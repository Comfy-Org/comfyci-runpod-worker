"""CI driver: run the manifest's workflows on RunPod for one ComfyUI commit,
compare against golden + previous-run baselines, publish everything to storage
(a GitHub results branch by default; GCS once credentials exist — see storage.py).

Exit code: 1 iff any enabled workflow's verdict is "fail" or "execution_error".
Infrastructure problems and missing baselines are GitHub warnings, never
failures, so RunPod flakiness cannot block core merges.

Usage (CI):
  python run_regression.py --commit $GITHUB_SHA --branch master \
      [--manifest ../manifest/workflows.json] [--workflows all]
Env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID; GITHUB_TOKEN to push results in
Actions (github storage) or GOOGLE_APPLICATION_CREDENTIALS (gcs storage).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from pathlib import Path

import compare
import storage
from submit_and_poll import INFRA_STATUSES, RunPodClient, run_workflow_job

HERE = Path(__file__).resolve().parent
EXEC_FAIL_STATUSES = {"execution_error", "validation_error", "timeout", "prompt_rejected"}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_value(entry: dict, defaults: dict, key: str):
    v = entry.get(key)
    return defaults.get(key) if v is None else v


def apply_seed_overrides(workflow: dict, overrides: dict) -> dict:
    wf = json.loads(json.dumps(workflow))
    for node_id, patch in (overrides or {}).items():
        if node_id in wf:
            wf[node_id]["inputs"].update(patch)
    return wf


def derive_thresholds(entry: dict, defaults: dict, noise_floor: dict | None) -> dict:
    """Manifest entry wins; otherwise loosen the defaults to 3x the golden's
    own re-run noise floor so nondeterministic workflows don't false-alarm."""
    if entry.get("thresholds"):
        return entry["thresholds"]
    th = dict(defaults["thresholds"])
    if noise_floor and not noise_floor.get("error") and not noise_floor.get("identical"):
        mse = noise_floor.get("mean_mse") or 0
        if mse > 0:
            th["max_mean_mse"] = max(th["max_mean_mse"], round(mse * 3, 6))
            # PSNR of 3x the noise-floor MSE is floor PSNR minus 10*log10(3) dB
            floor_psnr = noise_floor.get("mean_psnr_db")
            if floor_psnr is not None:
                th["min_mean_psnr_db"] = min(th["min_mean_psnr_db"],
                                             round(floor_psnr - 10 * math.log10(3), 2))
        pct = noise_floor.get("mean_pct_pixels_changed") or 0
        if pct > 0:
            th["max_pct_pixels_changed"] = max(th["max_pct_pixels_changed"], round(pct * 3, 3))
    return th


def classify_detail(verdict: str | None, vs_previous: dict | None,
                    previous_verdict: str | None, golden_tag: str | None = None,
                    previous_golden_tag: str | None = None) -> str | None:
    """Label a failure vs golden: 'new_drift' when this commit's outputs differ
    from the previous tested run (the change happened here), 'inherited' when
    they are bit-identical to a previous run that already failed against the
    same golden. None when there is nothing to compare against, or when the
    outputs match a previous run that did not fail against this golden (the
    golden changed in between, not the code)."""
    if verdict != "fail" or not vs_previous or vs_previous.get("error"):
        return None
    if vs_previous.get("identical"):
        same_golden = golden_tag is None or previous_golden_tag == golden_tag
        return "inherited" if previous_verdict == "fail" and same_golden else None
    return "new_drift"


def metrics_pass(metrics: dict, th: dict) -> bool:
    if metrics.get("error"):
        return False
    if metrics.get("identical"):
        return True
    if metrics.get("frame_count_ref") != metrics.get("frame_count_cand"):
        return False
    if metrics.get("mean_mse", 0) > th["max_mean_mse"]:
        return False
    psnr = metrics.get("mean_psnr_db")
    if psnr is not None and psnr < th["min_mean_psnr_db"]:
        return False
    if metrics.get("mean_pct_pixels_changed", 0) > th["max_pct_pixels_changed"]:
        return False
    return True


def first_output_sha(run_rec: dict) -> str | None:
    """sha256 of the first PNG (by filename) — the identity of a run's output."""
    pngs = sorted((o for o in run_rec.get("outputs") or []
                   if str(o.get("filename", "")).endswith(".png")),
                  key=lambda o: o["filename"])
    return pngs[0].get("sha256") if pngs else None


def process_workflow(wf_id: str, entry: dict, defaults: dict, args, repo_root: Path,
                     store: storage.Storage) -> dict:
    workdir = Path(args.workdir) / wf_id
    workdir.mkdir(parents=True, exist_ok=True)

    workflow = json.loads((repo_root / entry["workflow_path"]).read_text(encoding="utf-8"))
    workflow = apply_seed_overrides(workflow, entry.get("seed_overrides"))

    endpoint_id = entry.get("endpoint_id") or os.environ["RUNPOD_ENDPOINT_ID"]
    client = RunPodClient(endpoint_id)
    print(f"[{wf_id}] submitting to RunPod (endpoint {endpoint_id})")
    run_rec = run_workflow_job(
        client, wf_id, workflow,
        comfy_commit=args.commit,
        models=entry.get("models", []),
        timeout_s=entry_value(entry, defaults, "timeout_s"),
        comfy_flags=entry_value(entry, defaults, "comfy_flags"),
        workdir=workdir,
        comfy_repo=args.repo_url,
    )
    published = {k: v for k, v in run_rec.items() if k != "outputs_dir"}  # runner-local path
    (workdir / "run.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    status = run_rec.get("status")
    print(f"[{wf_id}] worker status: {status}")

    comparison: dict = {"verdict": None, "detail": None, "vs_golden": None,
                        "vs_previous": None, "thresholds_used": None, "golden_tag": None,
                        "previous_commit": None, "previous_verdict": None,
                        "previous_golden_tag": None, "figures": False}

    if status in INFRA_STATUSES:
        comparison["verdict"] = "infra_error"
    elif status in EXEC_FAIL_STATUSES:
        comparison["verdict"] = "execution_error"
    else:
        outputs_dir = Path(run_rec["outputs_dir"])
        figures_dir = workdir / "figures"

        # Previous run first (metrics only): it decides whether a failure vs
        # golden is new drift introduced by this commit or drift inherited
        # from an earlier commit, which in turn decides whether figures are
        # worth rendering again.
        prev_dir = workdir / "_previous"
        latest = store.latest_pointer(args.branch)
        if latest and latest.get("commit") and latest["commit"] != args.commit:
            prev = latest["commit"]
            comparison["previous_commit"] = prev
            if store.fetch_run_outputs(args.branch, prev, wf_id, prev_dir):
                comparison["vs_previous"] = compare.compare_dirs(prev_dir, outputs_dir)
                prev_summary = store.run_summary(args.branch, prev) or {}
                prev_wf = prev_summary.get("workflows", {}).get(wf_id) or {}
                comparison["previous_verdict"] = prev_wf.get("verdict")
                comparison["previous_golden_tag"] = prev_wf.get("golden_tag")

        golden_ptr = store.golden_current(wf_id)
        if golden_ptr and golden_ptr.get("tag"):
            tag = golden_ptr["tag"]
            comparison["golden_tag"] = tag
            golden_dir = workdir / "_golden"
            n = store.fetch_golden_outputs(wf_id, tag, golden_dir)
            if n:
                noise_floor = store.noise_floor(wf_id, tag)
                th = derive_thresholds(entry, defaults, noise_floor)
                comparison["thresholds_used"] = th
                vs_golden = compare.compare_dirs(golden_dir, outputs_dir)
                comparison["verdict"] = "pass" if metrics_pass(vs_golden, th) else "fail"
                comparison["detail"] = classify_detail(
                    comparison["verdict"], comparison["vs_previous"],
                    comparison["previous_verdict"], tag, comparison["previous_golden_tag"])
                # Inherited failures are bit-identical to an earlier run whose
                # figures already exist: don't render (and store) them again.
                want_figures = comparison["detail"] != "inherited"
                comparison["vs_golden"] = compare.compare_with_figures(
                    golden_dir, outputs_dir, figures_dir, "golden", f"golden {tag}",
                    metrics=vs_golden, figures=want_figures)
                comparison["figures"] = bool(want_figures and not vs_golden.get("identical")
                                             and not vs_golden.get("error"))
            else:
                comparison["verdict"] = "no_baseline"
        else:
            comparison["verdict"] = "no_baseline"

        if comparison["vs_previous"] is not None:
            comparison["vs_previous"] = compare.compare_with_figures(
                prev_dir, outputs_dir, figures_dir, "prev", "previous",
                metrics=comparison["vs_previous"])

    (workdir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # Publish run.json/comparison.json/outputs/figures; baseline copies stay local.
    publish_dir = workdir
    for sub in ("_golden", "_previous"):
        d = workdir / sub
        if d.exists():
            import shutil
            shutil.rmtree(d)
    if not args.skip_publish:
        store.publish_workflow_run(args.branch, args.commit, wf_id, publish_dir)
    return {"workflow_id": wf_id, "worker_status": status, "verdict": comparison["verdict"],
            "detail": comparison["detail"],
            "vs_golden": comparison["vs_golden"], "vs_previous": comparison["vs_previous"],
            "thresholds_used": comparison["thresholds_used"],
            "golden_tag": comparison["golden_tag"],
            "previous_commit": comparison["previous_commit"],
            "previous_verdict": comparison["previous_verdict"],
            "figures": comparison["figures"],
            "output_sha256": first_output_sha(run_rec),
            "gpu_name": run_rec.get("gpu_name"), "timings": run_rec.get("timings"),
            "vram_peak_mb": run_rec.get("vram_peak_mb"),
            "rss_peak_mb": run_rec.get("rss_peak_mb"),
            "comfy_version": run_rec.get("comfy_version"),
            "torch_version": run_rec.get("torch_version"),
            "python_version": run_rec.get("python_version")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifest" / "workflows.json"))
    ap.add_argument("--commit", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--workdir", default="./regression-work")
    ap.add_argument("--workflows", default="all", help="csv of workflow ids, or 'all'")
    ap.add_argument("--skip-publish", action="store_true", help="local dry run, no writes")
    storage.add_storage_args(ap)
    args = ap.parse_args()
    store = storage.from_args(args)

    manifest_path = Path(args.manifest).resolve()
    repo_root = manifest_path.parent.parent
    manifest = load_manifest(manifest_path)
    defaults = manifest["defaults"]

    selected = {
        wf_id: entry for wf_id, entry in manifest["workflows"].items()
        if (entry.get("enabled") if args.workflows == "all"
            else wf_id in args.workflows.split(","))
    }
    if not selected:
        print("no workflows selected; nothing to do")
        return 0

    if not args.skip_publish:
        store.snapshot_manifest(args.commit, manifest)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futs = {pool.submit(process_workflow, wf_id, entry, defaults, args, repo_root,
                            store): wf_id
                for wf_id, entry in selected.items()}
        for fut in concurrent.futures.as_completed(futs):
            wf_id = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{wf_id}] harness error: {e!r}")
                results.append({"workflow_id": wf_id, "worker_status": "harness_error",
                                "verdict": "infra_error", "error": repr(e)[:1000]})

    results.sort(key=lambda r: r["workflow_id"])
    verdicts = {r["workflow_id"]: r["verdict"] for r in results}
    overall = ("fail" if any(v in ("fail", "execution_error") for v in verdicts.values())
               else "pass")
    summary = {"schema_version": 2, "branch": args.branch, "commit": args.commit,
               "run_ts": int(time.time()), "overall": overall,
               "has_infra_error": any(v == "infra_error" for v in verdicts.values()),
               "workflows": {r["workflow_id"]: r for r in results}}

    if not args.skip_publish:
        store.publish_summary(args.branch, args.commit, summary)
        # Advance the previous-run pointer only for fully comparable runs: a commit
        # where every workflow at least produced outputs.
        if all(r["verdict"] in ("pass", "fail", "no_baseline") for r in results):
            store.update_latest(args.branch,
                                {"commit": args.commit, "run_ts": summary["run_ts"],
                                 "workflows": sorted(verdicts)})
        store.finalize(f"regression {args.branch}@{args.commit[:8]}: {overall}")

    for r in results:
        v = r["verdict"]
        line = f"{r['workflow_id']}: {v} (worker={r['worker_status']})"
        if v in ("fail", "execution_error"):
            print(f"::error::{line}")
        elif v in ("infra_error", "no_baseline"):
            print(f"::warning::{line}")
        else:
            print(line)
    print(f"overall: {overall}")
    return 1 if overall == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
