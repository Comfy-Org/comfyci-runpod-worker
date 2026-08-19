"""Generate (and optionally bless) golden baselines for a release tag.

Runs each selected workflow TWICE on RunPod at the tag's commit: r1 becomes the
golden outputs, r2-vs-r1 is the determinism noise floor that default regression
thresholds derive from. Results land under golden/<wf>/<tag>/ in storage;
blessing (--bless) flips golden/<wf>/current.json, which is what CI runs
compare against. Generate first, inspect on the dashboard, then bless.

Usage:
  python golden_baseline.py --ref v0.3.50 [--workflows all] [--bless]
  python golden_baseline.py --ref v0.3.50 --bless-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import compare
import storage
from run_regression import apply_seed_overrides, entry_value, load_manifest
from submit_and_poll import RunPodClient, run_workflow_job

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "https://github.com/Comfy-Org/ComfyUI"


def resolve_ref(repo_url: str, ref: str) -> str:
    out = subprocess.check_output(["git", "ls-remote", repo_url, ref, f"{ref}^{{}}"],
                                  text=True, timeout=60)
    peeled, plain = None, None
    for line in out.strip().splitlines():
        sha, name = line.split("\t")
        if name.endswith("^{}"):
            peeled = sha
        else:
            plain = sha
    sha = peeled or plain
    if not sha:
        raise SystemExit(f"ref {ref!r} not found in {repo_url}")
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifest" / "workflows.json"))
    ap.add_argument("--ref", required=True, help="release tag (or branch) to build goldens from")
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--workdir", default="./golden-work")
    ap.add_argument("--workflows", default="all")
    ap.add_argument("--bless", action="store_true", help="also flip current.json after generating")
    ap.add_argument("--bless-only", action="store_true", help="no runs; just point current.json at --ref")
    ap.add_argument("--blessed-by", default=os.environ.get("GITHUB_ACTOR", "manual"))
    storage.add_storage_args(ap)
    args = ap.parse_args()
    store = storage.from_args(args)

    manifest_path = Path(args.manifest).resolve()
    repo_root = manifest_path.parent.parent
    manifest = load_manifest(manifest_path)
    defaults = manifest["defaults"]
    selected = {
        wf_id: e for wf_id, e in manifest["workflows"].items()
        if (args.workflows == "all" and e.get("enabled")) or wf_id in args.workflows.split(",")
    }

    if args.bless_only:
        for wf_id in selected:
            store.upload_json(f"{args.prefix}/golden/{wf_id}/current.json",
                              {"tag": args.ref, "blessed_by": args.blessed_by,
                               "blessed_ts": int(time.time())})
            print(f"[{wf_id}] blessed golden {args.ref}")
        store.finalize(f"bless golden {args.ref} ({', '.join(sorted(selected))})")
        return 0

    commit = resolve_ref(args.repo_url, args.ref)
    print(f"golden ref {args.ref} -> {commit}")

    failed = []
    for wf_id, entry in selected.items():
        workflow = json.loads((repo_root / entry["workflow_path"]).read_text(encoding="utf-8"))
        workflow = apply_seed_overrides(workflow, entry.get("seed_overrides"))
        endpoint_id = entry.get("endpoint_id") or os.environ["RUNPOD_ENDPOINT_ID"]
        client = RunPodClient(endpoint_id)
        wf_dir = Path(args.workdir) / wf_id
        runs = {}
        for r in (1, 2):
            print(f"[{wf_id}] golden run r{r} @ {args.ref}")
            rec = run_workflow_job(
                client, wf_id, workflow, comfy_commit=commit,
                models=entry.get("models", []),
                timeout_s=entry_value(entry, defaults, "timeout_s"),
                comfy_flags=entry_value(entry, defaults, "comfy_flags"),
                workdir=wf_dir / f"r{r}", comfy_repo=args.repo_url)
            runs[r] = rec
            if rec.get("status") != "ok":
                print(f"::error::[{wf_id}] golden r{r} failed: {rec.get('status')} "
                      f"{json.dumps(rec.get('error'))[:500]}")
                break
        if runs.get(1, {}).get("status") != "ok" or runs.get(2, {}).get("status") != "ok":
            failed.append(wf_id)
            continue

        noise_floor = compare.compare_dirs(Path(runs[1]["outputs_dir"]),
                                           Path(runs[2]["outputs_dir"]))
        print(f"[{wf_id}] noise floor: "
              f"{'identical' if noise_floor.get('identical') else json.dumps(noise_floor)[:200]}")

        base = f"{args.prefix}/golden/{wf_id}/{args.ref}"
        store.upload_dir(f"{base}/outputs", Path(runs[1]["outputs_dir"]))
        store.upload_json(f"{base}/run_r1.json",
                          {k: v for k, v in runs[1].items() if k != "outputs_dir"})
        store.upload_json(f"{base}/run_r2.json",
                          {k: v for k, v in runs[2].items() if k != "outputs_dir"})
        store.upload_json(f"{base}/noise_floor.json", noise_floor)
        store.upload_json(f"{base}/blessed.json",
                          {"tag": args.ref, "commit": commit,
                           "gpu_name": runs[1].get("gpu_name"),
                           "torch_version": runs[1].get("torch_version"),
                           "generated_by": args.blessed_by, "generated_ts": int(time.time())})
        print(f"[{wf_id}] golden uploaded to {base}/")

        if args.bless:
            store.upload_json(f"{args.prefix}/golden/{wf_id}/current.json",
                              {"tag": args.ref, "blessed_by": args.blessed_by,
                               "blessed_ts": int(time.time())})
            print(f"[{wf_id}] blessed golden {args.ref}")

    store.finalize(f"golden {args.ref} ({', '.join(sorted(selected))})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
