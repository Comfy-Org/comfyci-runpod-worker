"""CI driver: run the manifest's workflows on RunPod for one ComfyUI commit,
compare against golden + previous-run baselines, publish everything to storage
(a GitHub results branch by default; GCS once credentials exist — see storage.py).

A workflow whose entry says `"level": "smoke"` is not compared: it passes when
it finished and every output file passes the checks of smoke.py. Its large
output files are published only when they repeat.

Exit code: 1 iff any enabled workflow's verdict is "fail" or "execution_error".
Infrastructure problems and missing baselines are GitHub warnings, never
failures, so RunPod flakiness cannot block core merges.

A nightly or weekly lane (cadence "nightly" or "weekly" in manifest/lanes.json)
runs many workflows once, for an hour or more, and nothing tests its commit
again the next minute. So such a run behaves in its own way:

- It has `max_in_flight` jobs on the endpoint at once (the lane's key), each
  with one attempt, 1800 s for the queue and a look every 3 s.
- It has a deadline of its own: the lane's `timeout_minutes` minus 25. A
  workflow whose job could still be running then is not started. Its cell
  says `not_reached`.
- After every 10 workflows it pushes what it has, with `partial` in the
  summary, so a run that dies leaves its results behind.
- At its end it tries once more what was not tested for a reason that can
  pass. Not a job that used up its whole time, and not one that another
  attempt could not change.
- When GitHub stops it, it cancels its RunPod jobs before it goes.
- Its exit code is also 1 when a workflow stayed untested: the next run is a
  day or a week away, and a red run is what GitHub mails.
- Its summary says for each red workflow how many commits lie between the
  workflow's own last result and this commit (`ranges`).

Usage (CI):
  python run_regression.py --commit $GITHUB_SHA --branch master \
      [--manifest ../manifest/workflows.json] [--workflows all] \
      [--summary-out summary.json] [--max-in-flight N] [--deadline-minutes N]
Env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID; GITHUB_TOKEN to push results in
Actions (github storage) or GOOGLE_APPLICATION_CREDENTIALS (gcs storage).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import commit_meta
import compare
import index_builder
import lanes as lanes_mod
import output_kinds
import packs
import regression_skip
import smoke
import storage
from submit_and_poll import (INFRA_STATUSES, QUEUE_DEADLINE_S, STOP_CANCEL_TIMEOUT_S, TIMED_OUT,
                             RunPodClient, poll_deadline_s, run_workflow_job)

HERE = Path(__file__).resolve().parent
# A run submits all its jobs at once, and the endpoint has fewer workers than a
# run has jobs. Each job further back in the line may wait this much longer in
# RunPod's queue, so none is cancelled only for standing behind its own run.
QUEUE_DEADLINE_STEP_S = 60
# A nightly or weekly run submits a job only when one of its `max_in_flight`
# slots is free, so the queue clock starts when a worker can take the job. The
# long deadline is for a worker that RunPod cannot start at once. The runner
# hands the next job to a worker itself, and the worker waits (billed) until
# it does: so it looks at a job every 3 s, not every 10.
LONG_QUEUE_DEADLINE_S = 1800
LONG_POLL_INTERVAL_S = 3
# The last minutes of such a lane's job belong to the publish and the report:
# the run's own deadline is the lane's `timeout_minutes` minus this.
DEADLINE_MARGIN_MIN = 25
# Such a run pushes what it has after this many workflows.
CHECKPOINT_EVERY = 10
# The worker status of a workflow that the run's deadline did not reach.
NOT_REACHED = "not_reached"
# What an untested workflow's status must not be for the pass at the end of a
# nightly or weekly run to try it again: another job would end the same way.
NO_SECOND_TRY = regression_skip.NO_RETRY | {"bad_request"}
# What GitHub sends a step's process when the run is cancelled or out of time:
# an interrupt, 7.5 s later a terminate, and 2.5 s after that it kills.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# custom_node_error: a custom-node pack the workflow requested did not import on
# the commit under test (whether or not the workflow uses any of its nodes).
EXEC_FAIL_STATUSES = {"execution_error", "validation_error", "timeout", "prompt_rejected",
                      "custom_node_error"}


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


def write_thumbnails(outputs_dir: Path, size: int = 256) -> str | None:
    """WebP previews next to the full PNGs (outputs/thumbs/); returns the first
    thumbnail's filename. Full PNGs are ~1.6 MB and raw.githubusercontent.com
    serves them slowly, so the dashboard's list views use these instead."""
    from PIL import Image
    thumbs_dir = outputs_dir / "thumbs"
    first = None
    for png in sorted(outputs_dir.rglob("*.png")):
        if thumbs_dir in png.parents:
            continue
        thumbs_dir.mkdir(exist_ok=True)
        name = png.stem + ".webp"
        with Image.open(png) as im:
            im.thumbnail((size, size))
            im.convert("RGB").save(thumbs_dir / name, "WEBP", quality=80, method=4)
        first = first or name
    return first


class ImageComparator(output_kinds.Comparator):
    """PNG outputs, compared as they always were: every method is one of the
    functions above or in compare.py, called with the arguments it always got."""
    kind = "image"
    has_figures = True

    def compare(self, ref_dir: Path, cand_dir: Path) -> dict:
        return compare.compare_dirs(ref_dir, cand_dir)

    def thresholds(self, entry: dict, defaults: dict, noise_floor: dict | None) -> dict:
        return derive_thresholds(entry, defaults, noise_floor)

    def passes(self, metrics: dict, thresholds: dict | None) -> bool:
        return metrics_pass(metrics, thresholds)

    def render(self, ref_dir: Path, cand_dir: Path, figures_dir: Path, label: str,
               ref_name: str, metrics: dict, figures: bool = True) -> dict:
        return compare.compare_with_figures(ref_dir, cand_dir, figures_dir, label, ref_name,
                                            metrics=metrics, figures=figures)

    def thumbnails(self, outputs_dir: Path) -> str | None:
        return write_thumbnails(outputs_dir)

    def primary_sha(self, run_rec: dict) -> str | None:
        return first_output_sha(run_rec)

    def zero_floor(self) -> dict:
        return {"identical": True, "mean_mse": 0.0, "mean_psnr_db": None, "min_psnr_db": None,
                "max_abs_diff": 0, "mean_pct_pixels_changed": 0.0}


COMPARATORS: dict[str, output_kinds.Comparator] = {
    "image": ImageComparator(), "text": output_kinds.TextComparator()}


def comparator_for(entry: dict | None) -> output_kinds.Comparator:
    """The comparator of a manifest entry's output kind (ValueError: an unknown
    kind, or a kind that nothing can compare yet)."""
    kind = output_kinds.kind_of(entry)
    if kind not in COMPARATORS:
        raise ValueError(f"no comparator for output_kind {kind!r} yet: such a workflow can "
                         'only run with "level": "smoke"')
    return COMPARATORS[kind]


def smoke_verdict(status: str | None, problems: list) -> str:
    """The verdict of a smoke-level run: "pass" when the workflow finished and
    the checks of its output files found nothing. "no_outputs" (it finished
    and wrote no file) is an infrastructure status at reference level, where a
    template that ends in a preview gets it. Here it is a failure: it is the
    first thing a smoke check exists to catch. A status this does not know
    means the workflow was not tested."""
    if status == "ok":
        return "fail" if problems else "pass"
    if status == "no_outputs":
        return "fail"
    return "execution_error" if status in EXEC_FAIL_STATUSES else "infra_error"


def held_back(run_rec: dict, previous_files: list | None) -> set[str]:
    """The names of the output files a smoke run does not publish: by the rule
    of smoke.unpublished over every file the worker listed (with
    `previous_files`, the `smoke.files` of the workflow's previous result),
    and every file the worker did not send, of which there is nothing to publish."""
    outputs = [o for o in run_rec.get("outputs") or []
               if isinstance(o, dict) and isinstance(o.get("filename"), str)]
    listed = [{"name": o["filename"], "bytes": o.get("bytes"), "sha256": o.get("sha256")}
              for o in outputs]
    return (smoke.unpublished(listed, previous_files)
            | {o["filename"] for o in outputs if o.get("truncated")})


def _not_published(entry: dict) -> dict:
    """A `smoke.files` entry that says its file is not published, with the key
    where every entry has it: after the worker's facts, before what was measured."""
    head = {k: entry[k] for k in ("name", "type", "bytes", "sha256", "sent") if k in entry}
    return {**head, "published": False, **{k: v for k, v in entry.items() if k not in head}}


def check_smoke(run_rec: dict, workflow: dict, entry: dict, status: str | None,
                previous_files: list | None) -> tuple[dict, set[str]]:
    """(the `smoke` object of a run, the names of the output files that are
    not published). An entry of `files` whose file is not published carries
    `"published": False`."""
    found = smoke.check_outputs(run_rec.get("outputs_dir"), run_rec, workflow,
                                entry.get("expect"), status)
    held = held_back(run_rec, previous_files)
    found["files"] = [_not_published(f) if f["name"] in held else f for f in found["files"]]
    return found, held


def queue_deadline_s(position: int) -> int:
    """Seconds the job submitted as number `position` of a run (from 0) may
    wait in RunPod's queue: 600 for the first, 60 more for each job before it."""
    return QUEUE_DEADLINE_S + QUEUE_DEADLINE_STEP_S * position


def process_workflow(wf_id: str, entry: dict, defaults: dict, args, repo_root: Path,
                     store: storage.Storage, queue_deadline: float = QUEUE_DEADLINE_S) -> dict:
    workdir = Path(args.workdir) / wf_id
    workdir.mkdir(parents=True, exist_ok=True)
    # At reference level the output is compared with the approved reference
    # and with the previous run. At smoke level nothing is compared, so there
    # is no comparator, and no reference or earlier output is read.
    level = lanes_mod.level_of(entry)
    kind = output_kinds.kind_of(entry)
    cmp_ = comparator_for(entry) if level == "reference" else None

    workflow = json.loads((repo_root / entry["workflow_path"]).read_text(encoding="utf-8"))
    workflow = apply_seed_overrides(workflow, entry.get("seed_overrides"))

    endpoint_id = entry.get("endpoint_id") or args.endpoint_id
    client = RunPodClient(endpoint_id)
    # A nightly or weekly run keeps its clients: when it is stopped it cancels
    # the jobs they have in flight. main() puts the list on `args`.
    clients = getattr(args, "clients", None)
    if clients is not None:
        clients.append(client)
    pack_records = packs.records_for(entry, repo_root)
    print(f"[{wf_id}] submitting to RunPod (endpoint {endpoint_id}, lane {args.lane_id}"
          + (f", packs {', '.join(r['name'] for r in pack_records)}" if pack_records else "") + ")")
    run_rec = run_workflow_job(
        client, wf_id, workflow,
        comfy_commit=args.commit,
        models=entry.get("models", []),
        timeout_s=entry_value(entry, defaults, "timeout_s"),
        comfy_flags=entry_value(entry, defaults, "comfy_flags"),
        workdir=workdir,
        comfy_repo=args.repo_url,
        lane=args.lane_id,
        custom_nodes=pack_records,
        queue_deadline_s=queue_deadline,
        # Only a nightly or weekly run says more: one attempt, a shorter poll interval.
        **(getattr(args, "job_options", None) or {}),
    )
    published = {k: v for k, v in run_rec.items() if k != "outputs_dir"}  # runner-local path
    (workdir / "run.json").write_text(json.dumps(published, indent=2), encoding="utf-8")
    status = run_rec.get("status")
    print(f"[{wf_id}] worker status: {status}")

    comparison: dict = {"verdict": None, "detail": None, "vs_golden": None,
                        "vs_previous": None, "thresholds_used": None, "golden_tag": None,
                        "previous_commit": None, "previous_verdict": None,
                        "previous_golden_tag": None, "figures": False}
    if kind != output_kinds.DEFAULT_KIND:  # an image comparison.json keeps its keys
        comparison["output_kind"] = kind

    held: set[str] = set()  # output files that stay off the results branch (smoke level only)
    if cmp_ is None:
        # Only a smoke comparison names its level: absent means reference.
        comparison["level"] = level
        # What the workflow wrote in its previous result, which main() read
        # for every smoke workflow of the run: a large file is published only
        # when it repeats.
        previous_files = (getattr(args, "previous_smoke", None) or {}).get(wf_id)
        comparison["smoke"], held = check_smoke(run_rec, workflow, entry, status,
                                                previous_files)
        comparison["verdict"] = smoke_verdict(status, comparison["smoke"]["problems"])
    elif status in INFRA_STATUSES:
        comparison["verdict"] = "infra_error"
    elif status in EXEC_FAIL_STATUSES:
        comparison["verdict"] = "execution_error"
    else:
        outputs_dir = Path(run_rec["outputs_dir"])
        figures_dir = workdir / "figures"
        # A compared file the worker listed and did not send (never so for images).
        lost = cmp_.transfer_error(run_rec)

        def against(ref_dir: Path) -> dict:
            if lost:
                return {"kind": cmp_.kind, "error": lost}
            return cmp_.compare(ref_dir, outputs_dir)

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
                comparison["vs_previous"] = against(prev_dir)
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
                th = cmp_.thresholds(entry, defaults, noise_floor)
                comparison["thresholds_used"] = th
                vs_golden = against(golden_dir)
                comparison["verdict"] = "pass" if cmp_.passes(vs_golden, th) else "fail"
                comparison["detail"] = classify_detail(
                    comparison["verdict"], comparison["vs_previous"],
                    comparison["previous_verdict"], tag, comparison["previous_golden_tag"])
                # Inherited failures are bit-identical to an earlier run whose
                # figures already exist: don't render (and store) them again.
                want_figures = comparison["detail"] != "inherited"
                comparison["vs_golden"] = cmp_.render(
                    golden_dir, outputs_dir, figures_dir, "golden", f"golden {tag}",
                    metrics=vs_golden, figures=want_figures)
                comparison["figures"] = bool(cmp_.has_figures and want_figures
                                             and not vs_golden.get("identical")
                                             and not vs_golden.get("error"))
            else:
                comparison["verdict"] = "no_baseline"
        else:
            comparison["verdict"] = "no_baseline"

        if comparison["vs_previous"] is not None:
            comparison["vs_previous"] = cmp_.render(
                prev_dir, outputs_dir, figures_dir, "prev", "previous",
                metrics=comparison["vs_previous"])

    (workdir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    thumbnail = None
    if run_rec.get("outputs_dir") and comparison["verdict"] not in ("infra_error", "execution_error"):
        try:
            # A smoke run gets a preview of every PNG it wrote, whatever its output kind.
            previews = cmp_.thumbnails if cmp_ else write_thumbnails
            thumbnail = previews(Path(run_rec["outputs_dir"]))
        except Exception as e:  # previews are a convenience, never a failure
            print(f"[{wf_id}] thumbnail generation failed: {e!r}")

    # Publish run.json/comparison.json/outputs/figures; baseline copies stay local.
    publish_dir = workdir
    for sub in ("_golden", "_previous"):
        d = workdir / sub
        if d.exists():
            shutil.rmtree(d)
    if not args.skip_publish and held:
        # The files held back stay in this folder, so the run's artifact has them.
        store.publish_workflow_run(args.branch, args.commit, wf_id, publish_dir,
                                   exclude=sorted(f"outputs/{name}" for name in held))
    elif not args.skip_publish:
        store.publish_workflow_run(args.branch, args.commit, wf_id, publish_dir)
    cell = {"workflow_id": wf_id, "worker_status": status, "verdict": comparison["verdict"],
            "detail": comparison["detail"],
            "vs_golden": comparison["vs_golden"], "vs_previous": comparison["vs_previous"],
            "thresholds_used": comparison["thresholds_used"],
            "golden_tag": comparison["golden_tag"],
            "previous_commit": comparison["previous_commit"],
            "previous_verdict": comparison["previous_verdict"],
            "figures": comparison["figures"],
            "output_sha256": (cmp_.primary_sha(run_rec) if cmp_
                              else smoke.main_sha(run_rec, wf_id, kind)),
            "thumbnail": thumbnail,
            "gpu_name": run_rec.get("gpu_name"), "timings": run_rec.get("timings"),
            "vram_peak_mb": run_rec.get("vram_peak_mb"),
            "rss_peak_mb": run_rec.get("rss_peak_mb"),
            "comfy_version": run_rec.get("comfy_version"),
            "torch_version": run_rec.get("torch_version"),
            "python_version": run_rec.get("python_version"),
            "env": run_rec.get("env"),
            "custom_nodes": run_rec.get("custom_nodes")}
    if kind != output_kinds.DEFAULT_KIND:  # absent means image: those cells keep their keys
        cell["output_kind"] = kind
    if cmp_ is None:  # absent means reference: those cells keep their keys too
        cell["level"] = level
        cell["smoke"] = comparison["smoke"]
    return cell


def previous_smoke_files(store: storage.Storage, branch: str, prior_entries: list[dict],
                         commit: str, wf_ids: list[str]) -> dict[str, list]:
    """Per workflow of `wf_ids`: the `smoke.files` of its previous result on
    the lane, which is its newest red or green result at another commit that
    the index (`prior_entries`) knows. A workflow without one is left out.
    Each earlier summary is read once, and alone: no run folder is fetched."""
    priors = index_builder.prior_verdicts(prior_entries, commit)
    summaries: dict[str, dict] = {}
    found: dict[str, list] = {}
    for wf_id in wf_ids:
        prior = (priors.get(wf_id) or {}).get("commit")
        if not prior:
            continue
        if prior not in summaries:
            summaries[prior] = store.published_summary(branch, prior) or {}
        cell = (summaries[prior].get("workflows") or {}).get(wf_id) or {}
        files = (cell.get("smoke") or {}).get("files")
        if isinstance(files, list):
            found[wf_id] = files
    return found


def published_new_bytes(results: list[dict], previous: dict[str, list]) -> int:
    """Bytes of the output files this run published that the previous result
    of their workflow had not published under the same name and sha256: about
    what the run added to the results branch, where equal files are stored
    once. Counted over the files that smoke cells list. `previous` is what
    `previous_smoke_files` returned."""
    total = 0
    for cell in results:
        before = {f.get("name"): f.get("sha256")
                  for f in previous.get(cell["workflow_id"]) or []
                  if isinstance(f, dict) and f.get("published", True) and f.get("sent", True)}
        for f in (cell.get("smoke") or {}).get("files") or []:
            size, sha = f.get("bytes"), f.get("sha256")
            there = f.get("published", True) and f.get("sent", True)
            new = not sha or before.get(f.get("name")) != sha
            if there and new and isinstance(size, int) and not isinstance(size, bool):
                total += size
    return total


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def actions_run() -> tuple[str | None, int | None]:
    """(url, id) of the Actions run this process belongs to; (None, None)
    outside Actions. The run's page shows the report of every lane it tested."""
    run_id, repo = os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_REPOSITORY")
    if not (run_id and repo):
        return None, None
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{repo}/actions/runs/{run_id}", int(run_id) if run_id.isdigit() else None


def attempt_marker(args, lane_id: str, attempt: int) -> dict:
    """runs/<branch>/<commit>/attempt.json: which attempt started, and the
    Actions run behind it (to find the log of one that never published)."""
    marker = {"commit": args.commit, "branch": args.branch, "lane": lane_id,
              "attempt": attempt, "started_ts": int(time.time())}
    run_url, _ = actions_run()
    if run_url:
        marker["run_url"] = run_url
    return marker


def merge_retest(old: dict, new: dict) -> dict:
    """The summary of a commit after a run that named its workflows tested
    some of them again. `old` is the published summary, `new` this run's.

    Only what the run tested changes: its cells replace the old ones, and
    `overall` and `has_infra_error` are worked out again over every cell. Every
    other cell and every other key stay as they are, because they describe the
    lane's run of the commit (its time, its Actions run, its tested range). The
    run records itself under `retests`. `scope` grows only when the old summary
    had one: a summary without it stays the lane's result of the commit."""
    cells = {**(old.get("workflows") or {}), **new["workflows"]}
    verdicts = [cell.get("verdict") for cell in cells.values()]
    tested = sorted(new["workflows"])
    merged = dict(old)
    merged["workflows"] = {wf_id: cells[wf_id] for wf_id in sorted(cells)}
    merged["overall"] = ("fail" if any(v in ("fail", "execution_error") for v in verdicts)
                         else "pass")
    merged["has_infra_error"] = "infra_error" in verdicts
    if "scope" in old:
        merged["scope"] = sorted({*(old["scope"] or []), *tested})
    # A range belongs to a cell: the one of a workflow tested again is this run's.
    ranges = {wf_id: r for wf_id, r in (old.get("ranges") or {}).items() if wf_id not in tested}
    ranges.update(new.get("ranges") or {})
    if ranges:
        merged["ranges"] = ranges
    else:
        merged.pop("ranges", None)
    merged["retests"] = [*(old.get("retests") or []),
                         {"run_ts": new["run_ts"], "run_url": new["run_url"],
                          "run_id": new["run_id"], "env": new["env"], "workflows": tested}]
    return merged


# -- a nightly or weekly run ------------------------------------------------------------
def _count(text: str) -> int:
    """A command-line number that counts something: a whole number above 0."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a whole number above 0")
    return value


def lane_count(lane: dict, key: str) -> int | None:
    """What a lane says under a key that counts something (`max_in_flight`,
    `timeout_minutes`): None when it says nothing. Anything but a whole number
    above 0 is a configuration error, which `lanes.py validate` reports too."""
    value = lane.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"{key} of the lane must be a whole number above 0, not {value!r}; "
                         "see manifest/lanes.json")
    return value


def run_deadline(minutes: int | None, lane: dict, long_cadence: bool) -> float | None:
    """The time after which the run submits no job whose worst case would pass
    it, in seconds since the epoch, or None: no deadline. `minutes` is
    --deadline-minutes, counted from now. Without it a nightly or weekly lane
    has its `timeout_minutes` (GitHub's limit for the job) minus
    DEADLINE_MARGIN_MIN, and a per-commit lane has none."""
    if minutes is None and long_cadence:
        limit = lane_count(lane, "timeout_minutes") or lanes_mod.DEFAULT_TIMEOUT_MINUTES
        minutes = max(limit - DEADLINE_MARGIN_MIN, 0)
    return None if minutes is None else time.time() + minutes * 60


def job_worst_case_s(entry: dict, defaults: dict, queue_deadline: float, attempts: int) -> float:
    """Seconds after which the runner has given up on a workflow's job at the
    latest: its poll deadline, once for each attempt."""
    return attempts * poll_deadline_s(entry_value(entry, defaults, "timeout_s"), queue_deadline)


def not_reached_cell(wf_id: str, entry: dict, left_s: float, need_s: float) -> dict:
    """The cell of a workflow that was not started, because its job could
    still have been running at the run's deadline. No job was submitted."""
    cell = {"workflow_id": wf_id, "worker_status": NOT_REACHED, "verdict": "infra_error",
            "error": (f"not started: its job can take {need_s / 60:.0f} min, and "
                      f"{max(left_s, 0) / 60:.0f} min were left before the run's deadline")}
    if lanes_mod.level_of(entry) == "smoke":  # every cell of a smoke workflow names its level
        cell["level"] = "smoke"
    return cell


def run_record(workdir: Path, wf_id: str) -> dict:
    """The record process_workflow wrote for a workflow of this run ({}: none)."""
    try:
        record = json.loads((workdir / wf_id / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def worth_another_try(cell: dict, record: dict) -> bool:
    """Whether the pass at the end of a nightly or weekly run tries a workflow
    again: it was not tested, and for a reason that can pass (the queue, the
    network, a worker error, a failed model download). Not when another job
    would end the same way (NO_SECOND_TRY), and not when its job used up its
    whole time: a second one can cost another hour. `record` is its run.json."""
    if cell.get("verdict") != "infra_error" or cell.get("worker_status") in NO_SECOND_TRY:
        return False
    return record.get("reason") != TIMED_OUT


def own_ranges(cells: list[dict], commit: str, commits_since) -> dict:
    """Per red workflow: the commits between its own last red or green result
    (`prior_commit`) and `commit`, as commit_meta.fetch_range answers
    (`commits_since(prev)`). A nightly or weekly run covers many commits, and
    its workflows did not all have their last result in the run before. A
    workflow without an earlier result has no range: it is on its first run."""
    ranges = {}
    for cell in cells:
        prior = cell.get("prior_commit")
        if cell.get("verdict") in ("fail", "execution_error") and prior and prior != commit:
            found = commits_since(prior)
            if found:
                ranges[cell["workflow_id"]] = found
    return ranges


def cancel_in_flight(clients: list) -> list[str]:
    """Cancel every RunPod job the run's clients still have in flight, all at
    once, and return their ids. Each call has a few seconds: GitHub kills the
    step 10 seconds after its first signal."""
    cancelled: list[str] = []
    threads = [threading.Thread(target=lambda c=client: cancelled.extend(c.cancel_live()),
                                daemon=True)
               for client in list(clients) if getattr(client, "live", None)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2 * STOP_CANCEL_TIMEOUT_S)
    return sorted(cancelled)


def stop_handler(clients: list, stopping: threading.Event, leave=os._exit):
    """The handler for GitHub's stop signals. Without one the RunPod jobs in
    flight would run on, billed, with nobody to read them. It cancels them and
    ends the process: there is no time to push anything, which is what the
    checkpoints are for. `stopping` keeps the run from starting another job
    in between. `leave` ends the process (the code is 128 plus the signal)."""
    def handle(signum, frame):
        again = stopping.is_set()
        stopping.set()
        try:
            if not again:  # the second signal finds the cancels sent
                cancelled = cancel_in_flight(clients)
                print(f"::error::stopped by signal {signum}: cancelled {len(cancelled)} RunPod "
                      f"job(s) in flight ({', '.join(cancelled) or 'none'})", flush=True)
        except Exception:  # noqa: BLE001 - whatever goes wrong here, the process ends
            pass
        finally:
            leave(128 + signum)
    return handle


def install_stop_handler(clients: list, stopping: threading.Event):
    """Handle STOP_SIGNALS with `stop_handler`. Returns a function that puts
    the earlier handlers back. Signals reach the main thread only: called from
    another one, this changes nothing."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    handler = stop_handler(clients, stopping)
    earlier = {sig: signal.signal(sig, handler) for sig in STOP_SIGNALS}

    def restore():
        for sig, was in earlier.items():
            signal.signal(sig, signal.SIG_DFL if was is None else was)
    return restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifest" / "workflows.json"))
    ap.add_argument("--commit", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--workdir", default="./regression-work")
    ap.add_argument("--workflows", default="all", help="csv of workflow ids, or 'all'")
    ap.add_argument("--skip-publish", action="store_true", help="local dry run, no writes")
    ap.add_argument("--lane", default=None, help="lane id from manifest/lanes.json "
                                                 "(default: the lane whose prefix matches)")
    ap.add_argument("--summary-out", default=None,
                    help="also write the run summary here (for the PR comment step); "
                         "written once results are published, whatever the verdict")
    ap.add_argument("--max-in-flight", type=_count, default=None,
                    help="jobs on the endpoint at once (default: the lane's max_in_flight, "
                         "else all of them)")
    ap.add_argument("--deadline-minutes", type=_count, default=None,
                    help="minutes from now after which no job is submitted that could still "
                         "be running then (default on a nightly or weekly lane: its "
                         "timeout_minutes minus 25; on a per-commit lane: none)")
    storage.add_storage_args(ap)
    args = ap.parse_args()

    # The lane decides the storage prefix; an unknown lane or prefix is a
    # configuration error, caught before any GPU job is submitted.
    lanes = index_builder.load_lanes()
    if args.lane:
        if args.lane not in lanes["lanes"]:
            raise SystemExit(f"unknown lane {args.lane!r}; see manifest/lanes.json")
        lane_id = args.lane
        args.prefix = lanes["lanes"][lane_id].get("prefix", args.prefix)
    else:
        lane_id = index_builder.lane_for_prefix(lanes, args.prefix)
        if not lane_id:
            raise SystemExit(f"no lane in manifest/lanes.json has prefix {args.prefix!r}; "
                             "pass --lane")
    if lanes_mod.is_retired(lanes["lanes"][lane_id]):
        raise SystemExit(f"lane {lane_id!r} is retired (cadence 'retired' in manifest/lanes.json): "
                         "its results stay readable but it runs nothing; pass --lane with an "
                         f"active lane (the primary is {lanes.get('primary')!r})")
    layout = lanes["lanes"][lane_id].get("layout", 1)
    # The lane names its endpoint (and so its worker image); a lane without one
    # is not provisioned yet and must not fall through to another lane's endpoint.
    args.lane_id = lane_id
    args.endpoint_id = index_builder.lane_endpoint(lanes, lane_id)
    if not args.endpoint_id:
        raise SystemExit(f"lane {lane_id!r} has no endpoint (endpoint_id is null in "
                         "manifest/lanes.json): create one for its image and commit the id "
                         "before running this lane")
    store = storage.from_args(args)
    manifest_path = Path(args.manifest).resolve()
    repo_root = manifest_path.parent.parent
    manifest = load_manifest(manifest_path)
    defaults = manifest["defaults"]
    # The pointer as it stands before this run names the previous tested
    # commit (the per-workflow comparisons read it again later, but they can
    # race with each other once the first workflow publishes).
    prev_commit = (store.latest_pointer(args.branch) or {}).get("commit")
    # The index lists an open failing chain only for a workflow the lane runs
    # today: enabled, not kept to other lanes, and of the lane's cadence.
    lane = lanes["lanes"][lane_id]
    try:
        active = lanes_mod.workflows_for_lane(manifest, lane_id, lane=lane)
    except ValueError as e:
        raise SystemExit(f"{e}; see manifest/workflows.json")
    builder = index_builder.IndexBuilder(store, args.branch, lane_id, layout, lanes,
                                         active=active)
    # The index as it stands before this run: it also records runs that had
    # execution errors, which the pointer above skips.
    prior_entries = (store.download_json(builder.head_path()) or {}).get("entries") or []
    # A re-run of a commit (the cron's one retry, or a dispatch naming it)
    # counts its attempts, so the cron knows when to stop. A run that died
    # before publishing left only its start marker, which counts too.
    published = store.published_summary(args.branch, args.commit)
    attempt = regression_skip.next_attempt(published,
                                           store.attempt_marker(args.branch, args.commit))
    if attempt > 1:
        print(f"{args.commit[:8]} was already tried on lane {lane_id}; this is attempt {attempt}")

    selected = {
        wf_id: entry for wf_id, entry in manifest["workflows"].items()
        if (entry.get("enabled") if args.workflows == "all"
            else wf_id in args.workflows.split(","))
    }
    # The lane filter is hard: a workflow restricted to other lanes (its packs
    # are only baked there), or of another cadence than the lane's, is skipped
    # even when named explicitly, and the dashboard shows its cell as not run.
    for wf_id, entry in list(selected.items()):
        try:
            why_not = lanes_mod.why_not_on_lane(entry, lane_id, lane)
        except ValueError as e:
            raise SystemExit(f"[{wf_id}] {e}; see manifest/workflows.json")
        if why_not:
            print(f"[{wf_id}] not run on lane {lane_id} ({why_not})")
            del selected[wf_id]
    if not selected:
        print(f"no workflows selected for lane {lane_id}; nothing to do")
        return 0
    # A level, a cadence, an expectation or an output kind that cannot be run
    # is a configuration error, like an unknown lane: caught here, before the
    # start marker and before any GPU job is submitted. A smoke-level workflow
    # is not compared, so its kind needs no comparator.
    for wf_id, entry in selected.items():
        try:
            level = lanes_mod.level_of(entry)
            lanes_mod.cadence_of(entry)
            lanes_mod.expect_of(entry)
            output_kinds.kind_of(entry)
            if level == "reference":
                comparator_for(entry)
        except ValueError as e:
            raise SystemExit(f"[{wf_id}] {e}; see manifest/workflows.json")
    # How the jobs are run. A per-commit lane submits them all at once, and
    # has no deadline. A nightly or weekly lane says how many it has on the
    # endpoint at a time, and stops submitting when its time runs out (see the
    # top of this file). Read here, before the start marker: a value that
    # cannot be used is a configuration error too.
    long_cadence = lanes_mod.is_long_cadence(lane)
    in_flight = min(len(selected), args.max_in_flight or lane_count(lane, "max_in_flight")
                    or len(selected))
    deadline = run_deadline(args.deadline_minutes, lane, long_cadence)
    if long_cadence:
        print(f"lane {lane_id}: {len(selected)} workflow(s), {in_flight} job(s) at a time, one "
              f"attempt each, and {(deadline - time.time()) / 60:.0f} min until the run's "
              "deadline")

    # A run that names its workflows is a re-test or a first look, not the
    # lane's run of the commit: it pushes no start marker, so the cron does not
    # count it as an attempt, and it never moves the previous-run pointer.
    named = args.workflows != "all"
    if not args.skip_publish:
        store.snapshot_manifest(args.commit, manifest)
    if not args.skip_publish and not named:
        # Pushed before any GPU job: if this run dies before it publishes (a
        # failed push, a killed job), the cron still counts the attempt and
        # stops after regression_skip.MAX_ATTEMPTS instead of re-running the
        # commit every tick. A marker that cannot be pushed stops the run here,
        # before it costs anything.
        store.mark_attempt(args.branch, args.commit, attempt_marker(args, lane_id, attempt))
        store.checkpoint(f"regression {args.branch}@{args.commit[:8]}: attempt {attempt} started")

    # A smoke run publishes a large output file only when it repeats: what each
    # smoke workflow wrote in its previous result, read here once for them all.
    # process_workflow finds it on `args`, like the lane and the endpoint.
    at_smoke = [wf_id for wf_id, entry in selected.items()
                if lanes_mod.level_of(entry) == "smoke"]
    args.previous_smoke = previous_smoke_files(store, args.branch, prior_entries, args.commit,
                                               at_smoke)

    results: dict[str, dict] = {}  # workflow id -> its cell, filled by the main thread
    position = {wf_id: i for i, wf_id in enumerate(selected)}
    stopping = threading.Event()
    tries = 2  # of a job: the second right away, inside run_workflow_job
    if long_cadence:
        # One attempt, and one pass at the end of the run for what can be tried
        # again: by then an outage of some minutes is over.
        tries = 1
        args.job_options = {"attempts": 1, "poll_interval_s": LONG_POLL_INTERVAL_S}
        args.clients = []

    def queue_wait(wf_id: str) -> float:
        return LONG_QUEUE_DEADLINE_S if long_cadence else queue_deadline_s(position[wf_id])

    def run_one(wf_id: str, again: bool = False) -> dict | None:
        """The cell of one workflow. With a deadline, the run first asks
        whether the job's worst case still fits before it. When it does not,
        no job is submitted: the cell says `not_reached`, or, in the pass at
        the end (`again`), the cell the workflow has stays (None)."""
        entry = selected[wf_id]
        if stopping.is_set():
            raise RuntimeError("the run is being stopped")
        if deadline is not None:
            left = deadline - time.time()
            need = job_worst_case_s(entry, defaults, queue_wait(wf_id), tries)
            if need > left:
                print(f"[{wf_id}] not started: its job can take {need / 60:.0f} min, and "
                      f"{max(left, 0) / 60:.0f} min are left before the run's deadline")
                return None if again else not_reached_cell(wf_id, entry, left, need)
        if again:
            # The folder is the second attempt's: what the first left in it goes.
            shutil.rmtree(Path(args.workdir) / wf_id, ignore_errors=True)
        return process_workflow(wf_id, entry, defaults, args, repo_root, store,
                                queue_wait(wf_id))

    def run_pass(wf_ids: list[str], again: bool = False):
        """Run these workflows, `in_flight` at a time, and yield each cell as
        it comes. A job is submitted only when one of those slots is free."""
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(wf_ids), in_flight)) as pool:
            futs = {pool.submit(run_one, wf_id, again): wf_id for wf_id in wf_ids}
            for fut in concurrent.futures.as_completed(futs):
                wf_id = futs[fut]
                try:
                    cell = fut.result()
                except Exception as e:
                    print(f"[{wf_id}] harness error: {e!r}")
                    cell = {"workflow_id": wf_id, "worker_status": "harness_error",
                            "verdict": "infra_error", "error": repr(e)[:1000]}
                    if wf_id in at_smoke:  # every cell of a smoke workflow names its level
                        cell["level"] = "smoke"
                if cell is not None:
                    results[wf_id] = cell
                    yield cell

    asked: dict = {"meta": None, "ranges": {}}  # what GitHub answered about the commit

    def commit_info() -> dict | None:
        if not asked["meta"]:
            asked["meta"] = commit_meta.fetch_commit_meta(args.commit)
        return asked["meta"]

    def commits_since(prev: str) -> dict | None:
        """The commits between `prev` and this commit, asked once. An answer
        without a count (GitHub did not answer) is asked for again."""
        if (asked["ranges"].get(prev) or {}).get("commits_between") is None:
            asked["ranges"][prev] = commit_meta.fetch_range(prev, args.commit)
        return asked["ranges"][prev]

    def summarise(partial: dict | None = None) -> dict:
        """The summary of the cells the run has. With `partial` it is one that
        is pushed before the run has them all, and says so."""
        cells = sorted(results.values(), key=lambda r: r["workflow_id"])
        verdicts = {r["workflow_id"]: r["verdict"] for r in cells}
        overall = ("fail" if any(v in ("fail", "execution_error") for v in verdicts.values())
                   else "pass")
        meta = commit_info()
        priors = index_builder.prior_verdicts(prior_entries, args.commit,
                                              (meta or {}).get("committed_ts"))
        for r in cells:
            prior = priors.get(r["workflow_id"]) or {}
            r["prior_verdict"] = prior.get("verdict")
            r["prior_commit"] = prior.get("commit")
        run_url, run_id = actions_run()
        summary = {"schema_version": 2, "branch": args.branch, "commit": args.commit,
                   "run_ts": int(time.time()), "overall": overall,
                   "has_infra_error": any(v == "infra_error" for v in verdicts.values()),
                   "lane": lane_id,
                   "attempt": attempt,
                   # The Actions run that tested the commit (null outside Actions):
                   # its page carries the full report, and the index links rows to it.
                   "run_url": run_url, "run_id": run_id,
                   # The worker image and host every workflow ran on (they share one endpoint).
                   "env": next((r.get("env") for r in cells if r.get("env")), None),
                   "commit_meta": meta,
                   "tested_range": (commits_since(prev_commit)
                                    if prev_commit and prev_commit != args.commit else None),
                   "workflows": {r["workflow_id"]: r for r in cells}}
        # What follows is said by a nightly or weekly lane only: a summary of a
        # per-commit lane keeps the keys it always had.
        if long_cadence:
            # The run covers many commits, and the lane's previous run says
            # little about one workflow: each red one gets its own range.
            ranges = own_ranges(cells, args.commit, commits_since)
            if ranges:
                summary["ranges"] = ranges
        if long_cadence and not args.skip_publish:
            # How much the run added to the results branch.
            summary["published_new_bytes"] = published_new_bytes(cells, args.previous_smoke)
        if partial:
            summary["partial"] = partial
        return summary

    # A nightly or weekly run that publishes pushes what it has every few
    # workflows, so a run that dies, is cancelled or loses its network leaves
    # its results behind. The index is written by the final publish alone, so
    # such a run is not in the history tables. A run that names its workflows
    # pushes nothing early: its summary is merged into the lane's at the end,
    # and a partial one would take that one's place.
    checkpoints = long_cadence and not named and not args.skip_publish

    def checkpoint():
        done = len(results)
        try:
            store.publish_summary(args.branch, args.commit,
                                  summarise({"done": done, "of": len(selected)}))
            store.checkpoint(f"regression {args.branch}@{args.commit[:8]}: "
                             f"{done} of {len(selected)} workflows")
        except Exception as e:  # noqa: BLE001 - a safety net: its failure must not end the run
            print(f"::warning::the checkpoint after {done} workflows failed, the run goes "
                  f"on: {e!r}")

    restore_handlers = install_stop_handler(args.clients, stopping) if long_cadence else None
    try:
        ran = saved = 0
        for cell in run_pass(list(selected)):
            if cell.get("worker_status") != NOT_REACHED:
                ran += 1
            if checkpoints and ran - saved >= CHECKPOINT_EVERY:
                checkpoint()
                saved = ran
        # The pass at the end: the one retry a job has, moved from "seconds
        # later" to "when everything else is done".
        again = [wf_id for wf_id in selected if long_cadence and worth_another_try(
            results[wf_id], run_record(Path(args.workdir), wf_id))]
        if again:
            if checkpoints and ran > saved:
                checkpoint()  # the pass can take an hour: what there is goes out first
            print(f"not tested, tried once more: {', '.join(again)}")
            for cell in run_pass(again, again=True):
                print(f"[{cell['workflow_id']}] second attempt: {cell['verdict']}")
    finally:
        if restore_handlers:
            restore_handlers()

    summary = summarise()
    cells = list(summary["workflows"].values())
    verdicts = {r["workflow_id"]: r["verdict"] for r in cells}
    overall = summary["overall"]

    if named and published and not args.skip_publish:
        # The commit already has a result on the lane: only the cells of the
        # workflows tested again are replaced, so one re-test does not cost the
        # results of all the others.
        summary = merge_retest(published, summary)
        print(f"{args.commit[:8]} already has a result on lane {lane_id}: kept, with "
              f"{', '.join(sorted(verdicts))} tested again (overall {summary['overall']})")
    elif named:
        summary["scope"] = sorted(verdicts)

    if not args.skip_publish:
        store.publish_summary(args.branch, args.commit, summary)
        # The previous-run pointer. A run that names its workflows is a re-test
        # or a first look, not the lane's run of the commit: it never moves it.
        # A full run of a per-commit lane moves it only when the run is fully
        # comparable: every workflow at least produced outputs. A full run of a
        # nightly or weekly lane always moves it: among that many workflows one
        # often has an error, and the pointer would stand still.
        comparable = all(r["verdict"] in ("pass", "fail", "no_baseline") for r in cells)
        if not named and (comparable or long_cadence):
            store.update_latest(args.branch,
                                {"commit": args.commit, "run_ts": summary["run_ts"],
                                 "workflows": sorted(verdicts)})
        # The run index is a derived file: regenerated on the freshest tree if
        # the push has to retry behind a concurrent publisher.
        entry = index_builder.entry_from_summary(summary, lane_id, layout)
        store.register_derived(lambda st, b=builder, e=entry: b.upsert([e]))
        store.finalize(f"regression {args.branch}@{args.commit[:8]}: {summary['overall']}")
    # After publishing, so the PR comment step never links to results that
    # did not make it to the store; before the exit code, so a red run has one.
    if args.summary_out:
        write_summary(Path(args.summary_out), summary)

    # A workflow that stayed untested fails a nightly or weekly run: nothing
    # tests the commit again before the lane's next run, a day or a week
    # later, and a red run is the one GitHub mails.
    untested = [r["workflow_id"] for r in cells if long_cadence and r["verdict"] == "infra_error"]
    for r in cells:
        v = r["verdict"]
        line = f"{r['workflow_id']}: {v} (worker={r['worker_status']})"
        if v in ("fail", "execution_error") or r["workflow_id"] in untested:
            print(f"::error::{line}")
        elif v in ("infra_error", "no_baseline"):
            print(f"::warning::{line}")
        else:
            print(line)
    print(f"overall: {overall}")
    if untested:
        print(f"not tested: {', '.join(untested)}. That fails a nightly or weekly run.")
    return 1 if overall == "fail" or untested else 0


if __name__ == "__main__":
    sys.exit(main())
