"""Generate (and optionally bless) golden baselines for a release tag.

Runs each selected workflow TWICE on RunPod at the tag's commit: r1 becomes the
golden outputs, r2-vs-r1 is the determinism noise floor that default regression
thresholds derive from. Results land under golden/<wf>/<tag>/ in storage;
blessing (--bless) flips golden/<wf>/current.json, which is what CI runs
compare against. Generate first, inspect on the dashboard, then bless.

Accepting an intentional numerics change (e.g. a kernel optimisation that
moved every pixel by a rounding error) is the same flow with a reason:
  --ref v0.38.0 --workflows flux_dev_t2i --reason "accept #16488 adaln drift" --bless
or, without spending GPU time, promote the outputs of a run that already
exists on the results branch (the golden is named after the commit):
  --from-run master/<sha> --workflows flux_dev_t2i --reason "..." --bless

Blessing and the run-index refresh are derived writers: they are re-applied
on the freshest tree if the publish has to retry behind another publisher.

Usage:
  python golden_baseline.py --ref v0.3.50 [--workflows all] [--bless] [--reason ...]
  python golden_baseline.py --ref v0.3.50 --bless-only
  python golden_baseline.py --from-run master/<commit> --workflows <id> --bless
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import index_builder
import lanes as lanes_mod
import output_kinds
import packs
import storage
from run_regression import (COMPARATORS, apply_seed_overrides, comparator_for, entry_value,
                            fetch_options, load_manifest)
from submit_and_poll import RunPodClient, run_workflow_job

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "https://github.com/Comfy-Org/ComfyUI"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_ref(repo_url: str, ref: str) -> str:
    """Commit sha for a tag/branch name; a full sha is returned as-is."""
    if SHA_RE.match(ref):
        return ref
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


def tag_name(ref: str) -> str:
    """Directory name for a golden: the tag itself, or a short sha."""
    return ref[:12] if SHA_RE.match(ref) else ref


def cuda_of(torch_version: str | None) -> str | None:
    """'2.11.0+cu128' -> '12.8'."""
    m = re.search(r"\+cu(\d+)(\d)$", torch_version or "")
    return f"{m.group(1)}.{m.group(2)}" if m else None


def blessed_record(tag: str, commit: str, run: dict, lane: str, source: str,
                   generated_by: str, reason: str | None, previous_tag: str | None,
                   comparator: output_kinds.Comparator | None = None) -> dict:
    """Environment identity of a golden. driver_version and image_sha are
    filled in once the worker reports them (null until then). `comparator` is
    the one of the workflow's output kind (default: image); a golden of another
    kind than image names its kind in `output_kind`."""
    cmp_ = comparator or COMPARATORS[output_kinds.DEFAULT_KIND]
    record = {
        "schema_version": 2,
        "tag": tag, "commit": commit, "source": source, "lane": lane,
        "gpu_name": run.get("gpu_name"),
        "torch_version": run.get("torch_version"),
        "python_version": run.get("python_version"),
        "cuda": cuda_of(run.get("torch_version")),
        "driver_version": run.get("driver_version"),
        "image_sha": (run.get("env") or {}).get("image_sha"),
        "output_sha256": cmp_.primary_sha(run),
        # The custom-node packs (sha, import status) the golden was generated with.
        "custom_nodes": run.get("custom_nodes"),
        "reason": reason,
        "previous_tag": previous_tag,
        "generated_by": generated_by, "generated_ts": int(time.time()),
    }
    if cmp_.kind != output_kinds.DEFAULT_KIND:
        record["output_kind"] = cmp_.kind
    return record


def golden_exists(store: storage.Storage, wf_id: str, tag: str) -> bool:
    return store.download_json(f"{store.prefix}/golden/{wf_id}/{tag}/blessed.json") is not None


def bless(store: storage.Storage, wf_id: str, tag: str, blessed_by: str,
          reason: str | None) -> dict:
    """Point golden/<wf>/current.json at `tag` and append to history.json.
    Idempotent per publish: safe to re-run as a derived writer."""
    base = f"{store.prefix}/golden/{wf_id}"
    previous = store.download_json(f"{base}/current.json") or {}
    blessed = store.download_json(f"{base}/{tag}/blessed.json") or {}
    current = {
        "tag": tag,
        "commit": blessed.get("commit"),
        "blessed_by": blessed_by, "blessed_ts": int(time.time()),
        "reason": reason,
        "supersedes": previous.get("tag") if previous.get("tag") != tag else previous.get("supersedes"),
        "source": blessed.get("source", "golden_run"),
        "output_sha256": blessed.get("output_sha256"),
    }
    store.upload_json(f"{base}/current.json", current)
    history = store.download_json(f"{base}/history.json") or {"blesses": []}
    history["blesses"].append(current)
    store.upload_json(f"{base}/history.json", history)
    return current


def left_in_golden(store: storage.Storage, wf_id: str, tag: str, outputs: Path,
                   comparator: output_kinds.Comparator) -> str | None:
    """Why `outputs` must not be written over golden `tag`; None when they may.
    The store adds files to a directory and removes none (--force). A compared
    file that only the earlier golden has would stay in the reference, and
    every run would then fail on the file count."""
    if not comparator.bless_gate:
        return None  # an image golden is written as it always was
    with tempfile.TemporaryDirectory() as old:
        store.fetch_golden_outputs(wf_id, tag, Path(old))
        stale = comparator.left_behind(Path(old), outputs)
    if not stale:
        return None
    return (f"golden {tag} holds {', '.join(stale[:5])}, which the new outputs do not have, and "
            f"writing over a golden removes no file. Remove it from golden/{wf_id}/{tag}/outputs "
            "on the results branch, or use another ref")


def unpublished_files(comparison: dict | None, sent: bool | None = None) -> list[str]:
    """The output files a run's comparison.json lists as not published
    (`"published": false` in `smoke.files`): none for a reference-level run.
    With `sent`, only those the worker sent (True: the run held them back) or
    did not send (False: no run has a copy of them)."""
    files = ((comparison or {}).get("smoke") or {}).get("files")
    return [str(f.get("name")) for f in (files if isinstance(files, list) else [])
            if isinstance(f, dict) and f.get("published") is False
            and (sent is None or (f.get("sent") is not False) is sent)]


def promote_run(store: storage.Storage, branch: str, commit: str, wf_id: str, tag: str,
                lane: str, generated_by: str, reason: str | None, workdir: Path,
                comparator: output_kinds.Comparator | None = None) -> bool:
    """Make an existing run's outputs the golden candidate for `tag` without a
    GPU run. The noise floor is inherited from the previous golden (same-env
    determinism has been bit-exact so far) and marked as such. `comparator` is
    the one of the workflow's output kind (default: image)."""
    cmp_ = comparator or COMPARATORS[output_kinds.DEFAULT_KIND]
    prefix = store.prefix
    run = store.download_json(f"{prefix}/runs/{branch}/{commit}/{wf_id}/run.json")
    outputs = workdir / wf_id / "outputs"
    n = store.fetch_run_outputs(branch, commit, wf_id, outputs)
    if not run or not n:
        print(f"::error::[{wf_id}] no published outputs for {branch}/{commit}")
        return False
    # A smoke run publishes a large output file only when it repeats. A run
    # that held one back is not whole on the results branch, and a reference
    # made from it would lack that file. A file the worker did not send is
    # held back by every run: no later one has it either.
    comparison = store.download_json(f"{prefix}/runs/{branch}/{commit}/{wf_id}/comparison.json")
    unsent = unpublished_files(comparison, sent=False)
    if unsent:
        print(f"::error::[{wf_id}] {branch}/{commit} cannot be a reference: the worker did not "
              f"send {', '.join(unsent[:5])} (over its size limit), so no run can publish it. "
              "Make the output smaller, or generate the reference with a golden run")
        return False
    held = unpublished_files(comparison, sent=True)
    if held:
        print(f"::error::[{wf_id}] {branch}/{commit} cannot be a reference: it was a smoke run "
              f"that did not publish {', '.join(held[:5])}. Use a later run that shows the "
              "same output: that one publishes every file")
        return False
    # An inherited floor describes the earlier reference. What the bless gate
    # asks about the output itself is read from the outputs that are promoted.
    own = cmp_.describe(outputs) if cmp_.bless_gate else {}
    # A compared file that the run listed and did not send is not among them:
    # the reference would lack it, and no later run could pass.
    lost = cmp_.transfer_error(run) or own.get("error")
    if lost:
        print(f"::error::[{wf_id}] {branch}/{commit} cannot be a reference: {lost}")
        return False
    previous = store.download_json(f"{prefix}/golden/{wf_id}/current.json") or {}
    floor = None
    if previous.get("tag"):
        floor = store.noise_floor(wf_id, previous["tag"])
        if floor and output_kinds.metrics_kind(floor) != cmp_.kind:
            floor = None  # the earlier reference was of another output kind
    if not floor:
        floor = cmp_.zero_floor()
    floor = {**floor, **own, "inherited_from": previous.get("tag"), "measured": False}
    # A floor that was not measured is held to no limit of the entry, so none is passed.
    why = cmp_.blessable(floor, {}) if cmp_.bless_gate and previous.get("tag") == tag else None
    if why:
        # As for a generated golden: current.json points at this directory, and
        # what may not be blessed must not replace the reference in it.
        print(f"::error::[{wf_id}] golden {tag} is not blessed: {why}. It is the current "
              "reference, so its files are left as they are")
        return False
    why = left_in_golden(store, wf_id, tag, outputs, cmp_)
    if why:
        print(f"::error::[{wf_id}] {why}")
        return False

    base = f"{prefix}/golden/{wf_id}/{tag}"
    store.upload_dir(f"{base}/outputs", outputs)
    store.upload_json(f"{base}/run_r1.json", {**run, "promoted_from": f"{branch}/{commit}"})
    store.upload_json(f"{base}/noise_floor.json", floor)
    store.upload_json(f"{base}/blessed.json",
                      blessed_record(tag, commit, run, lane, "from_run", generated_by, reason,
                                     previous.get("tag"), cmp_))
    print(f"[{wf_id}] promoted {branch}/{commit[:8]} outputs to golden {tag}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifest" / "workflows.json"))
    ap.add_argument("--ref", default=None,
                    help="release tag, branch, or full commit sha to build goldens from "
                         "(optional with --from-run, where it must resolve to that commit)")
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    ap.add_argument("--workdir", default="./golden-work")
    ap.add_argument("--workflows", default="all")
    ap.add_argument("--bless", action="store_true", help="also flip current.json after generating")
    ap.add_argument("--bless-only", action="store_true", help="no runs; just point current.json at --ref")
    ap.add_argument("--from-run", default=None, metavar="BRANCH/COMMIT",
                    help="promote an existing run's outputs instead of running on RunPod")
    ap.add_argument("--force", action="store_true", help="overwrite an existing golden/<wf>/<tag>/")
    ap.add_argument("--reason", default=None, help="why this baseline is being (re)blessed")
    ap.add_argument("--lane", default=None, help="lane id from manifest/lanes.json")
    ap.add_argument("--branch", default="master", help="branch whose run index to refresh")
    ap.add_argument("--blessed-by", default=os.environ.get("GITHUB_ACTOR", "manual"))
    storage.add_storage_args(ap)
    args = ap.parse_args()

    lanes = index_builder.load_lanes()
    if args.lane:
        if args.lane not in lanes["lanes"]:
            raise SystemExit(f"unknown lane {args.lane!r}; see manifest/lanes.json")
        lane = args.lane
        args.prefix = lanes["lanes"][lane].get("prefix", args.prefix)
    else:
        lane = index_builder.lane_for_prefix(lanes, args.prefix)
        if not lane:
            raise SystemExit(f"no lane in manifest/lanes.json has prefix {args.prefix!r}; pass --lane")
    layout = lanes["lanes"][lane].get("layout", 1)
    # Generating runs on the lane's own endpoint; blessing and promoting touch storage only.
    default_endpoint = index_builder.lane_endpoint(lanes, lane)
    store = storage.from_args(args)

    manifest_path = Path(args.manifest).resolve()
    repo_root = manifest_path.parent.parent
    manifest = load_manifest(manifest_path)
    defaults = manifest["defaults"]
    selected = {
        wf_id: e for wf_id, e in manifest["workflows"].items()
        if (args.workflows == "all" and e.get("enabled")) or wf_id in args.workflows.split(",")
    }
    # A golden only means something on a lane the workflow runs on: the lane
    # filter applies to generating, promoting and blessing alike. It knows the
    # cadence too, so `all` on a lane takes that lane's workflows. On a retired
    # lane only the manifest's `lanes` list counts.
    for wf_id, e in list(selected.items()):
        try:
            why_not = lanes_mod.why_not_on_lane(e, lane, lanes["lanes"][lane])
        except ValueError as err:
            raise SystemExit(f"[{wf_id}] {err}; see manifest/workflows.json")
        if why_not:
            print(f"[{wf_id}] not run on lane {lane} ({why_not}); skipped")
            del selected[wf_id]
    # A smoke-level workflow is not compared, so it has no golden to generate,
    # promote or bless: `all` leaves it out, and naming it is a mistake.
    at_smoke = []
    for wf_id, e in selected.items():
        try:
            if lanes_mod.level_of(e) == "smoke":
                at_smoke.append(wf_id)
        except ValueError as err:
            raise SystemExit(f"[{wf_id}] {err}; see manifest/workflows.json")
    if at_smoke and args.workflows != "all":
        raise SystemExit("; ".join(f"{wf_id} is a smoke-level workflow: it has no reference"
                                   for wf_id in at_smoke))
    for wf_id in at_smoke:
        print(f"::notice::[{wf_id}] is a smoke-level workflow: it has no reference; skipped")
        del selected[wf_id]
    if not selected:
        raise SystemExit(f"no workflows selected for lane {lane}")
    for wf_id, e in selected.items():
        try:
            comparator_for(e)
        except ValueError as err:
            raise SystemExit(f"[{wf_id}] {err}; see manifest/workflows.json")

    from_branch = from_commit = None
    if args.from_run:
        from_branch, _, from_commit = args.from_run.rpartition("/")
        if not (from_branch and SHA_RE.match(from_commit)):
            raise SystemExit("--from-run expects BRANCH/<40-hex commit>")
        if args.ref and resolve_ref(args.repo_url, args.ref) != from_commit:
            raise SystemExit(f"--ref {args.ref} does not resolve to the --from-run commit")
        tag = tag_name(args.ref or from_commit)
    elif args.ref:
        tag = tag_name(args.ref)
    else:
        raise SystemExit("--ref is required unless --from-run is given")

    blessed: list[str] = []

    def plan_bless(wf_id: str) -> bool:
        """Approve golden `tag` of a workflow, unless its kind says it must not be."""
        cmp_ = comparator_for(selected[wf_id])
        why = (cmp_.blessable(store.noise_floor(wf_id, tag), selected[wf_id])
               if cmp_.bless_gate else None)
        if why:
            print(f"::error::[{wf_id}] golden {tag} is not blessed: {why}")
            return False
        store.register_derived(lambda st, w=wf_id: bless(st, w, tag, args.blessed_by, args.reason))
        blessed.append(wf_id)
        print(f"[{wf_id}] blessing golden {tag}")
        return True

    def finish(message: str, failed: list[str]) -> int:
        if blessed:
            builder = index_builder.IndexBuilder(store, args.branch, lane, layout, lanes)
            store.register_derived(lambda st, b=builder, w=tuple(blessed): b.refresh(w))
        store.finalize(message)
        return 1 if failed else 0

    if args.bless_only:
        missing = [wf_id for wf_id in selected if not golden_exists(store, wf_id, tag)]
        if missing:
            print(f"::error::no generated golden {tag} for: {', '.join(missing)}; nothing blessed")
            return 1
        refused = [wf_id for wf_id in selected if not plan_bless(wf_id)]
        return finish(f"bless golden {tag} ({', '.join(sorted(selected))})", refused)

    failed: list[str] = []
    if args.from_run:
        for wf_id in selected:
            if golden_exists(store, wf_id, tag) and not args.force:
                print(f"::error::[{wf_id}] golden {tag} already exists (use --force to overwrite)")
                failed.append(wf_id)
                continue
            if promote_run(store, from_branch, from_commit, wf_id, tag, lane, args.blessed_by,
                           args.reason, Path(args.workdir), comparator_for(selected[wf_id])):
                if args.bless and not plan_bless(wf_id):
                    failed.append(wf_id)
            else:
                failed.append(wf_id)
        return finish(f"golden {tag} from {args.from_run[:20]} ({', '.join(sorted(selected))})",
                      failed)

    # Blessing and promoting stay possible on a retired lane (storage only);
    # generating would dispatch GPU work to an endpoint that is scaled to zero.
    if lanes_mod.is_retired(lanes["lanes"][lane]):
        raise SystemExit(f"lane {lane!r} is retired: its goldens can still be blessed or "
                         "promoted from a run, but not generated; pass lane= with an active lane")
    if not default_endpoint:
        raise SystemExit(f"lane {lane!r} has no endpoint (endpoint_id is null in "
                         "manifest/lanes.json): create one for its image and commit the id "
                         "before generating goldens for it")
    commit = resolve_ref(args.repo_url, args.ref)
    print(f"golden ref {args.ref} -> {commit}")

    for wf_id, entry in selected.items():
        if golden_exists(store, wf_id, tag) and not args.force:
            print(f"::error::[{wf_id}] golden {tag} already exists (use --force to overwrite)")
            failed.append(wf_id)
            continue
        workflow = json.loads((repo_root / entry["workflow_path"]).read_text(encoding="utf-8"))
        workflow = apply_seed_overrides(workflow, entry.get("seed_overrides"))
        endpoint_id = entry.get("endpoint_id") or default_endpoint
        client = RunPodClient(endpoint_id)
        wf_dir = Path(args.workdir) / wf_id
        runs = {}
        for r in (1, 2):
            print(f"[{wf_id}] golden run r{r} @ {args.ref} (lane {lane}, endpoint {endpoint_id})")
            rec = run_workflow_job(
                client, wf_id, workflow, comfy_commit=commit,
                models=entry.get("models", []),
                timeout_s=entry_value(entry, defaults, "timeout_s"),
                comfy_flags=entry_value(entry, defaults, "comfy_flags"),
                workdir=wf_dir / f"r{r}", comfy_repo=args.repo_url, lane=lane,
                custom_nodes=packs.records_for(entry, repo_root),
                # On a lane whose jobs fetch their models, both runs are told to.
                **fetch_options(entry, lanes["lanes"][lane]))
            runs[r] = rec
            if rec.get("status") != "ok":
                print(f"::error::[{wf_id}] golden r{r} failed: {rec.get('status')} "
                      f"{json.dumps(rec.get('error'))[:500]}")
                break
        if runs.get(1, {}).get("status") != "ok" or runs.get(2, {}).get("status") != "ok":
            failed.append(wf_id)
            continue

        cmp_ = comparator_for(entry)
        # A compared file that a run listed and did not send is in neither
        # directory: the files that are left would compare as identical.
        lost = cmp_.transfer_error(runs[1]) or cmp_.transfer_error(runs[2])
        noise_floor = ({"kind": cmp_.kind, "error": lost} if lost else
                       cmp_.compare(Path(runs[1]["outputs_dir"]), Path(runs[2]["outputs_dir"])))
        # Numbers only: the two lines of a text difference are model output.
        shown = json.dumps(output_kinds.loggable(noise_floor))[:200]
        print(f"[{wf_id}] noise floor: {'identical' if noise_floor.get('identical') else shown}")

        previous = store.download_json(f"{args.prefix}/golden/{wf_id}/current.json") or {}
        base = f"{args.prefix}/golden/{wf_id}/{tag}"
        why = (cmp_.blessable({**noise_floor, "measured": True}, entry)
               if cmp_.bless_gate and previous.get("tag") == tag else None)
        if why:
            # current.json points at this directory: a golden that may not be
            # blessed must not replace the reference every run is compared with.
            print(f"::error::[{wf_id}] golden {tag} is not blessed: {why}. It is the current "
                  "reference, so its files are left as they are")
            failed.append(wf_id)
            continue
        why = left_in_golden(store, wf_id, tag, Path(runs[1]["outputs_dir"]), cmp_)
        if why:
            print(f"::error::[{wf_id}] {why}")
            failed.append(wf_id)
            continue
        store.upload_dir(f"{base}/outputs", Path(runs[1]["outputs_dir"]))
        store.upload_json(f"{base}/run_r1.json",
                          {k: v for k, v in runs[1].items() if k != "outputs_dir"})
        store.upload_json(f"{base}/run_r2.json",
                          {k: v for k, v in runs[2].items() if k != "outputs_dir"})
        store.upload_json(f"{base}/noise_floor.json", {**noise_floor, "measured": True})
        store.upload_json(f"{base}/blessed.json",
                          blessed_record(tag, commit, runs[1], lane, "golden_run",
                                         args.blessed_by, args.reason, previous.get("tag"), cmp_))
        print(f"[{wf_id}] golden uploaded to {base}/")
        if args.bless and not plan_bless(wf_id):
            failed.append(wf_id)

    return finish(f"golden {tag} ({', '.join(sorted(selected))})", failed)


if __name__ == "__main__":
    sys.exit(main())
