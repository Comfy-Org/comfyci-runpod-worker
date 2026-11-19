"""Submit a model-sync job to the primary lane's RunPod endpoint and wait for it.

The download itself happens inside the worker (src/model_sync.py, mode
sync_models of src/handler.py, which every lane image ships) so the files land
on the network volume all lanes share. The job runs on the endpoint of the
`primary` lane in manifest/lanes.json, so it follows the primary when that
changes and never depends on a retired lane's endpoint; RUNPOD_ENDPOINT_ID is
only the fallback for a registry whose primary has no endpoint. One endpoint
writes the volume at a time: the workflow serialises syncs, and the sync state
file is only updated after a completed download.

Only what a lane reads from the volume is synced, the same files whose sizes
the manifest lint adds up (lanes.volume_models). That is every model of a
workflow that runs on a lane which keeps its models on the volume. A workflow
that runs only on lanes with `"model_storage": "job"` has its models fetched
by each job, so only its input files (`"directory": "input"`) are synced:
ComfyUI reads those from the volume on every lane. A workflow that no lane
with a volume runs is left out.

Run automatically by .github/workflows/sync-models.yml whenever manifest/**
changes; safe to run manually any time (idempotent).

Usage: python sync_models.py [--manifest ../manifest/workflows.json] [--workflows all]
Env: RUNPOD_API_KEY, optionally RUNPOD_ENDPOINT_ID (fallback, see above) and
HF_TOKEN (or set it as an endpoint env var so it never rides in request payloads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import index_builder
import lanes as lanes_mod
from run_regression import load_manifest
from submit_and_poll import RunPodClient, _poll

HERE = Path(__file__).resolve().parent
SYNC_TIMEOUT_S = 7200  # large model downloads


def sync_endpoint(lanes: dict | None, env=None) -> tuple[str, str]:
    """(endpoint id, where it came from) for the sync job: the primary lane's
    endpoint, else RUNPOD_ENDPOINT_ID."""
    env = os.environ if env is None else env
    primary = (lanes or {}).get("primary")
    endpoint = (index_builder.lane_config(lanes or {}, primary).get("endpoint_id")
                if primary else None)
    if endpoint:
        return endpoint, f"primary lane {primary}"
    if env.get("RUNPOD_ENDPOINT_ID"):
        return env["RUNPOD_ENDPOINT_ID"], (f"RUNPOD_ENDPOINT_ID (primary lane {primary!r} has "
                                           "no endpoint_id in manifest/lanes.json)")
    raise SystemExit(f"no endpoint to sync on: primary lane {primary!r} has no endpoint_id in "
                     "manifest/lanes.json and RUNPOD_ENDPOINT_ID is not set")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "manifest" / "workflows.json"))
    ap.add_argument("--workflows", default="all", help="csv of workflow ids, or 'all'")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    try:
        lanes = index_builder.load_lanes()
    except (OSError, ValueError):
        lanes = None
    # The volume holds what a lane reads from it, and nothing else: the
    # manifest lint adds up the sizes of the same files (lanes.volume_models),
    # so what it lets through is what arrives here. Of a workflow whose jobs
    # fetch their models that is its input files. A workflow that no lane with
    # a volume runs is left out. Without a registry: every model.
    try:
        on_volume = None if lanes is None else lanes_mod.volume_models(manifest, lanes)
    except ValueError as e:
        raise SystemExit(f"{e}; see manifest/workflows.json")
    models, seen, left_out, inputs_only = [], set(), [], []
    for wf_id, entry in manifest["workflows"].items():
        if args.workflows != "all" and wf_id not in args.workflows.split(","):
            continue
        # Sync disabled workflows too: the volume should be ready before enabling.
        wanted = entry.get("models", [])
        if on_volume is not None and len(on_volume.get(wf_id, [])) < len(wanted):
            (inputs_only if wf_id in on_volume else left_out).append(wf_id)
            wanted = on_volume.get(wf_id, [])
        for m in wanted:
            key = f"{m['directory']}/{m['name']}"
            if key not in seen:
                seen.add(key)
                models.append(m)
    if left_out:
        print(f"not synced, no lane reads their models from the volume: {', '.join(left_out)}")
    if inputs_only:
        print("only the input files are synced, each job fetches the models: "
              f"{', '.join(inputs_only)}")

    if not models:
        print("no models in manifest selection")
        return 0

    endpoint_id, source = sync_endpoint(lanes)
    client = RunPodClient(endpoint_id)
    job_input = {"mode": "sync_models", "models": models}
    if os.environ.get("HF_TOKEN"):
        job_input["hf_token"] = os.environ["HF_TOKEN"]
    print(f"syncing {len(models)} models to the network volume via endpoint {endpoint_id} "
          f"({source})...")
    job_id = client.run(job_input)
    st = _poll(client, job_id, SYNC_TIMEOUT_S)
    if st.get("status") != "COMPLETED":
        print(f"::error::sync job {st.get('status')}: {str(st.get('error'))[:1000]}")
        return 1
    out = st.get("output") or {}
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
