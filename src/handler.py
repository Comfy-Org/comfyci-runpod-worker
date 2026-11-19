"""RunPod serverless handler for ComfyUI CI regression runs.

Modes (input.mode):
  run          (default) — check out input.comfy_commit, install the custom-node
               packs in input.custom_nodes (if any), run input.workflow, return
               outputs base64-encoded. With input.fetch_missing, the models the
               volume does not hold are first downloaded to the worker's own
               disk (model_sync.fetch_local), input.fetch_streams files at once.
               With input.fetch_timeout_s that download is stopped when it has
               taken so many seconds, and the job answers with what it reached.
  sync_models  — sync input.models onto the network volume (see model_sync.py).

Every failure maps to a distinct output.status so the orchestrator can tell
infrastructure problems (checkout_error, missing_model, model_fetch_error,
model_access_error, server_start_timeout, custom_node_install_error) apart from
real execution failures
(execution_error, validation_error, custom_node_error) and never report an
infra flake as a code regression. The detail travels under `failure`, never
`error`: the RunPod SDK treats a top-level `error` as the job failing, rejects
a dict there (job-done 400) and drops it from the output.

Every `run` answer carries an `env` block naming the worker image (lane image
id, build sha, python, torch, cuda) and the host driver, so a result is always
attributable to the environment that produced it. A run that requested packs
also carries `custom_nodes`: per pack the sha, where it came from, the pip
time, and whether ComfyUI imported it. A run that fetched models carries
`model_fetch` (how many files and bytes, how fast, how many it found again)
and `timings.model_fetch_s`.

Output files travel in the answer, and RunPod allows an answer 10 MB. So the
files are sent up to MAX_INLINE_BYTES in all, the smallest first: one large
clip or mesh cannot hold back the picture. A file that is not sent is still
listed, with `truncated`, and the list is in name order either way.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
import traceback
from pathlib import Path

import runpod

import checkout
import comfy_runner
import custom_nodes
import model_sync

# RunPod's limit for the answer of a /run job is 10 MB. 7 MiB of files are
# 9.8 MB as base64, which leaves room for the rest of the answer.
MAX_INLINE_BYTES = 7 * 1024 * 1024
# Written at image build time (see Dockerfile): image id, build sha, python, cuda, torch spec.
IMAGE_INFO = Path(os.environ.get("IMAGE_INFO_FILE", "/comfy-config/image.json"))


def image_env(info_file: Path = IMAGE_INFO) -> dict:
    """The environment this worker is: what the image baked plus the host's driver."""
    env: dict = {}
    try:
        baked = json.loads(info_file.read_text())
        if isinstance(baked, dict):
            env.update(baked)
    except Exception:
        pass
    env["driver_version"] = comfy_runner.read_driver_version()
    return env


def fetch_streams(job_input: dict) -> int:
    """How many files a job downloads at once: input.fetch_streams, a whole
    number from 1 to 16, else model_sync.FETCH_STREAMS."""
    asked = job_input.get("fetch_streams")
    if isinstance(asked, int) and not isinstance(asked, bool) and 1 <= asked <= 16:
        return asked
    return model_sync.FETCH_STREAMS


def fetch_timeout_s(job_input: dict) -> float | None:
    """How many seconds a job's download may take: input.fetch_timeout_s, a
    number above 0. None for anything else, and for a job that names none:
    its download takes as long as it takes, as before the key existed."""
    asked = job_input.get("fetch_timeout_s")
    if (isinstance(asked, (int, float)) and not isinstance(asked, bool)
            and math.isfinite(asked) and asked > 0):
        return asked
    return None


def sent_inline(files: list[dict]) -> set[int]:
    """Which output files travel in the answer, as positions in `files`: the
    smallest first, while they fit MAX_INLINE_BYTES in all. Files of one size
    go in the order they are listed."""
    chosen: set[int] = set()
    total = 0
    for i in sorted(range(len(files)), key=lambda i: (files[i]["bytes"], i)):
        if total + files[i]["bytes"] > MAX_INLINE_BYTES:
            break  # every file after it is at least as large
        total += files[i]["bytes"]
        chosen.add(i)
    return chosen


def inline_outputs(files: list[dict]) -> list[dict]:
    """The `outputs` list of an answer, in the order of `files` (by name): each
    file with its size and sha256, and its bytes as base64 when it is sent
    (`sent_inline`). One that is not sent says `truncated`."""
    send = sent_inline(files)
    outputs = []
    for i, f in enumerate(files):
        entry = {"filename": f["filename"], "bytes": f["bytes"], "sha256": f["sha256"]}
        if i in send:
            entry["data_b64"] = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        else:
            entry["truncated"] = True
        outputs.append(entry)
    return outputs


def _run(job_input: dict) -> dict:
    commit = job_input.get("comfy_commit")
    workflow = job_input.get("workflow")
    if not commit or not workflow:
        return {"status": "bad_request", "failure": {"message": "comfy_commit and workflow are required"}}

    out: dict = {"status": "ok", "workflow_id": job_input.get("workflow_id"), "timings": {},
                 "lane": job_input.get("lane"), "env": image_env()}

    try:
        co = checkout.checkout_commit(commit, job_input.get("comfy_repo"))
    except Exception as e:
        return {**out, "status": "checkout_error", "failure": {"message": str(e)[:2000]}}
    out["commit_checked_out"] = co["commit_checked_out"]
    out["timings"]["checkout_s"] = co["checkout_s"]
    out["timings"]["pip_s"] = co["pip_s"]

    models = job_input.get("models", [])
    missing = model_sync.missing_models(models)
    if missing and job_input.get("fetch_missing") is True:
        # Input files stay on the volume: --input-directory points there. One
        # that is absent ends the job before anything is downloaded.
        fetch = [m for m in models
                 if f"{m['directory']}/{m['name']}" in missing and m["directory"] != "input"]
        missing = [key for key in missing if key.startswith("input/")]
        if fetch and not missing:
            t0 = time.monotonic()
            # Only a job that names a time limit hands one on.
            limit = fetch_timeout_s(job_input)
            limited = {} if limit is None else {"timeout_s": limit}
            try:
                report = model_sync.fetch_local(fetch, streams=fetch_streams(job_input),
                                                **limited)
            except Exception as e:
                return {**out, "status": "worker_error",
                        "failure": {"message": str(e)[:2000],
                                    "traceback": traceback.format_exc()[-3000:]}}
            out["timings"]["model_fetch_s"] = round(time.monotonic() - t0, 2)
            out["model_fetch"] = model_sync.fetch_brief(report)
            failed = model_sync.fetch_failure(report)
            if failed:
                return {**out, "status": failed[0], "failure": failed[1]}
    if missing:
        return {**out, "status": "missing_model", "failure": {"message": "models absent from volume",
                                                              "missing": missing}}

    # Always reconciled, even for a job without packs: a warm worker may still
    # hold the packs of an earlier job, and a job that requests none must run
    # the bare custom_nodes/ the image shipped.
    requested = list(job_input.get("custom_nodes") or [])
    try:
        installed = custom_nodes.prepare(requested)
    except custom_nodes.InstallError as e:
        return {**out, "status": "custom_node_install_error", "failure": {"message": str(e)[:2000]}}
    except Exception as e:
        return {**out, "status": "worker_error",
                "failure": {"message": str(e)[:2000], "traceback": traceback.format_exc()[-3000:]}}
    if requested:
        out["custom_nodes"] = installed["packs"]
        out["timings"]["custom_nodes_s"] = installed["seconds"]
        out["timings"]["custom_nodes_pip_s"] = installed["pip_s"]

    try:
        rec = comfy_runner.execute(
            workflow,
            timeout_s=float(job_input.get("timeout_s", 600)),
            extra_flags=custom_nodes.merge_flags(list(job_input.get("comfy_flags", [])), requested),
            required_custom_nodes=requested,
        )
    except comfy_runner.ExecutionFailure as e:
        if requested and e.detail.get("custom_nodes"):
            out["custom_nodes"] = custom_nodes.merge_status(out.get("custom_nodes"),
                                                            e.detail["custom_nodes"])
        return {**out, "status": e.status, "failure": e.detail}
    except Exception as e:
        return {**out, "status": "worker_error",
                "failure": {"message": str(e)[:2000], "traceback": traceback.format_exc()[-3000:]}}

    if requested:
        out["custom_nodes"] = custom_nodes.merge_status(out.get("custom_nodes"),
                                                        rec.get("custom_nodes"))
    out["comfy_version"] = rec.get("comfy_version")
    out["torch_version"] = rec.get("torch_version")
    out["python_version"] = rec.get("python_version")
    out["gpu_name"] = rec.get("gpu_name")
    out["vram_peak_mb"] = rec.get("vram_peak_mb")
    out["rss_peak_mb"] = rec.get("rss_peak_mb")
    out["validation"] = rec.get("validation")
    out["timings"]["server_start_s"] = rec.get("server_start_s")
    out["timings"]["prompt_exec_s"] = rec.get("prompt_exec_s")
    # The versions the server actually reported win over what the image claims.
    out["env"]["torch"] = rec.get("torch_version") or out["env"].get("torch")
    out["env"]["python"] = rec.get("python_version") or out["env"].get("python")
    out["env"]["runtime_backends"] = rec.get("runtime_backends") or None

    out["outputs"] = inline_outputs(rec["outputs"])
    return out


def handler(job):
    job_input = job.get("input") or {}
    mode = job_input.get("mode", "run")
    if mode == "sync_models":
        try:
            report = model_sync.sync(job_input.get("models", []), job_input.get("hf_token"))
            status = "ok" if not report["failed"] else "sync_partial"
            return {"status": status, "sync": report}
        except Exception as e:
            return {"status": "worker_error", "failure": {"message": str(e)[:2000]}}
    return _run(job_input)


runpod.serverless.start({"handler": handler})
