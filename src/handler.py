"""RunPod serverless handler for ComfyUI CI regression runs.

Modes (input.mode):
  run          (default) — check out input.comfy_commit, install the custom-node
               packs in input.custom_nodes (if any), run input.workflow, return
               outputs base64-encoded.
  sync_models  — sync input.models onto the network volume (see model_sync.py).

Every failure maps to a distinct output.status so the orchestrator can tell
infrastructure problems (checkout_error, missing_model, server_start_timeout,
custom_node_install_error) apart from real execution failures
(execution_error, validation_error, custom_node_error) and never report an
infra flake as a code regression. The detail travels under `failure`, never
`error`: the RunPod SDK treats a top-level `error` as the job failing, rejects
a dict there (job-done 400) and drops it from the output.

Every `run` answer carries an `env` block naming the worker image (lane image
id, build sha, python, torch, cuda) and the host driver, so a result is always
attributable to the environment that produced it. A run that requested packs
also carries `custom_nodes`: per pack the sha, where it came from, the pip
time, and whether ComfyUI imported it.
"""
from __future__ import annotations

import base64
import json
import os
import traceback
from pathlib import Path

import runpod

import checkout
import comfy_runner
import custom_nodes
import model_sync

MAX_INLINE_BYTES = 15 * 1024 * 1024  # stay under RunPod's ~20MB response cap after b64 overhead
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

    missing = model_sync.missing_models(job_input.get("models", []))
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

    total = 0
    outputs = []
    for f in rec["outputs"]:
        entry = {"filename": f["filename"], "bytes": f["bytes"], "sha256": f["sha256"]}
        total += f["bytes"]
        if total <= MAX_INLINE_BYTES:
            entry["data_b64"] = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        else:
            entry["truncated"] = True  # video-scale outputs: direct GCS upload lands in phase 5
        outputs.append(entry)
    out["outputs"] = outputs
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
