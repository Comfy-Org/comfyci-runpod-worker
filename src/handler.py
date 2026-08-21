"""RunPod serverless handler for ComfyUI CI regression runs.

Modes (input.mode):
  run          (default) — check out input.comfy_commit, run input.workflow,
               return outputs base64-encoded.
  sync_models  — sync input.models onto the network volume (see model_sync.py).

Every failure maps to a distinct output.status so the orchestrator can tell
infrastructure problems (checkout_error, missing_model, server_start_timeout)
apart from real execution failures (execution_error, validation_error) and
never report an infra flake as a code regression.
"""
from __future__ import annotations

import base64
import traceback
from pathlib import Path

import runpod

import checkout
import comfy_runner
import model_sync

MAX_INLINE_BYTES = 15 * 1024 * 1024  # stay under RunPod's ~20MB response cap after b64 overhead


def _run(job_input: dict) -> dict:
    commit = job_input.get("comfy_commit")
    workflow = job_input.get("workflow")
    if not commit or not workflow:
        return {"status": "bad_request", "error": {"message": "comfy_commit and workflow are required"}}

    out: dict = {"status": "ok", "workflow_id": job_input.get("workflow_id"), "timings": {}}

    try:
        co = checkout.checkout_commit(commit, job_input.get("comfy_repo"))
    except Exception as e:
        return {"status": "checkout_error", "error": {"message": str(e)[:2000]}}
    out["commit_checked_out"] = co["commit_checked_out"]
    out["timings"]["checkout_s"] = co["checkout_s"]
    out["timings"]["pip_s"] = co["pip_s"]

    missing = model_sync.missing_models(job_input.get("models", []))
    if missing:
        return {**out, "status": "missing_model", "error": {"message": "models absent from volume",
                                                            "missing": missing}}

    try:
        rec = comfy_runner.execute(
            workflow,
            timeout_s=float(job_input.get("timeout_s", 600)),
            extra_flags=list(job_input.get("comfy_flags", [])),
        )
    except comfy_runner.ExecutionFailure as e:
        return {**out, "status": e.status, "error": e.detail}
    except Exception as e:
        return {**out, "status": "worker_error",
                "error": {"message": str(e)[:2000], "traceback": traceback.format_exc()[-3000:]}}

    out["comfy_version"] = rec.get("comfy_version")
    out["torch_version"] = rec.get("torch_version")
    out["python_version"] = rec.get("python_version")
    out["gpu_name"] = rec.get("gpu_name")
    out["vram_peak_mb"] = rec.get("vram_peak_mb")
    out["rss_peak_mb"] = rec.get("rss_peak_mb")
    out["validation"] = rec.get("validation")
    out["timings"]["server_start_s"] = rec.get("server_start_s")
    out["timings"]["prompt_exec_s"] = rec.get("prompt_exec_s")

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
            return {"status": "worker_error", "error": {"message": str(e)[:2000]}}
    return _run(job_input)


runpod.serverless.start({"handler": handler})
