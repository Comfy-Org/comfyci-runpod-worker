"""RunPod serverless client: submit a workflow job, poll to completion, download outputs.

Infrastructure flakiness is isolated here: a job stuck IN_QUEUE beyond the
queue deadline is cancelled and retried once; anything that still fails maps
to status "infra_error" so the CI job can warn instead of going red.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests

RUNPOD_API = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 10
QUEUE_DEADLINE_S = 600
COLD_START_ALLOWANCE_S = 900

INFRA_STATUSES = {"checkout_error", "missing_model", "server_start_timeout",
                  "server_died", "worker_error", "infra_error", "no_outputs"}


class RunPodClient:
    def __init__(self, endpoint_id: str, api_key: str | None = None):
        self.base = f"{RUNPOD_API}/{endpoint_id}"
        key = api_key or os.environ["RUNPOD_API_KEY"]
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def run(self, job_input: dict) -> str:
        r = requests.post(f"{self.base}/run", headers=self.headers,
                          json={"input": job_input}, timeout=60)
        r.raise_for_status()
        return r.json()["id"]

    def status(self, job_id: str) -> dict:
        r = requests.get(f"{self.base}/status/{job_id}", headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def cancel(self, job_id: str):
        try:
            requests.post(f"{self.base}/cancel/{job_id}", headers=self.headers, timeout=60)
        except requests.RequestException:
            pass


def _poll(client: RunPodClient, job_id: str, exec_timeout_s: float) -> dict:
    """Poll until COMPLETED/FAILED. Returns RunPod's status body; raises TimeoutError."""
    t0 = time.monotonic()
    deadline = t0 + exec_timeout_s + COLD_START_ALLOWANCE_S
    while time.monotonic() < deadline:
        st = client.status(job_id)
        state = st.get("status")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return st
        if state == "IN_QUEUE" and time.monotonic() - t0 > QUEUE_DEADLINE_S:
            client.cancel(job_id)
            raise TimeoutError(f"job {job_id} stuck IN_QUEUE > {QUEUE_DEADLINE_S}s")
        time.sleep(POLL_INTERVAL_S)
    client.cancel(job_id)
    raise TimeoutError(f"job {job_id} exceeded deadline ({exec_timeout_s + COLD_START_ALLOWANCE_S}s)")


def _write_outputs(output_entries: list[dict], dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    inventory = []
    for f in output_entries:
        rec = {k: f[k] for k in ("filename", "bytes", "sha256") if k in f}
        if f.get("data_b64"):
            p = dest / f["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(f["data_b64"]))
        else:
            rec["truncated"] = True
        inventory.append(rec)
    return inventory


def run_workflow_job(client: RunPodClient, workflow_id: str, workflow_json: dict,
                     comfy_commit: str, models: list[dict], timeout_s: float,
                     comfy_flags: list[str], workdir: Path,
                     comfy_repo: str | None = None) -> dict:
    """Submit + poll one workflow with a single retry on infra flakes.

    Returns a run record: {"status": ..., "outputs_dir": ..., worker fields...}.
    Worker-reported statuses pass through; transport/queue problems become
    "infra_error".
    """
    job_input = {
        "mode": "run",
        "comfy_commit": comfy_commit,
        "comfy_repo": comfy_repo,
        "workflow_id": workflow_id,
        "workflow": workflow_json,
        "models": models,
        "timeout_s": timeout_s,
        "comfy_flags": comfy_flags,
    }
    last_err = None
    for attempt in (1, 2):
        try:
            job_id = client.run(job_input)
            st = _poll(client, job_id, timeout_s)
        except (requests.RequestException, TimeoutError) as e:
            last_err = str(e)[:1000]
            continue
        if st.get("status") != "COMPLETED":
            last_err = f"runpod status {st.get('status')}: {str(st.get('error'))[:1000]}"
            continue
        out = st.get("output") or {}
        record = {k: out.get(k) for k in
                  ("status", "error", "comfy_version", "torch_version", "gpu_name",
                   "vram_peak_mb", "validation", "timings", "commit_checked_out")}
        record["runpod_job_id"] = job_id
        record["attempt"] = attempt
        record["delay_s"] = st.get("delayTime")
        record["execution_ms"] = st.get("executionTime")
        outputs_dir = workdir / "outputs"
        record["outputs"] = _write_outputs(out.get("outputs") or [], outputs_dir)
        record["outputs_dir"] = str(outputs_dir)
        if record["status"] in INFRA_STATUSES and attempt == 1:
            last_err = f"worker infra status {record['status']}: {json.dumps(record.get('error'))[:500]}"
            continue
        return record
    return {"status": "infra_error", "error": {"message": last_err}, "attempt": 2}
