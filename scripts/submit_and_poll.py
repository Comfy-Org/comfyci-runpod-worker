"""RunPod serverless client: submit a workflow job, poll to completion, download outputs.

Infrastructure flakiness is isolated here: a job stuck IN_QUEUE beyond the
queue deadline is cancelled and retried once; anything that still fails maps
to status "infra_error" so the CI job can warn instead of going red.

A nightly or weekly run asks for one attempt (`attempts=1`) and decides itself,
at its end, what to try again. Its failure record then says when the job used
up its whole time (`"reason": "timed_out"`), which is not worth a second try.
The client remembers the jobs it has submitted that RunPod has not ended, so a
run that is stopped can cancel them.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests

import output_kinds

RUNPOD_API = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 10
QUEUE_DEADLINE_S = 600
COLD_START_ALLOWANCE_S = 900
# The states in which RunPod has ended a job: nothing is left to cancel.
ENDED_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")
# `reason` in the failure record of a job that used up its whole time: RunPod
# ended it at the endpoint's limit, or the runner gave up at its poll deadline.
TIMED_OUT = "timed_out"
# Seconds one cancel call may wait for RunPod, to connect and again to be
# answered, when a stopped run cancels its jobs: GitHub kills the step 10
# seconds after its first signal.
STOP_CANCEL_TIMEOUT_S = 3

# custom_node_install_error: a pack could not be fetched or its requirements
# could not be pip-installed on the worker (network, not the commit under test).
INFRA_STATUSES = {"checkout_error", "missing_model", "server_start_timeout",
                  "server_died", "worker_error", "infra_error", "no_outputs",
                  "bad_request", "custom_node_install_error"}
# bad_request is a deterministic orchestrator/handler contract error: retrying
# it would only buy a second identical failure.
RETRYABLE_STATUSES = INFRA_STATUSES - {"bad_request"}


def _error_detail(err) -> dict | None:
    """Normalise a job-level error: RunPod returns the handler's dict as a
    JSON-encoded string, so decode it back when possible."""
    if err is None:
        return None
    if isinstance(err, dict):
        return err
    if isinstance(err, str):
        try:
            decoded = json.loads(err)
            if isinstance(decoded, dict):
                return decoded
        except ValueError:
            pass
    return {"message": str(err)[:2000]}


class RunPodClient:
    def __init__(self, endpoint_id: str, api_key: str | None = None):
        self.base = f"{RUNPOD_API}/{endpoint_id}"
        key = api_key or os.environ["RUNPOD_API_KEY"]
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        # The jobs this client submitted that RunPod has not ended, as far as
        # the client was told: what a run that is stopped has to cancel.
        self.live: set[str] = set()

    def run(self, job_input: dict) -> str:
        r = requests.post(f"{self.base}/run", headers=self.headers,
                          json={"input": job_input}, timeout=60)
        r.raise_for_status()
        job_id = r.json()["id"]
        self.live.add(job_id)
        return job_id

    def status(self, job_id: str) -> dict:
        r = requests.get(f"{self.base}/status/{job_id}", headers=self.headers, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("status") in ENDED_STATES:
            self.live.discard(job_id)
        return body

    def cancel(self, job_id: str, timeout: float = 60):
        self.live.discard(job_id)
        try:
            requests.post(f"{self.base}/cancel/{job_id}", headers=self.headers, timeout=timeout)
        except requests.RequestException:
            pass

    def cancel_live(self, timeout: float = STOP_CANCEL_TIMEOUT_S) -> list[str]:
        """Cancel every job of this client that RunPod has not ended, one call
        each, and return their ids. For a run that is being stopped: nobody
        would read those jobs, and they would be billed to their end."""
        job_ids = sorted(self.live)
        for job_id in job_ids:
            self.cancel(job_id, timeout)
        return job_ids


class JobTimeout(TimeoutError):
    """The runner gave up on a job at its poll deadline: it used up its time."""


def poll_deadline_s(exec_timeout_s: float, queue_deadline_s: float = QUEUE_DEADLINE_S) -> float:
    """Seconds after its submission at which the runner gives up on a job. A
    job that may wait longer in the queue gets that much more, so it is not cut
    short afterwards for having stood in line."""
    return (exec_timeout_s + COLD_START_ALLOWANCE_S
            + max(0, queue_deadline_s - QUEUE_DEADLINE_S))


def _poll(client: RunPodClient, job_id: str, exec_timeout_s: float,
          queue_deadline_s: float = QUEUE_DEADLINE_S,
          poll_interval_s: float = POLL_INTERVAL_S) -> dict:
    """Poll until COMPLETED/FAILED. Returns RunPod's status body; raises TimeoutError
    (JobTimeout when the job was past its poll deadline, not waiting in the queue).

    Transient network failures (runner DNS blips etc.) are tolerated for up to
    ~5 minutes before giving up — the job keeps running server-side either way.
    """
    t0 = time.monotonic()
    allowed_s = poll_deadline_s(exec_timeout_s, queue_deadline_s)
    deadline = t0 + allowed_s
    net_errors = 0
    while time.monotonic() < deadline:
        try:
            st = client.status(job_id)
            net_errors = 0
        except requests.RequestException:
            net_errors += 1
            if net_errors > 30:
                raise
            time.sleep(poll_interval_s)
            continue
        state = st.get("status")
        if state in ENDED_STATES:
            return st
        if state == "IN_QUEUE" and time.monotonic() - t0 > queue_deadline_s:
            client.cancel(job_id)
            raise TimeoutError(f"job {job_id} stuck IN_QUEUE > {queue_deadline_s}s")
        time.sleep(poll_interval_s)
    client.cancel(job_id)
    raise JobTimeout(f"job {job_id} exceeded deadline ({allowed_s}s)")


def _write_outputs(output_entries: list[dict], dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    inventory = []
    for f in output_entries:
        rec = {k: f[k] for k in ("filename", "bytes", "sha256") if k in f}
        data = f.get("data_b64")
        # "" is an empty file. An empty text is a result, so it is written. Any
        # other empty file stays listed as not sent, as it always was: an empty
        # PNG on disk would change what an image run records.
        empty_text = data == "" and str(f.get("filename", "")).lower().endswith(
            output_kinds.TEXT_EXTENSIONS)
        if data or empty_text:
            p = dest / f["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(data))
        else:
            rec["truncated"] = True
        inventory.append(rec)
    return inventory


def run_workflow_job(client: RunPodClient, workflow_id: str, workflow_json: dict,
                     comfy_commit: str, models: list[dict], timeout_s: float,
                     comfy_flags: list[str], workdir: Path,
                     comfy_repo: str | None = None, lane: str | None = None,
                     custom_nodes: list[dict] | None = None,
                     queue_deadline_s: float = QUEUE_DEADLINE_S, attempts: int = 2,
                     poll_interval_s: float = POLL_INTERVAL_S) -> dict:
    """Submit + poll one workflow with a single retry on infra flakes.

    Returns a run record: {"status": ..., "outputs_dir": ..., worker fields...}.
    Worker-reported statuses pass through; transport/queue problems become
    "infra_error". `lane` is echoed by the worker so a record names the lane
    it was produced for. `custom_nodes` are the pack records (scripts/packs.py)
    the worker installs before the server starts; the record carries the
    worker's per-pack install and import report under the same key.
    `queue_deadline_s` is how long the job may wait in RunPod's queue before it
    is cancelled (and submitted once more). `poll_interval_s` is the pause
    between two looks at the job.

    `attempts` is how often the job is submitted at most: 2, the retry above.
    With 1 the caller decides about another attempt, so the failure record
    tells it what it needs for that: `"reason": "timed_out"` when the job used
    up its whole time (RunPod ended it, or the runner gave up at the poll
    deadline). Another attempt of such a job can cost as much again.
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
        "lane": lane,
        "custom_nodes": list(custom_nodes or []),
    }
    last_err = None
    spent = False  # the last attempt used up its whole time
    for attempt in range(1, attempts + 1):
        try:
            job_id = client.run(job_input)
            st = _poll(client, job_id, timeout_s, queue_deadline_s, poll_interval_s)
        except (requests.RequestException, TimeoutError) as e:
            last_err = str(e)[:1000]
            spent = isinstance(e, JobTimeout)
            continue
        if st.get("status") != "COMPLETED":
            last_err = f"runpod status {st.get('status')}: {str(st.get('error'))[:1000]}"
            spent = st.get("status") == "TIMED_OUT"
            continue
        out = st.get("output") or {}
        record = {k: out.get(k) for k in
                  ("status", "error", "comfy_version", "torch_version", "python_version",
                   "gpu_name", "vram_peak_mb", "rss_peak_mb", "validation", "timings",
                   "commit_checked_out", "env", "lane", "custom_nodes")}
        # The worker reports failure detail under `failure` (a top-level `error`
        # makes the RunPod SDK fail the job and drop a dict detail). Older images
        # still use `error`, which the SDK may lift to the job level: read all three.
        if record.get("error") is None:
            record["error"] = out.get("failure")
        if record.get("error") is None and st.get("error") is not None:
            record["error"] = _error_detail(st.get("error"))
        record["runpod_job_id"] = job_id
        record["attempt"] = attempt
        delay = st.get("delayTime")  # RunPod reports milliseconds
        record["delay_ms"] = delay if isinstance(delay, (int, float)) else None
        record["delay_s"] = round(delay / 1000, 3) if isinstance(delay, (int, float)) else None
        record["execution_ms"] = st.get("executionTime")
        outputs_dir = workdir / "outputs"
        record["outputs"] = _write_outputs(out.get("outputs") or [], outputs_dir)
        record["outputs_dir"] = str(outputs_dir)
        if record["status"] in RETRYABLE_STATUSES and attempt < attempts:
            last_err = f"worker infra status {record['status']}: {json.dumps(record.get('error'))[:500]}"
            continue
        return record
    failed = {"status": "infra_error", "error": {"message": last_err}, "attempt": attempts}
    if spent and attempts == 1:
        failed["reason"] = TIMED_OUT
    return failed
