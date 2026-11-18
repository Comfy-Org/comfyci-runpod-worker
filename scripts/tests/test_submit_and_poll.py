"""submit_and_poll: run records keep worker error detail, convert RunPod's
millisecond delay, and never retry a deterministic bad_request."""
import json

import submit_and_poll


class FakeClient:
    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.runs = 0

    def run(self, job_input):
        self.runs += 1
        return f"job-{self.runs}"

    def status(self, job_id):
        return self.bodies.pop(0)

    def cancel(self, job_id):
        pass


def _job(client, tmp_path):
    return submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                            timeout_s=10, comfy_flags=[], workdir=tmp_path)


def test_job_level_error_is_kept_and_decoded(tmp_path):
    detail = {"message": "pip install failed", "traceback": "..."}
    client = FakeClient([
        {"status": "COMPLETED", "delayTime": 7325, "executionTime": 907,
         "output": {"status": "checkout_error", "error": None, "timings": {}},
         "error": json.dumps(detail)},
        {"status": "COMPLETED", "delayTime": 100, "executionTime": 900,
         "output": {"status": "checkout_error", "error": None, "timings": {}},
         "error": "plain text"},
    ])
    rec = _job(client, tmp_path)
    assert client.runs == 2  # checkout_error is retried once
    assert rec["status"] == "checkout_error"
    assert rec["error"] == {"message": "plain text"}
    assert rec["delay_s"] == 0.1


def test_first_attempt_record_decodes_dict_errors(tmp_path):
    client = FakeClient([
        {"status": "COMPLETED", "delayTime": 7325, "executionTime": 907,
         "output": {"status": "execution_error", "error": None, "timings": {}},
         "error": json.dumps({"node_id": "3", "message": "boom"})},
    ])
    rec = _job(client, tmp_path)
    assert client.runs == 1  # execution errors are never retried
    assert rec["error"] == {"node_id": "3", "message": "boom"}
    assert rec["delay_s"] == 7.325 and rec["delay_ms"] == 7325 and rec["execution_ms"] == 907


def test_failure_detail_is_read_from_the_workers_failure_key(tmp_path):
    # The worker reports under `failure`: a top-level `error` makes the RunPod SDK
    # fail the job-done call and the detail vanishes from the stored output.
    detail = {"node_id": "44:2", "node_type": "UNETLoader", "message": "boom", "log_tail": "..."}
    client = FakeClient([
        {"status": "COMPLETED", "delayTime": 10, "executionTime": 900,
         "output": {"status": "execution_error", "failure": detail, "timings": {},
                    "env": {"image": "cu130-t2141"}, "lane": "py313-torch2.14.1-cu130"}},
    ])
    rec = _job(client, tmp_path)
    assert client.runs == 1
    assert rec["status"] == "execution_error"
    assert rec["error"] == detail
    assert rec["env"] == {"image": "cu130-t2141"} and rec["lane"] == "py313-torch2.14.1-cu130"


def test_custom_nodes_ride_in_the_job_input_and_back_in_the_record(tmp_path):
    class Recording(FakeClient):
        def run(self, job_input):
            self.sent = job_input
            return super().run(job_input)

    packs = [{"name": "ComfyUI-KJNodes", "repo": "https://github.com/kijai/ComfyUI-KJNodes",
              "sha": "e" * 40, "kind": "git", "requirements": True}]
    report = {"ComfyUI-KJNodes": {"sha": "e" * 40, "source": "cache", "pip_s": 0.0,
                                  "import": "ok", "seconds": 1.1}}
    client = Recording([
        {"status": "COMPLETED", "delayTime": 10, "executionTime": 900,
         "output": {"status": "ok", "timings": {"custom_nodes_s": 0.8}, "custom_nodes": report,
                    "outputs": []}},
    ])
    rec = submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                           timeout_s=10, comfy_flags=["--x"], workdir=tmp_path,
                                           custom_nodes=packs)
    assert client.sent["custom_nodes"] == packs and client.sent["comfy_flags"] == ["--x"]
    assert rec["custom_nodes"] == report and rec["timings"] == {"custom_nodes_s": 0.8}
    # Without packs the job input still carries the key, empty, so the worker reconciles.
    rec = _job(FakeClient([{"status": "COMPLETED", "output": {"status": "ok", "outputs": []}}]), tmp_path)
    assert rec["custom_nodes"] is None


def test_pack_install_failures_are_infra_and_retried(tmp_path):
    client = FakeClient([
        {"status": "COMPLETED", "output": {"status": "custom_node_install_error",
                                           "failure": {"message": "pinned commit could not be fetched"}}},
        {"status": "COMPLETED", "output": {"status": "custom_node_error",
                                           "failure": {"message": "ComfyUI-KJNodes: IMPORT FAILED"}}},
    ])
    rec = _job(client, tmp_path)
    assert client.runs == 2
    assert rec["status"] == "custom_node_error"  # an import failure is never retried
    assert "custom_node_install_error" in submit_and_poll.INFRA_STATUSES
    assert "custom_node_error" not in submit_and_poll.INFRA_STATUSES


def test_bad_request_is_infra_but_not_retried(tmp_path):
    client = FakeClient([
        {"status": "COMPLETED", "output": {"status": "bad_request",
                                           "error": {"message": "workflow required"}}},
    ])
    rec = _job(client, tmp_path)
    assert client.runs == 1
    assert rec["status"] == "bad_request"
    assert rec["status"] in submit_and_poll.INFRA_STATUSES
    assert rec["error"] == {"message": "workflow required"}
    assert rec["delay_s"] is None


class Clock:
    """Stands in for the time module: only sleep() moves it."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class QueuedClient(FakeClient):
    """Every job waits `queue_s` in the queue, runs `run_s` and completes."""

    def __init__(self, clock, queue_s, run_s=0):
        super().__init__([])
        self.clock, self.queue_s, self.run_s = clock, queue_s, run_s
        self.submitted, self.cancelled = {}, []

    def run(self, job_input):
        job_id = super().run(job_input)
        self.submitted[job_id] = self.clock.now
        return job_id

    def status(self, job_id):
        waited = self.clock.now - self.submitted[job_id]
        if waited < self.queue_s:
            return {"status": "IN_QUEUE"}
        if waited < self.queue_s + self.run_s:
            return {"status": "IN_PROGRESS"}
        return {"status": "COMPLETED", "output": {"status": "ok", "outputs": []}}

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def test_a_job_waits_600_seconds_in_the_queue_unless_told_otherwise(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)
    # Today's rule: cancelled after 600 s in the queue, submitted once more, then given up.
    client = QueuedClient(clock, queue_s=650)
    rec = _job(client, tmp_path)
    assert (client.runs, client.cancelled) == (2, ["job-1", "job-2"])
    assert rec == {"status": "infra_error", "attempt": 2,
                   "error": {"message": "job job-2 stuck IN_QUEUE > 600s"}}
    # A job further back in its run's line is given longer, and is not cancelled.
    client = QueuedClient(clock, queue_s=650)
    rec = submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                           timeout_s=10, comfy_flags=[], workdir=tmp_path,
                                           queue_deadline_s=720)
    assert (client.runs, client.cancelled, rec["status"]) == (1, [], "ok")
    client = QueuedClient(clock, queue_s=800)
    rec = submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                           timeout_s=10, comfy_flags=[], workdir=tmp_path,
                                           queue_deadline_s=720)
    assert client.cancelled == ["job-1", "job-2"]
    assert rec["error"] == {"message": "job job-2 stuck IN_QUEUE > 720s"}


def test_a_longer_wait_in_the_queue_is_not_taken_from_the_jobs_own_time(tmp_path, monkeypatch):
    """The runner gives up on a job after its timeout plus 900 s, counted from
    the submission. A job that may queue longer gets that much more."""
    assert submit_and_poll.poll_deadline_s(10) == 910
    assert submit_and_poll.poll_deadline_s(10, 600) == 910
    assert submit_and_poll.poll_deadline_s(10, 720) == 1030
    assert submit_and_poll.poll_deadline_s(10, 300) == 910  # never less than today
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)
    # 700 s in the queue and 300 s of work: past 910 s, inside 1030 s.
    client = QueuedClient(clock, queue_s=700, run_s=300)
    rec = submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                           timeout_s=10, comfy_flags=[], workdir=tmp_path,
                                           queue_deadline_s=720)
    assert (client.runs, client.cancelled, rec["status"]) == (1, [], "ok")
    # With today's queue deadline the limit and its message are what they were.
    client = QueuedClient(clock, queue_s=500, run_s=500)
    rec = _job(client, tmp_path)
    assert client.cancelled == ["job-1", "job-2"]
    assert rec["error"] == {"message": "job job-2 exceeded deadline (910s)"}


def test_an_empty_text_is_written_and_only_an_unsent_output_is_marked(tmp_path):
    """The worker sends an empty file as "" and leaves the data out of one past
    its size limit. An empty generated text is a result, not a lost file. Any
    other empty file is listed as not sent, as before text workflows: an empty
    PNG on disk would turn an image run's `fail` into an error of the harness."""
    sent = [{"filename": "wf_00001.txt", "bytes": 0, "sha256": "e" * 64, "data_b64": ""},
            {"filename": "wf_00002.TXT", "bytes": 2, "sha256": "f" * 64, "data_b64": "aGk="},
            {"filename": "sub/wf_00001_.png", "bytes": 9, "sha256": "a" * 64, "truncated": True},
            {"filename": "wf_00003.txt", "bytes": 5, "sha256": "b" * 64, "data_b64": None},
            {"filename": "wf_00002_.png", "bytes": 0, "sha256": "e" * 64, "data_b64": ""},
            {"filename": "notes.log", "bytes": 0, "sha256": "e" * 64, "data_b64": ""}]
    inventory = submit_and_poll._write_outputs(sent, tmp_path / "outputs")
    assert inventory == [
        {"filename": "wf_00001.txt", "bytes": 0, "sha256": "e" * 64},
        {"filename": "wf_00002.TXT", "bytes": 2, "sha256": "f" * 64},
        {"filename": "sub/wf_00001_.png", "bytes": 9, "sha256": "a" * 64, "truncated": True},
        {"filename": "wf_00003.txt", "bytes": 5, "sha256": "b" * 64, "truncated": True},
        {"filename": "wf_00002_.png", "bytes": 0, "sha256": "e" * 64, "truncated": True},
        {"filename": "notes.log", "bytes": 0, "sha256": "e" * 64, "truncated": True}]
    assert (tmp_path / "outputs" / "wf_00001.txt").read_bytes() == b""
    assert (tmp_path / "outputs" / "wf_00002.TXT").read_bytes() == b"hi"
    assert sorted(p.name for p in (tmp_path / "outputs").rglob("*") if p.is_file()) == [
        "wf_00001.txt", "wf_00002.TXT"]
