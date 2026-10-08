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
