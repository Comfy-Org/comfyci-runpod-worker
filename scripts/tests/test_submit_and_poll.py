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
