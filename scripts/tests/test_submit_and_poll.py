"""submit_and_poll: run records keep worker error detail, convert RunPod's
millisecond delay, and never retry a deterministic bad_request."""
import json

import pytest

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


# -- one attempt, for a run that decides itself what to try again ---------------------------
def _once(client, tmp_path, **kw):
    return submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                            timeout_s=10, comfy_flags=[], workdir=tmp_path,
                                            attempts=1, **kw)


def test_with_one_attempt_a_job_is_submitted_once(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)
    # A worker status that the two-attempt rule would try again comes back as it is.
    client = FakeClient([{"status": "COMPLETED", "output": {"status": "checkout_error"}}])
    rec = _once(client, tmp_path)
    assert (client.runs, rec["status"], rec["attempt"]) == (1, "checkout_error", 1)
    assert "reason" not in rec
    # A job that waited too long in the queue: not tested, and worth another try.
    client = QueuedClient(clock, queue_s=2000)
    rec = _once(client, tmp_path, queue_deadline_s=1800)
    assert (client.runs, client.cancelled) == (1, ["job-1"])
    assert rec == {"status": "infra_error", "attempt": 1,
                   "error": {"message": "job job-1 stuck IN_QUEUE > 1800s"}}
    # A result is a result, with one attempt as with two.
    rec = _once(FakeClient([{"status": "COMPLETED", "output": {"status": "ok"}}]), tmp_path)
    assert (rec["status"], rec["attempt"]) == ("ok", 1) and "reason" not in rec


def test_a_job_that_used_up_its_time_says_so_when_the_caller_decides_about_a_retry(
        tmp_path, monkeypatch):
    """A second attempt of such a job can cost another hour. The run that
    asked for one attempt reads `reason` and does not try it again."""
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)
    # The runner gave up at the poll deadline: 10 s of timeout plus 900 s.
    client = QueuedClient(clock, queue_s=100, run_s=5000)
    rec = _once(client, tmp_path)
    assert rec == {"status": "infra_error", "attempt": 1, "reason": "timed_out",
                   "error": {"message": "job job-1 exceeded deadline (910s)"}}
    assert client.cancelled == ["job-1"] and submit_and_poll.TIMED_OUT == "timed_out"
    # RunPod ended the job at the endpoint's own limit.
    rec = _once(FakeClient([{"status": "TIMED_OUT", "error": "job timed out after 1 retries"}]),
                tmp_path)
    assert rec["reason"] == "timed_out" and rec["error"]["message"].startswith(
        "runpod status TIMED_OUT: ")
    # Any other end without a result is not that.
    for body in ({"status": "FAILED", "error": "x"}, {"status": "CANCELLED"}):
        assert "reason" not in _once(FakeClient([body]), tmp_path)
    # With two attempts the function has already tried again, and its record is
    # the one it always was: three keys.
    client = QueuedClient(clock, queue_s=100, run_s=5000)
    rec = _job(client, tmp_path)
    assert rec == {"status": "infra_error", "attempt": 2,
                   "error": {"message": "job job-2 exceeded deadline (910s)"}}
    client = FakeClient([{"status": "TIMED_OUT"}, {"status": "TIMED_OUT"}])
    assert list(_job(client, tmp_path)) == ["status", "error", "attempt"] and client.runs == 2


def test_the_pause_between_two_looks_at_a_job_is_10_seconds_unless_told_otherwise(
        tmp_path, monkeypatch):
    class Counting(Clock):
        def __init__(self):
            super().__init__()
            self.pauses = []

        def sleep(self, seconds):
            self.pauses.append(seconds)
            super().sleep(seconds)

    clock = Counting()
    monkeypatch.setattr(submit_and_poll, "time", clock)
    assert _job(QueuedClient(clock, queue_s=30, run_s=30), tmp_path)["status"] == "ok"
    assert clock.pauses == [10] * 6
    clock.pauses.clear()
    rec = _once(QueuedClient(clock, queue_s=30, run_s=30), tmp_path, poll_interval_s=3)
    assert rec["status"] == "ok" and clock.pauses == [3] * 20

    # A look that fails on the network is followed by the same pause.
    class Flaky(FakeClient):
        def status(self, job_id):
            body = self.bodies.pop(0)
            if body is None:
                raise submit_and_poll.requests.RequestException("no network")
            return body

    done = {"status": "COMPLETED", "output": {"status": "ok"}}
    for interval, kw in ((10, {}), (3, {"poll_interval_s": 3})):
        clock.pauses.clear()
        rec = _once(Flaky([None, None, {"status": "IN_PROGRESS"}, done]), tmp_path, **kw)
        assert rec["status"] == "ok" and clock.pauses == [interval] * 3


class FakeRunPod:
    """Stands in for `requests`: an endpoint that numbers its jobs and
    answers a status call with what the test put in `states`."""
    RequestException = submit_and_poll.requests.RequestException

    def __init__(self):
        self.states, self.calls, self.jobs, self.down = {}, [], 0, False
        self.cancel_answer = 200  # the status code of an answer to a cancel call

    def post(self, url, headers=None, json=None, timeout=None):
        path = url.rsplit("/v2/", 1)[1]
        self.calls.append(("POST", path, timeout))
        if self.down:
            raise self.RequestException("no network")
        if "/cancel/" in path:
            return self._answer({"status": "CANCELLED"}, self.cancel_answer)
        self.jobs += 1
        return self._answer({"id": f"job-{self.jobs}"})

    def get(self, url, headers=None, timeout=None):
        return self._answer({"status": self.states[url.rsplit("/", 1)[1]]})

    @staticmethod
    def _answer(body, status_code=200):
        import types
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: body,
                                     status_code=status_code)


def test_the_client_knows_which_of_its_jobs_runpod_has_not_ended(monkeypatch):
    """What a run that is stopped has to cancel: a job that nobody reads any
    more runs on, billed, to its end."""
    api = FakeRunPod()
    monkeypatch.setattr(submit_and_poll, "requests", api)
    client = submit_and_poll.RunPodClient("ep", api_key="k")
    assert client.live == set()
    first, second, third = (client.run({}) for _ in range(3))
    assert client.live == {"job-1", "job-2", "job-3"}
    # A look at a job that is waiting or running changes nothing.
    for state in ("IN_QUEUE", "IN_PROGRESS"):
        api.states[first] = state
        client.status(first)
        assert first in client.live
    # RunPod ended it, whichever way: nothing is left to cancel.
    for job_id, state in ((first, "COMPLETED"), (second, "FAILED")):
        api.states[job_id] = state
        client.status(job_id)
    assert client.live == {third}
    for state in ("CANCELLED", "TIMED_OUT"):
        api.states[client.run({})] = state
    for job_id in sorted(client.live - {third}):
        client.status(job_id)
    assert client.live == {third}
    # A job the runner cancels itself (the queue, its deadline) is not live either.
    fourth = client.run({})
    assert client.cancel(fourth) is True
    assert client.live == {third} and api.calls[-1] == ("POST", "ep/cancel/job-6", 60)

    # The stop: one cancel call for each live job, with a few seconds each.
    fifth = client.run({})
    api.calls.clear()
    assert client.cancel_live() == (sorted([third, fifth]), [])
    assert api.calls == [("POST", f"ep/cancel/{third}", submit_and_poll.STOP_CANCEL_TIMEOUT_S),
                         ("POST", f"ep/cancel/{fifth}", submit_and_poll.STOP_CANCEL_TIMEOUT_S)]
    assert client.live == set() and client.cancel_live() == ([], [])
    # A cancel that cannot reach RunPod does not raise: the run is going down
    # anyway. But the job is not cancelled, and nothing may say that it is:
    # it stays live, so the next stop asks again.
    sixth = client.run({})
    api.down = True
    assert client.cancel(sixth) is False and client.live == {sixth}
    assert client.cancel_live() == ([], [sixth]) and client.live == {sixth}
    # Nor is a job cancelled for which RunPod answers an error.
    api.down = False
    for code in (500, 401, 404):
        api.cancel_answer = code
        assert client.cancel_live() == ([], [sixth]) and client.live == {sixth}, code
    api.cancel_answer = 200
    assert client.cancel_live() == ([sixth], []) and client.live == set()


def test_a_job_whose_status_stays_out_of_reach_is_given_up_after_five_minutes_and_cancelled(
        tmp_path, monkeypatch):
    """The runner counted 30 failed looks: 5 minutes at the usual 10 seconds,
    a minute and a half at the 3 seconds of a nightly or weekly run. And it
    left the job running, unread and billed, while such a run submits the
    workflow again at its end."""
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)

    class Unreachable(QueuedClient):
        """No look at a job is answered in its first `down_s` seconds."""

        def __init__(self, down_s, run_s):
            super().__init__(clock, queue_s=0, run_s=run_s)
            self.down_s, self.looks = down_s, 0

        def status(self, job_id):
            self.looks += 1
            if clock.now - self.submitted[job_id] < self.down_s:
                raise submit_and_poll.requests.RequestException("no network")
            return super().status(job_id)

    assert submit_and_poll.STATUS_PATIENCE_S == 300
    # 200 seconds without an answer, at either pause: the job is read when the
    # network is back. One job, and nothing is cancelled.
    for kw in ({}, {"poll_interval_s": 3}):
        client = Unreachable(down_s=200, run_s=250)
        rec = _once(client, tmp_path, **kw)
        assert (rec["status"], client.runs, client.cancelled) == ("ok", 1, []), kw
    # No answer at all: given up after 5 minutes, and the job is cancelled.
    for pause, looks in ((3, 101), (10, 31)):
        client, start = Unreachable(down_s=10 ** 6, run_s=10), clock.now
        rec = _once(client, tmp_path, poll_interval_s=pause)
        assert rec == {"status": "infra_error", "attempt": 1, "error": {"message": "no network"}}
        assert (client.runs, client.looks, client.cancelled) == (1, looks, ["job-1"]), pause
        assert clock.now - start == pause * (looks - 1) == 300
    # With two attempts each job has the same patience, and neither is left running.
    client = Unreachable(down_s=10 ** 6, run_s=10)
    rec = _job(client, tmp_path)
    assert rec == {"status": "infra_error", "attempt": 2, "error": {"message": "no network"}}
    assert (client.runs, client.looks, client.cancelled) == (2, 62, ["job-1", "job-2"])

    # A submission that fails has no job to cancel.
    class NoSubmit(QueuedClient):
        def run(self, job_input):
            self.runs += 1
            raise submit_and_poll.requests.RequestException("no network")

    client = NoSubmit(clock, queue_s=0)
    assert _job(client, tmp_path)["status"] == "infra_error"
    assert (client.runs, client.cancelled) == (2, [])
    # The wait itself cancels nothing: the model sync waits with it too, and
    # its download goes on while the runner cannot see it.
    client = Unreachable(down_s=10 ** 6, run_s=10)
    job_id = client.run({})
    with pytest.raises(submit_and_poll.requests.RequestException):
        submit_and_poll._poll(client, job_id, 7200)
    assert client.cancelled == [] and client.looks == 31


def test_an_output_whose_name_leaves_the_outputs_folder_is_written_nowhere(tmp_path):
    """The worker names its files. The folder above the outputs is published
    as it is, and the runner holds the token that pushes the results branch."""
    dest = tmp_path / "run" / "wf" / "outputs"
    names = ["../escape.png", "../../climbed.txt", "sub/../../side.png", "..", ".", "",
             str(tmp_path / "abs_outside.png")]
    kept = ["wf_00001_.png", "sub/wf_00002_.png", "sub/../wf_00003_.png",
            str(dest / "abs_inside.png")]
    sent = [{"filename": name, "bytes": 2, "sha256": "a" * 64, "data_b64": "aGk="}
            for name in [*names, *kept]]
    inventory = submit_and_poll._write_outputs(sent, dest)
    assert [rec["filename"] for rec in inventory] == [*names, *kept]
    # Each is listed as a file that was not sent, with what the worker said of it.
    assert [rec for rec in inventory if rec.get("truncated")] == [
        {"filename": name, "bytes": 2, "sha256": "a" * 64, "truncated": True} for name in names]
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
                     if p.is_file())
    assert written == ["run/wf/outputs/abs_inside.png", "run/wf/outputs/sub/wf_00002_.png",
                       "run/wf/outputs/wf_00001_.png", "run/wf/outputs/wf_00003_.png"]
    # 10 s after its first signal GitHub kills the step: connect and answer fit in that.
    assert 2 * submit_and_poll.STOP_CANCEL_TIMEOUT_S < 7.5
