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


def test_the_client_counts_the_submissions_runpod_has_not_answered(monkeypatch):
    """RunPod can have created a job before the runner knows its id. Such a
    job is in no list: a run that is stopped has to know that an answer is
    still on its way, and wait for it before its last cancel pass."""
    api = FakeRunPod()
    monkeypatch.setattr(submit_and_poll, "requests", api)
    client = submit_and_poll.RunPodClient("ep", api_key="k")
    assert client.submitting == 0
    seen, post = [], api.post

    def watched(url, **kw):
        seen.append((client.submitting, sorted(client.live)))
        return post(url, **kw)

    monkeypatch.setattr(api, "post", watched)
    first = client.run({})
    # Counted while the call is out, and not after the id is in the list.
    assert seen == [(1, [])] and (client.submitting, client.live) == (0, {first})
    second = client.run({})
    assert seen[1] == (1, [first]) and (client.submitting, client.live) == (0, {first, second})
    # A submission that failed is not on its way any more, and left no job.
    api.down = True
    with pytest.raises(api.RequestException):
        client.run({})
    assert (client.submitting, client.live) == (0, {first, second})
    # A cancel call is no submission.
    api.down = False
    client.cancel(first)
    assert seen[-1] == (0, [first, second]) and client.submitting == 0


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


# -- a job that fetches its models ----------------------------------------------------------
class Sent(FakeClient):
    """Keeps the input of each job it is given."""

    def __init__(self, bodies):
        super().__init__(bodies)
        self.inputs = []

    def run(self, job_input):
        self.inputs.append(job_input)
        return super().run(job_input)


DONE = {"status": "COMPLETED", "output": {"status": "ok", "outputs": []}}
# What a job of a lane with its models on the volume is sent, and what its
# record holds: every key, in this order. Neither may gain one.
JOB_INPUT = ('{"mode": "run", "comfy_commit": "cccccccccccccccccccccccccccccccccccccccc", '
             '"comfy_repo": null, "workflow_id": "wf", "workflow": {}, "models": [], '
             '"timeout_s": 10, "comfy_flags": [], "lane": null, "custom_nodes": []}')
RECORD_KEYS = ["status", "error", "comfy_version", "torch_version", "python_version", "gpu_name",
               "vram_peak_mb", "rss_peak_mb", "validation", "timings", "commit_checked_out",
               "env", "lane", "custom_nodes", "runpod_job_id", "attempt", "delay_ms", "delay_s",
               "execution_ms", "outputs", "outputs_dir"]


def test_the_input_and_the_record_of_a_job_that_fetches_nothing_are_unchanged(tmp_path):
    """This test passes on the code from before jobs could fetch models too:
    it pins what that code sent and recorded."""
    client = Sent([DONE])
    rec = _job(client, tmp_path)
    assert json.dumps(client.inputs[0]) == JOB_INPUT
    assert list(rec) == RECORD_KEYS


def test_only_fetch_missing_asks_for_a_fetch(tmp_path):
    client = Sent([DONE, DONE])
    for kw in ({"fetch_allowance_s": 400}, {"fetch_streams": 1}):
        submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                         timeout_s=10, comfy_flags=[], workdir=tmp_path, **kw)
    assert [json.dumps(sent) for sent in client.inputs] == [JOB_INPUT] * 2


def test_a_job_of_a_lane_that_fetches_is_told_to(tmp_path):
    def sent(**kw):
        client = Sent([DONE])
        submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                         timeout_s=10, comfy_flags=[], workdir=tmp_path, **kw)
        return json.dumps(client.inputs[0])

    asked = JOB_INPUT[:-1] + ', "fetch_missing": true}'
    assert sent(fetch_missing=True) == asked
    # How long the runner waits is its own business. What the worker hears of
    # it is how long the download may take: here 400 s and the workflow's 10,
    # less the 60 s the worker's answer is given.
    limited = asked[:-1] + ', "fetch_timeout_s": 350}'
    assert sent(fetch_missing=True, fetch_allowance_s=400) == limited
    assert sent(fetch_missing=True, fetch_streams=1) == asked[:-1] + ', "fetch_streams": 1}'
    assert sent(fetch_missing=True, fetch_allowance_s=400, fetch_streams=1) == (
        limited[:-1] + ', "fetch_streams": 1}')
    assert sent(fetch_missing=False, fetch_streams=1) == JOB_INPUT
    # The limit is for a job that fetches, like the allowance it comes from.
    assert sent(fetch_allowance_s=400) == JOB_INPUT


def test_the_record_carries_the_fetch_report_only_when_the_worker_sent_one(tmp_path):
    report = {"files": 2, "bytes": 800, "seconds": 4.0, "mb_per_s": 0.0, "streams": 4,
              "token": True, "reused": 0, "reused_bytes": 0, "evicted": 0}
    client = Sent([{"status": "COMPLETED", "output": {
        "status": "ok", "timings": {"model_fetch_s": 4.2, "prompt_exec_s": 3.0},
        "model_fetch": report, "outputs": []}}])
    rec = submit_and_poll.run_workflow_job(client, "wf", {}, comfy_commit="c" * 40, models=[],
                                           timeout_s=10, comfy_flags=[], workdir=tmp_path,
                                           fetch_missing=True)
    assert rec["model_fetch"] == report and rec["timings"]["model_fetch_s"] == 4.2
    assert list(rec) == RECORD_KEYS[:14] + ["model_fetch"] + RECORD_KEYS[14:]
    # A worker image from before the fetch sends none, also when it is asked.
    rec = submit_and_poll.run_workflow_job(Sent([DONE]), "wf", {}, comfy_commit="c" * 40,
                                           models=[], timeout_s=10, comfy_flags=[],
                                           workdir=tmp_path, fetch_missing=True)
    assert list(rec) == RECORD_KEYS


def test_the_runner_waits_longer_for_a_job_that_downloads_its_models(tmp_path, monkeypatch):
    """The runner gives up on a job after its timeout plus 900 s. A download
    of 60 GB at the planning speed takes 2,400 s more."""
    deadline = submit_and_poll.poll_deadline_s
    assert deadline(10, 600, 2400) == 3310 and deadline(10, 1800, 2400) == 4510
    # Without a fetch the limit is the whole number it was, as its message prints it.
    assert deadline(10) == 910 and isinstance(deadline(10), int)
    clock = Clock()
    monkeypatch.setattr(submit_and_poll, "time", clock)

    def job(run_s, **kw):
        client = QueuedClient(clock, queue_s=0, run_s=run_s)
        rec = submit_and_poll.run_workflow_job(
            client, "wf", {}, comfy_commit="c" * 40, models=[], timeout_s=10, comfy_flags=[],
            workdir=tmp_path, attempts=1, **kw)
        return client, rec

    # 3,000 s of work: past 910 s, inside 910 + 2,400.
    client, rec = job(3000, fetch_missing=True, fetch_allowance_s=2400)
    assert (client.cancelled, rec["status"]) == ([], "ok")
    client, rec = job(3400, fetch_missing=True, fetch_allowance_s=2400)
    assert client.cancelled == ["job-1"]
    assert rec == {"status": "infra_error", "attempt": 1, "reason": "timed_out",
                   "error": {"message": "job job-1 exceeded deadline (3310s)"}}
    # The time is for a fetch: a job that is not told to fetch does not get it.
    client, rec = job(3000, fetch_allowance_s=2400)
    assert rec["error"] == {"message": "job job-1 exceeded deadline (910s)"}


def test_a_failed_download_is_tried_again_and_a_refused_file_is_not(tmp_path):
    assert {"model_fetch_error", "model_access_error"} <= submit_and_poll.INFRA_STATUSES
    assert "model_fetch_error" in submit_and_poll.RETRYABLE_STATUSES
    assert submit_and_poll.INFRA_STATUSES - submit_and_poll.RETRYABLE_STATUSES == {
        "bad_request", "model_access_error"}
    failed = {"message": "model download failed: vae/x: HTTP 503 Service Unavailable"}
    client = FakeClient([
        {"status": "COMPLETED", "output": {"status": "model_fetch_error", "failure": failed}},
        DONE])
    rec = _job(client, tmp_path)
    assert (client.runs, rec["status"], rec["attempt"]) == (2, "ok", 2)
    # The host said no: a person has to act, and a second job would hear the same.
    refused = {"message": "the model host answered 403 for Lightricks/LTX-2.5",
               "host": "huggingface.co", "repo": "Lightricks/LTX-2.5", "http_status": 403}
    client = FakeClient([
        {"status": "COMPLETED", "output": {"status": "model_access_error", "failure": refused,
                                           "model_fetch": {"files": 0, "token": False}}},
        DONE])
    rec = _job(client, tmp_path)
    assert (client.runs, rec["status"], rec["error"]) == (1, "model_access_error", refused)
    assert rec["model_fetch"] == {"files": 0, "token": False}


# -- a time limit for the download, on the worker -----------------------------------------
def test_how_long_the_worker_gives_a_download():
    """The time the runner waits for the download and the workflow's own
    timeout, less the 60 s for the worker's answer. With the allowance alone
    as the limit, the planning speed of 25 MB a second was the slowest
    download that passed. Before there was a limit, 15.8 GB passed at 6 MB a
    second in a job that started and ran as jobs mostly do."""
    limit = submit_and_poll.fetch_timeout_s
    assert submit_and_poll.FETCH_TIMEOUT_MARGIN_S == 60
    # 15.8 GB at 25 MB a second are 632 s. With a workflow of 600 s the
    # download may take 1,172 s, which is 13.5 MB a second.
    assert limit(600, 632) == 1172 and 15.8e9 / 1e6 / limit(600, 632) < 13.5
    # 56.6 GB, the largest family: 2,264 s and 900 s, so 18.3 MB a second.
    assert limit(900, 2264) == 3104 and 56.6e9 / 1e6 / limit(900, 2264) < 18.3
    # A small download is connections and a rate limit's wait of up to 300 s
    # more than it is bytes. It has the workflow's time for them.
    assert limit(600, 12) == 552 and limit(900, 40) == 880 and limit(900, 0) == 840
    assert limit(10, 400) == 350
    # A whole number of seconds, as the job input carries it.
    assert limit(600.0, 2400.7) == 2940 and isinstance(limit(600.0, 2400.7), int)
    # Nothing is left of such a job's time: no limit, as before the key.
    assert limit(10, 0) is None and limit(60, 0) is None and limit(30, 30.5) is None


def test_the_worker_stops_a_download_before_the_runner_stops_waiting():
    """The runner gives a job 900 s to start, the allowance for its download
    and the workflow's timeout, and then cancels it. The worker's limit for
    the download ends 60 s before that, when the start took no more than its
    900 s: the time the worker waits for its downloads to end, its answer's
    way, and the pause until the runner looks again."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import model_sync
    import regression_skip
    # One word, in the worker's answer, in the run record and in the cell.
    assert model_sync.TIMED_OUT == submit_and_poll.TIMED_OUT == regression_skip.TIMED_OUT
    start, margin = submit_and_poll.COLD_START_ALLOWANCE_S, submit_and_poll.FETCH_TIMEOUT_MARGIN_S
    assert margin >= model_sync.FETCH_STOP_WAIT_S + submit_and_poll.POLL_INTERVAL_S + 30
    # The time in which the worker shuts connections is inside its wait.
    assert 0 < model_sync.FETCH_CUT_AFTER_S < model_sync.FETCH_STOP_WAIT_S
    for timeout_s in (60, 61, 300, 600, 900, 1200, 2400, 3300):
        for allowance in (0, 1, 12, 59, 60, 400, 600, 632, 2264, 2401):
            limit = submit_and_poll.fetch_timeout_s(timeout_s, allowance)
            if limit is None:
                assert allowance + timeout_s - margin < 1
                continue
            gives_up = submit_and_poll.poll_deadline_s(timeout_s, 600, allowance)
            # The answer has its 60 s, and the download every second but those.
            assert gives_up - (start + limit) == margin, (timeout_s, allowance)
            # RunPod ends a job 3,600 s after a worker took it. The lint asks
            # of a job that fetches that its timeout, its allowance and 300 s
            # fit in there (test_manifest_lint.py). The worker's answer to a
            # download it stopped then comes before RunPod's end as well, and
            # the checkout before it has those 300 s.
            if timeout_s + allowance + 300 <= 3600:
                assert 300 + limit + model_sync.FETCH_STOP_WAIT_S <= 3600


SLOW = {"message": "the download was too slow: 12.4 of 56.6 GB in 600 s (20.7 MB/s), "
                   "2 of 5 files complete",
        "reason": "timed_out", "timeout_s": 600, "received_bytes": 12_400_000_000,
        "wanted_bytes": 56_600_000_000, "seconds": 600.2, "mb_per_s": 20.7, "files_done": 2,
        "files_wanted": 5}
FETCHED = {"files": 2, "bytes": 3_200_000_000, "seconds": 600.2, "mb_per_s": 20.7, "streams": 4,
           "token": True, "reused": 0, "reused_bytes": 0, "evicted": 0}


def test_a_download_the_worker_stopped_is_not_submitted_again(tmp_path):
    """A `model_fetch_error` is tried again at once. Not this one: the worker
    stopped the download because its time was up, and a second job can take
    as long and cost as much. Its record says so where the record of a job
    that the runner gave up on says it."""
    stopped = {"status": "COMPLETED", "delayTime": 10, "executionTime": 601000, "output": {
        "status": "model_fetch_error", "failure": SLOW, "model_fetch": FETCHED,
        "timings": {"checkout_s": 1.9, "model_fetch_s": 600.3}}}
    for attempts in (2, 1):
        client = Sent([stopped, DONE])
        rec = submit_and_poll.run_workflow_job(
            client, "wf", {}, comfy_commit="c" * 40, models=[], timeout_s=600, comfy_flags=[],
            workdir=tmp_path, attempts=attempts, fetch_missing=True, fetch_allowance_s=600)
        assert client.runs == 1 and client.inputs[0]["fetch_timeout_s"] == 1140, attempts
        assert (rec["status"], rec["reason"], rec["attempt"]) == (
            "model_fetch_error", "timed_out", 1)
        assert rec["error"] == SLOW and rec["model_fetch"] == FETCHED
        assert list(rec) == (RECORD_KEYS[:14] + ["model_fetch"] + RECORD_KEYS[14:16] + ["reason"]
                             + RECORD_KEYS[16:])
    # What a worker image from before the limit answers to the same job, which
    # the key in its input changes nothing about: a download that failed is
    # tried again as it was, and a job that passed is recorded as it was.
    plain = {"message": "model download failed: vae/x: HTTP 503 Service Unavailable"}
    client = Sent([{"status": "COMPLETED", "output": {"status": "model_fetch_error",
                                                      "failure": plain}}, DONE])
    rec = submit_and_poll.run_workflow_job(
        client, "wf", {}, comfy_commit="c" * 40, models=[], timeout_s=600, comfy_flags=[],
        workdir=tmp_path, fetch_missing=True, fetch_allowance_s=600)
    assert (client.runs, rec["status"], rec["attempt"]) == (2, "ok", 2)
    assert list(rec) == RECORD_KEYS and [sent["fetch_timeout_s"] for sent in client.inputs] == [
        1140, 1140]
    # The mark is the worker's failure saying `reason`, and nothing else.
    marked = submit_and_poll.stopped_at_its_limit
    assert marked({"status": "model_fetch_error", "error": SLOW})
    for other in ({}, {"error": None}, {"error": "timed_out"}, {"error": {"reason": "slow"}},
                  {"error": {"message": "timed_out"}}, {"reason": "timed_out"},
                  {"status": "infra_error", "error": {"message": "x"}, "reason": "timed_out"}):
        assert not marked(other), other
