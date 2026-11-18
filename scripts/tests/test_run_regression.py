"""run_regression: pure decision helpers and the summary handed to later steps."""
import pytest

import index_builder
import run_regression

# A registry of its own, so these tests do not follow the committed lane state:
# "main" owns the default `regression` prefix, "other" is a layout-2 lane.
LANES = {"schema_version": 2, "primary": "main", "lanes": {
    "main": {"cadence": "per-commit", "role": "primary", "layout": 1, "prefix": "regression",
             "endpoint_id": "ep-main"},
    "other": {"cadence": "per-commit", "role": "lane", "layout": 2,
              "prefix": "regression/lanes/other", "endpoint_id": "ep-other"}}}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: LANES)

IDENT = {"identical": True, "mean_mse": 0.0}
DIFF = {"identical": False, "mean_mse": 3.4}


def test_classify_detail():
    c = run_regression.classify_detail
    assert c("pass", DIFF, "fail") is None
    assert c("fail", None, None) is None                      # nothing to compare against
    assert c("fail", {"error": "shape mismatch"}, "fail") is None
    assert c("fail", DIFF, "pass") == "new_drift"             # outputs changed here
    assert c("fail", DIFF, "fail") == "new_drift"             # changed again on top of a failure
    assert c("fail", IDENT, "fail") == "inherited"            # same bits as an earlier failure
    assert c("fail", IDENT, "pass") is None                   # golden changed, not the code
    assert c("fail", IDENT, "no_baseline") is None
    # The previous failure must have been judged against the same golden.
    assert c("fail", IDENT, "fail", "v0.38.0", "v0.38.0") == "inherited"
    assert c("fail", IDENT, "fail", "v0.38.0", "v0.33.2") is None


def test_metrics_pass_thresholds():
    th = {"max_mean_mse": 2.0, "min_mean_psnr_db": 45.0, "max_pct_pixels_changed": 1.0}
    ok = {"identical": False, "frame_count_ref": 1, "frame_count_cand": 1, "mean_mse": 1.0,
          "mean_psnr_db": 48.0, "mean_pct_pixels_changed": 0.5}
    assert run_regression.metrics_pass(ok, th)
    assert run_regression.metrics_pass({"identical": True}, th)
    assert not run_regression.metrics_pass({**ok, "mean_mse": 3.44}, th)
    assert not run_regression.metrics_pass({**ok, "mean_psnr_db": 42.76}, th)
    assert not run_regression.metrics_pass({**ok, "mean_pct_pixels_changed": 70.0}, th)
    assert not run_regression.metrics_pass({**ok, "frame_count_cand": 2}, th)
    assert not run_regression.metrics_pass({"error": "no PNGs"}, th)


def test_write_thumbnails(tmp_path):
    import numpy as np
    from PIL import Image
    out = tmp_path / "outputs"
    out.mkdir()
    for name in ("b_00002_.png", "a_00001_.png"):
        Image.fromarray(np.zeros((640, 480, 3), dtype=np.uint8)).save(out / name)
    assert run_regression.write_thumbnails(out) == "a_00001_.webp"
    thumbs = sorted(p.name for p in (out / "thumbs").iterdir())
    assert thumbs == ["a_00001_.webp", "b_00002_.webp"]
    with Image.open(out / "thumbs" / "a_00001_.webp") as im:
        assert max(im.size) == 256
    assert run_regression.write_thumbnails(tmp_path / "empty") is None


def test_first_output_sha_uses_first_png_by_name():
    rec = {"outputs": [{"filename": "b_00002_.png", "sha256": "two"},
                       {"filename": "a_00001_.png", "sha256": "one"},
                       {"filename": "video.mp4", "sha256": "mp4"}]}
    assert run_regression.first_output_sha(rec) == "one"
    assert run_regression.first_output_sha({"outputs": []}) is None
    assert run_regression.first_output_sha({}) is None


class FakeStore:
    """Read side of a results store, enough for a --skip-publish run."""
    root = "regression"

    def __init__(self, files):
        self.files = files

    def latest_pointer(self, branch):
        return self.files.get(f"regression/latest/{branch}.json")

    def download_json(self, path):
        return self.files.get(path)

    def published_summary(self, branch, commit):
        return self.files.get(f"regression/runs/{branch}/{commit}/summary.json")

    def attempt_marker(self, branch, commit):
        return self.files.get(f"regression/runs/{branch}/{commit}/attempt.json")


def _manifest(tmp_path, *wf_ids) -> str:
    """A manifest of its own with these workflows, all enabled, in this order:
    a run of `--workflows all` then tests exactly them."""
    import json
    path = tmp_path / "manifest" / "workflows.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [], "thresholds": {}},
        "workflows": {wf_id: {"enabled": True, "workflow_path": "x", "models": []}
                      for wf_id in wf_ids}}), encoding="utf-8")
    return str(path)


def test_summary_out_is_written_on_failure_with_prior_verdicts(tmp_path, monkeypatch):
    import json
    import sys
    import commit_meta
    import storage
    lane = "main"
    head = {"entries": [{"c": "prev", "t": 2, "m": {"ct": 2},
                         "w": {"flux_dev_t2i": {"v": "execution_error"}}},
                        {"c": "old", "t": 1, "m": {"ct": 1},
                         "w": {"flux_dev_t2i": {"v": "pass"}, "sdxl_t2i": {"v": "pass"}}}]}
    store = FakeStore({"regression/latest/master.json": {"commit": "old"},
                       f"regression/index/master/{lane}.json": head})
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta",
                        lambda sha: {"subject": "x (#9)", "pr": 9, "committed_ts": 3})
    monkeypatch.setattr(commit_meta, "fetch_range",
                        lambda prev, sha: {"prev": prev, "commits_between": 2, "compare_url": "u"})
    verdicts = {"flux_dev_t2i": "execution_error", "sdxl_t2i": "fail"}
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": verdicts[wf_id], "detail": None})
    out = tmp_path / "out" / "summary.json"
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--skip-publish", "--summary-out", str(out)])
    assert run_regression.main() == 1
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["overall"] == "fail" and summary["commit_meta"]["pr"] == 9
    assert summary["tested_range"]["commits_between"] == 2
    flux, sdxl = summary["workflows"]["flux_dev_t2i"], summary["workflows"]["sdxl_t2i"]
    assert (flux["prior_verdict"], flux["prior_commit"]) == ("execution_error", "prev")
    assert (sdxl["prior_verdict"], sdxl["prior_commit"]) == ("pass", "old")
    assert summary["attempt"] == 1


def test_summary_names_the_actions_run_that_tested_the_commit(tmp_path, monkeypatch):
    """The run report and the PR comment link to the run; the index keeps its
    id. Outside Actions both keys are null, never missing."""
    import json
    import sys
    import commit_meta
    import storage
    monkeypatch.setattr(storage, "from_args", lambda args: FakeStore({}))
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": "pass", "detail": None})
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--workflows", "sdxl_t2i", "--skip-publish",
                                      "--summary-out", str(out)])
    for key in ("GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_SERVER_URL"):
        monkeypatch.delenv(key, raising=False)
    assert run_regression.main() == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert "run_url" in summary and summary["run_url"] is None and summary["run_id"] is None
    assert run_regression.actions_run() == (None, None)

    monkeypatch.setenv("GITHUB_REPOSITORY", "Comfy-Org/comfyci-runpod-worker")
    monkeypatch.setenv("GITHUB_RUN_ID", "43658104127")
    assert run_regression.main() == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    run_url = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs/43658104127"
    assert summary["run_url"] == run_url and summary["run_id"] == 43658104127
    # One helper for the summary and the start marker: they name the same run.
    args = type("Args", (), {"commit": "new", "branch": "master"})()
    assert run_regression.attempt_marker(args, "main", 1)["run_url"] == run_url
    # Every key the summary had before is still there.
    assert {"schema_version", "branch", "commit", "run_ts", "overall", "has_infra_error", "lane",
            "attempt", "env", "commit_meta", "tested_range", "workflows"} <= set(summary)


def test_a_rerun_records_its_attempt(tmp_path, monkeypatch, capsys):
    """The cron retries a commit with infra errors once; the summary counts the
    attempts so the next tick knows the retry was spent."""
    import json
    import sys
    import commit_meta
    import storage
    store = FakeStore({"regression/runs/master/new/summary.json": {
        "commit": "new", "overall": "pass", "has_infra_error": True,
        "workflows": {"sdxl_t2i": {"verdict": "infra_error"}}}})
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": "pass", "detail": None})
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--workflows", "sdxl_t2i", "--skip-publish",
                                      "--summary-out", str(out)])
    assert run_regression.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["attempt"] == 2
    assert "this is attempt 2" in capsys.readouterr().out
    # A later marker (a retry that died before publishing) raises the count.
    store.files["regression/runs/master/new/attempt.json"] = {"attempt": 2}
    assert run_regression.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["attempt"] == 3


def test_the_start_marker_is_pushed_before_any_gpu_job(tmp_path, monkeypatch):
    """A run that dies after its GPU jobs (here: the results push fails) has
    already pushed its attempt marker, so the cron can count it."""
    import sys
    import commit_meta
    import storage
    events = []

    class RecordingStore(FakeStore):
        def snapshot_manifest(self, commit, manifest):
            events.append("snapshot")

        def mark_attempt(self, branch, commit, marker):
            events.append(("marker", branch, commit, marker["attempt"], marker["lane"],
                           marker.get("run_url")))

        def checkpoint(self, message):
            events.append(("checkpoint", message))

        def publish_summary(self, branch, commit, summary):
            raise RuntimeError("could not push results")

    store = RecordingStore({"regression/runs/master/new/attempt.json": {"attempt": 1}})
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "Comfy-Org/comfyci-runpod-worker")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")

    def gpu(wf_id, *a):
        events.append(("gpu", wf_id))
        return {"workflow_id": wf_id, "worker_status": "ok", "verdict": "pass", "detail": None}

    monkeypatch.setattr(run_regression, "process_workflow", gpu)
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--manifest", _manifest(tmp_path, "sdxl_t2i")])
    with pytest.raises(RuntimeError, match="could not push results"):
        run_regression.main()
    run_url = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs/42"
    # Attempt 1 left only its marker, so this is attempt 2.
    assert events == ["snapshot", ("marker", "master", "new", 2, "main", run_url),
                      ("checkpoint", "regression master@new: attempt 2 started"),
                      ("gpu", "sdxl_t2i")]


def test_a_marker_that_cannot_be_pushed_stops_the_run_before_the_gpu(tmp_path, monkeypatch):
    import sys
    import commit_meta
    import storage

    class NoPushStore(FakeStore):
        def snapshot_manifest(self, commit, manifest):
            pass

        def mark_attempt(self, branch, commit, marker):
            pass

        def checkpoint(self, message):
            raise RuntimeError("could not push checkpoint")

    monkeypatch.setattr(storage, "from_args", lambda args: NoPushStore({}))
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda *a: pytest.fail("no GPU job without a pushed marker"))
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--manifest", _manifest(tmp_path, "sdxl_t2i")])
    with pytest.raises(RuntimeError, match="could not push checkpoint"):
        run_regression.main()


def test_summary_out_is_not_written_when_publishing_fails(tmp_path, monkeypatch):
    import sys
    import pytest
    import commit_meta
    import storage

    class BrokenStore(FakeStore):
        def snapshot_manifest(self, commit, manifest):
            pass

        def mark_attempt(self, branch, commit, marker):
            pass

        def checkpoint(self, message):
            pass

        def publish_summary(self, branch, commit, summary):
            raise RuntimeError("push rejected")

    monkeypatch.setattr(storage, "from_args", lambda args: BrokenStore({}))
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": "fail", "detail": "new_drift"})
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--summary-out", str(out)])
    with pytest.raises(RuntimeError):
        run_regression.main()
    # No summary, so no PR comment linking to results that were never published.
    assert not out.exists()


def test_lane_filter_skips_workflows_restricted_to_other_lanes(tmp_path, monkeypatch, capsys):
    """A workflow restricted to another lane is skipped on this one even when
    named explicitly; the summary simply has no cell for it."""
    import json
    import sys
    import commit_meta
    import storage
    manifest = tmp_path / "manifest" / "workflows.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [], "thresholds": {}},
        "workflows": {
            "sdxl_t2i": {"enabled": True, "workflow_path": "x", "models": []},
            "kjnodes_sdxl_t2i": {"enabled": True, "workflow_path": "x", "models": [],
                                 "custom_nodes": ["ComfyUI-KJNodes"],
                                 "lanes": ["other"]},
        }}))
    monkeypatch.setattr(storage, "from_args", lambda args: FakeStore({}))
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    ran = []
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: ran.append(wf_id) or {"workflow_id": wf_id, "worker_status": "ok",
                                                                "verdict": "pass", "detail": None})
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--manifest", str(manifest), "--commit", "new",
                                      "--branch", "master", "--lane", "main",
                                      "--workflows", "sdxl_t2i,kjnodes_sdxl_t2i",
                                      "--skip-publish", "--summary-out", str(out)])
    assert run_regression.main() == 0
    assert ran == ["sdxl_t2i"]
    assert set(json.loads(out.read_text())["workflows"]) == {"sdxl_t2i"}
    assert "[kjnodes_sdxl_t2i] not run on lane main" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--manifest", str(manifest), "--commit", "new",
                                      "--branch", "master", "--lane", "main",
                                      "--workflows", "kjnodes_sdxl_t2i", "--skip-publish"])
    assert run_regression.main() == 0
    assert ran == ["sdxl_t2i"]
    assert "no workflows selected for lane" in capsys.readouterr().out


def test_custom_node_errors_are_execution_failures():
    assert "custom_node_error" in run_regression.EXEC_FAIL_STATUSES
    assert "custom_node_install_error" in run_regression.INFRA_STATUSES
    assert not run_regression.EXEC_FAIL_STATUSES & run_regression.INFRA_STATUSES


def test_a_retired_lane_runs_nothing(monkeypatch):
    """A retired lane is refused before any GPU job, named or reached through
    the default prefix; its endpoint is scaled to zero."""
    import sys
    import storage
    reg = {"primary": "new", "lanes": {
        "old": {"cadence": "retired", "role": "legacy", "layout": 1, "prefix": "regression",
                "endpoint_id": "ep-old"},
        "new": {"cadence": "per-commit", "role": "primary", "layout": 2,
                "prefix": "regression/lanes/new", "endpoint_id": "ep-new"}}}
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: reg)
    monkeypatch.setattr(storage, "from_args", lambda args: pytest.fail("no store for a retired lane"))
    for extra in (["--lane", "old"], []):
        monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                          "--skip-publish", *extra])
        with pytest.raises(SystemExit, match="lane 'old' is retired.*the primary is 'new'"):
            run_regression.main()


# -- a run that names its workflows -------------------------------------------------
SUMMARY_KEYS = ["schema_version", "branch", "commit", "run_ts", "overall", "has_infra_error",
                "lane", "attempt", "run_url", "run_id", "env", "commit_meta", "tested_range",
                "workflows"]


class RecordingStore(FakeStore):
    """FakeStore with the write side: `events` is what a run writes, in order."""

    def __init__(self, files=None):
        super().__init__(dict(files or {}))
        self.events = []

    def snapshot_manifest(self, commit, manifest):
        self.events.append("snapshot")

    def mark_attempt(self, branch, commit, marker):
        self.events.append("marker")
        self.files[f"regression/runs/{branch}/{commit}/attempt.json"] = marker

    def checkpoint(self, message):
        self.events.append("checkpoint")

    def publish_summary(self, branch, commit, summary):
        import json
        self.events.append("summary")
        self.files[f"regression/runs/{branch}/{commit}/summary.json"] = json.loads(
            json.dumps(summary))

    def update_latest(self, branch, pointer):
        self.events.append("latest")
        self.files[f"regression/latest/{branch}.json"] = pointer

    def register_derived(self, fn):
        self.events.append("index")

    def finalize(self, message):
        self.events.append(message)


def _run(tmp_path, monkeypatch, store, verdicts, *argv):
    """main() for commit `new` on a manifest of the workflows in `verdicts`,
    with a GPU that answers each with its verdict. Returns (exit code, the
    --summary-out file)."""
    import json
    import sys
    import commit_meta
    import storage
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": verdicts[wf_id], "detail": None,
                                           "env": {"image": "img"}})
    out = tmp_path / "summary-out.json"
    out.unlink(missing_ok=True)
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--commit", "new", "--branch", "master",
                                      "--manifest", _manifest(tmp_path, *verdicts),
                                      "--summary-out", str(out), *argv])
    code = run_regression.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_a_run_that_names_its_workflows_is_not_the_lanes_run_of_the_commit(tmp_path, monkeypatch):
    """A dispatch with workflows=<ids> at a commit the lane has no result for
    (a first look at master HEAD, a bisect step). It says which workflows it
    tested, pushes no start marker and leaves the previous-run pointer alone,
    so the cron still tests the commit and compares with the right run."""
    import regression_skip
    verdicts = {"a": "pass", "b": "pass", "c": "pass"}
    store = RecordingStore({"regression/latest/master.json": {"commit": "old"}})
    code, summary = _run(tmp_path, monkeypatch, store, verdicts, "--workflows", "b,a")
    assert code == 0
    assert list(summary) == SUMMARY_KEYS + ["scope"]
    assert summary["scope"] == ["a", "b"] == list(summary["workflows"])
    assert store.events == ["snapshot", "summary", "index", "regression master@new: pass"]
    assert store.latest_pointer("master") == {"commit": "old"}
    assert store.attempt_marker("master", "new") is None
    assert store.published_summary("master", "new") == summary

    def cron():
        return regression_skip.decide("new", store.published_summary("master", "new"),
                                      store.latest_pointer("master"),
                                      marker=store.attempt_marker("master", "new"))[0]

    assert cron()
    # The lane's own run of the commit: counted as the first attempt, and its
    # summary takes the place of the named one.
    store.events.clear()
    code, summary = _run(tmp_path, monkeypatch, store, verdicts)
    assert code == 0 and list(summary) == SUMMARY_KEYS and summary["attempt"] == 1
    assert store.events == ["snapshot", "marker", "checkpoint", "summary", "latest", "index",
                            "regression master@new: pass"]
    assert store.published_summary("master", "new") == summary
    assert store.latest_pointer("master") == {"commit": "new", "run_ts": summary["run_ts"],
                                              "workflows": ["a", "b", "c"]}
    assert not cron()


def test_a_second_named_run_adds_to_the_first(tmp_path, monkeypatch):
    """Two bisect steps of different workflows at one commit: the second keeps
    the first one's cell, and the summary still is not the lane's own."""
    import regression_skip
    verdicts = {"a": "pass", "b": "execution_error"}
    store = RecordingStore()
    _, first = _run(tmp_path, monkeypatch, store, verdicts, "--workflows", "a")
    code, summary = _run(tmp_path, monkeypatch, store, verdicts, "--workflows", "b")
    assert code == 1
    assert list(summary) == SUMMARY_KEYS + ["scope", "retests"]
    assert summary["scope"] == ["a", "b"] and summary["overall"] == "fail"
    assert summary["workflows"]["a"] == first["workflows"]["a"]
    assert [r["workflows"] for r in summary["retests"]] == [["b"]]
    assert store.events.count("marker") == 0 and store.events.count("latest") == 0
    assert store.events[-1] == "regression master@new: fail"
    assert regression_skip.decide("new", store.published_summary("master", "new"))[0]


def test_a_retest_that_passes_does_not_make_a_failing_commit_clean(tmp_path, monkeypatch):
    """The summary, and the message of the publish, speak for every cell of
    the commit. The job's own colour is that of the workflows it tested."""
    verdicts = {"a": "fail", "b": "infra_error"}
    store = RecordingStore()
    code, first = _run(tmp_path, monkeypatch, store, verdicts)
    assert code == 1 and first["has_infra_error"] and first["attempt"] == 1
    verdicts["b"] = "pass"
    store.events.clear()
    code, summary = _run(tmp_path, monkeypatch, store, verdicts, "--workflows", "b")
    assert code == 0
    assert (summary["overall"], summary["has_infra_error"]) == ("fail", False)
    assert store.events == ["snapshot", "summary", "index", "regression master@new: fail"]
    assert list(summary) == SUMMARY_KEYS + ["retests"]
    assert (summary["attempt"], summary["run_ts"]) == (1, first["run_ts"])
    assert summary["workflows"]["a"] == first["workflows"]["a"]


def test_a_first_look_with_publish_off_writes_nothing(tmp_path, monkeypatch):
    lane_result = {"commit": "new", "overall": "pass", "attempt": 1, "run_ts": 5,
                   "workflows": {"a": {"verdict": "pass"}, "b": {"verdict": "pass"}}}
    store = RecordingStore({"regression/runs/master/new/summary.json": lane_result})
    code, summary = _run(tmp_path, monkeypatch, store, {"a": "pass", "b": "pass", "c": "fail"},
                         "--workflows", "c", "--skip-publish")
    assert code == 1 and store.events == []
    assert store.published_summary("master", "new") == lane_result
    # The report of that run is built from this file: the run's own result, nothing merged.
    assert list(summary) == SUMMARY_KEYS + ["scope"]
    assert list(summary["workflows"]) == ["c"] == summary["scope"]


def test_merge_retest_key_by_key():
    import copy
    old = {"schema_version": 2, "branch": "master", "commit": "c", "run_ts": 100,
           "overall": "fail", "has_infra_error": True, "lane": "weekly", "attempt": 1,
           "run_url": "u1", "run_id": 1, "env": {"image": "old"}, "commit_meta": {"pr": 9},
           "tested_range": {"prev": "p", "commits_between": 44, "compare_url": "x"},
           "workflows": {"a": {"workflow_id": "a", "verdict": "fail", "exec": 104.19999999999999},
                         "b": {"workflow_id": "b", "verdict": "infra_error"},
                         "c": {"workflow_id": "c", "verdict": "pass"}},
           "ranges": {"a": {"prev": "p", "commits_between": 44},
                      "b": {"prev": "q", "commits_between": 3}}}
    new = {"schema_version": 2, "branch": "master", "commit": "c", "run_ts": 200,
           "overall": "pass", "has_infra_error": False, "lane": "weekly", "attempt": 2,
           "run_url": "u2", "run_id": 2, "env": {"image": "new"}, "commit_meta": {"pr": 10},
           "tested_range": None,
           "workflows": {"d": {"workflow_id": "d", "verdict": "no_baseline"},
                         "b": {"workflow_id": "b", "verdict": "pass"}}}
    before = copy.deepcopy(old)
    merged = run_regression.merge_retest(old, new)
    assert old == before, "the published summary is read, never changed in place"
    # Today's keys in today's order, then what the re-test adds.
    assert list(merged) == [*old, "retests"]
    for key in ("schema_version", "branch", "commit", "run_ts", "lane", "attempt", "run_url",
                "run_id", "env", "commit_meta", "tested_range"):
        assert merged[key] == old[key], key
    assert list(merged["workflows"]) == ["a", "b", "c", "d"]
    assert merged["workflows"]["a"] is old["workflows"]["a"]
    assert merged["workflows"]["c"] is old["workflows"]["c"]
    assert merged["workflows"]["b"] == {"workflow_id": "b", "verdict": "pass"}
    assert (merged["overall"], merged["has_infra_error"]) == ("fail", False)
    assert merged["ranges"] == {"a": {"prev": "p", "commits_between": 44}}
    assert merged["retests"] == [{"run_ts": 200, "run_url": "u2", "run_id": 2,
                                  "env": {"image": "new"}, "workflows": ["b", "d"]}]
    assert "scope" not in merged
    # A second re-test: the list grows, a new range replaces the old one, and
    # the last range that goes takes the key with it.
    fixed = {**new, "run_ts": 300, "run_id": 3,
             "workflows": {"a": {"workflow_id": "a", "verdict": "fail"}},
             "ranges": {"a": {"prev": "r", "commits_between": 1}}}
    again = run_regression.merge_retest(merged, fixed)
    assert [r["run_id"] for r in again["retests"]] == [2, 3]
    assert again["ranges"] == {"a": {"prev": "r", "commits_between": 1}}
    assert list(again) == list(merged)
    passed = {**new, "workflows": {"a": {"workflow_id": "a", "verdict": "pass"}}}
    clean = run_regression.merge_retest(again, passed)
    assert "ranges" not in clean and clean["overall"] == "pass"
    # The roll-up is over every cell: a kept infra error keeps the summary marked.
    kept = run_regression.merge_retest(old, passed)
    assert (kept["overall"], kept["has_infra_error"]) == ("pass", True)
    # A summary of named workflows stays one: its list grows.
    scoped = run_regression.merge_retest({**old, "scope": ["a", "b", "c"]}, new)
    assert scoped["scope"] == ["a", "b", "c", "d"]
    assert list(scoped) == [*old, "scope", "retests"]


def test_each_job_further_back_in_the_run_may_wait_longer_in_the_queue(tmp_path, monkeypatch):
    """The run submits every job at once to an endpoint with 3 workers. With a
    fixed 600 s the last jobs of 15 were cancelled for standing behind their
    own run, and submitted again."""
    import sys
    import types
    import commit_meta
    import storage
    assert [run_regression.queue_deadline_s(i) for i in (0, 1, 2, 14)] == [600, 660, 720, 1440]
    process_workflow = run_regression.process_workflow
    seen = {}
    monkeypatch.setattr(storage, "from_args", lambda args: FakeStore({}))
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: seen.update({wf_id: a[-1]}) or {
                            "workflow_id": wf_id, "worker_status": "ok", "verdict": "pass"})
    argv = ["run_regression.py", "--commit", "new", "--branch", "master", "--skip-publish",
            "--manifest", _manifest(tmp_path, "c", "a", "b")]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_regression.main() == 0
    assert seen == {"c": 600, "a": 660, "b": 720}  # the order of the manifest
    seen.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--workflows", "b,c"])
    assert run_regression.main() == 0
    assert seen == {"c": 600, "b": 660}

    # process_workflow hands it to the job. Called without it, the job gets today's 600 s.
    def job(client, wf_id, workflow, **kw):
        seen[wf_id] = kw["queue_deadline_s"]
        return {"status": "worker_error", "outputs": []}

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
    (tmp_path / "wf_api.json").write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(workdir=str(tmp_path / "work"), endpoint_id="ep", lane_id="main",
                                 commit="new", repo_url=None, branch="master", skip_publish=True)
    entry = {"workflow_path": "wf_api.json", "models": []}
    process_workflow("first", entry, DEFAULTS, args, tmp_path, None)
    process_workflow("fourth", entry, DEFAULTS, args, tmp_path, None, 780)
    assert (seen["first"], seen["fourth"]) == (600, 780)


NOW = 1794700000
RUNS = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs"


class ImageLane:
    """Three image workflows on the per-commit lane `other`, through main():
    a results branch in a local repository that holds their references, and a
    GPU that writes one PNG of each workflow's shade (100 is the reference's)."""
    PREFIX = "regression/lanes/other"
    IDS = ("wf_a", "wf_b", "wf_c")

    def __init__(self, tmp_path, monkeypatch):
        import json
        import commit_meta
        import storage
        from conftest import make_remote
        self.tmp, self.monkeypatch, self.runs = tmp_path, monkeypatch, 0
        self.shade = dict.fromkeys(self.IDS, 100)
        self.status = dict.fromkeys(self.IDS, "ok")
        repo, seed = tmp_path / "repo", {}
        (repo / "manifest" / "workflows").mkdir(parents=True)
        for wf_id in self.IDS:
            (repo / "manifest" / "workflows" / f"{wf_id}_api.json").write_text("{}")
            _png(tmp_path / "reference.png", 100)
            seed[f"{self.PREFIX}/golden/{wf_id}/current.json"] = {"tag": "v1"}
            seed[f"{self.PREFIX}/golden/{wf_id}/v1/outputs/{wf_id}_00001_.png"] = (
                tmp_path / "reference.png").read_bytes()
        self.manifest = repo / "manifest" / "workflows.json"
        self.manifest.write_text(json.dumps({"defaults": DEFAULTS, "workflows": {
            wf_id: {"enabled": True, "workflow_path": f"manifest/workflows/{wf_id}_api.json",
                    "models": []} for wf_id in self.IDS}}), encoding="utf-8")
        self.remote = make_remote(tmp_path, seed)
        monkeypatch.setattr(storage, "from_args", lambda args: storage.GitHubStorage(
            "local/test", "results", args.results_workdir, args.prefix,
            remote_url=str(self.remote)))
        monkeypatch.setattr(run_regression, "run_workflow_job", self.job)
        monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
        monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: {
            "subject": "change (#9)", "pr": 9, "committed_ts": {"c1": 10, "c2": 20}[sha]})
        monkeypatch.setattr(commit_meta, "fetch_range", lambda prev, sha: {
            "prev": prev, "commits_between": 1, "compare_url": f"{prev}...{sha}"})
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "Comfy-Org/comfyci-runpod-worker")

    def job(self, client, wf_id, workflow, workdir=None, **kw):
        import hashlib
        if self.status[wf_id] != "ok":
            return {"status": self.status[wf_id], "outputs": [], "error": {"message": "boom"}}
        png = workdir / "outputs" / f"{wf_id}_00001_.png"
        _png(png, self.shade[wf_id])
        return {"status": "ok", "error": None, "comfy_version": "0.45.0",
                "torch_version": "2.14.1+cu130", "python_version": "3.13.16",
                "gpu_name": "NVIDIA GeForce RTX 4090", "vram_peak_mb": 21480.5,
                "rss_peak_mb": 30110.25, "custom_nodes": None,
                "timings": {"checkout_s": 1.9, "prompt_exec_s": 104.19999999999999},
                "env": {"image": "img", "image_sha": "f" * 40}, "outputs_dir": str(png.parent),
                "outputs": [{"filename": png.name, "bytes": png.stat().st_size,
                             "sha256": hashlib.sha256(png.read_bytes()).hexdigest()}]}

    def run(self, commit, *argv, now=NOW, run_id=1):
        """One Actions run of main(): (exit code, the text of its --summary-out file)."""
        import sys
        import types
        self.runs += 1
        out = self.tmp / f"out{self.runs}.json"
        self.monkeypatch.setattr(run_regression, "time", types.SimpleNamespace(time=lambda: now))
        self.monkeypatch.setenv("GITHUB_RUN_ID", str(run_id))
        self.monkeypatch.setattr(sys, "argv", [
            "run_regression.py", "--manifest", str(self.manifest), "--commit", commit,
            "--branch", "master", "--lane", "other", "--workdir", str(self.tmp / f"w{self.runs}"),
            "--results-workdir", str(self.tmp / f"r{self.runs}"), "--summary-out", str(out),
            *argv])
        return run_regression.main(), out.read_text(encoding="utf-8")

    def published(self, path, prefix=PREFIX):
        """The text of a file on the results branch (None: there is none)."""
        import subprocess
        r = subprocess.run(["git", "-C", str(self.remote), "cat-file", "-p",
                            f"results:{prefix}/{path}"], capture_output=True)
        return r.stdout.decode("utf-8").replace("\r\n", "\n") if r.returncode == 0 else None


def _cell_text(summary_text, wf_id):
    """One workflow's cell as it stands in the text of a summary.json."""
    import re
    return re.search(rf'^    "{wf_id}": \{{\n.*?^    \}}', summary_text, re.S | re.M).group(0)


def test_a_full_image_run_on_a_per_commit_lane_publishes_what_it_always_did(tmp_path, monkeypatch):
    """What the pin of image behaviour cannot see, because it does not run
    main(): the keys of the summary, the start marker and the previous-run
    pointer of a full run of image workflows."""
    import json
    lane = ImageLane(tmp_path, monkeypatch)
    code, out = lane.run("c1")
    assert code == 0
    assert lane.published("latest/master.json") == (
        '{\n  "commit": "c1",\n  "run_ts": 1794700000,\n  "workflows": [\n    "wf_a",\n'
        '    "wf_b",\n    "wf_c"\n  ]\n}')
    text = lane.published("runs/master/c1/summary.json")
    assert text == out, "--summary-out is the published file"
    summary = json.loads(text)
    assert list(summary) == SUMMARY_KEYS and list(summary["workflows"]) == list(lane.IDS)
    for wf_id, cell in summary["workflows"].items():
        assert list(cell) == CELL_KEYS + ["prior_verdict", "prior_commit"], wf_id
    assert (summary["attempt"], summary["run_id"], summary["tested_range"]) == (1, 1, None)
    assert json.loads(lane.published("runs/master/c1/attempt.json")) == {
        "commit": "c1", "branch": "master", "lane": "other", "attempt": 1, "started_ts": NOW,
        "run_url": f"{RUNS}/1"}
    # A full run with an execution error is a result too. It does not move the pointer.
    lane.status["wf_c"] = "timeout"
    code, out = lane.run("c2", now=NOW + 600, run_id=2)
    summary = json.loads(out)
    assert code == 1 and list(summary) == SUMMARY_KEYS
    assert summary["tested_range"] == {"prev": "c1", "commits_between": 1,
                                       "compare_url": "c1...c2"}
    assert json.loads(lane.published("latest/master.json"))["commit"] == "c1"
    assert json.loads(lane.published("runs/master/c2/attempt.json"))["attempt"] == 1


def test_a_retest_of_one_workflow_keeps_every_other_cell_byte_for_byte(tmp_path, monkeypatch):
    """A dispatch with workflows=<id> at a commit the lane already has a
    result for. Before, its summary took the place of the lane's: one workflow
    tested again cost the cells of all the others."""
    import json
    import regression_skip
    lane = ImageLane(tmp_path, monkeypatch)
    assert lane.run("c1")[0] == 0
    # The lane's run of c2: one output changed, and one workflow met a worker error.
    lane.shade["wf_b"], lane.status["wf_c"] = 180, "worker_error"
    assert lane.run("c2", now=NOW + 600, run_id=2)[0] == 1
    old_text = lane.published("runs/master/c2/summary.json")
    old = json.loads(old_text)
    assert (old["overall"], old["has_infra_error"]) == ("fail", True)
    assert {wf: cell["verdict"] for wf, cell in old["workflows"].items()} == {
        "wf_a": "pass", "wf_b": "fail", "wf_c": "infra_error"}
    pointer = lane.published("latest/master.json")
    marker = lane.published("runs/master/c2/attempt.json")
    assert json.loads(pointer)["commit"] == "c1"

    # An hour later, in another Actions run, wf_c is tested again at c2.
    lane.status["wf_c"] = "ok"
    code, out = lane.run("c2", "--workflows", "wf_c", now=NOW + 4200, run_id=3)
    new_text = lane.published("runs/master/c2/summary.json")
    new = json.loads(new_text)
    assert code == 0 and new_text == out
    for wf_id in ("wf_a", "wf_b"):
        assert _cell_text(new_text, wf_id) == _cell_text(old_text, wf_id), wf_id
        assert list(new["workflows"][wf_id]) == CELL_KEYS + ["prior_verdict", "prior_commit"]
    assert '"prompt_exec_s": 104.19999999999999' in _cell_text(new_text, "wf_a")
    assert new["workflows"]["wf_c"]["verdict"] == "pass"
    assert new["workflows"]["wf_c"]["prior_commit"] == "c1"
    # The summary stays the lane's result of c2: its time, its run, its range.
    assert list(new) == SUMMARY_KEYS + ["retests"]
    assert (new["run_ts"], new["run_id"], new["run_url"]) == (NOW + 600, 2, f"{RUNS}/2")
    assert new["tested_range"] == {"prev": "c1", "commits_between": 1, "compare_url": "c1...c2"}
    assert new["retests"] == [{"run_ts": NOW + 4200, "run_url": f"{RUNS}/3", "run_id": 3,
                               "env": {"image": "img", "image_sha": "f" * 40},
                               "workflows": ["wf_c"]}]
    # The whole file: the old one with that cell, the roll-up over every cell, and the re-test.
    expected = {**old, "has_infra_error": False, "retests": new["retests"],
                "workflows": {**old["workflows"], "wf_c": new["workflows"]["wf_c"]}}
    assert expected["overall"] == "fail"  # wf_b still fails: the commit is not clean
    assert new_text == json.dumps(expected, indent=2)
    # No start marker of its own, and the previous-run pointer stays where it was.
    assert lane.published("runs/master/c2/attempt.json") == marker
    assert lane.published("latest/master.json") == pointer
    # The history row of c2 still links the lane's run, with the new cell.
    head = json.loads(lane.published("index/master/other.json", prefix="regression"))
    entry = next(e for e in head["entries"] if e["c"] == "c2")
    assert (entry["run"], entry["t"], entry["o"], entry["infra"]) == (2, NOW + 600, "fail", False)
    assert {wf: cell["v"] for wf, cell in entry["w"].items()} == {
        "wf_a": "pass", "wf_b": "fail", "wf_c": "pass"}
    # And the cron leaves the commit alone: its infra error is gone.
    assert regression_skip.decide("c2", old, json.loads(pointer), marker=json.loads(marker))[0]
    assert not regression_skip.decide("c2", new, json.loads(pointer),
                                      marker=json.loads(marker))[0]


# -- limits -------------------------------------------------------------------------
DEFAULTS = {"timeout_s": 600, "comfy_flags": [],
            "thresholds": {"max_mean_mse": 2.0, "min_mean_psnr_db": 45.0,
                           "max_pct_pixels_changed": 1.0}}


def test_derive_thresholds():
    d = run_regression.derive_thresholds
    assert d({}, DEFAULTS, None) == DEFAULTS["thresholds"]
    assert d({}, DEFAULTS, None) is not DEFAULTS["thresholds"]  # a copy: loosening never leaks
    # An identical or failed re-run of the golden loosens nothing.
    assert d({}, DEFAULTS, {"identical": True, "mean_mse": 0.0}) == DEFAULTS["thresholds"]
    assert d({}, DEFAULTS, {"error": "no PNGs", "mean_mse": 9.0}) == DEFAULTS["thresholds"]
    # A noisy golden: three times its own noise, never tighter than the defaults.
    noisy = {"identical": False, "mean_mse": 1.5, "mean_psnr_db": 46.37,
             "mean_pct_pixels_changed": 2.0}
    assert d({}, DEFAULTS, noisy) == {"max_mean_mse": 4.5, "min_mean_psnr_db": 41.6,
                                      "max_pct_pixels_changed": 6.0}
    quiet = {"identical": False, "mean_mse": 0.1, "mean_psnr_db": 58.13,
             "mean_pct_pixels_changed": 0.1}
    assert d({}, DEFAULTS, quiet) == DEFAULTS["thresholds"]
    # The entry's own limits win over both.
    own = {"max_mean_mse": 9.0, "min_mean_psnr_db": 30.0, "max_pct_pixels_changed": 50.0}
    assert d({"thresholds": own}, DEFAULTS, noisy) is own


# -- an image workflow through a run ------------------------------------------------
CELL_KEYS = ["workflow_id", "worker_status", "verdict", "detail", "vs_golden", "vs_previous",
             "thresholds_used", "golden_tag", "previous_commit", "previous_verdict", "figures",
             "output_sha256", "thumbnail", "gpu_name", "timings", "vram_peak_mb", "rss_peak_mb",
             "comfy_version", "torch_version", "python_version", "env", "custom_nodes"]
COMPARISON_KEYS = ["verdict", "detail", "vs_golden", "vs_previous", "thresholds_used",
                   "golden_tag", "previous_commit", "previous_verdict", "previous_golden_tag",
                   "figures"]
METRIC_KEYS = ["frames_compared", "frame_count_ref", "frame_count_cand", "identical", "mean_mse",
               "mean_psnr_db", "min_psnr_db", "max_abs_diff", "mean_pct_pixels_changed",
               "worst_frame_index"]


def _png(path, shade):
    import numpy as np
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 24, 3), shade, dtype=np.uint8)).save(path)


class ImageStore:
    """A results store for one image workflow: a golden and a previous run,
    each a PNG of one shade (None: there is none)."""

    def __init__(self, golden=100, previous=100, previous_verdict="pass", floor=None):
        self.golden, self.previous = golden, previous
        self.previous_verdict, self.floor = previous_verdict, floor
        self.published = None

    def _write(self, dest, shade):
        _png(dest / "wf_00001_.png", shade)
        return 1

    def latest_pointer(self, branch):
        return {"commit": "prev"} if self.previous is not None else None

    def fetch_run_outputs(self, branch, commit, wf_id, dest):
        return self._write(dest, self.previous)

    def run_summary(self, branch, commit):
        return {"workflows": {"wf": {"verdict": self.previous_verdict, "golden_tag": "v1"}}}

    def golden_current(self, wf_id):
        return {"tag": "v1"} if self.golden is not None else None

    def fetch_golden_outputs(self, wf_id, tag, dest):
        return self._write(dest, self.golden)

    def noise_floor(self, wf_id, tag):
        return self.floor

    def publish_workflow_run(self, branch, commit, wf_id, local_dir):
        self.published = sorted(p.relative_to(local_dir).as_posix()
                                for p in local_dir.rglob("*") if p.is_file())


def _image_run(tmp_path, monkeypatch, store, shade=100, status="ok", extra=()):
    """process_workflow for an image workflow whose job wrote one PNG of
    `shade` (plus the files named in `extra`); returns (cell, comparison.json)."""
    import json
    import types

    def job(client, wf_id, workflow, workdir=None, **kw):
        if status != "ok":
            return {"status": status, "outputs": [], "error": {"message": "boom"}}
        out = workdir / "outputs"
        _png(out / "wf_00001_.png", shade)
        for name in extra:
            if name.endswith(".png"):
                _png(out / name, shade)
            else:
                (out / name).write_bytes(b"not a png")
        return {"status": "ok", "outputs_dir": str(out), "gpu_name": "gpu",
                "timings": {"prompt_exec_s": 3.0},
                "outputs": [{"filename": p.name, "bytes": p.stat().st_size, "sha256": p.name}
                            for p in sorted(out.iterdir())]}

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "wf_api.json").write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(workdir=str(tmp_path / "work"), endpoint_id="ep", lane_id="main",
                                 commit="new", repo_url=None, branch="master", skip_publish=False)
    cell = run_regression.process_workflow("wf", {"workflow_path": "wf_api.json", "models": []},
                                           DEFAULTS, args, tmp_path, store)
    comparison = json.loads((tmp_path / "work" / "wf" / "comparison.json").read_text())
    # What is published for an image workflow never names a kind: absent means image.
    assert list(cell) == CELL_KEYS and list(comparison) == COMPARISON_KEYS
    for key in COMPARISON_KEYS:
        if key != "previous_golden_tag":
            assert cell[key] == comparison[key], key
    return cell, comparison


def test_an_image_workflow_that_matches_passes(tmp_path, monkeypatch):
    store = ImageStore()
    cell, comparison = _image_run(tmp_path, monkeypatch, store)
    assert (cell["verdict"], cell["detail"], cell["worker_status"]) == ("pass", None, "ok")
    assert list(cell["vs_golden"]) == METRIC_KEYS == list(cell["vs_previous"])
    assert cell["vs_golden"]["identical"] is True and cell["vs_previous"]["identical"] is True
    assert cell["thresholds_used"] == DEFAULTS["thresholds"] and cell["golden_tag"] == "v1"
    assert (cell["previous_commit"], cell["previous_verdict"]) == ("prev", "pass")
    assert comparison["previous_golden_tag"] == "v1"
    assert cell["figures"] is False and cell["thumbnail"] == "wf_00001_.webp"
    assert cell["output_sha256"] == "wf_00001_.png"  # the first PNG's sha (here: its name)
    assert store.published == ["comparison.json", "outputs/thumbs/wf_00001_.webp",
                               "outputs/wf_00001_.png", "run.json"]


def test_a_changed_image_fails_as_new_then_as_inherited(tmp_path, monkeypatch):
    store = ImageStore()
    cell, _ = _image_run(tmp_path / "new", monkeypatch, store, shade=180)
    assert (cell["verdict"], cell["detail"], cell["figures"]) == ("fail", "new_drift", True)
    assert cell["vs_golden"]["mean_mse"] == 6400.0 and cell["vs_golden"]["max_abs_diff"] == 80
    assert cell["vs_previous"]["identical"] is False
    # Figures against the reference and against the previous run.
    assert store.published == [
        "comparison.json", "figures/diff_heatmap_golden.png", "figures/diff_heatmap_prev.png",
        "figures/side_by_side_golden.png", "figures/side_by_side_prev.png",
        "outputs/thumbs/wf_00001_.webp", "outputs/wf_00001_.png", "run.json"]

    # The same output again, after a run that already failed: nothing is rendered twice.
    store = ImageStore(previous=180, previous_verdict="fail")
    cell, _ = _image_run(tmp_path / "again", monkeypatch, store, shade=180)
    assert (cell["verdict"], cell["detail"], cell["figures"]) == ("fail", "inherited", False)
    assert not [name for name in store.published if name.startswith("figures/")]
    # Identical to a previous run that passed: the reference changed, not the output.
    store = ImageStore(previous=180, previous_verdict="pass")
    cell, _ = _image_run(tmp_path / "moved", monkeypatch, store, shade=180)
    assert (cell["verdict"], cell["detail"], cell["figures"]) == ("fail", None, True)


def test_a_small_difference_passes_within_the_limits_and_a_noisy_golden_widens_them(
        tmp_path, monkeypatch):
    cell, _ = _image_run(tmp_path / "near", monkeypatch, ImageStore(), shade=101)
    assert cell["verdict"] == "fail"  # every pixel moved: past the 1 % default
    floor = {"identical": False, "mean_mse": 1.0, "mean_psnr_db": 48.13,
             "mean_pct_pixels_changed": 100.0}
    cell, _ = _image_run(tmp_path / "noisy", monkeypatch, ImageStore(floor=floor), shade=101)
    assert cell["verdict"] == "pass" and cell["vs_golden"]["identical"] is False
    assert cell["thresholds_used"] == {"max_mean_mse": 3.0, "min_mean_psnr_db": 43.36,
                                       "max_pct_pixels_changed": 300.0}
    assert cell["figures"] is True  # a pass that is not bit-identical still shows its figures


def test_an_image_workflow_without_a_reference_or_a_previous_run(tmp_path, monkeypatch):
    store = ImageStore(golden=None)
    cell, _ = _image_run(tmp_path / "no-golden", monkeypatch, store)
    assert (cell["verdict"], cell["vs_golden"], cell["thresholds_used"]) == ("no_baseline", None,
                                                                            None)
    assert cell["golden_tag"] is None and cell["vs_previous"]["identical"] is True
    assert cell["thumbnail"] == "wf_00001_.webp" and cell["output_sha256"] == "wf_00001_.png"

    cell, comparison = _image_run(tmp_path / "first", monkeypatch, ImageStore(previous=None),
                                  shade=180)
    assert (cell["verdict"], cell["detail"]) == ("fail", None)  # nothing says whether it is new
    assert cell["vs_previous"] is None and cell["previous_commit"] is None
    assert comparison["previous_golden_tag"] is None and cell["figures"] is True


def test_another_frame_fails_and_a_file_that_is_no_png_is_not_compared(tmp_path, monkeypatch):
    store = ImageStore()
    cell, _ = _image_run(tmp_path, monkeypatch, store, extra=("wf_00002_.png", "wf_clip.gif"))
    assert cell["verdict"] == "fail"
    assert (cell["vs_golden"]["frame_count_ref"], cell["vs_golden"]["frame_count_cand"]) == (1, 2)
    assert cell["vs_golden"]["mean_mse"] == 0.0 and cell["vs_golden"]["identical"] is False
    assert cell["output_sha256"] == "wf_00001_.png"
    assert "outputs/wf_clip.gif" in store.published  # published, never compared


@pytest.mark.parametrize("status, verdict", [("worker_error", "infra_error"),
                                             ("no_outputs", "infra_error"),
                                             ("timeout", "execution_error"),
                                             ("validation_error", "execution_error")])
def test_a_run_that_produced_nothing_is_not_compared(tmp_path, monkeypatch, status, verdict):
    assert status in (run_regression.INFRA_STATUSES | run_regression.EXEC_FAIL_STATUSES)
    store = ImageStore()
    cell, comparison = _image_run(tmp_path, monkeypatch, store, status=status)
    assert (cell["verdict"], cell["worker_status"]) == (verdict, status)
    assert comparison == {"verdict": verdict, "detail": None, "vs_golden": None,
                          "vs_previous": None, "thresholds_used": None, "golden_tag": None,
                          "previous_commit": None, "previous_verdict": None,
                          "previous_golden_tag": None, "figures": False}
    assert cell["output_sha256"] is None and cell["thumbnail"] is None
    assert store.published == ["comparison.json", "run.json"]
