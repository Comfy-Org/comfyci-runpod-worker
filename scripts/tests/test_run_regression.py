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


# -- check levels: a smoke-level workflow through a run -----------------------------
MIB = 1024 * 1024
EMPTY_SMOKE = {"files": [], "problems": [], "notes": []}
# A 3D workflow as the design has it: a picture of the mesh under the workflow
# id, and the mesh itself as a second file.
SMOKE_WORKFLOW = {
    "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "wf", "images": ["1", 0]}},
    "4": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "wf_mesh", "mesh": ["2", 0]}}}


def _fixture(name):
    from pathlib import Path
    return (Path(__file__).resolve().parent / "fixtures" / "smoke" / name).read_bytes()


def _picture(seed=1, side=96, height=64):
    """The bytes of a PNG that passes the smoke checks: noise, `side` px wide."""
    import io
    import numpy as np
    from PIL import Image
    pixels = np.random.default_rng(seed).integers(0, 256, (height, side, 3), dtype=np.uint8)
    data = io.BytesIO()
    Image.fromarray(pixels).save(data, "PNG")
    return data.getvalue()


def _sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _smoke_job(files, status="ok", unsent=()):
    """A RunPod job that wrote `files` ({name: bytes}). A name in `unsent` is
    listed and not written, like a file over the worker's size limit."""
    def job(client, wf_id, workflow, workdir=None, **kw):
        if status not in ("ok", "no_outputs"):
            return {"status": status, "error": {"message": "boom"}}
        out = workdir / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        listed = []
        for name, data in sorted(files.items()):
            rec = {"filename": name, "bytes": len(data), "sha256": _sha(data)}
            if name in unsent:
                rec["truncated"] = True
            else:
                (out / name).write_bytes(data)
            listed.append(rec)
        return {"status": status, "outputs_dir": str(out), "gpu_name": "gpu",
                "timings": {"prompt_exec_s": 3.0}, "outputs": listed}
    return job


class SmokeStore:
    """A results store for one smoke-level workflow. It records what is
    published, and every read of a reference or of an earlier run: a smoke run
    makes none."""
    READS = ("latest_pointer", "fetch_run_outputs", "run_summary", "golden_current",
             "fetch_golden_outputs", "noise_floor", "download_json", "published_summary")

    def __init__(self):
        self.reads, self.published, self.exclude = [], None, None

    def __getattr__(self, name):
        if name not in self.READS:
            raise AttributeError(name)
        return lambda *args: self.reads.append(name)

    def publish_workflow_run(self, branch, commit, wf_id, local_dir, exclude=()):
        self.exclude = list(exclude)
        names = (p.relative_to(local_dir).as_posix() for p in local_dir.rglob("*") if p.is_file())
        self.published = sorted(name for name in names if name not in exclude)


def _smoke_run(tmp_path, monkeypatch, files, status="ok", entry=None, workflow=None,
               previous_files=None, unsent=()):
    """process_workflow for a smoke-level workflow whose job wrote `files`;
    returns (cell, comparison.json, the store)."""
    import json
    import types
    store = SmokeStore()
    monkeypatch.setattr(run_regression, "run_workflow_job", _smoke_job(files, status, unsent))
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "wf_api.json").write_text(
        json.dumps(SMOKE_WORKFLOW if workflow is None else workflow), encoding="utf-8")
    args = types.SimpleNamespace(workdir=str(tmp_path / "work"), endpoint_id="ep", lane_id="main",
                                 commit="new", repo_url=None, branch="master", skip_publish=False)
    if previous_files is not None:
        # What main() read for the run: each smoke workflow's files of its previous result.
        args.previous_smoke = {"another": [{"name": "x", "sha256": "y"}], "wf": previous_files}
    entry = {"workflow_path": "wf_api.json", "models": [], "level": "smoke", **(entry or {})}
    cell = run_regression.process_workflow("wf", entry, DEFAULTS, args, tmp_path, store)
    comparison = json.loads((tmp_path / "work" / "wf" / "comparison.json").read_text())
    kind = [] if entry.get("output_kind") in (None, "image") else ["output_kind"]
    # Today's keys in today's order, then the kind when it is not image, then the two new ones.
    assert list(cell) == CELL_KEYS + kind + ["level", "smoke"]
    assert list(comparison) == COMPARISON_KEYS + kind + ["level", "smoke"]
    for key in COMPARISON_KEYS + ["level", "smoke"]:
        if key != "previous_golden_tag":
            assert cell[key] == comparison[key], key
    # Nothing is compared, so nothing was read: no reference and no earlier run.
    assert store.reads == []
    for key in ("detail", "vs_golden", "vs_previous", "thresholds_used", "golden_tag",
                "previous_commit", "previous_verdict"):
        assert cell[key] is None, key
    assert cell["figures"] is False and cell["level"] == "smoke"
    return cell, comparison, store


def test_a_smoke_workflow_that_wrote_usable_files_passes(tmp_path, monkeypatch):
    png, glb = _picture(), _fixture("mesh_300.glb")
    cell, comparison, store = _smoke_run(tmp_path, monkeypatch, {
        "wf_00001_.png": png, "wf_mesh_00001_.glb": glb})
    assert (cell["verdict"], cell["worker_status"]) == ("pass", "ok")
    assert cell["smoke"]["problems"] == [] and cell["smoke"]["notes"] == []
    picture, mesh = cell["smoke"]["files"]
    assert picture == {"name": "wf_00001_.png", "type": "image", "bytes": len(png),
                       "sha256": _sha(png), "width": 96, "height": 64,
                       "std": picture["std"], "flat_share": picture["flat_share"]}
    assert list(mesh) == ["name", "type", "bytes", "sha256", "vertices", "faces", "extent"]
    assert (mesh["name"], mesh["type"], mesh["vertices"]) == ("wf_mesh_00001_.glb", "mesh", 300)
    # The main output of an image workflow is its first PNG, as at reference level.
    assert cell["output_sha256"] == _sha(png) and cell["thumbnail"] == "wf_00001_.webp"
    assert store.exclude == [] and store.published == [
        "comparison.json", "outputs/thumbs/wf_00001_.webp", "outputs/wf_00001_.png",
        "outputs/wf_mesh_00001_.glb", "run.json"]


def test_a_smoke_workflow_with_a_missing_or_an_unusable_file_fails(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    passed, _, _ = _smoke_run(tmp_path / "ok", monkeypatch, {
        "wf_00001_.png": _picture(), "wf_mesh_00001_.glb": _fixture("mesh_300.glb")})
    # The mesh branch of the graph wrote nothing: ComfyUI ran the other one.
    cell, _, store = _smoke_run(tmp_path / "missing", monkeypatch, {"wf_00001_.png": _picture()})
    assert cell["verdict"] == "fail" and cell["smoke"]["problems"] == [
        {"file": None, "check": "missing",
         "text": "SaveGLB wrote no 3D file with the prefix wf_mesh"}]
    # A failing cell differs from a passing one in its verdict and its smoke object only.
    assert {k for k in cell if cell[k] != passed[k]} == {"verdict", "smoke"}
    assert cell["thumbnail"] == "wf_00001_.webp" and "outputs/wf_00001_.png" in store.published

    black = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(black, "PNG")
    cell, _, _ = _smoke_run(tmp_path / "blank", monkeypatch, {
        "wf_00001_.png": black.getvalue(), "wf_mesh_00001_.glb": _fixture("mesh_300.glb")})
    assert cell["verdict"] == "fail"
    assert [(p["file"], p["check"]) for p in cell["smoke"]["problems"]] == [
        ("wf_00001_.png", "blank")]
    # A cut file is a failure of the workflow, never a harness error.
    cut = _fixture("mesh_300.glb")
    cell, _, _ = _smoke_run(tmp_path / "cut", monkeypatch, {
        "wf_00001_.png": _picture(), "wf_mesh_00001_.glb": cut[:len(cut) // 2]})
    assert cell["verdict"] == "fail"
    assert [(p["file"], p["check"]) for p in cell["smoke"]["problems"]] == [
        ("wf_mesh_00001_.glb", "decode")]


def test_a_smoke_workflow_that_wrote_nothing_fails(tmp_path, monkeypatch):
    """`no_outputs` is an infrastructure status at reference level. At smoke
    level it is the failure the level exists to catch."""
    assert "no_outputs" in run_regression.INFRA_STATUSES
    cell, comparison, store = _smoke_run(tmp_path, monkeypatch, {}, status="no_outputs")
    assert (cell["verdict"], cell["worker_status"]) == ("fail", "no_outputs")
    assert cell["smoke"] == {"files": [], "notes": [], "problems": [
        {"file": None, "check": "missing", "text": "the workflow finished and wrote no file"}]}
    assert cell["output_sha256"] is None and cell["thumbnail"] is None
    assert store.published == ["comparison.json", "run.json"]


@pytest.mark.parametrize("status, verdict", [("timeout", "execution_error"),
                                             ("validation_error", "execution_error"),
                                             ("custom_node_error", "execution_error"),
                                             ("worker_error", "infra_error"),
                                             ("missing_model", "infra_error"),
                                             ("a_status_of_tomorrow", "infra_error")])
def test_a_smoke_workflow_that_did_not_finish_is_not_checked(tmp_path, monkeypatch, status,
                                                             verdict):
    cell, comparison, store = _smoke_run(tmp_path, monkeypatch, {"wf_00001_.png": _picture()},
                                         status=status)
    assert (cell["verdict"], cell["worker_status"]) == (verdict, status)
    assert comparison == {"verdict": verdict, "detail": None, "vs_golden": None,
                          "vs_previous": None, "thresholds_used": None, "golden_tag": None,
                          "previous_commit": None, "previous_verdict": None,
                          "previous_golden_tag": None, "figures": False, "level": "smoke",
                          "smoke": EMPTY_SMOKE}
    assert cell["output_sha256"] is None and cell["thumbnail"] is None
    assert store.published == ["comparison.json", "run.json"]


def test_the_verdict_of_a_smoke_run_by_worker_status():
    verdict = run_regression.smoke_verdict
    assert verdict("ok", []) == "pass"
    assert verdict("ok", [{"file": None, "check": "missing", "text": "x"}]) == "fail"
    assert verdict("no_outputs", []) == "fail"
    for status in sorted(run_regression.EXEC_FAIL_STATUSES):
        assert verdict(status, []) == "execution_error", status
    for status in sorted(run_regression.INFRA_STATUSES - {"no_outputs"}):
        assert verdict(status, []) == "infra_error", status
    # A status this does not know, or none at all, was not tested. It never passes.
    assert verdict(None, []) == verdict("harness_error", []) == "infra_error"


def test_an_audio_workflow_at_smoke_level_needs_no_comparator(tmp_path, monkeypatch):
    monkeypatch.setattr(run_regression, "comparator_for",
                        lambda entry: pytest.fail("a smoke workflow is not compared"))
    flac = _fixture("audio_1s.flac")
    workflow = {"9": {"class_type": "SaveAudioAdvanced",
                      "inputs": {"filename_prefix": "wf", "format": "flac", "audio": ["1", 0]}}}
    cell, comparison, store = _smoke_run(
        tmp_path / "pass", monkeypatch, {"wf_00001.flac": flac, "notes_00001.flac": b"x"},
        entry={"output_kind": "audio", "expect": {"seconds": 1}}, workflow=workflow)
    assert cell["verdict"] == "fail"  # the second file is no FLAC: every file is checked
    cell, comparison, store = _smoke_run(
        tmp_path / "one", monkeypatch, {"wf_00001.flac": flac},
        entry={"output_kind": "audio", "expect": {"seconds": 1}}, workflow=workflow)
    assert (cell["verdict"], cell["output_kind"], cell["thumbnail"]) == ("pass", "audio", None)
    # The main output of an audio workflow: its first FLAC named after the workflow.
    assert cell["output_sha256"] == _sha(flac)
    assert cell["smoke"]["files"][0]["seconds"] == 1.0
    assert store.published == ["comparison.json", "outputs/wf_00001.flac", "run.json"]
    # What the entry expects is held against the file.
    cell, _, _ = _smoke_run(tmp_path / "long", monkeypatch, {"wf_00001.flac": flac},
                            entry={"output_kind": "audio", "expect": {"seconds": 30}},
                            workflow=workflow)
    assert cell["verdict"] == "fail"
    assert [p["check"] for p in cell["smoke"]["problems"]] == ["expect"]


def test_a_text_workflow_at_smoke_level_is_checked_and_not_compared(tmp_path, monkeypatch):
    workflow = {"9": {"class_type": "SaveText",
                      "inputs": {"filename_prefix": "wf", "format": "txt", "text": ["1", 0]}}}
    said = "A short answer that is long enough to be a text.".encode()
    cell, comparison, store = _smoke_run(tmp_path / "said", monkeypatch, {"wf_00001.txt": said},
                                         entry={"output_kind": "text"}, workflow=workflow)
    assert (cell["verdict"], cell["output_kind"]) == ("pass", "text")
    assert cell["output_sha256"] == _sha(said) and cell["thumbnail"] is None
    cell, _, _ = _smoke_run(tmp_path / "empty", monkeypatch, {"wf_00001.txt": b""},
                            entry={"output_kind": "text"}, workflow=workflow)
    assert cell["verdict"] == "fail"
    assert [p["check"] for p in cell["smoke"]["problems"]] == ["empty"]


# -- check levels: which output files of a smoke run are published ------------------
def test_a_large_file_is_published_only_when_it_repeats(tmp_path, monkeypatch):
    png, small, large = _picture(), b"s" * (MIB // 2), b"L" * (2 * MIB)
    workflow = {**SMOKE_WORKFLOW,
                "5": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "wf_part"}}}
    files = {"wf_00001_.png": png, "wf_part_00001_.obj": small, "wf_mesh_00001_.obj": large}
    cell, comparison, store = _smoke_run(tmp_path / "first", monkeypatch, files,
                                         workflow=workflow)
    assert cell["verdict"] == "pass"
    held = cell["smoke"]["files"][1]
    # The key stands after what the worker reported, and only on a file that is held back.
    assert held == {"name": "wf_mesh_00001_.obj", "type": None, "bytes": 2 * MIB,
                    "sha256": _sha(large), "published": False}
    assert ["published" in f for f in cell["smoke"]["files"]] == [False, True, False]
    assert store.exclude == ["outputs/wf_mesh_00001_.obj"]
    # The picture, its thumbnail, the small file and the two records are always published.
    assert store.published == ["comparison.json", "outputs/thumbs/wf_00001_.webp",
                               "outputs/wf_00001_.png", "outputs/wf_part_00001_.obj", "run.json"]
    # The file stays on the runner, so the run's artifact has it.
    assert (tmp_path / "first" / "work" / "wf" / "outputs" / "wf_mesh_00001_.obj").stat(
        ).st_size == 2 * MIB

    # The next run wrote the same bytes: now the file is published.
    again, _, store = _smoke_run(tmp_path / "second", monkeypatch, files, workflow=workflow,
                                 previous_files=cell["smoke"]["files"])
    assert not [f for f in again["smoke"]["files"] if "published" in f]
    assert store.exclude == [] and "outputs/wf_mesh_00001_.obj" in store.published
    # A run that wrote other bytes holds it back again, whatever the run before published.
    other = {**files, "wf_mesh_00001_.obj": b"M" * (2 * MIB)}
    moved, _, store = _smoke_run(tmp_path / "third", monkeypatch, other, workflow=workflow,
                                 previous_files=again["smoke"]["files"])
    assert moved["smoke"]["files"][1]["published"] is False
    assert store.exclude == ["outputs/wf_mesh_00001_.obj"]
    # The limit itself is published: a file of 1 MiB, on its first run.
    edge, _, store = _smoke_run(tmp_path / "limit", monkeypatch,
                                {**files, "wf_mesh_00001_.obj": b"L" * MIB}, workflow=workflow)
    assert not [f for f in edge["smoke"]["files"] if "published" in f] and store.exclude == []


def test_a_file_the_worker_did_not_send_is_never_published(tmp_path, monkeypatch):
    png, clip = _picture(), b"c" * 4096
    workflow = {"3": SMOKE_WORKFLOW["3"],
                "4": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "wf_video"}}}
    files = {"wf_00001_.png": png, "wf_video_00001_.mp4": clip}
    cell, _, store = _smoke_run(tmp_path / "a", monkeypatch, files, workflow=workflow,
                                unsent=("wf_video_00001_.mp4",))
    assert cell["verdict"] == "pass"  # on its size alone, and the cell says so
    assert cell["smoke"]["notes"] == ["wf_video_00001_.mp4 was not sent (over the worker's size "
                                      "limit): checked by size only"]
    assert cell["smoke"]["files"][1] == {"name": "wf_video_00001_.mp4", "type": "video",
                                         "bytes": 4096, "sha256": _sha(clip), "sent": False,
                                         "published": False}
    assert list(cell["smoke"]["files"][1])[-2:] == ["sent", "published"]
    assert store.exclude == ["outputs/wf_video_00001_.mp4"]
    assert "outputs/wf_video_00001_.mp4" not in store.published
    # Also when it repeats: there is no file to publish.
    again, _, _ = _smoke_run(tmp_path / "b", monkeypatch, files, workflow=workflow,
                             unsent=("wf_video_00001_.mp4",),
                             previous_files=cell["smoke"]["files"])
    assert again["smoke"]["files"][1]["published"] is False


def test_a_file_that_is_held_back_is_still_checked_and_says_so_before_its_numbers(tmp_path,
                                                                                 monkeypatch):
    """A second picture of 1.2 MB, as a 3D workflow writes it: checked like the
    first, and held back until it repeats. Every reader finds `published`
    where the worker's own facts end."""
    small, large = _picture(1), _picture(2, side=640, height=640)
    assert len(large) > MIB
    cell, _, store = _smoke_run(tmp_path, monkeypatch, {"wf_00001_.png": small,
                                                        "wf_00002_.png": large},
                                workflow={"3": SMOKE_WORKFLOW["3"], "4": SMOKE_WORKFLOW["3"]})
    assert cell["verdict"] == "pass" and store.exclude == ["outputs/wf_00002_.png"]
    first, second = cell["smoke"]["files"]
    assert list(first) == ["name", "type", "bytes", "sha256", "width", "height", "std",
                           "flat_share"]
    assert list(second) == ["name", "type", "bytes", "sha256", "published", "width", "height",
                            "std", "flat_share"]
    assert (second["published"], second["width"], second["height"]) == (False, 640, 640)
    # Its preview is small and is published, like the picture it was made from is not.
    assert "outputs/thumbs/wf_00002_.webp" in store.published
    assert "outputs/wf_00002_.png" not in store.published
    # The main output is still the first picture.
    assert cell["output_sha256"] == _sha(small) and cell["thumbnail"] == "wf_00001_.webp"


def test_the_publish_rule_reaches_files_the_cell_does_not_list(tmp_path, monkeypatch):
    """`smoke.files` holds 50 entries. A large file after them is held back too."""
    import smoke
    files = {f"wf_{i:05d}_.bin": b"x" for i in range(1, smoke.FILES_MAX + 1)}
    files["wf_99999_.bin"] = b"L" * (MIB + 1)
    cell, _, store = _smoke_run(tmp_path, monkeypatch, files, workflow={})
    assert len(cell["smoke"]["files"]) == smoke.FILES_MAX and cell["verdict"] == "pass"
    assert store.exclude == ["outputs/wf_99999_.bin"]
    assert run_regression.held_back({"outputs": "none"}, None) == set()


def test_previous_smoke_files_reads_each_earlier_summary_once():
    files_a = [{"name": "a_00001_.png", "bytes": 9, "sha256": "a1"}]
    files_b = [{"name": "b_00001_.glb", "bytes": 9, "sha256": "b1", "published": False}]
    head = [{"c": "c2", "t": 2, "m": {"ct": 2},
             "w": {"a": {"v": "pass"}, "b": {"v": "fail"}, "c": {"v": "infra_error"},
                   "e": {"v": "pass"}}},
            {"c": "c1", "t": 1, "m": {"ct": 1}, "w": {"c": {"v": "pass"}, "d": {"v": "pass"}}}]
    summaries = {
        "c2": {"workflows": {"a": {"smoke": {"files": files_a}}, "b": {"smoke": {"files": files_b}},
                             "e": {"verdict": "pass"}}},
        "c1": {"workflows": {"c": {"smoke": {"files": []}}}}}
    reads = []

    class Store:
        def published_summary(self, branch, commit):
            reads.append((branch, commit))
            return summaries.get(commit)

    found = run_regression.previous_smoke_files(Store(), "master", head, "c3",
                                                ["a", "b", "c", "d", "e", "new"])
    # c: its newest red or green result is at c1. d: no smoke result there. e: it was at
    # reference level then. new: no result at all.
    assert found == {"a": files_a, "b": files_b, "c": []}
    assert sorted(reads) == [("master", "c1"), ("master", "c2")]
    # A re-run of c2 itself reads the results before it, never its own.
    reads.clear()
    assert run_regression.previous_smoke_files(Store(), "master", head, "c2", ["a", "c"]) == {
        "c": []}
    assert reads == [("master", "c1")]
    # A run without smoke workflows reads nothing.
    reads.clear()
    assert run_regression.previous_smoke_files(Store(), "master", head, "c3", []) == {}
    assert reads == []


def test_published_new_bytes_counts_what_the_previous_result_had_not_published():
    def cell(wf_id, *files):
        return {"workflow_id": wf_id, "smoke": {"files": list(files)}}

    def one(name, size, sha, **more):
        return {"name": name, "type": None, "bytes": size, "sha256": sha, **more}

    count = run_regression.published_new_bytes
    first = [cell("a", one("x.png", 100, "p1"), one("x.glb", 5000, "g1", published=False)),
             cell("b", one("y.flac", 30, "f1")),
             {"workflow_id": "ref", "verdict": "pass"}]  # a reference cell lists no files
    assert count(first, {}) == 130
    before = {"a": first[0]["smoke"]["files"], "b": first[1]["smoke"]["files"]}
    # The same bytes again: the large file is published now, and nothing else is new.
    second = [cell("a", one("x.png", 100, "p1"), one("x.glb", 5000, "g1")),
              cell("b", one("y.flac", 30, "f1"))]
    assert count(second, before) == 5000
    assert count(second, {"a": second[0]["smoke"]["files"], "b": before["b"]}) == 0
    # A changed file is new, a file that was not sent is not there, a size that is no
    # number adds nothing.
    third = [cell("a", one("x.png", 120, "p2"), one("x.glb", 5000, "g1", sent=False,
                                                    published=False)),
             cell("b", one("y.flac", None, "f2"), one("z.txt", 7, None))]
    assert count(third, before) == 127


# -- check levels: through main() ----------------------------------------------------
LEVEL_NOW = 1794700000
LEVEL_LANES = {"schema_version": 2, "primary": "main", "lanes": {
    "main": {"cadence": "per-commit", "role": "primary", "layout": 2,
             "prefix": "regression/lanes/main", "endpoint_id": "ep-main"},
    "nightly": {"cadence": "nightly", "role": "lane", "layout": 2,
                "prefix": "regression/lanes/nightly", "endpoint_id": "ep-nightly"}}}
RUN_SUMMARY_KEYS = ["schema_version", "branch", "commit", "run_ts", "overall", "has_infra_error",
                    "lane", "attempt", "run_url", "run_id", "env", "commit_meta", "tested_range",
                    "workflows"]
INDEX_CELL_KEYS = ["v", "d", "sha", "mse", "psnr", "pct", "pv", "exec", "vram", "g", "th"]


def _shade(shade):
    """The bytes of a small PNG of one shade, as a reference-level run compares it."""
    import io
    import numpy as np
    from PIL import Image
    data = io.BytesIO()
    Image.fromarray(np.full((16, 24, 3), shade, dtype=np.uint8)).save(data, "PNG")
    return data.getvalue()


class FolderStore(run_regression.storage.Storage):
    """A results store in a folder: what a run publishes, without git."""

    def __init__(self, root, prefix):
        super().__init__(prefix)
        self.dir, self.summary_reads = root, []

    def download_json(self, path):
        import json
        f = self.dir / path
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def read_blob_json(self, path):
        if path.endswith("/summary.json"):
            self.summary_reads.append(path.split("/")[-2])
        return self.download_json(path)

    def upload_json(self, path, obj, compact=False):
        import json
        f = self.dir / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def upload_dir(self, blob_prefix, local_dir, exclude=()):
        import shutil
        for p in sorted(local_dir.rglob("*")):
            rel = p.relative_to(local_dir).as_posix()
            if p.is_file() and rel not in exclude:
                dest = self.dir / blob_prefix / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

    def download_dir(self, blob_prefix, local_dir):
        import shutil
        src = self.dir / blob_prefix
        files = [p for p in sorted(src.rglob("*")) if p.is_file()] if src.is_dir() else []
        for p in files:
            dest = local_dir / p.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        return len(files)


class LevelLane:
    """main() on a registry of two lanes, the per-commit lane `main` and the
    lane `nightly`, with a results store in a folder and a GPU that writes for
    each workflow the files it holds at that moment (`files`), or ends with
    its `status`. `entries` are the manifest entries, `graphs` their workflows."""

    def __init__(self, tmp_path, monkeypatch, entries, graphs=None):
        import json
        import commit_meta
        self.tmp, self.monkeypatch, self.runs, self.last = tmp_path, monkeypatch, 0, None
        self.files = {wf_id: {} for wf_id in entries}
        self.status = dict.fromkeys(entries, "ok")
        repo, manifest = tmp_path / "repo", {"defaults": DEFAULTS, "workflows": {}}
        (repo / "manifest" / "workflows").mkdir(parents=True)
        for wf_id, entry in entries.items():
            path = f"manifest/workflows/{wf_id}_api.json"
            (repo / path).write_text(json.dumps((graphs or {}).get(wf_id, {})), encoding="utf-8")
            manifest["workflows"][wf_id] = {"enabled": True, "workflow_path": path,
                                            "models": [], **entry}
        self.manifest = repo / "manifest" / "workflows.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: LEVEL_LANES)
        monkeypatch.setattr(run_regression.storage, "from_args", self.store)
        monkeypatch.setattr(run_regression, "run_workflow_job", self.job)
        monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
        # Commit c1 was committed at 10, c2 at 20, and so on.
        monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: {
            "subject": "change (#9)", "pr": 9, "committed_ts": int(sha[1:]) * 10})
        monkeypatch.setattr(commit_meta, "fetch_range", lambda prev, sha: {
            "prev": prev, "commits_between": 1, "compare_url": f"{prev}...{sha}"})
        for key in ("GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_SERVER_URL"):
            monkeypatch.delenv(key, raising=False)

    def store(self, args):
        self.last = FolderStore(self.tmp / "store", args.prefix)
        return self.last

    def job(self, client, wf_id, workflow, workdir=None, **kw):
        if self.status[wf_id] == "harness":
            raise RuntimeError("the runner lost the job")
        return _smoke_job(self.files[wf_id], self.status[wf_id])(client, wf_id, workflow,
                                                                 workdir=workdir)

    def reference(self, lane, wf_id, data):
        """Approve `data` as the PNG reference of a workflow on a lane."""
        base = self.tmp / "store" / LEVEL_LANES["lanes"][lane]["prefix"] / "golden" / wf_id
        (base / "v1" / "outputs").mkdir(parents=True)
        (base / "current.json").write_text('{"tag": "v1"}', encoding="utf-8")
        (base / "v1" / "outputs" / f"{wf_id}_00001_.png").write_bytes(data)

    def run(self, commit, lane, *argv, now=LEVEL_NOW):
        """main() once: (exit code, the summary it wrote for the steps after it)."""
        import json
        import sys
        import types
        self.runs += 1
        out = self.tmp / f"out{self.runs}.json"
        self.monkeypatch.setattr(run_regression, "time", types.SimpleNamespace(time=lambda: now))
        self.monkeypatch.setattr(sys, "argv", [
            "run_regression.py", "--manifest", str(self.manifest), "--commit", commit,
            "--branch", "master", "--lane", lane, "--workdir", str(self.tmp / f"w{self.runs}"),
            "--summary-out", str(out), *argv])
        code = run_regression.main()
        return code, json.loads(out.read_text(encoding="utf-8"))

    def text(self, path):
        """The text of a published file (None: there is none)."""
        f = self.tmp / "store" / path
        return f.read_text(encoding="utf-8") if f.is_file() else None

    def json(self, path):
        import json
        return json.loads(self.text(path))


def test_an_image_lane_through_main_publishes_what_it_did_before_check_levels(tmp_path,
                                                                             monkeypatch):
    """What the pin of image behaviour cannot see, because it does not run
    main(): the previous-run pointer and the shape of summary.json after a
    full run of image workflows on a per-commit lane."""
    import json
    lane = LevelLane(tmp_path, monkeypatch, {"wf_a": {}, "wf_b": {}, "wf_c": {}})
    for wf_id in lane.files:
        lane.files[wf_id] = {f"{wf_id}_00001_.png": _shade(100)}
        lane.reference("main", wf_id, _shade(100))
    prefix = "regression/lanes/main"
    code, summary = lane.run("c1", "main")
    assert code == 0
    pointer = lane.text(f"{prefix}/latest/master.json")
    assert pointer == ('{\n  "commit": "c1",\n  "run_ts": 1794700000,\n  "workflows": [\n'
                       '    "wf_a",\n    "wf_b",\n    "wf_c"\n  ]\n}')
    text = lane.text(f"{prefix}/runs/master/c1/summary.json")
    assert text == json.dumps(summary, indent=2), "--summary-out is the published file"
    assert list(summary) == RUN_SUMMARY_KEYS and list(summary["workflows"]) == list(lane.files)
    assert {k: summary[k] for k in RUN_SUMMARY_KEYS[:10]} == {
        "schema_version": 2, "branch": "master", "commit": "c1", "run_ts": LEVEL_NOW,
        "overall": "pass", "has_infra_error": False, "lane": "main", "attempt": 1,
        "run_url": None, "run_id": None}
    assert summary["tested_range"] is None and summary["commit_meta"]["pr"] == 9
    for wf_id, cell in summary["workflows"].items():
        assert list(cell) == CELL_KEYS + ["prior_verdict", "prior_commit"], wf_id
        assert (cell["verdict"], cell["prior_verdict"], cell["prior_commit"]) == ("pass", None,
                                                                                 None)
    # The index: no cell names a level, every reference is listed, no chain is open.
    head = lane.json("regression/index/master/main.json")
    assert [list(cell) for cell in head["entries"][0]["w"].values()] == [INDEX_CELL_KEYS] * 3
    assert "wr" not in head["entries"][0] and head["first_bad"] == {}
    assert sorted(lane.json("regression/index/lanes.json")["lanes"]["main"]["goldens"]) == [
        "wf_a", "wf_b", "wf_c"]

    # A full run with an execution error is a result too. It does not move the pointer.
    lane.status["wf_c"] = "timeout"
    code, summary = lane.run("c2", "main", now=LEVEL_NOW + 600)
    assert code == 1 and list(summary) == RUN_SUMMARY_KEYS
    assert summary["tested_range"] == {"prev": "c1", "commits_between": 1,
                                       "compare_url": "c1...c2"}
    assert lane.text(f"{prefix}/latest/master.json") == pointer
    assert lane.json("regression/index/master/main.json")["first_bad"]["wf_c"] == {
        "commit": "c2", "run_ts": LEVEL_NOW + 600, "kind": "execution_error", "prev_good": "c1",
        "pr": 9, "author": None, "golden": None, "runs": 1}
    # A run that names its workflows is not the lane's run of the commit: it
    # leaves the pointer alone, also when everything it ran can be compared.
    code, summary = lane.run("c3", "main", "--workflows", "wf_a,wf_b", now=LEVEL_NOW + 1200)
    assert code == 0 and [c["verdict"] for c in summary["workflows"].values()] == ["pass", "pass"]
    assert lane.text(f"{prefix}/latest/master.json") == pointer
    # The next full run that can be compared moves it.
    lane.status["wf_c"] = "ok"
    assert lane.run("c4", "main", now=LEVEL_NOW + 1800)[0] == 0
    assert lane.json(f"{prefix}/latest/master.json") == {
        "commit": "c4", "run_ts": LEVEL_NOW + 1800, "workflows": ["wf_a", "wf_b", "wf_c"]}


THREE_D = {"3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "wf_3d"}},
           "4": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "wf_3d_mesh"}},
           "5": {"class_type": "SaveGLB", "inputs": {"filename_prefix": "wf_3d_part"}}}
SOUND = {"9": {"class_type": "SaveAudioAdvanced",
               "inputs": {"filename_prefix": "wf_audio", "format": "flac"}}}


def test_a_nightly_lane_publishes_a_large_smoke_output_from_its_second_run_on(tmp_path,
                                                                             monkeypatch):
    monkeypatch.setattr(run_regression, "comparator_for",
                        lambda entry: pytest.fail("a smoke workflow is not compared"))
    lane = LevelLane(tmp_path, monkeypatch, {
        "wf_3d": {"level": "smoke", "cadence": "nightly", "expect": {"width": 96}},
        "wf_audio": {"level": "smoke", "cadence": "nightly", "output_kind": "audio"}},
        {"wf_3d": THREE_D, "wf_audio": SOUND})
    png, part, mesh, flac = _picture(), b"s" * (MIB // 2), b"L" * (2 * MIB), _fixture(
        "audio_1s.flac")
    lane.files["wf_3d"] = {"wf_3d_00001_.png": png, "wf_3d_part_00001_.obj": part,
                           "wf_3d_mesh_00001_.obj": mesh}
    lane.files["wf_audio"] = {"wf_audio_00001.flac": flac}
    prefix = "regression/lanes/nightly"

    def published(commit):
        root = tmp_path / "store" / prefix / "runs" / "master" / commit
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    code, summary = lane.run("c1", "nightly")
    assert code == 0 and list(summary) == RUN_SUMMARY_KEYS + ["published_new_bytes"]
    assert {wf: cell["verdict"] for wf, cell in summary["workflows"].items()} == {
        "wf_3d": "pass", "wf_audio": "pass"}
    cell = summary["workflows"]["wf_3d"]
    assert list(cell) == CELL_KEYS + ["level", "smoke", "prior_verdict", "prior_commit"]
    assert list(summary["workflows"]["wf_audio"]) == CELL_KEYS + [
        "output_kind", "level", "smoke", "prior_verdict", "prior_commit"]
    assert [(f["name"], f.get("published")) for f in cell["smoke"]["files"]] == [
        ("wf_3d_00001_.png", None), ("wf_3d_mesh_00001_.obj", False),
        ("wf_3d_part_00001_.obj", None)]
    # The first run: everything but the 2 MiB file, which stays on the runner.
    assert published("c1") == [
        "attempt.json", "summary.json", "wf_3d/comparison.json",
        "wf_3d/outputs/thumbs/wf_3d_00001_.webp", "wf_3d/outputs/wf_3d_00001_.png",
        "wf_3d/outputs/wf_3d_part_00001_.obj", "wf_3d/run.json", "wf_audio/comparison.json",
        "wf_audio/outputs/wf_audio_00001.flac", "wf_audio/run.json"]
    assert (tmp_path / "w1" / "wf_3d" / "outputs" / "wf_3d_mesh_00001_.obj").is_file()
    assert summary["published_new_bytes"] == len(png) + len(part) + len(flac)
    assert lane.json(f"{prefix}/runs/master/c1/wf_3d/comparison.json")["smoke"] == cell["smoke"]
    assert lane.last.summary_reads == ["c1"]  # its own attempts: there is no earlier result

    # The second run wrote the same bytes: now the large file is published too.
    code, summary = lane.run("c2", "nightly", now=LEVEL_NOW + 86400)
    assert code == 0
    assert not [f for f in summary["workflows"]["wf_3d"]["smoke"]["files"] if "published" in f]
    assert "wf_3d/outputs/wf_3d_mesh_00001_.obj" in published("c2")
    assert summary["published_new_bytes"] == len(mesh)
    # Both workflows had their previous result at c1: its summary was read once.
    assert lane.last.summary_reads == ["c2", "c1"]
    # A third run of the same bytes adds nothing.
    code, summary = lane.run("c3", "nightly", now=LEVEL_NOW + 2 * 86400)
    assert summary["published_new_bytes"] == 0 and len(published("c3")) == len(published("c2"))
    # Other bytes are held back again, and a changed small file counts as new.
    lane.files["wf_3d"]["wf_3d_mesh_00001_.obj"] = b"M" * (2 * MIB)
    lane.files["wf_3d"]["wf_3d_part_00001_.obj"] = b"t" * 1000
    code, summary = lane.run("c4", "nightly", now=LEVEL_NOW + 3 * 86400)
    assert "wf_3d/outputs/wf_3d_mesh_00001_.obj" not in published("c4")
    assert summary["published_new_bytes"] == 1000

    # The index: every cell names its level, and the lane lists no reference.
    head = lane.json("regression/index/master/nightly.json")
    cells = head["entries"][0]["w"]
    assert list(cells["wf_3d"]) == INDEX_CELL_KEYS + ["l"]
    assert list(cells["wf_audio"]) == INDEX_CELL_KEYS + ["k", "l"]
    assert (cells["wf_audio"]["k"], cells["wf_audio"]["l"], cells["wf_audio"]["th"]) == (
        "audio", "smoke", None)
    assert cells["wf_3d"] == {"v": "pass", "d": None, "sha": _sha(png), "mse": None,
                              "psnr": None, "pct": None, "pv": None, "exec": 3.0, "vram": None,
                              "g": None, "th": "wf_3d_00001_.webp", "l": "smoke"}
    assert lane.json("regression/index/lanes.json")["lanes"]["nightly"]["goldens"] == {}
    assert lane.json(f"{prefix}/latest/master.json")["commit"] == "c4"


def test_a_full_run_of_a_nightly_lane_always_moves_the_pointer(tmp_path, monkeypatch):
    """Among 30 to 90 workflows one often has an error. With the rule of a
    per-commit lane the pointer of such a lane would stand still."""
    lane = LevelLane(tmp_path, monkeypatch, {
        "wf_a": {"level": "smoke", "cadence": "nightly"},
        "wf_b": {"level": "smoke", "cadence": "nightly"}})
    lane.files["wf_a"], lane.files["wf_b"] = {"wf_a_00001_.png": _picture(1)}, {
        "wf_b_00001_.png": _picture(2)}
    latest = "regression/lanes/nightly/latest/master.json"
    assert lane.run("c1", "nightly")[0] == 0
    assert lane.json(latest)["commit"] == "c1"
    lane.status["wf_b"] = "timeout"
    code, summary = lane.run("c2", "nightly", now=LEVEL_NOW + 86400)
    assert code == 1 and summary["workflows"]["wf_b"]["verdict"] == "execution_error"
    assert lane.json(latest) == {"commit": "c2", "run_ts": LEVEL_NOW + 86400,
                                 "workflows": ["wf_a", "wf_b"]}
    # Not tested at all is no reason either.
    lane.status["wf_b"] = "worker_error"
    code, summary = lane.run("c3", "nightly", now=LEVEL_NOW + 2 * 86400)
    assert code == 0 and summary["has_infra_error"] is True
    assert lane.json(latest)["commit"] == "c3"
    # A run that names its workflows never moves it, on this lane as on any other.
    lane.status["wf_b"] = "ok"
    code, summary = lane.run("c4", "nightly", "--workflows", "wf_b", now=LEVEL_NOW + 3 * 86400)
    assert code == 0 and lane.json(latest)["commit"] == "c3"


def test_a_smoke_workflow_on_a_per_commit_lane_from_its_first_failure_to_its_fix(tmp_path,
                                                                                monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    lane = LevelLane(tmp_path, monkeypatch, {"wf_img": {}, "wf_smoke": {"level": "smoke"}},
                     {"wf_smoke": {"3": {"class_type": "SaveImage",
                                         "inputs": {"filename_prefix": "wf_smoke"}}}})
    good, black = _picture(), io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(black, "PNG")
    lane.files["wf_img"] = {"wf_img_00001_.png": _shade(100)}
    lane.files["wf_smoke"] = {"wf_smoke_00001_.png": good}
    lane.reference("main", "wf_img", _shade(100))
    latest, index = "regression/lanes/main/latest/master.json", "regression/index/master/main.json"

    code, summary = lane.run("c1", "main")
    # A per-commit lane keeps the summary keys it always had, and its image cell its keys.
    assert code == 0 and list(summary) == RUN_SUMMARY_KEYS
    assert list(summary["workflows"]["wf_img"]) == CELL_KEYS + ["prior_verdict", "prior_commit"]
    assert list(summary["workflows"]["wf_smoke"]) == CELL_KEYS + [
        "level", "smoke", "prior_verdict", "prior_commit"]
    assert lane.json(latest)["commit"] == "c1"
    cells = lane.json(index)["entries"][0]["w"]
    assert list(cells["wf_img"]) == INDEX_CELL_KEYS
    assert list(cells["wf_smoke"]) == INDEX_CELL_KEYS + ["l"]

    # The workflow did not finish: an execution error, and the pointer stays.
    lane.status["wf_smoke"] = "timeout"
    assert lane.run("c2", "main", now=LEVEL_NOW + 600)[0] == 1
    assert lane.json(latest)["commit"] == "c1"
    # It finished with a blank picture: a failure that can be compared with nothing,
    # yet every workflow produced its outputs, so the pointer moves.
    lane.status["wf_smoke"] = "ok"
    lane.files["wf_smoke"] = {"wf_smoke_00001_.png": black.getvalue()}
    code, summary = lane.run("c3", "main", now=LEVEL_NOW + 1200)
    cell = summary["workflows"]["wf_smoke"]
    assert code == 1 and (cell["verdict"], cell["prior_verdict"]) == ("fail", "execution_error")
    assert [p["check"] for p in cell["smoke"]["problems"]] == ["blank"]
    assert lane.json(latest)["commit"] == "c3"
    head = lane.json(index)
    assert head["entries"][0]["w"]["wf_smoke"]["d"] is None
    # The chain is the one the execution error opened: it does not start again.
    assert head["first_bad"] == {"wf_smoke": {
        "commit": "c2", "run_ts": LEVEL_NOW + 600, "kind": "execution_error", "prev_good": "c1",
        "pr": 9, "author": None, "golden": None, "runs": 2, "level": "smoke"}}
    # The same failure again is a standing one.
    code, summary = lane.run("c4", "main", now=LEVEL_NOW + 1800)
    assert summary["workflows"]["wf_smoke"]["prior_verdict"] == "fail"
    head = lane.json(index)
    assert head["entries"][0]["w"]["wf_smoke"]["d"] == "inherited"
    assert head["first_bad"]["wf_smoke"]["runs"] == 3
    # Fixed: the chain closes.
    lane.files["wf_smoke"] = {"wf_smoke_00001_.png": good}
    assert lane.run("c5", "main", now=LEVEL_NOW + 2400)[0] == 0
    assert lane.json(index)["first_bad"] == {}
    assert lane.json("regression/index/lanes.json")["lanes"]["main"]["goldens"] == {
        "wf_img": {"tag": "v1", "sha": None}}


def test_the_index_lists_a_chain_only_for_a_workflow_the_lane_still_runs(tmp_path, monkeypatch):
    import json
    lane = LevelLane(tmp_path, monkeypatch, {"wf_a": {"level": "smoke"},
                                             "wf_b": {"level": "smoke"}})
    lane.files = {wf_id: {f"{wf_id}_00001_.png": _picture()} for wf_id in lane.files}
    index = "regression/index/master/main.json"
    assert lane.run("c1", "main")[0] == 0
    lane.status["wf_b"] = "timeout"
    assert lane.run("c2", "main", now=LEVEL_NOW + 600)[0] == 1
    assert sorted(lane.json(index)["first_bad"]) == ["wf_b"]
    # A run that names another workflow closes nothing: the lane still runs wf_b.
    assert lane.run("c3", "main", "--workflows", "wf_a", now=LEVEL_NOW + 1200)[0] == 0
    assert sorted(lane.json(index)["first_bad"]) == ["wf_b"]
    # Switched off while it was red: no later run could close its chain.
    manifest = json.loads(lane.manifest.read_text(encoding="utf-8"))
    manifest["workflows"]["wf_b"]["enabled"] = False
    lane.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    assert lane.run("c4", "main", now=LEVEL_NOW + 1800)[0] == 0
    head = lane.json(index)
    assert head["first_bad"] == {}
    # Its history stays where it is.
    assert [sorted(e["w"]) for e in head["entries"]] == [["wf_a"], ["wf_a"], ["wf_a", "wf_b"],
                                                        ["wf_a", "wf_b"]]


def test_a_harness_error_cell_of_a_smoke_workflow_names_its_level(tmp_path, monkeypatch):
    lane = LevelLane(tmp_path, monkeypatch, {
        "wf_img": {}, "wf_smoke": {"level": "smoke"},
        "wf_night": {"level": "smoke", "cadence": "nightly"}})
    lane.status = dict.fromkeys(lane.status, "harness")
    code, summary = lane.run("c1", "main", "--skip-publish")
    assert code == 0 and summary["has_infra_error"] is True
    error = "RuntimeError('the runner lost the job')"
    assert summary["workflows"] == {
        "wf_img": {"workflow_id": "wf_img", "worker_status": "harness_error",
                   "verdict": "infra_error", "error": error, "prior_verdict": None,
                   "prior_commit": None},
        "wf_smoke": {"workflow_id": "wf_smoke", "worker_status": "harness_error",
                     "verdict": "infra_error", "error": error, "level": "smoke",
                     "prior_verdict": None, "prior_commit": None}}
    # Publishing is off: nothing was written, and a nightly lane names no bytes either.
    assert not (tmp_path / "store").exists()
    night = lane.run("c1", "nightly", "--skip-publish")[1]
    assert list(night["workflows"]) == ["wf_night"] and "published_new_bytes" not in night


def test_a_lane_runs_the_workflows_of_its_own_cadence(tmp_path, monkeypatch, capsys):
    """`all` on a lane takes the workflows of the lane's cadence. A workflow of
    another cadence that a dispatch names is skipped with a line that says
    why: its `lanes` is null, and joining that for the line used to be a
    TypeError. The index keeps open failures for the lane's own workflows."""
    import sys
    lane = LevelLane(tmp_path, monkeypatch, {
        "wf_img": {}, "wf_pack": {"lanes": ["main"]},
        "wf_night": {"level": "smoke", "cadence": "nightly"},
        "wf_week": {"level": "smoke", "cadence": "weekly"},
        "wf_off": {"enabled": False, "cadence": "nightly"}})
    lane.status = dict.fromkeys(lane.status, "harness")
    builder, seen = index_builder.IndexBuilder, []

    def recording(*args, **kw):
        seen.append(kw.get("active"))
        return builder(*args, **kw)
    monkeypatch.setattr(run_regression.index_builder, "IndexBuilder", recording)

    code, summary = lane.run("c1", "main", "--skip-publish")
    assert code == 0 and list(summary["workflows"]) == ["wf_img", "wf_pack"]
    out = capsys.readouterr().out
    assert "[wf_night] not run on lane main (cadence nightly, the lane is per-commit)\n" in out
    assert "[wf_week] not run on lane main (cadence weekly, the lane is per-commit)\n" in out
    assert "wf_off" not in out
    code, summary = lane.run("c1", "nightly", "--skip-publish")
    assert list(summary["workflows"]) == ["wf_night"]
    out = capsys.readouterr().out
    assert "[wf_img] not run on lane nightly (cadence per-commit, the lane is nightly)\n" in out
    # The `lanes` list is said first, as it always was.
    assert "[wf_pack] not run on lane nightly (manifest lanes: main)\n" in out
    assert "[wf_week] not run on lane nightly (cadence weekly, the lane is nightly)\n" in out
    assert seen == [["wf_img", "wf_pack"], ["wf_night"]]

    # Named in a dispatch on a lane of another cadence: skipped, and nothing is left to do.
    for named, lane_id, why in (
            ("wf_night", "main", "cadence nightly, the lane is per-commit"),
            ("wf_off", "main", "cadence nightly, the lane is per-commit"),
            ("wf_img", "nightly", "cadence per-commit, the lane is nightly")):
        monkeypatch.setattr(sys, "argv", [
            "run_regression.py", "--manifest", str(lane.manifest), "--commit", "c2",
            "--branch", "master", "--lane", lane_id, "--workflows", named, "--skip-publish"])
        assert run_regression.main() == 0
        assert capsys.readouterr().out == (
            f"[{named}] not run on lane {lane_id} ({why})\n"
            f"no workflows selected for lane {lane_id}; nothing to do\n")
    # A workflow that is switched off runs when a dispatch names it on a lane of its cadence.
    code, summary = lane.run("c2", "nightly", "--workflows", "wf_off", "--skip-publish")
    assert list(summary["workflows"]) == ["wf_off"]


@pytest.mark.parametrize("entry, said", [
    ({"level": "main"}, r"\[wf\] unknown level 'main' \(known: reference, smoke\); see "),
    ({"level": 1}, r"\[wf\] unknown level 1 \(known: reference, smoke\); see "),
    ({"cadence": "daily"},
     r"\[wf\] unknown cadence 'daily' \(known: per-commit, nightly, weekly\); see "),
    ({"expect": {"width": 64}}, r"\[wf\] expect is for a smoke-level workflow only; see "),
    ({"level": "smoke", "expect": {"width": 0}},
     r"\[wf\] expect.width must be a number above 0, not 0; see "),
    ({"level": "smoke", "expect": {"size": 3}}, r"\[wf\] unknown expect key 'size' \(known: "),
    ({"level": "smoke", "output_kind": "video"}, r"\[wf\] unknown output_kind 'video' \(known: "),
    # Audio has no comparator yet: a reference-level entry of that kind cannot run.
    ({"output_kind": "audio"},
     r"\[wf\] no comparator for output_kind 'audio' yet: such a workflow can only run with "
     r'"level": "smoke"; see manifest/workflows.json'),
    ({"level": "reference", "output_kind": "audio"}, r"\[wf\] no comparator for output_kind"),
])
def test_an_entry_that_cannot_be_run_stops_the_run_before_its_start_marker(tmp_path, monkeypatch,
                                                                           entry, said):
    """A clean message, no traceback, no GPU job, and nothing written: a run
    that left a start marker would count as an attempt at the commit."""
    import json
    import sys
    import storage

    class ReadOnlyStore(FakeStore):
        def __getattr__(self, name):
            pytest.fail(f"{name}: nothing is written or published before the check")

    manifest = tmp_path / "manifest" / "workflows.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"defaults": DEFAULTS, "workflows": {
        "ok": {"enabled": True, "workflow_path": "x", "models": []},
        "wf": {"enabled": True, "workflow_path": "x", "models": [], **entry}}}))
    monkeypatch.setattr(storage, "from_args", lambda args: ReadOnlyStore({}))
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda *a: pytest.fail("no GPU job for an entry that cannot be run"))
    monkeypatch.setattr(sys, "argv", ["run_regression.py", "--manifest", str(manifest),
                                      "--commit", "new", "--branch", "master"])
    with pytest.raises(SystemExit, match=said):
        run_regression.main()


def test_the_comparators_are_those_of_the_kinds_that_can_have_a_reference():
    assert set(run_regression.COMPARATORS) == {"image", "text"}
    assert run_regression.comparator_for({}).kind == "image"
    assert run_regression.comparator_for({"output_kind": "text"}).kind == "text"
    with pytest.raises(ValueError, match="no comparator for output_kind 'audio' yet"):
        run_regression.comparator_for({"output_kind": "audio"})
    with pytest.raises(ValueError, match="unknown output_kind 'mesh'"):
        run_regression.comparator_for({"output_kind": "mesh"})
