"""run_regression: pure decision helpers and the summary handed to later steps."""
import os

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


def test_one_png_that_cannot_be_read_does_not_cost_the_others_their_thumbnail(tmp_path,
                                                                             monkeypatch, capsys):
    """A smoke run with a good picture and a cut one fails, which is right.
    Its row, the one a person opens, had no picture: the first error ended
    the loop, and the cell's thumbnail was null beside a published preview."""
    import io
    import numpy as np
    from PIL import Image
    data = io.BytesIO()
    Image.fromarray(np.zeros((640, 480, 3), dtype=np.uint8)).save(data, "PNG")
    good = data.getvalue()
    for label, files in (("cut-last", {"a_00001_.png": good, "b_00002_.png": good[:200]}),
                         ("cut-first", {"a_00001_.png": good[:200], "b_00002_.png": good}),
                         ("no-image", {"a_00001_.png": b"not a picture", "b_00002_.png": good})):
        out = tmp_path / label
        out.mkdir()
        for name, content in files.items():
            (out / name).write_bytes(content)
        kept = next(name for name, content in files.items() if content == good)
        assert run_regression.write_thumbnails(out) == kept[:-4] + ".webp", label
        assert [p.name for p in (out / "thumbs").iterdir()] == [kept[:-4] + ".webp"], label
    assert capsys.readouterr().out.count("no thumbnail for ") == 3
    # None of them readable: no thumbnail, and no error.
    out = tmp_path / "none"
    out.mkdir()
    (out / "a_00001_.png").write_bytes(good[:200])
    assert run_regression.write_thumbnails(out) is None
    # Through a run: the failing cell has the picture of the file that is one.
    workflow = {"3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "wf"}},
                "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "wf"}}}
    cell, _, store = _smoke_run(tmp_path / "run", monkeypatch, {
        "wf_00001_.png": good[:200], "wf_00002_.png": _picture()}, workflow=workflow)
    assert cell["verdict"] == "fail" and cell["thumbnail"] == "wf_00002_.webp"
    assert [(p["file"], p["check"]) for p in cell["smoke"]["problems"]] == [
        ("wf_00001_.png", "decode")]
    assert "outputs/thumbs/wf_00002_.webp" in store.published


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
    published = {"commit": "new", "overall": "pass", "has_infra_error": True,
                 "workflows": {"sdxl_t2i": {"verdict": "infra_error"}}}
    store = RecordingStore({"regression/runs/master/new/summary.json": published})
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(run_regression, "process_workflow",
                        lambda wf_id, *a: {"workflow_id": wf_id, "worker_status": "ok",
                                           "verdict": "pass", "detail": None})
    out = tmp_path / "summary.json"
    argv = ["run_regression.py", "--commit", "new", "--branch", "master",
            "--manifest", _manifest(tmp_path, "sdxl_t2i"), "--summary-out", str(out)]

    def attempt(*more):
        monkeypatch.setattr(sys, "argv", [*argv, *more])
        assert run_regression.main() == 0
        return json.loads(out.read_text(encoding="utf-8"))["attempt"], capsys.readouterr().out

    # A run that names its workflows, or that publishes nothing, is no attempt:
    # it pushes no start marker and the cron does not count it. Its summary
    # records the attempts counted so far, and it does not say it is the next.
    for more in (["--skip-publish"], ["--workflows", "sdxl_t2i", "--skip-publish"]):
        number, said = attempt(*more)
        assert number == 1 and "attempt" not in said, more
    assert store.events == [] and store.published_summary("master", "new") == published
    # The lane's run of the commit is one.
    number, said = attempt()
    assert number == 2 and "new was already tried on lane main; this is attempt 2\n" in said
    assert store.attempt_marker("master", "new")["attempt"] == 2
    # A later marker (a retry that died before publishing) raises the count.
    store.files["regression/runs/master/new/attempt.json"] = {"attempt": 3}
    assert attempt("--skip-publish")[0] == 3
    assert attempt()[0] == 4
    # A commit that nothing was published for: a run that is no attempt records 1.
    store.files.clear()
    assert attempt("--workflows", "sdxl_t2i", "--skip-publish")[0] == 1


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
    # Not tested at all is no reason either. On this lane it makes the run red.
    lane.status["wf_b"] = "worker_error"
    code, summary = lane.run("c3", "nightly", now=LEVEL_NOW + 2 * 86400)
    assert code == 1 and summary["overall"] == "pass" and summary["has_infra_error"] is True
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


def test_a_workflow_that_changes_its_level_while_it_is_red_starts_a_new_failure(tmp_path,
                                                                               monkeypatch):
    """Its output differs from its reference, then it moves to smoke level and
    fails its first smoke check. That is another failure: the cell was marked
    inherited, and the open chain still read as an output that changed against
    a reference the workflow no longer has."""
    import json
    lane = LevelLane(tmp_path, monkeypatch, {"wf": {}}, {"wf": {"3": {
        "class_type": "SaveImage", "inputs": {"filename_prefix": "wf"}}}})
    lane.reference("main", "wf", _shade(100))
    index = "regression/index/master/main.json"

    def level(value):
        manifest = json.loads(lane.manifest.read_text(encoding="utf-8"))
        manifest["workflows"]["wf"]["level"] = value
        lane.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def chain():
        head = lane.json(index)
        found = head["first_bad"]["wf"]
        return (found["commit"], found["prev_good"], found["golden"], found.get("level"),
                found["runs"]), head["entries"][0]["w"]["wf"]["d"]

    lane.files["wf"] = {"wf_00001_.png": _shade(100)}
    assert lane.run("c1", "main")[0] == 0
    # A flat picture of another shade: it differs from the reference.
    lane.files["wf"] = {"wf_00001_.png": _shade(200)}
    assert lane.run("c2", "main", now=LEVEL_NOW + 600)[0] == 1
    assert chain()[0] == ("c2", "c1", "v1", None, 1)
    # At smoke level the same picture fails as blank. The last result that says
    # something about a smoke check is the pass of c1, not the changed output of c2.
    level("smoke")
    code, summary = lane.run("c3", "main", now=LEVEL_NOW + 1200)
    cell = summary["workflows"]["wf"]
    assert code == 1 and "blank" in [p["check"] for p in cell["smoke"]["problems"]]
    assert (cell["prior_verdict"], cell["prior_commit"]) == ("pass", "c1")
    assert chain() == (("c3", "c1", None, "smoke", 1), None)
    # The same failure again is a standing one, of that chain.
    code, summary = lane.run("c4", "main", now=LEVEL_NOW + 1800)
    assert summary["workflows"]["wf"]["prior_commit"] == "c3"
    assert chain() == (("c3", "c1", None, "smoke", 2), "inherited")
    # Back at reference level, still with the other shade: a reference failure again.
    level("reference")
    code, summary = lane.run("c5", "main", now=LEVEL_NOW + 2400)
    assert code == 1 and summary["workflows"]["wf"]["prior_commit"] == "c4"
    assert chain()[0] == ("c5", "c1", "v1", None, 1)


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


# -- a nightly or weekly run ------------------------------------------------------------
# How such a lane runs its jobs: a few at a time, one attempt each, with a
# deadline of its own, a push every 10 workflows, one pass at the end for what
# can be tried again, and a red run when a workflow stayed untested.
LONG_LANES = {"schema_version": 2, "primary": "main", "lanes": {
    "main": {"cadence": "per-commit", "role": "primary", "layout": 1, "prefix": "regression",
             "endpoint_id": "ep-main"},
    "night": {"cadence": "nightly", "role": "lane", "layout": 2,
              "prefix": "regression/lanes/night", "endpoint_id": "ep-long", "max_in_flight": 2,
              "timeout_minutes": 120},
    "week": {"cadence": "weekly", "role": "lane", "layout": 2,
             "prefix": "regression/lanes/week", "endpoint_id": "ep-long", "max_in_flight": 2,
             "timeout_minutes": 350}}}


class Gpu:
    """Stands in for process_workflow. `script` says per workflow what each
    attempt answers (the last answer repeats): a verdict, or the fields of
    its cell. A `reason` among them goes into the run record, where a job that
    used up its time says so. `calls` lists the workflows in the order their
    jobs started, and `seen` what each call was given."""

    def __init__(self, script, clock=None, job_minutes=0):
        import threading
        self.script = {wf: a if isinstance(a, list) else [a] for wf, a in script.items()}
        self.calls, self.seen, self.lock = [], [], threading.Lock()
        self.clock, self.job_minutes, self.left_over = clock, job_minutes, None

    def __call__(self, wf_id, entry, defaults, args, repo_root, store, queue_deadline):
        import json
        from pathlib import Path
        with self.lock:
            self.calls.append(wf_id)
            attempt = self.calls.count(wf_id)
            if self.clock:
                self.clock.now += self.job_minutes * 60
        self.seen.append((wf_id, queue_deadline, getattr(args, "job_options", None)))
        answers = self.script[wf_id]
        answer = answers[min(attempt, len(answers)) - 1]
        answer = {"verdict": answer} if isinstance(answer, str) else dict(answer)
        reason = answer.pop("reason", None)
        cell = {"workflow_id": wf_id, "worker_status": "ok", "verdict": "pass", "detail": None,
                "env": {"image": "img"}, **answer}
        if cell["verdict"] == "infra_error" and "worker_status" not in answer:
            cell["worker_status"] = "infra_error"
        folder = Path(args.workdir) / wf_id
        self.left_over = (folder / "run.json").exists()  # what the attempt before left
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.json").write_text(json.dumps(
            {"status": cell["worker_status"], **({"reason": reason} if reason else {})}))
        return cell


class Clock:
    """Stands in for the time module in run_regression: it moves when told."""

    def __init__(self, now=NOW):
        self.now = now

    def time(self):
        return self.now


def _long_run(tmp_path, monkeypatch, store, gpu, *argv, lane="night", entries=None,
              commit="new", registry=LONG_LANES):
    """main() on a lane of LONG_LANES, for a manifest of the workflows in the
    gpu's script (each of the lane's cadence, with `entries[wf]` added).
    Returns (exit code, the --summary-out file)."""
    import json
    import sys
    import commit_meta
    import storage
    cadence = registry["lanes"][lane]["cadence"]
    manifest = tmp_path / "manifest" / "workflows.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [], "thresholds": {}},
        "workflows": {wf_id: {"enabled": True, "workflow_path": "x", "models": [],
                              "cadence": cadence, **(entries or {}).get(wf_id, {})}
                      for wf_id in gpu.script}}), encoding="utf-8")
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: registry)
    monkeypatch.setattr(storage, "from_args", lambda args: store)
    monkeypatch.setattr(commit_meta, "fetch_commit_meta", lambda sha: None)
    monkeypatch.setattr(commit_meta, "fetch_range", lambda prev, sha: {
        "prev": prev, "commits_between": 3, "compare_url": f"{prev}...{sha}"})
    monkeypatch.setattr(run_regression, "process_workflow", gpu)
    out = tmp_path / "summary-out.json"
    out.unlink(missing_ok=True)
    monkeypatch.setattr(sys, "argv", [
        "run_regression.py", "--commit", commit, "--branch", "master", "--lane", lane,
        "--manifest", str(manifest), "--workdir", str(tmp_path / "work"),
        "--summary-out", str(out), *argv])
    code = run_regression.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


class CountingEndpoint:
    """Stands in for RunPod: counts the jobs that are on the endpoint at one
    time. A job stays in progress until `together` jobs have been there at
    once (or 10 s have passed), so the count does not depend on timing."""

    def __init__(self, together):
        import threading
        self.lock, self.together = threading.Lock(), together
        self.on, self.most, self.jobs, self.looks, self.started = 0, 0, 0, {}, {}

    def client(self, endpoint_id):
        return self

    def run(self, job_input):
        import time
        with self.lock:
            self.jobs += 1
            self.on += 1
            self.most = max(self.most, self.on)
            job_id = f"job-{self.jobs}"
            self.looks[job_id], self.started[job_id] = [job_input["workflow_id"], 0], time.time()
        return job_id

    def status(self, job_id):
        import base64
        import time
        with self.lock:
            self.looks[job_id][1] += 1
            waited = time.time() - self.started[job_id]
            if self.looks[job_id][1] < 3 or (self.most < self.together and waited < 10):
                return {"status": "IN_PROGRESS"}
            self.on -= 1
        name, data = f"{self.looks[job_id][0]}_00001_.png", _picture()
        return {"status": "COMPLETED", "delayTime": 5, "executionTime": 9, "output": {
            "status": "ok", "timings": {"prompt_exec_s": 3.0}, "outputs": [
                {"filename": name, "bytes": len(data), "sha256": _sha(data),
                 "data_b64": base64.b64encode(data).decode()}]}}

    def cancel(self, job_id):
        pytest.fail(f"{job_id} was cancelled")


def _counted_run(tmp_path, monkeypatch, lane_id, registry, together, *argv):
    """Six smoke workflows through main(), process_workflow and
    run_workflow_job, on an endpoint that counts its jobs. Returns (exit code,
    the endpoint, the keyword arguments each job was submitted with)."""
    import types
    import submit_and_poll
    cadence = registry["lanes"][lane_id]["cadence"]
    lane = LevelLane(tmp_path, monkeypatch, {
        f"wf_{i}": {"level": "smoke", "cadence": cadence} for i in range(6)})
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: registry)
    endpoint, jobs, submit, pauses = CountingEndpoint(together), [], \
        submit_and_poll.run_workflow_job, []

    def job(client, wf_id, workflow, **kw):
        jobs.append(kw)
        return submit(client, wf_id, workflow, **kw)

    def pause(seconds):
        import time
        pauses.append(seconds)
        time.sleep(0.01)

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "RunPodClient", endpoint.client)
    monkeypatch.setattr(submit_and_poll, "time", types.SimpleNamespace(
        monotonic=submit_and_poll.time.monotonic, sleep=pause))
    code, summary = lane.run("c1", lane_id, "--skip-publish", *argv)
    assert [cell["verdict"] for cell in summary["workflows"].values()] == ["pass"] * 6
    return code, endpoint, jobs, pauses


def test_a_lane_with_max_in_flight_2_never_has_more_than_2_jobs_on_its_endpoint(tmp_path,
                                                                               monkeypatch):
    """Fifty jobs submitted at once to an endpoint with two workers would
    mostly be cancelled for waiting in the queue. The lane says how many it
    has there at a time, and no flag is needed for it."""
    night = {**LEVEL_LANES["lanes"]["nightly"], "max_in_flight": 2}
    registry = {**LEVEL_LANES, "lanes": {**LEVEL_LANES["lanes"], "nightly": night}}
    code, endpoint, jobs, pauses = _counted_run(tmp_path / "a", monkeypatch, "nightly", registry, 2)
    assert code == 0 and (endpoint.jobs, endpoint.most, endpoint.on) == (6, 2, 0)
    # What else such a lane gives a job: one attempt, 1800 s for the queue, a look every 3 s.
    assert {(kw["attempts"], kw["queue_deadline_s"], kw["poll_interval_s"]) for kw in jobs} == {
        (1, 1800, 3)}
    assert set(pauses) == {3}
    # The flag only overrides the lane.
    code, endpoint, _, _ = _counted_run(tmp_path / "b", monkeypatch, "nightly", registry, 3,
                                        "--max-in-flight", "3")
    assert (endpoint.jobs, endpoint.most) == (6, 3)
    code, endpoint, _, _ = _counted_run(tmp_path / "c", monkeypatch, "nightly", registry, 1,
                                        "--max-in-flight", "1")
    assert (endpoint.jobs, endpoint.most) == (6, 1)
    # A lane that says nothing has them all there at once, as every lane had.
    code, endpoint, jobs, pauses = _counted_run(tmp_path / "d", monkeypatch, "nightly",
                                                LEVEL_LANES, 6)
    assert (endpoint.jobs, endpoint.most) == (6, 6)
    # The per-commit lane: every job at once, and nothing new in what a job is given.
    code, endpoint, jobs, pauses = _counted_run(tmp_path / "e", monkeypatch, "main",
                                                LEVEL_LANES, 6)
    assert code == 0 and (endpoint.jobs, endpoint.most) == (6, 6)
    assert sorted(kw["queue_deadline_s"] for kw in jobs) == [600, 660, 720, 780, 840, 900]
    assert not [kw for kw in jobs if "attempts" in kw or "poll_interval_s" in kw]
    assert set(pauses) == {10}
    # It takes the cap too when it is asked to.
    code, endpoint, _, _ = _counted_run(tmp_path / "f", monkeypatch, "main", LEVEL_LANES, 2,
                                        "--max-in-flight", "2")
    assert (endpoint.jobs, endpoint.most) == (6, 2)


def test_what_a_lane_says_about_its_jobs_is_checked_before_anything_is_written(tmp_path,
                                                                             monkeypatch):
    class ReadOnlyStore(FakeStore):
        def __getattr__(self, name):
            pytest.fail(f"{name}: nothing is written before the check")

    for key, value in (("max_in_flight", 0), ("max_in_flight", "2"), ("max_in_flight", True),
                       ("timeout_minutes", -5), ("timeout_minutes", 1.5)):
        night = {**LONG_LANES["lanes"]["night"], key: value}
        registry = {**LONG_LANES, "lanes": {**LONG_LANES["lanes"], "night": night}}
        gpu = Gpu({"a": "pass"})
        with pytest.raises(SystemExit, match=f"{key} of the lane must be a whole number above 0"):
            _long_run(tmp_path, monkeypatch, ReadOnlyStore({}), gpu, registry=registry)
        assert gpu.calls == []
    for flag in ("--max-in-flight", "--deadline-minutes"):
        for value in ("0", "-1", "two"):
            with pytest.raises(SystemExit):
                _long_run(tmp_path, monkeypatch, ReadOnlyStore({}), Gpu({"a": "pass"}), flag, value)
    assert run_regression.lane_count({}, "max_in_flight") is None
    assert run_regression.lane_count({"max_in_flight": 3}, "max_in_flight") == 3


def test_a_workflow_whose_job_could_pass_the_runs_deadline_is_not_started(tmp_path, monkeypatch,
                                                                         capsys):
    """GitHub ends a job after its time limit, and a RunPod job in flight then
    runs on with nobody to read it. So a run does not submit a job whose worst
    case would pass its own deadline. A smaller job later in the line still
    runs. The run is red: a workflow was not tested."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    # On such a lane the runner gives up on a job after its timeout plus 900 s
    # plus the 1200 s more it may wait in the queue: 95 minutes for `big`, 45
    # for `small`. 60 minutes are left.
    entries = {"big": {"timeout_s": 3600}, "small": {"timeout_s": 600}}
    gpu = Gpu({"big": "pass", "small": "pass"})
    store = RecordingStore()
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--deadline-minutes", "60",
                              "--max-in-flight", "1", entries=entries)
    assert code == 1 and gpu.calls == ["small"]
    assert summary["workflows"]["big"] == {
        "workflow_id": "big", "worker_status": "not_reached", "verdict": "infra_error",
        "error": "not started: its job can take 95 min, and 60 min were left before the run's "
                 "deadline",
        "prior_verdict": None, "prior_commit": None}
    assert summary["workflows"]["small"]["verdict"] == "pass"
    assert (summary["overall"], summary["has_infra_error"]) == ("pass", True)
    out = capsys.readouterr().out
    assert ("lane night: 2 workflow(s), 1 job(s) at a time, one attempt each, and 60 min until "
            "the run's deadline\n") in out
    assert "[big] not started: its job can take 95 min, and 60 min are left" in out
    assert "::error::big: infra_error (worker=not_reached)\n" in out
    assert "not tested: big. That fails a nightly or weekly run.\n" in out
    # Nothing of it is in the runner's folder, and it is not tried at the end.
    assert not (tmp_path / "work" / "big").exists()
    # It is the lane's run of the commit all the same: published, and the pointer moves.
    assert store.events[-4:] == ["summary", "latest", "index", "regression master@new: pass"]
    # The cron leaves the commit alone: another run would not reach it either.
    import regression_skip
    assert not regression_skip.decide("new", store.published_summary("master", "new"))[0]

    # Without the flag the deadline is the lane's timeout_minutes minus 25, counted
    # from the start of the run: 95 minutes on this lane, which `big` just fits.
    gpu = Gpu({"big": "pass", "small": "pass"})
    code, _ = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, entries=entries)
    assert code == 0 and sorted(gpu.calls) == ["big", "small"]
    night = {**LONG_LANES["lanes"]["night"], "timeout_minutes": 119}
    registry = {**LONG_LANES, "lanes": {**LONG_LANES["lanes"], "night": night}}
    gpu = Gpu({"big": "pass", "small": "pass"})
    code, _ = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, entries=entries,
                        registry=registry)
    assert code == 1 and gpu.calls == ["small"]
    # A lane that names no limit has GitHub's own 360 minutes.
    night = {k: v for k, v in LONG_LANES["lanes"]["night"].items() if k != "timeout_minutes"}
    assert run_regression.run_deadline(None, night, True) == NOW + (360 - 25) * 60
    assert run_regression.run_deadline(None, {"timeout_minutes": 10}, True) == NOW
    assert run_regression.run_deadline(90, {"timeout_minutes": 10}, True) == NOW + 90 * 60

    # Time goes by while the jobs run: each takes 20 minutes here, one at a time.
    # After two of them 55 of the 95 minutes are left, 10 more than a job needs,
    # and after three there are 35.
    clock.now = NOW
    gpu = Gpu(dict.fromkeys("abcde", "pass"), clock, job_minutes=20)
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu,
                              "--max-in-flight", "1")
    assert code == 1 and gpu.calls == ["a", "b", "c"]
    assert [cell["worker_status"] for cell in summary["workflows"].values()] == [
        "ok", "ok", "ok", "not_reached", "not_reached"]
    # A smoke-level workflow that was not reached names its level, like every cell of one.
    cell = run_regression.not_reached_cell("wf", {"level": "smoke"}, 30.0, 2700.0)
    assert list(cell) == ["workflow_id", "worker_status", "verdict", "error", "level"]
    assert cell["error"].startswith("not started: its job can take 45 min, and 0 min were left")


def test_the_per_commit_lane_has_no_deadline_unless_it_is_given_one(tmp_path, monkeypatch,
                                                                   capsys):
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    assert run_regression.run_deadline(None, LONG_LANES["lanes"]["main"], False) is None
    gpu = Gpu({"a": "pass", "b": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, lane="main",
                              entries={"a": {"timeout_s": 86400}})
    assert code == 0 and sorted(gpu.calls) == ["a", "b"]
    # Its log is the one it had: no line about jobs at a time or a deadline.
    assert capsys.readouterr().out == "a: pass (worker=ok)\nb: pass (worker=ok)\noverall: pass\n"
    # With the flag: a job there has two attempts, so its worst case counts twice.
    assert run_regression.job_worst_case_s({}, {"timeout_s": 600}, 600, 2) == 2 * 1500
    assert run_regression.job_worst_case_s({"timeout_s": 60}, {"timeout_s": 600}, 1800, 1) == 2160
    gpu = Gpu({"a": "pass", "b": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, "--deadline-minutes",
                              "55", lane="main", entries={"a": {"timeout_s": 900}})
    # `a` could take 2 x (900 + 900) s, an hour. `b`, second in line, 2 x (600 + 900 + 60) s:
    # 52 minutes. A workflow that was not tested is a warning on this lane, as it was.
    assert code == 0 and gpu.calls == ["b"]
    assert summary["workflows"]["a"]["worker_status"] == "not_reached"
    assert summary["has_infra_error"] is True


class CheckpointStore(RecordingStore):
    """RecordingStore that keeps every summary it was given, and can fail a
    checkpoint."""

    def __init__(self, files=None, fail_at=()):
        super().__init__(files)
        self.summaries, self.pushes, self.fail_at = [], 0, fail_at

    def publish_summary(self, branch, commit, summary):
        import json
        super().publish_summary(branch, commit, summary)
        self.summaries.append(json.loads(json.dumps(summary)))

    def checkpoint(self, message):
        self.pushes += 1
        self.events.append(message if "workflows" in message else "checkpoint")
        if self.pushes in self.fail_at:
            raise RuntimeError("could not push checkpoint: remote hung up")


def test_a_nightly_or_weekly_run_pushes_what_it_has_every_10_workflows(tmp_path, monkeypatch,
                                                                      capsys):
    """A weekly run takes hours, and GitHub ends its job after six. A run
    that published once, at its end, would have spent the money and left
    nothing. The summary of a checkpoint says that it is one."""
    import regression_skip
    ids = [f"wf_{i:02d}" for i in range(25)]
    store = CheckpointStore()
    gpu = Gpu({**dict.fromkeys(ids, "pass"), "wf_03": "execution_error"})
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1",
                              lane="week")
    assert code == 1 and gpu.calls == ids
    assert store.events == [
        "snapshot", "marker", "checkpoint",
        "summary", "regression master@new: 10 of 25 workflows",
        "summary", "regression master@new: 20 of 25 workflows",
        "summary", "latest", "index", "regression master@new: fail"]
    first, second, final = store.summaries
    keys = SUMMARY_KEYS + ["published_new_bytes"]
    assert list(first) == keys + ["partial"] and first["partial"] == {"done": 10, "of": 25}
    assert second["partial"] == {"done": 20, "of": 25} and list(second["workflows"]) == ids[:20]
    assert list(first["workflows"]) == ids[:10] and first["overall"] == "fail"
    assert (first["attempt"], first["lane"], first["commit"]) == (1, "week", "new")
    assert all("prior_verdict" in cell for cell in first["workflows"].values())
    # The final summary has no such key: it is the lane's result of the commit.
    assert list(final) == keys and final == summary and list(final["workflows"]) == ids
    # A run that died after its first checkpoint: the cron gives the commit one more run.
    run, why = regression_skip.decide("new", first, marker=store.attempt_marker("master", "new"))
    assert run and "did not finish" in why
    assert not regression_skip.decide("new", final)[0]

    # A checkpoint that cannot be pushed is a warning. The run goes on and publishes.
    store = CheckpointStore(fail_at=(2,))
    gpu = Gpu(dict.fromkeys(ids, "pass"))
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1",
                              lane="week")
    assert code == 0 and len(store.summaries) == 3 and "partial" not in store.summaries[-1]
    assert store.events[-1] == "regression master@new: pass"
    assert ("::warning::the checkpoint after 10 workflows failed, the run goes on: "
            "RuntimeError('could not push checkpoint: remote hung up')") in capsys.readouterr().out

    # Not a run that names its workflows: its summary is merged into the lane's
    # at the end, and a partial one would have taken that one's place.
    store = CheckpointStore({"regression/runs/master/new/summary.json": summary})
    code, merged = _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")),
                             "--workflows", ",".join(ids[:12]), "--max-in-flight", "1",
                             lane="week")
    assert code == 0 and len(store.summaries) == 1 and "partial" not in merged
    assert store.events == ["snapshot", "summary", "index", "regression master@new: pass"]
    # Not a run that writes nothing.
    store = CheckpointStore()
    _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")), "--skip-publish",
              lane="week")
    assert store.events == [] and store.summaries == []
    # And not the per-commit lane: its run is over in minutes.
    store = CheckpointStore()
    code, summary = _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")),
                              lane="main")
    assert store.events == ["snapshot", "marker", "checkpoint", "summary", "latest", "index",
                            "regression master@new: pass"]
    assert list(summary) == SUMMARY_KEYS


def test_workflows_that_were_not_reached_do_not_count_for_a_checkpoint(tmp_path, monkeypatch):
    """When the deadline comes, every workflow left gets its cell at once.
    That is not ten more results to push."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    ids = [f"wf_{i:02d}" for i in range(40)]
    store = CheckpointStore()
    gpu = Gpu(dict.fromkeys(ids, "pass"), clock, job_minutes=5)
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1")
    # 95 minutes, 45 for a job's worst case, 5 minutes a job: 11 jobs start.
    assert code == 1 and gpu.calls == ids[:11]
    assert [e for e in store.events if "workflows" in e] == [
        "regression master@new: 10 of 40 workflows"]
    assert sum(cell["worker_status"] == "not_reached"
               for cell in summary["workflows"].values()) == 29


def test_the_pass_at_the_end_tries_again_what_can_pass_and_nothing_else(tmp_path, monkeypatch,
                                                                       capsys):
    """One attempt in the pool, and one more when everything else is done:
    by then an outage of some minutes is over. A job that used up its whole
    time is not tried again (a second one can cost another hour), nor is a
    workflow that another job could not change."""
    untested = {"verdict": "infra_error"}
    script = {
        "fetch": [{**untested, "worker_status": "model_fetch_error"}, "pass"],
        "queue": [untested, "fail"],                      # the queue, the network
        "worker": [{**untested, "worker_status": "worker_error"}] * 2,   # still not tested
        "spent": [{**untested, "reason": "timed_out"}, "pass"],
        "gated": [{**untested, "worker_status": "model_access_error"}, "pass"],
        "absent": [{**untested, "worker_status": "missing_model"}, "pass"],
        "broken": [{**untested, "worker_status": "bad_request"}, "pass"],
        "red": "execution_error", "green": "pass"}
    gpu = Gpu(script)
    store = RecordingStore()
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1")
    assert gpu.calls[:9] == list(script)  # the pool, in the manifest's order
    assert sorted(gpu.calls[9:]) == ["fetch", "queue", "worker"]
    assert {wf: (cell["verdict"], cell["worker_status"])
            for wf, cell in summary["workflows"].items()} == {
        "absent": ("infra_error", "missing_model"), "broken": ("infra_error", "bad_request"),
        "fetch": ("pass", "ok"), "gated": ("infra_error", "model_access_error"),
        "green": ("pass", "ok"), "queue": ("fail", "ok"), "red": ("execution_error", "ok"),
        "spent": ("infra_error", "infra_error"), "worker": ("infra_error", "worker_error")}
    assert code == 1 and summary["overall"] == "fail"
    out = capsys.readouterr().out
    assert "not tested, tried once more: fetch, queue, worker\n" in out
    assert "[fetch] second attempt: pass\n" in out
    # The folder of a second attempt is its own: what the first left is gone.
    assert gpu.left_over is False
    # What the pool produced is pushed before the pass starts, and the result after it.
    assert store.events == ["snapshot", "marker", "checkpoint", "summary", "checkpoint",
                            "summary", "latest", "index", "regression master@new: fail"]

    # When the second attempt tests everything, the run is green.
    gpu = Gpu({"fetch": script["fetch"], "green": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu)
    assert code == 0 and sorted(gpu.calls) == ["fetch", "fetch", "green"]
    assert summary["has_infra_error"] is False
    # A workflow whose only problem is that it was not tested makes the run red.
    gpu = Gpu({"gated": script["gated"], "green": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu)
    assert code == 1 and (summary["overall"], sorted(gpu.calls)) == ("pass", ["gated", "green"])
    # A run that names its workflows has the pass too.
    gpu = Gpu({"fetch": script["fetch"], "green": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, "--workflows",
                              "fetch")
    assert code == 0 and gpu.calls == ["fetch", "fetch"] and summary["scope"] == ["fetch"]

    # The per-commit lane has no such pass: its job had its second attempt
    # right away, and the cron tests the commit once more. Its exit code is
    # the one it always was.
    gpu = Gpu({"worker": script["worker"], "green": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, lane="main")
    assert code == 0 and sorted(gpu.calls) == ["green", "worker"]
    assert summary["has_infra_error"] is True
    assert [seen[2] for seen in gpu.seen] == [None, None]

    assert run_regression.NO_SECOND_TRY == {"missing_model", "model_access_error",
                                            "not_reached", "bad_request"}
    worth = run_regression.worth_another_try
    assert worth({"verdict": "infra_error", "worker_status": "harness_error"}, {})
    assert not worth({"verdict": "infra_error", "worker_status": "not_reached"}, {})
    assert not worth({"verdict": "execution_error", "worker_status": "timeout"}, {})
    assert not worth({"verdict": "infra_error", "worker_status": "infra_error"},
                     {"reason": "timed_out"})
    assert run_regression.run_record(tmp_path / "nowhere", "wf") == {}
    (tmp_path / "work" / "odd").mkdir()
    (tmp_path / "work" / "odd" / "run.json").write_text("[1]")
    assert run_regression.run_record(tmp_path / "work", "odd") == {}


def test_the_pass_at_the_end_starts_no_job_that_could_pass_the_deadline(tmp_path, monkeypatch):
    """The cell of the first attempt stays: it says why the workflow was not
    tested, which `not_reached` would hide."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    flaky = [{"verdict": "infra_error", "worker_status": "worker_error"}, "pass"]
    # 95 minutes, and each job takes 30: after the two of the pool 35 are left,
    # 10 short of the 45 a job may need.
    gpu = Gpu({"a": flaky, "b": "pass"}, clock, job_minutes=30)
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu,
                              "--max-in-flight", "1")
    assert code == 1 and gpu.calls == ["a", "b"]
    assert summary["workflows"]["a"]["worker_status"] == "worker_error"
    # With 10 minutes a job there is time for it.
    clock.now = NOW
    gpu = Gpu({"a": flaky, "b": "pass"}, clock, job_minutes=10)
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu,
                              "--max-in-flight", "1")
    assert code == 0 and gpu.calls == ["a", "b", "a"]


def test_a_pass_at_the_end_is_checkpointed_before_it_starts(tmp_path, monkeypatch):
    """The second attempts can take an hour. What the pool produced goes out
    first, if it has not all gone out yet."""
    ids = [f"wf_{i:02d}" for i in range(12)]
    flaky = [{"verdict": "infra_error", "worker_status": "worker_error"}, "pass"]
    store = CheckpointStore()
    gpu = Gpu({**dict.fromkeys(ids, "pass"), "wf_05": flaky})
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1")
    assert code == 0 and gpu.calls == [*ids, "wf_05"]
    assert [e for e in store.events if "workflows" in e] == [
        "regression master@new: 10 of 12 workflows", "regression master@new: 12 of 12 workflows"]
    assert [s.get("partial") for s in store.summaries] == [
        {"done": 10, "of": 12}, {"done": 12, "of": 12}, None]
    assert store.summaries[1]["workflows"]["wf_05"]["verdict"] == "infra_error"
    assert summary["workflows"]["wf_05"]["verdict"] == "pass"
    # Nothing new since the last push: no second one.
    ids = ids[:10]
    store = CheckpointStore()
    gpu = Gpu({**dict.fromkeys(ids, "pass"), "wf_05": flaky})
    _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1")
    assert [e for e in store.events if "workflows" in e] == [
        "regression master@new: 10 of 10 workflows"]


def test_a_checkpoint_counts_the_workflows_that_have_a_result(tmp_path, monkeypatch):
    """A workflow the deadline did not reach has a cell and no result. Four of
    those ahead of ten that ran read as "14 of 14 workflows have a result"."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    late = {f"big_{i}": {"timeout_s": 7200} for i in range(4)}  # 155 minutes: never fits in 95
    ids = [f"wf_{i:02d}" for i in range(10)]
    store = CheckpointStore()
    gpu = Gpu({**dict.fromkeys(late, "pass"), **dict.fromkeys(ids, "pass")}, clock)
    code, summary = _long_run(tmp_path, monkeypatch, store, gpu, "--max-in-flight", "1",
                              entries=late)
    assert code == 1 and gpu.calls == ids and len(summary["workflows"]) == 14
    assert [e for e in store.events if "workflows" in e] == [
        "regression master@new: 10 of 14 workflows"]
    first, final = store.summaries
    assert first["partial"] == {"done": 10, "of": 14} and len(first["workflows"]) == 14
    assert "partial" not in final


def test_a_second_full_run_of_a_commit_pushes_nothing_over_its_full_result(tmp_path, monkeypatch):
    """The cron runs a commit once more when a workflow stayed untested, and a
    dispatch can name the commit. A checkpoint of that run replaced the lane's
    full result with a part of one. If the run then died, the commit was left
    with fewer cells than it had, and the cron took it for tested."""
    import regression_skip
    ids = [f"wf_{i:02d}" for i in range(14)]
    untested = {"verdict": "infra_error", "worker_status": "worker_error"}
    store = CheckpointStore()
    code, first = _long_run(tmp_path, monkeypatch, store,
                            Gpu({**dict.fromkeys(ids, "pass"), "wf_13": untested}),
                            "--max-in-flight", "1")
    assert code == 1 and len(first["workflows"]) == 14 and first["attempt"] == 1
    assert [s.get("partial") for s in store.summaries] == [
        {"done": 10, "of": 14}, {"done": 14, "of": 14}, None]
    marker, latest = store.attempt_marker("master", "new"), store.latest_pointer("master")
    run, why = regression_skip.decide("new", first, latest, marker=marker)
    assert run and why == "new attempt 1 had infra errors; retrying (attempt 2 of 2)"

    # The second run. While its jobs run, the branch holds the first run's result.
    seen = []

    class Watching(Gpu):
        def __call__(self, *args):
            seen.append(store.published_summary("master", "new"))
            return super().__call__(*args)

    store.summaries.clear()
    store.events.clear()
    code, second = _long_run(tmp_path, monkeypatch, store, Watching(dict.fromkeys(ids, "pass")),
                             "--max-in-flight", "1")
    assert code == 0 and second["attempt"] == 2 and len(seen) == 14
    assert all(s == first for s in seen)
    assert [s.get("partial") for s in store.summaries] == [None]
    assert store.events == ["snapshot", "marker", "checkpoint", "summary", "latest", "index",
                            "regression master@new: pass"]
    # What a run that did not finish left, or a result of named workflows only,
    # is no full result: a run over it pushes on its way as any first run does.
    for left in ({**first, "partial": {"done": 10, "of": 14}}, {**first, "scope": ids[:3]}):
        store = CheckpointStore({"regression/runs/master/new/summary.json": left})
        _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")),
                  "--max-in-flight", "1")
        assert [s.get("partial") for s in store.summaries] == [{"done": 10, "of": 14}, None]


def test_a_retest_at_a_commit_whose_run_died_does_not_merge_into_what_it_left(tmp_path,
                                                                               monkeypatch):
    """What a run pushed on its way is no result of the lane: it is in no
    history table, and the cron gives the commit another run. A re-test of
    two workflows there used to merge into it. The summary kept `partial`
    with its old count, and the publish put the dead run into the index. It
    now writes a summary of its own, as at a commit without one."""
    import regression_skip
    ids = [f"wf_{i:02d}" for i in range(12)]
    store = CheckpointStore()
    _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")), "--max-in-flight",
              "1", lane="week")
    died = store.summaries[0]
    assert died["partial"] == {"done": 10, "of": 12} and len(died["workflows"]) == 10
    path = "regression/runs/master/new"
    store = CheckpointStore({f"{path}/summary.json": died, f"{path}/attempt.json": {"attempt": 1}})
    code, summary = _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")),
                              "--workflows", "wf_10,wf_11", lane="week")
    assert code == 0 and list(summary["workflows"]) == ["wf_10", "wf_11"]
    assert summary["scope"] == ["wf_10", "wf_11"]
    assert "partial" not in summary and "retests" not in summary
    assert store.published_summary("master", "new") == summary
    assert store.events == ["snapshot", "summary", "index", "regression master@new: pass"]
    # It is no attempt, and it does not end the count: the cron still gives the
    # commit the second attempt a run that died is owed.
    assert summary["attempt"] == 1
    run, why = regression_skip.decide("new", summary, marker=store.attempt_marker("master", "new"))
    assert run and why == ("new attempt 1 started but published no result; retrying (attempt 2 "
                           "of 2)")
    # A full result is still merged into, as on every lane.
    full = {k: v for k, v in died.items() if k != "partial"}
    store = CheckpointStore({f"{path}/summary.json": full})
    code, merged = _long_run(tmp_path, monkeypatch, store, Gpu(dict.fromkeys(ids, "pass")),
                             "--workflows", "wf_10,wf_11", lane="week")
    assert len(merged["workflows"]) == 12 and "scope" not in merged
    assert [r["workflows"] for r in merged["retests"]] == [["wf_10", "wf_11"]]


def test_the_result_of_a_second_job_says_that_it_is_one(tmp_path, monkeypatch):
    """The pass at the end gives an untested workflow a second job. Its folder
    replaces the first job's, and its run.json is a first attempt's, so only
    the cell can say that the workflow was tried twice."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    untested = {"verdict": "infra_error", "worker_status": "worker_error"}
    gpu = Gpu({"again_bad": untested, "again_good": [untested, "pass"], "good": "pass",
               "gated": {"verdict": "infra_error", "worker_status": "missing_model"}})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu)
    assert code == 1 and sorted(gpu.calls) == ["again_bad", "again_bad", "again_good",
                                               "again_good", "gated", "good"]
    assert {wf: cell.get("tries") for wf, cell in summary["workflows"].items()} == {
        "again_bad": 2, "again_good": 2, "gated": None, "good": None}
    assert "tries" not in summary["workflows"]["good"]
    # A second job that the deadline did not allow was not started: no mark.
    clock.now = NOW
    gpu = Gpu({"a": untested, "b": "pass"}, clock, job_minutes=30)
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu,
                              "--max-in-flight", "1")
    assert gpu.calls == ["a", "b"] and "tries" not in summary["workflows"]["a"]
    # A harness error in the second job is a second try all the same.
    clock.now = NOW

    class Breaking(Gpu):
        def __call__(self, wf_id, *args):
            if self.calls.count(wf_id) == 1:
                self.calls.append(wf_id)
                raise RuntimeError("the runner lost the job")
            return super().__call__(wf_id, *args)

    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), Breaking({"a": untested}))
    assert (summary["workflows"]["a"]["worker_status"], summary["workflows"]["a"]["tries"]) == (
        "harness_error", 2)
    # The per-commit lane has no such pass, and its cells have no such key.
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(),
                              Gpu({"a": untested, "b": "pass"}), lane="main")
    assert all("tries" not in cell for cell in summary["workflows"].values())


def test_the_two_numbers_of_a_job_that_was_not_started_never_read_as_if_it_fitted(
        tmp_path, monkeypatch, capsys):
    """A job that can take 95 minutes, a deadline 95 minutes after the start,
    and a second gone by: "can take 95 min, and 95 min are left" read as a
    contradiction. What the job needs is rounded up, what is left down."""
    class Ticking(Clock):
        def time(self):
            self.now += 1  # a second goes by between any two looks at the clock
            return self.now

    monkeypatch.setattr(run_regression, "time", Ticking())
    gpu = Gpu({"slow": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu,
                              entries={"slow": {"timeout_s": 3600}})
    assert code == 1 and gpu.calls == []
    assert summary["workflows"]["slow"]["error"] == (
        "not started: its job can take 95 min, and 94 min were left before the run's deadline")
    assert "[slow] not started: its job can take 95 min, and 94 min are left before the run's " \
           "deadline\n" in capsys.readouterr().out
    assert run_regression.not_started(59.9, 60.1) == (
        "not started: its job can take 2 min, and 0 min are left before the run's deadline")
    assert run_regression.not_started(-5, 2700, were=True) == (
        "not started: its job can take 45 min, and 0 min were left before the run's deadline")
    # The numbers are those validate holds a lane and its workflows to.
    import lanes
    import submit_and_poll
    assert run_regression.DEADLINE_MARGIN_MIN == lanes.DEADLINE_MARGIN_MIN
    assert lanes.LONG_JOB_ALLOWANCE_S == submit_and_poll.poll_deadline_s(
        0, run_regression.LONG_QUEUE_DEADLINE_S)
    lane = LONG_LANES["lanes"]["night"]
    monkeypatch.setattr(run_regression, "time", Clock())
    assert run_regression.run_deadline(None, lane, True) == NOW + lanes.deadline_window_s(lane)
    assert run_regression.job_worst_case_s(
        {"timeout_s": 3600}, {}, run_regression.LONG_QUEUE_DEADLINE_S, 1) == (
        3600 + lanes.LONG_JOB_ALLOWANCE_S)
    # And validate names this very pair before any run finds it.
    said = lanes.validate(LONG_LANES, {"defaults": {"timeout_s": 600}, "workflows": {
        "slow": {"enabled": True, "cadence": "nightly", "timeout_s": 3600}}})
    assert [p for p in said if p.startswith("workflow ")] == [
        "workflow slow: it can never start on lane night: its job can take 95 min (timeout_s "
        "3600 and 2100 s for the queue and a cold start), and a run's deadline is 95 min after "
        "its start. Raise the lane's timeout_minutes or lower timeout_s"]


# -- who broke it: each workflow's own range ------------------------------------------------
def _published(lane):
    """Every JSON file of the lane's store, as run_report reads a results tree."""
    import json
    root = lane.tmp / "store"
    return {p.relative_to(root).as_posix(): json.loads(p.read_text(encoding="utf-8"))
            for p in root.rglob("*.json")}


def test_a_weekly_failure_is_placed_by_the_workflows_own_last_result(tmp_path, monkeypatch):
    """A weekly run covers about 44 commits, and a workflow's last result is
    not always in the run before: that run may not have tested it. So the
    summary says for each red workflow how many commits lie between its own
    last result and this commit, and the report words the failure from that.
    A workflow without an earlier result is on its first run. A step of a
    bisect, at an older commit, reads its own short range."""
    import io
    import json
    import commit_meta
    import numpy as np
    import run_report
    from PIL import Image
    week = {"cadence": "weekly", "role": "lane", "layout": 2, "label": "weekly",
            "prefix": "regression/lanes/weekly", "endpoint_id": "ep-nightly", "max_in_flight": 2}
    registry = {**LEVEL_LANES, "lanes": {**LEVEL_LANES["lanes"], "weekly": week}}
    smoke = {"level": "smoke", "cadence": "weekly"}
    lane = LevelLane(tmp_path, monkeypatch, {
        "wf_exec": smoke, "wf_music": smoke, "wf_ok": smoke, "wf_new": {**smoke, "enabled": False}})
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: registry)
    asked = []

    def commits_between(prev, sha):
        # The number in a commit's name is its place on master.
        asked.append(prev)
        return {"prev": prev, "commits_between": max(int(sha[1:]) - int(prev[1:]), 0),
                "compare_url": f"https://github.com/Comfy-Org/ComfyUI/compare/{prev}...{sha}"}

    monkeypatch.setattr(commit_meta, "fetch_range", commits_between)
    black = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(black, "PNG")
    for wf_id in lane.files:
        lane.files[wf_id] = {f"{wf_id}_00001_.png": _picture()}
    first, second, third, step = "c0000100", "c0000144", "c0000188", "c0000105"
    prefix, day = "regression/lanes/weekly", 86400

    # Week 1: everything passes. Week 2: one workflow is not tested, twice.
    assert lane.run(first, "weekly")[0] == 0
    lane.status["wf_music"] = "worker_error"
    code, summary = lane.run(second, "weekly", now=LEVEL_NOW + 7 * day)
    assert code == 1 and summary["overall"] == "pass" and "ranges" not in summary
    assert lane.json(f"{prefix}/latest/master.json")["commit"] == second

    # Week 3: an execution error, a smoke failure of the workflow that week 2
    # did not test, and a new workflow that fails its first run.
    manifest = json.loads(lane.manifest.read_text(encoding="utf-8"))
    manifest["workflows"]["wf_new"]["enabled"] = True
    lane.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    lane.status.update(wf_exec="timeout", wf_music="ok")
    for wf_id in ("wf_music", "wf_new"):
        lane.files[wf_id] = {f"{wf_id}_00001_.png": black.getvalue()}
    asked.clear()
    code, summary = lane.run(third, "weekly", now=LEVEL_NOW + 14 * day)
    assert code == 1 and summary["overall"] == "fail"
    assert {wf: (cell["verdict"], cell["prior_verdict"], cell["prior_commit"])
            for wf, cell in summary["workflows"].items()} == {
        "wf_exec": ("execution_error", "pass", second), "wf_music": ("fail", "pass", first),
        "wf_new": ("fail", None, None), "wf_ok": ("pass", "pass", second)}
    compare = "https://github.com/Comfy-Org/ComfyUI/compare"
    # The run covers the 44 commits since the lane's previous full run.
    assert summary["tested_range"] == {"prev": second, "commits_between": 44,
                                       "compare_url": f"{compare}/{second}...{third}"}
    # Each red workflow has its own range. The one on its first run has none.
    assert summary["ranges"] == {
        "wf_exec": {"prev": second, "commits_between": 44,
                    "compare_url": f"{compare}/{second}...{third}"},
        "wf_music": {"prev": first, "commits_between": 88,
                     "compare_url": f"{compare}/{first}...{third}"}}
    assert list(summary) == RUN_SUMMARY_KEYS + ["ranges", "published_new_bytes"]
    assert sorted(asked) == [first, second], "GitHub is asked once for each range"
    # A full run of such a lane moves the pointer, red or not.
    assert lane.json(f"{prefix}/latest/master.json") == {
        "commit": third, "run_ts": LEVEL_NOW + 14 * day,
        "workflows": ["wf_exec", "wf_music", "wf_new", "wf_ok"]}
    # The index carries the ranges, and each chain the one it started with.
    head = lane.json("regression/index/master/weekly.json")
    assert head["entries"][0]["wr"] == {
        "wf_exec": {"n": 44, "url": f"{compare}/{second}...{third}", "prev": second},
        "wf_music": {"n": 88, "url": f"{compare}/{first}...{third}", "prev": first}}
    assert head["first_bad"]["wf_music"]["range"] == {
        "n": 88, "url": f"{compare}/{first}...{third}"}
    assert "range" not in head["first_bad"]["wf_new"]

    def report(commit):
        return run_report.lane_report(run_report.DictReader(_published(lane), "d" * 40),
                                      registry, commit, "master", "weekly").split("\n")

    lines = report(third)
    assert lines[0] == "## ❌ GPU regression · weekly · `c000014`..`c000018` (44 commits)"
    assert ("3 of 4 workflows failed: 2 new in this run, 1 on its first run. 1 passed its smoke "
            "checks.") in lines
    assert "#### ❌ `wf_music`: smoke check failed (new since `c000010`, 88 commits)" in lines
    assert "#### ❌ `wf_exec`: execution error (new since `c000014`, 44 commits)" in lines
    assert "#### ❌ `wf_new`: smoke check failed (first run)" in lines
    assert not [line for line in lines if line.startswith("#### ") and "new at this commit" in line]
    assert (f"It last passed at [`c000010`](https://github.com/Comfy-Org/ComfyUI/commit/{first}). "
            f"88 commits since then: [compare]({compare}/{first}...{third}). To test one commit "
            "of that range (needs write access to the worker repository):") in lines
    assert ("The workflow did not finish on this commit. Worker status: <code>timeout</code>. "
            "Before this commit it last ran to the end at [`c000014`](https://github.com/"
            f"Comfy-Org/ComfyUI/commit/{second}) (smoke pass): the error started after that "
            f"commit. 44 commits since then: [compare]({compare}/{second}...{third}). To test one "
            "commit of that range (needs write access to the worker repository):") in lines
    for wf_id in ("wf_exec", "wf_music"):
        assert ("gh workflow run nightly-weekly.yml --repo Comfy-Org/comfyci-runpod-worker "
                f"-f cadence=weekly -f commit=<sha> -f workflows={wf_id}") in lines
    assert run_report.FIRST_RUN in lines

    # A step of the bisect: the workflow alone, 5 commits after its last pass.
    asked.clear()
    code, summary = lane.run(step, "weekly", "--workflows", "wf_music",
                             now=LEVEL_NOW + 15 * day)
    assert code == 1 and summary["scope"] == ["wf_music"]
    assert summary["workflows"]["wf_music"]["prior_commit"] == first
    assert summary["ranges"] == {"wf_music": {"prev": first, "commits_between": 5,
                                             "compare_url": f"{compare}/{first}...{step}"}}
    # The run's own range starts at the pointer, which is ahead of this commit.
    assert summary["tested_range"]["commits_between"] == 0
    assert lane.json(f"{prefix}/latest/master.json")["commit"] == third
    lines = report(step)
    assert "#### ❌ `wf_music`: smoke check failed (new since `c000010`, 5 commits)" in lines
    assert "1 of 1 workflow failed: 1 new in this run." in lines
    assert not [line for line in lines if "new at this commit" in line and "Result labels" not in
                line]
    # The chain now starts at the step: 5 commits to look at, not 88.
    chain = lane.json("regression/index/master/weekly.json")["first_bad"]["wf_music"]
    assert (chain["commit"], chain["prev_good"], chain["range"]["n"]) == (step, first, 5)

    # The next step passes, 2 commits before the failing one. The chain keeps
    # its start and has a nearer last pass. The 5 commits the failing step
    # counted lie between it and the pass before, so they are not the chain's
    # any more, and the index cannot count commits: the range is given from
    # the new last pass, with its compare link and without a number.
    passing = "c0000103"
    lane.files["wf_music"] = {"wf_music_00001_.png": _picture()}
    code, summary = lane.run(passing, "weekly", "--workflows", "wf_music",
                             now=LEVEL_NOW + 16 * day)
    lane.files["wf_music"] = {"wf_music_00001_.png": black.getvalue()}
    assert code == 0 and summary["scope"] == ["wf_music"] and "ranges" not in summary
    chain = lane.json("regression/index/master/weekly.json")["first_bad"]["wf_music"]
    assert (chain["commit"], chain["prev_good"]) == (step, passing)
    assert chain["range"] == {"n": None, "url": f"{compare}/{passing}...{step}"}
    lines = report(passing)
    commits = "https://github.com/Comfy-Org/ComfyUI/commit"
    (open_line,) = [line for line in lines if line.startswith("- `wf_music`: ")]
    assert open_line.startswith(
        f"- `wf_music`: smoke check failed, failing since [`c000010`]({commits}/{step}) (one of "
        f"the commits since its last pass, [compare]({compare}/{passing}...{step})), first seen ")
    assert open_line.endswith(f"last passed at [`c000010`]({commits}/{passing})")
    assert "covered" not in open_line and "5 commits" not in open_line
    # In the lane's history a run of one workflow says so: its green row is no
    # week in which the lane was green. A full run's row has no such note.
    rows = {line.split(" | ")[0]: line for line in lines if line.startswith("| [`c0000")}
    assert "✅ smoke pass · 1 of 4 workflows |" in rows[f"| [`c000010`]({commits}/{passing}) "
                                                        "**this report**"]
    assert "❌ 1 of 1 failed · 1 of 4 workflows |" in rows[f"| [`c000010`]({commits}/{step})"]
    assert "workflows |" not in rows[f"| [`c000018`]({commits}/{third}) +44 commits"]

    # A range of one commit is the tested commit itself.
    lane.status["wf_ok"] = "timeout"
    code, summary = lane.run("c0000189", "weekly", now=LEVEL_NOW + 21 * day)
    assert summary["ranges"]["wf_ok"]["commits_between"] == 1
    # A failure that was already there has its range too: back to its last result.
    assert summary["ranges"]["wf_music"]["prev"] == third
    lines = report("c0000189")
    assert "#### ❌ `wf_ok`: execution error (new at this commit)" in lines
    assert ("The workflow did not finish on this commit. Worker status: <code>timeout</code>. "
            "Before this commit it last ran to the end at [`c000018`](https://github.com/"
            f"Comfy-Org/ComfyUI/commit/{third}) (smoke pass), so the error is new at this commit."
            ) in lines
    # Only a red cell with an earlier result at another commit has a range.
    assert run_regression.own_ranges(
        [{"workflow_id": "a", "verdict": "fail", "prior_commit": "p"},
         {"workflow_id": "b", "verdict": "pass", "prior_commit": "p"},
         {"workflow_id": "c", "verdict": "execution_error", "prior_commit": None},
         {"workflow_id": "d", "verdict": "fail", "prior_commit": "here"},
         {"workflow_id": "e", "verdict": "infra_error", "prior_commit": "p"}],
        "here", lambda prev: {"prev": prev}) == {"a": {"prev": "p"}}


def test_a_retest_on_a_weekly_lane_keeps_the_other_cells_and_their_ranges(tmp_path, monkeypatch):
    """Running one workflow of a weekly run again must not cost the other
    fifty. The summary stays the lane's run of the commit: its cells byte for
    byte, its range, its ranges, and what it said it published."""
    import io
    import json
    import commit_meta
    import numpy as np
    from PIL import Image
    week = {"cadence": "weekly", "role": "lane", "layout": 2,
            "prefix": "regression/lanes/weekly", "endpoint_id": "ep-nightly", "max_in_flight": 2}
    registry = {**LEVEL_LANES, "lanes": {**LEVEL_LANES["lanes"], "weekly": week}}
    smoke = {"level": "smoke", "cadence": "weekly"}
    lane = LevelLane(tmp_path, monkeypatch, {"wf_a": smoke, "wf_b": smoke, "wf_c": smoke})
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: registry)
    asked = []
    monkeypatch.setattr(commit_meta, "fetch_range", lambda prev, sha: asked.append(prev) or {
        "prev": prev, "commits_between": int(sha[1:]) - int(prev[1:]), "compare_url": "u"})
    black = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(black, "PNG")
    for wf_id in lane.files:
        lane.files[wf_id] = {f"{wf_id}_00001_.png": _picture()}
    first, second, prefix = "c0000100", "c0000144", "regression/lanes/weekly"
    assert lane.run(first, "weekly")[0] == 0

    # The lane's run of the second commit: one failure, one workflow not tested.
    lane.files["wf_a"] = {"wf_a_00001_.png": black.getvalue()}
    lane.status["wf_b"] = "worker_error"
    code, old = lane.run(second, "weekly", now=LEVEL_NOW + 7 * 86400)
    assert code == 1 and (old["overall"], old["has_infra_error"]) == ("fail", True)
    assert list(old) == RUN_SUMMARY_KEYS + ["ranges", "published_new_bytes"]
    assert old["ranges"] == {"wf_a": {"prev": first, "commits_between": 44, "compare_url": "u"}}
    path = f"{prefix}/runs/master/{second}/summary.json"
    old_text, pointer = lane.text(path), lane.text(f"{prefix}/latest/master.json")

    # A day later the workflow that was not tested is run again, alone.
    lane.status["wf_b"] = "ok"
    asked.clear()
    code, new = lane.run(second, "weekly", "--workflows", "wf_b", now=LEVEL_NOW + 8 * 86400)
    new_text = lane.text(path)
    assert code == 0 and json.loads(new_text) == new
    for wf_id in ("wf_a", "wf_c"):
        assert _cell_text(new_text, wf_id) == _cell_text(old_text, wf_id), wf_id
    assert new["workflows"]["wf_b"]["verdict"] == "pass"
    assert list(new) == RUN_SUMMARY_KEYS + ["ranges", "published_new_bytes", "retests"]
    for key in ("run_ts", "attempt", "tested_range", "ranges", "published_new_bytes"):
        assert new[key] == old[key], key
    assert (new["overall"], new["has_infra_error"]) == ("fail", False)
    assert new["retests"] == [{"run_ts": LEVEL_NOW + 8 * 86400, "run_url": None, "run_id": None,
                               "env": None, "workflows": ["wf_b"]}]
    assert "partial" not in new and "scope" not in new
    # It is not the lane's run of the commit: the pointer stays where it was.
    assert lane.text(f"{prefix}/latest/master.json") == pointer
    # The pointer stands at this commit and a green cell has no range: GitHub was not asked.
    assert asked == []

    # The failing workflow tested again: its range is worked out again, the rest kept.
    code, again = lane.run(second, "weekly", "--workflows", "wf_a", now=LEVEL_NOW + 9 * 86400)
    assert code == 1 and again["ranges"] == old["ranges"] and asked == [first]
    assert [r["workflows"] for r in again["retests"]] == [["wf_b"], ["wf_a"]]
    assert _cell_text(lane.text(path), "wf_c") == _cell_text(old_text, "wf_c")
    # Fixed: its range goes, and the commit is clean.
    lane.files["wf_a"] = {"wf_a_00001_.png": _picture()}
    code, fixed = lane.run(second, "weekly", "--workflows", "wf_a", now=LEVEL_NOW + 10 * 86400)
    assert code == 0 and "ranges" not in fixed and fixed["overall"] == "pass"
    assert list(fixed) == RUN_SUMMARY_KEYS + ["published_new_bytes", "retests"]


# -- the stop --------------------------------------------------------------------------
class LiveClient:
    """A RunPod client with jobs in flight, as the stop handler sees it.
    `refuse` are the jobs whose cancel RunPod does not take."""

    def __init__(self, *job_ids, log=None, refuse=()):
        self.live, self.log = set(job_ids), log if log is not None else []
        self.refuse = set(refuse)

    def cancel_live(self, timeout=None):
        job_ids = sorted(self.live)
        self.log.extend(job_ids)
        cancelled = [job_id for job_id in job_ids if job_id not in self.refuse]
        self.live -= set(cancelled)
        return cancelled, [job_id for job_id in job_ids if job_id in self.refuse]


class StopClock:
    """The time of a stop. Every look takes its seconds, and a moment of real
    time in which a cancel call that is answered at once is over. `at` maps a
    time to what happens then."""

    def __init__(self, at=None):
        self.now, self.at = 0.0, dict(at or {})

    def __call__(self):
        return self.now

    def pause(self, seconds):
        import time
        time.sleep(0.002)
        self.now = round(self.now + seconds, 6)
        for when in sorted(t for t in self.at if t <= self.now):
            self.at.pop(when)()


def test_a_stopped_run_cancels_once_more_a_second_later(monkeypatch, capsys):
    """A job whose submission RunPod had not answered when the signal came is
    in no client's list yet: it was created, ran on to its end and was billed,
    and the log said that every job was cancelled. A cancel that RunPod did
    not take was reported as done too."""
    import signal
    import threading
    # A job that gets its id between the two passes: the second one cancels it.
    log = []
    late = LiveClient(log=log)
    clock = StopClock({0.5: lambda: late.live.add("job-2")})
    clients = [LiveClient("job-1", log=log), late]
    assert run_regression.cancel_in_flight(clients, pause=clock.pause, clock=clock) == (
        ["job-1", "job-2"], [], 0)
    assert log == ["job-1", "job-2"]
    # No client said that a submission was on its way, so the second pass came
    # a second after the first, as it always did.
    second = run_regression.STOP_SECOND_PASS_S
    assert second == 1.0 and second <= clock.now < second + 1.0

    # A cancel that failed gets the second call. A job that is still not
    # cancelled then is named as such, and not among the cancelled ones.
    log.clear()
    clients = [LiveClient("job-1", "job-2", log=log, refuse=["job-2"]),
               LiveClient("job-3", log=log, refuse=["job-3"])]
    clock = StopClock({0.5: lambda: clients[1].refuse.clear()})
    assert run_regression.cancel_in_flight(clients, pause=clock.pause, clock=clock) == (
        ["job-1", "job-3"], ["job-2"], 0)
    assert sorted(log) == ["job-1", "job-2", "job-2", "job-3", "job-3"]
    assert [sorted(client.live) for client in clients] == [["job-2"], []]
    # RunPod did not answer the cancel call, and a look at the job then finds
    # it ended: it is in flight no more, so it is not named as one that may
    # run on. The line used to say so of jobs RunPod had cancelled.
    ended = LiveClient("job-1", refuse=["job-1"])
    clock = StopClock({0.5: lambda: ended.live.discard("job-1")})
    assert run_regression.cancel_in_flight([ended], pause=clock.pause, clock=clock) == (
        ["job-1"], [], 0)
    monkeypatch.setattr(run_regression, "STOP_SECOND_PASS_S", 0)
    left = []
    clients = [LiveClient("job-1", "job-2", refuse=["job-2"]),
               LiveClient("job-3", refuse=["job-3"])]
    run_regression.stop_handler(clients, threading.Event(), leave=left.append)(signal.SIGINT, None)
    assert capsys.readouterr().out == (
        f"::error::stopped by signal {int(signal.SIGINT)}: cancelled 1 RunPod job(s) in flight "
        "(job-1). Not cancelled, RunPod did not take the call: job-2, job-3. They may run on to "
        "their end: cancel them by hand\n")
    assert left == [128 + signal.SIGINT]


class SilentClient(LiveClient):
    """A client whose cancel calls RunPod takes and does not answer."""

    def __init__(self, *job_ids):
        import threading
        super().__init__(*job_ids)
        self.release = threading.Event()

    def cancel_live(self, timeout=None):
        self.log.extend(sorted(self.live))
        self.release.wait(60)
        return [], sorted(self.live)


def test_a_stopped_run_waits_for_the_submissions_runpod_has_not_answered(monkeypatch, capsys):
    """The second pass came one second after the first, whatever was still on
    its way. A submission that RunPod answered three seconds after the signal
    left its job running, billed, and the log said "cancelled 0 RunPod job(s)
    in flight (none)"."""
    import signal
    import threading
    import time
    last, budget = run_regression.STOP_LAST_PASS_S, run_regression.STOP_BUDGET_S
    # GitHub's second signal comes 7.5 s after the first, and ends the process
    # without a word. The stop is over a second before the 7 s it has at most.
    assert run_regression.STOP_SECOND_PASS_S < last < budget <= 6.0
    # RunPod answers the submission 3 s after the signal: the job is cancelled.
    log = []
    late = LiveClient(log=log)
    late.submitting = 1

    def answered():
        late.live.add("job-2")
        late.submitting = 0

    clock = StopClock({3.0: answered})
    assert run_regression.cancel_in_flight([LiveClient("job-1", log=log), late],
                                           pause=clock.pause, clock=clock) == (
        ["job-1", "job-2"], [], 0)
    assert log == ["job-1", "job-2"] and 3.0 <= clock.now < 4.0
    # It never answers: the last pass is not put off for ever, and the
    # submission is counted. Its job, if there is one, has no id to name.
    never = LiveClient()
    never.submitting = 1
    clock = StopClock()
    assert run_regression.cancel_in_flight([LiveClient("job-1"), never, None],
                                           pause=clock.pause, clock=clock) == (["job-1"], [], 1)
    assert last <= clock.now < last + 1.0
    # RunPod takes the cancel calls and stays silent: the stop does not wait
    # for them past its limit, and the job is named as not cancelled.
    silent = SilentClient("job-1")
    clock = StopClock()
    assert run_regression.cancel_in_flight([silent, LiveClient("job-2")], pause=clock.pause,
                                           clock=clock) == (["job-2"], ["job-1"], 0)
    assert budget <= clock.now < budget + 0.1 and silent.log == ["job-1", "job-1"]
    silent.release.set()

    # The same with the real clock, on limits a test can wait for. Both waits
    # are held, and the last line says what is left.
    for name, seconds in (("STOP_SECOND_PASS_S", 0.05), ("STOP_LAST_PASS_S", 0.3),
                          ("STOP_BUDGET_S", 0.6), ("STOP_LOOK_S", 0.01)):
        monkeypatch.setattr(run_regression, name, seconds)
    silent, never, left = SilentClient("job-1"), LiveClient("job-3"), []
    never.submitting = 2
    start = time.monotonic()
    run_regression.stop_handler([silent, never], threading.Event(), leave=left.append)(
        signal.SIGINT, None)
    assert 0.6 <= time.monotonic() - start < 1.6 and left == [128 + signal.SIGINT]
    silent.release.set()
    assert capsys.readouterr().out == (
        f"::error::stopped by signal {int(signal.SIGINT)}: cancelled 1 RunPod job(s) in flight "
        "(job-3). Not cancelled, RunPod did not take the call: job-1. They may run on to "
        "their end: cancel them by hand. 2 submission(s) RunPod had not answered yet: each may "
        "have started a job that nothing cancelled. Look for them on the endpoint and cancel "
        "them by hand\n")


def test_a_stop_signal_cancels_the_jobs_in_flight_and_ends_the_process(monkeypatch, capsys):
    """GitHub sends an interrupt, 7.5 s later a terminate, then it kills.
    Without a handler the RunPod jobs would run on, billed, to their end."""
    import signal
    import threading
    monkeypatch.setattr(run_regression, "STOP_SECOND_PASS_S", 0)
    log, left = [], []
    stopping = threading.Event()
    # A client per workflow: two have a job, one has ended its job, one is no client.
    clients = [LiveClient("job-7", log=log), LiveClient(log=log), None,
               LiveClient("job-9", "job-8", log=log)]
    handle = run_regression.stop_handler(clients, stopping, leave=left.append)
    handle(signal.SIGINT, None)
    assert sorted(log) == ["job-7", "job-8", "job-9"] and stopping.is_set()
    assert left == [128 + signal.SIGINT]
    assert (f"::error::stopped by signal {int(signal.SIGINT)}: cancelled 3 RunPod job(s) in "
            "flight (job-7, job-8, job-9)\n") in capsys.readouterr().out
    # The second signal finds the cancels sent: it only ends the process.
    clients.append(LiveClient("job-10", log=log))
    handle(signal.SIGTERM, None)
    assert left == [128 + signal.SIGINT, 128 + signal.SIGTERM] and "job-10" not in log
    assert capsys.readouterr().out == ""
    # No job in flight: nothing to cancel, and it still ends.
    run_regression.stop_handler([], threading.Event(), leave=left.append)(signal.SIGTERM, None)
    assert left[-1] == 143 and "cancelled 0 RunPod job(s) in flight (none)" in (
        capsys.readouterr().out)
    assert run_regression.cancel_in_flight([None, LiveClient()]) == ([], [], 0)
    assert run_regression.STOP_SIGNALS == (signal.SIGINT, signal.SIGTERM)


def test_the_stop_handler_is_set_for_the_jobs_of_a_nightly_or_weekly_run_only(tmp_path,
                                                                             monkeypatch):
    import signal
    process_workflow = run_regression.process_workflow
    before = {sig: signal.getsignal(sig) for sig in run_regression.STOP_SIGNALS}
    during = []

    class Watching(Gpu):
        def __call__(self, *args):
            during.append({sig: signal.getsignal(sig) for sig in run_regression.STOP_SIGNALS})
            return super().__call__(*args)

    code, _ = _long_run(tmp_path, monkeypatch, RecordingStore(), Watching({"a": "pass"}))
    assert code == 0 and len(during) == 1
    for sig in run_regression.STOP_SIGNALS:
        assert during[0][sig] is not before[sig] and callable(during[0][sig])
        assert signal.getsignal(sig) is before[sig], "put back when the jobs are done"
    # Put back also when the run ends in an error.
    class Failing(RecordingStore):
        def publish_summary(self, branch, commit, summary):
            raise RuntimeError("the push failed")

    with pytest.raises(RuntimeError, match="the push failed"):
        _long_run(tmp_path, monkeypatch, Failing(), Gpu({"a": "pass"}))
    assert {sig: signal.getsignal(sig) for sig in run_regression.STOP_SIGNALS} == before
    # The per-commit lane is as it was: no handler.
    during.clear()
    _long_run(tmp_path, monkeypatch, RecordingStore(), Watching({"a": "pass"}), lane="main")
    assert during == [before]
    # process_workflow hands the run its clients, on such a lane only.
    import types
    made = []
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: made.append(
        LiveClient()) or made[-1])
    monkeypatch.setattr(run_regression, "run_workflow_job",
                        lambda client, wf_id, workflow, **kw: {"status": "worker_error"})
    (tmp_path / "wf_api.json").write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(workdir=str(tmp_path / "w"), endpoint_id="ep", lane_id="night",
                                 commit="new", repo_url=None, branch="master", skip_publish=True,
                                 clients=[])
    entry = {"workflow_path": "wf_api.json", "models": []}
    process_workflow("wf", entry, DEFAULTS, args, tmp_path, None)
    assert args.clients == made and len(made) == 1
    del args.clients
    process_workflow("wf", entry, DEFAULTS, args, tmp_path, None)
    assert len(made) == 2


@pytest.mark.skipif(os.name == "nt",
                    reason="on Windows a signal does not interrupt the wait for a worker thread")
def test_a_real_interrupt_reaches_the_run_while_it_waits_for_its_jobs(tmp_path, monkeypatch):
    """The run's main thread waits for its worker threads. The handler must
    run there and then, not when a job ends an hour later: it cancels the two
    jobs in flight, and no job of the workflows still in line is submitted."""
    import signal
    import threading
    import time
    cancelled, submitted, in_flight, released = [], [], threading.Semaphore(0), threading.Event()

    def client(endpoint_id):
        return LiveClient(log=cancelled)

    def job(client, wf_id, workflow, **kw):
        submitted.append(wf_id)
        client.live.add(f"job-{wf_id}")
        in_flight.release()
        assert released.wait(60), "the interrupt never came"
        return {"status": "worker_error", "error": {"message": "cancelled"}}

    def leave(code):
        released.set()
        raise SystemExit(code)

    def install(clients, stopping):
        earlier = signal.signal(signal.SIGINT, run_regression.stop_handler(clients, stopping,
                                                                           leave=leave))
        return lambda: signal.signal(signal.SIGINT, earlier)

    def interrupt():
        for _ in range(2):  # both slots of the lane hold a job
            assert in_flight.acquire(timeout=60)
        time.sleep(0.3)  # the run has handed its pool the other three and waits
        os.kill(os.getpid(), signal.SIGINT)

    lane = LevelLane(tmp_path, monkeypatch, {
        f"wf_{i}": {"level": "smoke", "cadence": "nightly"} for i in range(5)})
    night = {**LEVEL_LANES["lanes"]["nightly"], "max_in_flight": 2}
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: {
        **LEVEL_LANES, "lanes": {**LEVEL_LANES["lanes"], "nightly": night}})
    monkeypatch.setattr(run_regression, "RunPodClient", client)
    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "install_stop_handler", install)
    before = signal.getsignal(signal.SIGINT)
    killer = threading.Thread(target=interrupt)
    killer.start()
    with pytest.raises(SystemExit) as stopped:
        lane.run("c1", "nightly", "--skip-publish")
    killer.join(60)
    assert stopped.value.code == 128 + signal.SIGINT
    assert len(submitted) == 2 and sorted(cancelled) == sorted(f"job-{wf}" for wf in submitted)
    assert signal.getsignal(signal.SIGINT) is before


# -- a lane whose jobs fetch their models ---------------------------------------------------
# The weekly lane says `"model_storage": "job"`: each job downloads the models
# the volume does not hold. The runner tells the job to, and waits longer.
FETCH_LANES = {**LEVEL_LANES, "lanes": {**LEVEL_LANES["lanes"], "weekly": {
    "cadence": "weekly", "role": "lane", "layout": 2, "prefix": "regression/lanes/weekly",
    "endpoint_id": "ep-long", "model_storage": "job", "timeout_minutes": 350}}}
# What process_workflow gives run_workflow_job on a per-commit lane, besides
# the client, the id and the workflow. A lane with its models on the volume
# gains nothing here.
JOB_KEYWORDS = {"comfy_commit", "models", "timeout_s", "comfy_flags", "workdir", "comfy_repo",
                "lane", "custom_nodes", "queue_deadline_s"}


def _model(name, size, directory="vae"):
    return {"directory": directory, "name": name, "bytes": size, "sha256": "ab" * 32,
            "url": f"https://huggingface.co/org/repo/resolve/main/{name}"}


def _fetch_lane(tmp_path, monkeypatch, entries, answers=None):
    """A LevelLane on FETCH_LANES whose GPU keeps what each job was given
    (`lane.sent`) and adds `answers[wf_id]` to what the job answers."""
    lane = LevelLane(tmp_path, monkeypatch, entries)
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: FETCH_LANES)
    lane.sent = {}

    def job(client, wf_id, workflow, **kw):
        lane.sent[wf_id] = kw
        rec = _smoke_job(lane.files[wf_id], lane.status[wf_id])(client, wf_id, workflow,
                                                                workdir=kw["workdir"])
        return {**rec, **(answers or {}).get(wf_id, {})}

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    for wf_id in entries:
        lane.files[wf_id] = {f"{wf_id}_00001_.png": _picture()}
    return lane


def test_a_lane_that_fetches_tells_each_job_to_and_gives_it_the_time(tmp_path, monkeypatch):
    smoke = {"level": "smoke", "cadence": "weekly"}
    lane = _fetch_lane(tmp_path, monkeypatch, {
        "big": {**smoke, "models": [_model("a.safetensors", 60 * 10**9),
                                    _model("b.safetensors", 1),
                                    _model("clip.mp4", 10**9, "input")]},
        "bare": dict(smoke)})
    code, summary = lane.run("c1", "weekly", "--skip-publish")
    assert code == 0 and [c["verdict"] for c in summary["workflows"].values()] == ["pass"] * 2
    # 60 GB and a byte at 25 MB a second, rounded up. The input file is not fetched.
    assert lane.sent["big"]["fetch_missing"] is True
    assert lane.sent["big"]["fetch_allowance_s"] == 2401
    assert (lane.sent["bare"]["fetch_missing"], lane.sent["bare"]["fetch_allowance_s"]) == (True, 0)
    assert set(lane.sent["big"]) == JOB_KEYWORDS | {"attempts", "poll_interval_s",
                                                    "fetch_missing", "fetch_allowance_s"}
    # The job is given its whole model list, as on every lane: the worker
    # decides what the volume lacks.
    assert [m["name"] for m in lane.sent["big"]["models"]] == ["a.safetensors", "b.safetensors",
                                                              "clip.mp4"]
    # How many files at once is the worker's own choice, unless the run says it.
    code, _ = lane.run("c2", "weekly", "--skip-publish", "--fetch-streams", "1")
    assert code == 0 and lane.sent["big"]["fetch_streams"] == 1
    assert run_regression.fetch_options({"models": [_model("a", 25_000_001)]},
                                        FETCH_LANES["lanes"]["weekly"]) == {
        "fetch_missing": True, "fetch_allowance_s": 2}


def test_a_lane_with_its_models_on_the_volume_submits_its_jobs_as_before(tmp_path, monkeypatch):
    """This test passes on the code from before jobs could fetch models too:
    it pins what that code gave a job."""
    models = [_model("a.safetensors", 60 * 10**9)]
    lane = _fetch_lane(tmp_path / "main", monkeypatch, {"wf": {"level": "smoke",
                                                               "models": models}})
    code, _ = lane.run("c1", "main", "--skip-publish")
    assert code == 0 and set(lane.sent["wf"]) == JOB_KEYWORDS
    lane = _fetch_lane(tmp_path / "nightly", monkeypatch, {"wf": {
        "level": "smoke", "cadence": "nightly", "models": models}})
    code, _ = lane.run("c1", "nightly", "--skip-publish")
    assert code == 0 and set(lane.sent["wf"]) == JOB_KEYWORDS | {"attempts", "poll_interval_s"}


def test_only_a_lane_that_fetches_gives_its_jobs_the_fetch_options(tmp_path, monkeypatch):
    models = [_model("a.safetensors", 60 * 10**9)]
    lane = _fetch_lane(tmp_path, monkeypatch, {"wf": {"level": "smoke", "models": models}})
    # The flag is for a lane that fetches. On another it changes nothing.
    code, _ = lane.run("c1", "main", "--skip-publish", "--fetch-streams", "1")
    assert code == 0 and set(lane.sent["wf"]) == JOB_KEYWORDS
    for other in (None, {}, FETCH_LANES["lanes"]["main"], FETCH_LANES["lanes"]["nightly"],
                  {"cadence": "retired", "model_storage": "job"}, {"model_storage": "volume"}):
        assert run_regression.fetch_options({"models": models}, other, streams=1) == {}, other


def test_the_record_and_the_cell_of_a_job_that_fetched_say_how_it_went(tmp_path, monkeypatch):
    report = {"files": 2, "bytes": 800, "seconds": 4.0, "mb_per_s": 0.0, "streams": 4,
              "token": True, "reused": 0, "reused_bytes": 0, "evicted": 0}
    answers = {"wf": {"model_fetch": report,
                      "timings": {"model_fetch_s": 4.2, "prompt_exec_s": 3.0}}}
    lane = _fetch_lane(tmp_path, monkeypatch, {"wf": {
        "level": "smoke", "cadence": "weekly", "models": [_model("a.safetensors", 800)]}},
        answers)
    code, summary = lane.run("c1", "weekly")
    assert code == 0
    record = lane.json("regression/lanes/weekly/runs/master/c1/wf/run.json")
    assert record["model_fetch"] == report and record["timings"]["model_fetch_s"] == 4.2
    cell = summary["workflows"]["wf"]
    # The cell has the keys of any smoke cell. The download time rides in
    # `timings`, apart from the workflow's own seconds.
    assert list(cell) == CELL_KEYS + ["level", "smoke", "prior_verdict", "prior_commit"]
    assert cell["timings"] == {"model_fetch_s": 4.2, "prompt_exec_s": 3.0}


def test_a_model_that_could_not_be_fetched_is_not_tested_whatever_the_level(tmp_path,
                                                                           monkeypatch):
    """`model_fetch_error` and `model_access_error` are worker statuses of a
    workflow that was not tested. A reference-level workflow has no output to
    compare then: before the runner knew the two, it went on to compare."""
    for status in ("model_fetch_error", "model_access_error"):
        lane = _fetch_lane(tmp_path / status, monkeypatch, {
            "ref": {"cadence": "weekly"}, "smoke": {"level": "smoke", "cadence": "weekly"}})
        lane.status.update(ref=status, smoke=status)
        code, summary = lane.run("c1", "weekly", "--skip-publish")
        for cell in summary["workflows"].values():
            assert (cell["worker_status"], cell["verdict"]) == (status, "infra_error"), status
        # A weekly run with an untested workflow is red.
        assert code == 1 and summary["overall"] == "pass" and summary["has_infra_error"] is True


def test_the_runs_deadline_counts_the_time_a_job_may_fetch(tmp_path, monkeypatch):
    """A job of a lane that fetches can take its download longer. A run does
    not start one that could still be running at its deadline."""
    clock = Clock()
    monkeypatch.setattr(run_regression, "time", clock)
    week = {**LONG_LANES["lanes"]["week"], "model_storage": "job"}
    registry = {**LONG_LANES, "lanes": {**LONG_LANES["lanes"], "week": week}}
    # 600 s of timeout, 900 s to start, 1,200 s more in the queue: 45 minutes.
    # The 60 GB of `big` add 2,400 s: 85 minutes. 60 minutes are left.
    entries = {"big": {"models": [_model("a.safetensors", 60 * 10**9)]},
               "small": {"models": [_model("b.safetensors", 10**9)]}}
    gpu = Gpu({"big": "pass", "small": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, "--deadline-minutes",
                              "60", "--max-in-flight", "1", entries=entries, lane="week",
                              registry=registry)
    assert code == 1 and gpu.calls == ["small"]
    assert summary["workflows"]["big"]["worker_status"] == "not_reached"
    assert summary["workflows"]["big"]["error"] == (
        "not started: its job can take 85 min, and 60 min were left before the run's deadline")
    # The same workflows on a lane with its models on the volume: both fit.
    gpu = Gpu({"big": "pass", "small": "pass"})
    code, summary = _long_run(tmp_path, monkeypatch, RecordingStore(), gpu, "--deadline-minutes",
                              "60", "--max-in-flight", "1", entries=entries, lane="week")
    assert code == 0 and gpu.calls == ["big", "small"]
    worst = run_regression.job_worst_case_s
    assert worst({}, {"timeout_s": 600}, 1800, 1) == 2700
    assert worst({}, {"timeout_s": 600}, 1800, 1, 2400) == 5100
    assert worst({}, {"timeout_s": 600}, 600, 2, 100) == 3200


def test_the_run_refuses_a_number_of_streams_the_worker_would_not_take(tmp_path, monkeypatch,
                                                                      capsys):
    """The worker takes 1 to 16 and uses its own 4 for anything else, without
    a word. A run that asked for 32 would be read as a measurement of 32."""
    import argparse
    streams = run_regression._streams
    assert [streams(text) for text in ("1", "4", "16")] == [1, 4, 16]
    for text in ("0", "-1", "17", "32"):
        with pytest.raises(argparse.ArgumentTypeError):
            streams(text)
    lane = _fetch_lane(tmp_path, monkeypatch, {"wf": {"level": "smoke", "cadence": "weekly"}})
    with pytest.raises(SystemExit) as stopped:
        lane.run("c1", "weekly", "--skip-publish", "--fetch-streams", "32")
    assert stopped.value.code == 2 and lane.sent == {}  # before any job
    assert "--fetch-streams: must be at most 16" in capsys.readouterr().err


# -- a picture that did not arrive --------------------------------------------------------
# The worker sends output files up to a size limit, the smallest first, and
# lists the rest as not sent (src/handler.py).
FRAMES = {"wf_00001_.png": 10, "wf_00002_.png": 100, "wf_00003_.png": 200}


class FramesStore(ImageStore):
    """A golden and a previous run of three pictures, each of its own shade."""

    def _write(self, dest, shade):
        for name, own in FRAMES.items():
            _png(dest / name, own)
        return len(FRAMES)


def _frames_run(tmp_path, monkeypatch, not_sent=()):
    """process_workflow for an image workflow whose job wrote the three
    pictures of the reference, bit for bit, and did not send those named."""
    import types

    def job(client, wf_id, workflow, workdir=None, **kw):
        out = workdir / "outputs"
        for name, shade in FRAMES.items():
            if name not in not_sent:
                _png(out / name, shade)
        return {"status": "ok", "outputs_dir": str(out), "gpu_name": "gpu",
                "timings": {"prompt_exec_s": 3.0},
                "outputs": [{"filename": name, "bytes": 9, "sha256": name,
                             **({"truncated": True} if name in not_sent else {})}
                            for name in FRAMES]}

    monkeypatch.setattr(run_regression, "run_workflow_job", job)
    monkeypatch.setattr(run_regression, "RunPodClient", lambda endpoint_id: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "wf_api.json").write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(workdir=str(tmp_path / "work"), endpoint_id="ep", lane_id="main",
                                 commit="new", repo_url=None, branch="master", skip_publish=True)
    return run_regression.process_workflow("wf", {"workflow_path": "wf_api.json", "models": []},
                                           DEFAULTS, args, tmp_path, FramesStore())


def test_a_picture_that_was_not_sent_is_not_measured_against_another(tmp_path, monkeypatch):
    """The pictures of a run and of its reference are paired by their place in
    name order. With the smallest files sent first, the picture that stays
    behind can be the first: the two that arrived were measured against the
    reference's first two, and a run that equals its reference bit for bit
    was published as drift, with numbers and figures."""
    cell = _frames_run(tmp_path / "all", monkeypatch)
    assert cell["verdict"] == "pass" and cell["vs_golden"]["identical"] is True
    assert cell["vs_golden"]["frames_compared"] == 3 and list(cell) == CELL_KEYS

    said = "output not transferred, over the worker's inline size limit: wf_00001_.png"
    cell = _frames_run(tmp_path / "first", monkeypatch, not_sent=("wf_00001_.png",))
    assert cell["vs_golden"] == {"kind": "image", "error": said}
    assert cell["vs_previous"] == cell["vs_golden"]
    # It is still a failure, as a run without all its pictures always was. It
    # says why, and shows no figure of pictures that were never a pair.
    assert (cell["verdict"], cell["detail"], cell["figures"]) == ("fail", None, False)
    assert not (tmp_path / "first" / "work" / "wf" / "figures").exists()
    # The run is known by its first picture's sha256, which the worker listed.
    assert cell["output_sha256"] == "wf_00001_.png" and list(cell) == CELL_KEYS

    cell = _frames_run(tmp_path / "two", monkeypatch, not_sent=("wf_00003_.png", "wf_00002_.png"))
    assert cell["vs_golden"]["error"].endswith("size limit: wf_00002_.png, wf_00003_.png")
    assert cell["verdict"] == "fail"
    # A file of another kind that stayed behind is not the comparison's business.
    image = run_regression.COMPARATORS["image"]
    assert image.transfer_error({"outputs": [
        {"filename": "wf_00001_.png", "sha256": "p"},
        {"filename": "wf_mesh_00001_.glb", "sha256": "m", "truncated": True}]}) is None
