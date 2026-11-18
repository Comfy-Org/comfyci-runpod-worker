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
                                      "--workflows", "sdxl_t2i"])
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
                                      "--workflows", "sdxl_t2i"])
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
