"""run_regression: pure decision helpers and the summary handed to later steps."""
import run_regression

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


def test_summary_out_is_written_on_failure_with_prior_verdicts(tmp_path, monkeypatch):
    import json
    import sys
    import commit_meta
    import storage
    lane = "py312-torch2.11.0-cu128"
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
