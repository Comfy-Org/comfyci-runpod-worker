"""golden_baseline: blessing records, history, promoting an existing run, and
the command-line flows that need no RunPod (bless-only, from-run)."""
import json
import subprocess
import sys

import pytest

import golden_baseline
import storage
from conftest import make_remote

RUN = {"status": "ok", "gpu_name": "cuda:0 NVIDIA GeForce RTX 4090", "torch_version": "2.11.0+cu128",
       "python_version": "3.12.3 (main)", "outputs": [{"filename": "flux_00001_.png", "sha256": "s2",
                                                       "bytes": 9}]}
COMMIT = "b" * 40
LANES = {"schema_version": 1, "primary": "legacy",
         "lanes": {"legacy": {"label": "legacy", "python": "3.12", "torch": "2.11.0", "cuda": "12.8",
                              "gpu": "RTX 4090", "cadence": "per-commit", "blocking": True,
                              "role": "primary", "layout": 1, "prefix": "regression"}}}


def _remote(tmp_path):
    return make_remote(tmp_path, {
        f"regression/runs/master/{COMMIT}/summary.json": {
            "commit": COMMIT, "run_ts": 1725235200, "overall": "fail",
            "workflows": {"flux": {"verdict": "fail", "golden_tag": "v0.33.2", "output_sha256": "s2",
                                   "vs_golden": {"identical": False, "mean_mse": 3.4},
                                   "vs_previous": {"identical": False}}}},
        f"regression/runs/master/{COMMIT}/flux/run.json": RUN,
        f"regression/runs/master/{COMMIT}/flux/outputs/flux_00001_.png": b"\x89PNG new",
        "regression/latest/master.json": {"commit": COMMIT},
        "regression/golden/flux/current.json": {"tag": "v0.33.2", "blessed_by": "x", "blessed_ts": 1},
        "regression/golden/flux/v0.33.2/blessed.json": {"tag": "v0.33.2", "commit": "a" * 40},
        "regression/golden/flux/v0.33.2/noise_floor.json": {"identical": True, "mean_mse": 0.0},
        "regression/golden/flux/v0.33.2/run_r1.json": {"outputs": [{"filename": "flux_00001_.png", "sha256": "s1"}]},
    })


def _manifest(tmp_path):
    root = tmp_path / "repo"
    (root / "manifest").mkdir(parents=True)
    (root / "manifest" / "workflows.json").write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [], "thresholds": {}},
        "workflows": {"flux": {"enabled": True, "workflow_path": "manifest/workflows/flux.json",
                               "models": []}}}))
    (root / "manifest" / "lanes.json").write_text(json.dumps(LANES))
    return root


def test_resolve_and_tag_helpers():
    sha = "c" * 40
    assert golden_baseline.resolve_ref("unused", sha) == sha
    assert golden_baseline.tag_name(sha) == "c" * 12
    assert golden_baseline.tag_name("v0.38.0") == "v0.38.0"
    assert golden_baseline.cuda_of("2.11.0+cu128") == "12.8"
    assert golden_baseline.cuda_of("2.14.1+cu130") == "13.0"
    assert golden_baseline.cuda_of(None) is None


def test_promote_run_then_bless_records_history(tmp_path):
    remote = _remote(tmp_path)
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    tag = golden_baseline.tag_name(COMMIT)
    assert golden_baseline.promote_run(store, "master", COMMIT, "flux", tag, "lane-x", "dev",
                                       "accept #16488 drift", tmp_path / "work")
    base = f"regression/golden/flux/{tag}"
    blessed = store.download_json(f"{base}/blessed.json")
    assert blessed["schema_version"] == 2 and blessed["source"] == "from_run"
    assert blessed["output_sha256"] == "s2" and blessed["cuda"] == "12.8"
    assert blessed["previous_tag"] == "v0.33.2" and blessed["lane"] == "lane-x"
    floor = store.download_json(f"{base}/noise_floor.json")
    assert floor["identical"] is True and floor["measured"] is False
    assert floor["inherited_from"] == "v0.33.2"
    assert (store.workdir / base / "outputs" / "flux_00001_.png").read_bytes() == b"\x89PNG new"
    assert golden_baseline.golden_exists(store, "flux", tag)
    assert not golden_baseline.golden_exists(store, "flux", "nope")

    current = golden_baseline.bless(store, "flux", tag, "dev", "accept #16488 drift")
    assert current["tag"] == tag and current["supersedes"] == "v0.33.2"
    assert current["output_sha256"] == "s2" and current["reason"] == "accept #16488 drift"
    history = store.download_json("regression/golden/flux/history.json")
    assert [b["tag"] for b in history["blesses"]] == [tag]

    # Blessing the same tag again keeps the original supersedes pointer.
    again = golden_baseline.bless(store, "flux", tag, "dev", None)
    assert again["supersedes"] == "v0.33.2"
    assert len(store.download_json("regression/golden/flux/history.json")["blesses"]) == 2


def test_promote_run_without_outputs_fails_cleanly(tmp_path, capsys):
    remote = _remote(tmp_path)
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    ok = golden_baseline.promote_run(store, "master", "f" * 40, "flux", "nope", "lane", "dev",
                                     None, tmp_path / "work")
    assert ok is False
    assert "no published outputs" in capsys.readouterr().out


def _run_main(tmp_path, remote, root, *extra):
    argv = ["golden_baseline.py", "--manifest", str(root / "manifest" / "workflows.json"),
            "--workdir", str(tmp_path / "work"), "--results-workdir", str(tmp_path / "wt"),
            "--blessed-by", "dev", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        # Point the lane registry at the test manifest and the store at the bare remote.
        golden_baseline.index_builder.LANES_FILE = root / "manifest" / "lanes.json"
        orig = storage.from_args

        def from_args(args):
            return storage.GitHubStorage(args.results_repo, args.results_branch,
                                         args.results_workdir, args.prefix, remote_url=str(remote))
        storage.from_args = from_args
        try:
            return golden_baseline.main()
        finally:
            storage.from_args = orig
    finally:
        sys.argv = old


def _clone(tmp_path, remote, name="check"):
    check = tmp_path / name
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    return check


def test_from_run_blesses_through_derived_writers_and_refreshes_index(tmp_path, capsys):
    remote = _remote(tmp_path)
    root = _manifest(tmp_path)
    rc = _run_main(tmp_path, remote, root, "--from-run", f"master/{COMMIT}", "--workflows", "flux",
                   "--bless", "--reason", 'accept "adaln" drift ($100)')
    assert rc == 0
    check = _clone(tmp_path, remote)
    tag = COMMIT[:12]
    current = json.loads((check / "regression/golden/flux/current.json").read_text())
    assert current["tag"] == tag and current["reason"] == 'accept "adaln" drift ($100)'
    history = json.loads((check / "regression/golden/flux/history.json").read_text())
    assert [b["tag"] for b in history["blesses"]] == [tag]
    assert not (check / f"regression/golden/flux/{tag}/run_r2.json").exists()
    lanes = json.loads((check / "regression/index/lanes.json").read_text())
    assert lanes["lanes"]["legacy"]["goldens"]["flux"] == {"tag": tag, "sha": "s2"}

    # Second attempt without --force refuses to overwrite the golden directory.
    rc = _run_main(tmp_path / "second", remote, root, "--from-run", f"master/{COMMIT}",
                   "--workflows", "flux", "--bless")
    assert rc == 1
    assert "already exists" in capsys.readouterr().out


def test_from_run_rejects_a_mismatching_ref_and_bad_shapes(tmp_path):
    remote = _remote(tmp_path)
    root = _manifest(tmp_path)
    with pytest.raises(SystemExit, match="does not resolve"):
        _run_main(tmp_path, remote, root, "--from-run", f"master/{COMMIT}", "--ref", "c" * 40,
                  "--workflows", "flux")
    with pytest.raises(SystemExit, match="expects BRANCH"):
        _run_main(tmp_path, remote, root, "--from-run", "master/short", "--workflows", "flux")
    with pytest.raises(SystemExit, match="--ref is required"):
        _run_main(tmp_path, remote, root, "--workflows", "flux")
    with pytest.raises(SystemExit, match="unknown lane"):
        _run_main(tmp_path, remote, root, "--ref", "v1", "--workflows", "flux", "--lane", "nope")


def test_bless_only_validates_every_workflow_first(tmp_path, capsys):
    remote = _remote(tmp_path)
    root = _manifest(tmp_path)
    rc = _run_main(tmp_path, remote, root, "--ref", "v0.38.0", "--workflows", "flux", "--bless-only")
    assert rc == 1
    assert "no generated golden v0.38.0 for: flux" in capsys.readouterr().out
    check = _clone(tmp_path, remote)
    assert json.loads((check / "regression/golden/flux/current.json").read_text())["tag"] == "v0.33.2"

    rc = _run_main(tmp_path / "ok", remote, root, "--ref", "v0.33.2", "--workflows", "flux",
                   "--bless-only", "--reason", "re-point")
    assert rc == 0
    check = _clone(tmp_path, remote, "check2")
    current = json.loads((check / "regression/golden/flux/current.json").read_text())
    assert current["tag"] == "v0.33.2" and current["reason"] == "re-point"
