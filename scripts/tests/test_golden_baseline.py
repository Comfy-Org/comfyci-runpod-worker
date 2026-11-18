"""golden_baseline: blessing records, history, promoting an existing run, and
the command-line flows that need no RunPod (bless-only, from-run)."""
import hashlib
import json
import subprocess
import sys

import pytest

import golden_baseline
import output_kinds
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
        lanes_file = golden_baseline.index_builder.LANES_FILE
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
            golden_baseline.index_builder.LANES_FILE = lanes_file
    finally:
        sys.argv = old


def _clone(tmp_path, remote, name="check"):
    check = tmp_path / name
    # The files as the branch holds them, whatever this machine's git does to line ends.
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", "-b", "results",
                    str(remote), str(check)], check=True)
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


def test_blessed_record_keeps_the_packs_a_golden_was_generated_with():
    packs = {"ComfyUI-KJNodes": {"sha": "e" * 40, "source": "cache", "import": "ok", "seconds": 1.2}}
    rec = golden_baseline.blessed_record("v1", COMMIT, {**RUN, "custom_nodes": packs}, "lane", "golden_run",
                                         "dev", None, None)
    assert rec["custom_nodes"] == packs
    assert golden_baseline.blessed_record("v1", COMMIT, RUN, "lane", "golden_run", "dev", None,
                                          None)["custom_nodes"] is None


def test_lane_filter_skips_workflows_restricted_to_other_lanes(tmp_path, capsys):
    """A workflow whose packs are only baked on another lane is skipped on this
    one, bless-only included, and nothing is left to do when all are skipped."""
    remote = _remote(tmp_path)
    root = _manifest(tmp_path)
    manifest = json.loads((root / "manifest" / "workflows.json").read_text())
    manifest["workflows"]["flux"]["lanes"] = ["elsewhere"]
    (root / "manifest" / "workflows.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="no workflows selected for lane legacy"):
        _run_main(tmp_path, remote, root, "--ref", "v0.33.2", "--workflows", "flux", "--bless-only")
    assert "[flux] not run on lane legacy (manifest lanes: elsewhere); skipped" in capsys.readouterr().out


def test_a_retired_lane_can_be_blessed_but_not_generated(tmp_path):
    """Generating dispatches GPU work, which a retired lane never gets;
    blessing and promoting only touch storage, so its goldens stay manageable."""
    remote = _remote(tmp_path)
    root = _manifest(tmp_path)
    retired = json.loads(json.dumps(LANES))
    retired["lanes"]["legacy"].update({"cadence": "retired", "role": "legacy",
                                       "endpoint_id": "ep-retired"})
    (root / "manifest" / "lanes.json").write_text(json.dumps(retired))
    with pytest.raises(SystemExit, match="is retired"):
        _run_main(tmp_path, remote, root, "--ref", "c" * 40, "--workflows", "flux")
    rc = _run_main(tmp_path / "bless", remote, root, "--ref", "v0.33.2", "--workflows", "flux",
                   "--bless-only", "--reason", "still readable")
    assert rc == 0


# -- text workflows -------------------------------------------------------------------
TEXT_LANES = {"schema_version": 1, "primary": "legacy",
              "lanes": {"legacy": {**LANES["lanes"]["legacy"], "endpoint_id": "ep-text"}}}
SAID = "The model wrote this line.\nAnd this second line."
OTHER = "The model wrote that line.\nAnd this second line."
TEXT_COMMIT = "d" * 40
TEXT_TAG = TEXT_COMMIT[:12]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_manifest(tmp_path, **entry):
    root = tmp_path / "repo"
    (root / "manifest" / "workflows").mkdir(parents=True, exist_ok=True)
    (root / "manifest" / "workflows" / "llm.json").write_text("{}", encoding="utf-8")
    (root / "manifest" / "workflows.json").write_text(json.dumps({
        "defaults": {"timeout_s": 600, "comfy_flags": [],
                     "thresholds": {"max_mean_mse": 2.0, "min_mean_psnr_db": 45.0,
                                    "max_pct_pixels_changed": 1.0}},
        "workflows": {"llm": {"enabled": True, "output_kind": "text",
                              "workflow_path": "manifest/workflows/llm.json", "models": [],
                              "thresholds": None, **entry}}}))
    (root / "manifest" / "lanes.json").write_text(json.dumps(TEXT_LANES))
    return root


def _text_jobs(monkeypatch, *texts):
    """RunPod jobs that write one text of `texts` each, in order; returns the calls made."""
    left, calls = list(texts), []

    def job(client, wf_id, workflow, workdir=None, **kw):
        calls.append(wf_id)
        out = workdir / "outputs"
        out.mkdir(parents=True)
        data = left.pop(0).encode("utf-8")
        (out / f"{wf_id}_00001.txt").write_bytes(data)
        return {"status": "ok", "torch_version": "2.14.1+cu130", "outputs_dir": str(out),
                "outputs": [{"filename": f"{wf_id}_00001.txt", "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()}]}

    monkeypatch.setattr(golden_baseline, "run_workflow_job", job)
    monkeypatch.setattr(golden_baseline, "RunPodClient", lambda endpoint_id: None)
    return calls


def _text_remote(tmp_path):
    return make_remote(tmp_path, {"regression/latest/master.json": {"commit": "a" * 40}})


def test_a_text_golden_whose_two_runs_agree_is_blessed(tmp_path, monkeypatch, capsys):
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path)
    # The second run differs in line ends only: the same text.
    calls = _text_jobs(monkeypatch, SAID + "\n", SAID.replace("\n", "\r\n"))
    rc = _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm", "--bless")
    assert rc == 0 and calls == ["llm", "llm"]
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    floor = json.loads((base / TEXT_TAG / "noise_floor.json").read_text())
    assert floor == {"kind": "text", "files_compared": 1, "file_count_ref": 1,
                     "file_count_cand": 1, "identical": True, "similarity": 1.0,
                     "similarity_basis": "words", "chars_ref": len(SAID), "chars_cand": len(SAID),
                     "lines_ref": 2, "lines_cand": 2, "measured": True}
    blessed = json.loads((base / TEXT_TAG / "blessed.json").read_text())
    # The sha is the text file's own, by the rule a run's cell follows; the kind comes last.
    assert blessed["output_sha256"] == _sha(SAID + "\n") and blessed["source"] == "golden_run"
    assert blessed["output_kind"] == "text" and list(blessed)[-1] == "output_kind"
    assert (base / TEXT_TAG / "outputs" / "llm_00001.txt").read_bytes() == (SAID + "\n").encode()
    assert (base / TEXT_TAG / "run_r2.json").exists()
    current = json.loads((base / "current.json").read_text())
    assert current["tag"] == TEXT_TAG and current["output_sha256"] == _sha(SAID + "\n")
    lanes = json.loads((base.parents[1] / "index" / "lanes.json").read_text())
    assert lanes["lanes"]["legacy"]["goldens"]["llm"] == {"tag": TEXT_TAG,
                                                         "sha": _sha(SAID + "\n")}
    assert "[llm] noise floor: identical" in capsys.readouterr().out


def test_a_text_golden_whose_two_runs_differ_is_not_blessed(tmp_path, monkeypatch, capsys):
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path)
    _text_jobs(monkeypatch, SAID, OTHER)
    rc = _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm", "--bless")
    out = capsys.readouterr().out
    assert rc == 1
    assert (f"::error::[llm] golden {TEXT_TAG} is not blessed: its two runs produced different "
            "text (similarity 0.888889), so an exact-match reference would fail at random: make "
            "the workflow deterministic, or set thresholds.min_text_similarity in its manifest "
            "entry") in out
    # What the model wrote never reaches the job log, not even as the noise floor.
    assert "model wrote" not in out and "second line" not in out
    assert '[llm] noise floor: {"kind": "text"' in out
    # The golden is uploaded all the same: the first run's text, and where the
    # second one parts from it. Nothing points at it.
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    floor = json.loads((base / TEXT_TAG / "noise_floor.json").read_text())
    assert floor["identical"] is False and floor["measured"] is True
    # "this" and "that" share two letters: the texts part at column 19.
    assert floor["first_diff"] == {"file": "llm_00001.txt", "line": 1, "col": 19, "offset": 18,
                                   "ref": "The model wrote this line.",
                                   "cand": "The model wrote that line."}
    assert (base / TEXT_TAG / "outputs" / "llm_00001.txt").exists()
    assert (base / TEXT_TAG / "blessed.json").exists() and not (base / "current.json").exists()

    # Approving it later is refused for the same reason.
    rc = _run_main(tmp_path / "later", remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm",
                   "--bless-only")
    assert rc == 1 and "is not blessed: its two runs produced different text" in \
        capsys.readouterr().out
    assert not (_clone(tmp_path, remote, "check2") / "regression/golden/llm/current.json").exists()


def test_a_similarity_floor_in_the_entry_lets_a_differing_text_golden_be_blessed(
        tmp_path, monkeypatch, capsys):
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path)
    _text_jobs(monkeypatch, SAID, OTHER)
    assert _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm") == 0
    # A limit the two runs do not reach opens nothing: the second run would fail.
    root = _text_manifest(tmp_path, thresholds={"min_text_similarity": 0.95})
    rc = _run_main(tmp_path / "high", remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm",
                   "--bless-only")
    assert rc == 1 and ("is not blessed: its two runs produced different text (similarity "
                        "0.888889), which does not pass thresholds.min_text_similarity 0.95 of "
                        "its manifest entry") in capsys.readouterr().out
    check = _clone(tmp_path, remote, "high-check")
    assert not (check / "regression/golden/llm/current.json").exists()
    root = _text_manifest(tmp_path, thresholds={"min_text_similarity": 0.7})
    rc = _run_main(tmp_path / "bless", remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm",
                   "--bless-only")
    assert rc == 0 and f"[llm] blessing golden {TEXT_TAG}" in capsys.readouterr().out
    current = _clone(tmp_path, remote) / "regression/golden/llm/current.json"
    assert json.loads(current.read_text())["tag"] == TEXT_TAG


def test_empty_text_and_text_that_could_not_be_compared_are_not_blessed(
        tmp_path, monkeypatch, capsys):
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path)
    _text_jobs(monkeypatch, "", "")
    rc = _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT, "--workflows", "llm", "--bless")
    assert rc == 1 and "is not blessed: the generated text is empty" in capsys.readouterr().out

    def no_file(client, wf_id, workflow, workdir=None, **kw):
        (workdir / "outputs").mkdir(parents=True)
        return {"status": "ok", "outputs_dir": str(workdir / "outputs"), "outputs": []}

    monkeypatch.setattr(golden_baseline, "run_workflow_job", no_file)
    rc = _run_main(tmp_path / "none", remote, root, "--ref", "e" * 40, "--workflows", "llm",
                   "--bless")
    assert rc == 1 and ("is not blessed: its two runs could not be compared (no text files to "
                        "compare (ref=0, cand=0))") in capsys.readouterr().out

    def one_not_sent(client, wf_id, workflow, workdir=None, **kw):
        """Both runs list a second text and send only the first: what arrived is the same."""
        (workdir / "outputs").mkdir(parents=True)
        (workdir / "outputs" / "llm_00001.txt").write_bytes(SAID.encode())
        return {"status": "ok", "outputs_dir": str(workdir / "outputs"), "outputs": [
            {"filename": "llm_00001.txt", "bytes": len(SAID), "sha256": _sha(SAID)},
            {"filename": "llm_thinking_00001.txt", "bytes": 9, "sha256": "t" * 64,
             "truncated": True}]}

    monkeypatch.setattr(golden_baseline, "run_workflow_job", one_not_sent)
    rc = _run_main(tmp_path / "lost", remote, root, "--ref", "f" * 40, "--workflows", "llm",
                   "--bless")
    assert rc == 1 and ("is not blessed: its two runs could not be compared (output not "
                        "transferred, over the worker's inline size limit: "
                        "llm_thinking_00001.txt)") in capsys.readouterr().out

    def empty_answer(client, wf_id, workflow, workdir=None, **kw):
        """Both runs write an empty answer next to a second text that is not empty."""
        (workdir / "outputs").mkdir(parents=True)
        (workdir / "outputs" / "llm_00001.txt").write_bytes(b"")
        (workdir / "outputs" / "llm_thinking_00001.txt").write_bytes(b"thoughts")
        return {"status": "ok", "outputs_dir": str(workdir / "outputs"), "outputs": [
            {"filename": "llm_00001.txt", "bytes": 0, "sha256": _sha("")},
            {"filename": "llm_thinking_00001.txt", "bytes": 8, "sha256": _sha("thoughts")}]}

    monkeypatch.setattr(golden_baseline, "run_workflow_job", empty_answer)
    rc = _run_main(tmp_path / "half", remote, root, "--ref", "9" * 40, "--workflows", "llm",
                   "--bless")
    out = capsys.readouterr().out
    assert rc == 1 and "[llm] noise floor: identical" in out
    assert f"::error::[llm] golden {'9' * 12} is not blessed: it has an empty text file" in out
    assert not (_clone(tmp_path, remote) / "regression/golden/llm/current.json").exists()


def test_a_golden_that_is_not_blessed_never_replaces_the_current_reference(
        tmp_path, monkeypatch, capsys):
    """Generating the current tag again (--force) writes into the directory
    current.json points at, the one every run is compared with."""
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path)
    _text_jobs(monkeypatch, SAID, SAID, OTHER, SAID, OTHER, OTHER)
    flags = ("--ref", TEXT_COMMIT, "--workflows", "llm", "--bless")
    assert _run_main(tmp_path, remote, root, *flags) == 0
    capsys.readouterr()
    rc = _run_main(tmp_path / "again", remote, root, *flags, "--force")
    out = capsys.readouterr().out
    assert rc == 1 and (f"::error::[llm] golden {TEXT_TAG} is not blessed: its two runs produced "
                        "different text (similarity 0.888889)") in out
    assert "It is the current reference, so its files are left as they are" in out
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert (base / TEXT_TAG / "outputs" / "llm_00001.txt").read_bytes() == SAID.encode()
    assert json.loads((base / TEXT_TAG / "noise_floor.json").read_text())["identical"] is True
    assert json.loads((base / "current.json").read_text())["output_sha256"] == _sha(SAID)
    # Two runs that agree replace it, as they always did.
    assert _run_main(tmp_path / "third", remote, root, *flags, "--force") == 0
    base = _clone(tmp_path, remote, "check2") / "regression/golden/llm"
    assert (base / TEXT_TAG / "outputs" / "llm_00001.txt").read_bytes() == OTHER.encode()
    assert json.loads((base / "current.json").read_text())["output_sha256"] == _sha(OTHER)


def _published_text_run(previous_floor, text=OTHER, name="llm_00001.txt"):
    run = {"status": "ok", "torch_version": "2.14.1+cu130",
           "outputs": [{"filename": name, "bytes": len(text), "sha256": _sha(text)}]}
    return {f"regression/runs/master/{COMMIT}/llm/run.json": run,
            f"regression/runs/master/{COMMIT}/llm/outputs/{name}": text.encode(),
            "regression/latest/master.json": {"commit": COMMIT},
            "regression/golden/llm/current.json": {"tag": "v1"},
            "regression/golden/llm/v1/noise_floor.json": previous_floor}


TEXT_FLOOR = {"kind": "text", "identical": True, "similarity": 1.0, "chars_ref": 40,
              "measured": True}
# What a promoted text says about itself: the gate reads these, not the earlier reference's.
OWN = {"files_compared": 1, "file_count_ref": 1, "file_count_cand": 1, "chars_ref": len(OTHER),
       "chars_cand": len(OTHER), "lines_ref": 2, "lines_cand": 2}


def test_promoting_a_text_run_takes_over_a_text_noise_floor_only(tmp_path):
    text = output_kinds.TextComparator()
    floor = TEXT_FLOOR
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt",
                                  remote_url=str(make_remote(tmp_path, _published_text_run(floor))))
    assert golden_baseline.promote_run(store, "master", COMMIT, "llm", "new", "lane", "dev", None,
                                       tmp_path / "work", text)
    assert store.download_json("regression/golden/llm/new/noise_floor.json") == {
        **floor, **OWN, "inherited_from": "v1", "measured": False}
    blessed = store.download_json("regression/golden/llm/new/blessed.json")
    assert blessed["output_sha256"] == _sha(OTHER) and blessed["output_kind"] == "text"
    promoted = store.workdir / "regression/golden/llm/new/outputs/llm_00001.txt"
    assert output_kinds.read_text(promoted) == OTHER  # a checkout may rewrite line ends

    # The earlier reference was an image: its floor says nothing about a text.
    image = {"identical": False, "mean_mse": 0.4, "mean_psnr_db": 52.1}
    other = tmp_path / "image"
    other.mkdir()
    store = storage.GitHubStorage("local/test", "results", other / "wt",
                                  remote_url=str(make_remote(other, _published_text_run(image))))
    assert golden_baseline.promote_run(store, "master", COMMIT, "llm", "new", "lane", "dev", None,
                                       other / "work", text)
    assert store.download_json("regression/golden/llm/new/noise_floor.json") == {
        "kind": "text", "identical": True, "similarity": 1.0, **OWN, "inherited_from": "v1",
        "measured": False}


def test_from_run_blesses_a_text_run_under_its_text_sha(tmp_path, capsys):
    """Through the command line: main() hands the workflow's comparator on."""
    remote = make_remote(tmp_path, _published_text_run(TEXT_FLOOR))
    root = _text_manifest(tmp_path)
    rc = _run_main(tmp_path, remote, root, "--from-run", f"master/{COMMIT}", "--workflows", "llm",
                   "--bless")
    assert rc == 0 and f"[llm] blessing golden {COMMIT[:12]}" in capsys.readouterr().out
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    blessed = json.loads((base / COMMIT[:12] / "blessed.json").read_text())
    assert (blessed["output_kind"], blessed["source"]) == ("text", "from_run")
    assert blessed["output_sha256"] == _sha(OTHER)
    assert json.loads((base / COMMIT[:12] / "noise_floor.json").read_text()) == {
        **TEXT_FLOOR, **OWN, "inherited_from": "v1", "measured": False}
    current = json.loads((base / "current.json").read_text())
    assert current["tag"] == COMMIT[:12] and current["output_sha256"] == _sha(OTHER)
    lanes = json.loads((base.parents[1] / "index" / "lanes.json").read_text())
    assert lanes["lanes"]["legacy"]["goldens"]["llm"] == {"tag": COMMIT[:12], "sha": _sha(OTHER)}


def test_from_run_does_not_bless_an_empty_text_or_a_run_without_a_text(tmp_path, capsys):
    """The gate judges the text that is promoted: the earlier reference's
    noise floor (40 characters) says nothing about it."""
    root = _text_manifest(tmp_path)
    flags = ("--from-run", f"master/{COMMIT}", "--workflows", "llm", "--bless")
    remote = make_remote(tmp_path, _published_text_run(TEXT_FLOOR, text=""))
    assert _run_main(tmp_path, remote, root, *flags) == 1
    assert (f"::error::[llm] golden {COMMIT[:12]} is not blessed: the generated text is empty"
            in capsys.readouterr().out)
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert json.loads((base / "current.json").read_text()) == {"tag": "v1"}
    assert json.loads((base / COMMIT[:12] / "noise_floor.json").read_text())["chars_ref"] == 0
    # Approving it later is refused as well.
    assert _run_main(tmp_path / "later", remote, root, "--ref", COMMIT, "--workflows", "llm",
                     "--bless-only") == 1
    assert "is not blessed: the generated text is empty" in capsys.readouterr().out

    other = tmp_path / "png"
    other.mkdir()
    remote = make_remote(other, _published_text_run(TEXT_FLOOR, name="llm_00001_.png"))
    assert _run_main(other, remote, root, *flags) == 1
    assert (f"::error::[llm] master/{COMMIT} cannot be a reference: no text files to compare "
            "(ref=0, cand=0)") in capsys.readouterr().out
    assert not (_clone(other, remote) / "regression/golden/llm" / COMMIT[:12]).exists()


def test_a_promoted_text_that_is_not_blessed_never_replaces_the_current_reference(
        tmp_path, capsys):
    """Promoting onto the tag that is current (--force) writes into the
    directory current.json points at, like generating that tag again."""
    tag = COMMIT[:12]
    remote = make_remote(tmp_path, {
        **_published_text_run(TEXT_FLOOR, text=""),
        "regression/golden/llm/current.json": {"tag": tag, "output_sha256": _sha(SAID)},
        f"regression/golden/llm/{tag}/blessed.json": {"tag": tag},
        f"regression/golden/llm/{tag}/noise_floor.json": TEXT_FLOOR,
        f"regression/golden/llm/{tag}/outputs/llm_00001.txt": SAID.encode()})
    root = _text_manifest(tmp_path)
    flags = ("--from-run", f"master/{COMMIT}", "--workflows", "llm", "--force")
    for bless in ((), ("--bless",)):
        assert _run_main(tmp_path / f"run{len(bless)}", remote, root, *flags, *bless) == 1
        assert (f"::error::[llm] golden {tag} is not blessed: the generated text is empty. It is "
                "the current reference, so its files are left as they are"
                ) in capsys.readouterr().out
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert (base / tag / "outputs" / "llm_00001.txt").read_bytes() == SAID.encode()
    assert json.loads((base / tag / "noise_floor.json").read_text()) == TEXT_FLOOR
    assert json.loads((base / tag / "blessed.json").read_text()) == {"tag": tag}
    # A text that may be blessed replaces it, as it always did.
    other = tmp_path / "fine"
    other.mkdir()
    remote = make_remote(other, {
        **_published_text_run(TEXT_FLOOR),
        "regression/golden/llm/current.json": {"tag": tag, "output_sha256": _sha(SAID)},
        f"regression/golden/llm/{tag}/blessed.json": {"tag": tag},
        f"regression/golden/llm/{tag}/noise_floor.json": TEXT_FLOOR,
        f"regression/golden/llm/{tag}/outputs/llm_00001.txt": SAID.encode()})
    assert _run_main(other, remote, root, *flags, "--bless") == 0
    base = _clone(other, remote) / "regression/golden/llm"
    assert (base / tag / "outputs" / "llm_00001.txt").read_bytes() == OTHER.encode()
    assert json.loads((base / "current.json").read_text())["output_sha256"] == _sha(OTHER)


def test_from_run_does_not_promote_a_run_that_did_not_send_every_text(tmp_path, capsys):
    """Generating refuses outputs with a text that the worker listed and did
    not send. A promoted run is held to the same: its reference would lack
    that file, and no later run could pass against it."""
    files = _published_text_run(TEXT_FLOOR)
    files[f"regression/runs/master/{COMMIT}/llm/run.json"]["outputs"].append(
        {"filename": "llm_thinking_00001.txt", "bytes": 9, "sha256": "t" * 64, "truncated": True})
    remote = make_remote(tmp_path, files)
    root = _text_manifest(tmp_path)
    rc = _run_main(tmp_path, remote, root, "--from-run", f"master/{COMMIT}", "--workflows", "llm",
                   "--bless")
    assert rc == 1
    assert (f"::error::[llm] master/{COMMIT} cannot be a reference: output not transferred, over "
            "the worker's inline size limit: llm_thinking_00001.txt") in capsys.readouterr().out
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert not (base / COMMIT[:12]).exists()
    assert json.loads((base / "current.json").read_text()) == {"tag": "v1"}


def test_a_text_golden_is_not_written_over_one_that_holds_another_text_file(
        tmp_path, monkeypatch, capsys):
    """--force adds files to golden/<id>/<tag>/outputs and removes none. A text
    file that only the earlier golden has would stay in the reference: the gate
    reads the new outputs and passes, and every later run fails on the file count."""
    tag = COMMIT[:12]
    remote = make_remote(tmp_path, {
        **_published_text_run(TEXT_FLOOR),
        "regression/golden/llm/current.json": {"tag": tag, "output_sha256": _sha(SAID)},
        f"regression/golden/llm/{tag}/blessed.json": {"tag": tag},
        f"regression/golden/llm/{tag}/noise_floor.json": TEXT_FLOOR,
        f"regression/golden/llm/{tag}/outputs/llm_00001.txt": SAID.encode(),
        f"regression/golden/llm/{tag}/outputs/llm_thinking_00001.txt": b"thoughts"})
    root = _text_manifest(tmp_path)
    calls = _text_jobs(monkeypatch, OTHER, OTHER)
    refused = (f"::error::[llm] golden {tag} holds llm_thinking_00001.txt, which the new outputs "
               "do not have, and writing over a golden removes no file. Remove it from "
               f"golden/llm/{tag}/outputs on the results branch, or use another ref")
    for name, flags in (("promote", ("--from-run", f"master/{COMMIT}")), ("generate", ())):
        rc = _run_main(tmp_path / name, remote, root, *flags, "--ref", COMMIT, "--workflows",
                       "llm", "--force", "--bless")
        assert rc == 1 and refused in capsys.readouterr().out, name
    assert calls == ["llm", "llm"]
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert (base / tag / "outputs" / "llm_00001.txt").read_bytes() == SAID.encode()
    assert json.loads((base / tag / "blessed.json").read_text()) == {"tag": tag}
    assert json.loads((base / "current.json").read_text())["output_sha256"] == _sha(SAID)
    # The same text file, next to a file of another kind: written as it always was.
    _text_jobs(monkeypatch, OTHER, OTHER)
    other = tmp_path / "same"
    other.mkdir()
    remote = make_remote(other, {
        **_published_text_run(TEXT_FLOOR),
        f"regression/golden/llm/{tag}/blessed.json": {"tag": tag},
        f"regression/golden/llm/{tag}/outputs/llm_00001.txt": SAID.encode(),
        f"regression/golden/llm/{tag}/outputs/llm_00001_.png": b"\x89PNG of an earlier kind"})
    assert _run_main(other, remote, root, "--ref", COMMIT, "--workflows", "llm", "--force") == 0
    base = _clone(other, remote) / "regression/golden/llm"
    assert (base / tag / "outputs" / "llm_00001.txt").read_bytes() == OTHER.encode()
    # An image golden too: the outputs of the earlier one are not even read.
    image = tmp_path / "image"
    image.mkdir()
    store = storage.GitHubStorage("local/test", "results", image / "wt",
                                  remote_url=str(_remote(image)))
    monkeypatch.setattr(store, "fetch_golden_outputs",
                        lambda *a: pytest.fail("not read for an image golden"))
    assert golden_baseline.promote_run(store, "master", COMMIT, "flux", "v0.33.2", "lane", "dev",
                                       None, image / "work")


def test_bless_only_refuses_a_golden_of_another_output_kind(tmp_path, capsys):
    """The id was an image workflow before, or the wrong tag was named: the
    golden holds a PNG and an image noise floor. An identical image floor is
    no text floor, and no run of a text workflow could pass against a PNG."""
    remote = make_remote(tmp_path, {
        "regression/latest/master.json": {"commit": "a" * 40},
        "regression/golden/llm/v1/blessed.json": {"tag": "v1", "output_sha256": "p"},
        "regression/golden/llm/v1/noise_floor.json": {"identical": True, "mean_mse": 0.0,
                                                      "measured": True},
        "regression/golden/llm/v1/outputs/llm_00001_.png": b"\x89PNG"})
    root = _text_manifest(tmp_path)
    rc = _run_main(tmp_path, remote, root, "--ref", "v1", "--workflows", "llm", "--bless-only")
    assert rc == 1 and ("::error::[llm] golden v1 is not blessed: its noise floor is not a text "
                        "comparison, so it is a golden of another kind") in capsys.readouterr().out
    assert not (_clone(tmp_path, remote) / "regression/golden/llm/current.json").exists()


def test_an_image_golden_names_no_kind():
    """Absent means image: an image golden keeps the keys it always had, in order."""
    rec = golden_baseline.blessed_record("v1", COMMIT, RUN, "lane", "golden_run", "dev", None, None)
    assert list(rec) == ["schema_version", "tag", "commit", "source", "lane", "gpu_name",
                         "torch_version", "python_version", "cuda", "driver_version", "image_sha",
                         "output_sha256", "custom_nodes", "reason", "previous_tag", "generated_by",
                         "generated_ts"]
    assert rec["output_sha256"] == "s2"
    text = golden_baseline.blessed_record("v1", COMMIT, RUN, "lane", "golden_run", "dev", None,
                                          None, output_kinds.TextComparator())
    assert list(text) == [*rec, "output_kind"] and text["output_sha256"] is None


def test_an_unknown_output_kind_stops_golden_generation_before_any_job(tmp_path, monkeypatch):
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path, output_kind="video")
    calls = _text_jobs(monkeypatch, SAID, SAID)
    for flags in (("--ref", TEXT_COMMIT), ("--ref", TEXT_COMMIT, "--bless-only"),
                  ("--from-run", f"master/{COMMIT}")):
        with pytest.raises(SystemExit, match=r"\[llm\] unknown output_kind 'video' \(known: "
                                             r"image, text, audio\); see manifest/workflows.json"):
            _run_main(tmp_path, remote, root, *flags, "--workflows", "llm")
    assert calls == []


# -- check levels -----------------------------------------------------------------------
ALL_FLOWS = (("--ref", TEXT_COMMIT), ("--ref", TEXT_COMMIT, "--bless-only"),
             ("--from-run", f"master/{COMMIT}", "--bless"))


def test_a_reference_level_audio_workflow_stops_before_any_job(tmp_path, monkeypatch):
    """Audio is a known kind with no comparator yet: a clean stop, no traceback."""
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path, output_kind="audio")
    calls = _text_jobs(monkeypatch, SAID, SAID)
    for flags in ALL_FLOWS:
        with pytest.raises(SystemExit, match=r"^\[llm\] no comparator for output_kind 'audio' "
                                             r'yet: such a workflow can only run with "level": '
                                             r'"smoke"; see manifest/workflows.json$'):
            _run_main(tmp_path, remote, root, *flags, "--workflows", "llm")
    assert calls == []


def test_a_named_smoke_workflow_is_refused_and_all_leaves_it_out(tmp_path, monkeypatch, capsys):
    """A smoke-level workflow is not compared: there is no golden to generate,
    to promote or to bless for it."""
    remote, root = _text_remote(tmp_path), _text_manifest(tmp_path, level="smoke")
    calls = _text_jobs(monkeypatch, SAID, SAID, SAID, SAID)
    for flags in ALL_FLOWS:
        with pytest.raises(SystemExit,
                           match="^llm is a smoke-level workflow: it has no reference$"):
            _run_main(tmp_path, remote, root, *flags, "--workflows", "llm")
    skipped = "::notice::[llm] is a smoke-level workflow: it has no reference; skipped"
    with pytest.raises(SystemExit, match="no workflows selected for lane legacy"):
        _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT)
    assert skipped in capsys.readouterr().out and calls == []

    # Next to a reference-level workflow: `all` generates that one only.
    path = root / "manifest" / "workflows.json"
    manifest = json.loads(path.read_text())
    manifest["workflows"]["ref"] = {**manifest["workflows"]["llm"], "level": "reference"}
    (root / "manifest" / "workflows" / "ref.json").write_text("{}", encoding="utf-8")
    path.write_text(json.dumps(manifest))
    assert _run_main(tmp_path, remote, root, "--ref", TEXT_COMMIT, "--bless") == 0
    assert skipped in capsys.readouterr().out and calls == ["ref", "ref"]
    goldens = _clone(tmp_path, remote) / "regression/golden"
    assert (goldens / "ref" / "current.json").exists() and not (goldens / "llm").exists()
    with pytest.raises(SystemExit, match="^llm is a smoke-level workflow: it has no reference$"):
        _run_main(tmp_path / "both", remote, root, "--ref", "e" * 40, "--workflows", "ref,llm")
    assert calls == ["ref", "ref"]

    # A level that is not known is a mistake in the manifest, as an unknown kind is.
    manifest["workflows"]["llm"]["level"] = "main"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match=r"^\[llm\] unknown level 'main' \(known: reference, "
                                         r"smoke\); see manifest/workflows.json$"):
        _run_main(tmp_path / "odd", remote, root, "--ref", "e" * 40)


def _smoke_comparison(*files):
    return {"verdict": "pass", "detail": None, "vs_golden": None, "vs_previous": None,
            "thresholds_used": None, "golden_tag": None, "previous_commit": None,
            "previous_verdict": None, "previous_golden_tag": None, "figures": False,
            "output_kind": "text", "level": "smoke",
            "smoke": {"files": list(files), "problems": [], "notes": []}}


def test_from_run_refuses_a_smoke_run_that_held_a_file_back(tmp_path, capsys):
    """Smoke to reference is one manifest edit and one approval of a published
    smoke run. A run that did not publish every output file is not whole on
    the results branch: the reference would lack what it held back."""
    said = {"name": "llm_00001.txt", "type": "text", "bytes": len(OTHER), "sha256": _sha(OTHER),
            "chars": 40, "words": 8}
    clip = {"name": "llm_video_00001_.mp4", "type": "video", "bytes": 3000000, "sha256": "v" * 64,
            "published": False, "width": 640, "height": 352, "frames": 25, "seconds": 1.0,
            "bytes_per_frame_mp": 14280}
    mesh = {"name": "llm_mesh_00001_.glb", "type": "mesh", "bytes": 9000000, "sha256": "m" * 64,
            "sent": False, "published": False}
    assert golden_baseline.unpublished_files(_smoke_comparison(said, clip, mesh)) == [
        "llm_video_00001_.mp4", "llm_mesh_00001_.glb"]
    assert golden_baseline.unpublished_files(_smoke_comparison(said)) == []
    for none in (None, {}, {"verdict": "pass", "vs_golden": {"identical": True}},
                 {"smoke": None}, {"smoke": {"files": "x"}}):
        assert golden_baseline.unpublished_files(none) == []

    root = _text_manifest(tmp_path)  # the entry is at reference level now
    flags = ("--from-run", f"master/{COMMIT}", "--workflows", "llm", "--bless")
    files = _published_text_run(TEXT_FLOOR)
    files[f"regression/runs/master/{COMMIT}/llm/comparison.json"] = _smoke_comparison(
        said, clip, mesh)
    remote = make_remote(tmp_path, files)
    assert _run_main(tmp_path, remote, root, *flags) == 1
    assert (f"::error::[llm] master/{COMMIT} cannot be a reference: it was a smoke run that did "
            "not publish llm_video_00001_.mp4, llm_mesh_00001_.glb. Use a later run that shows "
            "the same output: that one publishes every file") in capsys.readouterr().out
    base = _clone(tmp_path, remote) / "regression/golden/llm"
    assert not (base / COMMIT[:12]).exists()
    assert json.loads((base / "current.json").read_text()) == {"tag": "v1"}

    # A smoke run that published every file is approved like any other run.
    other = tmp_path / "whole"
    other.mkdir()
    files[f"regression/runs/master/{COMMIT}/llm/comparison.json"] = _smoke_comparison(said)
    remote = make_remote(other, files)
    assert _run_main(other, remote, root, *flags) == 0
    current = _clone(other, remote) / "regression/golden/llm/current.json"
    assert json.loads(current.read_text())["tag"] == COMMIT[:12]
