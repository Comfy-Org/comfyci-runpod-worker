"""GitHubStorage against a local bare repo: sparse materialisation, derived
writers, and recovery from a concurrent publisher."""
import json
import subprocess

import storage


def _store(tmp_path, remote, name):
    return storage.GitHubStorage("local/test", "results", tmp_path / name,
                                 remote_url=str(remote))


def _index_writer(entry):
    def write(store):
        head = store.download_json("regression/index/master/legacy.json") or {"entries": []}
        entries = [e for e in head["entries"] if e["c"] != entry["c"]] + [entry]
        store.upload_json("regression/index/master/legacy.json",
                          {"entries": sorted(entries, key=lambda e: e["c"])})
    return write


def test_sparse_checkout_materialises_on_demand(tmp_path, remote):
    s = _store(tmp_path, remote, "a")
    assert s.latest_pointer("master") == {"commit": "aaa"}
    # Run outputs are outside the base cone until asked for.
    assert not (s.workdir / "regression" / "runs").exists()
    dest = tmp_path / "prev"
    assert s.fetch_run_outputs("master", "aaa", "wf", dest) == 1
    assert (dest / "wf_00001_.png").read_bytes() == b"\x89PNG seed"
    assert s.run_summary("master", "aaa")["overall"] == "pass"
    assert s.list_json("regression/runs/master", "summary.json") == [
        "regression/runs/master/aaa/summary.json"]
    assert s.read_blob_json("regression/runs/master/aaa/summary.json")["commit"] == "aaa"


def test_prefetch_batches_blobs_without_materialising(tmp_path, remote):
    s = _store(tmp_path, remote, "a")
    paths = ["regression/runs/master/aaa/summary.json",
             "regression/runs/master/aaa/wf/outputs/wf_00001_.png"]
    blobs = s.list_blobs("regression/runs/master")
    assert set(paths) <= set(blobs)
    s.prefetch(paths)
    # Objects are now local: reading must not need the remote.
    s.remote_url = str(tmp_path / "gone")
    s._git("remote", "set-url", "origin", str(tmp_path / "gone"))
    assert s.read_blob_json(paths[0])["commit"] == "aaa"
    assert not (s.workdir / "regression" / "runs").exists()


def test_concurrent_publishers_keep_each_others_derived_files(tmp_path, remote):
    a = _store(tmp_path, remote, "a")
    b = _store(tmp_path, remote, "b")
    for store, commit in ((a, "bbb"), (b, "ccc")):
        store.publish_summary("master", commit, {"commit": commit, "overall": "pass"})
        store.update_latest("master", {"commit": commit})
        store.register_derived(_index_writer({"c": commit, "o": "pass"}))
    a.finalize("regression master@bbb: pass")
    # b's first push is rejected (remote advanced); it must rebuild its index
    # on top of a's tree rather than restore its stale bytes.
    b.PUSH_ATTEMPTS = 3
    b.finalize("regression master@ccc: pass")

    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    head = json.loads((check / "regression" / "index" / "master" / "legacy.json").read_text())
    assert [e["c"] for e in head["entries"]] == ["bbb", "ccc"]
    for commit in ("aaa", "bbb", "ccc"):
        assert (check / "regression" / "runs" / "master" / commit / "summary.json").exists()
    assert json.loads((check / "regression" / "latest" / "master.json").read_text()) == {
        "commit": "ccc"}


def test_nothing_to_publish_is_a_noop(tmp_path, remote, capsys):
    s = _store(tmp_path, remote, "a")
    s.finalize("empty")
    assert "nothing new" in capsys.readouterr().out


def test_failing_derived_writer_never_blocks_primary_results(tmp_path, remote, capsys):
    s = _store(tmp_path, remote, "a")
    s.publish_summary("master", "bbb", {"commit": "bbb", "overall": "pass"})

    def broken(store):
        store.upload_json("regression/index/master/legacy.json", {"entries": ["partial"]})
        raise RuntimeError("boom")

    s.register_derived(broken)
    s.finalize("regression master@bbb: pass")
    out = capsys.readouterr().out
    assert "::warning::derived writer broken failed" in out and "pushed results" in out

    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    assert (check / "regression" / "runs" / "master" / "bbb" / "summary.json").exists()
    assert not (check / "regression" / "index").exists()  # partial output discarded


def test_fetch_failure_other_than_missing_branch_is_fatal(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="could not fetch"):
        storage.GitHubStorage("local/test", "results", tmp_path / "wt",
                              remote_url=str(tmp_path / "does-not-exist"))


def test_missing_results_branch_starts_empty(tmp_path):
    bare = tmp_path / "empty.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "results", str(bare)], check=True)
    s = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(bare))
    assert s.latest_pointer("master") is None
    s.update_latest("master", {"commit": "aaa"})
    s.finalize("first publish")
    assert subprocess.run(["git", "-C", str(bare), "rev-parse", "results"],
                          capture_output=True).returncode == 0
