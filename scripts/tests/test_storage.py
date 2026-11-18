"""GitHubStorage against a local bare repo: sparse materialisation, derived
writers, and recovery from a concurrent publisher."""
import json
import subprocess

import pytest

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


def test_upload_json_compact_drops_whitespace_and_round_trips(tmp_path, remote):
    s = _store(tmp_path, remote, "a")
    obj = {"schema_version": 1, "entries": [{"c": "abc", "t": 1, "w": {"x": None}}]}
    s.upload_json("frontend/ci/main.json", obj, compact=True)
    s.upload_json("frontend/ci/pretty.json", obj)
    compact = (s.workdir / "frontend" / "ci" / "main.json").read_text(encoding="utf-8")
    assert compact == '{"schema_version":1,"entries":[{"c":"abc","t":1,"w":{"x":null}}]}'
    pretty = (s.workdir / "frontend" / "ci" / "pretty.json").read_text(encoding="utf-8")
    assert pretty.startswith('{\n  "schema_version": 1,') and json.loads(pretty) == obj
    assert s.download_json("frontend/ci/main.json") == obj
    s.finalize("compact")
    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    assert (check / "frontend" / "ci" / "main.json").read_text(encoding="utf-8") == compact


def test_checkpoint_pushes_the_start_marker_before_the_results(tmp_path, remote, capsys):
    """The marker lands on the remote on its own (derived writers wait for
    finalize), even behind a concurrent publisher, and finalize still pushes
    the rest of the run on top."""
    a = _store(tmp_path, remote, "a")
    other = _store(tmp_path, remote, "other")
    other.publish_summary("master", "ccc", {"commit": "ccc", "overall": "pass"})
    other.finalize("regression master@ccc: pass")  # a's first checkpoint push is rejected

    a.mark_attempt("master", "bbb", {"commit": "bbb", "attempt": 1})
    a.register_derived(_index_writer({"c": "bbb", "o": "pass"}))
    a.PUSH_ATTEMPTS = 3
    a.checkpoint("regression master@bbb: attempt 1 started")
    assert "pushed checkpoint" in capsys.readouterr().out

    def remote_file(path):
        r = subprocess.run(["git", "-C", str(remote), "cat-file", "-p", f"results:{path}"],
                           capture_output=True)
        return json.loads(r.stdout) if r.returncode == 0 else None

    assert remote_file("regression/runs/master/bbb/attempt.json") == {"commit": "bbb", "attempt": 1}
    assert remote_file("regression/runs/master/ccc/summary.json")["commit"] == "ccc"
    assert remote_file("regression/runs/master/bbb/summary.json") is None
    assert remote_file("regression/index/master/legacy.json") is None  # derived: not yet
    assert a.attempt_marker("master", "bbb") == {"commit": "bbb", "attempt": 1}

    a.publish_summary("master", "bbb", {"commit": "bbb", "overall": "pass"})
    a.finalize("regression master@bbb: pass")
    assert remote_file("regression/runs/master/bbb/summary.json")["overall"] == "pass"
    assert remote_file("regression/runs/master/bbb/attempt.json")["attempt"] == 1
    assert [e["c"] for e in remote_file("regression/index/master/legacy.json")["entries"]] == ["bbb"]


def test_a_checkpoint_with_nothing_written_pushes_nothing(tmp_path, remote, capsys):
    s = _store(tmp_path, remote, "a")
    s.checkpoint("nothing")
    assert "pushed" not in capsys.readouterr().out


def test_a_run_folder_is_published_without_the_files_it_holds_back(tmp_path, remote):
    """A smoke run keeps a large output file that did not repeat off the
    results branch. The file stays in the runner's folder, for the artifact."""
    run = tmp_path / "work" / "wf"
    (run / "outputs" / "thumbs").mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "outputs" / "wf_00001_.png").write_bytes(b"\x89PNG small")
    (run / "outputs" / "wf_mesh_00001_.glb").write_bytes(b"glTF large")
    (run / "outputs" / "thumbs" / "wf_00001_.webp").write_bytes(b"RIFF")
    s = _store(tmp_path, remote, "a")
    s.publish_workflow_run("master", "bbb", "wf", run, exclude=["outputs/wf_mesh_00001_.glb"])
    # The same folder under another commit, with nothing held back.
    s.publish_workflow_run("master", "ccc", "wf", run)
    s.finalize("regression master@bbb: pass")
    assert (run / "outputs" / "wf_mesh_00001_.glb").exists()
    listed = subprocess.run(["git", "-C", str(remote), "ls-tree", "-r", "--name-only", "results",
                             "regression/runs/master/bbb", "regression/runs/master/ccc"],
                            capture_output=True, text=True, check=True).stdout.split()
    assert listed == [
        "regression/runs/master/bbb/wf/outputs/thumbs/wf_00001_.webp",
        "regression/runs/master/bbb/wf/outputs/wf_00001_.png",
        "regression/runs/master/bbb/wf/run.json",
        "regression/runs/master/ccc/wf/outputs/thumbs/wf_00001_.webp",
        "regression/runs/master/ccc/wf/outputs/wf_00001_.png",
        "regression/runs/master/ccc/wf/outputs/wf_mesh_00001_.glb",
        "regression/runs/master/ccc/wf/run.json"]


def test_a_backend_that_predates_exclude_still_publishes_a_run_that_holds_nothing_back(tmp_path):
    """A run of reference-level workflows calls the backend the way it always
    did, so a store written before files could be held back keeps working."""
    calls = []

    class Old(storage.Storage):
        def upload_dir(self, blob_prefix, local_dir):
            calls.append((blob_prefix, local_dir))

    old = Old("regression/lanes/x")
    old.publish_workflow_run("master", "abc", "wf", tmp_path)
    old.publish_workflow_run("master", "abc", "wf", tmp_path, exclude=())
    assert calls == [("regression/lanes/x/runs/master/abc/wf", tmp_path)] * 2

    class New(storage.Storage):
        def upload_dir(self, blob_prefix, local_dir, exclude=()):
            calls.append((blob_prefix, local_dir, list(exclude)))

    New().publish_workflow_run("master", "abc", "wf", tmp_path, exclude=["outputs/a.glb"])
    assert calls[-1] == ("regression/runs/master/abc/wf", tmp_path, ["outputs/a.glb"])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # raised while the bucket client loads
def test_the_bucket_backend_holds_the_same_files_back(tmp_path, monkeypatch):
    import gcs_publish
    uploaded = []

    class Bucket:
        def bucket(self, name):
            return self

        def blob(self, name):
            self.name = name
            return self

        def upload_from_filename(self, path):
            uploaded.append(self.name)

    monkeypatch.setattr(gcs_publish, "client", Bucket)
    run = tmp_path / "wf"
    (run / "outputs").mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "outputs" / "wf_00001_.png").write_bytes(b"\x89PNG")
    (run / "outputs" / "wf_mesh_00001_.glb").write_bytes(b"glTF")
    store = storage.GcsStorage("bucket", "regression/lanes/x")
    base = "regression/lanes/x/runs/master/abc/wf"
    store.publish_workflow_run("master", "abc", "wf", run, exclude=["outputs/wf_mesh_00001_.glb"])
    assert uploaded == [f"{base}/outputs/wf_00001_.png", f"{base}/run.json"]
    uploaded.clear()
    store.publish_workflow_run("master", "abc", "wf", run)
    assert uploaded == [f"{base}/outputs/wf_00001_.png", f"{base}/outputs/wf_mesh_00001_.glb",
                        f"{base}/run.json"]
