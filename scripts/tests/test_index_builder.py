"""index_builder: entry shape, drift classification, first-bad detection,
merging, and the backfill + incremental upsert against a results checkout."""
import json
import subprocess

import commit_meta
import index_builder
import storage
from conftest import make_remote

LANE = "py312-torch2.11.0-cu128"
LANES = {"schema_version": 1, "primary": LANE,
         "lanes": {LANE: {"label": "legacy", "python": "3.12", "torch": "2.11.0",
                          "cuda": "12.8", "gpu": "RTX 4090", "cadence": "per-commit",
                          "blocking": True, "role": "primary", "layout": 1,
                          "prefix": "regression"}}}
IDENT = {"identical": True, "mean_mse": 0.0, "mean_psnr_db": None,
         "mean_pct_pixels_changed": 0.0}
DIFF = {"identical": False, "mean_mse": 3.442175, "mean_psnr_db": 42.76,
        "mean_pct_pixels_changed": 69.986}


def _wf(verdict, vs_golden, vs_previous, prev, exec_s=100.0, sha=None, detail=None,
        golden="v0.33.2"):
    d = {"workflow_id": "flux", "worker_status": "ok", "verdict": verdict,
         "vs_golden": vs_golden, "vs_previous": vs_previous, "golden_tag": golden,
         "previous_commit": prev, "timings": {"prompt_exec_s": exec_s},
         "vram_peak_mb": 21500.0, "comfy_version": "0.38.0"}
    if sha:
        d["output_sha256"] = sha
    if detail:
        d["detail"] = detail
    return d


def _summary(commit, ts, verdict, vs_golden, vs_previous, prev, **kw):
    return {"branch": "master", "commit": commit, "run_ts": ts, "overall": verdict,
            "workflows": {"flux": _wf(verdict, vs_golden, vs_previous, prev, **kw)}}


def _entry(c, t, v, pr=None, ct=None, sha=None, wf="flux"):
    return {"c": c, "t": t, "o": v if v in ("pass", "fail") else "pass",
            "m": {"pr": pr, "a": "dev", "ct": ct, "s": "x"} if (pr or ct) else None,
            "w": {wf: {"v": v, "g": "v0.33.2", "sha": sha}}}


def test_helpers():
    assert commit_meta.pr_number("Fix thing (#1234)") == 1234
    assert commit_meta.pr_number("Fix thing (#1234) ") == 1234
    assert commit_meta.pr_number("No pr here") is None
    assert commit_meta.pr_number(None) is None
    assert index_builder.lane_for_prefix(LANES, "regression") == LANE
    assert index_builder.lane_for_prefix(LANES, "regression/lanes/x") is None
    assert index_builder.month_of(1791142918) == "2026-10"
    assert index_builder.sort_ts({"t": 5, "m": {"ct": 3}}) == 3
    assert index_builder.sort_ts({"t": 5, "m": None}) == 5


def test_entry_classifies_legacy_summaries_from_previous_entry():
    prev_entry = {"c": "a" * 40, "w": {"flux": {"v": "fail", "g": "v0.33.2"}}}
    s = _summary("b" * 40, 1700000100, "fail", DIFF, IDENT, "a" * 40)
    e = index_builder.entry_from_summary(s, LANE, 1, prev_entry=prev_entry,
                                         output_shas={"flux": "deadbeef"})
    assert e["w"]["flux"]["d"] == "inherited"
    assert e["w"]["flux"]["sha"] == "deadbeef"
    assert e["w"]["flux"]["pv"] is True and e["w"]["flux"]["mse"] == 3.442175
    assert e["prev"] == "a" * 40 and e["layout"] == 1 and e["lane"] == LANE

    # Same outputs as a failure against a different golden: not inherited.
    other_golden = {"c": "a" * 40, "w": {"flux": {"v": "fail", "g": "v0.30.0"}}}
    assert index_builder.entry_from_summary(s, LANE, 1, prev_entry=other_golden)["w"]["flux"]["d"] is None

    s2 = _summary("c" * 40, 1700000200, "fail", DIFF, DIFF, "b" * 40)
    e2 = index_builder.entry_from_summary(s2, LANE, 1, prev_entry={"w": {"flux": {"v": "pass"}}})
    assert e2["w"]["flux"]["d"] == "new_drift"

    s3 = _summary("d" * 40, 1700000300, "fail", DIFF, IDENT, "c" * 40, detail="inherited",
                  sha="feed")
    e3 = index_builder.entry_from_summary(s3, LANE, 1, meta={"subject": "x (#12)", "author": "me",
                                                             "pr": 12, "committed_ts": 1},
                                          tested_range={"commits_between": 3, "compare_url": "u"})
    assert e3["w"]["flux"]["d"] == "inherited" and e3["w"]["flux"]["sha"] == "feed"
    assert e3["m"] == {"s": "x (#12)", "a": "me", "pr": 12, "ct": 1}
    assert e3["range"] == {"n": 3, "url": "u"}

    # A previous comparison that errored is no evidence either way.
    s4 = _summary("e" * 40, 1700000400, "fail", DIFF, {"error": "shape mismatch"}, "d" * 40)
    assert index_builder.entry_from_summary(s4, LANE, 1, prev_entry=prev_entry)["w"]["flux"]["d"] is None


def test_first_bad_finds_the_start_of_the_open_failing_chain():
    newest_first = [_entry("f", 6, "fail"), _entry("e", 5, "infra_error"), _entry("d", 4, "fail"),
                    _entry("c", 3, "fail", pr=16488), _entry("b", 2, "pass"), _entry("a", 1, "fail")]
    fb = index_builder.first_bad(newest_first)
    assert fb == {"flux": {"commit": "c", "run_ts": 3, "kind": "fail", "prev_good": "b",
                           "pr": 16488, "author": "dev", "golden": "v0.33.2", "runs": 3}}
    assert index_builder.first_bad([_entry("b", 2, "pass"), _entry("a", 1, "fail")]) == {}
    # A crash opens a chain too, and the chain records what opened it.
    crash = index_builder.first_bad([_entry("c", 3, "fail"), _entry("b", 2, "execution_error"),
                                     _entry("a", 1, "pass")])
    assert crash["flux"]["commit"] == "b" and crash["flux"]["kind"] == "execution_error"
    assert crash["flux"]["runs"] == 2


def test_first_bad_walks_commit_time_not_run_time():
    # 'old' was re-tested on demand after 'new' (bisecting): it still precedes it.
    entries = [_entry("old", 90, "pass", ct=10), _entry("new", 50, "fail", ct=20)]
    assert index_builder.first_bad(entries)["flux"] == {
        "commit": "new", "run_ts": 50, "kind": "fail", "prev_good": "old", "pr": None,
        "author": "dev", "golden": "v0.33.2", "runs": 1}


def test_prior_verdicts_look_through_infra_errors_and_see_execution_errors():
    entries = [_entry("e", 5, "fail"), _entry("d", 4, "infra_error"),
               _entry("c", 3, "execution_error"), _entry("b", 2, "pass")]
    # Re-running 'e' itself: its own entry is not its prior.
    assert index_builder.prior_verdicts(entries, "e") == {
        "flux": {"commit": "c", "verdict": "execution_error"}}
    assert index_builder.prior_verdicts(entries, "f")["flux"]["commit"] == "e"
    assert index_builder.prior_verdicts([_entry("a", 1, "no_baseline")], "b") == {}
    assert index_builder.prior_verdicts([], "a") == {}


def test_prior_verdicts_skip_newer_commits_on_a_retest():
    entries = [_entry("new", 50, "execution_error", ct=20), _entry("old", 40, "pass", ct=10)]
    assert index_builder.prior_verdicts(entries, "mid", committed_ts=15)["flux"]["commit"] == "old"
    assert index_builder.prior_verdicts(entries, "next", committed_ts=30)["flux"]["commit"] == "new"


def test_merge_entry_keeps_other_workflows_on_partial_rerun():
    old = {"c": "x", "t": 1, "o": "fail", "infra": False, "m": {"s": "x"}, "prev": "p",
           "w": {"flux": {"v": "fail"}, "sdxl": {"v": "pass"}}}
    new = {"c": "x", "t": 2, "o": "pass", "infra": False, "m": None, "prev": None,
           "w": {"flux": {"v": "pass"}}}
    merged = index_builder.merge_entry(old, new)
    assert merged["w"] == {"flux": {"v": "pass"}, "sdxl": {"v": "pass"}}
    assert merged["o"] == "pass" and merged["t"] == 2
    assert merged["m"] == {"s": "x"} and merged["prev"] == "p"  # nulls never erase
    assert index_builder.merge_entry(None, new) is new


def test_entry_carries_the_actions_run_id():
    """History rows link to the run that tested each commit. Summaries from
    before the id was recorded give None; an id that is not an int is dropped."""
    s = _summary("a" * 40, 1700000100, "pass", IDENT, None, None)
    assert index_builder.entry_from_summary(s, LANE, 1)["run"] is None
    assert index_builder.entry_from_summary({**s, "run_id": 43658104127}, LANE, 1)["run"] == (
        43658104127)
    for bad in ("43658104127", None, True, 1.5):
        assert index_builder.entry_from_summary({**s, "run_id": bad}, LANE, 1)["run"] is None


def test_entry_names_the_repository_the_run_was_in():
    """The action also runs from other repositories (ComfyUI's own workflow):
    an id alone would link a history row into the wrong one."""
    s = {**_summary("a" * 40, 1700000100, "pass", IDENT, None, None), "run_id": 555}
    worker = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs/555"
    assert index_builder.entry_from_summary({**s, "run_url": worker}, LANE, 1)["run_repo"] == (
        "Comfy-Org/comfyci-runpod-worker")
    comfy = "https://github.com/Comfy-Org/ComfyUI/actions/runs/555"
    assert index_builder.entry_from_summary({**s, "run_url": comfy}, LANE, 1)["run_repo"] == (
        "Comfy-Org/ComfyUI")
    # Only a plain Actions run on github.com names one.
    for bad in (None, 5, "https://evil.example/a/b/actions/runs/555", comfy + "/../x",
                "https://github.com.evil.example/a/b/actions/runs/555", "http" + comfy[5:]):
        entry = index_builder.entry_from_summary({**s, "run_url": bad}, LANE, 1)
        assert entry["run"] == 555 and entry["run_repo"] is None
    # No id, no repository: the pair names one run.
    assert index_builder.entry_from_summary({**s, "run_id": None, "run_url": comfy}, LANE, 1)[
        "run_repo"] is None


def test_merge_entry_keeps_the_run_id_a_rerun_does_not_have():
    old = {"c": "x", "t": 1, "o": "pass", "run": 7, "w": {"flux": {"v": "pass"}}}
    # A run outside Actions (run None) never erases the link.
    assert index_builder.merge_entry(old, {"c": "x", "t": 2, "o": "pass", "run": None,
                                           "w": {"flux": {"v": "pass"}}})["run"] == 7
    # A later run in Actions replaces it: the row links to the newest test.
    assert index_builder.merge_entry(old, {"c": "x", "t": 2, "o": "pass", "run": 9,
                                           "w": {"flux": {"v": "pass"}}})["run"] == 9
    # An entry indexed before the key existed gains it, and stays valid without it.
    legacy = {"c": "x", "t": 1, "o": "pass", "w": {"flux": {"v": "pass"}}}
    assert index_builder.merge_entry(legacy, {"c": "x", "t": 2, "o": "pass", "run": 9,
                                              "w": {}})["run"] == 9
    assert "run" not in index_builder.merge_entry(legacy, {"c": "x", "t": 2, "o": "pass",
                                                           "run": None, "w": {}})
    # The repository travels with the id: a newer run replaces both, also when
    # its own repository is not known, and a run outside Actions keeps both.
    old = {**old, "run_repo": "Comfy-Org/ComfyUI"}
    newer = {"c": "x", "t": 2, "o": "pass", "run": 9, "run_repo": "Comfy-Org/worker", "w": {}}
    merged = index_builder.merge_entry(old, newer)
    assert (merged["run"], merged["run_repo"]) == (9, "Comfy-Org/worker")
    merged = index_builder.merge_entry(old, {**newer, "run_repo": None})
    assert (merged["run"], merged["run_repo"]) == (9, None)
    merged = index_builder.merge_entry(old, {**newer, "run": None, "run_repo": None})
    assert (merged["run"], merged["run_repo"]) == (7, "Comfy-Org/ComfyUI")


def _seed_files():
    return {
        "regression/runs/master/aaa/summary.json": _summary("aaa", 1722470400, "pass", IDENT, None, None),
        "regression/runs/master/aaa/flux/run.json": {"outputs": [{"filename": "flux_00001_.png", "sha256": "s1"}]},
        "regression/runs/master/bbb/summary.json": _summary("bbb", 1725148800, "fail", DIFF, DIFF, "aaa"),
        "regression/runs/master/bbb/flux/run.json": {"outputs": [{"filename": "flux_00001_.png", "sha256": "s2"}]},
        "regression/runs/master/ccc/summary.json": _summary("ccc", 1725235200, "fail", DIFF, IDENT, "bbb"),
        "regression/runs/master/ccc/flux/run.json": {"outputs": [{"filename": "flux_00001_.png", "sha256": "s2"}]},
        "regression/latest/master.json": {"commit": "ccc"},
        "regression/golden/flux/current.json": {"tag": "v0.33.2"},
        "regression/golden/flux/v0.33.2/run_r1.json": {"outputs": [{"filename": "flux_00001_.png", "sha256": "s1"}]},
    }


def test_backfill_then_incremental_upsert(tmp_path):
    remote = make_remote(tmp_path, _seed_files())
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
    assert builder.backfill(with_meta=False, log=lambda *_: None) == 3

    head = store.download_json(builder.head_path())
    assert [e["c"] for e in head["entries"]] == ["ccc", "bbb", "aaa"]
    assert head["shards"] == ["2024-08", "2024-09"] and head["count"] == 3
    assert head["first_bad"]["flux"]["commit"] == "bbb"
    assert head["first_bad"]["flux"]["prev_good"] == "aaa"
    assert head["first_bad"]["flux"]["runs"] == 2
    cells = {e["c"]: e["w"]["flux"] for e in head["entries"]}
    assert cells["bbb"]["d"] == "new_drift" and cells["ccc"]["d"] == "inherited"
    assert cells["ccc"]["sha"] == "s2" and cells["aaa"]["sha"] == "s1"
    assert store.download_json(builder.shard_path("2024-09"))["entries"][0]["c"] == "ccc"
    lanes = store.download_json(builder.lanes_path())
    assert lanes["primary"] == LANE
    assert lanes["lanes"][LANE]["branches"]["master"]["latest"]["commit"] == "ccc"
    assert lanes["lanes"][LANE]["goldens"]["flux"] == {"tag": "v0.33.2", "sha": "s1"}
    assert lanes["lanes"][LANE]["python"] == "3.12"

    # Incremental: a new passing run after a re-bless closes the failing chain.
    s = _summary("ddd", 1725321600, "pass", IDENT, DIFF, "ccc", sha="s3")
    s["commit_meta"] = {"subject": "Re-bless (#1)", "author": "dev", "pr": 1, "committed_ts": 1725321500}
    builder.upsert([index_builder.entry_from_summary(s, LANE, 1)])
    head = store.download_json(builder.head_path())
    assert head["entries"][0]["c"] == "ddd" and head["count"] == 4
    assert head["first_bad"] == {}
    assert head["entries"][0]["m"]["pr"] == 1

    store.finalize("index test")
    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    on_remote = json.loads((check / "regression" / "index" / "master" / f"{LANE}.json").read_text())
    assert on_remote["count"] == 4


def test_accepted_drift_closes_the_chain_and_refresh_sees_a_new_bless(tmp_path):
    remote = make_remote(tmp_path, _seed_files())
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
    builder.backfill(with_meta=False, log=lambda *_: None)
    assert "flux" in store.download_json(builder.head_path())["first_bad"]
    # Bless the drifted output (s2) as the new golden: the chain is accepted.
    store.upload_json("regression/golden/flux/current.json", {"tag": "v0.38.0", "output_sha256": "s2"})
    builder.refresh()
    head = store.download_json(builder.head_path())
    assert head["first_bad"] == {} and head["count"] == 3
    assert store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"]["flux"] == {
        "tag": "v0.38.0", "sha": "s2"}


def test_rerun_moves_entry_between_shards_without_duplicates(tmp_path):
    remote = make_remote(tmp_path, _seed_files())
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
    builder.backfill(with_meta=False, log=lambda *_: None)
    # A re-test of aaa whose commit time lands in September moves it.
    s = _summary("aaa", 1727740800, "pass", IDENT, None, None, sha="s1")
    s["commit_meta"] = {"subject": "aaa", "author": "dev", "pr": None, "committed_ts": 1725200000}
    builder.upsert([index_builder.entry_from_summary(s, LANE, 1)])
    aug = store.download_json(builder.shard_path("2024-08"))["entries"]
    sep = store.download_json(builder.shard_path("2024-09"))["entries"]
    assert [e["c"] for e in aug] == []
    assert sorted(e["c"] for e in sep) == ["aaa", "bbb", "ccc"]
    head = store.download_json(builder.head_path())
    assert head["count"] == 3 and head["shards"] == ["2024-09"]
    # Ordered by commit time: aaa's commit (14:13Z) is newer than bbb's run (00:00Z).
    assert [e["c"] for e in head["entries"]] == ["ccc", "aaa", "bbb"]


def test_two_publishers_both_land_in_the_index(tmp_path):
    remote = make_remote(tmp_path, _seed_files())
    a = storage.GitHubStorage("local/test", "results", tmp_path / "a", remote_url=str(remote))
    b = storage.GitHubStorage("local/test", "results", tmp_path / "b", remote_url=str(remote))
    for store, commit, ts in ((a, "ddd", 1725321600), (b, "eee", 1725408000)):
        s = _summary(commit, ts, "pass", IDENT, DIFF, "ccc", sha="s3")
        store.publish_summary("master", commit, s)
        builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
        entry = index_builder.entry_from_summary(s, LANE, 1)
        store.register_derived(lambda st, bl=builder, e=entry: bl.upsert([e]))
    a.finalize("regression master@ddd: pass")
    b.PUSH_ATTEMPTS = 3
    b.finalize("regression master@eee: pass")
    check = tmp_path / "check"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(check)], check=True)
    head = json.loads((check / "regression" / "index" / "master" / f"{LANE}.json").read_text())
    assert [e["c"] for e in head["entries"]] == ["eee", "ddd"]  # fresh index, two runs
    shard = json.loads((check / "regression" / "index" / "master" / LANE / "2024-09.json").read_text())
    assert sorted(e["c"] for e in shard["entries"]) == ["ddd", "eee"]


# -- text workflows -------------------------------------------------------------------
TEXT_SAME = {"kind": "text", "files_compared": 1, "file_count_ref": 1, "file_count_cand": 1,
             "identical": True, "similarity": 1.0, "similarity_basis": "words", "chars_ref": 40,
             "chars_cand": 40, "lines_ref": 2, "lines_cand": 2}
TEXT_DIFF = {**TEXT_SAME, "identical": False, "similarity": 0.864865, "chars_cand": 41,
             "first_diff": {"file": "llm_00001.txt", "line": 2, "col": 22, "offset": 108,
                            "ref": "what the reference says", "cand": "what the run says"}}


def _text_summary(commit, ts, verdict, vs_golden, vs_previous, prev, sha, detail=None):
    wf = {**_wf(verdict, vs_golden, vs_previous, prev, sha=sha, detail=detail),
          "workflow_id": "llm", "thresholds_used": None, "figures": False, "thumbnail": None,
          "output_kind": "text"}
    return {"branch": "master", "commit": commit, "run_ts": ts, "overall": verdict,
            "workflows": {"llm": wf}}


def test_a_text_cell_names_its_kind_and_an_image_cell_keeps_its_keys():
    s = _text_summary("b" * 40, 1700000100, "fail", TEXT_DIFF, TEXT_DIFF, "a" * 40, "t2",
                      detail="new_drift")
    cell = index_builder.entry_from_summary(s, LANE, 1)["w"]["llm"]
    # The image numbers are null, as on a cell whose comparison failed; `k` says why.
    assert cell == {"v": "fail", "d": "new_drift", "sha": "t2", "mse": None, "psnr": None,
                    "pct": None, "pv": False, "exec": 100.0, "vram": 21500.0, "g": "v0.33.2",
                    "th": None, "k": "text"}
    assert list(cell)[-1] == "k"
    same = _text_summary("c" * 40, 1700000200, "pass", TEXT_SAME, TEXT_SAME, "b" * 40, "t1")
    assert index_builder.entry_from_summary(same, LANE, 1)["w"]["llm"]["pv"] is True

    image = index_builder.entry_from_summary(
        _summary("d" * 40, 1700000300, "pass", IDENT, IDENT, "c" * 40, sha="s1"), LANE, 1)
    assert list(image["w"]["flux"]) == ["v", "d", "sha", "mse", "psnr", "pct", "pv", "exec",
                                       "vram", "g", "th"]
    # Only a kind other than image is written; anything that is no string is not a kind.
    for odd in ("image", None, ["text"], 3):
        s["workflows"]["llm"]["output_kind"] = odd
        assert "k" not in index_builder.entry_from_summary(s, LANE, 1)["w"]["llm"]


def test_blessing_a_changed_text_closes_its_failing_chain(tmp_path):
    """Accepted drift works by sha. A text golden carries the text file's sha,
    the same one its runs carry, so approving the changed text ends the chain."""
    text_run = {"outputs": [{"filename": "llm_00001.txt", "sha256": "t2"}]}
    remote = make_remote(tmp_path, {
        "regression/runs/master/aaa/summary.json":
            _text_summary("aaa", 1722470400, "pass", TEXT_SAME, None, None, "t1"),
        "regression/runs/master/bbb/summary.json":
            _text_summary("bbb", 1725148800, "fail", TEXT_DIFF, TEXT_DIFF, "aaa", "t2",
                          detail="new_drift"),
        "regression/runs/master/bbb/llm/run.json": text_run,
        "regression/runs/master/ccc/summary.json":
            _text_summary("ccc", 1725235200, "fail", TEXT_DIFF, TEXT_SAME, "bbb", "t2",
                          detail="inherited"),
        "regression/latest/master.json": {"commit": "ccc"},
        "regression/golden/llm/current.json": {"tag": "v0.33.2", "output_sha256": "t1"},
    })
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
    assert builder.backfill(with_meta=False, log=lambda *_: None) == 3
    head = store.download_json(builder.head_path())
    assert head["first_bad"]["llm"]["commit"] == "bbb" and head["first_bad"]["llm"]["runs"] == 2
    assert [e["w"]["llm"].get("k") for e in head["entries"]] == ["text", "text", "text"]
    assert store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"]["llm"] == {
        "tag": "v0.33.2", "sha": "t1"}

    store.upload_json("regression/golden/llm/current.json",
                      {"tag": "ccc", "output_sha256": "t2"})
    builder.refresh()
    assert store.download_json(builder.head_path())["first_bad"] == {}
    assert store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"]["llm"] == {
        "tag": "ccc", "sha": "t2"}


# -- smoke-level workflows ------------------------------------------------------------
def _smoke_summary(commit, ts, verdict, sha="m1", prior=None, ct=None, **cell):
    wf = {"workflow_id": "mesh", "worker_status": "ok", "verdict": verdict, "detail": None,
          "vs_golden": None, "vs_previous": None, "thresholds_used": None, "golden_tag": None,
          "previous_commit": None, "previous_verdict": None, "figures": False,
          "output_sha256": sha, "thumbnail": "mesh_00001_.webp",
          "timings": {"prompt_exec_s": 104.2}, "vram_peak_mb": 21480.0, "comfy_version": "0.45.0",
          "custom_nodes": None, "level": "smoke",
          "smoke": {"files": [], "problems": [], "notes": []},
          "prior_verdict": prior, "prior_commit": None, **cell}
    summary = {"branch": "master", "commit": commit, "run_ts": ts,
               "overall": "fail" if verdict in index_builder.RED else "pass",
               "workflows": {"mesh": wf}}
    if ct:
        summary["commit_meta"] = {"subject": "x", "author": "dev", "pr": None, "committed_ts": ct}
    return summary


def _smoke_entry(*args, **kw):
    return index_builder.entry_from_summary(_smoke_summary(*args, **kw), LANE, 1)


def test_a_smoke_cell_names_its_level_and_has_no_comparison_numbers():
    cell = _smoke_entry("a" * 40, 1700000100, "pass")["w"]["mesh"]
    assert cell == {"v": "pass", "d": None, "sha": "m1", "mse": None, "psnr": None, "pct": None,
                    "pv": None, "exec": 104.2, "vram": 21480.0, "g": None,
                    "th": "mesh_00001_.webp", "l": "smoke"}
    assert list(cell)[-1] == "l"
    # After the kind, before the packs: `l` is the next key a reader of `k` meets.
    packs = {"ComfyUI-KJNodes": {"import": "ok"}}
    cell = _smoke_entry("a" * 40, 1700000100, "pass", output_kind="audio",
                        custom_nodes=packs)["w"]["mesh"]
    assert list(cell)[-3:] == ["k", "l", "cn"] and (cell["k"], cell["l"]) == ("audio", "smoke")
    # Absent means reference. Nothing else than "smoke" is written.
    for level in ("reference", None, "Smoke", 1, ["smoke"]):
        assert "l" not in _smoke_entry("a" * 40, 1700000100, "pass", level=level)["w"]["mesh"]
    image = index_builder.entry_from_summary(
        _summary("d" * 40, 1700000300, "pass", IDENT, IDENT, "c" * 40, sha="s1"), LANE, 1)
    assert "l" not in image["w"]["flux"] and "wr" not in image


def test_a_standing_smoke_failure_is_marked_inherited():
    """The mark a standing reference failure has, so a reader that looks for
    "started here" does not announce the same smoke failure on every run."""
    def detail(verdict, prior, **cell):
        return _smoke_entry("b" * 40, 1700000100, verdict, prior=prior, **cell)["w"]["mesh"]["d"]

    assert detail("fail", "fail") == "inherited"
    assert detail("fail", "pass") is None           # new at this run
    assert detail("fail", "execution_error") is None
    assert detail("fail", None) is None             # its first run
    assert detail("pass", "fail") is None
    assert detail("execution_error", "fail") is None
    # Nothing is compared at smoke level, so the cell's own detail says nothing.
    assert detail("fail", "pass", detail="new_drift") is None
    # A reference cell is classified as it always was, whatever came before it.
    s = _summary("b" * 40, 1700000100, "fail", DIFF, DIFF, "a" * 40)
    s["workflows"]["flux"]["prior_verdict"] = "fail"
    assert index_builder.entry_from_summary(s, LANE, 1)["w"]["flux"]["d"] == "new_drift"


def test_a_smoke_chain_opens_with_its_level_and_closes_on_the_next_pass():
    runs = [_smoke_entry("a", 1, "pass"), _smoke_entry("b", 2, "fail", prior="pass"),
            _smoke_entry("c", 3, "infra_error"), _smoke_entry("d", 4, "fail", prior="fail")]
    assert index_builder.first_bad(runs) == {"mesh": {
        "commit": "b", "run_ts": 2, "kind": "fail", "prev_good": "a", "pr": None, "author": None,
        "golden": None, "runs": 2, "level": "smoke"}}
    assert index_builder.first_bad([*runs, _smoke_entry("e", 5, "pass", prior="fail")]) == {}
    # The level is that of the cell that opened the chain.
    moved = [_entry("a", 1, "fail", wf="mesh"), _smoke_entry("b", 2, "fail", prior="fail")]
    assert "level" not in index_builder.first_bad(moved)["mesh"]
    assert index_builder.first_bad(moved)["mesh"]["runs"] == 2
    assert index_builder.smoke_workflows(moved) == {"mesh"}
    assert index_builder.smoke_workflows(list(reversed(runs))) == {"mesh"}
    # The newest cell decides, in commit order: this workflow is at reference level again.
    back = [*moved, _entry("c", 3, "pass", wf="mesh")]
    assert index_builder.smoke_workflows(back) == set()
    assert index_builder.smoke_workflows([]) == set()


def test_an_entry_carries_the_range_of_each_workflow_and_a_chain_the_one_it_opened_with():
    ranges = {"mesh": {"prev": "p" * 40, "commits_between": 7, "compare_url": "u7"},
              "other": {"prev": "q" * 40, "commits_between": None, "compare_url": "u"},
              "odd": "not a range"}
    summary = {**_smoke_summary("b", 2, "fail", prior="pass"), "ranges": ranges}
    entry = index_builder.entry_from_summary(summary, LANE, 1)
    assert entry["wr"] == {"mesh": {"n": 7, "url": "u7"}, "other": {"n": None, "url": "u"}}
    assert list(entry)[-1] == "wr"
    for none in ({}, None, ["mesh"]):
        assert "wr" not in index_builder.entry_from_summary({**summary, "ranges": none}, LANE, 1)
    chain = index_builder.first_bad([_smoke_entry("a", 1, "pass"), entry,
                                     _smoke_entry("c", 3, "fail", prior="fail")])["mesh"]
    # "A run that covered 7 commits": the chain's commit is only the newest of them.
    assert chain == {"commit": "b", "run_ts": 2, "kind": "fail", "prev_good": "a", "pr": None,
                     "author": None, "golden": None, "runs": 2, "level": "smoke",
                     "range": {"n": 7, "url": "u7"}}
    # The range is that of the run that opened the chain, for this workflow.
    late = index_builder.entry_from_summary(
        {**_smoke_summary("c", 3, "fail", prior="fail"), "ranges": ranges}, LANE, 1)
    assert "range" not in index_builder.first_bad([_smoke_entry("b", 2, "fail"), late])["mesh"]
    other = {**entry, "wr": {"other": {"n": 3, "url": "u"}}}
    assert "range" not in index_builder.first_bad([other])["mesh"]


def test_merge_entry_takes_the_ranges_of_the_newest_summary():
    old = {"c": "x", "t": 1, "o": "fail", "w": {"mesh": {"v": "fail"}, "b": {"v": "fail"}},
           "wr": {"mesh": {"n": 7, "url": "u"}, "b": {"n": 7, "url": "u"}}}
    new = {"c": "x", "t": 2, "o": "fail", "w": {"mesh": {"v": "pass"}},
           "wr": {"b": {"n": 7, "url": "u"}}}
    assert index_builder.merge_entry(old, new)["wr"] == {"b": {"n": 7, "url": "u"}}
    # A summary without ranges has no red cell left that covers several commits.
    assert "wr" not in index_builder.merge_entry(old, {"c": "x", "t": 2, "o": "pass",
                                                       "w": {"mesh": {"v": "pass"}}})
    assert "wr" not in index_builder.merge_entry({"c": "x", "t": 1, "w": {}},
                                                 {"c": "x", "t": 2, "w": {}})


def _reference_then_smoke(tmp_path):
    """A workflow that passed at reference level (reference sha m1), then moved
    to smoke level. Its reference files are still on the results branch."""
    remote = make_remote(tmp_path, {
        "regression/latest/master.json": {"commit": "aaa"},
        "regression/golden/mesh/current.json": {"tag": "v1", "output_sha256": "m1"},
        "regression/golden/flux/current.json": {"tag": "v0.33.2", "output_sha256": "s1"}})
    store = storage.GitHubStorage("local/test", "results", tmp_path / "wt", remote_url=str(remote))
    builder = index_builder.IndexBuilder(store, "master", LANE, 1, LANES)
    first = _summary("aaa", 1722470400, "pass", IDENT, None, None, sha="s1")
    first["workflows"]["mesh"] = {**first["workflows"]["flux"], "workflow_id": "mesh",
                                  "golden_tag": "v1", "output_sha256": "m1"}
    builder.upsert([index_builder.entry_from_summary(first, LANE, 1)])
    return store, builder


def test_a_smoke_failure_is_never_accepted_drift_of_a_left_over_reference(tmp_path, monkeypatch):
    store, builder = _reference_then_smoke(tmp_path)
    goldens = store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"]
    assert goldens == {"flux": {"tag": "v0.33.2", "sha": "s1"}, "mesh": {"tag": "v1", "sha": "m1"}}

    # At smoke level it fails, with the very bytes the old reference has.
    looked_up = []
    lookup = builder._golden_sha
    monkeypatch.setattr(builder, "_golden_sha", lambda wf: looked_up.append(wf) or lookup(wf))
    second = _smoke_summary("bbb", 1725148800, "fail", sha="m1", prior="pass")
    second["workflows"]["flux"] = _wf("fail", DIFF, DIFF, "aaa", sha="s1")
    builder.upsert([index_builder.entry_from_summary(second, LANE, 1)])
    head = store.download_json(builder.head_path())
    # The chain stays: the sha of a reference that no longer counts closes nothing.
    assert head["first_bad"]["mesh"]["commit"] == "bbb"
    assert head["first_bad"]["mesh"]["level"] == "smoke"
    # A reference workflow whose output is its blessed reference is accepted, as before.
    assert "flux" not in head["first_bad"]
    # No reference is read for the smoke workflow, and the lane no longer lists one.
    assert looked_up == ["flux"]
    assert store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"] == {
        "flux": {"tag": "v0.33.2", "sha": "s1"}}
    builder.refresh()
    assert "mesh" in store.download_json(builder.head_path())["first_bad"]

    # Back at reference level, a bless names the workflow: its reference counts again.
    builder.refresh(("mesh",))
    assert store.download_json(builder.head_path())["first_bad"] == {}
    assert store.download_json(builder.lanes_path())["lanes"][LANE]["goldens"]["mesh"] == {
        "tag": "v1", "sha": "m1"}


def test_a_chain_is_listed_only_while_the_lane_runs_its_workflow(tmp_path):
    """A workflow that was renamed, moved to another cadence or removed while
    it was red would keep its chain open for ever: no later run closes it."""
    store, builder = _reference_then_smoke(tmp_path)
    red = _smoke_summary("bbb", 1725148800, "fail", sha="m2", prior="pass")
    red["workflows"]["flux"] = _wf("fail", DIFF, DIFF, "aaa", sha="s2")
    builder.upsert([index_builder.entry_from_summary(red, LANE, 1)])
    assert sorted(store.download_json(builder.head_path())["first_bad"]) == ["flux", "mesh"]

    # The next run is of a manifest without `mesh`: run_regression.py names what the lane runs.
    running = index_builder.IndexBuilder(store, "master", LANE, 1, LANES, active=["flux"])
    third = _summary("ccc", 1725235200, "fail", DIFF, IDENT, "bbb", sha="s2")
    running.upsert([index_builder.entry_from_summary(third, LANE, 1)])
    head = store.download_json(running.head_path())
    assert sorted(head["first_bad"]) == ["flux"] and head["first_bad"]["flux"]["runs"] == 2
    # Its history stays, only the open chain goes.
    assert [sorted(e["w"]) for e in head["entries"]] == [["flux"], ["flux", "mesh"],
                                                        ["flux", "mesh"]]
    # Without the ids (a bless, a backfill) every chain is listed, as before.
    builder.refresh()
    assert sorted(store.download_json(builder.head_path())["first_bad"]) == ["flux", "mesh"]
    assert index_builder.IndexBuilder(store, "master", LANE, 1, LANES, active=()).active == set()
