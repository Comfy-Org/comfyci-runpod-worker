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
