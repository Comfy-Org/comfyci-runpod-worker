"""frontend_index: the perf statistics mirror, the ci and gate record
builders, and the publisher end to end against a local results remote with
a URL-routed GitHub API fake."""
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pytest
import requests

import frontend_index as fi
import storage
from conftest import make_remote

FIX = Path(__file__).resolve().parent / "fixtures" / "frontend"
REPO = fi.FRONTEND_REPO
API = f"{fi.GITHUB_API}/repos/{REPO}"
NOW = int(datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc).timestamp())
H = 3600
PIN = "b08e6cf35fac50d3ca8470dffb3f9a1fbb7187d2"
WORKFLOW_TEXT = f"name: CI: Tests Custom Nodes\nenv:\n  COMFYUI_PIN: {PIN}\n"
WORKFLOWS = {
    "ci-tests-unit.yaml": "CI: Tests Unit", "ci-tests-e2e.yaml": "CI: Tests E2E",
    "ci-tests-storybook.yaml": "CI: Tests Storybook", "ci-perf-report.yaml": "CI: Performance Report",
    "ci-ecosystem-matrix.yaml": "CI: Custom Nodes Ecosystem Matrix",
    "ci-lint-format-queue.yaml": "CI: Lint Format", "ci-json-validation.yaml": "CI: JSON Validation",
    "ci-size-data.yaml": "CI: Size Data", "ci-website-unit.yaml": "CI: Website Unit",
    "ci-website-e2e.yaml": "CI: Website E2E", "ci-website-build.yaml": "CI: Website Build",
    "ci-billing-web-unit.yaml": "CI: Billing Web Unit", "ci-billing-web-e2e.yaml": "CI: Billing Web E2E",
    "ci-oss-assets-validation.yaml": "CI: OSS Assets Validation",
    "ci-adr-links-validation.yaml": "CI: ADR Link Validation",
    "cloud-dispatch-build.yaml": "Cloud Frontend Build Dispatch",
    "detect-unreviewed-merge.yml": "Detect Unreviewed Merge",
}
KNOWN_IDS = {c["id"] for c in fi.COLUMNS}
SHARDS = ["core:1/2", "core:2/2", "1/5", "2/5", "3/5", "4/5", "5/5"]
PROOFS = [("2/5", 9), ("4/5", 3), ("4/5", 2), ("4/5", 1), ("core:2/2", 15)]
SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# -- GitHub API fake ------------------------------------------------------------
class Resp:
    def __init__(self, status, body, remaining=900, limit=1000):
        self.status_code = status
        self._body = body
        self.headers = {"X-RateLimit-Remaining": str(remaining), "X-RateLimit-Limit": str(limit)}

    def json(self):
        if isinstance(self._body, bytes):
            raise ValueError("not json")
        return self._body

    @property
    def content(self):
        return self._body if isinstance(self._body, bytes) else json.dumps(self._body).encode()

    @property
    def text(self):
        return self.content.decode()


class FakeApi:
    """requests.get replacement routed by URL path and query parameters;
    unmatched requests answer 404. Both frontend_index and commit_meta call
    requests.get directly, so one fake covers them."""

    def __init__(self, monkeypatch):
        self.routes = []
        self.calls = []
        monkeypatch.setattr(requests, "get", self.get)

    def on(self, path, resp, **params):
        self.routes.append((f"{API}{path}", {k: str(v) for k, v in params.items()}, resp))
        return self

    def get(self, url, params=None, headers=None, timeout=None):
        params = {k: str(v) for k, v in (params or {}).items()}
        self.calls.append(url + ("?" + urlencode(params) if params else ""))
        for route_url, route_params, resp in self.routes:
            if url == route_url and all(params.get(k) == v for k, v in route_params.items()):
                if isinstance(resp, Exception):
                    raise resp
                return resp(url, params) if callable(resp) else resp
        return Resp(404, {"message": "Not Found"})

    def count(self, fragment):
        return sum(1 for c in self.calls if fragment in c)


# -- fixture builders -----------------------------------------------------------
def ci_run(path, sha, ct, conclusion="success", status="completed", attempt=1, start=None,
           subject="feat: thing (#21295)", author="Terry Jia", event="push", run_id=None):
    start = ct + 600 if start is None else start
    return {"id": run_id or int(sha[:4], 16) * 100 + list(WORKFLOWS).index(path),
            "name": WORKFLOWS[path], "path": f".github/workflows/{path}", "event": event,
            "status": status, "conclusion": conclusion if status == "completed" else None,
            "run_attempt": attempt, "head_sha": sha, "head_branch": "main",
            "created_at": iso(start), "run_started_at": iso(start), "updated_at": iso(start + 400),
            "head_commit": {"id": sha, "timestamp": iso(ct), "message": f"{subject}\n\nbody",
                            "author": {"name": author}}}


def commit_runs(sha, ct, **overrides):
    """One run per workflow (15 CI columns + 2 non-CI); overrides maps a
    file name to ci_run keyword arguments."""
    return [ci_run(path, sha, ct, **overrides.get(path.split(".")[0].replace("-", "_"), {}))
            for path in WORKFLOWS]


def runs_page(runs):
    return Resp(200, {"total_count": len(runs), "workflow_runs": runs})


def commit_api(fake, sha, ct, subject="feat: thing (#21295)", login="terryjia", pr=21295,
               base="main"):
    fake.on(f"/commits/{sha}", Resp(200, {
        "sha": sha, "author": {"login": login},
        "commit": {"message": f"{subject}\n\nbody", "author": {"name": "Terry Jia"},
                   "committer": {"date": iso(ct)}}, "parents": [{"sha": "p" * 40}]}))
    fake.on(f"/commits/{sha}/pulls", Resp(200, [{
        "number": pr, "title": "PR title", "merged_at": iso(ct), "base": {"ref": base},
        "merge_commit_sha": sha, "user": {"login": login, "avatar_url": f"https://avatars/{login}"},
        "labels": [], "html_url": f"https://github.com/{REPO}/pull/{pr}"}]))


def job(name, conclusion, steps=(), status="completed"):
    return {"id": abs(hash(name)) % 10 ** 9, "name": name, "status": status,
            "conclusion": conclusion if status == "completed" else None,
            "steps": [{"name": n, "number": i + 1, "status": status,
                       "conclusion": c if status == "completed" else None}
                      for i, (n, c) in enumerate(steps)]}


def gate_jobs(clean="failure", proofs="failure", final="failure", status="completed"):
    jobs = [job("changes", "success", [("Detect", "success")], status)]
    for shard in SHARDS:
        jobs.append(job(f"custom_nodes_e2e ({shard}, 0)", clean,
                        [("Run custom-node suite", clean),
                         ("Forbid failed, skipped, or flaky tests", clean)], status))
    for shard, row in PROOFS:
        jobs.append(job(f"custom_nodes_e2e ({shard}, {row})", proofs,
                        [("Run custom-node suite", "success"),
                         ("Assert deliberate failure independence", proofs)], status))
    jobs.append(job(fi.GATE_VERDICT_JOB, final, [("Enforce suite result", final)], status))
    return jobs


def gate_list_run(run_id, sha, start, event="schedule", conclusion="failure",
                  status="completed", attempt=1):
    return {"id": run_id, "name": "CI: Tests Custom Nodes", "event": event, "status": status,
            "conclusion": conclusion if status == "completed" else None, "run_attempt": attempt,
            "head_sha": sha, "head_branch": "main", "created_at": iso(start),
            "run_started_at": iso(start), "updated_at": iso(start + 900),
            "path": f".github/workflows/{fi.GATE_WORKFLOW}"}


def gate_api(fake, runs, jobs_by_id, pins):
    fake.on(f"/actions/workflows/{fi.GATE_WORKFLOW}/runs", runs_page(runs))
    for run_id, jobs in jobs_by_id.items():
        fake.on(f"/actions/runs/{run_id}/jobs", Resp(200, {"total_count": len(jobs), "jobs": jobs}))
    for sha, text in pins.items():
        fake.on(f"/contents/.github/workflows/{fi.GATE_WORKFLOW}", Resp(200, text.encode()),
                ref=sha)


PERF_FILES = ["perf-20261019T103805Z-17b0d4e4.json", "perf-20261019T105541Z-e38b73dd.json",
              "perf-20261019T130522Z-dd678395.json"]


def perf_api(fake, files=PERF_FILES, blob_status=None):
    fake.on("/git/trees/perf-data", Resp(200, {"sha": "t", "truncated": False, "tree": [
        {"path": "baselines/.gitkeep", "type": "blob", "sha": "keep"},
        *({"path": f"baselines/{f}", "type": "blob", "sha": f"blob-{f[-13:-5]}"} for f in files)]}))
    for f in files:
        status = (blob_status or {}).get(f, 200)
        body = (FIX / f).read_bytes() if status == 200 else {"message": "Not Found"}
        fake.on(f"/git/blobs/blob-{f[-13:-5]}", Resp(status, body))


def raw_baseline(name="perf-20261019T130522Z-dd678395.json"):
    return load(name)


# -- storage helpers -------------------------------------------------------------
class MemStore(storage.Storage):
    """In-memory backend for the writers; reads return copies like a real
    backend parsing files."""

    def __init__(self):
        super().__init__("frontend")
        self.files = {}
        self.written = []

    def download_json(self, path):
        return copy.deepcopy(self.files.get(path))

    def upload_json(self, path, obj, compact=False):
        self.files[path] = copy.deepcopy(obj)
        self.written.append(path)


def _store(tmp_path, remote, name):
    return storage.GitHubStorage("local/test", "results", tmp_path / name, prefix="frontend",
                                 remote_url=str(remote))


def remote_text(bare, path):
    r = subprocess.run(["git", "-C", str(bare), "cat-file", "-p", f"results:{path}"],
                       capture_output=True, check=False)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def remote_json(bare, path):
    text = remote_text(bare, path)
    return json.loads(text) if text is not None else None


def remote_head(bare):
    return subprocess.run(["git", "-C", str(bare), "rev-parse", "results"], capture_output=True,
                          text=True, check=True).stdout.strip()


def args(**kw):
    a = fi.parse_args([])
    a.only = kw.pop("only", "perf,ci,gate")
    a.backfill_days = kw.pop("backfill_days", None)
    a.skip_publish = kw.pop("skip_publish", False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def remote(tmp_path, monkeypatch):
    for var in ("FRONTEND_BACKFILL_DAYS", "FRONTEND_ONLY", "FRONTEND_GH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
    return make_remote(tmp_path, {"regression/latest/master.json": {"commit": "aaa"}})


# -- perf: registry, entries, statistics --------------------------------------------
def test_metric_registry_mirrors_the_frontend_order_and_gates():
    assert [m["key"] for m in fi.METRICS] == [
        "frameDurationMs", "p95FrameDurationMs", "layoutDurationMs", "styleRecalcDurationMs",
        "layouts", "styleRecalcs", "taskDurationMs", "scriptDurationMs", "totalBlockingTimeMs",
        "heapUsedBytes", "domNodes", "eventListeners"]
    assert [m["unit"] for m in fi.METRICS] == ["ms"] * 4 + [""] * 2 + ["ms"] * 3 + ["bytes", "", ""]
    assert {m["key"] for m in fi.METRICS if m["min_abs_delta"] == 5} == {
        "layouts", "styleRecalcs", "domNodes", "eventListeners"}
    assert fi.TARGETS == {"p5_fps": 52} and fi.PERF_WINDOW == 20


def test_perf_entry_means_exceptions_nulls_and_rounding():
    def sample(name, **kw):
        base = {"name": name, "frameDurationMs": 1.0, "p95FrameDurationMs": 1.0,
                "layoutDurationMs": 1.0, "styleRecalcDurationMs": 1.0, "layouts": 1,
                "styleRecalcs": 1, "taskDurationMs": 1.0, "scriptDurationMs": 1.0,
                "totalBlockingTimeMs": 1.0, "heapUsedBytes": 1, "domNodes": 1, "eventListeners": 1}
        return {**base, **kw}
    raw = {"timestamp": "2026-10-19T13:05:19.646Z", "gitSha": "d" * 40, "measurements": [
        sample("a", frameDurationMs=1 / 3, heapUsedBytes=10, layouts=1),
        sample("a", frameDurationMs=2 / 3, heapUsedBytes=11, layouts=2),
        sample("a", frameDurationMs=1.0, heapUsedBytes=11, layouts=float("nan")),
        sample("b", frameDurationMs=5.0, totalBlockingTimeMs=float("inf")),
        sample("b", frameDurationMs=7.0, totalBlockingTimeMs=float("inf")),
    ]}
    e = fi.perf_entry("20261019T130522Z-dd678395", raw)
    assert e["id"] == "20261019T130522Z-dd678395" and e["c"] == "d" * 40
    assert e["t"] == fi._ts("2026-10-19T13:05:19Z") and e["n"] == 3 and e["nx"] == {"b": 2}
    assert e["s"]["a"][0] == 0.66666667          # 8 significant digits
    assert e["s"]["a"][9] == 11                  # bytes: integer, half up (10.666 -> 11)
    assert e["s"]["a"][4] == 1.5                 # NaN sample dropped from the mean
    assert e["s"]["b"][8] is None                # no finite value at all
    assert e["s"]["b"][0] == 6.0 and "cls" not in e
    meds = fi.perf_medians(raw)
    assert meds["a"][0] == 2 / 3 and meds["b"][0] == 6.0 and meds["b"][8] is None
    assert fi.perf_entry("x", {"measurements": []})["n"] == 0


def test_baseline_id_parsing():
    assert fi.baseline_id("baselines/perf-20261019T130522Z-dd678395.json") == "20261019T130522Z-dd678395"
    assert fi.baseline_id("baselines/.gitkeep") is None
    assert fi.baseline_id("perf-20261019T130522Z-dd678395.json") is None
    assert fi.baseline_id("baselines/perf-20261019T130522Z-dd678395.json.bak") is None
    assert fi.id_ts("20261019T130522Z-dd678395") == fi._ts("2026-10-19T13:05:22Z")
    assert fi.id_ts("garbage") is None


def test_stats_mirror_perf_stats_ts():
    s = fi.compute_stats([2, 4, 4, 4, 5, 5, 7, 9])
    assert s["mean"] == 5 and s["n"] == 8
    assert s["stddev"] == pytest.approx((32 / 7) ** 0.5)      # sample variance, n - 1
    assert fi.compute_stats([]) == {"mean": 0.0, "stddev": 0.0, "n": 0}
    assert fi.compute_stats([3.5]) == {"mean": 3.5, "stddev": 0.0, "n": 1}
    assert fi.z_score(7, s) == pytest.approx(2 / (32 / 7) ** 0.5)
    assert fi.z_score(7, fi.compute_stats([5, 5, 5])) is None          # stddev 0
    assert fi.z_score(7, fi.compute_stats([5])) is None                # n < 2
    assert fi.coefficient_of_variation(s) == pytest.approx(s["stddev"] / 5 * 100)
    assert fi.coefficient_of_variation({"mean": 0, "stddev": 1, "n": 3}) == 0
    assert fi.coefficient_of_variation({"mean": -2, "stddev": 1, "n": 3}) == 0


def test_classify_change_order_and_strict_thresholds():
    assert fi.classify_change(10, 51) == "noisy"                   # CV gate first
    assert fi.classify_change(None, 50) == "neutral"
    assert fi.classify_change(7, 10, 4, 5) == "neutral"            # effect-size gate
    assert fi.classify_change(7, 10, -4.9, 5) == "neutral"
    assert fi.classify_change(7, 10, 5, 5) == "regression"
    assert fi.classify_change(7, 10, 4, None) == "regression"      # no gate without min_abs_delta
    assert fi.classify_change(2.0, 10) == "neutral"                # strict > 2
    assert fi.classify_change(2.01, 10) == "regression"
    assert fi.classify_change(-2.0, 10) == "neutral"
    assert fi.classify_change(-2.01, 10) == "improvement"
    assert fi.classify_change(-9, 10, -4, 5) == "neutral"


def test_compare_metric_marks_insufficient_history():
    assert fi.compare_metric(3.0, []) == [3.0, None, None, 0, None, "-"]
    assert fi.compare_metric(3.0, [2.0]) == [3.0, 2.0, None, 1, None, "-"]
    assert fi.compare_metric(None, [2.0, 3.0]) == [None, 2.5, None, 2, None, "-"]
    cur, mean, std, n, z, cls = fi.compare_metric(9.0, [4.0, 6.0])
    assert (cur, mean, n, cls) == (9.0, 5.0, 2, "regression"[0]) and std == pytest.approx(2 ** 0.5)
    assert z == pytest.approx(4 / 2 ** 0.5)
    assert fi.compare_metric(9.0, [5.0, 5.0])[4:] == [None, "n"]   # identical history: z null


def test_preceding_window_excludes_the_entry_and_caps_at_twenty():
    entries = [{"id": f"e{i}", "t": 1000 + i, "s": {}} for i in range(25)]
    window = fi.preceding(entries, entries[22])
    assert [e["id"] for e in window] == [f"e{i}" for i in range(21, 1, -1)]
    assert fi.preceding(entries, entries[0]) == []
    assert [e["id"] for e in fi.preceding(entries, entries[2])] == ["e1", "e0"]
    # Ties on t fall back to the id, so an entry never precedes itself.
    twin = {"id": "e22-twin", "t": entries[22]["t"], "s": {}}
    assert twin not in fi.preceding(entries + [twin], twin)
    assert "e22" in [e["id"] for e in fi.preceding(entries + [twin], twin)]


def test_latest_comparison_matches_the_frontend_report():
    """expected_latest_cmp.json was produced by the frontend's own
    perf-stats.ts functions (see fixtures/frontend/README.md)."""
    expected = load("expected_latest_cmp.json")
    entries = load("entries.json")["entries"]
    newest = entries[0]
    assert newest["id"] == expected["id"] == "20261019T130522Z-dd678395"
    history = fi.preceding(entries, newest)
    assert [e["id"] for e in history] == expected["history_ids"][::-1] and len(history) == 19
    cmp = fi.compare_entry(fi.perf_medians(raw_baseline()), history)
    assert list(cmp) == list(expected["cmp"])
    for scenario, rows in expected["cmp"].items():
        for i, (want, got) in enumerate(zip(rows, cmp[scenario])):
            unit = fi.METRICS[i]["unit"]
            assert got[5] == want[5] and got[3] == want[3], (scenario, i, want, got)
            for k in (0, 1, 2):
                if want[k] is None:
                    assert got[k] is None
                else:
                    tol = 1 if unit == "bytes" else abs(want[k]) * 1e-6 + 1e-9
                    assert got[k] == pytest.approx(want[k], abs=tol), (scenario, i, k)
            if want[4] is None:
                assert got[4] is None
            else:
                assert got[4] == pytest.approx(want[4], abs=0.006, rel=1e-6), (scenario, i)
    assert fi.classification(cmp) == newest["cls"]
    letters = "".join(newest["cls"].values())
    assert len(letters) == 26 * 12 and "-" not in letters and "r" in letters and "x" in letters
    assert fi.perf_entry(newest["id"], raw_baseline()) == {k: v for k, v in newest.items()
                                                           if k != "cls"}


def test_write_perf_classifies_at_ingest_and_is_idempotent():
    store = MemStore()
    new = [(fi.baseline_id(f"baselines/{f}"), load(f)) for f in reversed(PERF_FILES)]
    assert fi.write_perf(store, REPO, new, NOW) == 3
    head = store.files["frontend/perf/head.json"]
    shard = store.files["frontend/perf/2026-10.json"]
    ids = [e["id"] for e in head["entries"]]
    assert ids == sorted(ids, reverse=True) == [e["id"] for e in shard["entries"]]
    assert head["count"] == 3 and head["shards"] == {"2026-10": 3} and head["updated_ts"] == NOW
    assert head["schema_version"] == 1 and head["repo"] == REPO
    assert head["metrics"] == fi.METRICS and head["window"] == 20
    assert head["stats"]["delta_base"] == "history_mean"
    assert len(head["scenarios"]) == 26 and head["scenarios"] == sorted(head["scenarios"])
    oldest, middle, newest = head["entries"][2], head["entries"][1], head["entries"][0]
    assert set("".join(oldest["cls"].values())) == {"-"}      # no history
    assert set("".join(middle["cls"].values())) == {"-"}      # one value: insufficient
    assert set("".join(newest["cls"].values())) <= set("rinx")
    assert head["latest"]["id"] == newest["id"] and head["latest"]["n_hist"] == 2
    row = head["latest"]["cmp"]["canvas-idle"][0]
    assert row[3] == 2 and row[5] == newest["cls"]["canvas-idle"][0]
    store.written.clear()
    assert fi.write_perf(store, REPO, new, NOW + 1800) == 0
    assert store.written == []                                # unchanged content, no write
    assert fi.write_perf(store, REPO, [], NOW + 1800) == 0 and store.written == []


# -- ci: cells, entries, chains ---------------------------------------------------
def test_cell_and_overall_mapping():
    done = fi.cell(ci_run("ci-tests-unit.yaml", SHA_A, NOW - 2 * H))
    assert done == {"id": done["id"], "c": "success", "t": NOW - 2 * H + 600, "d": 400}
    running = fi.cell(ci_run("ci-tests-unit.yaml", SHA_A, NOW - 2 * H, status="in_progress"))
    assert running["c"] is None and running["s"] == "in_progress" and "d" not in running
    rerun = fi.cell(ci_run("ci-tests-unit.yaml", SHA_A, NOW - 2 * H, attempt=2))
    assert rerun["a"] == 2
    ok = {"tests-unit": {"c": "success"}, "website-build": {"c": "failure"}}
    assert fi.overall(ok, fi.COLUMNS) == "success"            # non-core never blocks
    assert fi.overall({**ok, "tests-e2e": {"c": "timed_out"}}, fi.COLUMNS) == "failure"
    assert fi.overall({**ok, "lint": {"c": "startup_failure"}}, fi.COLUMNS) == "failure"
    assert fi.overall({**ok, "lint": {"c": None, "s": "queued"}}, fi.COLUMNS) == "pending"
    assert fi.overall({**ok, "lint": {"c": "cancelled"}}, fi.COLUMNS) == "cancelled"
    assert fi.overall({**ok, "lint": {"c": "skipped"}}, fi.COLUMNS) == "success"
    assert fi.overall({**ok, "lint": {"c": "failure"}, "size-data": {"c": None, "s": "queued"}},
                      fi.COLUMNS) == "failure"
    assert fi.overall({}, fi.COLUMNS) == "success"


def test_merge_keeps_the_latest_attempt_and_only_upgrades_meta():
    old = {"id": 1, "c": "failure", "t": 100}
    assert fi.pick_cell(None, old) is old
    assert fi.pick_cell(old, {"id": 1, "c": "success", "t": 100, "a": 2})["a"] == 2
    assert fi.pick_cell({"id": 1, "c": "success", "t": 100, "a": 2}, old)["a"] == 2
    assert fi.pick_cell(old, {"id": 2, "c": "success", "t": 200})["id"] == 2
    assert fi.pick_cell({"id": 2, "c": "success", "t": 200}, old)["id"] == 2
    assert fi.pick_cell(old, {"id": 1, "c": None, "t": 100, "s": "in_progress"})["s"] == "in_progress"

    full = {"s": "x", "a": "me", "pr": 1, "ct": 50, "pt": "PR", "pa": "me", "av": "u"}
    prov = {"s": "x", "a": "Me", "pr": 1, "ct": 50}
    stored = {"c": SHA_A, "ct": 50, "o": "success", "m": full, "w": {"lint": old}}
    fetched = {"c": SHA_A, "ct": 50, "o": None, "m": prov, "mi": True,
               "w": {"tests-unit": {"id": 3, "c": "success", "t": 120}}}
    merged = fi.merge_ci_entry(stored, fetched)
    assert merged["m"] == full and "mi" not in merged
    assert set(merged["w"]) == {"lint", "tests-unit"}
    upgraded = fi.merge_ci_entry({**stored, "m": prov, "mi": True}, {**fetched, "m": full, "mi": False})
    assert upgraded["m"] == full and "mi" not in upgraded
    still = fi.merge_ci_entry({**stored, "m": prov, "mi": True}, fetched)
    assert still["m"] == prov and still["mi"] is True
    assert fi.merge_ci_entry(None, fetched) is fetched


def test_provisional_meta_from_head_commit():
    run = ci_run("ci-tests-unit.yaml", SHA_A, NOW - H, subject="fix: thing (#42)")
    assert fi.provisional_meta(run) == {"s": "fix: thing (#42)", "a": "Terry Jia", "pr": 42,
                                        "ct": NOW - H}
    assert fi.provisional_meta({"head_commit": {"message": "Direct push"}}) == {
        "s": "Direct push", "a": None, "pr": None, "ct": None}
    assert fi.provisional_meta({})["s"] is None


def test_entries_from_runs_group_by_commit_and_ignore_non_ci_workflows():
    columns = fi.known_columns({})
    runs = commit_runs(SHA_A, NOW - 2 * H) + commit_runs(SHA_B, NOW - H, ci_tests_unit={
        "status": "in_progress"})
    runs.append(ci_run("ci-tests-unit.yaml", "", NOW, run_id=5))        # no sha: dropped
    entries = fi.ci_entries(runs, columns)
    assert set(entries) == {SHA_A, SHA_B}
    assert set(entries[SHA_A]["w"]) == KNOWN_IDS and entries[SHA_A]["mi"] is True
    assert entries[SHA_A]["ct"] == NOW - 2 * H and entries[SHA_A]["m"]["pr"] == 21295
    assert entries[SHA_B]["w"]["tests-unit"]["s"] == "in_progress"
    assert [c["id"] for c in columns] == [c["id"] for c in fi.COLUMNS]   # nothing discovered


def test_column_registry_lint_paths_and_discovery():
    columns = fi.known_columns({})
    assert fi.column_for(columns, ".github/workflows/ci-lint-format-queue.yaml", "CI: Lint Format")["id"] == "lint"
    assert fi.column_for(columns, ".github/workflows/ci-lint-format.yaml", "CI: Lint Format")["id"] == "lint"
    assert fi.column_for(columns, ".github/workflows/cloud-dispatch-build.yaml", "Cloud Build") is None
    new = fi.column_for(columns, ".github/workflows/ci-foo-bar.yaml", "CI: Foo Bar")
    assert new == {"id": "foo-bar", "label": "Foo Bar", "paths": [".github/workflows/ci-foo-bar.yaml"],
                   "name": "CI: Foo Bar", "group": "other"}
    assert columns[-1] is new and len(columns) == len(fi.COLUMNS) + 1
    assert fi.column_for(columns, ".github/workflows/ci-foo-bar.yaml", "CI: Foo Bar") is new
    # A stored discovered column survives, in order, after the registry.
    again = fi.known_columns({"columns": [dict(new), {"id": "lint", "label": "x", "paths": []}]})
    assert [c["id"] for c in again] == [c["id"] for c in fi.COLUMNS] + ["foo-bar"]
    assert again[5]["paths"] == fi.COLUMNS[5]["paths"]
    assert fi.COLUMNS[5]["group"] == "core" and {c["group"] for c in fi.COLUMNS} == {
        "core", "website", "billing", "validation"}


def test_first_bad_ignores_cancelled_and_pending_cells():
    def entry(sha, ct, **cells):
        return {"c": sha, "ct": ct, "m": {"pr": ct, "a": "me"},
                "w": {k: ({"c": None, "s": "in_progress"} if v == "pending" else {"c": v})
                      for k, v in cells.items()}}
    entries = [
        entry("1", 1, lint="success", unit="failure"),
        entry("2", 2, lint="failure", unit="cancelled"),
        entry("3", 3, lint="cancelled", unit="pending"),
        entry("4", 4, lint="pending", unit="failure"),
        entry("5", 5, lint="timed_out", unit="success"),
        entry("6", 6, lint="failure", unit="failure"),
    ]
    chains = fi.first_bad(list(reversed(entries)))
    assert chains["lint"] == {"commit": "2", "ct": 2, "prev_good": "1", "pr": 2, "author": "me",
                              "runs": 3}
    assert chains["unit"] == {"commit": "6", "ct": 6, "prev_good": "5", "pr": 6, "author": "me",
                              "runs": 1}
    assert fi.first_bad(entries[:1] + [entry("7", 7, unit="success")]) == {}


def test_unsettled_cursor():
    columns = fi.known_columns({})
    full = {cid: {"c": "success"} for cid in KNOWN_IDS}
    settled = {"c": SHA_A, "ct": NOW - 5 * H, "w": full}
    pending = {"c": SHA_B, "ct": NOW - 4 * H, "w": {**full, "lint": {"c": None, "s": "queued"}}}
    young_missing = {"c": SHA_C, "ct": NOW - H, "w": {k: v for k, v in full.items() if k != "lint"}}
    old_missing = {"c": "d" * 40, "ct": NOW - 9 * H, "w": {k: v for k, v in full.items() if k != "lint"}}
    no_website = {"c": "e" * 40, "ct": NOW - H, "w": {k: v for k, v in full.items() if k != "website-e2e"}}
    assert fi.unsettled_cursor([settled, old_missing, no_website], columns, NOW) is None
    assert fi.unsettled_cursor([settled, pending, young_missing], columns, NOW) == NOW - 4 * H
    assert fi.unsettled_cursor([young_missing], columns, NOW) == NOW - H


# -- gate -----------------------------------------------------------------------------
def test_parse_pin_quoted_unquoted_comment_and_absent():
    assert fi.parse_pin("env:\n  COMFYUI_PIN: b08e6cf3\n") == "b08e6cf3"
    assert fi.parse_pin(f"env:\n  COMFYUI_PIN: '{PIN}'\n") == PIN
    assert fi.parse_pin(f'env:\n  COMFYUI_PIN: "{PIN}" # core\n') == PIN
    assert fi.parse_pin("env:\n  # COMFYUI_PIN: deadbeef\n  OTHER: 1\n") is None
    assert fi.parse_pin("env:\n  COMFYUI_PIN: ${{ inputs.ref }}\n") is None
    assert fi.parse_pin("") is None and fi.parse_pin(None) is None


def test_gate_run_rows_final_failed_steps_and_dispatch_ref():
    listed = gate_list_run(39231543476, SHA_A, NOW - 4 * H)
    r = fi.gate_run(listed, gate_jobs(proofs="failure"), PIN)
    assert r["id"] == 39231543476 and r["t"] == NOW - 4 * H and r["d"] == 900
    assert r["ev"] == "schedule" and r["c"] == "failure" and r["sha"] == SHA_A
    assert r["pin"] == PIN and r["ref"] == PIN and r["rs"] == "pin"
    assert r["final"] == "failure" and r["jobs"] == [12, 12] and "s" not in r and "a" not in r
    assert set(r["rows"]["0"]) == set(SHARDS) and set(r["rows"]["0"].values()) == {"failure"}
    assert r["rows"]["15"] == {"core:2/2": "failure"} and r["rows"]["9"] == {"2/5": "failure"}
    assert r["fs"]["0"]["core:1/2"] == "Run custom-node suite"
    assert r["fs"]["9"]["2/5"] == "Assert deliberate failure independence"

    passing = fi.gate_run(gate_list_run(2, SHA_B, NOW - 3 * H, event="workflow_dispatch",
                                        conclusion="success", attempt=2),
                          gate_jobs("success", "success", "success"), PIN)
    assert passing["ref"] is None and passing["rs"] == "dispatch" and passing["pin"] == PIN
    assert passing["jobs"] == [12, 0] and passing["fs"] == {} and passing["a"] == 2
    assert fi.resolve_dispatch_ref(fi.GitHubApi(None), listed) is None     # phase 2 hook
    assert fi.clean_state(passing) == "pass" and fi.clean_state(r) == "fail"

    running = fi.gate_run(gate_list_run(3, SHA_C, NOW - H, status="in_progress"),
                          gate_jobs(status="in_progress"), None)
    assert running["s"] == "in_progress" and running["d"] is None and running["final"] is None
    assert set(running["rows"]["0"].values()) == {None} and not fi.gate_settled(running)
    assert fi.clean_state(running) is None

    no_jobs = fi.gate_run(listed, None, PIN)
    assert no_jobs["jobs"] is None and no_jobs["rows"] == {} and not fi.gate_settled(no_jobs)
    assert fi.gate_settled(r) and fi.clean_state(no_jobs) is None
    mixed = {"rows": {"0": {"a": "success", "b": "cancelled"}}}
    assert fi.clean_state(mixed) is None and fi.clean_state({"rows": {"1": {"a": "failure"}}}) is None


def test_streak_across_a_state_change_and_capped():
    def run(i, t, state):
        rows = {} if state is None else {"0": {s: ("failure" if state == "fail" else "success")
                                              for s in SHARDS}}
        return {"id": i, "t": t, "rows": rows}
    day = 86400
    runs = [run(5, NOW, "fail"), run(4, NOW - day, None), run(3, NOW - 2 * day + 1, "fail"),
            run(2, NOW - 3 * day, "pass"), run(1, NOW - 4 * day, "fail")]
    assert fi.streak(runs) == {"state": "fail", "since": NOW - 2 * day + 1, "days": 2, "runs": 2,
                               "capped": False}
    assert fi.streak(runs[:3]) == {"state": "fail", "since": NOW - 2 * day + 1, "days": 2,
                                   "runs": 2, "capped": True}
    assert fi.streak(runs[:1]) == {"state": "fail", "since": NOW, "days": 1, "runs": 1,
                                   "capped": True}
    assert fi.streak(runs[1:2]) is None and fi.streak([]) is None
    assert fi.streak([run(1, NOW, "pass"), run(0, NOW - 39 * day, "pass")])["days"] == 39
    # A dispatch run tests an unknown core ref: it neither starts nor breaks the streak.
    dispatch = {**run(6, NOW - day // 2, "pass"), "rs": "dispatch"}
    assert fi.streak([runs[0], dispatch] + runs[1:]) == fi.streak(runs)
    assert fi.streak([dispatch]) is None


# -- GitHub API client ---------------------------------------------------------------
def test_api_budget_errors_and_retry(monkeypatch, capsys):
    monkeypatch.setattr(fi.time, "sleep", lambda s: None)
    fake = FakeApi(monkeypatch)
    fake.on("/a", Resp(200, {"ok": 1}, remaining=400))
    fake.on("/b", Resp(200, {"ok": 2}, remaining=150))
    fake.on("/c", Resp(403, {"message": "rate limited"}))
    fake.on("/net", requests.ConnectionError("down"))
    calls = iter([Resp(502, {}), Resp(200, b"raw body")])
    fake.on("/flaky", lambda url, params: next(calls))
    api = fi.GitHubApi("tok", reserve=150)
    assert api.get(f"{API}/a") == {"ok": 1} and api.remaining == 400 and not api.stopped
    assert api.get(f"{API}/net") is None and not api.stopped
    assert api.get(f"{API}/flaky", raw=True) == b"raw body" and api.calls == 4
    assert api.get(f"{API}/missing") is None
    assert api.get(f"{API}/b") == {"ok": 2}            # the response that hit the reserve is kept
    assert api.stopped and "reserve" in api.stopped
    assert api.get(f"{API}/a") is None and api.calls == 6  # no request once stopped
    out = capsys.readouterr().out
    assert "::warning::frontend-index: fetching stopped" in out and "request failed" in out

    api = fi.GitHubApi(None, reserve=150)
    assert api.get(f"{API}/c") is None and api.stopped.startswith("GitHub API 403")
    api.external_calls(2)
    small = fi.GitHubApi(None, reserve=150)
    fake.on("/small", Resp(200, {}, remaining=30, limit=60))
    assert small.get(f"{API}/small") == {} and not small.stopped   # reserve scales to limit / 4
    small.external_calls(20)
    assert small.remaining == 10 and small.stopped                 # external calls are budgeted too


# -- end to end ------------------------------------------------------------------------
def standard_api(fake, now=NOW):
    perf_api(fake)
    runs = commit_runs(SHA_A, now - 2 * H, ci_tests_e2e={"conclusion": "failure"}) + \
        commit_runs(SHA_B, now - H)
    fake.on("/actions/runs", runs_page(runs))
    commit_api(fake, SHA_A, now - 2 * H)
    commit_api(fake, SHA_B, now - H, subject="Direct push", pr=None)
    gate_api(fake, [gate_list_run(901, SHA_A, now - 4 * H),
                    gate_list_run(900, SHA_C, now - 28 * H, event="workflow_dispatch",
                                  conclusion="success"),
                    gate_list_run(899, SHA_C, now - 30 * H, event="pull_request",
                                  conclusion="success")],
             {901: gate_jobs(), 900: gate_jobs("success", "success", "success")},
             {SHA_A: WORKFLOW_TEXT, SHA_C: WORKFLOW_TEXT.replace(PIN, "deadbeef1")})
    return fake


def test_first_tick_publishes_every_file_then_an_unchanged_tick_pushes_nothing(
        tmp_path, remote, monkeypatch, capsys):
    fake = standard_api(FakeApi(monkeypatch))
    assert fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW) == 0
    assert fake.count("/git/trees/") == 1 and fake.count("/git/blobs/") == 3
    assert fake.count(f"/workflows/{fi.GATE_WORKFLOW}/runs") == 1 and fake.count("/jobs") == 2
    assert fake.count("/contents/") == 2 and fake.count("/actions/runs?") == 1
    assert fake.count("/commits/") == 4 and len(fake.calls) == 14
    assert f"created=%3E%3D{iso(NOW - 6 * H).replace(':', '%3A')}" in fake.calls[9]
    assert "pushed results" in capsys.readouterr().out

    perf = remote_json(remote, "frontend/perf/head.json")
    assert perf["count"] == 3 and perf["latest"]["id"] == "20261019T130522Z-dd678395"
    assert perf["latest"]["cmp"]["canvas-idle"][0][3] == 2 and perf["shards"] == {"2026-10": 3}
    assert len(remote_json(remote, "frontend/perf/2026-10.json")["entries"]) == 3
    text = remote_text(remote, "frontend/ci/main.json")
    ci = json.loads(text)
    assert text == json.dumps(ci, separators=(",", ":"))                 # compact
    assert ci["schema_version"] == 1 and ci["repo"] == REPO and ci["updated_ts"] == NOW
    assert [c["id"] for c in ci["columns"]] == [c["id"] for c in fi.COLUMNS]
    assert ci["latest"] == {"c": SHA_B, "ct": NOW - H, "o": "success"}
    assert ci["count"] == 2 and ci["shards"] == {"2026-10": 2}
    assert ci["coverage"] == {"from_ts": NOW - 6 * H, "to_ts": NOW, "partial": False}
    b, a = ci["entries"]
    assert set(a["w"]) == KNOWN_IDS and set(b["w"]) == KNOWN_IDS
    assert a["o"] == "failure" and a["w"]["tests-e2e"]["c"] == "failure"
    assert a["m"] == {"s": "feat: thing (#21295)", "a": "terryjia", "pr": 21295, "ct": NOW - 2 * H,
                      "pt": "PR title", "pa": "terryjia", "av": "https://avatars/terryjia"}
    assert "mi" not in a and "mi" not in b and b["m"]["pr"] is None
    assert ci["first_bad"] == {}                     # B's green e2e run closed A's chain
    assert remote_json(remote, "frontend/ci/main/2026-10.json")["entries"] == ci["entries"]
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert [r["id"] for r in gate["runs"]] == [901, 900] and gate["pin"] == PIN
    assert gate["runs"][1]["pin"] == "deadbeef1" and gate["runs"][1]["ref"] is None
    # The passing dispatch run (900) tests an unknown ref, so it does not break the streak,
    # which therefore reaches the oldest pinned run and is capped.
    assert gate["streak"] == {"state": "fail", "since": NOW - 4 * H, "days": 1, "runs": 1,
                              "capped": True}
    assert gate["limit"] == 90 and gate["workflow"] == fi.GATE_WORKFLOW
    head_before = remote_head(remote)

    fake = standard_api(FakeApi(monkeypatch))
    assert fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800) == 0
    assert len(fake.calls) == 3
    assert "nothing new to publish" in capsys.readouterr().out
    assert remote_head(remote) == head_before


def test_in_progress_cell_completes_and_a_rerun_overrides_a_failure(tmp_path, remote, monkeypatch):
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(
        commit_runs(SHA_A, NOW - 2 * H, ci_tests_unit={"status": "in_progress"})
        + commit_runs(SHA_B, NOW - H, ci_tests_e2e={"conclusion": "failure"})))
    commit_api(fake, SHA_A, NOW - 2 * H)
    commit_api(fake, SHA_B, NOW - H)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    ci = remote_json(remote, "frontend/ci/main.json")
    assert ci["entries"][0]["o"] == "failure" and ci["entries"][1]["o"] == "pending"
    assert ci["first_bad"]["tests-e2e"]["commit"] == SHA_B and ci["latest"]["o"] == "failure"

    now2 = NOW + 1800
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([
        ci_run("ci-tests-unit.yaml", SHA_A, NOW - 2 * H),
        ci_run("ci-tests-e2e.yaml", SHA_B, NOW - H, attempt=2, start=NOW + 60)]))
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=now2)
    assert len(fake.calls) == 3          # unsettled cursor is still within the 6 h lookback
    assert f"created=%3E%3D{iso(now2 - 6 * H).replace(':', '%3A')}" in fake.calls[2]
    ci = remote_json(remote, "frontend/ci/main.json")
    b, a = ci["entries"]
    assert a["o"] == "success" and a["w"]["tests-unit"]["c"] == "success" and "s" not in a["w"]["tests-unit"]
    assert b["o"] == "success" and b["w"]["tests-e2e"] == {
        "id": b["w"]["tests-e2e"]["id"], "c": "success", "t": NOW + 60, "d": 400, "a": 2}
    assert set(a["w"]) == set(b["w"]) == KNOWN_IDS and ci["first_bad"] == {}
    assert ci["updated_ts"] == now2 and ci["latest"]["o"] == "success"


def test_backfill_merges_into_existing_shards_without_duplicates(tmp_path, remote, monkeypatch):
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_C, NOW - H)))
    commit_api(fake, SHA_C, NOW - H)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert remote_json(remote, "frontend/ci/main.json")["coverage"]["from_ts"] == NOW - 6 * H

    day = 86400
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_C, NOW - H)), created=f">={iso(NOW - 6 * H)}")
    fake.on("/actions/runs", runs_page(commit_runs(SHA_B, NOW - 20 * H) + commit_runs(SHA_C, NOW - H)),
            created=f"{iso(NOW - 30 * H)}..{iso(NOW - 6 * H)}")
    fake.on("/actions/runs", runs_page(commit_runs(SHA_A, NOW - 40 * H)),
            created=f"{iso(NOW - 2 * day)}..{iso(NOW - 30 * H)}")
    commit_api(fake, SHA_B, NOW - 20 * H)
    commit_api(fake, SHA_A, NOW - 40 * H)
    fi.run(args(backfill_days=2), store=_store(tmp_path, remote, "b"), now=NOW)
    assert fake.count("/actions/runs?") == 3 and fake.count("/commits/") == 4
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_C, SHA_B, SHA_A]
    assert all("mi" not in e for e in ci["entries"])
    assert ci["coverage"] == {"from_ts": NOW - 2 * day, "to_ts": NOW, "partial": False}
    shard = remote_json(remote, "frontend/ci/main/2026-10.json")["entries"]
    assert [e["c"] for e in shard] == [SHA_C, SHA_B, SHA_A] and ci["count"] == 3

    # Budget runs out after the first day window: the window counts, the
    # rest is flagged partial and a later backfill resumes from there.
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]), created=f">={iso(NOW - 6 * H)}")
    fake.on("/actions/runs", Resp(200, {"total_count": 0, "workflow_runs": []}, remaining=100),
            created=f"{iso(NOW - 3 * day)}..{iso(NOW - 2 * day)}")
    fi.run(args(backfill_days=4), store=_store(tmp_path, remote, "c"), now=NOW)
    assert fake.count("/actions/runs?") == 2
    assert remote_json(remote, "frontend/ci/main.json")["coverage"] == {
        "from_ts": NOW - 3 * day, "to_ts": NOW, "partial": True}
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]), created=f">={iso(NOW - 6 * H)}")
    fake.on("/actions/runs", runs_page([]), created=f"{iso(NOW - 4 * day)}..{iso(NOW - 3 * day)}")
    fi.run(args(backfill_days=4), store=_store(tmp_path, remote, "d"), now=NOW)
    assert fake.count("/actions/runs?") == 2
    assert remote_json(remote, "frontend/ci/main.json")["coverage"] == {
        "from_ts": NOW - 4 * day, "to_ts": NOW, "partial": False}


def test_backfill_splits_a_window_with_more_than_a_thousand_runs(tmp_path, remote, monkeypatch):
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_C, NOW - H)))
    commit_api(fake, SHA_C, NOW - H)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)

    # The day window reports 1,500 runs (more than ten pages hold): it is split
    # in half after its first page instead of being paged and dropped, and the
    # halves list completely, so the backfill reaches its target.
    day = 86400
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_C, NOW - H)), created=f">={iso(NOW - 6 * H)}")
    fake.on("/actions/runs", Resp(200, {"total_count": 1500, "workflow_runs": []}),
            created=f"{iso(NOW - day)}..{iso(NOW - 6 * H)}")
    fake.on("/actions/runs", runs_page(commit_runs(SHA_A, NOW - 20 * H)),
            created=f"{iso(NOW - day)}..{iso(NOW - 15 * H)}")
    fake.on("/actions/runs", runs_page(commit_runs(SHA_B, NOW - 10 * H)),
            created=f"{iso(NOW - 15 * H)}..{iso(NOW - 6 * H)}")
    commit_api(fake, SHA_A, NOW - 20 * H)
    commit_api(fake, SHA_B, NOW - 10 * H)
    fi.run(args(backfill_days=1), store=_store(tmp_path, remote, "b"), now=NOW)
    assert fake.count("/actions/runs?") == 4
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_C, SHA_B, SHA_A]
    assert ci["coverage"] == {"from_ts": NOW - day, "to_ts": NOW, "partial": False}


def test_window_truncates_an_hour_that_still_overflows(monkeypatch, capsys):
    fake = FakeApi(monkeypatch)
    full_page = [ci_run("ci-tests-unit.yaml", SHA_A, NOW - 2 * H, run_id=1000 + i) for i in range(100)]
    fake.on("/actions/runs", Resp(200, {"total_count": 1500, "workflow_runs": full_page}),
            created=f"{iso(NOW - 2 * H)}..{iso(NOW - 2 * H + 1800)}")
    api = fi.GitHubApi(None, reserve=150)
    fetch = fi.CiFetch(fi.known_columns({}))
    # Half an hour cannot be split further: it is paged to the limit, what was
    # listed is kept, and the window still counts as covered.
    assert fi._window(api, REPO, fetch, NOW - 2 * H, NOW - 2 * H + 1800) is True
    assert fake.count("/actions/runs?") == fi.BACKFILL_PAGE_LIMIT and SHA_A in fetch.entries
    assert "truncated" in capsys.readouterr().out


def test_cursor_follows_the_last_complete_listing_across_long_gaps(tmp_path, remote, monkeypatch,
                                                                   capsys):
    day = 86400
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_A, NOW - H)), created=f">={iso(NOW - 6 * H)}")
    commit_api(fake, SHA_A, NOW - H)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert remote_json(remote, "frontend/ci/main.json")["coverage"] == {
        "from_ts": NOW - 6 * H, "to_ts": NOW, "partial": False}

    # Ten hours later (a stalled cron): the listing starts at the last
    # complete listing rather than six hours back, so the commit that
    # landed in between is indexed instead of silently skipped.
    now2 = NOW + 10 * H
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_B, NOW + 2 * H)), created=f">={iso(NOW)}")
    commit_api(fake, SHA_B, NOW + 2 * H)
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=now2)
    assert fake.count("/actions/runs?") == 1
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_B, SHA_A]
    assert ci["coverage"] == {"from_ts": NOW - 6 * H, "to_ts": now2, "partial": False}

    # Three days away: the listing since to_ts overflows its pages, so the
    # gap is closed from to_ts in day windows. The budget ends after the
    # first window; coverage stops there and the next tick resumes from it.
    now3 = now2 + 3 * day
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    full = (commit_runs(SHA_C, now3 - H) * 6)[:100]
    fake.on("/actions/runs", lambda url, params: runs_page(full), created=f">={iso(now2)}")
    fake.on("/actions/runs", Resp(200, {"total_count": 17, "workflow_runs": commit_runs("d" * 40, now2 + 5 * H)},
                                  remaining=150), created=f"{iso(now2)}..{iso(now2 + day)}")
    fi.run(args(), store=_store(tmp_path, remote, "c"), now=now3)
    assert fake.count("/actions/runs?") == 6            # five full pages, then one day window
    assert f"the index is complete up to {iso(now2 + day)}" in capsys.readouterr().out
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_C, "d" * 40, SHA_B, SHA_A]
    assert ci["coverage"] == {"from_ts": NOW - 6 * H, "to_ts": now2 + day, "partial": False}

    now4 = now3 + 1800
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs("e" * 40, now2 + 2 * day)),
            created=f">={iso(now2 + day)}")
    fi.run(args(), store=_store(tmp_path, remote, "d"), now=now4)
    assert fake.count("/actions/runs?") == 1
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_C, "e" * 40, "d" * 40, SHA_B, SHA_A]
    assert ci["coverage"] == {"from_ts": NOW - 6 * H, "to_ts": now4, "partial": False}

    # to_ts alone never causes a write: a tick that learns nothing else
    # pushes nothing and the next cursor starts from the stored to_ts.
    head_before = remote_head(remote)
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]), created=f">={iso(now4)}")
    fi.run(args(), store=_store(tmp_path, remote, "e"), now=now4 + 8 * H)
    assert fake.count("/actions/runs?") == 1 and remote_head(remote) == head_before
    assert remote_json(remote, "frontend/ci/main.json")["coverage"]["to_ts"] == now4

    # A head written before to_ts existed counts as complete up to its last tick.
    assert fi._coverage({"coverage": {"from_ts": 5, "partial": True}, "updated_ts": 9}, 20) == {
        "from_ts": 5, "to_ts": 9, "partial": True}
    assert fi._coverage({}, 20) == {"from_ts": 20, "to_ts": 20, "partial": False}


def test_budget_exhaustion_publishes_partial_results_then_resumes(tmp_path, remote, monkeypatch,
                                                                  capsys):
    fake = FakeApi(monkeypatch)
    perf_api(fake)
    fake.routes = [(u, p, r) for u, p, r in fake.routes if "17b0d4e4" not in u]
    fake.on("/git/blobs/blob-17b0d4e4", Resp(200, (FIX / PERF_FILES[0]).read_bytes(), remaining=150))
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    assert fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW) == 0
    assert len(fake.calls) == 2           # tree + first blob, then nothing else is requested
    out = capsys.readouterr().out
    assert "fetching stopped: rate budget exhausted" in out and "incomplete tick" in out
    head = remote_json(remote, "frontend/perf/head.json")
    assert [e["id"] for e in head["entries"]] == ["20261019T103805Z-17b0d4e4"]
    ci = remote_json(remote, "frontend/ci/main.json")
    assert ci["count"] == 0 and ci["coverage"] == {"from_ts": NOW, "to_ts": NOW, "partial": False}

    fake = FakeApi(monkeypatch)
    perf_api(fake)
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count("/git/blobs/") == 2 and len(fake.calls) == 5
    head = remote_json(remote, "frontend/perf/head.json")
    assert len(head["entries"]) == 3 and head["latest"]["n_hist"] == 2
    ci = remote_json(remote, "frontend/ci/main.json")
    assert ci["count"] == 0 and ci["coverage"] == {"from_ts": NOW + 1800 - 6 * H, "to_ts": NOW + 1800,
                                                   "partial": False}


def test_blob_404_is_skipped_and_retried(tmp_path, remote, monkeypatch, capsys):
    fake = FakeApi(monkeypatch)
    perf_api(fake, blob_status={PERF_FILES[1]: 404})
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert "GitHub API 404" in capsys.readouterr().out
    head = remote_json(remote, "frontend/perf/head.json")
    assert [e["id"] for e in head["entries"]] == ["20261019T130522Z-dd678395",
                                                   "20261019T103805Z-17b0d4e4"]
    assert head["latest"]["n_hist"] == 1 and head["count"] == 2

    fake = FakeApi(monkeypatch)
    perf_api(fake)
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count("/git/blobs/") == 1 and len(fake.calls) == 4
    head = remote_json(remote, "frontend/perf/head.json")
    assert [e["id"] for e in head["entries"]] == [
        "20261019T130522Z-dd678395", "20261019T105541Z-e38b73dd", "20261019T103805Z-17b0d4e4"]
    assert set("".join(head["entries"][1]["cls"].values())) == {"-"}    # one older entry only
    assert head["latest"]["n_hist"] == 1                                 # newest was not new


def test_provisional_meta_is_upgraded_within_the_cap(tmp_path, remote, monkeypatch):
    monkeypatch.setattr(fi, "META_UPGRADES_PER_TICK", 2)
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_A, NOW - 3 * H) + commit_runs(SHA_B, NOW - 2 * H)
                                       + commit_runs(SHA_C, NOW - H)))
    for sha, ct in ((SHA_A, NOW - 3 * H), (SHA_B, NOW - 2 * H), (SHA_C, NOW - H)):
        commit_api(fake, sha, ct)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert fake.count("/commits/") == 4 and fake.count(f"/commits/{SHA_A}") == 0
    c, b, a = remote_json(remote, "frontend/ci/main.json")["entries"]
    assert "mi" not in c and "mi" not in b and a["mi"] is True and a["m"]["a"] == "Terry Jia"

    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    commit_api(fake, SHA_A, NOW - 3 * H)
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count(f"/commits/{SHA_A}") == 2 and len(fake.calls) == 5
    entries = remote_json(remote, "frontend/ci/main.json")["entries"]
    assert all("mi" not in e for e in entries) and entries[2]["m"]["a"] == "terryjia"


def test_provisional_meta_in_a_shard_is_upgraded(tmp_path, remote, monkeypatch):
    monkeypatch.setattr(fi, "CI_HEAD_LIMIT", 1)
    monkeypatch.setattr(fi, "META_UPGRADES_PER_TICK", 1)
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page(commit_runs(SHA_A, NOW - 3 * H) + commit_runs(SHA_B, NOW - 2 * H)))
    commit_api(fake, SHA_A, NOW - 3 * H)
    commit_api(fake, SHA_B, NOW - 2 * H)
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert fake.count(f"/commits/{SHA_B}") == 2 and fake.count(f"/commits/{SHA_A}") == 0
    ci = remote_json(remote, "frontend/ci/main.json")
    assert [e["c"] for e in ci["entries"]] == [SHA_B] and ci["count"] == 2
    shard = remote_json(remote, "frontend/ci/main/2026-10.json")["entries"]
    assert [e["c"] for e in shard] == [SHA_B, SHA_A] and shard[1]["mi"] is True

    # A now lives only in the monthly shard; the next tick still upgrades it.
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    commit_api(fake, SHA_A, NOW - 3 * H)
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count(f"/commits/{SHA_A}") == 2 and len(fake.calls) == 5
    shard = remote_json(remote, "frontend/ci/main/2026-10.json")["entries"]
    assert [e["c"] for e in shard] == [SHA_B, SHA_A] and all("mi" not in e for e in shard)
    assert shard[1]["m"]["a"] == "terryjia" and shard[1]["m"]["pt"] == "PR title"
    assert [e["c"] for e in remote_json(remote, "frontend/ci/main.json")["entries"]] == [SHA_B]

    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    gate_api(fake, [], {}, {})
    fake.on("/actions/runs", runs_page([]))
    fi.run(args(), store=_store(tmp_path, remote, "c"), now=NOW + 3600)
    assert fake.count("/commits/") == 0 and len(fake.calls) == 3


def test_gate_filters_pr_runs_and_refetches_only_unsettled_runs(tmp_path, remote, monkeypatch):
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, [gate_list_run(3, SHA_C, NOW - H, event="pull_request", conclusion="success"),
                    gate_list_run(2, SHA_B, NOW - 2 * H, status="in_progress"),
                    gate_list_run(1, SHA_A, NOW - 26 * H)],
             {2: gate_jobs(status="in_progress"), 1: gate_jobs(), 3: gate_jobs("success")},
             {SHA_A: WORKFLOW_TEXT, SHA_B: WORKFLOW_TEXT})
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert fake.count("/jobs") == 2 and fake.count("/contents/") == 2 and fake.count("/runs/3/") == 0
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert [r["id"] for r in gate["runs"]] == [2, 1] and gate["runs"][0]["s"] == "in_progress"
    assert gate["streak"] == {"state": "fail", "since": NOW - 26 * H, "days": 1, "runs": 1,
                              "capped": True}

    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, [gate_list_run(2, SHA_B, NOW - 2 * H, conclusion="success"),
                    gate_list_run(1, SHA_A, NOW - 26 * H)],
             {2: gate_jobs("success", "success", "success")}, {})
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count("/jobs") == 1 and fake.count("/contents/") == 0 and len(fake.calls) == 4
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    done = gate["runs"][0]
    assert "s" not in done and done["c"] == "success" and done["pin"] == PIN and done["d"] == 900
    assert gate["streak"] == {"state": "pass", "since": NOW - 2 * H, "days": 1, "runs": 1,
                              "capped": False}

    # A rerun (new attempt) of a settled run is fetched again.
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, [gate_list_run(2, SHA_B, NOW - 2 * H, conclusion="success"),
                    gate_list_run(1, SHA_A, NOW - 26 * H, conclusion="success", attempt=2)],
             {1: gate_jobs("success", "success", "success")}, {})
    fi.run(args(), store=_store(tmp_path, remote, "c"), now=NOW + 3600)
    assert fake.count("/jobs") == 1 and fake.count("/runs/1/") == 1
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert gate["runs"][1]["a"] == 2 and gate["streak"]["runs"] == 2 and gate["streak"]["capped"]


def test_gate_pin_is_recovered_after_a_budget_stop_or_a_failed_read(tmp_path, remote, monkeypatch):
    # The jobs response exhausts the budget: the run is stored with its
    # jobs and no pin, and the next tick reads only the workflow file.
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, [gate_list_run(901, SHA_A, NOW - 4 * H)], {}, {SHA_A: WORKFLOW_TEXT})
    fake.on("/actions/runs/901/jobs", Resp(200, {"total_count": 13, "jobs": gate_jobs()}, remaining=150))
    fi.run(args(), store=_store(tmp_path, remote, "a"), now=NOW)
    assert fake.count("/jobs") == 1 and fake.count("/contents/") == 0
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert gate["runs"][0]["jobs"] == [12, 12] and gate["runs"][0]["pin"] is None
    assert gate["runs"][0]["ref"] is None and gate["pin"] is None

    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, [gate_list_run(901, SHA_A, NOW - 4 * H)], {}, {SHA_A: WORKFLOW_TEXT})
    fi.run(args(), store=_store(tmp_path, remote, "b"), now=NOW + 1800)
    assert fake.count("/jobs") == 0 and fake.count("/contents/") == 1 and len(fake.calls) == 4
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert gate["runs"][0]["pin"] == PIN and gate["runs"][0]["ref"] == PIN and gate["pin"] == PIN
    assert gate["runs"][0]["jobs"] == [12, 12]

    # A failed read is not cached: the second run of the same sha tries
    # again in the same tick, and the next tick repairs both with one read
    # (a dispatch run gets its pin but keeps ref null).
    listed = [gate_list_run(903, SHA_B, NOW - 3 * H),
              gate_list_run(902, SHA_B, NOW - 27 * H, event="workflow_dispatch", conclusion="success"),
              gate_list_run(901, SHA_A, NOW - 4 * H)]
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, listed, {903: gate_jobs(), 902: gate_jobs("success", "success", "success")}, {})
    fi.run(args(), store=_store(tmp_path, remote, "c"), now=NOW + 3600)
    assert fake.count("/jobs") == 2 and fake.count("/contents/") == 2
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert [(r["id"], r["pin"]) for r in gate["runs"]] == [(903, None), (901, PIN), (902, None)]
    assert gate["pin"] == PIN

    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, listed, {}, {SHA_B: WORKFLOW_TEXT.replace(PIN, "deadbeef1")})
    fi.run(args(), store=_store(tmp_path, remote, "d"), now=NOW + 5400)
    assert fake.count("/jobs") == 0 and fake.count("/contents/") == 1 and len(fake.calls) == 4
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert [(r["pin"], r["ref"]) for r in gate["runs"]] == [
        ("deadbeef1", "deadbeef1"), (PIN, PIN), ("deadbeef1", None)]
    assert gate["pin"] == "deadbeef1" and gate["runs"][0]["jobs"] == [12, 12]

    # A file read without a pin is cached for the tick: one read for two
    # runs of that sha, not one per run.
    listed = [gate_list_run(905, SHA_C, NOW - 2 * H), gate_list_run(904, SHA_C, NOW - 26 * H)] + listed
    fake = FakeApi(monkeypatch)
    perf_api(fake, [])
    fake.on("/actions/runs", runs_page([]))
    gate_api(fake, listed, {905: gate_jobs(), 904: gate_jobs()}, {SHA_C: "env:\n  OTHER: 1\n"})
    fi.run(args(), store=_store(tmp_path, remote, "e"), now=NOW + 7200)
    assert fake.count("/jobs") == 2 and fake.count("/contents/") == 1
    gate = remote_json(remote, "frontend/ci/custom-nodes.json")
    assert [r["id"] for r in gate["runs"]] == [905, 903, 901, 904, 902]
    assert gate["runs"][0]["pin"] is None and gate["runs"][3]["pin"] is None
    assert gate["pin"] == "deadbeef1"


def test_only_and_skip_publish_run_the_writers_locally(tmp_path, remote, monkeypatch, capsys):
    fake = FakeApi(monkeypatch)
    perf_api(fake)
    head_before = remote_head(remote)
    store = _store(tmp_path, remote, "a")
    assert fi.run(args(only="perf", skip_publish=True), store=store, now=NOW) == 0
    assert len(fake.calls) == 4 and "dry run" in capsys.readouterr().out
    assert remote_head(remote) == head_before
    local = json.loads((store.workdir / "frontend" / "perf" / "head.json").read_text())
    assert local["count"] == 3 and (store.workdir / "frontend" / "perf" / "2026-10.json").exists()
    assert not (store.workdir / "frontend" / "ci").exists()
    with pytest.raises(SystemExit):
        fi.run(args(only="perf,bogus"), store=store, now=NOW)


def test_parse_args_reads_the_dispatch_inputs_from_env(monkeypatch):
    monkeypatch.setenv("FRONTEND_BACKFILL_DAYS", "90")
    monkeypatch.setenv("FRONTEND_ONLY", "ci")
    a = fi.parse_args([])
    assert a.backfill_days == 90 and a.only == "ci" and a.prefix == "frontend"
    assert a.repo == REPO and a.budget_reserve == 150 and a.storage == "github"
    monkeypatch.setenv("FRONTEND_BACKFILL_DAYS", "")
    monkeypatch.delenv("FRONTEND_ONLY")
    a = fi.parse_args(["--backfill-days", "7"])
    assert a.backfill_days == 7 and a.only == "perf,ci,gate"
