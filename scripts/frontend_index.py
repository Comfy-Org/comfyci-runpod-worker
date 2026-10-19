"""Frontend CI index: ComfyUI_frontend's CI state on main, its perf baselines
and the daily custom-nodes gate, published as compact static JSON under
frontend/ on the results branch for the dashboard's Frontend tab.

Files (compact JSON, UTC epoch-second ints, schema_version 1; every head
carries updated_ts):
  frontend/perf/head.json          metric registry, rule, scenarios, newest
                                   PERF_HEAD_LIMIT entries, latest comparison
  frontend/perf/<YYYY-MM>.json     complete monthly shards (newest first)
  frontend/ci/main.json            column registry, newest CI_HEAD_LIMIT main
                                   commits, first_bad per column, coverage
  frontend/ci/main/<YYYY-MM>.json  complete monthly shards
  frontend/ci/custom-nodes.json    newest GATE_LIMIT gate runs, pin, streak

Sources, all public and read with the Actions token: the runs API for pushes
to main, the perf-data branch (baselines/perf-<ts>-<sha8>.json, of which the
frontend keeps the newest 20) and the ci-tests-custom-nodes.yaml runs with
their jobs.

A tick is a fetch phase (GitHubApi, budgeted by X-RateLimit-Remaining) and a
write phase. The three writers are derived writers: they re-run on the
freshest tree when a push retries behind a concurrent publisher, merge by key
and only write a file whose content changed (updated_ts aside), so a tick
with nothing new pushes nothing. An exhausted budget or a failed request is a
warning: what was collected is published and the rest waits for the next
tick.

Usage:
  python frontend_index.py [--backfill-days N] [--only perf,ci,gate] [--skip-publish]
Env: FRONTEND_GH_TOKEN (else GITHUB_TOKEN / GH_TOKEN) for the frontend repo
reads; FRONTEND_BACKFILL_DAYS and FRONTEND_ONLY mirror the two flags for the
workflow's dispatch inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import requests

import commit_meta
import storage
from commit_meta import _ts, pr_number
from index_builder import compact_meta, month_of

FRONTEND_REPO = "Comfy-Org/ComfyUI_frontend"
GITHUB_API = commit_meta.GITHUB_API
SCHEMA_VERSION = 1
TIMEOUT_S = 30
BUDGET_RESERVE = 150

# -- perf ---------------------------------------------------------------------
PERF_BRANCH = "perf-data"
PERF_HEAD_LIMIT = 100
PERF_WINDOW = 20
PERF_RULE = {"cv_noisy_pct": 50, "z": 2}
PERF_STATS = {"current": "median", "history": "mean", "stddev": "sample",
              "delta_base": "history_mean"}
TARGETS = {"p5_fps": 52}
# The frontend's REPORTED_METRICS (scripts/perf-report.ts), in its order.
METRICS = [
    {"key": "frameDurationMs", "label": "avg frame time", "unit": "ms", "min_abs_delta": None},
    {"key": "p95FrameDurationMs", "label": "p95 frame time", "unit": "ms", "min_abs_delta": None},
    {"key": "layoutDurationMs", "label": "layout duration", "unit": "ms", "min_abs_delta": None},
    {"key": "styleRecalcDurationMs", "label": "style recalc duration", "unit": "ms",
     "min_abs_delta": None},
    {"key": "layouts", "label": "layout count", "unit": "", "min_abs_delta": 5},
    {"key": "styleRecalcs", "label": "style recalc count", "unit": "", "min_abs_delta": 5},
    {"key": "taskDurationMs", "label": "task duration", "unit": "ms", "min_abs_delta": None},
    {"key": "scriptDurationMs", "label": "script duration", "unit": "ms", "min_abs_delta": None},
    {"key": "totalBlockingTimeMs", "label": "TBT", "unit": "ms", "min_abs_delta": None},
    {"key": "heapUsedBytes", "label": "heap used", "unit": "bytes", "min_abs_delta": None},
    {"key": "domNodes", "label": "DOM nodes", "unit": "", "min_abs_delta": 5},
    {"key": "eventListeners", "label": "event listeners", "unit": "", "min_abs_delta": 5},
]
CLS = {"regression": "r", "improvement": "i", "neutral": "n", "noisy": "x"}
INSUFFICIENT = "-"
BASELINE_RE = re.compile(r"^baselines/perf-(\d{8}T\d{6}Z-[0-9a-f]{7,40})\.json$")

# -- ci -----------------------------------------------------------------------
CI_BRANCH = "main"
CI_HEAD_LIMIT = 300
CI_PAGE_LIMIT = 5
BACKFILL_PAGE_LIMIT = 10
# A backfill window that holds more runs than BACKFILL_PAGE_LIMIT pages can
# list is halved until it fits or is this wide; narrower is never needed.
MIN_WINDOW_S = 3600
META_UPGRADES_PER_TICK = 25
UNSETTLED_AGE_S = 3 * 3600
LOOKBACK_S = 6 * 3600
RED = ("failure", "timed_out", "startup_failure")


def _column(cid: str, label: str, group: str, *files: str) -> dict:
    return {"id": cid, "label": label, "paths": [f".github/workflows/{f}" for f in files],
            "name": f"CI: {label}", "group": group}


# Columns are keyed by workflow path: lint runs under ci-lint-format-queue.yaml
# on main and ci-lint-format.yaml on PRs with the same display name.
COLUMNS = [
    _column("tests-unit", "Tests Unit", "core", "ci-tests-unit.yaml"),
    _column("tests-e2e", "Tests E2E", "core", "ci-tests-e2e.yaml"),
    _column("tests-storybook", "Tests Storybook", "core", "ci-tests-storybook.yaml"),
    _column("perf-report", "Performance Report", "core", "ci-perf-report.yaml"),
    _column("ecosystem-matrix", "Custom Nodes Ecosystem Matrix", "core",
            "ci-ecosystem-matrix.yaml"),
    _column("lint", "Lint Format", "core", "ci-lint-format-queue.yaml", "ci-lint-format.yaml"),
    _column("json-validation", "JSON Validation", "core", "ci-json-validation.yaml"),
    _column("size-data", "Size Data", "core", "ci-size-data.yaml"),
    _column("website-unit", "Website Unit", "website", "ci-website-unit.yaml"),
    _column("website-e2e", "Website E2E", "website", "ci-website-e2e.yaml"),
    _column("website-build", "Website Build", "website", "ci-website-build.yaml"),
    _column("billing-web-unit", "Billing Web Unit", "billing", "ci-billing-web-unit.yaml"),
    _column("billing-web-e2e", "Billing Web E2E", "billing", "ci-billing-web-e2e.yaml"),
    _column("oss-assets-validation", "OSS Assets Validation", "validation",
            "ci-oss-assets-validation.yaml"),
    _column("adr-links-validation", "ADR Link Validation", "validation",
            "ci-adr-links-validation.yaml"),
]

# -- gate ---------------------------------------------------------------------
GATE_WORKFLOW = "ci-tests-custom-nodes.yaml"
GATE_LIMIT = 90
GATE_PAGE_LIMIT = 6
GATE_EVENTS = ("schedule", "workflow_dispatch")
GATE_VERDICT_JOB = "E2E Custom Nodes Test"
GATE_JOB_RE = re.compile(r"^custom_nodes_e2e \((?P<shard>[^,]+), (?P<row>\d+)\)$")
PIN_RE = re.compile(r"^\s*COMFYUI_PIN:\s*['\"]?([0-9a-f]{7,40})", re.MULTILINE)


# -- paths --------------------------------------------------------------------
def perf_head_path() -> str:
    return "frontend/perf/head.json"


def perf_shard_path(month: str) -> str:
    return f"frontend/perf/{month}.json"


def ci_head_path() -> str:
    return "frontend/ci/main.json"


def ci_shard_path(month: str) -> str:
    return f"frontend/ci/main/{month}.json"


def gate_path() -> str:
    return "frontend/ci/custom-nodes.json"


# -- GitHub API ---------------------------------------------------------------
class GitHubApi:
    """requests.get against the GitHub REST API with a per-tick budget.

    get() returns the decoded body (bytes when raw) or None for any failure:
    a non-200 status, a 5xx that survives one retry, a network error. The
    budget comes from X-RateLimit-Remaining: once it falls to the reserve, or
    the API answers 403/429, fetching stops (later calls return None without
    a request) and `stopped` names the reason, so the tick publishes what it
    collected and the rest waits for the next one.
    """

    def __init__(self, token: str | None = None, reserve: int = BUDGET_RESERVE):
        self.token = token
        self.reserve = reserve
        self.calls = 0
        self.remaining: int | None = None
        self.stopped: str | None = None

    def _headers(self, raw: bool) -> dict:
        h = {"Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _stop(self, reason: str):
        if not self.stopped:
            self.stopped = reason
            print(f"::warning::frontend-index: fetching stopped: {reason}")

    def _account(self, headers) -> None:
        rem = headers.get("X-RateLimit-Remaining")
        if rem is None or not str(rem).isdigit():
            return
        self.remaining = int(rem)
        limit = headers.get("X-RateLimit-Limit")
        reserve = self.reserve
        if limit is not None and str(limit).isdigit() and int(limit) < 4 * reserve:
            reserve = int(limit) // 4  # unauthenticated or small limits
        if self.remaining <= reserve:
            self._stop(f"rate budget exhausted ({self.remaining} requests left, "
                       f"reserve {reserve})")

    def external_calls(self, n: int):
        """Account for requests made outside this client (commit_meta)."""
        self.calls += n
        if self.remaining is not None:
            self.remaining -= n
            if self.remaining <= self.reserve:
                self._stop(f"rate budget exhausted ({self.remaining} requests left, "
                           f"reserve {self.reserve})")

    def get(self, url: str, params: dict | None = None, raw: bool = False):
        if self.stopped:
            return None
        for attempt in (1, 2):
            self.calls += 1
            try:
                r = requests.get(url, params=params, headers=self._headers(raw),
                                 timeout=TIMEOUT_S)
            except requests.RequestException as e:
                print(f"::warning::frontend-index: request failed for {url}: {e!r}")
                return None
            self._account(r.headers)
            if r.status_code in (403, 429):
                self._stop(f"GitHub API {r.status_code} for {url}")
                return None
            if r.status_code >= 500 and attempt == 1:
                time.sleep(1)
                continue
            if r.status_code != 200:
                print(f"::warning::frontend-index: GitHub API {r.status_code} for {url}")
                return None
            if raw:
                return r.content
            try:
                return r.json()
            except ValueError as e:
                print(f"::warning::frontend-index: bad JSON from {url}: {e!r}")
                return None
        return None


# -- perf: stats (mirror of the frontend's perf-stats.ts / perf-report.ts) ----
def compute_stats(values: list[float]) -> dict:
    """{mean, stddev, n} with the sample standard deviation (n - 1). Sums
    are accumulated left to right like the frontend's reduce(), not with
    Python's compensated sum(), so a near-constant series gets the same
    floating-point residue (and the same verdict) in both."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "stddev": 0.0, "n": 0}
    if n == 1:
        return {"mean": values[0], "stddev": 0.0, "n": 1}
    total = 0.0
    for v in values:
        total += v
    mean = total / n
    squares = 0.0
    for v in values:
        squares += (v - mean) * (v - mean)
    return {"mean": mean, "stddev": math.sqrt(squares / (n - 1)), "n": n}


def z_score(value: float, stats: dict) -> float | None:
    if stats["stddev"] == 0 or stats["n"] < 2:
        return None
    return (value - stats["mean"]) / stats["stddev"]


def coefficient_of_variation(stats: dict) -> float:
    return stats["stddev"] / stats["mean"] * 100 if stats["mean"] > 0 else 0.0


def classify_change(z: float | None, cv: float, abs_delta: float | None = None,
                    min_abs_delta: float | None = None) -> str:
    """regression / improvement / neutral / noisy, in the frontend's order of
    checks: the CV gate first, then the effect-size gate for count metrics,
    then the strict +-2 z-score rule."""
    if cv > PERF_RULE["cv_noisy_pct"]:
        return "noisy"
    if z is None:
        return "neutral"
    if min_abs_delta is not None and abs_delta is not None and abs(abs_delta) < min_abs_delta:
        return "neutral"
    if z > PERF_RULE["z"]:
        return "regression"
    if z < -PERF_RULE["z"]:
        return "improvement"
    return "neutral"


def compare_metric(current: float | None, history: list[float],
                   min_abs_delta: float | None = None) -> list:
    """One latest.cmp row: [current, hist_mean, hist_stddev, hist_n, z, cls].
    cls is '-' when there is no current value or fewer than two history
    values (the frontend would say neutral; the distinction is kept)."""
    stats = compute_stats(history)
    if current is None or stats["n"] < 2:
        return [current, stats["mean"] if stats["n"] else None, None, stats["n"], None,
                INSUFFICIENT]
    z = z_score(current, stats)
    sig = classify_change(z, coefficient_of_variation(stats), current - stats["mean"],
                          min_abs_delta)
    return [current, stats["mean"], stats["stddev"], stats["n"], z, CLS[sig]]


# -- perf: entries ------------------------------------------------------------
def baseline_id(path: str) -> str | None:
    """'<ts>-<sha8>' from a perf-data tree path, None for any other file."""
    m = BASELINE_RE.match(path)
    return m.group(1) if m else None


def id_ts(bid: str) -> int | None:
    try:
        return int(datetime.strptime(bid.split("-", 1)[0], "%Y%m%dT%H%M%SZ")
                   .replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def _finite(sample: dict, key: str) -> float | None:
    v = sample.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _values(samples: list[dict], key: str) -> list[float]:
    return [v for v in (_finite(s, key) for s in samples) if v is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    mid = len(vs) // 2
    return (vs[mid - 1] + vs[mid]) / 2 if len(vs) % 2 == 0 else vs[mid]


def _round(v: float | None, unit: str):
    """Stored precision: bytes to an integer (half up, like Math.round),
    everything else to 8 significant digits. Two decimals would quantise
    frame times (16.67 in every run) and turn the frontend's z-score into
    noise; 8 digits keep its verdicts while halving the entry size."""
    if v is None:
        return None
    return int(math.floor(v + 0.5)) if unit == "bytes" else float(f"{v:.8g}")


def scenario_samples(raw: dict) -> dict[str, list[dict]]:
    """Measurements grouped by scenario name, in first-seen order."""
    groups: dict[str, list[dict]] = {}
    for m in raw.get("measurements") or []:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            groups.setdefault(m["name"], []).append(m)
    return groups


def perf_medians(raw: dict) -> dict[str, list[float | None]]:
    """Per scenario, the unrounded median of each metric's samples (the
    frontend's 'current' value)."""
    return {name: [_median(_values(samples, m["key"])) for m in METRICS]
            for name, samples in scenario_samples(raw).items()}


def perf_entry(bid: str, raw: dict) -> dict:
    """Compact entry for one baseline: per-scenario metric means (rounded,
    null when absent), the modal sample count n and the exceptions nx. The
    writer adds cls, which needs the history."""
    groups = scenario_samples(raw)
    counts = Counter(len(s) for s in groups.values())
    n = counts.most_common(1)[0][0] if counts else 0
    nx = {name: len(s) for name, s in groups.items() if len(s) != n}
    entry = {
        "id": bid,
        "c": raw.get("gitSha") or None,
        "t": _ts(raw.get("timestamp")) or id_ts(bid) or 0,
        "n": n,
        "s": {name: [_round(_mean(_values(samples, m["key"])), m["unit"]) for m in METRICS]
              for name, samples in groups.items()},
    }
    if nx:
        entry["nx"] = nx
    return entry


def perf_key(e: dict):
    return (e.get("t") or 0, e["id"])


def preceding(entries: list[dict], entry: dict, window: int = PERF_WINDOW) -> list[dict]:
    """The `window` entries before `entry` in time order, newest first; the
    entry itself is never part of its own history."""
    key = perf_key(entry)
    older = [e for e in entries if e["id"] != entry["id"] and perf_key(e) < key]
    older.sort(key=perf_key, reverse=True)
    return older[:window]


def compare_entry(medians: dict[str, list[float | None]],
                  history: list[dict]) -> dict[str, list[list]]:
    """latest.cmp rows per scenario: the current medians against the history
    entries' stored means, one row per metric in METRICS order."""
    cmp = {}
    for scenario, meds in medians.items():
        rows = []
        for i, m in enumerate(METRICS):
            hist = [e["s"][scenario][i] for e in history
                    if scenario in (e.get("s") or {}) and e["s"][scenario][i] is not None]
            cur, mean, std, n, z, cls = compare_metric(meds[i], hist, m["min_abs_delta"])
            rows.append([_round(cur, m["unit"]), _round(mean, m["unit"]), _round(std, m["unit"]),
                         n, None if z is None else round(z, 2), cls])
        cmp[scenario] = rows
    return cmp


def classification(cmp: dict[str, list[list]]) -> dict[str, str]:
    """cls per scenario: one letter per metric (r, i, n, x, -)."""
    return {scenario: "".join(row[5] for row in rows) for scenario, rows in cmp.items()}


# -- perf: fetch and write ----------------------------------------------------
def collect_perf(api: GitHubApi, repo: str, known_ids: set[str]) -> list[tuple[str, dict]]:
    """(id, raw baseline) for every baseline on perf-data that is not indexed
    yet, oldest first so a budget stop leaves the newest for the next tick
    with their history complete. A blob that fails to download is skipped
    and picked up again next tick."""
    tree = api.get(f"{GITHUB_API}/repos/{repo}/git/trees/{PERF_BRANCH}",
                   params={"recursive": "1"})
    if not isinstance(tree, dict):
        return []
    blobs = {}
    for node in tree.get("tree") or []:
        bid = baseline_id(node.get("path") or "")
        if bid and node.get("type") == "blob" and node.get("sha"):
            blobs[bid] = node["sha"]
    out = []
    for bid in sorted(blobs):
        if bid in known_ids or api.stopped:
            continue
        body = api.get(f"{GITHUB_API}/repos/{repo}/git/blobs/{blobs[bid]}", raw=True)
        if body is None:
            continue
        try:
            raw = json.loads(body)
        except ValueError as e:
            print(f"::warning::frontend-index: perf baseline {bid} is not JSON: {e!r}")
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("measurements"), list):
            print(f"::warning::frontend-index: perf baseline {bid} has no measurements")
            continue
        out.append((bid, raw))
    print(f"frontend-index: perf: {len(blobs)} baselines on {PERF_BRANCH}, {len(out)} new")
    return out


def _load_shards(store, head: dict, shard_path) -> dict[str, list[dict]]:
    return {m: (store.download_json(shard_path(m)) or {}).get("entries", [])
            for m in sorted(head.get("shards") or {})}


def _content(obj: dict) -> str:
    """The comparable content of a file: tick times are left out
    (updated_ts, and coverage.to_ts, which every complete listing moves to
    the tick time), so a tick that learns nothing else commits nothing."""
    o = {k: v for k, v in obj.items() if k != "updated_ts"}
    if isinstance(o.get("coverage"), dict):
        o["coverage"] = {k: v for k, v in o["coverage"].items() if k != "to_ts"}
    return json.dumps(o, sort_keys=True)


def _write_if_changed(store, path: str, obj: dict) -> bool:
    """Write `obj` unless the stored file already has the same content
    (tick times aside), so an unchanged tick commits nothing."""
    old = store.download_json(path)
    if old is not None and _content(old) == _content(obj):
        return False
    store.upload_json(path, obj, compact=True)
    return True


def write_perf(store, repo: str, new_raw: list[tuple[str, dict]], now: int) -> int:
    """Merge the new baselines into the shards and rewrite the head. cls is
    computed once, at ingest, against the entries preceding each new one;
    latest.cmp is recomputed only when the newest entry is new this tick
    (it needs the raw samples). Returns the number of entries added."""
    head = store.download_json(perf_head_path()) or {}
    shards = _load_shards(store, head, perf_shard_path)
    pool = {e["id"]: e for es in shards.values() for e in es}
    added = 0
    newest_cmp = None
    for bid, raw in sorted(new_raw, key=lambda x: x[0]):
        if bid in pool:
            continue
        entry = perf_entry(bid, raw)
        history = preceding(list(pool.values()), entry)
        cmp = compare_entry(perf_medians(raw), history)
        entry["cls"] = classification(cmp)
        pool[bid] = entry
        added += 1
        newest_cmp = (entry, cmp, len(history))
    union = sorted(pool.values(), key=perf_key, reverse=True)
    latest = head.get("latest")
    if union and newest_cmp and newest_cmp[0]["id"] == union[0]["id"]:
        entry, cmp, n_hist = newest_cmp
        latest = {"id": entry["id"], "c": entry["c"], "t": entry["t"], "n_hist": n_hist,
                  "cmp": cmp}
    by_month: dict[str, list[dict]] = {}
    for e in union:
        by_month.setdefault(month_of(e["t"]), []).append(e)
    for month, entries in by_month.items():
        _write_if_changed(store, perf_shard_path(month), {
            "schema_version": SCHEMA_VERSION, "repo": repo, "shard": month,
            "entries": entries})
    _write_if_changed(store, perf_head_path(), {
        "schema_version": SCHEMA_VERSION, "repo": repo, "branch": PERF_BRANCH,
        "updated_ts": now,
        "metrics": METRICS, "targets": TARGETS, "window": PERF_WINDOW, "rule": PERF_RULE,
        "stats": PERF_STATS,
        "scenarios": sorted((union[0].get("s") or {}) if union else []),
        "count": len(union),
        "shards": {m: len(es) for m, es in sorted(by_month.items())},
        "latest": latest,
        "entries": union[:PERF_HEAD_LIMIT],
    })
    return added


# -- ci: entries --------------------------------------------------------------
def known_columns(head: dict) -> list[dict]:
    """The registry plus every column a previous tick discovered."""
    # Copy the path lists too: column_for appends discovered paths, and the
    # module-level registry must not change between ticks.
    cols = [{**c, "paths": list(c["paths"])} for c in COLUMNS]
    ids = {c["id"] for c in cols}
    for c in head.get("columns") or []:
        if isinstance(c, dict) and c.get("id") and c["id"] not in ids:
            cols.append({**c, "paths": list(c.get("paths") or [])})
            ids.add(c["id"])
    return cols


def column_for(columns: list[dict], path: str, name: str) -> dict | None:
    """The column a run belongs to, by workflow path. An unknown workflow
    whose display name starts with 'CI:' is appended to `columns` as group
    'other' (id from its file stem); any other workflow is ignored."""
    for c in columns:
        if path in c["paths"]:
            return c
    if not (name or "").startswith("CI:") or not path:
        return None
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cid = stem[3:] if stem.startswith("ci-") and len(stem) > 3 else stem
    for c in columns:
        if c["id"] == cid:
            c["paths"].append(path)
            return c
    col = {"id": cid, "label": name[3:].strip() or cid, "paths": [path], "name": name,
           "group": "other"}
    columns.append(col)
    return col


def cell(run: dict) -> dict:
    """{id, c, s?, t, d?, a?} for one run: conclusion, status while not
    completed, start time, duration in seconds, attempt when > 1."""
    t = _ts(run.get("run_started_at")) or _ts(run.get("created_at"))
    c = {"id": run.get("id"), "c": run.get("conclusion"), "t": t}
    if run.get("status") != "completed":
        c["s"] = run.get("status")
    else:
        end = _ts(run.get("updated_at"))
        if t is not None and end is not None and end >= t:
            c["d"] = end - t
    if (run.get("run_attempt") or 1) > 1:
        c["a"] = run["run_attempt"]
    return c


def pick_cell(old: dict | None, new: dict) -> dict:
    """The newer of two cells for one column: a later attempt of the same
    run, or the later-started run."""
    if not old:
        return new
    if new.get("id") == old.get("id"):
        return new if (new.get("a") or 1) >= (old.get("a") or 1) else old
    return new if (new.get("t") or 0) >= (old.get("t") or 0) else old


def provisional_meta(run: dict) -> dict:
    """compact_meta-shaped metadata from the run's head_commit (free, no
    lookup): subject, author name, PR from the subject suffix, commit time."""
    hc = run.get("head_commit") or {}
    subject = (hc.get("message") or "").split("\n", 1)[0].strip()[:200]
    return {"s": subject or None, "a": (hc.get("author") or {}).get("name"),
            "pr": pr_number(subject), "ct": _ts(hc.get("timestamp"))}


def overall(w: dict, columns: list[dict]) -> str:
    """Rollup of the core columns: failure, pending, cancelled or success.
    skipped / neutral / action_required never block."""
    core = {c["id"] for c in columns if c.get("group") == "core"}
    cells = [v for k, v in w.items() if k in core]
    if any(x.get("c") in RED for x in cells):
        return "failure"
    if any("s" in x for x in cells):
        return "pending"
    if any(x.get("c") == "cancelled" for x in cells):
        return "cancelled"
    return "success"


def ci_entries(runs: list[dict], columns: list[dict]) -> dict[str, dict]:
    """Provisional entries keyed by head sha from a runs listing."""
    by_sha: dict[str, dict] = {}
    for run in runs:
        sha = run.get("head_sha")
        col = column_for(columns, run.get("path") or "", run.get("name") or "")
        if not sha or not col:
            continue
        e = by_sha.get(sha)
        if e is None:
            m = provisional_meta(run)
            e = by_sha[sha] = {"c": sha, "ct": m["ct"] or _ts(run.get("created_at")) or 0,
                               "o": None, "m": m, "mi": True, "w": {}}
        e["w"][col["id"]] = pick_cell(e["w"].get(col["id"]), cell(run))
    return by_sha


def merge_ci_entry(old: dict | None, new: dict) -> dict:
    """Keep the latest attempt per column; metadata is only ever upgraded
    (a provisional head_commit never replaces a looked-up one)."""
    if not old:
        return new
    merged = {**old, "w": dict(old.get("w") or {})}
    for cid, c in (new.get("w") or {}).items():
        merged["w"][cid] = pick_cell(merged["w"].get(cid), c)
    if old.get("mi") and (not new.get("mi") or not old.get("m")):
        merged["m"] = new.get("m")
        if not new.get("mi"):
            merged.pop("mi", None)
    merged["ct"] = old.get("ct") or new.get("ct")
    return merged


def ci_key(e: dict):
    return (e.get("ct") or 0, e["c"])


def first_bad(entries: list[dict]) -> dict:
    """Per column: the start of the currently open failing chain, walking the
    commits in order. A red conclusion opens a chain, success closes it;
    cancelled, skipped, pending and missing cells do neither."""
    chains: dict[str, dict | None] = {}
    last_good: dict[str, dict | None] = {}
    for e in sorted(entries, key=ci_key):
        for cid, c in (e.get("w") or {}).items():
            v = c.get("c")
            if v == "success":
                chains[cid] = None
                last_good[cid] = e
            elif v in RED:
                if chains.get(cid) is None:
                    chains[cid] = {"commit": e["c"], "ct": e.get("ct"),
                                   "prev_good": (last_good.get(cid) or {}).get("c"),
                                   "pr": (e.get("m") or {}).get("pr"),
                                   "author": (e.get("m") or {}).get("a"), "runs": 0}
                chains[cid]["runs"] += 1
    return {cid: chain for cid, chain in chains.items() if chain}


# -- ci: fetch and write ------------------------------------------------------
def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_runs(api: GitHubApi, repo: str, params: dict, page_limit: int,
              force: bool = False) -> tuple[list, bool]:
    """Pages of a runs listing: (runs, complete). Incomplete when a page
    failed or the page limit was hit. Unless `force`, a listing whose
    total_count already exceeds what the page limit can hold stops after
    the first page, so the caller can narrow the window instead of paging
    a result the API would truncate anyway."""
    runs: list[dict] = []
    for page in range(1, page_limit + 1):
        data = api.get(f"{GITHUB_API}/repos/{repo}/actions/runs",
                       params={**params, "per_page": 100, "page": page})
        if not isinstance(data, dict):
            return runs, False
        items = data.get("workflow_runs") or []
        runs.extend(items)
        if page == 1 and not force and (data.get("total_count") or 0) > page_limit * 100:
            return runs, False
        if len(items) < 100:
            return runs, True
    return runs, False


def unsettled_cursor(entries: list[dict], columns: list[dict], now: int) -> int | None:
    """Oldest commit time among entries that may still change: a cell that
    is not completed, or a core column missing on a commit younger than
    UNSETTLED_AGE_S."""
    core = [c["id"] for c in columns if c.get("group") == "core"]
    cts = []
    for e in entries:
        w = e.get("w") or {}
        pending = any("s" in c for c in w.values())
        young = now - (e.get("ct") or 0) < UNSETTLED_AGE_S
        if pending or (young and any(cid not in w for cid in core)):
            cts.append(e.get("ct") or 0)
    return min(cts) if cts else None


class CiFetch:
    """What the ci fetch phase hands to its writer."""

    def __init__(self, columns: list[dict]):
        self.columns = columns
        self.entries: dict[str, dict] = {}
        self.meta: dict[str, dict] = {}
        self.coverage: dict | None = None

    def merge(self, entries: dict[str, dict]):
        for sha, e in entries.items():
            self.entries[sha] = merge_ci_entry(self.entries.get(sha), e)


def _base_params() -> dict:
    return {"branch": CI_BRANCH, "event": "push", "exclude_pull_requests": "true"}


def _coverage(head: dict, now: int) -> dict:
    """The stored coverage interval {from_ts, to_ts, partial}, or a fresh
    one at `now`: the index is complete for runs created between from_ts
    and to_ts. A head written before to_ts existed counts as complete up to
    its last tick."""
    c = head.get("coverage") or {}
    if not c.get("from_ts"):
        return {"from_ts": now, "to_ts": now, "partial": False}
    return {"from_ts": c["from_ts"], "to_ts": c.get("to_ts") or head.get("updated_ts") or now,
            "partial": bool(c.get("partial"))}


def _window(api: GitHubApi, repo: str, fetch: CiFetch, start: int, end: int) -> bool:
    """List the runs created in one window into `fetch`; True when the
    window is covered. A window holding more runs than BACKFILL_PAGE_LIMIT
    pages can list (a busy day on main) is halved until it fits or is
    MIN_WINDOW_S wide; an hour that still overflows is paged to the limit
    and its tail dropped with a warning, so a backfill never stalls on it."""
    params = {**_base_params(), "created": f"{_iso(start)}..{_iso(end)}"}
    if end - start > MIN_WINDOW_S:
        runs, complete = list_runs(api, repo, params, BACKFILL_PAGE_LIMIT)
        if complete:
            fetch.merge(ci_entries(runs, fetch.columns))
            return True
        if api.stopped:
            return False
        mid = start + (end - start) // 2
        return _window(api, repo, fetch, start, mid) and _window(api, repo, fetch, mid, end)
    runs, complete = list_runs(api, repo, params, BACKFILL_PAGE_LIMIT, force=True)
    fetch.merge(ci_entries(runs, fetch.columns))
    if api.stopped:
        return False
    if not complete:
        print(f"::warning::frontend-index: ci: more than {BACKFILL_PAGE_LIMIT * 100} runs "
              f"between {_iso(start)} and {_iso(end)}; the listing was truncated")
    return True


def collect_ci(api: GitHubApi, repo: str, head: dict, now: int,
               backfill_days: int | None = None, stored: list[dict] | None = None) -> CiFetch:
    """Runs created since the cursor (the oldest unsettled commit, at least
    LOOKBACK_S ago and no later than the last complete listing), then
    optionally day windows back from the oldest covered time, then commit
    metadata upgrades within the per-tick cap. `stored` is every stored
    entry, head and shards (the head's entries when None); the head's
    entries alone decide the cursor, since anything unsettled is recent,
    while the upgrade candidates come from all of them."""
    fetch = CiFetch(known_columns(head))
    recent = head.get("entries") or []
    stored = recent if stored is None else stored
    coverage = _coverage(head, now)
    # The cursor never passes to_ts, so a gap between ticks longer than the
    # lookback (a stalled cron, a broken tick) is listed, not skipped.
    cursor = min(now - LOOKBACK_S, coverage["to_ts"])
    u = unsettled_cursor(recent, fetch.columns, now)
    if u is not None:
        cursor = min(cursor, u)
    runs, complete = list_runs(api, repo, {**_base_params(), "created": f">={_iso(cursor)}"},
                               CI_PAGE_LIMIT)
    fetch.merge(ci_entries(runs, fetch.columns))
    print(f"frontend-index: ci: {len(runs)} runs since {_iso(cursor)}, "
          f"{len(fetch.entries)} commits")

    # Each complete incremental listing extends the coverage interval to
    # now (and down to its cursor); a backfill walks from_ts back day by
    # day. An incomplete listing (a failed page, or more than CI_PAGE_LIMIT
    # pages since the cursor after days away) has the newest runs in hand
    # but leaves the index only known complete up to to_ts, so the gap is
    # closed from there in day windows while the budget lasts.
    if complete:
        coverage["from_ts"] = min(coverage["from_ts"], cursor)
        coverage["to_ts"] = now
    else:
        lo = coverage["to_ts"]
        windows = 0
        while lo < now and not api.stopped:
            hi = min(now, lo + 86400)
            if not _window(api, repo, fetch, lo, hi):
                break
            lo = hi
            windows += 1
            coverage["to_ts"] = lo
        if lo < now:
            print(f"::warning::frontend-index: ci: runs since {_iso(cursor)} did not list "
                  f"completely; the index is complete up to {_iso(lo)} and the next tick "
                  "continues from there")
        elif windows:
            print(f"frontend-index: ci: runs since {_iso(cursor)} did not list completely; "
                  f"closed the gap to {_iso(now)} in {windows} day windows")
    if backfill_days:
        target = now - backfill_days * 86400
        from_ts = coverage["from_ts"]
        while from_ts > target and not api.stopped:
            start = max(target, from_ts - 86400)
            if not _window(api, repo, fetch, start, from_ts):
                break
            from_ts = start
        coverage["from_ts"] = from_ts
        coverage["partial"] = from_ts > target
        print(f"frontend-index: ci: backfill covered down to {_iso(from_ts)}"
              f"{' (partial)' if coverage['partial'] else ''}, {len(fetch.entries)} commits")
    fetch.coverage = coverage

    # Metadata upgrades: newest first, fetched commits and every stored
    # provisional entry (shards included, so a backfill's older commits are
    # upgraded too), within the cap.
    stored_by_sha = {e["c"]: e for e in stored}
    candidates = {sha: e for sha, e in fetch.entries.items()
                  if not stored_by_sha.get(sha) or stored_by_sha[sha].get("mi")}
    for e in stored:
        if e.get("mi") and e["c"] not in candidates:
            candidates[e["c"]] = e
    for sha in sorted(candidates, key=lambda s: ci_key(candidates[s]), reverse=True)[
            :META_UPGRADES_PER_TICK]:
        if api.stopped:
            break
        meta = commit_meta.fetch_commit_meta(sha, repo, base=CI_BRANCH)
        api.external_calls(2)
        if meta:
            fetch.meta[sha] = compact_meta(meta)
    return fetch


def write_ci(store, repo: str, fetch: CiFetch, now: int) -> int:
    """Merge the fetched entries and metadata into the shards and rewrite the
    head. Every shard is regenerated and written only when its content
    changed. Returns the number of commits merged."""
    head = store.download_json(ci_head_path()) or {}
    shards = _load_shards(store, head, ci_shard_path)
    columns = known_columns(head)
    ids = {c["id"] for c in columns}
    for c in fetch.columns:
        if c["id"] not in ids:
            columns.append(dict(c))
            ids.add(c["id"])
    merged = {e["c"]: e for es in shards.values() for e in es}
    for sha, e in fetch.entries.items():
        merged[sha] = merge_ci_entry(merged.get(sha), e)
    for sha, m in fetch.meta.items():
        if sha in merged:
            merged[sha] = {**merged[sha], "m": m}
            merged[sha].pop("mi", None)
    for e in merged.values():
        e["o"] = overall(e.get("w") or {}, columns)
    union = sorted(merged.values(), key=ci_key, reverse=True)
    by_month: dict[str, list[dict]] = {}
    for e in union:
        by_month.setdefault(month_of(e.get("ct") or 0), []).append(e)
    for month, entries in by_month.items():
        _write_if_changed(store, ci_shard_path(month), {
            "schema_version": SCHEMA_VERSION, "repo": repo, "branch": CI_BRANCH,
            "shard": month, "entries": entries})
    latest = union[0] if union else None
    coverage = fetch.coverage or _coverage(head, now)
    _write_if_changed(store, ci_head_path(), {
        "schema_version": SCHEMA_VERSION, "repo": repo, "branch": CI_BRANCH,
        "updated_ts": now,
        "columns": columns,
        "count": len(union),
        "shards": {m: len(es) for m, es in sorted(by_month.items())},
        "latest": ({"c": latest["c"], "ct": latest["ct"], "o": latest["o"]} if latest else None),
        "first_bad": first_bad(union),
        "coverage": coverage,
        "entries": union[:CI_HEAD_LIMIT],
    })
    return len(fetch.entries)


# -- gate ---------------------------------------------------------------------
def parse_pin(text: str) -> str | None:
    """COMFYUI_PIN from the gate workflow file (quoted or bare, 7-40 hex)."""
    m = PIN_RE.search(text or "")
    return m.group(1) if m else None


def read_pin(api: GitHubApi, repo: str, sha: str) -> tuple[bool, str | None]:
    """(read, pin): whether the gate workflow file at `sha` could be read,
    and the COMFYUI_PIN it carries (None when the file has none). A failed
    read leaves the pin unknown rather than absent."""
    body = api.get(f"{GITHUB_API}/repos/{repo}/contents/.github/workflows/{GATE_WORKFLOW}",
                   params={"ref": sha}, raw=True)
    if body is None:
        return False, None
    return True, parse_pin(body.decode("utf-8", "replace"))


def with_pin(run: dict, pin: str) -> dict:
    """A stored run with its pin filled in (and the ref it implies for a
    pinned run)."""
    r = {**run, "pin": pin}
    if r.get("rs") == "pin":
        r["ref"] = pin
    return r


def failed_step(job: dict) -> str | None:
    """Name of the first step that did not succeed or skip."""
    for step in job.get("steps") or []:
        if step.get("conclusion") not in (None, "success", "skipped"):
            return step.get("name")
    return None


def gate_run(run: dict, jobs: list[dict] | None, pin: str | None) -> dict:
    """{id, t, ev, c, d, sha, pin, ref, rs, final, rows, fs, jobs, s?, a?}:
    rows[proof_row][shard] holds the matrix job conclusions, fs the failed
    step of each non-success matrix job, final the verdict job's conclusion,
    jobs [matrix jobs, red matrix jobs] (null when the jobs lookup failed,
    so the run is fetched again next tick)."""
    t = _ts(run.get("run_started_at")) or _ts(run.get("created_at"))
    ev = run.get("event")
    dispatch = ev == "workflow_dispatch"
    r = {"id": run.get("id"), "t": t, "ev": ev, "c": run.get("conclusion"), "d": None,
         "sha": run.get("head_sha"), "pin": pin, "ref": None if dispatch else pin,
         "rs": "dispatch" if dispatch else "pin", "final": None, "rows": {}, "fs": {},
         "jobs": None}
    if run.get("status") == "completed":
        end = _ts(run.get("updated_at"))
        if t is not None and end is not None and end >= t:
            r["d"] = end - t
    else:
        r["s"] = run.get("status")
    if (run.get("run_attempt") or 1) > 1:
        r["a"] = run["run_attempt"]
    if jobs is None:
        return r
    total = red = 0
    for job in jobs:
        name = job.get("name") or ""
        conclusion = job.get("conclusion")
        if name == GATE_VERDICT_JOB:
            r["final"] = conclusion
            continue
        m = GATE_JOB_RE.match(name)
        if not m:
            continue
        total += 1
        row, shard = m.group("row"), m.group("shard")
        r["rows"].setdefault(row, {})[shard] = conclusion
        if conclusion in RED:
            red += 1
        if conclusion not in (None, "success"):
            step = failed_step(job)
            if step:
                r["fs"].setdefault(row, {})[shard] = step
    r["jobs"] = [total, red]
    return r


def clean_state(run: dict) -> str | None:
    """pass when every clean shard (proof row 0) succeeded, fail when any is
    red, None when the run has no decided clean shard."""
    rows = (run.get("rows") or {}).get("0")
    if not rows:
        return None
    values = list(rows.values())
    if any(v in RED for v in values):
        return "fail"
    if all(v == "success" for v in values):
        return "pass"
    return None


def streak(runs: list[dict]) -> dict | None:
    """{state, since, days, runs, capped} for the current unbroken pass or
    fail state over `runs` (newest first); capped when the streak reaches
    the oldest classifiable stored run. Dispatch runs test whatever core
    ref they were given (unknown until phase 2), so only pinned runs count:
    the streak describes the daily gate against the pinned core."""
    classified = [(r, s) for r in runs if r.get("rs") != "dispatch"
                  for s in [clean_state(r)] if s]
    if not classified:
        return None
    state = classified[0][1]
    newest_t = classified[0][0].get("t") or 0
    since, n, capped = newest_t, 0, True
    for r, s in classified:
        if s != state:
            capped = False
            break
        since = r.get("t") or since
        n += 1
    days = max(1, math.ceil((newest_t - since) / 86400))
    return {"state": state, "since": since, "days": days, "runs": n, "capped": capped}


def gate_settled(run: dict) -> bool:
    """Completed with its jobs known; a missing pin is repaired separately,
    without fetching the jobs again."""
    return "s" not in run and run.get("jobs") is not None


def resolve_dispatch_ref(api: GitHubApi, run: dict) -> str | None:
    """The comfyui_ref a dispatch run was given. Not recoverable through the
    REST API without reading the run log (a token with Actions read on the
    frontend repo, phase 2), so None for now."""
    return None


def collect_gate(api: GitHubApi, repo: str, head: dict) -> list[dict]:
    """Gate runs (schedule and dispatch events; PR runs are filtered out)
    that are new, re-run or still unsettled, with their jobs and the core
    pin at their head sha, plus stored runs whose pin is still missing once
    it can be read. Pins are cached per head sha from the stored runs; a
    read that fails (budget, 404, network) is not cached, so a later run of
    the same sha and the next tick try again, while a file read without a
    pin is cached for the tick (one read per such sha per tick)."""
    stored = {r["id"]: r for r in head.get("runs") or []}
    pins = {r["sha"]: r["pin"] for r in stored.values() if r.get("sha") and r.get("pin")}

    def pin_at(sha: str | None) -> str | None:
        if sha in pins:
            return pins[sha]
        if not sha or api.stopped:
            return None
        read, pin = read_pin(api, repo, sha)
        if read:
            pins[sha] = pin
        return pin

    listed: list[dict] = []
    for page in range(1, GATE_PAGE_LIMIT + 1):
        data = api.get(f"{GITHUB_API}/repos/{repo}/actions/workflows/{GATE_WORKFLOW}/runs",
                       params={"per_page": 100, "page": page, "exclude_pull_requests": "true"})
        if not isinstance(data, dict):
            break
        items = data.get("workflow_runs") or []
        listed.extend(r for r in items if r.get("event") in GATE_EVENTS)
        known = any(r.get("id") in stored for r in items)
        if len(items) < 100 or len(listed) >= GATE_LIMIT or (known and len(stored) >= GATE_LIMIT):
            break
    listed.sort(key=lambda r: (_ts(r.get("run_started_at")) or _ts(r.get("created_at")) or 0,
                               r.get("id") or 0), reverse=True)
    built: list[dict] = []
    for run in listed[:GATE_LIMIT]:
        old = stored.get(run.get("id"))
        if old and gate_settled(old) and (old.get("a") or 1) == (run.get("run_attempt") or 1):
            continue
        if api.stopped:
            break
        data = api.get(f"{GITHUB_API}/repos/{repo}/actions/runs/{run['id']}/jobs",
                       params={"filter": "latest", "per_page": 100})
        jobs = data.get("jobs") if isinstance(data, dict) else None
        r = gate_run(run, jobs, pin_at(run.get("head_sha")))
        if r["rs"] == "dispatch":
            r["ref"] = resolve_dispatch_ref(api, run)
        built.append(r)
    # Pin repair: a stored run whose pin read was skipped or failed keeps
    # its jobs and only needs the workflow file, once per distinct sha.
    fetched = len(built)
    rebuilt = {r["id"] for r in built}
    for old in stored.values():
        if old["id"] in rebuilt or old.get("pin") or not old.get("sha"):
            continue
        pin = pin_at(old["sha"])
        if pin:
            built.append(with_pin(old, pin))
    print(f"frontend-index: gate: {len(listed)} runs listed, {fetched} fetched, "
          f"{len(built) - fetched} pins repaired")
    return built


def write_gate(store, repo: str, built: list[dict], now: int) -> int:
    """Merge the fetched runs into the stored window and rewrite the head."""
    head = store.download_json(gate_path()) or {}
    runs = {r["id"]: r for r in head.get("runs") or []}
    for r in built:
        runs[r["id"]] = r
    ordered = sorted(runs.values(), key=lambda r: (r.get("t") or 0, r["id"]),
                     reverse=True)[:GATE_LIMIT]
    _write_if_changed(store, gate_path(), {
        "schema_version": SCHEMA_VERSION, "repo": repo, "updated_ts": now,
        "workflow": GATE_WORKFLOW, "events": list(GATE_EVENTS), "verdict_job": GATE_VERDICT_JOB,
        "limit": GATE_LIMIT,
        "pin": next((r["pin"] for r in ordered if r.get("pin")), None),
        "streak": streak(ordered),
        "runs": ordered,
    })
    return len(built)


# -- main ---------------------------------------------------------------------
def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _env_int(name: str) -> int | None:
    v = (os.environ.get(name) or "").strip()
    return int(v) if v.isdigit() else None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish the ComfyUI_frontend CI index.")
    ap.add_argument("--repo", default=FRONTEND_REPO)
    ap.add_argument("--backfill-days", type=int, default=_env_int("FRONTEND_BACKFILL_DAYS"),
                    help="walk main's push runs this many days back (empty = incremental)")
    ap.add_argument("--only", default=os.environ.get("FRONTEND_ONLY") or "perf,ci,gate",
                    help="csv of perf, ci, gate")
    ap.add_argument("--skip-publish", action="store_true",
                    help="run the writers against the local checkout, no push")
    ap.add_argument("--budget-reserve", type=int, default=BUDGET_RESERVE,
                    help="stop fetching when this many requests are left in the hour")
    storage.add_storage_args(ap)
    ap.set_defaults(prefix="frontend")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, store=None, now: int | None = None) -> int:
    only = {p.strip() for p in args.only.split(",") if p.strip()}
    unknown = only - {"perf", "ci", "gate"}
    if unknown or not only:
        raise SystemExit(f"--only takes a csv of perf, ci, gate; got {args.only!r}")
    now = now if now is not None else _now()
    token = (os.environ.get("FRONTEND_GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
             or os.environ.get("GH_TOKEN"))
    api = GitHubApi(token, args.budget_reserve)
    store = store or storage.from_args(args)
    repo = args.repo
    writers = []
    summary = []

    if "perf" in only:
        head = store.download_json(perf_head_path()) or {}
        known = {e["id"] for es in _load_shards(store, head, perf_shard_path).values()
                 for e in es}
        new_raw = collect_perf(api, repo, known)
        summary.append(f"perf +{len(new_raw)}")

        def write_perf_index(st):
            write_perf(st, repo, new_raw, now)
        writers.append(write_perf_index)

    if "gate" in only:
        built = collect_gate(api, repo, store.download_json(gate_path()) or {})
        summary.append(f"gate +{len(built)}")

        def write_gate_index(st):
            write_gate(st, repo, built, now)
        writers.append(write_gate_index)

    if "ci" in only:
        head = store.download_json(ci_head_path()) or {}
        stored = [e for es in _load_shards(store, head, ci_shard_path).values() for e in es]
        fetch = collect_ci(api, repo, head, now, args.backfill_days, stored)
        summary.append(f"ci +{len(fetch.entries)} commits, {len(fetch.meta)} meta")

        def write_ci_index(st):
            write_ci(st, repo, fetch, now)
        writers.append(write_ci_index)

    print(f"frontend-index: {api.calls} API calls, rate budget left: {api.remaining}")
    if args.skip_publish:
        for w in writers:
            w(store)
        print("frontend-index: dry run, nothing published")
    else:
        for w in writers:
            store.register_derived(w)
        store.finalize(f"frontend index: {', '.join(summary)}")
    if api.stopped:
        print(f"::warning::frontend-index: incomplete tick ({api.stopped}); "
              "the next tick continues")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
