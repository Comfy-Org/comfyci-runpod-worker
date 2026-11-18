"""Run index: the dashboard's history view, derived from the run summaries.

Files (all under <root>/index/, shared by every lane):
  <branch>/<lane>.json              head: newest HEAD_LIMIT entries + first_bad
  <branch>/<lane>/<YYYY-MM>.json    complete monthly shards (newest first)
  lanes.json                        lane registry + per-branch latest/count/shards
                                    + current golden tag/sha per workflow

Entries are compact (about 1 KB with commit metadata) so the head file stays a
single fetch for years. Entries are ordered by commit time (falling back to
run time), so re-testing an older commit on demand never reorders history.
first_bad reports the start of each open failing chain (fail or
execution_error), computed over every shard; a chain whose latest output is
the currently blessed golden (accepted drift) is not reported. A cell of a
smoke-level workflow carries `"l": "smoke"`: it has no golden, so its chain
is never accepted drift, and the lane lists no golden for it.

The entry of a run that named its workflows at a commit without a full result
carries `"named": true`. It has a row like any other, and it is not the lane's
run: `latest`, in the head and in lanes.json, is the newest entry without the
mark.

The builder is registered as a derived writer: it is re-run on the freshest
tree whenever a publish has to retry, so concurrent lanes never lose each
other's entries.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import commit_meta

HEAD_LIMIT = 500
SCHEMA_VERSION = 1
RED = ("fail", "execution_error")
LANES_FILE = Path(__file__).resolve().parents[1] / "manifest" / "lanes.json"
RUN_URL_RE = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/actions/runs/[0-9]+", re.ASCII)


def load_lanes(path: Path | None = None) -> dict:
    return json.loads(Path(path or LANES_FILE).read_text(encoding="utf-8"))


def lane_config(lanes: dict, lane_id: str) -> dict:
    """One lane's entry from the registry ({} when unknown)."""
    return (lanes.get("lanes") or {}).get(lane_id) or {}


def lane_endpoint(lanes: dict, lane_id: str, env: dict | None = None) -> str | None:
    """The RunPod endpoint a lane submits to.

    A lane that names its endpoint in lanes.json uses that one. A lane that lists
    `"endpoint_id": null` has not been provisioned yet and gets None, never the
    RUNPOD_ENDPOINT_ID fallback: that would quietly run it on another lane's
    image and file the results under this lane. Only a lane with no endpoint_id
    key at all (older registries) falls back to the environment.
    """
    cfg = lane_config(lanes, lane_id)
    if "endpoint_id" in cfg:
        return cfg["endpoint_id"] or None
    return (os.environ if env is None else env).get("RUNPOD_ENDPOINT_ID") or None


def lane_for_prefix(lanes: dict, prefix: str) -> str | None:
    """Lane id whose storage prefix matches (legacy lane == bare 'regression')."""
    for lane_id, lane in lanes.get("lanes", {}).items():
        if lane.get("prefix") == prefix:
            return lane_id
    return None


def month_of(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def sort_ts(e: dict) -> int:
    """Commit time when known, else run time: stable under re-tests."""
    return (e.get("m") or {}).get("ct") or e["t"]


def sort_key(e: dict):
    return (sort_ts(e), e["c"])


def classify_detail(verdict: str | None, prev_identical: bool | None,
                    previous_verdict: str | None, golden_tag: str | None = None,
                    previous_golden_tag: str | None = None) -> str | None:
    """Same rule as run_regression.classify_detail, for summaries that predate
    the `detail` field."""
    if verdict != "fail" or prev_identical is None:
        return None
    if prev_identical:
        same_golden = golden_tag is None or previous_golden_tag == golden_tag
        return "inherited" if previous_verdict == "fail" and same_golden else None
    return "new_drift"


def _num(v):
    return v if isinstance(v, (int, float)) else None


def compact_meta(meta: dict | None) -> dict | None:
    """Commit metadata for an index entry. The PR title, author and avatar
    (pt/pa/av) are only present when the PR lookup found them, so entries
    from older summaries keep their shape."""
    if not meta:
        return None
    m = {"s": meta.get("subject"), "a": meta.get("author"), "pr": meta.get("pr"),
         "ct": meta.get("committed_ts")}
    for key, src in (("pt", "pr_title"), ("pa", "pr_author"), ("av", "pr_avatar")):
        if meta.get(src):
            m[key] = meta[src]
    return m


def run_repo(summary: dict) -> str | None:
    """'owner/repo' of the Actions run a summary names in its run_url."""
    url = summary.get("run_url")
    m = RUN_URL_RE.fullmatch(url) if isinstance(url, str) else None
    return m.group(1) if m else None


def entry_from_summary(summary: dict, lane: str, layout: int,
                       meta: dict | None = None, tested_range: dict | None = None,
                       prev_entry: dict | None = None,
                       output_shas: dict | None = None) -> dict:
    """Compact index entry. `prev_entry` (the previous tested run's entry)
    supplies verdicts for drift classification of legacy summaries;
    `output_shas` maps workflow id -> sha256 for summaries without output_sha256."""
    wfs = summary.get("workflows") or {}
    first = next(iter(wfs.values()), {}) if wfs else {}
    meta = meta if meta is not None else summary.get("commit_meta")
    tested_range = tested_range if tested_range is not None else summary.get("tested_range")
    prev = next((wr.get("previous_commit") for wr in wfs.values() if wr.get("previous_commit")),
                None)
    if prev is None and tested_range:
        prev = tested_range.get("prev")

    w = {}
    for wf_id, wr in wfs.items():
        vg = wr.get("vs_golden") or {}
        vp = wr.get("vs_previous") or {}
        pv = vp.get("identical") if vp and not vp.get("error") else None
        at_smoke = wr.get("level") == "smoke"
        detail = wr.get("detail")
        if at_smoke:
            # Nothing is compared at smoke level. A failure that was already
            # there is marked the way a standing reference failure is, so it
            # is not shown as new on every run.
            standing = wr.get("verdict") == "fail" and wr.get("prior_verdict") == "fail"
            detail = "inherited" if standing else None
        elif detail is None and wr.get("verdict") == "fail":
            prev_cell = (prev_entry or {}).get("w", {}).get(wf_id) or {}
            prev_verdict = wr.get("previous_verdict") or prev_cell.get("v")
            prev_golden = wr.get("previous_golden_tag") or prev_cell.get("g")
            detail = classify_detail(wr.get("verdict"), pv, prev_verdict,
                                     wr.get("golden_tag"), prev_golden)
        sha = wr.get("output_sha256") or (output_shas or {}).get(wf_id)
        w[wf_id] = {
            "v": wr.get("verdict"),
            "d": detail,
            "sha": sha,
            "mse": _num(vg.get("mean_mse")),
            "psnr": _num(vg.get("mean_psnr_db")),
            "pct": _num(vg.get("mean_pct_pixels_changed")),
            "pv": pv,
            "exec": _num((wr.get("timings") or {}).get("prompt_exec_s")),
            "vram": _num(wr.get("vram_peak_mb")),
            "g": wr.get("golden_tag"),
            "th": wr.get("thumbnail"),
        }
        kind = wr.get("output_kind")
        if isinstance(kind, str) and kind != "image":
            # What the output is, when it is not an image: an image cell keeps its keys.
            w[wf_id]["k"] = kind
        if at_smoke:
            # Checked at smoke level, not compared: a reference cell keeps its keys.
            w[wf_id]["l"] = "smoke"
        if wr.get("custom_nodes"):
            # Per custom-node pack the run requested: whether ComfyUI imported it.
            w[wf_id]["cn"] = {name: (info or {}).get("import")
                              for name, info in wr["custom_nodes"].items()}

    verdicts = [x["v"] for x in w.values()]
    run_id = summary.get("run_id")
    run_id = run_id if isinstance(run_id, int) and not isinstance(run_id, bool) else None
    entry = {
        "c": summary["commit"],
        "t": int(summary.get("run_ts") or 0),
        "o": summary.get("overall"),
        "infra": bool(summary.get("has_infra_error", "infra_error" in verdicts)),
        # The Actions run that tested the commit, so a history row can link to it
        # (None for summaries written before the id was recorded), and the
        # repository it ran in: the action also runs from other repositories.
        "run": run_id,
        "run_repo": run_repo(summary) if run_id else None,
        "cv": first.get("comfy_version"),
        "lane": lane,
        "layout": layout,
        "m": compact_meta(meta),
        "prev": prev,
        "range": ({"n": tested_range.get("commits_between"),
                   "url": tested_range.get("compare_url")} if tested_range else None),
        "w": w,
        "s": summary.get("suites") or None,
    }
    if "scope" in summary:
        # The run named its workflows, and the commit has no result of a full
        # run: the entry holds those workflows alone, and is not the lane's
        # run of the commit. Only such an entry has the key.
        entry["named"] = True
    ranges = summary.get("ranges")
    if isinstance(ranges, dict) and ranges:
        # Per workflow, the commits between its own last result and this one. A
        # summary names them only where one run covers many commits. `prev` is
        # that last result's commit: a later run can put a result inside the
        # range, and the count is then no longer the chain's (`first_bad`).
        entry["wr"] = {wf_id: {"n": r.get("commits_between"), "url": r.get("compare_url"),
                               "prev": r.get("prev")}
                       for wf_id, r in ranges.items() if isinstance(r, dict)}
    return entry


def merge_entry(old: dict | None, new: dict) -> dict:
    """Re-running a subset of workflows for an already indexed commit keeps the
    other workflows' cells; the rollup is recomputed over the merged cells."""
    if not old:
        return new
    merged = dict(old)
    merged.update({k: v for k, v in new.items() if k != "w" and v is not None})
    if "wr" not in new:
        merged.pop("wr", None)  # the ranges are those of the newest summary, or none
    if "named" not in new:
        merged.pop("named", None)  # a full run came after the run of named workflows
    if new.get("run") is not None:
        merged["run_repo"] = new.get("run_repo")  # the id and its repository name one run
    merged["t"] = max(old.get("t", 0), new.get("t", 0))
    merged["w"] = {**(old.get("w") or {}), **(new.get("w") or {})}
    verdicts = [c.get("v") for c in merged["w"].values()]
    merged["o"] = "fail" if any(v in RED for v in verdicts) else "pass"
    merged["infra"] = "infra_error" in verdicts
    return merged


def chain_range(covered: dict, prev_good: str | None, commit: str) -> dict:
    """The `range` of a chain that starts at `commit`, from what the run there
    recorded for the workflow (`covered`, the entry's `wr`): the commits since
    the workflow's last result, counted when the run was made. A later run can
    pass inside that range (a step of a search for the commit). The chain's
    last pass (`prev_good`) is then newer than the range's start, and the
    count is not the chain's any more: the range is given from the chain's own
    last pass, without a count. Nothing here can count commits: only GitHub
    knows how many lie between two tested ones."""
    if covered.get("prev") == prev_good:
        return {"n": covered.get("n"), "url": covered.get("url")}
    url = (f"https://github.com/{commit_meta.DEFAULT_REPO}/compare/{prev_good}...{commit}"
           if prev_good else None)
    return {"n": None, "url": url}


def first_bad(entries: list[dict]) -> dict:
    """Per workflow: the start of the currently open failing chain, walking the
    runs in commit order. Runs without a red or green verdict (infra errors,
    missing baselines) neither start nor break a chain. A chain that a
    smoke-level cell opened says so in `level`, and a workflow that changes
    its level while it is red starts a new chain there: a failed smoke check
    and an output that differs from a reference are not one failure. A chain
    whose first run covered several commits for this workflow (the entry's
    `wr`) carries that `range` (`chain_range`): its commit is then only the
    newest of them."""
    chains: dict[str, dict | None] = {}
    last_good: dict[str, dict | None] = {}
    for e in sorted(entries, key=sort_key):
        for wf_id, cell in (e.get("w") or {}).items():
            v = cell.get("v")
            if v == "pass":
                chains[wf_id] = None
                last_good[wf_id] = e
            elif v in RED:
                at_smoke = cell.get("l") == "smoke"
                chain = chains.get(wf_id)
                if chain is None or (chain.get("level") == "smoke") != at_smoke:
                    chain = chains[wf_id] = {
                        "commit": e["c"], "run_ts": e["t"], "kind": v,
                        "prev_good": (last_good.get(wf_id) or {}).get("c"),
                        "pr": (e.get("m") or {}).get("pr"),
                        "author": (e.get("m") or {}).get("a"),
                        "golden": cell.get("g"), "runs": 0}
                    if at_smoke:
                        chain["level"] = "smoke"
                    covered = (e.get("wr") or {}).get(wf_id)
                    if isinstance(covered, dict):
                        chain["range"] = chain_range(covered, chain["prev_good"], e["c"])
                chain["runs"] += 1
    return {wf_id: chain for wf_id, chain in chains.items() if chain}


def smoke_workflows(entries: list[dict]) -> set[str]:
    """The workflows whose newest cell is a smoke-level one, in commit order."""
    newest: dict[str, dict] = {}
    for e in sorted(entries, key=sort_key, reverse=True):
        for wf_id, cell in (e.get("w") or {}).items():
            newest.setdefault(wf_id, cell)
    return {wf_id for wf_id, cell in newest.items() if cell.get("l") == "smoke"}


def prior_verdicts(entries: list[dict], commit: str, committed_ts: int | None = None,
                   smoke=()) -> dict:
    """Per workflow: the most recent red or green verdict before `commit` in
    commit order, as {wf_id: {"commit", "verdict"}}. Unlike the previous-run
    pointer (advanced only by fully comparable runs), this sees runs that had
    execution errors, so it tells a new execution error from a standing one.
    Entries newer than `committed_ts` (a re-test of an older commit) are
    skipped; infra errors and missing baselines are looked through.

    `smoke`: the workflows that are at smoke level now. For those a failure at
    reference level is looked through too. It says that an output differed
    from its reference, which is no failed smoke check: a smoke failure after
    it is not the same failure, and must not read as inherited from it."""
    out: dict[str, dict] = {}
    for e in sorted(entries, key=sort_key, reverse=True):
        if e.get("c") == commit or (committed_ts and sort_ts(e) > committed_ts):
            continue
        for wf_id, cell in (e.get("w") or {}).items():
            v = cell.get("v")
            if v == "fail" and wf_id in smoke and cell.get("l") != "smoke":
                continue
            if wf_id not in out and (v == "pass" or v in RED):
                out[wf_id] = {"commit": e["c"], "verdict": v}
    return out


class IndexBuilder:
    def __init__(self, store, branch: str, lane: str, layout: int, lanes: dict | None = None,
                 active=None):
        """`active`: the workflow ids the lane runs today. An open failing
        chain is then listed only for those, which closes the chain of a
        workflow that was renamed, moved or removed while it was red. None
        (a bless, a backfill): every chain is listed."""
        self.store = store
        self.branch = branch
        self.lane = lane
        self.layout = layout
        self.lanes = lanes if lanes is not None else load_lanes()
        self.active = None if active is None else set(active)
        self.index_root = f"{store.root}/index"

    # -- paths -----------------------------------------------------------------
    def head_path(self) -> str:
        return f"{self.index_root}/{self.branch}/{self.lane}.json"

    def shard_path(self, month: str) -> str:
        return f"{self.index_root}/{self.branch}/{self.lane}/{month}.json"

    def lanes_path(self) -> str:
        return f"{self.index_root}/lanes.json"

    # -- writes ----------------------------------------------------------------
    def _load_shards(self, months) -> dict[str, list[dict]]:
        return {m: (self.store.download_json(self.shard_path(m)) or {}).get("entries", [])
                for m in sorted(set(months))}

    def _write_shard(self, month: str, entries: list[dict]):
        self.store.upload_json(self.shard_path(month), {
            "schema_version": SCHEMA_VERSION, "branch": self.branch, "lane": self.lane,
            "shard": month, "entries": sorted(entries, key=sort_key, reverse=True)})

    def upsert(self, entries: list[dict]):
        """Insert or replace entries, then rewrite shards, head and lanes.json."""
        if not entries:
            return
        head = self.store.download_json(self.head_path()) or {}
        months = set(head.get("shards", [])) | {month_of(sort_ts(e)) for e in entries}
        shards = self._load_shards(months)
        old_by_commit = {e["c"]: e for es in shards.values() for e in es}
        upserted = {e["c"] for e in entries}
        touched = {m for m, es in shards.items() if any(e["c"] in upserted for e in es)}
        for m in shards:
            shards[m] = [e for e in shards[m] if e["c"] not in upserted]
        for e in entries:
            merged = merge_entry(old_by_commit.get(e["c"]), e)
            month = month_of(sort_ts(merged))
            shards.setdefault(month, []).append(merged)
            touched.add(month)
        for m in sorted(touched):
            self._write_shard(m, shards[m])
        self._write_aggregates(shards)

    def refresh(self, workflows=()):
        """Rewrite head and lanes.json from the existing shards (after a bless:
        golden shas and accepted chains change without a new run). `workflows`
        names the workflows whose golden must be published even when no run
        is indexed yet."""
        head = self.store.download_json(self.head_path()) or {}
        shards = self._load_shards(head.get("shards", []))
        if not any(shards.values()) and not workflows:
            return
        self._write_aggregates(shards, workflows)

    def _write_aggregates(self, shards: dict[str, list[dict]], workflows=()):
        union = sorted((e for es in shards.values() for e in es), key=sort_key, reverse=True)
        months = sorted(m for m, es in shards.items() if es)
        newest = union[0] if union else None
        # The lane's latest run, as the head and lanes.json name it. A run
        # that named its workflows (`named`) is not the lane's run of its
        # commit: the newest entry without the mark stays the latest.
        latest = next((e for e in union if not e.get("named")), None)
        wf_ids = set((newest or {}).get("w", {})) | set(workflows)
        # A smoke-level workflow has no reference. One that had a reference
        # before keeps those files on the results branch, so they are not
        # looked up: a smoke failure with the old reference's sha is no
        # accepted drift. A workflow that was just blessed is at reference level.
        unreferenced = smoke_workflows(union) - set(workflows)
        goldens = {wf_id: self._golden_sha(wf_id) for wf_id in sorted(wf_ids - unreferenced)}
        chains = first_bad(union)
        for wf_id in list(chains):
            # Accepted drift: the latest output is the blessed golden itself.
            cell = (newest or {}).get("w", {}).get(wf_id) or {}
            golden = goldens.get(wf_id) or {}
            if cell.get("sha") and golden.get("sha") == cell["sha"]:
                del chains[wf_id]
            elif self.active is not None and wf_id not in self.active:
                del chains[wf_id]  # the lane no longer runs it
        self.store.upload_json(self.head_path(), {
            "schema_version": SCHEMA_VERSION, "branch": self.branch, "lane": self.lane,
            "layout": self.layout, "generated_ts": _now(), "count": len(union),
            "shards": months,
            "latest": ({"commit": latest["c"], "run_ts": latest["t"], "overall": latest["o"]}
                       if latest else None),
            "first_bad": chains,
            "entries": union[:HEAD_LIMIT],
        })
        self._write_lanes(latest, len(union), months, goldens, unreferenced)

    def _golden_sha(self, wf_id: str) -> dict | None:
        prefix = self.store.prefix
        cur = self.store.download_json(f"{prefix}/golden/{wf_id}/current.json")
        if not cur or not cur.get("tag"):
            return None
        sha = cur.get("output_sha256")
        if not sha:
            r1 = self.store.download_json(f"{prefix}/golden/{wf_id}/{cur['tag']}/run_r1.json")
            pngs = sorted((o for o in (r1 or {}).get("outputs") or []
                           if str(o.get("filename", "")).endswith(".png")),
                          key=lambda o: o["filename"])
            sha = pngs[0].get("sha256") if pngs else None
        return {"tag": cur["tag"], "sha": sha}

    def _write_lanes(self, latest: dict | None, count: int, shards: list[str],
                     goldens: dict, unreferenced=()):
        """`unreferenced`: workflows that are at smoke level now. A reference
        the lane listed for one of them before is taken off the list."""
        existing = self.store.download_json(self.lanes_path()) or {}
        lanes_out = existing.get("lanes") or {}
        for lane_id, lane in self.lanes.get("lanes", {}).items():
            cur = lanes_out.get(lane_id) or {}
            cur.update({k: lane.get(k) for k in ("label", "python", "torch", "cuda", "gpu",
                                                  "cadence", "blocking", "role", "layout")})
            cur.setdefault("branches", {})
            cur.setdefault("goldens", {})
            lanes_out[lane_id] = cur
        mine = lanes_out.setdefault(self.lane, {"branches": {}, "goldens": {}})
        mine["branches"][self.branch] = {
            "latest": ({"commit": latest["c"], "run_ts": latest["t"], "overall": latest["o"]}
                       if latest else None),
            "count": count, "shards": shards}
        for wf_id, golden in goldens.items():
            if golden:
                mine["goldens"][wf_id] = golden
        for wf_id in unreferenced:
            mine["goldens"].pop(wf_id, None)
        self.store.upload_json(self.lanes_path(), {
            "schema_version": SCHEMA_VERSION, "generated_ts": _now(),
            "primary": self.lanes.get("primary"), "lanes": lanes_out})

    # -- backfill ----------------------------------------------------------------
    def collect(self, with_meta: bool = True, log=print) -> list[dict]:
        """Entries for every committed summary (the expensive, run-once half of
        a backfill; feed the result to upsert() through a derived writer)."""
        runs_prefix = f"{self.store.prefix}/runs/{self.branch}"
        paths = self.store.list_json(runs_prefix, "summary.json")
        prefetch = getattr(self.store, "prefetch", None)
        if prefetch:
            # One round trip for every summary and run record instead of a
            # lazy fetch per file.
            prefetch(paths + self.store.list_json(runs_prefix, "run.json"))
        summaries = []
        for p in paths:
            s = self.store.read_blob_json(p)
            # A summary with `partial` is what a run that did not finish pushed
            # on its way. No publish indexed it, so a rebuild does not either:
            # a run that died is in no history table. A later run of named
            # workflows at that commit does not merge into it: it replaces it.
            if s and s.get("commit") and s.get("run_ts") and "partial" not in s:
                summaries.append(s)
        summaries.sort(key=lambda s: (s["run_ts"], s["commit"]))
        log(f"index: {len(summaries)} summaries under {runs_prefix}")

        entries: list[dict] = []
        by_commit: dict[str, dict] = {}
        missing_meta = 0
        for i, s in enumerate(summaries):
            shas = {}
            for wf_id, wr in (s.get("workflows") or {}).items():
                if not wr.get("output_sha256"):
                    run = self.store.read_blob_json(
                        f"{runs_prefix}/{s['commit']}/{wf_id}/run.json") or {}
                    pngs = sorted((o for o in run.get("outputs") or []
                                   if str(o.get("filename", "")).endswith(".png")),
                                  key=lambda o: o["filename"])
                    if pngs:
                        shas[wf_id] = pngs[0].get("sha256")
            prev_commit = next((wr.get("previous_commit") for wr in (s.get("workflows") or {}).values()
                                if wr.get("previous_commit")), None)
            prev_entry = by_commit.get(prev_commit) if prev_commit else (entries[-1] if entries else None)
            meta = s.get("commit_meta")
            rng = s.get("tested_range")
            if with_meta:
                meta = meta or commit_meta.fetch_commit_meta(s["commit"])
                if prev_commit and rng is None:
                    rng = commit_meta.fetch_range(prev_commit, s["commit"])
            if meta is None:
                missing_meta += 1
            e = entry_from_summary(s, self.lane, self.layout, meta=meta, tested_range=rng,
                                   prev_entry=prev_entry, output_shas=shas)
            entries.append(e)
            by_commit[e["c"]] = e
            if (i + 1) % 25 == 0:
                log(f"index: {i + 1}/{len(summaries)}")
        if missing_meta:
            log(f"::warning::index: {missing_meta} entries without commit metadata")
        return entries

    def backfill(self, with_meta: bool = True, log=print) -> int:
        """collect() + upsert() in one go (tests and local runs)."""
        entries = self.collect(with_meta=with_meta, log=log)
        self.upsert(entries)
        return len(entries)


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())
