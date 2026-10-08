"""PULSE PR comments: after a GPU regression run, leave one comment on the
ComfyUI pull request that landed the tested commit when that commit changed
GPU outputs, and keep it up to date.

Policy (per run summary, see run_regression.py --summary-out):
  - comment when a workflow's outputs changed here (detail 'new_drift') or a
    workflow hit an execution error where its prior run produced outputs, and
    only when the tested commit is ahead of the previous run;
  - never for pass, inherited drift, infra errors or missing baselines;
  - an existing comment on the PR is updated in place (hidden marker, one
    comment per PR, authored by the token's user), and switched to a short
    resolved note when a later run of the PR's commit is clean;
  - only PRs authored by a ComfyUI code owner (the users on the '*' rule of
    CODEOWNERS, or PULSE_ALLOWED_AUTHORS); only the primary lane on master.

Never fails the job: any problem is a warning and exit code 0.

Usage (CI):
  python notify_pr.py --summary "$RUNNER_TEMP/summary.json"
Env:
  PULSE_PR_COMMENTS      off | dry-run (default) | on
  PULSE_GH_TOKEN         token that posts in 'on' mode; its owner authors the comment
  PULSE_ALLOWED_AUTHORS  comma list of PR authors to comment on (overrides CODEOWNERS)
  PULSE_DASHBOARD_URL    dashboard base (default https://ci.comfy.org)
  GITHUB_TOKEN / GH_TOKEN  read-only lookups; GITHUB_STEP_SUMMARY for the dry-run report
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

import commit_meta
import index_builder

GITHUB_API = "https://api.github.com"
REPO = "Comfy-Org/ComfyUI"
BRANCH = "master"
MARKER = "<!-- pulse-regression:v1 -->"
FOOTER = "<sub>Non-blocking GPU regression check; this comment updates in place.</sub>"
CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")  # GitHub's order
MODES = ("off", "dry-run", "on")
MAX_BODY = 2048
PER_PAGE = 100
MAX_PAGES = 30
TIMEOUT_S = 20
VERDICT_LABELS = {"pass": "match", "fail": "changed", "execution_error": "error",
                  "infra_error": "infra error", "no_baseline": "no golden"}


class ApiError(Exception):
    pass


def api(method: str, path: str, token: str | None = None, params: dict | None = None,
        body: dict | None = None):
    url = f"{GITHUB_API}/{path.lstrip('/')}"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.request(method, url, headers=headers, params=params, json=body,
                             timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise ApiError(f"{method} {url}: {e!r}") from e
    if r.status_code >= 300:
        raise ApiError(f"{method} {url}: HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise ApiError(f"{method} {url}: invalid JSON") from e


# -- policy -------------------------------------------------------------------
def change_kind(wr: dict) -> str | None:
    """'new_drift' / 'new_error' for a workflow worth a comment, else None."""
    v = wr.get("verdict")
    if v == "fail" and wr.get("detail") == "new_drift":
        return "new_drift"
    # Only when the prior run produced outputs: a workflow with no prior verdict
    # (just added to the manifest, or never compared) erroring is not news
    # about this commit.
    if v == "execution_error" and wr.get("prior_verdict") in ("pass", "fail"):
        return "new_error"
    return None


def decide(summary: dict) -> str | None:
    """'changed' when the run warrants a comment, 'clean' when an existing
    comment may be resolved, None otherwise (nothing to say)."""
    wfs = summary.get("workflows") or {}
    if any(change_kind(wr) for wr in wfs.values()):
        return "changed"
    if wfs and all(wr.get("verdict") in ("pass", "no_baseline") for wr in wfs.values()):
        return "clean"
    return None


# -- code owners ----------------------------------------------------------------
def default_owners(text: str) -> list[str]:
    """User logins on the last '*' rule of a CODEOWNERS file (the last matching
    rule wins). Team entries (@org/team) and e-mail owners are ignored."""
    owners: list[str] = []
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if parts and parts[0] == "*":
            owners = [o[1:] for o in parts[1:] if o.startswith("@") and "/" not in o]
    return owners


def fetch_codeowners(token: str | None) -> str | None:
    for path in CODEOWNERS_PATHS:
        try:
            data = api("GET", f"repos/{REPO}/contents/{path}", token)
        except ApiError:
            continue
        if isinstance(data, dict) and data.get("content"):
            return base64.b64decode(data["content"]).decode("utf-8", "replace")
    return None


def allowed_authors(read_token: str | None) -> list[str] | None:
    override = [a.strip().lstrip("@") for a in
                (os.environ.get("PULSE_ALLOWED_AUTHORS") or "").split(",")]
    override = [a for a in override if a and "/" not in a]
    if override:
        return override
    text = fetch_codeowners(read_token)
    return default_owners(text) if text else None


# -- target PR ------------------------------------------------------------------
def target_pr(summary: dict, read_token: str | None) -> dict | None:
    """{number, author, url} of the PR that landed the tested commit."""
    meta = summary.get("commit_meta") or {}
    if meta.get("pr") and meta.get("pr_author"):
        return {"number": meta["pr"], "author": meta["pr_author"], "url": meta.get("pr_url")}
    try:
        pulls = api("GET", f"repos/{REPO}/commits/{summary['commit']}/pulls", read_token)
    except ApiError as e:
        print(f"::warning::PULSE: PR lookup failed: {e}")
        return None
    pr = commit_meta.merged_pr(pulls, summary["commit"], BRANCH)
    if not pr:
        return None
    return {"number": pr["number"], "author": (pr.get("user") or {}).get("login"),
            "url": pr.get("html_url")}


# -- comment body -----------------------------------------------------------------
def _fmt(v, spec: str, suffix: str = "") -> str:
    return f"{v:{spec}}{suffix}" if isinstance(v, (int, float)) else "–"


def _row(wf_id: str, wr: dict) -> str:
    vg = wr.get("vs_golden") or {}
    kind = change_kind(wr)
    change = {"new_drift": "new", "new_error": "new"}.get(kind) or (
        "inherited" if wr.get("detail") == "inherited" else "–")
    psnr = "identical" if vg.get("identical") else _fmt(vg.get("mean_psnr_db"), ".2f")
    vram = wr.get("vram_peak_mb")
    cells = [f"`{wf_id}`", VERDICT_LABELS.get(wr.get("verdict"), str(wr.get("verdict"))),
             change, psnr, _fmt(vg.get("mean_mse"), ".4g"),
             _fmt(vg.get("mean_pct_pixels_changed"), ".2f", "%"),
             _fmt((wr.get("timings") or {}).get("prompt_exec_s"), ".1f"),
             _fmt(vram / 1024 if isinstance(vram, (int, float)) else None, ".1f", " GB")]
    return "| " + " | ".join(cells) + " |"


def lane_line(summary: dict, lanes: dict | None) -> str:
    lane = ((lanes or {}).get("lanes") or {}).get(summary.get("lane")) or {}
    first = next(iter((summary.get("workflows") or {}).values()), {})
    py = first.get("python_version") or lane.get("python")
    py = py.split()[0] if py else None  # "3.12.3 (main, ...)" -> "3.12.3"
    parts = [f"python {py}" if py else None,
             f"torch {first.get('torch_version') or lane.get('torch')}"
             if first.get("torch_version") or lane.get("torch") else None,
             f"CUDA {lane['cuda']}" if lane.get("cuda") else None,
             first.get("gpu_name") or lane.get("gpu")]
    return " · ".join(p for p in parts if p)


def dashboard_url(summary: dict, lanes: dict | None) -> str:
    base = (os.environ.get("PULSE_DASHBOARD_URL") or "https://ci.comfy.org").rstrip("/")
    url = f"{base}/regression/{summary.get('branch') or BRANCH}/{summary['commit']}"
    lane = summary.get("lane")
    if lane and lanes and lane != lanes.get("primary"):
        url += f"?lane={lane}"
    return url


def render_changed(summary: dict, lanes: dict | None, max_len: int = MAX_BODY) -> str:
    wfs = summary.get("workflows") or {}
    kinds = {wf_id: change_kind(wr) for wf_id, wr in wfs.items()}
    sha7 = summary["commit"][:7]
    headline = ("GPU output changed vs golden" if "new_drift" in kinds.values()
                else "GPU workflow execution error")
    head = [f"**(PULSE)** {headline} · `{sha7}`", ""]
    table = ["| Workflow | Verdict | New / inherited | PSNR dB | Mean MSE | Pixels changed "
             "| Exec s | Peak VRAM |", "|---|---|---|---|---|---|---|---|"]
    tail = [""]
    lane = lane_line(summary, lanes)
    if lane:
        tail.append(f"Lane: {lane}")
    rng = summary.get("tested_range") or {}
    n = rng.get("commits_between")
    if isinstance(n, int) and n > 1 and rng.get("compare_url"):
        tail.append(f"Appeared in a batch of {n} commits: [compare]({rng['compare_url']})")
    tail += [f"[Run details]({dashboard_url(summary, lanes)})", "", FOOTER, MARKER]

    def build(ids, more=0):
        rows = [_row(wf_id, wfs[wf_id]) for wf_id in ids]
        if more:
            rows.append(f"| +{more} more on the dashboard | | | | | | | |")
        return "\n".join(head + table + rows + tail)

    notable = [w for w in sorted(wfs) if kinds[w]]
    body = build(sorted(wfs))
    # Over budget: keep only the workflows that changed, then as many as fit.
    keep = notable
    while len(body) > max_len and keep:
        body = build(keep, len(wfs) - len(keep))
        if len(body) > max_len:
            keep = keep[:-1]
    return body


def render_resolved(summary: dict, lanes: dict | None) -> str:
    sha7 = summary["commit"][:7]
    n = len(summary.get("workflows") or {})
    lines = [f"**(PULSE)** No GPU output changes vs golden · `{sha7}`", "",
             f"Re-checked at `{sha7}`: all {n} workflows match the golden "
             "(or have no golden yet). The earlier report on this PR no longer applies."]
    lane = lane_line(summary, lanes)
    if lane:
        lines += ["", f"Lane: {lane}"]
    lines += [f"[Run details]({dashboard_url(summary, lanes)})", "", FOOTER, MARKER]
    return "\n".join(lines)


# -- sticky comment ----------------------------------------------------------------
def find_sticky(pr: int, token: str | None, login: str | None) -> dict | None:
    """The marker comment on the PR written by `login` (any author when None)."""
    for page in range(1, MAX_PAGES + 1):
        batch = api("GET", f"repos/{REPO}/issues/{pr}/comments", token,
                    params={"per_page": PER_PAGE, "page": page})
        for c in batch or []:
            author = ((c.get("user") or {}).get("login") or "").lower()
            if MARKER in (c.get("body") or "") and (login is None or author == login.lower()):
                return c
        if len(batch or []) < PER_PAGE:
            return None
    return None


def step_summary(text: str):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def resolve_mode() -> tuple[str, str | None]:
    mode = (os.environ.get("PULSE_PR_COMMENTS") or "dry-run").strip().lower()
    if mode not in MODES:
        print(f"::warning::PULSE: unknown PULSE_PR_COMMENTS={mode!r}; using dry-run")
        mode = "dry-run"
    token = os.environ.get("PULSE_GH_TOKEN") or None
    if mode == "on" and not token:
        print("::warning::PULSE: PULSE_PR_COMMENTS=on but PULSE_GH_TOKEN is not set; "
              "falling back to dry-run")
        mode = "dry-run"
    return mode, token


def run(summary_path: Path) -> str:
    """Do the work; returns a one-line outcome (also used by the tests)."""
    mode, write_token = resolve_mode()
    if mode == "off":
        return "off"
    if not summary_path.exists():
        return "skip: no summary (the run did not get far enough)"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("branch") != BRANCH:
        return f"skip: branch {summary.get('branch')!r} is not {BRANCH}"
    try:
        lanes = index_builder.load_lanes()
    except (OSError, ValueError):
        lanes = None
    if lanes and summary.get("lane") and summary["lane"] != lanes.get("primary"):
        return f"skip: lane {summary['lane']} is not the primary lane"
    state = decide(summary)
    if state is None:
        return "skip: nothing new to report"
    # A dispatch of an older commit is compared with the newer previous run,
    # so its "new" drift says nothing about its own PR. The compare from the
    # previous run then has no commits (or could not be read): don't comment.
    rng = summary.get("tested_range")
    if state == "changed" and rng and not (isinstance(rng.get("commits_between"), int)
                                           and rng["commits_between"] > 0):
        return "skip: the tested commit is not ahead of the previous run"

    read_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None
    pr = target_pr(summary, read_token)
    if not pr:
        return "skip: no merged PR for this commit"
    owners = allowed_authors(read_token)
    if owners is None:
        return "skip: CODEOWNERS unavailable and PULSE_ALLOWED_AUTHORS not set"
    if (pr.get("author") or "").lower() not in {o.lower() for o in owners}:
        return f"skip: #{pr['number']} author {pr.get('author')} is not a code owner"

    # The comment is found by marker and by the token owner's login, so a
    # rerun of the same commit edits it instead of adding another.
    if write_token:
        login = api("GET", "user", write_token).get("login")
        if not login:
            raise ApiError("GET /user returned no login")
    else:
        login = None
    list_token = write_token if mode == "on" else (write_token or read_token)
    existing = find_sticky(pr["number"], list_token, login)
    body = render_changed(summary, lanes) if state == "changed" else render_resolved(summary, lanes)
    if state == "clean" and not existing:
        return f"skip: #{pr['number']} is clean and has no earlier comment"
    if existing and (existing.get("body") or "").replace("\r\n", "\n").strip() == body.strip():
        return f"unchanged: #{pr['number']} comment {existing.get('id')} is up to date"
    action = "update" if existing else "create"

    pr_ref = f"{REPO}#{pr['number']}"
    if mode == "dry-run":
        pr_url = pr.get("url") or f"https://github.com/{REPO}/pull/{pr['number']}"
        report = (f"### PR comment (PULSE): dry run\n\nWould {action} a comment on "
                  f"[{pr_ref}]({pr_url}) (author @{pr['author']}). Set the "
                  f"`PULSE_PR_COMMENTS` variable to `on` to post.\n\n---\n\n{body}\n")
        print(report)
        step_summary(report)
        return f"dry-run: would {action} on #{pr['number']}"
    if existing:
        api("PATCH", f"repos/{REPO}/issues/comments/{existing['id']}", write_token,
            body={"body": body})
    else:
        api("POST", f"repos/{REPO}/issues/{pr['number']}/comments", write_token,
            body={"body": body})
    step_summary(f"PR comment (PULSE): {action}d comment on {pr_ref} as @{login}.")
    return f"{action}d on #{pr['number']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.json from run_regression.py")
    args = ap.parse_args(argv)
    try:
        outcome = run(Path(args.summary))
        print(f"PULSE: {outcome}")
    except Exception as e:  # a comment is a courtesy, never a failure
        print(f"::warning::PULSE: PR comment step failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
