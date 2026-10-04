"""ComfyUI commit metadata for run records and the run index.

Thin GitHub REST client, null-tolerant: any network or API problem yields
None so a metadata hiccup can never fail a regression publish. Uses
GITHUB_TOKEN / GH_TOKEN when present (the Actions token allows 1,000
requests per hour per repository; a backfill of a few hundred runs makes
about two requests per run).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "Comfy-Org/ComfyUI"
PR_SUFFIX = re.compile(r"\(#(\d+)\)\s*$")
TIMEOUT_S = 20


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT_S)
        if r.status_code != 200:
            print(f"::warning::GitHub API {r.status_code} for {url} "
                  f"(rate limit remaining: {r.headers.get('X-RateLimit-Remaining')})")
            return None
        return r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"::warning::GitHub API request failed for {url}: {e!r}")
        return None


def _ts(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00"))
                   .astimezone(timezone.utc).timestamp())
    except ValueError:
        return None


def pr_number(subject: str | None) -> int | None:
    """PR number from a squash-merge subject like 'Fix thing (#1234)'."""
    m = PR_SUFFIX.search(subject or "")
    return int(m.group(1)) if m else None


def fetch_commit_meta(sha: str, repo: str = DEFAULT_REPO) -> dict | None:
    """{subject, author, committed_ts, parents, pr} for a commit, or None."""
    data = _get(f"{GITHUB_API}/repos/{repo}/commits/{sha}")
    if not data:
        return None
    commit = data.get("commit") or {}
    subject = (commit.get("message") or "").split("\n", 1)[0].strip()
    author = (data.get("author") or {}).get("login") or (commit.get("author") or {}).get("name")
    return {
        "subject": subject[:200],
        "author": author,
        "committed_ts": _ts((commit.get("committer") or {}).get("date")),
        "parents": [p.get("sha") for p in data.get("parents") or [] if p.get("sha")],
        "pr": pr_number(subject),
    }


def fetch_range(prev: str, sha: str, repo: str = DEFAULT_REPO) -> dict | None:
    """How many commits lie between two tested commits (HEAD polling can skip
    some), with the GitHub compare link for them."""
    if not prev or prev == sha:
        return None
    data = _get(f"{GITHUB_API}/repos/{repo}/compare/{prev}...{sha}")
    compare_url = f"https://github.com/{repo}/compare/{prev}...{sha}"
    if not data:
        return {"prev": prev, "commits_between": None, "compare_url": compare_url}
    return {"prev": prev, "commits_between": data.get("total_commits"),
            "compare_url": data.get("html_url") or compare_url}
