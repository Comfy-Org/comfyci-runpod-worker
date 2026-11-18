"""Run report: everything a GPU regression run found, as Markdown for the job
summary ($GITHUB_STEP_SUMMARY), so reading a result never means leaving the
Actions run.

  python scripts/run_report.py lane --commit SHA --branch master --lane LANE_ID
        [--summary FILE] [--workdir DIR] [--skipped] [--artifact-url URL]
        [--results-dir DIR] [--no-fetch] [--out FILE]
  python scripts/run_report.py overview --commit SHA --branch master
        [--results-dir DIR] [--no-fetch] [--out FILE]

`lane` reports one commit on one lane: the result of every workflow, what
needs attention and why, the environment and the lane's recent history.
`overview` puts every lane side by side, for the commit and for recent commits.

A workflow at reference level is compared with its approved reference. One at
smoke level (`"level": "smoke"` in its cell) is only checked to finish and to
write usable files: its results read "smoke pass" and "smoke check failed",
and the report shows what the checks of smoke.py measured. A page without a
smoke-level result reads exactly as it did before that level existed.

The data is the results branch, read at one pinned commit. By default the
script fetches the branch into --results-dir (the workflow's own checkout) and
reads files with `git cat-file`; nothing is checked out. --summary and
--workdir name what the run that just finished left on the runner: they win
over the published copy, and they let a run that could not publish still report.

Every string read from results is untrusted (commit subjects, PR titles, error
messages, log tails, the text a text workflow generated). It is escaped before
it enters the summary and never printed: a log line that starts with "::" is a
workflow command. Links are built only from checked parts (hex shas, integer
ids, fixed hosts).

GitHub drops a step summary over 1 MiB whole, so the report stays under
MAX_BYTES by shortening, in this order: log tails, tracebacks, output file
lists, history rows, generated texts. A generated text is never shown past
Limits.text_chars characters in the first place. The report never fails the
job: main() returns 0.

Env: GITHUB_STEP_SUMMARY (where to write; otherwise --out), RESULTS_REPO /
RESULTS_BRANCH (as in storage.py), REPORT_DASHBOARD_URL (footer link; empty
drops the line), GITHUB_REPOSITORY / GITHUB_RUN_ID (links to Actions runs),
REPORT_WORKER_REPO (the repository of golden-baseline.yml and the image builds,
when the report runs in another one: action.yml sets it).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import lanes as lanes_mod
import output_kinds
import regression_skip
import smoke

# Standard library only, like lanes.py, output_kinds.py and regression_skip.py: a
# job renders the report without installing anything. Importing smoke.py needs
# no more: it loads numpy and Pillow only inside its checks, which never run here.
COMFY_URL = "https://github.com/Comfy-Org/ComfyUI"
DEFAULT_RESULTS_REPO = "Comfy-Org/comfyci-runpod-worker"  # the defaults storage.py uses
DEFAULT_RESULTS_BRANCH = "results"
DEFAULT_DASHBOARD_URL = "https://ci.comfy.org"
MAX_BYTES = 900_000
HISTORY_ROWS = 20
MATRIX_ROWS = 25
GIT_TIMEOUT_S = 180
THUMB_PX = 96
FIGURE_PX = 720
INLINE_PNG_MAX_BYTES = 4 * 1024 * 1024
TRACEBACK_CHARS = 8000
TEXT_READ_MAX = 256 * 1024  # bytes of one text output the report reads
USUAL_RUNS = 30       # passing runs a workflow's usual time range is taken from
USUAL_MIN_RUNS = 8    # fewer than this and no range is shown
SLOW_FACTOR = 1.5     # a time this many times the top of the range is marked slow
EXEC_NOTE = ("Exec s is how long the workflow ran. The usual range is what 8 in 10 of the lane's "
             f"last {USUAL_RUNS} passing runs took, and a time more than {SLOW_FACTOR:g} times its "
             "top is marked slow. Times differ between workers and are not part of the result.")

SHA_RE = re.compile(r"[0-9a-f]{7,40}")
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9_.-]{1,100}")
WORKFLOW_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,99}")
LANE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
BRANCH_RE = re.compile(r"[A-Za-z0-9._/-]{1,100}")
RUN_URL_RE = re.compile(r"https://github\.com/" + SLUG_RE.pattern + r"/actions/runs/[0-9]{1,20}")
ARTIFACT_URL_RE = re.compile(RUN_URL_RE.pattern + r"/artifacts/[0-9]{1,20}")
PACK_REPO_RE = re.compile(r"https://github\.com/" + SLUG_RE.pattern)
DASHBOARD_RE = re.compile(r"https?://[A-Za-z0-9.-]{1,200}(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]{0,200})?")


class ReportError(Exception):
    """The report cannot be built from what it was given (a fixed, safe message)."""


# -- small helpers --------------------------------------------------------------
def as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def as_list(v) -> list:
    return v if isinstance(v, list) else []


def is_num(v) -> bool:
    """A finite number that a float can hold. JSON also carries Infinity, NaN
    and integers past the float range: none of them formats or divides here."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e300


def as_int(v) -> int | None:
    """`v` when it is a real integer (JSON true is not a count), else None."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def fmt(v, spec: str, suffix: str = "") -> str:
    return f"{v:{spec}}{suffix}" if is_num(v) else ""


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def utc(ts) -> str:
    """'2026-11-14 12:30 UTC', or '' when `ts` is not a usable epoch."""
    if not is_num(ts) or ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (OverflowError, OSError, ValueError):
        return ""


def size_text(n) -> str:
    if not is_num(n) or n < 0:
        return ""
    if n < 1024:
        return f"{int(n)} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def is_sha(v) -> bool:
    return isinstance(v, str) and bool(SHA_RE.fullmatch(v))


def is_branch(v) -> bool:
    """A branch name that is safe in a path and a URL (no '..', no empty part)."""
    if not isinstance(v, str) or not BRANCH_RE.fullmatch(v) or v.startswith("-") or ".." in v:
        return False
    return all(part and not part.startswith(".") for part in v.split("/"))


def is_workflow_id(v) -> bool:
    return isinstance(v, str) and bool(WORKFLOW_RE.fullmatch(v))


# -- escaping -------------------------------------------------------------------
ZW = chr(0x200B)  # zero-width space: breaks @mentions, #123 and URL autolinks
# An OSC sequence is dropped with its text only when it is closed on its own
# line. One that never closes loses its two opening bytes (the last branch) and
# keeps the rest: it must not take the remainder of a log with it.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b\r\n]*(?:\x07|\x1b\\)|\x1b[@-_]?")
# C0/C1 controls (tab and newline aside), then by code point: zero-width and
# direction marks, line separators, invisible operators, lone surrogates (UTF-8
# cannot encode them, so the summary could not be written), the byte order mark.
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f"
    + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x200B, 0x200F), (0x2028, 0x202E),
                                               (0x2060, 0x206F), (0xD800, 0xDFFF),
                                               (0xFEFF, 0xFEFF))) + "]")
_SPECIAL_RE = re.compile(r"[\\`*_\[\]~#|<>&$@]|://|[wW]{3}\.|[gG][hH]-(?=[0-9])")
_ENTITIES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "|": "&#124;"}


def clean(value, limit: int | None = None, lines: bool = False) -> str:
    """Untrusted text without ANSI codes or control characters; one line unless
    `lines`. Longer than `limit`, it is cut and marked."""
    text = value if isinstance(value, str) else "" if value is None else str(value)
    text = _ANSI_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    if lines:
        text = text.replace("\t", "    ")
    else:
        text = " ".join(text.split())
    text = _CONTROL_RE.sub("", text)
    if limit is not None and len(text) > limit:
        text = text[:max(limit - 1, 0)].rstrip() + "…"
    return text


def _escape_special(m: re.Match) -> str:
    t = m.group(0)
    if t in _ENTITIES:
        return _ENTITIES[t]
    if t == "#":
        return "\\#" + ZW
    if t == "@":
        return "@" + ZW
    if t == "$":  # GitHub renders $...$ as math
        return ZW + "$" + ZW
    if t == "://":
        return ":" + ZW + "//"
    if len(t) > 1:  # "www." and "GH-"
        return t[:-1] + ZW + t[-1]
    return "\\" + t


def md(value, limit: int | None = 300) -> str:
    """Untrusted text as inline Markdown: no markup, no HTML, no table break,
    no mention, no issue or URL autolink."""
    return _SPECIAL_RE.sub(_escape_special, clean(value, limit))


def code(value, limit: int | None = 200) -> str:
    """Untrusted text in code type. Backticks would let it close the span."""
    text = md(value, limit)
    return f"<code>{text}</code>" if text else ""


def tick(value: str) -> str:
    """Code type for a value already checked to be a plain token."""
    return f"`{value}`"


def pre(text: str) -> str:
    """A <pre> block of untrusted text. No line may start with '::', nor with
    blanks and then '::': the runner trims a line before it looks for a command."""
    body = html.escape(text, quote=False)
    body = re.sub(r"(?m)^([^\S\n]*)::", r"\1&#58;:", body)
    return f"<pre>\n{body}\n</pre>"


def attr(value: str) -> str:
    return html.escape(value, quote=True)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        out.append("| " + " | ".join((c or "").replace("\n", " ") for c in row) + " |")
    return out


def details(summary: str, body: list[str]) -> list[str]:
    return [f"<details><summary>{summary}</summary>", "", *body, "", "</details>"]


# -- links ------------------------------------------------------------------------
def results_repo() -> str:
    repo = os.environ.get("RESULTS_REPO") or DEFAULT_RESULTS_REPO
    return repo if SLUG_RE.fullmatch(repo) else DEFAULT_RESULTS_REPO


def results_branch() -> str:
    branch = os.environ.get("RESULTS_BRANCH") or DEFAULT_RESULTS_BRANCH
    return branch if is_branch(branch) else DEFAULT_RESULTS_BRANCH


def actions_repo() -> str:
    """The repository whose Actions runs test commits (this one, in CI)."""
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    return repo if SLUG_RE.fullmatch(repo) else results_repo()


def worker_repo() -> str:
    """The repository that holds golden-baseline.yml and builds the worker
    image. That is the workflow's own repository, except in the composite
    action, which runs in its caller's: action.yml names it in REPORT_WORKER_REPO."""
    repo = os.environ.get("REPORT_WORKER_REPO") or ""
    return repo if SLUG_RE.fullmatch(repo) else actions_repo()


def run_url(run_id, repo=None) -> str | None:
    """An Actions run by its id, in `repo` when that is a plain slug (an index
    entry names the repository that ran it), else in this workflow's own."""
    if (as_int(run_id) or 0) <= 0:
        return None
    repo = repo if isinstance(repo, str) and SLUG_RE.fullmatch(repo) else actions_repo()
    return f"https://github.com/{repo}/actions/runs/{run_id}"


def entry_run_url(entry: dict) -> str | None:
    """The Actions run that tested an index entry's commit."""
    return run_url(entry.get("run"), entry.get("run_repo"))


def checked_run_url(url) -> str | None:
    return url if isinstance(url, str) and RUN_URL_RE.fullmatch(url) else None


def this_run_url() -> str | None:
    run_id = os.environ.get("GITHUB_RUN_ID") or ""
    return run_url(int(run_id)) if run_id.isdigit() else None


def sha_link(sha, n: int = 7) -> str:
    """`sha7` linked to the ComfyUI commit; escaped text when it is not a sha."""
    if is_sha(sha):
        return f"[{tick(sha[:n])}]({COMFY_URL}/commit/{sha})"
    return code(sha, 40)


def pr_link(pr) -> str:
    if (as_int(pr) or 0) <= 0:
        return ""
    return f"[#{pr}]({COMFY_URL}/pull/{pr})"


def compare_link(prev, sha) -> str:
    if is_sha(prev) and is_sha(sha):
        return f"[compare]({COMFY_URL}/compare/{prev}...{sha})"
    return ""


def join_path(*parts) -> str | None:
    """A results path from untrusted parts; None when a segment is not a plain name."""
    segments: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            return None
        for seg in part.split("/"):
            if (seg in ("", ".", "..") or len(seg) > 200 or "\\" in seg
                    or clean(seg, lines=True) != seg or "\n" in seg):
                return None
            segments.append(seg)
    return "/".join(segments)


def dashboard_url() -> str | None:
    base = os.environ.get("REPORT_DASHBOARD_URL")
    if base is None:
        return DEFAULT_DASHBOARD_URL
    base = base.strip().rstrip("/")
    return base if base and DASHBOARD_RE.fullmatch(base) else None


# -- results reader ---------------------------------------------------------------
class Reader:
    """Read-only view of the results branch at one commit. This base class
    reads nothing: it stands in when the branch cannot be reached."""

    sha: str | None = None  # the results commit every read and link is pinned to

    def read_json(self, path: str) -> dict | None:
        """The JSON object at `path`; None when absent or unreadable."""
        return None

    def read_text(self, path: str, limit: int = TEXT_READ_MAX) -> str | None:
        """The start of the text file at `path` (at most `limit` bytes, bytes
        that are not UTF-8 replaced); None when absent or unreadable."""
        return None

    def list_dir(self, path: str) -> list[str]:
        """Names directly under `path`, from the trees alone."""
        return []

    def prefetch(self, paths: list[str]) -> None:
        """Load many files in one round trip; reads work without it."""


class DictReader(Reader):
    """In-memory results tree {path: object}, for tests and local renders."""

    def __init__(self, files: dict[str, dict], sha: str | None = "0" * 40):
        self.files = files
        self.sha = sha

    def read_json(self, path: str) -> dict | None:
        data = self.files.get(path)
        return data if isinstance(data, dict) else None

    def read_text(self, path: str, limit: int = TEXT_READ_MAX) -> str | None:
        data = self.files.get(path)
        return data.encode("utf-8", "replace")[:limit].decode("utf-8", "replace") \
            if isinstance(data, str) else None

    def list_dir(self, path: str) -> list[str]:
        base = path.strip("/") + "/"
        return sorted({p[len(base):].split("/", 1)[0] for p in self.files if p.startswith(base)})


class GitReader(Reader):
    """The results branch through a git checkout that can fetch it.

    The fetch is blobless, which also makes origin a promisor remote: trees are
    local and a file's content is fetched the first time it is read. Every read
    names the commit the fetch returned, never FETCH_HEAD, so a publish that
    lands while the report is being built cannot mix two states.
    """

    def __init__(self, workdir: str | Path, fetch: bool = True, branch: str | None = None):
        self.workdir = str(workdir)
        ref = "HEAD"
        if fetch:
            self._git("fetch", "--depth", "1", "--filter=blob:none", "origin",
                      branch or results_branch(), check=True)
            ref = "FETCH_HEAD"
        sha = self._git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}",
                        check=True).stdout.decode("ascii", "replace").strip()
        if not FULL_SHA_RE.fullmatch(sha):
            raise ReportError("the results checkout has no commit")
        self.sha = sha
        # Only a partial clone has blobs left to fetch. A filtered fetch into a
        # full clone would turn it into one, so prefetch leaves those alone.
        self._partial = fetch or self._git(
            "config", "--get", "remote.origin.promisor").stdout.strip() == b"true"

    def _git(self, *args: str, check: bool = False, stdin: bytes | None = None):
        # Output is captured, never echoed: it can quote file content.
        r = subprocess.run(["git", "-C", self.workdir, *args], input=stdin, capture_output=True,
                           timeout=GIT_TIMEOUT_S, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        if check and r.returncode != 0:
            raise ReportError("git could not read the results branch")
        return r

    def read_json(self, path: str) -> dict | None:
        try:
            r = self._git("cat-file", "-p", f"{self.sha}:{path}")
            data = json.loads(r.stdout.decode("utf-8")) if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError, ValueError, RecursionError):
            return None  # RecursionError: JSON nested deeper than the parser goes
        return data if isinstance(data, dict) else None

    def read_text(self, path: str, limit: int = TEXT_READ_MAX) -> str | None:
        # Through a pipe, and no further than `limit`: a blob can be any size.
        try:
            with subprocess.Popen(
                    ["git", "-C", self.workdir, "cat-file", "blob",  # a file, never a tree
                     f"{self.sha}:{path}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                    start_new_session=True) as git:
                def stop():
                    # A blob that is not local is fetched by a child of git, and that
                    # child holds the pipe too: the read ends only when the whole
                    # group is gone. Windows has no process group to signal.
                    try:
                        if git.returncode is None:  # once git was waited for, its id is free
                            os.killpg(git.pid, signal.SIGKILL)
                    except (AttributeError, OSError):
                        git.kill()

                watchdog = threading.Timer(GIT_TIMEOUT_S, stop)
                watchdog.start()
                try:
                    data = git.stdout.read(limit)
                    cut = bool(git.stdout.read(1))
                    if cut:
                        git.kill()  # the rest is not read
                    code = git.wait()
                finally:
                    watchdog.cancel()
        except (OSError, subprocess.SubprocessError):
            return None
        return data.decode("utf-8", "replace") if cut or code == 0 else None

    def list_dir(self, path: str) -> list[str]:
        try:
            r = self._git("ls-tree", "-z", "--name-only", f"{self.sha}:{path.strip('/')}")
        except (OSError, subprocess.SubprocessError):
            return []
        if r.returncode != 0:
            return []
        return [n for n in r.stdout.decode("utf-8", "replace").split("\0") if n]

    def prefetch(self, paths: list[str]) -> None:
        """One fetch for the blobs behind `paths`, the way GitHubStorage.prefetch
        does it. A lazy read costs a round trip per file against GitHub."""
        wanted = sorted({p for p in paths if p})
        oids: list[str] = []
        if not self._partial:
            return
        try:
            for i in range(0, len(wanted), 50):
                r = self._git("ls-tree", "-z", self.sha, "--", *wanted[i:i + 50])
                if r.returncode != 0:
                    return
                for line in r.stdout.decode("utf-8", "replace").split("\0"):
                    meta = line.partition("\t")[0].split()
                    if len(meta) == 3 and meta[1] == "blob":
                        oids.append(meta[2])
            if oids:
                # Bytes, not text: text mode would write CRLF on Windows.
                self._git("-c", "fetch.negotiationAlgorithm=noop", "fetch", "--no-tags",
                          "--no-write-fetch-head", "--recurse-submodules=no",
                          "--filter=blob:none", "--stdin", "origin",
                          stdin=("\n".join(oids) + "\n").encode())
        except (OSError, subprocess.SubprocessError):
            return  # reads fall back to one lazy fetch per file


def raw_url(reader: Reader, path: str | None) -> str | None:
    """A file on the results branch, pinned to the commit the report read."""
    if not reader.sha or not path:
        return None
    return (f"https://raw.githubusercontent.com/{results_repo()}/{reader.sha}/"
            f"{quote(path, safe='/')}")


def tree_url(reader: Reader, path: str | None) -> str | None:
    if not reader.sha or not path:
        return None
    return f"https://github.com/{results_repo()}/tree/{reader.sha}/{quote(path, safe='/')}"


# -- results and labels -----------------------------------------------------------
# One icon and one wording per result, used in every table and sentence.
CHECK, CROSS, WARN, AMBER, BLANK, DASH, QUERY = "✅", "❌", "⚠️", "🟡", "⚪", "➖", "❔"
LABELS = {
    "pass": (CHECK, "pass"),
    "new": (CROSS, "output changed (new at this commit)"),
    "inherited": (CROSS, "output changed (inherited)"),
    "changed": (CROSS, "output changed"),
    "accepted": (AMBER, "output changed (accepted)"),
    "exec": (CROSS, "execution error"),
    "infra": (WARN, "infra error"),
    "no_ref": (BLANK, "no reference yet"),
    "not_run": (DASH, "not run"),
    "unknown": (QUERY, "unknown"),
    # A smoke-level workflow is checked, not compared: its results have names of their own.
    "smoke_pass": (CHECK, "smoke pass"),
    "smoke_fail": (CROSS, "smoke check failed"),
}
# A run that covers several commits cannot pin a new change on the one it tested.
NEW_IN_RUN = "output changed (new in this run)"
RED = ("new", "inherited", "changed", "exec", "smoke_fail")
GREEN = ("pass", "accepted", "smoke_pass")
# What needs no attention. Not GREEN: an accepted change is green and still shown.
PASSED = ("pass", "smoke_pass")
DRIFT = ("new", "inherited", "changed", "accepted")
ATTENTION_ORDER = ("new", "exec", "smoke_fail", "changed", "inherited", "infra", "no_ref",
                   "unknown", "accepted")
LEGEND = (f"Result labels: {CHECK} pass, the output matches the reference within its limits. "
          f"{CROSS} output changed, it differs beyond the limits: new at this commit (new in "
          "this run, when the run covers several commits), inherited from an earlier commit, "
          f"or {AMBER} accepted, when the changed output was approved as the reference "
          f"afterwards. {CROSS} execution error, the workflow did not finish. "
          f"{WARN} infra error, an infrastructure problem and not a verdict on the commit. "
          f"{BLANK} no reference yet. {DASH} not run. The reference is the approved (golden) "
          "output of a workflow on a lane.")
# Said only on a page that shows a smoke-level result. Every other page keeps the
# legend it had.
SMOKE_LEGEND = (f"{CHECK} smoke pass: the workflow finished and its output passed the smoke "
                f"checks. {CROSS} smoke check failed: it finished, and an output is missing, "
                "unreadable, blank or the wrong size. A workflow without the word smoke is at "
                "reference level (the main tests): its output is compared with the reference.")
SMOKE_NOTE = ("Smoke-level workflows are only checked to finish and to write a usable file. "
              "Their output is not compared with an approved reference.")
FIRST_RUN = ("This is the first result of this workflow on this lane. No earlier run says that "
             "a commit broke it: look at the workflow itself first.")


def legend(smoke_level: bool = False) -> str:
    """The result labels in words, with the smoke labels on a page that has one."""
    return f"{LEGEND} {SMOKE_LEGEND}" if smoke_level else LEGEND


def label(state: str, batch: bool = False) -> str:
    return f"{icon(state)} {words(state, batch)}"


def icon(state: str) -> str:
    return LABELS.get(state, LABELS["unknown"])[0]


def words(state: str, batch: bool = False) -> str:
    """The wording of a result; `batch` when its run covers several commits."""
    if state == "new" and batch:
        return NEW_IN_RUN
    return LABELS.get(state, LABELS["unknown"])[1]


def is_batch(n) -> bool:
    """Whether a tested range of `n` commits holds more than the tested one."""
    return (as_int(n) or 0) > 1


def cell_state(verdict, detail=None, sha=None, tag=None, reference=None, level=None) -> str:
    """The result label a verdict gets. A failure whose output is the lane's
    reference today was approved after the run: it is no longer open. The
    reference must have changed since the run (`tag` is the one the run was
    compared with): the sha is the first PNG alone, and a run that fails on a
    later frame can share it with the reference it failed against.

    `level` "smoke": the workflow was checked, not compared. Its pass and its
    failure have their own labels, and no reference can accept its output."""
    # A list or an object is not a verdict, and cannot be a dict key below.
    verdict = verdict if isinstance(verdict, str) else None
    detail = detail if isinstance(detail, str) else None
    if level == "smoke":
        return {"pass": "smoke_pass", "fail": "smoke_fail", "execution_error": "exec",
                "infra_error": "infra"}.get(verdict, "unknown")
    if verdict == "pass":
        return "pass"
    if verdict == "fail":
        ref = as_dict(reference)
        if sha and sha == ref.get("sha") and tag and ref.get("tag") and tag != ref["tag"]:
            return "accepted"
        return {"inherited": "inherited", "new_drift": "new"}.get(detail, "changed")
    return {"execution_error": "exec", "infra_error": "infra",
            "no_baseline": "no_ref"}.get(verdict, "unknown")


def workflow_name(wf) -> str:
    return tick(wf) if is_workflow_id(wf) else code(wf, 100)


def rollup(states: dict[str, str]) -> str:
    """One result for a commit on a lane, from its workflows' results."""
    vals = list(states.values())
    n = len(vals)
    if not n:
        return label("not_run")
    red = sum(s in RED for s in vals)
    if red:
        return f"{CROSS} {red} of {n} failed"
    for state in ("infra", "unknown", "no_ref"):
        k = vals.count(state)
        if k:
            return f"{icon(state)} {k} of {n} {words(state)}"
    # Green. A smoke pass is not a match with a reference, so it is not called one.
    checked = vals.count("smoke_pass")
    if checked == n:
        return label("smoke_pass")
    if checked:
        return f"{label('pass')} ({n - checked} reference, {checked} smoke)"
    return label("pass")


def not_passing(states: dict[str, str], cap: int = 6, batch: bool = False) -> str:
    """The workflows that did not pass, spelled out with their result."""
    names = [f"{workflow_name(wf)} {words(s, batch)}" for wf, s in states.items()
             if s not in PASSED]
    more = len(names) - cap
    return "; ".join(names[:cap]) + (f"; +{more} more" if more > 0 else "")


def first_png(run: dict) -> dict:
    """The first PNG by file name: the output that identifies a run."""
    pngs = sorted((o for o in as_list(run.get("outputs")) if isinstance(o, dict)
                   and str(o.get("filename", "")).endswith(".png")),
                  key=lambda o: str(o.get("filename")))
    return pngs[0] if pngs else {}


def cell_kind(cell: dict) -> str:
    """The output kind of a summary cell: the one it names, else the one its
    comparison carries. Neither says so for an image."""
    kind = cell.get("output_kind")
    if isinstance(kind, str) and kind in output_kinds.KINDS:
        return kind
    for key in ("vs_golden", "vs_previous"):
        kind = output_kinds.metrics_kind(cell.get(key))
        if kind != output_kinds.DEFAULT_KIND:
            return kind
    return output_kinds.DEFAULT_KIND


def output_sha(run: dict, kind: str, wf: str = ""):
    """The sha256 that identifies a run's output, from its record. An audio
    workflow (`wf` is its id) is identified by its FLAC file, by the rule the
    runner uses."""
    if kind == "text":
        return output_kinds.text_sha(run)
    if kind == "audio":
        return smoke.main_sha(run, wf, kind)
    return first_png(run).get("sha256")


def entries_of(head: dict | None) -> list[dict]:
    """Index entries, newest commit first (commit time, else run time)."""
    entries = [e for e in as_list(as_dict(head).get("entries"))
               if isinstance(e, dict) and isinstance(e.get("c"), str)]
    return sorted(entries, key=entry_key, reverse=True)


def entry_key(e: dict):
    ct, t = as_dict(e.get("m")).get("ct"), e.get("t")
    return (ct if is_num(ct) and ct else t if is_num(t) else 0, e["c"])


def entry_states(entry: dict, references: dict) -> dict[str, str]:
    out = {}
    for wf, cell in as_dict(entry.get("w")).items():
        cell = as_dict(cell)
        out[wf] = cell_state(cell.get("v"), cell.get("d"), cell.get("sha"), cell.get("g"),
                             references.get(wf), cell.get("l"))
    return out


def is_smoke_chain(fb) -> bool:
    """An open failing chain that a smoke-level result started."""
    return as_dict(fb).get("level") == "smoke"


def failing_since(fb: dict, entries: list[dict] | tuple = ()) -> str:
    """'failing since <sha7> (#pr, author, subject), first seen <date>, N runs,
    reference <tag>, last passed at <sha7>'. A chain that started in a run of
    several commits names that run: it cannot be pinned on one pull request.
    A chain with a `range` of its own (a nightly or weekly lane) is read from
    that: the commits since the workflow's own last result, not the run's.
    Its count is not known when GitHub did not answer, or when a later run
    passed inside the range: the line then names the commits since the last
    pass, with their compare link, and no number.
    A smoke-level chain of a workflow that never passed on the lane names no
    pull request and no range: its first run failed, and nothing says a commit
    did it."""
    fb = as_dict(fb)
    if not fb.get("commit"):
        return ""
    start = next((e for e in entries if e["c"] == fb["commit"]), {})
    own = fb.get("range")
    n = as_dict(own if isinstance(own, dict) else start.get("range")).get("n")
    if is_smoke_chain(fb) and not fb.get("prev_good"):
        who = "the workflow's first run on this lane"
    elif isinstance(own, dict) and as_int(n) != 1:
        cmp_ = compare_link(fb.get("prev_good"), fb["commit"])
        if is_batch(n):
            who = f"a run that covered {n} commits"
        elif cmp_:
            who = "one of the commits since its last pass"
        else:
            who = "a run that covered an unknown number of commits"
        who += f", {cmp_}" if cmp_ else ""
    elif is_batch(n):
        who = f"a run that covered {n} commits"
    else:
        who = ", ".join(x for x in (pr_link(fb.get("pr")), md(fb.get("author"), 60),
                                    md(as_dict(start.get("m")).get("s"), 80)) if x)
    parts = [f"failing since {sha_link(fb['commit'])}" + (f" ({who})" if who else "")]
    if utc(fb.get("run_ts")):
        parts.append(f"first seen {utc(fb['run_ts'])[:10]}")
    if is_num(fb.get("runs")):
        parts.append(plural(int(fb["runs"]), "run"))
    golden = fb.get("golden")
    if isinstance(golden, str) and SEGMENT_RE.fullmatch(golden):
        parts.append(f"reference {tick(golden)}")
    if fb.get("prev_good"):
        parts.append(f"last passed at {sha_link(fb['prev_good'])}")
    return ", ".join(parts)


def open_failures(head: dict | None, entries: list[dict] | tuple = (),
                  runs: set | None = None) -> list[str]:
    """One line for each open failing chain of a lane. With `runs`, the ids of
    the workflows the lane runs today, only their chains: the chain of a
    workflow that was renamed, moved or removed while it was red stays in the
    index until the lane's next run."""
    kinds = {"fail": words("changed"), "execution_error": words("exec")}
    out = []
    for wf, fb in as_dict(as_dict(head).get("first_bad")).items():
        if runs is not None and wf not in runs:
            continue
        since = failing_since(fb, entries)
        if since:
            kind = as_dict(fb).get("kind")
            if kind == "fail" and is_smoke_chain(fb):
                kind = words("smoke_fail")
            else:
                kind = kinds.get(kind, "") if isinstance(kind, str) else ""
            out.append(f"- {workflow_name(wf)}: " + (f"{kind}, " if kind else "") + since)
    return out


def other_cadence(entry: dict, lane: dict | None) -> bool:
    """Whether a manifest entry is of another cadence than the lane, which then
    does not run it (lanes.why_not_on_lane). Never so for a retired lane or a
    lane whose own cadence is not known here, and an entry whose cadence
    cannot be read counts as the lane's own: a report leaves nothing out on a
    guess."""
    cadence = as_dict(lane).get("cadence")
    if not isinstance(cadence, str) or cadence not in lanes_mod.WORKFLOW_CADENCES:
        return False
    try:
        return lanes_mod.cadence_of(entry) != cadence
    except ValueError:
        return False


def lane_workflows(manifest: dict | None, lane_id: str,
                   lane: dict | None = None) -> set[str] | None:
    """The ids of the workflows the manifest runs on a lane today: switched on,
    not kept to other lanes and, with `lane` (the lane's entry in the
    registry), of the lane's cadence. None when the manifest lists no
    workflow, so a report made without one leaves nothing out."""
    entries = as_dict(as_dict(manifest).get("workflows"))
    if not entries:
        return None
    found = set()
    for wf, entry in entries.items():
        entry = as_dict(entry)
        only = entry.get("lanes")
        if (entry.get("enabled") and not (isinstance(only, list) and lane_id not in only)
                and not other_cadence(entry, lane)):
            found.add(wf)
    return found


# -- size budget --------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """How much of the long parts a render keeps."""
    log_chars: int = 6000
    traceback_lines: int = 40
    files_per_workflow: int = 200
    history_rows: int = HISTORY_ROWS
    matrix_rows: int = MATRIX_ROWS
    text_chars: int = 4000

    def shortened(self) -> list[str]:
        full, out = Limits(), []
        if self.log_chars < full.log_chars:
            out.append(f"log tails cut to their last {self.log_chars} characters"
                       if self.log_chars else "log tails left out")
        if self.traceback_lines < full.traceback_lines:
            out.append(f"tracebacks cut to their last {self.traceback_lines} lines"
                       if self.traceback_lines else "tracebacks left out")
        if self.files_per_workflow < full.files_per_workflow:
            out.append(f"output file lists cut to {self.files_per_workflow} files per workflow"
                       if self.files_per_workflow else "output file lists left out")
        if self.history_rows < full.history_rows or self.matrix_rows < full.matrix_rows:
            rows = max(self.history_rows, self.matrix_rows)
            out.append(f"recent commits cut to {rows} rows" if rows
                       else "recent commits left out")
        if self.text_chars < full.text_chars:
            out.append(f"generated texts cut to their first {self.text_chars} characters"
                       if self.text_chars else "generated texts left out")
        return out


# The order GitHub's limit takes things away in: the least needed first. The
# generated texts go last: for a text workflow they are the result itself.
SHRINK_STEPS = ({"log_chars": 1500}, {"log_chars": 0},
                {"traceback_lines": 5}, {"traceback_lines": 0},
                {"files_per_workflow": 3}, {"files_per_workflow": 0},
                {"history_rows": 10, "matrix_rows": 10}, {"history_rows": 5, "matrix_rows": 5},
                {"history_rows": 0, "matrix_rows": 0},
                {"text_chars": 1000}, {"text_chars": 0})


def byte_size(text: str) -> int:
    return len(text.encode("utf-8"))


def fit(render, max_bytes: int = MAX_BYTES) -> str | None:
    """render(limits) at the fullest limits that fit; None when nothing fits."""
    limits = Limits()
    text = render(limits)
    for step in SHRINK_STEPS:
        if byte_size(text) <= max_bytes:
            break
        limits = replace(limits, **step)
        text = render(limits)
    return text if byte_size(text) <= max_bytes else None


def shortened_note(limits: Limits) -> list[str]:
    cut = limits.shortened()
    if not cut:
        return []
    return ["Shortened to stay under GitHub's size limit for a job summary: "
            + "; ".join(cut) + ". The linked files hold everything."]


# -- one lane: gather ---------------------------------------------------------------
@dataclass
class View:
    """One workflow of the run, with its result worked out."""
    wf: str
    name: str            # the id as Markdown
    safe: bool           # the id is a plain name: usable in paths and commands
    cell: dict
    run: dict
    state: str
    origin: str | None   # a red result: "new", "inherited" or "first" (run) when known
    prior: dict          # {verdict, commit} of its last red or green result before
    sha: str | None      # output sha256
    chain: dict = field(default_factory=dict)  # the open failing chain it is part of
    kind: str = output_kinds.DEFAULT_KIND      # what its output is: "image", "text", "audio"
    level: str | None = None                   # "smoke": checked, not compared with a reference
    # {prev, n}, its own range: of a new smoke failure, and of a new execution
    # error on a nightly or weekly lane. With `later`, the commit of a pass
    # that a later run added between `prev` and this commit (`later_pass`).
    since: dict = field(default_factory=dict)


@dataclass
class LaneData:
    commit: str
    branch: str
    lane_id: str
    lane: dict
    prefix: str
    run_dir: str | None
    reader: Reader
    summary: dict | None = None
    source: str | None = None       # "local", "published" or "workdir"
    published: bool = False         # the results branch holds this very summary
    earlier: dict = field(default_factory=dict)  # a published summary the report does not show
    left_nothing: bool = False      # a run that tested left no record: the published one is shown
    marker: dict = field(default_factory=dict)
    head: dict = field(default_factory=dict)
    registry: dict = field(default_factory=dict)
    entries: list[dict] = field(default_factory=list)  # the head's entries, newest first
    entry: dict = field(default_factory=dict)     # this commit's index entry
    views: list[View] = field(default_factory=list)
    references: dict = field(default_factory=dict)  # wf -> golden current.json
    noise: dict = field(default_factory=dict)       # wf -> noise_floor.json
    figures: dict = field(default_factory=dict)     # wf -> figure file names
    first_figures: dict = field(default_factory=dict)  # wf -> (dir, names) at first_bad
    texts: dict = field(default_factory=dict)       # wf -> the texts of a text workflow
    manifest: dict = field(default_factory=dict)
    manifest_is_snapshot: bool = False  # the manifest the run used, not today's
    packs: dict = field(default_factory=dict)
    skipped: bool = False
    artifact_url: str | None = None

    @property
    def files_ok(self) -> bool:
        """This run's files can be linked: published, at a known results commit."""
        return bool(self.published and self.reader.sha and self.run_dir)

    @property
    def lane_references(self) -> dict:
        lanes = as_dict(self.registry.get("lanes"))
        return as_dict(as_dict(lanes.get(self.lane_id)).get("goldens"))

    @property
    def first_bad(self) -> dict:
        return as_dict(self.head.get("first_bad"))

    @property
    def retired(self) -> bool:
        return lanes_mod.is_retired(self.lane)


def read_json_file(path: Path | str | None) -> dict | None:
    try:
        return regression_skip.read_optional_json(str(path) if path else None)
    except RecursionError:  # JSON nested deeper than the parser goes
        return None


def load_json_or_empty(path: Path) -> dict:
    return read_json_file(path) or {}


def lane_prefix(lane: dict) -> str:
    prefix = lane.get("prefix")
    return prefix if isinstance(prefix, str) and join_path(prefix) else "regression"


def summary_from_workdir(workdir: str | None, commit: str, branch: str,
                         lane_id: str) -> dict | None:
    """A summary rebuilt from <workdir>/<wf>/{run,comparison}.json: what a run
    that stopped before it published still knows. Same cell shape as
    run_regression.process_workflow returns."""
    root = Path(workdir) if workdir else None
    if not root or not root.is_dir():
        return None
    cells = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and is_workflow_id(p.name)):
        run, cmp_ = read_json_file(d / "run.json"), read_json_file(d / "comparison.json")
        if not run and not cmp_:
            continue
        run, cmp_ = run or {}, cmp_ or {}
        thumbs = d / "outputs" / "thumbs"
        thumb = min((p.name for p in thumbs.glob("*.webp")), default=None) if thumbs.is_dir() \
            else None
        cells[d.name] = {
            "workflow_id": d.name, "worker_status": run.get("status"),
            **{k: cmp_.get(k) for k in ("verdict", "detail", "vs_golden", "vs_previous",
                                        "thresholds_used", "golden_tag", "previous_commit",
                                        "previous_verdict", "figures")},
            "output_sha256": output_sha(run, cell_kind(cmp_), d.name), "thumbnail": thumb,
            **{k: run.get(k) for k in ("gpu_name", "timings", "vram_peak_mb", "rss_peak_mb",
                                       "comfy_version", "torch_version", "python_version",
                                       "env", "custom_nodes")}}
        if cell_kind(cmp_) != output_kinds.DEFAULT_KIND:
            cells[d.name]["output_kind"] = cell_kind(cmp_)
        if cmp_.get("level") == "smoke":  # only a smoke comparison names its level
            cells[d.name].update(level="smoke", smoke=cmp_.get("smoke"))
    if not cells:
        return None
    verdicts = [c.get("verdict") for c in cells.values()]
    return {"schema_version": 2, "branch": branch, "commit": commit, "lane": lane_id,
            "run_ts": None,
            "overall": "fail" if any(v in ("fail", "execution_error") for v in verdicts)
            else "pass",
            "has_infra_error": "infra_error" in verdicts,
            "env": next((c["env"] for c in cells.values() if c.get("env")), None),
            "workflows": cells}


def prior_result(entries: list[dict], commit: str, wf: str, cell: dict,
                 committed_ts=None) -> dict:
    """{verdict, commit} of the workflow's most recent red or green result
    before `commit`. The summary carries it; one that does not (an older one,
    or one rebuilt from the runner's files) gets it from the index, by the rule
    of index_builder.prior_verdicts: for a smoke-level cell a failure at
    reference level is looked through, as it is no failed smoke check."""
    if cell.get("prior_verdict"):
        return {"verdict": cell["prior_verdict"], "commit": cell.get("prior_commit")}
    at_smoke = cell.get("level") == "smoke"
    for e in entries:  # newest first
        if e["c"] == commit or (is_num(committed_ts) and entry_key(e)[0] > committed_ts):
            continue
        earlier = as_dict(as_dict(e.get("w")).get(wf))
        verdict = earlier.get("v")
        if verdict == "fail" and at_smoke and earlier.get("l") != "smoke":
            continue
        if verdict in ("pass", "fail", "execution_error"):
            return {"verdict": verdict, "commit": e["c"]}
    return {}


def origin_of(state: str, prior: dict, fb: dict, commit: str,
              long_cadence: bool = False) -> str | None:
    """Whether a red result started at this commit or was already there.
    "first": the workflow has no earlier red or green result on the lane. A
    workflow that fails its first run is a problem of the workflow, and must
    not read like a change in the commits the run covers. That holds for a
    smoke failure on any lane and, with `long_cadence` (a nightly or weekly
    lane), for an execution error."""
    if state in ("new", "inherited"):
        return state
    if state not in ("exec", "smoke_fail"):
        return None
    before = prior.get("verdict")
    if state == "smoke_fail" and before in ("pass", "fail", "execution_error"):
        # Nothing is compared, so only a failed check before this one is the same failure.
        return "inherited" if before == "fail" else "new"
    if before in ("pass", "fail"):
        return "new"
    if before == "execution_error":
        return "inherited"
    if state == "smoke_fail" or long_cadence:
        # A chain that started at another commit says the failure is older than this run.
        return "inherited" if fb.get("commit") and fb["commit"] != commit else "first"
    if fb.get("commit"):
        return "new" if fb["commit"] == commit else "inherited"
    return None


def own_range(summary: dict | None, wf: str, prior: dict) -> dict:
    """{prev, n}: the commit of the workflow's own last red or green result,
    and how many commits the run adds to it (1: only the tested one). The
    summary of a nightly or weekly run says so per workflow (`ranges`). Else
    the run's range is the workflow's own when both start at the same commit.
    `n` is None when the count is not known."""
    summary = as_dict(summary)
    prev = prior.get("commit")
    own = as_dict(as_dict(summary.get("ranges")).get(wf))
    if own:
        return {"prev": own.get("prev") if is_sha(own.get("prev")) else prev,
                "n": as_int(own.get("commits_between"))}
    rng = as_dict(summary.get("tested_range"))
    same = is_sha(prev) and rng.get("prev") == prev
    return {"prev": prev, "n": as_int(rng.get("commits_between")) if same else None}


def later_pass(entries: list[dict], fb: dict, prior: dict, since: dict) -> str | None:
    """The last pass of a workflow's open chain (`fb`), when it is a later one
    than the failing run recorded (`since`, with `prior` a pass): a run that
    named the workflow passed afterwards at a commit between the two, as a
    step of a search for the commit does. What the run counted then reaches
    back past that pass, so its number is no longer the failure's."""
    good, prev = fb.get("prev_good"), since.get("prev")
    if prior.get("verdict") != "pass" or not is_sha(good) or not is_sha(prev) or good == prev:
        return None
    at = {e["c"]: entry_key(e) for e in entries if e["c"] in (good, prev)}
    # An entry that is no longer in the head is older than every one that is.
    return good if good in at and (prev not in at or at[good] > at[prev]) else None


def not_identical(cell: dict) -> bool:
    """A comparison ran and found a difference, with the reference or the previous run."""
    return any(m and not m.get("error") and not m.get("identical")
               for m in (as_dict(cell.get("vs_golden")), as_dict(cell.get("vs_previous"))))


def gather_text(d: LaneData, wf: str, cell: dict, run: dict, workdir: str | None) -> dict:
    """{name, bytes, cand, ref, cand_url, ref_url, diff} of a text workflow, as
    far as each can be read: the generated text from the runner's files or the
    results branch, the reference text from the results branch. The reference
    is read only when the comparison did not find the two identical (an
    identical reference is the same text); `ref` is then None when it could
    not be read. A file over TEXT_READ_MAX is linked, not read. `diff` is the
    generated text with its changes marked, "" when the two texts show none."""
    sent = {"outputs": [o for o in as_list(run.get("outputs"))
                        if isinstance(o, dict) and not o.get("truncated")]}
    files = output_kinds.outputs_of(sent, output_kinds.TEXT_EXTENSIONS)
    vg = as_dict(cell.get("vs_golden"))
    named = as_dict(vg.get("first_diff")).get("file")
    entry = next((o for o in files if o.get("filename") == named), files[0] if files else {})
    rel = join_path("outputs", entry.get("filename"))
    if not rel or not is_num(entry.get("bytes")) or entry["bytes"] < 0:
        return {}
    out: dict = {"name": entry["filename"], "bytes": entry["bytes"]}
    published = f"{d.run_dir}/{wf}/{rel}" if d.files_ok else None
    if published:
        out["cand_url"] = raw_url(d.reader, published)
    if entry["bytes"] > TEXT_READ_MAX:
        return out
    local = Path(workdir) / wf / rel if workdir and d.source != "published" else None
    try:
        if local and local.is_file():
            with open(local, "rb") as f:  # the size in run.json is only what the run said
                out["cand"] = f.read(TEXT_READ_MAX).decode("utf-8", "replace")
    except OSError:
        pass
    if "cand" not in out and published:
        out["cand"] = d.reader.read_text(published)
    tag = cell.get("golden_tag")
    if vg and not vg.get("identical") and isinstance(tag, str) and SEGMENT_RE.fullmatch(tag):
        path = f"{d.prefix}/golden/{wf}/{tag}/{rel}"
        out["ref"] = d.reader.read_text(path) if d.reader.sha else None
        if out["ref"] is not None:
            out["ref_url"] = raw_url(d.reader, path)
        if isinstance(out["ref"], str) and isinstance(out.get("cand"), str):
            # Marked once, here: a report that has to shrink is rendered many times.
            a, b = (output_kinds.normalise_text(clean(x, lines=True))
                    for x in (out["ref"], out["cand"]))
            out["diff"] = output_kinds.word_diff(a, b) if a != b else ""
    return out


def chain_of(d: LaneData, wf: str, committed_ts=None) -> dict:
    """The lane's open failing chain of `wf`, when this commit is part of it.
    head.first_bad describes the chains that are open now: a report about an
    older commit must not take a chain that started later as its own."""
    fb = as_dict(d.first_bad.get(wf))
    start = next((e for e in d.entries if e["c"] == fb.get("commit")), None)
    here = entry_key(d.entry)[0] if d.entry else committed_ts
    if start is None or fb.get("commit") == d.commit or not is_num(here) or not here:
        return fb  # no order to check: the start is older than the head, or this commit is new
    return fb if entry_key(start)[0] <= here else {}


def gather_lane(reader: Reader, lanes: dict, commit: str, branch: str, lane_id: str,
                summary_file: str | None = None, workdir: str | None = None,
                skipped: bool = False, artifact_url: str | None = None,
                manifest: dict | None = None, packs: dict | None = None) -> LaneData:
    lane = as_dict(as_dict(lanes.get("lanes")).get(lane_id))
    if not lane:
        raise ReportError("the lane is not in manifest/lanes.json")
    prefix = lane_prefix(lane)
    root = prefix.split("/")[0]
    run_dir = join_path(prefix, "runs", branch, commit)
    d = LaneData(commit=commit, branch=branch, lane_id=lane_id, lane=lane, prefix=prefix,
                 run_dir=run_dir, reader=reader, skipped=skipped, packs=as_dict(packs),
                 artifact_url=artifact_url if isinstance(artifact_url, str)
                 and ARTIFACT_URL_RE.fullmatch(artifact_url) else None)

    head_path = join_path(root, "index", branch, f"{lane_id}.json")
    registry_path = f"{root}/index/lanes.json"
    snapshot_path = join_path(prefix, "manifest-snapshot", f"{commit}.json")
    summary_path = run_dir and f"{run_dir}/summary.json"
    marker_path = run_dir and f"{run_dir}/attempt.json"
    reader.prefetch([p for p in (head_path, registry_path, snapshot_path, summary_path,
                                 marker_path) if p])

    published = reader.read_json(summary_path) if summary_path else None
    local = read_json_file(summary_file)
    if local and local.get("commit") == commit and local.get("lane") == lane_id:
        d.summary, d.source = local, "local"
        d.published = bool(published) and published.get("run_ts") == local.get("run_ts")
    else:
        # A run that tested the commit and could not publish reports from its own
        # records: a summary an earlier attempt published must not stand in for it.
        rebuilt = None if skipped else summary_from_workdir(workdir, commit, branch, lane_id)
        if rebuilt:
            d.summary, d.source = rebuilt, "workdir"
        elif published:
            d.summary, d.source, d.published = published, "published", True
            # It was told where a run's files would be, and found none of them.
            d.left_nothing = not skipped and bool(summary_file or workdir)
    if published and not d.published:
        d.earlier = published
    d.marker = (reader.read_json(marker_path) if marker_path else None) or {}
    d.head = (reader.read_json(head_path) if head_path else None) or {}
    d.registry = reader.read_json(registry_path) or {}
    d.entries = entries_of(d.head)
    d.entry = next((e for e in d.entries if e["c"] == commit), {})
    # The manifest the run used, when its snapshot is published; else today's.
    snapshot = reader.read_json(snapshot_path) if snapshot_path else None
    d.manifest, d.manifest_is_snapshot = snapshot or as_dict(manifest), bool(snapshot)
    if not d.summary:
        return d

    cells = {wf: as_dict(cell) for wf, cell in as_dict(d.summary.get("workflows")).items()}
    # The index entry and the runner's records each describe one run: the entry
    # the published one, the records the run that just finished here.
    index_cells = as_dict(d.entry.get("w")) if d.published else {}
    run_paths = {wf: join_path(run_dir, wf, "run.json") if run_dir else None for wf in cells}
    local_runs = {wf: read_json_file(Path(workdir) / wf / "run.json")
                  for wf in cells if workdir and d.source != "published" and is_workflow_id(wf)}
    wanted = [p for wf, p in run_paths.items() if p and d.published and not local_runs.get(wf)]
    for wf, cell in cells.items():
        # A smoke failure has no reference to read.
        if (cell.get("verdict") == "fail" and is_workflow_id(wf)
                and cell.get("level") != "smoke"):
            wanted.append(f"{prefix}/golden/{wf}/current.json")
            tag = cell.get("golden_tag")
            if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag):
                wanted.append(f"{prefix}/golden/{wf}/{tag}/noise_floor.json")
    reader.prefetch(wanted)

    committed_ts = commit_meta(d).get("committed_ts")
    for wf, cell in cells.items():
        safe = is_workflow_id(wf)
        run = local_runs.get(wf) or {}
        if not run and run_paths[wf] and d.published:  # else it is another run's record
            run = reader.read_json(run_paths[wf]) or {}
        index_cell = as_dict(index_cells.get(wf))
        # Summaries older than `detail` were classified by the index builder.
        detail = cell.get("detail") or index_cell.get("d")
        kind = cell_kind(cell)
        sha = cell.get("output_sha256") or output_sha(run, kind, wf) or index_cell.get("sha")
        level = "smoke" if "smoke" in (cell.get("level"), index_cell.get("l")) else None
        state = cell_state(cell.get("verdict"), detail, sha,
                           cell.get("golden_tag") or index_cell.get("g"),
                           d.lane_references.get(wf), level)
        fb = chain_of(d, wf, committed_ts)
        prior = prior_result(d.entries, commit, wf, cell, committed_ts)
        long = lanes_mod.is_long_cadence(lane)
        origin = origin_of(state, prior, fb, commit, long)
        # A new failure that nothing compares with the run before is placed by
        # the workflow's own last result: a smoke failure on any lane, and an
        # execution error on a nightly or weekly lane, whose run covers many
        # commits.
        placed = origin == "new" and (state == "smoke_fail" or (long and state == "exec"))
        since = own_range(d.summary, wf, prior) if placed else {}
        moved = later_pass(d.entries, fb, prior, since)
        if moved:
            since["later"] = moved
        view = View(wf=wf, name=workflow_name(wf), safe=safe, cell=cell, run=run,
                    state=state, origin=origin,
                    prior=prior, sha=sha if isinstance(sha, str) else None, chain=fb,
                    kind=kind, level=level, since=since)
        d.views.append(view)
        if kind == "text" and safe and state in (*GREEN, *DRIFT, "no_ref", "smoke_fail"):
            d.texts[wf] = gather_text(d, wf, cell, run, workdir)
            if d.texts[wf].get("name") in held_back(view):
                d.texts[wf].pop("cand_url", None)  # kept off the results branch: no link
        if not safe or not (state in DRIFT or (state == "pass" and not_identical(cell))):
            continue
        if state in DRIFT:
            d.references[wf] = reader.read_json(f"{prefix}/golden/{wf}/current.json") or {}
        tag = cell.get("golden_tag")
        if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag):
            d.noise[wf] = reader.read_json(f"{prefix}/golden/{wf}/{tag}/noise_floor.json") or {}
        if kind == "text":
            continue  # no figures: its texts were read above
        if d.files_ok:
            d.figures[wf] = reader.list_dir(f"{run_dir}/{wf}/figures")
        start = fb.get("commit")
        if state == "inherited" and is_sha(start) and start != commit:
            first_dir = join_path(prefix, "runs", branch, start, wf, "figures")
            if first_dir:
                d.first_figures[wf] = (first_dir, reader.list_dir(first_dir))
    return d


# -- one lane: sections ---------------------------------------------------------------
def lane_label(lane_id: str, lane: dict) -> str:
    return md(lane.get("label") or lane_id, 120)


def lane_icon(views: list[View]) -> str:
    """The icon of the lane's result: the one its row in a history table gets."""
    return rollup({v.wf: v.state for v in views}).split(" ", 1)[0]


def has_smoke(views: list[View]) -> bool:
    return any(v.level == "smoke" for v in views)


def smoke_headline(views: list[View], covers=None) -> str:
    """The headline of a run with smoke-level workflows, or with a failure
    that has a range of its own. Failures are counted by where they started,
    and a workflow on its first run is counted apart from one that a commit
    broke. Smoke passes are counted apart from matches with a reference: they
    say less."""
    n = len(views)
    count = {s: sum(v.state == s for v in views) for s in LABELS}
    red = [v for v in views if v.state in RED]
    matched, checked = count["pass"], count["smoke_pass"]
    passes = ", ".join(p for p in (
        f"{matched} {'matches its' if matched == 1 else 'match their'} reference"
        if matched else "",
        f"{checked} passed {'its' if checked == 1 else 'their'} smoke checks"
        if checked else "") if p)
    out = []
    if red:
        new, inherited, first = (sum(v.origin == o for v in red)
                                 for o in ("new", "inherited", "first"))
        # A smoke failure can reach back past the commits of the run: to the
        # workflow's own last result.
        one_commit = not is_batch(covers) and not any(
            is_sha(v.since.get("prev")) and v.since.get("n") != 1 for v in red)
        other = len(red) - new - inherited - first
        parts = [f"{new} new {'at this commit' if one_commit else 'in this run'}" if new else "",
                 f"{inherited} inherited" if inherited else "",
                 f"{first} on {'its' if first == 1 else 'their'} first run" if first else "",
                 f"{other} not classified" if other else ""]
        out.append(f"{len(red)} of {plural(n, 'workflow')} failed: "
                   + ", ".join(p for p in parts if p) + ".")
    elif checked == n:
        out.append("The workflow passed its smoke checks." if n == 1
                   else f"All {n} workflows passed their smoke checks.")
    else:
        out.append(f"No failures: {passes}." if passes else "No failures.")
    if count["infra"]:
        out.append(f"{count['infra']} {'was' if count['infra'] == 1 else 'were'} not tested.")
    if red and passes:
        out.append(f"{passes}.")
    return " ".join(out)


def headline(views: list[View], covers=None) -> str:
    """One sentence for the run; `covers` is how many commits it tested at once."""
    n = len(views)
    if not n:
        return "The summary lists no workflows."
    count = {s: sum(v.state == s for v in views) for s in LABELS}
    red = [v for v in views if v.state in RED]
    extra = []
    # A failure with a range of its own is counted by where it started, as a
    # smoke failure is: the run's range is not the failure's.
    if has_smoke(views) or any(v.since for v in red):
        text = smoke_headline(views, covers)
    elif red:
        new = sum(v.origin == "new" for v in red)
        inherited = sum(v.origin == "inherited" for v in red)
        here = "new in this run" if is_batch(covers) else "new at this commit"
        text = f"{len(red)} of {plural(n, 'workflow')} failed"
        if new == len(red):
            text += f", {here}" + (f", which covers {covers} commits." if is_batch(covers) else ".")
        elif inherited == len(red):
            text += ", inherited from an earlier commit."
        elif new or inherited:
            parts = [f"{new} {here}" if new else "",
                     f"{inherited} inherited" if inherited else "",
                     f"{len(red) - new - inherited} not classified"
                     if len(red) > new + inherited else ""]
            text += ": " + ", ".join(p for p in parts if p) + "."
        else:
            text += "."
        if count["infra"]:
            extra.append(f"{plural(count['infra'], 'workflow')} hit infrastructure errors "
                         "(not a verdict on the commit).")
    elif count["infra"]:
        text = (f"No failures; {plural(count['infra'], 'workflow')} hit infrastructure errors "
                "(not a verdict on the commit).")
    elif count["pass"] == n:
        text = ("The workflow matches its reference." if n == 1
                else f"All {n} workflows match their reference.")
    elif count["pass"]:
        text = f"No failures; {count['pass']} of {plural(n, 'workflow')} match their reference."
    else:
        text = "No failures."
    if count["accepted"]:
        extra.append(f"{plural(count['accepted'], 'changed output')} accepted as the reference "
                     "after the run.")
    if count["no_ref"]:
        extra.append(f"{plural(count['no_ref'], 'workflow')} with no reference yet.")
    if count["unknown"]:
        extra.append(f"{plural(count['unknown'], 'workflow')} without a result.")
    return " ".join([text, *extra])


def tested_run_url(d: LaneData) -> str | None:
    """The Actions run that tested the commit."""
    summary = as_dict(d.summary)
    url = checked_run_url(summary.get("run_url")) or run_url(summary.get("run_id"))
    if not url and d.published:  # the index entry names the run that published
        url = entry_run_url(d.entry)
    if not url and d.source in ("local", "workdir") and not d.skipped:
        url = this_run_url()
    # The start marker names the run of the latest attempt: this one, if it matches.
    if not url and d.marker and (not d.summary or d.marker.get("attempt") == run_attempt(d)):
        url = checked_run_url(d.marker.get("run_url"))
    return url


def run_attempt(d: LaneData) -> int | None:
    """Which attempt at the commit the reported run was. A run rebuilt from the
    runner's files has no summary to say so; its start marker does."""
    attempt = as_int(as_dict(d.summary).get("attempt"))
    if attempt is None and d.source == "workdir":
        attempt = as_int(d.marker.get("attempt"))
    return attempt


def attempts_spent(d: LaneData) -> int:
    """The attempts the cron counts for the commit (regression_skip.attempts):
    the latest of the reported run and the start marker, which a run that died
    before it published still pushed."""
    return max(run_attempt(d) or 1, as_int(d.marker.get("attempt")) or 0, 1)


def no_attempt(d: LaneData) -> bool:
    """Whether the reported run is no attempt of the lane at the commit: it
    named its workflows (`scope` in its summary), or it published nothing (its
    summary is on the runner alone, and the results branch was read). The
    cron counts neither, so the attempt such a summary records is that of the
    runs before it, and what the cron does next does not depend on it."""
    return "scope" in as_dict(d.summary) or bool(
        d.source == "local" and not d.published and d.reader.sha)


def attempt_text(attempt) -> str:
    if (as_int(attempt) or 0) < 1:
        return ""
    if attempt <= regression_skip.MAX_ATTEMPTS:
        return f"attempt {attempt} of {regression_skip.MAX_ATTEMPTS}"
    return f"attempt {attempt}"


def earlier_text(summary: dict, url: str | None) -> str:
    """', tested <when> in [an earlier run](url)', as far as both are known."""
    when = utc(summary.get("run_ts"))
    return (f", tested {when}" if when else "") + (f" in [an earlier run]({url})" if url else "")


def smoke_on_page(d: LaneData, limits: Limits | None = None) -> bool:
    """Whether a lane report shows a smoke-level result anywhere: in this run,
    in the history rows it lists, or among the lane's open failures."""
    rows = d.entries[:(limits or Limits()).history_rows] if d.head else []
    return (has_smoke(d.views)
            or any(as_dict(cell).get("l") == "smoke"
                   for e in rows for cell in as_dict(e.get("w")).values())
            or any(is_smoke_chain(fb) for fb in d.first_bad.values()))


def title_commits(d: LaneData) -> str:
    """The commits a report is about, as its title names them: the tested one.
    A nightly or weekly run tests the newest of several commits, so its title
    names the range and how many commits it holds."""
    rng = tested_range(d)
    n, prev = as_int(rng.get("n")), rng.get("prev")
    if d.summary and lanes_mod.is_long_cadence(d.lane) and is_batch(n) and is_sha(prev):
        return f"{tick(prev[:7])}..{tick(d.commit[:7])} ({n} commits)"
    return tick(d.commit[:8])


def progress_text(summary) -> str:
    """'40 of 55 workflows have a result', for a summary that a run pushed
    before it had every result (`partial`: a nightly or weekly run does so
    every few workflows). '' for any other summary. The run's final summary
    takes its place, so one that is read belongs to a run that did not finish,
    or that is still going."""
    part = as_dict(summary).get("partial")
    if not isinstance(part, dict):
        return ""
    done, of = as_int(part.get("done")), as_int(part.get("of"))
    if done is None or of is None or not 0 <= done <= of:
        return "not every workflow has a result"
    return f"{done} of {plural(of, 'workflow')} {'has' if of == 1 else 'have'} a result"


def stopped_early(d: LaneData) -> bool:
    """A nightly or weekly run that reports from the files on the runner: it
    did not reach its final publish (stopped, killed, or the push failed), so
    what it has is not the lane's result of the commit. Such a run tests its
    workflows a few at a time, and one that is stopped has started only some
    of them: a page with every started workflow green is not a green run."""
    return d.source == "workdir" and lanes_mod.is_long_cadence(d.lane)


def retests_of(summary) -> dict[str, dict]:
    """Per workflow that a later run tested again at this commit (`retests`
    in the summary): the last such run. Its cell is that run's, and every
    other fact of the summary is still the lane's own run of the commit."""
    found: dict[str, dict] = {}
    for run in as_list(as_dict(summary).get("retests")):
        for wf in as_list(as_dict(run).get("workflows")):
            if isinstance(wf, str):
                found[wf] = run
    return found


def retest_url(run) -> str | None:
    """The Actions run of a re-test, when its record names one."""
    run = as_dict(run)
    return checked_run_url(run.get("run_url")) or run_url(run.get("run_id"))


def title_block(d: LaneData, limits: Limits | None = None) -> list[str]:
    mark = lane_icon(d.views) if d.summary else DASH
    summary = as_dict(d.summary)
    unfinished = progress_text(summary)
    if (unfinished or stopped_early(d)) and mark == CHECK:
        mark = WARN  # green so far is not green: workflows are still without a result
    out = [f"## {mark} GPU regression · {lane_label(d.lane_id, d.lane)} · {title_commits(d)}"]
    if unfinished:
        # A page that only shows what is published (a report-only run, a
        # skipped commit) is another run's than the one that did not finish.
        who = "The run that tested this commit" if d.skipped else "This run"
        out += ["", "> [!WARNING]",
                f"> {who} did not finish: {unfinished}. The report shows what the run "
                "pushed before it stopped, and the other workflows are not in it. A run that "
                "is still going reads the same until it publishes its last result."]
    if d.skipped and d.summary:
        has = "part of a result" if unfinished else "a result"
        note = (f"This run did not test this commit again: the lane already has {has} for it"
                + earlier_text(summary, tested_run_url(d)))
        out += ["", "> [!NOTE]", f"> {note}. The report below shows that result."]
    if d.summary and not d.reader.sha:
        out += ["", "> [!WARNING]",
                "> The results branch could not be read. History, reference details, images "
                "and links to published files are left out."]
    elif d.source == "workdir":
        text = ("> This run stopped before it published its results. The report is rebuilt "
                "from the files on the runner, so images and links to published files are "
                "missing. The attached files hold the outputs.")
        if stopped_early(d):
            text += (" It shows the workflows that had a result when the run stopped, and no "
                     "other.")
        if progress_text(d.earlier):
            # What a run pushed on its way, this one or an attempt before it.
            text += (" The results branch holds part of a result for this commit: "
                     f"{progress_text(d.earlier)}, pushed by a run before it had them all.")
        elif d.earlier:
            url = (checked_run_url(d.earlier.get("run_url")) or run_url(d.earlier.get("run_id"))
                   or entry_run_url(d.entry))
            text += (" The results branch holds an earlier result for this commit"
                     + earlier_text(d.earlier, url) + ": the history table still shows that one.")
        out += ["", "> [!WARNING]", text]
    elif d.source == "local" and not d.published:
        out += ["", "> [!WARNING]",
                "> The results branch does not hold this run's summary. The report is built "
                "from the files on the runner, so images and links to published files are "
                "left out."]
    elif d.left_nothing:
        out += ["", "> [!WARNING]",
                "> This run left no result on the runner. The report below shows the result "
                "that is already published for this commit"
                + earlier_text(summary, tested_run_url(d)) + "."]
    if d.summary:
        out += ["", headline(d.views, tested_range(d).get("n"))]
        if smoke_on_page(d, limits):
            out += ["", SMOKE_NOTE]
    return out


def no_summary_block(d: LaneData) -> list[str]:
    """Nothing is published for the commit: say so, show the start marker."""
    if not d.reader.sha:
        return ["The results branch could not be read, so the published result of this commit "
                "on this lane cannot be shown."]
    if d.skipped:
        out = ["This run did not test this commit, and the lane has no published result for it."]
    else:
        out = ["No result is published for this commit on this lane: the run stopped before "
               "it wrote a summary."]
    if d.marker:
        bits = [attempt_text(d.marker.get("attempt")),
                f"started {utc(d.marker.get('started_ts'))}" if utc(d.marker.get("started_ts"))
                else ""]
        url = checked_run_url(d.marker.get("run_url"))
        if url:
            bits.append(f"[the run that started it]({url})")
        bits = [b for b in bits if b]
        if bits:
            out += ["", "The last start marker on the results branch: " + ", ".join(bits) + "."]
        if d.skipped and (as_int(d.marker.get("attempt")) or 0) >= regression_skip.MAX_ATTEMPTS:
            out.append(f"The cron stops after {plural(regression_skip.MAX_ATTEMPTS, 'attempt')}. "
                       "Dispatch **GPU regression** with this commit to test it again.")
    else:
        out += ["", "There is no start marker for it either."]
    return out


def commit_meta(d: LaneData) -> dict:
    meta = as_dict(as_dict(d.summary).get("commit_meta"))
    if meta:
        return meta
    m = as_dict(d.entry.get("m"))  # summaries older than commit_meta: the index has it
    return {"subject": m.get("s"), "author": m.get("a"), "committed_ts": m.get("ct"),
            "pr": m.get("pr"), "pr_title": m.get("pt"), "pr_author": m.get("pa")}


def tested_range(d: LaneData) -> dict:
    rng = as_dict(as_dict(d.summary).get("tested_range"))
    if rng:
        return {"prev": rng.get("prev"), "n": rng.get("commits_between")}
    if d.entry.get("prev"):
        return {"prev": d.entry.get("prev"), "n": as_dict(d.entry.get("range")).get("n")}
    return {}


def tested_between(d: LaneData, prev) -> int:
    """How many commits after `prev` and before this one the lane has results
    for. The range starts at the previous-run pointer, which skips runs that
    did not finish every workflow: commits inside it can have been tested."""
    start = next((e for e in d.entries if e["c"] == prev), None)
    if not start or not d.entry:
        return 0
    low, high = entry_key(start), entry_key(d.entry)
    return sum(low < entry_key(e) < high for e in d.entries)


def commit_rows(commit: str, meta: dict) -> list[list[str]]:
    bits = [sha_link(commit, 12), md(meta.get("subject"), 200)]
    if meta.get("author"):
        bits.append(f"by {md(meta['author'], 60)}")
    if utc(meta.get("committed_ts")):
        bits.append(f"committed {utc(meta['committed_ts'])}")
    parents = [sha_link(p) for p in as_list(meta.get("parents"))[:4] if is_sha(p)]
    if parents:
        bits.append(("parent " if len(parents) == 1 else "parents ") + ", ".join(parents))
    rows = [["Commit", " · ".join(b for b in bits if b)]]
    if pr_link(meta.get("pr")):
        bits = [pr_link(meta["pr"]), md(meta.get("pr_title"), 200)]
        if meta.get("pr_author"):
            bits.append(f"by {md(meta['pr_author'], 60)}")
        rows.append(["Pull request", " · ".join(b for b in bits if b)])
    return rows


def backends_text(lines) -> str:
    """Kernel backends from ComfyUI's startup log lines. The worker sends other
    startup lines along (torch version, the GPU): they are not backends, and
    the row already has them."""
    out = []
    for line in as_list(lines)[:12]:
        m = re.search(r"backend\s+([\w.-]+):\s*\{'available':\s*(True|False)"
                      r"(?:,\s*'disabled':\s*(True|False))?", clean(line, 400))
        if not m:
            continue
        if m.group(2) == "False":
            out.append(f"{md(m.group(1))} (not available)")
        else:
            out.append(md(m.group(1)) + (" (disabled)" if m.group(3) == "True" else ""))
    return ", ".join(o for o in out if o)


def environment_text(summary: dict) -> str:
    env = as_dict(summary.get("env"))
    cells = [as_dict(c) for c in as_dict(summary.get("workflows")).values()]

    def first(key):
        return next((c[key] for c in cells if c.get(key)), None)

    python = env.get("python") or first("python_version")
    python = python.split()[0] if isinstance(python, str) and python.split() else None
    parts = [("GPU", first("gpu_name")), ("driver", env.get("driver_version")),
             ("Python", python), ("torch", env.get("torch") or first("torch_version")),
             ("CUDA", env.get("cuda")), ("ComfyUI", first("comfy_version")),
             ("worker image", env.get("image"))]
    out = [f"{name} {md(value, 80)}" for name, value in parts if value]
    build = env.get("image_sha")  # the commit of the worker repository the image was built from
    if is_sha(build):
        out.append(f"image build [{tick(build[:7])}](https://github.com/{worker_repo()}/commit/"
                   f"{build})")
    baked = env.get("custom_nodes_sha256")  # the custom-node packs baked into the image
    if isinstance(baked, str) and HEX64_RE.fullmatch(baked):
        out.append(f"baked packs {tick(baked[:12])}")
    backends = backends_text(env.get("runtime_backends"))
    if backends:
        out.append(f"backends: {backends}")
    return " · ".join(out)


def facts_block(d: LaneData) -> list[str]:
    summary = as_dict(d.summary)
    meta = commit_meta(d)
    rows = commit_rows(d.commit, meta)

    tested = [utc(summary.get("run_ts"))]
    run_ts, committed = summary.get("run_ts"), meta.get("committed_ts")
    if is_num(run_ts) and is_num(committed) and run_ts >= committed > 0:
        minutes = (run_ts - committed) / 60
        tested.append(f"{minutes:.0f} min after the commit" if minutes < 120
                      else f"{minutes / 60:.1f} h after the commit")
    # The attempt is named where it tells something: a retry happened or can follow.
    # "attempt 1 of 2" on a run with a verdict reads as if a second one were due,
    # and so it does when the only untested workflows are ones the cron leaves alone.
    # A run that is no attempt of the lane names none.
    attempt = run_attempt(d)
    if not no_attempt(d) and ((attempt or 0) > 1 or d.source == "workdir"
                              or cron_would_retry(d)):
        tested.append(attempt_text(attempt))
    url = tested_run_url(d)
    if url:
        tested.append(f"[Actions run]({url})")
    tested = [t for t in tested if t]
    if tested:
        rows.append(["Tested", " · ".join(tested)])
    if progress_text(summary):
        rows.append(["Progress", f"{progress_text(summary)}. The run did not finish"])
    # Workflows a later run tested again at this commit: their cells are that
    # run's, and every other row of this table is the lane's own run.
    again = []
    for run in as_list(summary.get("retests"))[-10:]:
        run = as_dict(run)
        names = [workflow_name(wf) for wf in as_list(run.get("workflows"))[:20]]
        url = retest_url(run)
        if names:
            again.append(" · ".join(x for x in (", ".join(names), utc(run.get("run_ts")),
                                                 f"[Actions run]({url})" if url else "") if x))
    if again:
        rows.append(["Tested again", "; ".join(again)])

    rng = tested_range(d)
    if rng.get("prev"):
        n, prev = as_int(rng.get("n")), sha_link(rng["prev"])
        cmp_ = compare_link(rng["prev"], d.commit)
        between = tested_between(d, rng["prev"])
        if n == 1:
            covers = f"1 commit since {prev}"
        elif n is not None and n > 1 and lanes_mod.is_long_cadence(d.lane):
            # Every full run of a nightly or weekly lane moves the pointer. The
            # range is a fact about the run: a failure says its own.
            covers = (f"{n} commits since {prev}" + (f" ({cmp_})" if cmp_ else "")
                      + ", the commit of the lane's previous full run. Each failure says "
                      "where it started: a workflow's own last result can be older")
        elif n is not None and n > 1 and between:
            # The previous-run pointer only moves on a run that finished every workflow.
            covers = (f"{n} commits since {prev}" + (f" ({cmp_})" if cmp_ else "")
                      + ", the commit of the previous run that finished every workflow. The "
                      f"lane tested {between} of them in between; outputs are still compared "
                      f"with that run, so a change seen here is in one of the {n}")
        elif n is not None and n > 1:
            covers = (f"{n} commits since {prev}" + (f" ({cmp_})" if cmp_ else "")
                      + ": a change seen here is in one of them")
        elif n is not None:
            covers = (f"no new commit since {prev}, the previous tested commit: this commit "
                      "is not ahead of it")
        else:
            covers = f"commits since {prev}, count unknown" + (f" ({cmp_})" if cmp_ else "")
        rows.append(["Covers", covers])

    lane = d.lane
    lane_bits = [lane_label(d.lane_id, lane),
                 tick(d.lane_id) if LANE_RE.fullmatch(d.lane_id) else code(d.lane_id),
                 f"branch {tick(d.branch) if is_branch(d.branch) else code(d.branch)}",
                 f"role {md(lane.get('role'), 30)}" if lane.get("role") else "",
                 "retired: history only" if d.retired
                 else f"cadence {md(lane.get('cadence'), 30)}" if lane.get("cadence") else "",
                 "blocking" if lane.get("blocking") else "informational"]
    rows.append(["Lane", " · ".join(b for b in lane_bits if b)])
    env = environment_text(summary)
    if env:
        rows.append(["Environment", env])
    # Only a nightly or weekly run counts what it added to the results branch:
    # the output files of its smoke-level workflows. The number is the lane's
    # own run's, also on a page that a later run renders (a skipped night, a
    # re-test), so the row does not say "this run".
    added = as_int(summary.get("published_new_bytes"))
    if added is not None and added >= 0:
        rows.append(["New smoke-level output the lane's run published", size_text(added)])
    return table(["", ""], rows)


def with_limit(value, spec: str, limit, sign: str, crossed: bool, extra: str = "") -> str:
    """A number next to its limit; bold when it crossed the limit."""
    if not is_num(value):
        return ""
    text = f"{value:{spec}}"
    if not is_num(limit):
        return text + extra
    return (f"**{text}**" if crossed else text) + f"{extra} (limit {sign} {limit:g})"


def metric_cells(metrics, th=None, other: str = "the reference") -> list[str]:
    """Mean MSE, PSNR mean (min), % pixels changed, max abs diff, frames."""
    m, th = as_dict(metrics), as_dict(th)
    if m.get("error"):
        return [f"comparison failed: {md(m['error'], 300)}", "", "", "", ""]
    if not m:
        return ["not compared", "", "", "", ""]
    ref, cand = m.get("frame_count_ref"), m.get("frame_count_cand")
    frames = fmt(m.get("frames_compared"), "g")
    if is_num(ref) and is_num(cand) and ref != cand:
        frames = f"**{cand:g} here, {ref:g} in {other}**"
    if m.get("identical"):
        return ["bit-identical", "", "", "", frames]
    mse, psnr, pct = m.get("mean_mse"), m.get("mean_psnr_db"), m.get("mean_pct_pixels_changed")
    lim_mse, lim_psnr, lim_pct = (th.get("max_mean_mse"), th.get("min_mean_psnr_db"),
                                  th.get("max_pct_pixels_changed"))
    low = fmt(m.get("min_psnr_db"), ".2f")
    return [with_limit(mse, ".4f", lim_mse, "≤", is_num(mse) and is_num(lim_mse)
                       and mse > lim_mse),
            with_limit(psnr, ".2f", lim_psnr, "≥", is_num(psnr) and is_num(lim_psnr)
                       and psnr < lim_psnr, f", min {low}" if low else ""),
            with_limit(pct, ".3f", lim_pct, "≤", is_num(pct) and is_num(lim_pct)
                       and pct > lim_pct),
            fmt(m.get("max_abs_diff"), "g"), frames]


TEXT_COLUMNS = ["Compared with", "Result", "Similarity", "First difference", "Characters",
                "Lines"]


def text_metric_cells(metrics, th=None, other: str = "the reference",
                      judged: bool = True) -> list[str]:
    """Result, similarity, first difference, characters and lines of a text
    comparison. Bold marks what fails, so only in a row that is `judged`: the
    one against the reference."""
    m, th = as_dict(metrics), as_dict(th)
    if m.get("error"):
        return [f"comparison failed: {md(m['error'], 300)}", "", "", "", ""]
    if not m:
        return ["not compared", "", "", "", ""]
    bold = "**" if judged else ""

    def pair(key: str) -> str:  # a count here and in the other text, said once when equal
        cand, ref = m.get(f"{key}_cand"), m.get(f"{key}_ref")
        if is_num(cand) and is_num(ref) and cand != ref:
            return f"{cand:.0f} here, {ref:.0f} in {other}"
        return fmt(cand, ".0f")

    # Files pair by name: one without its pair was never compared.
    files = ""
    ref_n, cand_n, paired = (m.get(k) for k in ("file_count_ref", "file_count_cand",
                                                "files_compared"))
    if is_num(ref_n) and is_num(cand_n) and ref_n != cand_n:
        files = f"{plural(round(cand_n), 'file')} here, {ref_n:.0f} in {other}"
    elif is_num(ref_n) and is_num(paired) and paired < ref_n:
        files = (f"only {paired:.0f} of {ref_n:.0f} text files" if paired else
                 "no text file") + f" with the same name as in {other}"
    files = f" ({bold}{files}{bold})" if files else ""
    if m.get("identical"):
        return ["identical" + files, "", "", pair("chars"), pair("lines")]
    floor, sim = th.get("min_text_similarity"), m.get("similarity")
    shown = ""
    if is_num(sim) and paired != 0:  # with no pair of files there is nothing to be similar
        shown = f"{output_kinds.percent(sim)} %{output_kinds.basis_note(m)}"
        if is_num(floor):
            shown = (f"**{shown}**" if sim < floor else shown) + f" (limit ≥ {floor * 100:g} %)"
    diff = as_dict(m.get("first_diff"))
    where = ""
    if is_num(diff.get("line")) and is_num(diff.get("col")):
        where = f"line {diff['line']:.0f}, column {diff['col']:.0f}"
        if is_num(ref_n) and ref_n > 1 and diff.get("file"):
            where += f" of {code(diff['file'], 120)}"
    within = is_num(sim) and is_num(floor) and sim >= floor and not files
    return [("differs" if within or not judged else "**differs**") + files, shown, where,
            pair("chars"), pair("lines")]


def text_shown(text: str, limit: int) -> tuple[str, bool, str]:
    """(what a report shows of a text, whether that is less than all of it, a
    note for its heading): no control characters, long lines folded, cut to
    `limit` characters. The note says when lines were folded: a line and
    column in the table count in the file, not in what is shown."""
    plain = clean(text, lines=True).strip("\n")
    text = output_kinds.wrap_lines(plain)
    folded = ", long lines folded" if text.count("\n") > plain.count("\n") else ""
    if len(text) <= limit:
        return text, False, folded
    return text[:limit].rstrip() + "\n…", True, folded


def text_lines(d: LaneData, v: View, limits: Limits, diff: bool = True) -> list[str]:
    """What a text workflow generated, the reference text and their difference.
    All three are model output: untrusted, and shown only inside <pre> blocks.
    A text is cut to limits.text_chars characters, and its heading says so."""
    t = as_dict(d.texts.get(v.wf))
    cand, ref = (t.get(k) if isinstance(t.get(k), str) else None for k in ("cand", "ref"))
    link = f" [The whole file]({t['cand_url']})." if t.get("cand_url") else ""
    if is_num(t.get("bytes")) and t["bytes"] > TEXT_READ_MAX:
        return [f"The generated text is {size_text(t['bytes'])}, too large to show here.{link}"]
    if cand is None and ref is None:
        return []
    if not limits.text_chars:
        return [f"The generated text is left out here to keep the summary small.{link}"]
    out: list[str] = []
    vg = as_dict(v.cell.get("vs_golden"))
    marked = t.get("diff")  # set when both texts were read
    if diff and isinstance(marked, str) and marked:
        shown, cut, folded = text_shown(marked, limits.text_chars * 2)
        note = f", cut to its first {limits.text_chars * 2} characters" if cut else ""
        # Texts too long to mark word by word come as a line diff, which has other signs.
        form = ("line by line, with <code>-</code> before a removed line and <code>+</code> "
                "before an added one" if marked.startswith(output_kinds.LINE_DIFF_HEAD) else
                "with removed text as <code>[-…-]</code> and added text as <code>{+…+}</code>")
        out += [f"**Difference from the reference**, {form}{folded}{note}:", "", pre(shown), ""]
    elif diff and isinstance(marked, str) and as_dict(vg.get("first_diff")):
        # The comparison found a difference that the cleaned texts no longer have.
        out += ["The two texts differ only in characters that are not shown here (control or "
                "zero-width characters, or a tab where the other text has blanks). Compare the "
                "two files.", ""]
    elif diff and cand is not None and ref is None and "ref" in t:
        if vg.get("files_compared") == 0:  # the reference is there, under another name
            out += [f"The reference has no text file named {code(t.get('name'), 120)}, so the "
                    "difference is not shown.", ""]
        else:
            out += ["The reference text could not be read, so the difference is not shown.", ""]
    for text, title, url in ((cand, "Generated text", t.get("cand_url")),
                             (ref, "Reference text", t.get("ref_url"))):
        if text is None:
            continue
        shown, cut, folded = text_shown(text, limits.text_chars)
        size = plural(len(output_kinds.normalise_text(text)), "character")
        if cut:
            size += f", the first {limits.text_chars} shown"
        size += folded
        body = [pre(shown)] if shown else ["The text is empty."]
        if url:
            body += ["", f"[The whole file]({url})"]
        out += [*details(f"{title} ({size})", body), ""]
    return out[:-1]


def reference_text(ref: dict) -> str:
    """Where a reference came from: tag, when it was approved, by whom, why."""
    tag = ref.get("tag")
    text = tick(tag) if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag) else code(tag, 60)
    approved = " ".join(b for b in (utc(ref.get("blessed_ts")),
                                    f"by {md(ref['blessed_by'], 60)}" if ref.get("blessed_by")
                                    else "") if b)
    if approved:
        text += f", approved {approved}"
    if ref.get("reason"):
        text += f". Reason: {md(ref['reason'], 400)}"
    return text


FIGURES = (("side_by_side_golden.png", "Reference, this commit and their difference"),
           ("diff_heatmap_golden.png", "Difference from the reference, worst frame"),
           ("side_by_side_prev.png", "Previous tested commit, this commit and their difference"),
           ("diff_heatmap_prev.png", "Difference from the previous tested commit, worst frame"))


def figure_lines(d: LaneData, v: View) -> list[str]:
    if not d.files_ok or not v.safe:  # an id that is no plain name builds no URL
        return []
    listed = set(d.figures.get(v.wf) or [])
    if not listed and v.cell.get("figures") is True:
        listed = {name for name, _ in FIGURES[:2]}  # the summary says they were rendered
    out: list[str] = []
    previous: list[str] = []
    for name, caption in FIGURES:
        url = raw_url(d.reader, f"{d.run_dir}/{v.wf}/figures/{name}")
        if name not in listed or not url:
            continue
        img = (f'<a href="{attr(url)}"><img src="{attr(url)}" width="{FIGURE_PX}" '
               f'alt="{attr(caption)}"></a>')
        (previous if name.endswith("_prev.png") else out).extend([f"**{caption}**", "", img, ""])
    if previous:
        out += details("Figures against the previous tested commit", previous[:-1]) + [""]
    return out[:-1] if out else out


def rebless_lines(d: LaneData, v: View) -> list[str]:
    """The two commands that make this commit's output the reference."""
    if d.retired or not (FULL_SHA_RE.fullmatch(d.commit) and v.safe
                         and LANE_RE.fullmatch(d.lane_id)):
        return []
    base = (f"gh workflow run golden-baseline.yml --repo {worker_repo()} -f ref={d.commit} "
            f"-f workflows={v.wf} -f lane={d.lane_id}")
    return ["If the change is intended, approve the new output as the reference. Make a "
            "candidate from this commit first, look at its output, then approve it with the "
            "second command:", "", "```", f"{base} -f bless=false", "```", "", "```",
            f"{base} -f bless_only=true", "```"]


def batch_size(d: LaneData) -> int | None:
    """How many commits the run tested at once, when that is more than one."""
    n = tested_range(d).get("n")
    return n if is_batch(n) else None


def covers_text(d: LaneData, clause: str) -> str:
    """' This run covers N commits, so <clause> (compare).' for a run that
    tested several commits at once, else ''."""
    n = batch_size(d)
    if not n:
        return ""
    cmp_ = compare_link(tested_range(d).get("prev"), d.commit)
    return f" This run covers {n} commits, so {clause}" + (f" ({cmp_})." if cmp_ else ".")


def comparison_table(d: LaneData, v: View) -> list[str]:
    """The output against its reference, the previous run and the noise floor,
    each number next to its limit."""
    cell = v.cell
    text = v.kind == "text"
    cells = text_metric_cells if text else metric_cells
    # A text is judged against its reference only: no bold in the other rows.
    unjudged = {"judged": False} if text else {}
    prev = cell.get("previous_commit")
    tag = cell.get("golden_tag")
    tag_text = tick(tag) if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag) else code(tag, 60)
    rows = [[f"Reference {tag_text}".strip(),
             *cells(cell.get("vs_golden"), cell.get("thresholds_used"))]]
    if cell.get("vs_previous") is not None or prev:
        rows.append([f"Previous tested commit {sha_link(prev) if prev else ''}".strip(),
                     *cells(cell.get("vs_previous"), None, "the previous run", **unjudged)])
    noise = as_dict(d.noise.get(v.wf))
    if noise:
        how = "" if noise.get("measured", True) else ", not measured, taken from " + (
            code(noise.get("inherited_from"), 60) or "an earlier reference")
        rows.append([f"Noise floor (the reference {'generated' if text else 'rendered'} "
                     f"twice{how})",
                     *cells(noise, None, "the first run" if text else "the first render",
                            **unjudged)])
    if text:
        return table(TEXT_COLUMNS, rows)
    return table(["Compared with", "Mean MSE", "PSNR dB mean, min", "% pixels changed",
                  "Max abs diff", "Frames"], rows)


def differs_text(v: View) -> str:
    """How a failing output differs from its reference, in the words of its kind."""
    if v.kind != "text":
        return "The output differs from the reference beyond its limits"
    if as_dict(v.cell.get("vs_golden")).get("error"):
        return "The generated text could not be compared with the reference"
    if is_num(as_dict(v.cell.get("thresholds_used")).get("min_text_similarity")):
        return "The generated text differs from the reference beyond its similarity limit"
    return "The generated text differs from the reference"


def drift_block(d: LaneData, v: View, limits: Limits | None = None) -> list[str]:
    cell = v.cell
    prev = cell.get("previous_commit")
    vs_prev = as_dict(cell.get("vs_previous"))
    differs = differs_text(v)
    out: list[str] = []
    if v.state == "accepted":
        out.append("The output differs from the reference this run was compared with. The "
                   "lane's reference was approved again after the run and is now this very "
                   "output, so nothing is open.")
    elif v.state == "new":
        out.append(f"{differs}, and it differs from "
                   f"the previous tested commit{' ' + sha_link(prev) if prev else ''}: the change "
                   f"is new {'in this run' if batch_size(d) else 'at this commit'}."
                   + covers_text(d, "the change is in one of them"))
    elif v.state == "inherited":
        out.append(f"{differs}. It is identical "
                   f"to the previous tested commit{' ' + sha_link(prev) if prev else ''}, which "
                   "already failed: the change is inherited from an earlier commit.")
    elif vs_prev.get("identical"):
        out.append(f"{differs}, yet it is "
                   f"identical to the previous tested commit{' ' + sha_link(prev) if prev else ''}"
                   ", which did not fail: the reference changed in between, not the output.")
    elif is_sha(prev) and not vs_prev and lanes_mod.is_long_cadence(d.lane):
        # A full run of a nightly or weekly lane always moves the pointer, so
        # the run before can be one in which this workflow wrote nothing.
        out.append(f"{differs}. The run before, at {sha_link(prev)}, left no output of this "
                   "workflow, so nothing tells whether the change is new in this run.")
    else:
        out.append(f"{differs}. There is no "
                   "previous run to tell whether the change is new at this commit.")
    since = failing_since(v.chain, d.entries)
    if since and v.state != "accepted":
        out += ["", since[0].upper() + since[1:] + "."]
    out += ["", *comparison_table(d, v)]

    tag = cell.get("golden_tag")
    ref = as_dict(d.references.get(v.wf))
    if ref.get("tag"):
        if ref["tag"] == tag:
            out += ["", f"Reference: {reference_text(ref)}"]
        else:
            tag_text = (tick(tag) if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag)
                        else code(tag, 60))
            out += ["", f"The lane's reference today is {reference_text(ref)}",
                    "", f"It is not the reference this run was compared with ({tag_text})."]
    if v.state == "accepted":
        return out
    # Runs from before inherited failures stopped rendering figures have their own.
    # A text workflow has no figures: it shows the texts and their difference.
    if v.kind == "text":
        figures = text_lines(d, v, limits or Limits())
    else:
        figures = figure_lines(d, v) if v.state != "inherited" or d.figures.get(v.wf) else []
    if figures:
        out += ["", *figures]
    elif v.state == "inherited" and v.kind != "text":
        first_dir, names = d.first_figures.get(v.wf) or (None, [])
        links = [f"[{caption}]({raw_url(d.reader, f'{first_dir}/{name}')})"
                 for name, caption in FIGURES if name in names and raw_url(d.reader, first_dir)]
        if links:
            out += ["", "An inherited failure has no figures of its own. The figures of the "
                    f"first failing commit {sha_link(v.chain.get('commit'))} show the same "
                    "difference: " + ", ".join(links) + "."]
        else:
            out += ["", "An inherited failure has no figures of its own. They are on the commit "
                    "where the change started."]
    commands = rebless_lines(d, v)
    if commands:
        out += ["", *commands]
    return out


def failure_detail(d: LaneData, v: View, limits: Limits) -> list[str]:
    """Why a workflow has no verdict: the worker's failure record by shape."""
    run, cell = v.run, v.cell
    err = run.get("error")
    if err is None:
        err = cell.get("error")  # a harness error is a string on the summary cell
    if isinstance(err, str):
        err = {"message": err}
    err = as_dict(err)
    bullets: list[str] = []
    if err.get("message"):
        bullets.append(f"**Message:** {md(err['message'], 2000)}")
    if err.get("node_type") or err.get("node_id"):
        bullets.append(f"**Node:** type {code(err.get('node_type'), 120) or 'unknown'}, "
                       f"id {code(err.get('node_id'), 40) or 'unknown'}")
    if is_num(err.get("timeout_s")):
        text = f"**Timeout:** the worker gave up after {err['timeout_s']:g} s"
        if err.get("prompt_id"):
            text += f" (prompt {code(err['prompt_id'], 80)})"
        bullets.append(text)
    if is_num(err.get("returncode")):
        bullets.append(f"**Server exit code:** {err['returncode']:g}")
    if err.get("startup_failure"):
        bullets.append(f"**Server start:** {md(err['startup_failure'], 300)}")
    for key, name in (("missing", "Missing models"), ("missing_nodes", "Missing node classes"),
                      ("stripped_inputs", "Inputs removed"),
                      ("filled_defaults", "Inputs filled from defaults")):
        items = as_list(err.get(key))
        if items:
            more = f", +{len(items) - 20} more" if len(items) > 20 else ""
            bullets.append(f"**{name}:** " + ", ".join(code(i, 200) for i in items[:20]) + more)
    if err.get("error"):
        bullets.append(f"**Validation:** {md(err['error'], 1000)}")
    packs = as_dict(err.get("custom_nodes"))
    if packs:
        bullets.append("**Custom-node packs:** " + "; ".join(
            f"{code(name, 80)} {md(as_dict(info).get('import'), 30) or 'unknown'}"
            + (f" ({as_dict(info)['seconds']:g} s)" if is_num(as_dict(info).get("seconds"))
               else "") for name, info in list(packs.items())[:20]))
    if not err:
        bullets.append("The worker recorded no detail for this failure.")

    job = []
    if run.get("runpod_job_id"):
        job.append(f"RunPod job {code(run['runpod_job_id'], 80)}")
    delay = run.get("delay_s")
    if is_num(run.get("delay_ms")):
        delay = run["delay_ms"] / 1000
    elif is_num(delay) and not is_num(as_dict(d.summary).get("schema_version")):
        delay = delay / 1000  # records older than schema 2 hold milliseconds here
    if is_num(delay):
        job.append(f"queued {delay:.1f} s")
    if is_num(run.get("execution_ms")):
        job.append(f"ran {run['execution_ms'] / 1000:.1f} s")
    if second_try(v):
        # Each job of such a run is submitted once, so "attempt 1" would hide the first.
        job.append("the run's second job for this workflow")
    elif is_num(run.get("attempt")):
        job.append(f"RunPod attempt {run['attempt']:g}")
    if job:
        bullets.append(", ".join(job))
    out = [f"- {b}" for b in bullets]

    if err.get("detail"):
        out += ["", "The server rejected the prompt:", "",
                pre(clean(err["detail"], 4000, lines=True))]
    trace = err.get("traceback")
    if isinstance(trace, list):
        trace = "\n".join(str(t).rstrip("\n") for t in trace)
    if isinstance(trace, str) and trace.strip():
        lines = clean(trace, lines=True).strip("\n").split("\n")
        if limits.traceback_lines:
            cut = lines[-limits.traceback_lines:]
            title = ("Traceback" if len(cut) == len(lines)
                     else f"Traceback (last {len(cut)} of {len(lines)} lines)")
            out += ["", f"{title}:", "", pre("\n".join(cut)[-TRACEBACK_CHARS:])]
        else:
            out += ["", "A traceback was recorded; it is left out here to keep the summary small."]
    tail = err.get("log_tail")
    if isinstance(tail, str) and tail.strip():
        text = clean(tail, lines=True).strip("\n")
        if limits.log_chars:
            shown = text[-limits.log_chars:]
            what = (f"Server log tail ({len(shown)} characters)" if len(shown) == len(text)
                    else f"Server log tail (last {len(shown)} of {len(text)} characters)")
            out += ["", *details(what, [pre(shown)])]
        else:
            out += ["", "A server log tail was recorded; it is left out here to keep the "
                    "summary small."]
    return out


NO_RETRY_WHY = {"missing_model": "Its models are not on the volume yet",
                "model_access_error": "The model host refuses a file it needs",
                "not_reached": "The run's deadline came before it could start"}


def cron_would_retry(d: LaneData) -> bool:
    """Whether the commit has an untested workflow that another run could
    test: the cron's reason for its one retry (regression_skip.has_infra_error).
    A workflow whose status is in regression_skip.NO_RETRY is not one."""
    statuses = [v.cell.get("worker_status") for v in d.views if v.state == "infra"]
    return any(not (isinstance(s, str) and s in regression_skip.NO_RETRY) for s in statuses)


def second_try(v: View) -> bool:
    """Whether a workflow's result is that of the second job a nightly or
    weekly run gave it, in the pass at its end (`"tries": 2` in its cell)."""
    return as_int(v.cell.get("tries")) == 2


def untested_on_a_long_lane(d: LaneData, v: View) -> list[str]:
    """What happens to a workflow that a nightly or weekly run did not test.
    The run itself tries it once more at its end, when that can help and its
    deadline allows: the text says whether it did. After the run nothing
    does: the lane's next run is a day or a week away, so the command is given."""
    status = v.cell.get("worker_status")
    if second_try(v):
        tried = ("The run tried it once more at its end, and this is the result of that "
                 "second job.")
    elif v.run.get("reason") == "timed_out":  # submit_and_poll.TIMED_OUT, in its run.json
        tried = ("Its job used up its whole time, so the run did not try it again: a second "
                 "job can cost as much.")
    elif isinstance(status, str) and status in NO_RETRY_WHY:
        tried = (f"{NO_RETRY_WHY[status]}. Another attempt would have ended the same way, so "
                 "the run did not try it again.")
    elif status == "bad_request":
        tried = ("The worker refused the request as it was sent. Another attempt would have "
                 "ended the same way, so the run did not try it again.")
    else:
        tried = ("The run did not try it again at its end: its deadline did not allow a "
                 "second job, or the run stopped before it came to that.")
    after = "Nothing tests it again before the lane's next run."
    if not (v.safe and is_sha(d.commit)):
        return [f"{tried} {after}"]
    return [f"{tried} {after} To test it now (needs write access to the worker repository):",
            "", *dispatch_command(d, v.wf, d.commit)]


def error_block(d: LaneData, v: View, limits: Limits) -> list[str]:
    cell = v.cell
    status = code(cell.get("worker_status") or v.run.get("status"), 60) or "unknown"
    if v.state == "exec":
        out = [f"The workflow did not finish on this commit. Worker status: {status}."]
        prior = v.prior.get("verdict")
        at = f" at {sha_link(v.prior['commit'])}" if v.prior.get("commit") else ""
        here = "in this run" if batch_size(d) else "at this commit"
        started = covers_text(d, "the error started in one of them")
        # Its last result, in the words of its level: nothing was compared for
        # a smoke-level workflow, so its pass is no "pass" of the legend.
        smoke_level = v.level == "smoke"
        result = words(("smoke_pass" if smoke_level else "pass") if prior == "pass"
                       else "smoke_fail" if smoke_level else "changed")
        if v.origin == "new" and is_sha(v.since.get("prev")):
            # A nightly or weekly lane: the error is placed by the workflow's
            # own last result, not by the commits the run covers.
            last = f"Before this commit it last ran to the end{at} ({result})"
            moved = passed_since(v)
            if moved:
                said = f"{moved} The error started after that pass."
            elif v.since.get("n") == 1:
                said = f"{last}, so the error is new at this commit."
            else:
                said = f"{last}: the error started after that commit."
            out += since_lines(d, v, said)
        elif v.origin == "new" and prior in ("pass", "fail"):
            out.append(f"Before this commit it last ran to the end{at} ({result}), so the "
                       f"error is new {here}.{started}")
        elif v.origin == "new":
            out.append(f"This is its first failing run, so the error is new {here}.{started}")
        elif v.origin == "inherited":
            out.append(f"It already ended in an execution error{at}, so the error is "
                       "inherited from an earlier commit.")
        elif v.origin == "first":
            out.append(FIRST_RUN)
        else:
            out.append("It has no earlier result, so it is not known whether the error is new "
                       "at this commit.")
        since = failing_since(v.chain, d.entries)
        if since:
            out += ["", since[0].upper() + since[1:] + "."]
    else:
        out = [f"An infrastructure problem stopped this workflow. Worker status: {status}. "
               "This is not a verdict on the commit."]
        # A summary from before attempts were counted is a first attempt.
        attempt = max(run_attempt(d) or 1, 1)
        spent = attempts_spent(d)
        held = cell.get("worker_status")
        if d.retired:
            out.append("The lane is retired, so nothing runs it again.")
        elif lanes_mod.is_long_cadence(d.lane):
            out += untested_on_a_long_lane(d, v)
        elif no_attempt(d):
            # Its attempt number is that of the runs before it: nothing about a
            # retry can be read from it.
            out.append("This run is no attempt of the lane, so it changes nothing for the cron.")
        elif (spent < regression_skip.MAX_ATTEMPTS and isinstance(held, str)
              and held in NO_RETRY_WHY and not cron_would_retry(d)):
            out.append(f"{NO_RETRY_WHY[held]}. Another run right away would end the same way, "
                       "so the cron does not test the commit again. Dispatch **GPU regression** "
                       "with this commit to test it again.")
        elif spent < regression_skip.MAX_ATTEMPTS:
            out.append(f"This was {attempt_text(attempt)}: the cron tests the commit once more "
                       "while it is still the newest commit on master.")
        elif spent > attempt:
            # The cron counts the start marker of an attempt that died before it published.
            out.append(f"This was {attempt_text(attempt)}. Attempt {spent} started after it and "
                       "published no result, so the cron does not retry it. Dispatch "
                       "**GPU regression** with this commit to test it again.")
        else:
            out.append(f"This was {attempt_text(attempt)}: the cron does not retry it. Dispatch "
                       "**GPU regression** with this commit to test it again.")
    return [" ".join(out[:2]), *out[2:], "", *failure_detail(d, v, limits)]


def no_reference_block(d: LaneData, v: View, limits: Limits | None = None) -> list[str]:
    out = ["No reference is approved for this workflow on this lane yet, so the output was not "
           "compared with anything."]
    text = v.kind == "text"
    shown = text_lines(d, v, limits or Limits(), diff=False) if text else []
    shown = ["", *shown] if shown else []
    if d.retired or not (v.safe and LANE_RE.fullmatch(d.lane_id)):
        return out + shown
    base = (f"gh workflow run golden-baseline.yml --repo {worker_repo()} -f ref=RELEASE_TAG "
            f"-f workflows={v.wf} -f lane={d.lane_id}")
    return out + [f"To make one, {'generate' if text else 'render'} it from a release tag, look "
                  "at the output, then approve "
                  "it with the second command:", "", "```", f"{base} -f bless=false", "```", "",
                  "```", f"{base} -f bless_only=true", "```"] + shown


# -- smoke level ------------------------------------------------------------------------
# A smoke run publishes what it checked: `smoke` in the workflow's cell, with
# the output files, what was measured on each and the problems found. File
# names, sizes and problem sentences come from the worker's files, so they are
# escaped like every other name, and links are built from checked parts only.
SMOKE_TYPES = tuple(dict.fromkeys(smoke.TYPES.values()))
NOT_SENT = "not sent, checked by size only"
PUBLISH_MAX = f"{smoke.PUBLISH_MAX_BYTES // (1024 * 1024)} MiB"
HELD_BACK = f"not published: over {PUBLISH_MAX}, and the run before did not write the same file"


def smoke_files(v: View) -> list[dict]:
    """The output files a smoke run listed, each with what was measured on it."""
    return [f for f in as_list(as_dict(v.cell.get("smoke")).get("files"))
            if isinstance(f, dict) and isinstance(f.get("name"), str)]


def file_type(f: dict) -> str | None:
    """The type a file was checked as; None for a file that no check knows."""
    kind = f.get("type")
    return kind if isinstance(kind, str) and kind in SMOKE_TYPES else None


def counted(n, word: str, many: str | None = None) -> str:
    """'1 frame', '35,781 vertices'; '' when `n` is not a count."""
    n = as_int(n)
    if n is None or n < 0:
        return ""
    return f"{n:,} {word if n == 1 else many or word + 's'}"


def pixels(f: dict) -> str:
    width, height = as_int(f.get("width")), as_int(f.get("height"))
    return f"{width}x{height}" if width is not None and height is not None else ""


def smoke_facts(f: dict) -> str:
    """What a smoke check measured on one output file, in the words of the
    problem sentences that smoke.py writes."""
    kind = file_type(f)
    if kind is None:
        return ("not sent, and " if f.get("sent") is False else "") + (
            "not checked: no check for this file type")
    if f.get("sent") is False:
        return NOT_SENT
    rate, seconds = as_int(f.get("sample_rate")), fmt(f.get("seconds"), ".1f", " s")
    if kind == "image":
        flat = f.get("flat_share")
        bits = [pixels(f), fmt(f.get("std"), ".2f") and f"brightness spread {f['std']:.2f}",
                is_num(flat) and f"{flat * 100:.1f} % of the pixels have one brightness"]
    elif kind == "text":
        bits = [counted(f.get("chars"), "non-blank character"), counted(f.get("words"), "word")]
    elif kind == "audio":
        bits = [seconds, rate is not None and f"{rate} Hz", counted(f.get("channels"), "channel"),
                counted(f.get("bytes_per_second"), "byte") and f"{f['bytes_per_second']:,} bytes/s"]
    elif kind == "mesh":
        box = [side for side in as_list(f.get("extent")) if is_num(side)]
        bits = [counted(f.get("vertices"), "vertex", "vertices"), counted(f.get("faces"), "face"),
                len(box) == 3 and "box " + " x ".join(f"{side:.3g}" for side in box)]
    else:
        bits = [pixels(f), counted(f.get("frames"), "frame"), seconds,
                counted(f.get("bytes_per_frame_mp"), "byte")
                and f"{f['bytes_per_frame_mp']:,} bytes a frame for each megapixel after the "
                    "first frame"]
    return " · ".join(b for b in bits if b)


def brief_facts(f: dict) -> str:
    """The one or two numbers that describe a checked file in a table row."""
    kind = file_type(f)
    if kind == "image":
        return pixels(f)
    if kind == "text":
        # The same words as in the file list: the heading of a shown text
        # counts every character, blanks too, and gives another number.
        return counted(f.get("chars"), "non-blank character")
    if kind == "audio":
        rate, channels = as_int(f.get("sample_rate")), as_int(f.get("channels"))
        return " · ".join(b for b in (
            fmt(f.get("seconds"), ".1f", " s"), rate is not None and f"{rate} Hz",
            {1: "mono", 2: "stereo"}.get(channels) or counted(channels, "channel")) if b)
    if kind == "mesh":
        return counted(f.get("vertices"), "vertex", "vertices")
    return " · ".join(b for b in (pixels(f), counted(f.get("frames"), "frame")) if b)


def checked_cell(v: View) -> str:
    """The Checked cell of a workflow's row: what its smoke checks looked at,
    the files by type with the numbers that describe them. Empty for a
    workflow at reference level, which is compared and not checked this way."""
    if v.level != "smoke":
        return ""
    files = smoke_files(v)
    if not files:
        return "no file" if v.state == "smoke_fail" else ""
    groups: dict[str | None, list[dict]] = {}
    unsent = 0
    for f in files:
        if file_type(f) and f.get("sent") is False:
            unsent += 1
        else:
            groups.setdefault(file_type(f), []).append(f)
    many = {"text": "text files", "audio": "audio files", "mesh": "meshes", "video": "videos"}
    parts = []
    for kind in SMOKE_TYPES:
        group = groups.get(kind) or []
        if not group:
            continue
        facts = {brief_facts(f) for f in group}
        facts = facts.pop() if len(facts) == 1 else ""
        if kind == "image":
            name = plural(len(group), "image")
        elif len(group) > 1:
            name = f"{len(group)} {many[kind]}"
        else:
            # An audio workflow's one file needs no name: its numbers say what it is.
            name = "" if kind == "audio" and set(groups) == {"audio"} and not unsent else kind
        parts.append(f"{name} {facts}".strip())
    if groups.get(None):
        parts.append(f"{plural(len(groups[None]), 'other file')}, not checked")
    if unsent:
        parts.append(f"{plural(unsent, 'file')} {NOT_SENT}")
    return " · ".join(parts)


def held_back(v: View) -> set[str]:
    """The names of the output files of a smoke run that are not on the
    results branch: the worker did not send them, or the run kept them off it.
    A file over 1 MiB is published only when the run before wrote the same
    file, so an output that differs from run to run does not fill the branch."""
    return {f["name"] for f in smoke_files(v)
            if f.get("published") is False or f.get("sent") is False}


def published_url(d: LaneData, v: View, *name) -> str | None:
    """The link to an output file of this run on the results branch."""
    if not (d.files_ok and v.safe):
        return None
    return raw_url(d.reader, join_path(d.run_dir, v.wf, "outputs", *name))


def attached_url(d: LaneData) -> str | None:
    """Where the files are that a run kept off the results branch: its
    artifact, or the page of the run that tested the commit, which lists it."""
    return d.artifact_url or tested_run_url(d)


def attached_link(d: LaneData) -> str:
    """The words that point to those files; "" when the report knows no run.
    Only the run that just finished knows its artifact. A report made later
    points to the run's page."""
    if d.artifact_url:
        return f"in the [attached files]({d.artifact_url})"
    url = tested_run_url(d)
    return f"attached to [the run that tested it]({url})" if url else ""


def smoke_output_cell(d: LaneData, v: View) -> str:
    """The Output cell of a smoke row: the thumbnail when the run wrote a PNG,
    else a link named after what the output is. An output that is not on the
    results branch points to the files attached to the run."""
    held = held_back(v)
    names = sorted(f["name"] for f in smoke_files(v))
    thumb = v.cell.get("thumbnail")
    if isinstance(thumb, str) and thumb.endswith(".webp"):
        thumb_url = published_url(d, v, "thumbs", thumb)
        png = thumb[:-len(".webp")] + ".png"
        if thumb_url:
            url = attached_url(d) if png in held else published_url(d, v, png)
            img = (f'<img src="{attr(thumb_url)}" width="{THUMB_PX}" '
                   f'alt="{attr(v.wf)} output">')
            return f'<a href="{attr(url)}">{img}</a>' if url else img
    if v.kind == "text":
        name = as_dict(d.texts.get(v.wf)).get("name")
    elif v.kind == "audio":
        flacs = [n for n in names if n.lower().endswith(".flac")]
        name = next((n for n in flacs if n.startswith(v.wf)), flacs[0] if flacs else None)
    else:
        name = next((n for n in names if n.endswith(".png")), None)
    word = v.kind if v.kind in ("text", "audio") else "PNG"
    if not isinstance(name, str):
        return ""
    if any(f["name"] == name and f.get("sent") is False for f in smoke_files(v)):
        # The worker did not send it, so the run has no copy: not on the
        # results branch, and not in the files attached to the run either.
        return f"{word}, not sent"
    if name in held:
        return f"{word}, {attached_link(d) or 'not published'}"
    url = published_url(d, v, name)
    return f"[{word}]({url})" if url else ""


def smoke_file_note(d: LaneData, o: dict, listed: dict) -> tuple[bool, str]:
    """(whether an output file of a smoke run is on the results branch, the
    note of its row in the file list): why it is not published, and what was
    measured on it. `listed` is the run's `smoke.files` by name. The run says
    what it published for those alone, at most 50: a larger file past them is
    not linked."""
    name, size = o.get("filename"), o.get("bytes")
    f = listed.get(name) if isinstance(name, str) else None
    if o.get("truncated"):
        return False, "not published: over the worker's size limit" + (
            ", checked by size only" if f is not None and file_type(f) else "")
    notes = []
    held = f is not None and f.get("published") is False
    unknown = f is None and not (is_num(size) and size <= smoke.PUBLISH_MAX_BYTES)
    if held:
        where = attached_link(d)
        notes.append(HELD_BACK + (f" ({where})" if where else ""))
    elif unknown:
        notes.append(f"not linked: past the {smoke.FILES_MAX} files the run says it published")
    if f is not None:
        notes.append(smoke_facts(f))
    return not (held or unknown), "; ".join(n for n in notes if n)


def smoke_files_table(d: LaneData, v: View, limits: Limits) -> list[str]:
    """The files of a smoke run with what was measured on each."""
    rows, held = [], held_back(v)
    files = smoke_files(v)
    for f in files[:limits.files_per_workflow]:
        name = md(f["name"], 200)
        url = None if f["name"] in held else published_url(d, v, f["name"])
        if f.get("published") is False and f.get("sent") is not False:
            where = attached_link(d)
            name += ", not published" + (f" ({where})" if where else "")
        rows.append([f"[{name}]({url})" if url else name, file_type(f) or "other",
                     size_text(f.get("bytes")), smoke_facts(f)])
    if not rows:
        return []
    out = table(["File", "Type", "Size", "Measured"], rows)
    if len(files) > len(rows):
        # Only a shortened report cuts this list, and it cuts the list under
        # Output files to the same length: the record holds them all.
        out += ["", f"{len(files) - len(rows)} more files are listed in run.json."]
    if any(f.get("published") is False and f.get("sent") is not False for f in files):
        out += ["", f"A file over {PUBLISH_MAX} is published only when the run before wrote "
                "the same file. Until then it is in the files attached to the run, which "
                "GitHub keeps 90 days."]
    return out


def origin_words(v: View, batch: bool = False) -> str:
    """Where a smoke failure started, for its heading: from the workflow's
    own last result, not from the run before."""
    if v.origin == "first":
        return "first run"
    if v.origin == "inherited":
        return "inherited"
    prev, n = v.since.get("prev"), v.since.get("n")
    if not is_sha(prev):
        return "new in this run" if batch else "new at this commit"
    if v.since.get("later"):
        # The run's own count reaches back past a later pass: it is not said.
        return f"new since {tick(v.since['later'][:7])}"
    if n == 1:
        return "new at this commit"
    return f"new since {tick(prev[:7])}" + (f", {n} commits" if is_batch(n) else "")


def dispatch_command(d: LaneData, workflows: str, commit: str = "<sha>") -> list[str]:
    """The command that tests workflows of a nightly or weekly lane at one
    commit, as a code block. The caller has checked the lane and the ids."""
    return ["```",
            f"gh workflow run nightly-weekly.yml --repo {worker_repo()} "
            f"-f cadence={d.lane['cadence']} -f commit={commit} -f workflows={workflows}", "```"]


def passed_since(v: View) -> str:
    """What a failure's own block says of its last pass when a later run has
    passed between that pass and this commit (`later` in `v.since`), '' when
    none has. The open failure names the later pass, so the block must not
    name the older one as the last."""
    prev, later = v.since.get("prev"), v.since.get("later")
    if not later:
        return ""
    return (f"When this run was made it had last passed at {sha_link(prev)}. A later run of "
            f"this workflow passed at {sha_link(later)}, which lies between that commit and "
            "this one.")


def since_lines(d: LaneData, v: View, text: str) -> list[str]:
    """`text`, which says where the workflow's last result stands, then the
    commits since that result (`v.since`, its own range) and, on a nightly or
    weekly lane, how to test one of them. Nothing more for a range of one
    commit: the tested one. After a later pass (`passed_since`) the commits
    are those of the open chain, from that pass on, and they are not counted:
    the run's own count reaches back past it."""
    prev, n, later = v.since.get("prev"), v.since.get("n"), v.since.get("later")
    if later:
        cmp_ = compare_link(later, v.chain.get("commit")) or compare_link(later, d.commit)
        text += f" The commits since that pass: {cmp_}." if cmp_ else ""
    elif n == 1:
        return [text]
    else:
        cmp_ = compare_link(prev, d.commit)
        if cmp_:
            text += (f" {n} commits since then: {cmp_}." if is_batch(n)
                     else f" The commits since then: {cmp_}.")
    if not (lanes_mod.is_long_cadence(d.lane) and v.safe):
        return [text]
    return [text + " To test one commit of that range (needs write access to the worker "
            "repository):", "", *dispatch_command(d, v.wf)]


def smoke_origin_lines(d: LaneData, v: View) -> list[str]:
    """Where a smoke failure started and, for a new one that covers several
    commits, how to find the commit."""
    if v.origin == "first":
        return [FIRST_RUN]
    before = v.prior.get("commit")
    if v.origin == "inherited":
        out = ["It already failed" + (f" at {sha_link(before)}" if before else "")
               + ": the failure is inherited from an earlier run."]
        since = failing_since(v.chain, d.entries)
        return out + (["", since[0].upper() + since[1:] + "."] if since else [])
    prev = v.since.get("prev")
    if not is_sha(prev):
        return []
    if v.prior.get("verdict") == "pass":
        text = passed_since(v) or f"It last passed at {sha_link(prev)}."
    else:
        text = (f"Its last result before this one, at {sha_link(prev)}, was an execution "
                "error: it finishes now, and its output does not pass.")
    return since_lines(d, v, text)


def smoke_block(d: LaneData, v: View, limits: Limits | None = None) -> list[str]:
    """Why a smoke-level workflow failed: each problem its checks found, its
    files with what was measured, and where the failure started. A failed
    text workflow shows its text here. No command approves anything: a
    smoke-level workflow has no reference."""
    limits = limits or Limits()
    found = as_dict(v.cell.get("smoke"))
    out = ["The workflow finished, and its output did not pass the smoke checks.", ""]
    problems = [p for p in as_list(found.get("problems")) if isinstance(p, dict)]
    for p in problems[:smoke.PROBLEMS_MAX]:
        name = code(p.get("file"), 200) if isinstance(p.get("file"), str) else ""
        text = md(p.get("text"), 400) or "no detail recorded"
        out.append(f"- {name}: {text}" if name else f"- {text}")
    if not problems:
        out.append("- The run recorded no problem for this failure.")
    out += [f"- Note: {md(note, 400)}" for note in as_list(found.get("notes"))[:smoke.PROBLEMS_MAX]
            if isinstance(note, str) and note.strip()]
    for part in (smoke_files_table(d, v, limits),
                 text_lines(d, v, limits, diff=False) if v.kind == "text" else [],
                 smoke_origin_lines(d, v)):
        if part:
            out += ["", *part]
    return out


def not_reached(v: View) -> bool:
    """A workflow that a run's deadline did not reach: no job was submitted
    (run_regression.NOT_REACHED)."""
    return v.state == "infra" and v.cell.get("worker_status") == "not_reached"


def not_reached_block(d: LaneData, views: list[View]) -> list[str]:
    """The workflows the run's deadline did not reach, as one list: each has
    the same story, and a weekly run can leave dozens."""
    out = ["", f"#### {WARN} Not reached before the run's deadline ({len(views)})", "",
           "The run stops submitting jobs when one could still be running at its deadline. "
           "No job was submitted for these workflows, so they have no result. This is not a "
           "verdict on the commit.", "", ", ".join(v.name for v in views)]
    after = ("Nothing tests them again before the lane's next run."
             if lanes_mod.is_long_cadence(d.lane) else "")
    if not (lanes_mod.is_long_cadence(d.lane) and is_sha(d.commit)
            and all(v.safe for v in views)):
        return out + (["", after] if after else [])
    return out + ["", f"{after} To test them now (needs write access to the worker "
                  "repository):", "",
                  *dispatch_command(d, ",".join(v.wf for v in views), d.commit)]


def attention_block(d: LaneData, limits: Limits) -> list[str]:
    todo = sorted((v for v in d.views if v.state not in PASSED),
                  key=lambda v: (ATTENTION_ORDER.index(v.state), v.wf))
    batch = bool(batch_size(d))
    long = lanes_mod.is_long_cadence(d.lane)
    late = [v for v in todo if not_reached(v)]
    open_ = [v for v in todo if v.state != "accepted" and not not_reached(v)]
    out: list[str] = []
    # An accepted change is closed: it is shown, but not as something to act on.
    for title, group in (("What needs attention", open_),
                         ("Accepted after the run", [v for v in todo if v.state == "accepted"])):
        if group or (late and group is open_):
            out += ([""] if out else []) + [f"### {title}"]
        for v in group:
            # A failure that is placed by the workflow's own last result says so
            # in its heading: a smoke failure, and on a nightly or weekly lane an
            # execution error.
            placed = v.state == "smoke_fail" or (long and v.state == "exec" and v.origin)
            where = f" ({origin_words(v, batch)})" if placed else ""
            out += ["", f"#### {icon(v.state)} {v.name}: {words(v.state, batch)}{where}", ""]
            if v.state in DRIFT:
                out += drift_block(d, v, limits)
            elif v.state == "smoke_fail":
                out += smoke_block(d, v, limits)
            elif v.state in ("exec", "infra"):
                out += error_block(d, v, limits)
            elif v.state == "no_ref":
                out += no_reference_block(d, v, limits)
            else:
                status = code(v.cell.get("worker_status"), 60)
                out.append("The run recorded no verdict for this workflow."
                           + (f" Worker status: {status}." if status else ""))
                out += ["", *failure_detail(d, v, limits)]
        if late and group is open_:
            out += not_reached_block(d, late)
    return out


def passed_with_a_difference(v: View) -> bool:
    """A pass that is listed with its numbers and limits. For an image: any
    difference, from the reference or from the previous run. For a text: a
    difference from the reference (within a similarity limit). A text that
    equals its reference is shown as that, whatever the run before produced:
    a text is not compared bit by bit and that pass crossed no limit."""
    if v.state != "pass":
        return False
    if v.kind != "text":
        return not_identical(v.cell)
    vg = as_dict(v.cell.get("vs_golden"))
    return bool(vg) and not vg.get("error") and not vg.get("identical")


def within_limits_block(d: LaneData, limits: Limits | None = None) -> list[str]:
    """Workflows that passed without being bit-identical: each number next to
    its limit, the noise floor and the figures the run rendered (for a text
    workflow: the texts and their difference)."""
    out: list[str] = []
    for v in d.views:
        if not passed_with_a_difference(v):
            continue
        body = comparison_table(d, v)
        figures = (text_lines(d, v, limits or Limits()) if v.kind == "text"
                   else figure_lines(d, v))
        name = f"<code>{v.wf}</code>" if v.safe else v.name  # no Markdown inside <summary>
        out += ["", *details(f"{name}: numbers and limits",
                             [*body, "", *figures] if figures else body)]
    if not out:
        return []
    return ["### Passed without being bit-identical", "",
            "These outputs differ from their reference or from the previous tested commit, "
            "within the limits.", *out]


def text_outputs_block(d: LaneData, limits: Limits) -> list[str]:
    """The texts that need no attention: of a text workflow that matches its
    reference, and of one at smoke level that passed its checks. A text that
    changed, or that failed its checks, is shown where that is reported."""
    out: list[str] = []
    for v in d.views:
        if v.kind != "text" or v.state not in PASSED or passed_with_a_difference(v):
            continue
        shown = text_lines(d, v, limits, diff=False)
        how = ("smoke level, not compared" if v.state == "smoke_pass"
               else "identical to its reference")
        if shown:
            out += ["", f"{v.name}, {how}:", "", *shown]
    return ["### Generated text", *out] if out else []


def output_cell(d: LaneData, v: View) -> str:
    """The thumbnail, linking to the full PNG (a text workflow: a link to its text)."""
    if not (d.files_ok and v.safe) or v.state in ("exec", "infra"):
        return ""
    if v.level == "smoke":
        return smoke_output_cell(d, v)
    if v.kind == "text":
        url = as_dict(d.texts.get(v.wf)).get("cand_url")
        return f"[text]({url})" if url else ""
    # A PNG past the worker's size limit is listed but was never published.
    png = first_png({"outputs": [o for o in as_list(v.run.get("outputs"))
                                 if isinstance(o, dict) and not o.get("truncated")]})
    thumb = v.cell.get("thumbnail")
    name = png.get("filename")
    if not name and isinstance(thumb, str) and thumb.endswith(".webp"):
        name = thumb[:-len(".webp")] + ".png"
    png_url = raw_url(d.reader, join_path(d.run_dir, v.wf, "outputs", name)) if name else None
    thumb_url = (raw_url(d.reader, join_path(d.run_dir, v.wf, "outputs", "thumbs", thumb))
                 if isinstance(thumb, str) else None)
    if not png_url:
        return ""
    if not thumb_url:
        # Runs older than the thumbnails: the PNG itself, unless it is very large.
        size = png.get("bytes")
        if not is_num(size) or size > INLINE_PNG_MAX_BYTES:
            return f"[PNG]({png_url})"
        thumb_url = png_url
    return (f'<a href="{attr(png_url)}"><img src="{attr(thumb_url)}" width="{THUMB_PX}" '
            f'alt="{attr(v.wf)} output"></a>')


def previous_cell(v: View) -> str:
    vs = as_dict(v.cell.get("vs_previous"))
    prev = v.cell.get("previous_commit")
    link = f" {sha_link(prev)}" if prev else ""
    if vs.get("error"):
        return "comparison failed"
    if not vs:
        return ""
    if vs.get("identical"):
        return f"identical to{link}" if link else "identical"
    return f"changed from{link}".strip() + (f" (MSE {vs['mean_mse']:.4f})"
                                            if is_num(vs.get("mean_mse")) else "")


def reference_cell(v: View) -> str:
    vg = as_dict(v.cell.get("vs_golden"))
    if v.level == "smoke":
        return ""  # checked, not compared: the Checked cell says what was looked at
    if vg.get("error"):
        return "comparison failed"
    if not vg:
        return ""
    if output_kinds.metrics_kind(vg) == "text":
        if vg.get("identical"):
            return "identical text"
        if vg.get("files_compared") == 0:  # no pair of files, so no similarity to show
            return "text file under another name"
        line = as_dict(vg.get("first_diff")).get("line")
        bits = [is_num(vg.get("similarity"))
                and f"{output_kinds.percent(vg['similarity'])} % similar"
                    f"{output_kinds.basis_note(vg)}",
                is_num(line) and f"differs from line {line:.0f}"]
        return " · ".join(b for b in bits if b) or "differs"
    if vg.get("identical"):
        return "bit-identical"
    bits = [fmt(vg.get("mean_mse"), ".4f") and f"MSE {vg['mean_mse']:.4f}",
            fmt(vg.get("mean_psnr_db"), ".2f") and f"PSNR {vg['mean_psnr_db']:.2f} dB",
            fmt(vg.get("mean_pct_pixels_changed"), ".3f")
            and f"{vg['mean_pct_pixels_changed']:.3f} % px"]
    return " · ".join(b for b in bits if b)


def usual_exec(d: LaneData, wf: str) -> tuple[float, float] | None:
    """(low, high) seconds a workflow usually takes on the lane: the 10th and
    90th percentile over its newest USUAL_RUNS passing runs, this commit left
    out. None until USUAL_MIN_RUNS of them exist.

    Workers are different machines and the models load from shared storage, so
    the same code takes up to a third longer from one run to the next. A time
    is read against this range, not against the previous run."""
    times = []
    for e in d.entries:
        cell = as_dict(as_dict(e.get("w")).get(wf))
        if e.get("c") != d.commit and cell.get("v") == "pass" and is_num(cell.get("exec")):
            times.append(cell["exec"])
            if len(times) == USUAL_RUNS:
                break
    if len(times) < USUAL_MIN_RUNS:
        return None
    times.sort()
    return tuple(times[round(p * (len(times) - 1))] for p in (0.1, 0.9))


def exec_cell(d: LaneData, v: View) -> str:
    """Exec seconds next to the workflow's usual range; a time well above the
    range is marked slow. Timing is never part of the verdict."""
    secs = as_dict(v.cell.get("timings")).get("prompt_exec_s")
    if not is_num(secs):
        return ""
    usual = usual_exec(d, v.wf)
    if not usual:
        return f"{secs:.1f}"
    low, high = usual
    slow = f" {WARN} slow" if secs > high * SLOW_FACTOR else ""
    spec = ".0f" if high >= 10 else ".1f"  # whole seconds say nothing about a 0.5 s workflow
    return f"{secs:.1f} (usual {low:{spec}} to {high:{spec}}){slow}"


def start_cell(timings: dict) -> str:
    """Server start seconds, with the checkout seconds in brackets."""
    start, checkout = (fmt(timings.get(k), ".1f") for k in ("server_start_s", "checkout_s"))
    return " ".join(x for x in (start, f"({checkout})" if checkout else "") if x)


def workflows_block(d: LaneData, limits: Limits) -> list[str]:
    rows = []
    batch = bool(batch_size(d))
    # A smoke row is checked, not compared: it fills Checked and none of the
    # three reference columns. A lane of smoke rows alone drops those three,
    # and a lane without a smoke row has no Checked column.
    checked = has_smoke(d.views)
    compared = not checked or any(v.level != "smoke" for v in d.views)
    # A row whose workflow a later run tested again says so: its result is
    # that run's, not the one of the run the facts table names.
    again = retests_of(d.summary)
    for v in d.views:
        cell, timings = v.cell, as_dict(v.cell.get("timings"))
        tag = cell.get("golden_tag")
        sha = v.sha[:12] if v.sha and HEX64_RE.fullmatch(v.sha) else None
        against = [reference_cell(v), previous_cell(v),
                   tick(tag) if isinstance(tag, str) and SEGMENT_RE.fullmatch(tag)
                   else code(tag, 60)] if compared else []
        result = label(v.state, batch)
        if v.wf in again:
            url = retest_url(again[v.wf])
            result += " · tested again in " + (f"[a later run]({url})" if url else "a later run")
        rows.append([v.name, result, output_cell(d, v), *against,
                     *([checked_cell(v)] if checked else []), exec_cell(d, v),
                     start_cell(timings),
                     fmt(cell.get("vram_peak_mb") / 1024, ".1f")
                     if is_num(cell.get("vram_peak_mb")) else "",
                     fmt(cell.get("rss_peak_mb") / 1024, ".1f")
                     if is_num(cell.get("rss_peak_mb")) else "",
                     tick(sha) if sha else ""])
    # The thumbnail comes early: the table is wider than a narrow page, and the
    # columns that scroll out of view should be the ones needed least.
    out = [f"### All workflows ({len(rows)})", "",
           *table(["Workflow", "Result", "Output",
                   *(["vs reference", "vs previous", "Reference"] if compared else []),
                   *(["Checked"] if checked else []),
                   "Exec s (usual range)", "Server start s (checkout s)", "Peak VRAM GB",
                   "Peak RSS GB", "Output sha256"], rows),
           "", EXEC_NOTE]

    file_rows, total = [], 0
    for v in d.views:
        outputs = [o for o in as_list(v.run.get("outputs")) if isinstance(o, dict)]
        total += len(outputs)
        listed = {f["name"]: f for f in smoke_files(v)}
        for o in outputs[:limits.files_per_workflow]:
            name = o.get("filename")
            path = join_path(d.run_dir, v.wf, "outputs", name) if d.files_ok and v.safe else None
            there = not o.get("truncated")
            note = "" if there else "not published: over the worker's size limit"
            if v.level == "smoke":
                there, note = smoke_file_note(d, o, listed)
            url = raw_url(d.reader, path) if path and there else None
            sha = o.get("sha256")
            file_rows.append([
                v.name, f"[{md(name, 200)}]({url})" if url else md(name, 200),
                size_text(o.get("bytes")),
                tick(sha) if isinstance(sha, str) and HEX64_RE.fullmatch(sha) else code(sha, 80),
                note])
        if len(outputs) > limits.files_per_workflow > 0:
            file_rows.append([v.name, f"+{len(outputs) - limits.files_per_workflow} more files, "
                              "listed in run.json", "", "", ""])
    if file_rows:
        out += ["", *details(f"Output files ({total})",
                             table(["Workflow", "File", "Size", "sha256", "Note"], file_rows))]
    records = [f"[{v.name}]({raw_url(d.reader, f'{d.run_dir}/{v.wf}/run.json')})"
               for v in d.views if d.files_ok and v.safe and v.run]
    if records and limits.files_per_workflow:  # cut together with the file lists
        out += ["", "Run records (run.json): " + ", ".join(records) + "."]
    return out


def schema_block(d: LaneData) -> list[str]:
    names = (("missing_nodes", "node classes this commit does not have"),
             ("stripped_inputs", "inputs not in this commit's schema, removed"),
             ("filled_defaults", "missing inputs filled from the schema's defaults"))
    out = []
    for v in d.views:
        validation = as_dict(v.run.get("validation"))
        for key, what in names:
            items = as_list(validation.get(key))
            if items:
                more = f", +{len(items) - 20} more" if len(items) > 20 else ""
                out.append(f"- {v.name}: {what}: "
                           + ", ".join(code(i, 200) for i in items[:20]) + more)
    if not out:
        return []
    return ["### Schema notes", "", "The worker checks each workflow against the node schema of "
            "the commit under test and records what it had to change:", "", *out]


def packs_block(d: LaneData) -> list[str]:
    registry = as_dict(d.packs.get("packs"))
    rows = []
    for v in d.views:
        for pack, info in as_dict(v.cell.get("custom_nodes") or v.run.get("custom_nodes")).items():
            info = as_dict(info)
            known = as_dict(registry.get(pack))
            sha, repo = info.get("sha"), known.get("repo")
            if is_sha(sha) and isinstance(repo, str) and PACK_REPO_RE.fullmatch(repo):
                pin = f"[{tick(sha[:7])}]({repo}/tree/{sha})"
            else:
                pin = tick(sha[:7]) if is_sha(sha) else code(sha, 40)
            status = md(info.get("import"), 30) or "unknown"
            name = code(pack, 80)
            if known.get("label") and known.get("sha") == sha:
                name += f" ({md(known['label'], 60)})"
            rows.append([v.name, name, status if status == "ok" else f"**{status}**", pin,
                         md(info.get("source"), 30), fmt(info.get("install_s"), "g"),
                         fmt(info.get("pip_s"), "g"), fmt(info.get("seconds"), "g")])
    if not rows:
        return []
    return ["### Custom-node packs", "",
            *table(["Workflow", "Pack", "Import", "Pinned commit", "Source", "Install s",
                    "Pip s", "Import s"], rows)]


def not_run_block(d: LaneData) -> list[str]:
    ran = {v.wf for v in d.views}
    elsewhere: dict[str, list[str]] = {}  # the lanes a workflow is restricted to -> workflows
    left_out = []
    for wf, entry in as_dict(d.manifest.get("workflows")).items():
        entry = as_dict(entry)
        if not entry.get("enabled") or wf in ran:
            continue
        # A workflow of another cadence belongs to other lanes. It is not
        # missing here, so it is not listed.
        if other_cadence(entry, d.lane):
            continue
        only = entry.get("lanes")
        if isinstance(only, list) and d.lane_id not in only:
            lanes = ", ".join(tick(x) if isinstance(x, str) and LANE_RE.fullmatch(x)
                              else code(x, 80) for x in only[:8])
            elsewhere.setdefault(lanes, []).append(workflow_name(wf))
        else:
            left_out.append(workflow_name(wf))
    out = [f"- {label('not_run')}, the manifest runs them only on {lanes}: {', '.join(names)}"
           for lanes, names in elsewhere.items()]
    if left_out and progress_text(d.summary):
        out.append(f"- {label('not_run')}, on this lane, and the run had no result for them "
                   f"when it pushed this summary: {', '.join(left_out)}")
    elif left_out and stopped_early(d):
        out.append(f"- {label('not_run')}, on this lane, and the run left no result for them: "
                   "it stopped before it published (a dispatch can also name only some "
                   f"workflows): {', '.join(left_out)}")
    elif left_out and d.manifest_is_snapshot:
        out.append(f"- {label('not_run')}, on this lane but not part of this run (a dispatch "
                   f"can name only some workflows): {', '.join(left_out)}")
    elif left_out:
        # Today's manifest, not the run's: the run may simply be older than a workflow.
        out.append(f"- {label('not_run')}, on this lane in today's manifest but without a "
                   "result in this run (a dispatch can name only some workflows, and a run can "
                   f"be older than a workflow): {', '.join(left_out)}")
    return ["### Workflows not run on this lane", "", *out] if out else []


def history_block(d: LaneData, limits: Limits) -> list[str]:
    if not d.head:
        return []
    out = ["### Recent commits on this lane", ""]
    failures = open_failures(d.head, d.entries)
    out += (["Open failures on this lane:", "", *failures] if failures
            else ["Open failures on this lane: none."])
    shown = d.entries[:limits.history_rows]
    if not shown:
        return out
    # On a nightly or weekly lane a run that names one workflow (a step of a
    # search for a commit) has a row like any other. Its result is of that
    # workflow alone, so the row says how many of the lane's workflows it has.
    # Only the row of such a run (`named` in its index entry): a full run
    # from before a workflow joined the lane lacks one too, and is no step.
    runs = (lane_workflows(d.manifest, d.lane_id, d.lane)
            if lanes_mod.is_long_cadence(d.lane) else None)
    rows = []
    for e in shown:
        m = as_dict(e.get("m"))
        states = entry_states(e, d.lane_references)
        n = as_int(as_dict(e.get("range")).get("n"))
        commit = sha_link(e["c"])
        if is_batch(n):
            commit += f" +{n} commits"
        if e["c"] == d.commit:
            commit += " **this report**"
        when = utc(e.get("t")).replace(" UTC", "")
        url = entry_run_url(e)
        result = rollup(states)
        if runs and e.get("named") is True:
            result += f" · {len(runs & set(states))} of {plural(len(runs), 'workflow')}"
        rows.append([commit, md(m.get("s"), 90), pr_link(m.get("pr")), md(m.get("a"), 40),
                     f"[{when}]({url})" if url and when else when, result,
                     not_passing(states, batch=is_batch(n))])
    out += ["", *table(["Commit", "Subject", "PR", "Author", "Tested (UTC)", "Result",
                        "Not passing"], rows)]
    count = d.head.get("count")
    note = f"The newest {len(shown)} of {count:g} runs on this lane." if is_num(count) \
        else f"The newest {len(shown)} runs on this lane."
    if not any(e["c"] == d.commit for e in shown):
        note += " The commit of this report is not among them."
    return out + ["", note]


def dashboard_line(d: LaneData | None = None) -> list[str]:
    """The footer link to the dashboard: its page for this commit on this lane
    when the report is about one, else its front page."""
    url = dashboard_url()
    if not url:
        return []
    line = f"Longer history and charts: [{md(url.split('//', 1)[-1], 200)}]({url})"
    if (d and d.published and is_branch(d.branch) and is_sha(d.commit)
            and LANE_RE.fullmatch(d.lane_id)):
        page = f"{url}/regression/{quote(d.branch, safe='')}/{d.commit}?lane={d.lane_id}"
        line += f" · [this commit on this lane]({page})"
    return [line]


def footer_block(d: LaneData, limits: Limits) -> list[str]:
    out = ["---", ""]
    links = []
    if d.files_ok:
        links.append(f"[summary.json]({raw_url(d.reader, f'{d.run_dir}/summary.json')})")
        links.append(f"[every file of this run]({tree_url(d.reader, d.run_dir)})")
    elif d.run_dir and d.reader.sha and d.marker:
        links.append(f"[the run's directory]({tree_url(d.reader, d.run_dir)})")
    if d.artifact_url:
        links.append(f"[the files attached to this run]({d.artifact_url}) "
                     "(a zip, kept 90 days, needs a GitHub login)")
    if links:
        pinned = (f" The results links are pinned to results commit {tick(d.reader.sha[:8])}."
                  if d.reader.sha and (d.files_ok or d.marker) else "")
        out += ["Files: " + ", ".join(links) + "." + pinned, ""]
    for extra in ([legend(smoke_on_page(d, limits))] if d.summary else [],
                  shortened_note(limits), dashboard_line(d)):
        if extra:
            out += [*extra, ""]
    return out[:-1]


def render_lane(d: LaneData, limits: Limits) -> str:
    if not d.summary:
        blocks = [title_block(d), no_summary_block(d), footer_block(d, limits)]
    else:
        blocks = [title_block(d, limits), facts_block(d), attention_block(d, limits),
                  workflows_block(d, limits), within_limits_block(d, limits),
                  text_outputs_block(d, limits), schema_block(d),
                  packs_block(d), not_run_block(d), history_block(d, limits),
                  footer_block(d, limits)]
    return "\n\n".join("\n".join(b) for b in blocks if b) + "\n"


def lane_report(reader: Reader, lanes: dict, commit: str, branch: str, lane_id: str,
                **kw) -> str | None:
    """The Markdown for one commit on one lane, within the size limit."""
    d = gather_lane(reader, lanes, commit, branch, lane_id, **kw)
    return fit(lambda limits: render_lane(d, limits))


# -- every lane -----------------------------------------------------------------------
@dataclass
class LaneColumn:
    lane_id: str
    lane: dict
    summary: dict | None
    head: dict
    entries: list[dict]  # the head's entries, newest first
    references: dict

    @property
    def active(self) -> bool:
        """Tested on every commit: per-commit cadence and an endpoint to run on."""
        return self.lane.get("cadence") == "per-commit" and bool(self.lane.get("endpoint_id"))

    @property
    def retired(self) -> bool:
        return lanes_mod.is_retired(self.lane)

    @property
    def long_cadence(self) -> bool:
        """Tested once a night or once a week: that cadence and an endpoint to run on."""
        return lanes_mod.is_long_cadence(self.lane) and bool(self.lane.get("endpoint_id"))

    def states(self, commit: str) -> dict[str, str]:
        """Results of `commit` on this lane, from its summary. What an old
        summary lacks (the drift class, the output sha) the index entry has."""
        cells = as_dict(as_dict(self.summary).get("workflows"))
        indexed = as_dict(next((e for e in self.entries if e["c"] == commit), {}).get("w"))
        out = {}
        for wf, cell in cells.items():
            cell, known = as_dict(cell), as_dict(indexed.get(wf))
            out[wf] = cell_state(cell.get("verdict"), cell.get("detail") or known.get("d"),
                                 cell.get("output_sha256") or known.get("sha"),
                                 cell.get("golden_tag") or known.get("g"),
                                 self.references.get(wf),
                                 "smoke" if "smoke" in (cell.get("level"), known.get("l"))
                                 else None)
        return out

    def covers(self, commit: str):
        """How many commits the run of `commit` tested at once, as far as recorded."""
        rng = as_dict(as_dict(self.summary).get("tested_range"))
        if rng:
            return rng.get("commits_between")
        entry = next((e for e in self.entries if e["c"] == commit), {})
        return as_dict(entry.get("range")).get("n")


@dataclass
class OverviewData:
    commit: str
    branch: str
    reader: Reader
    primary: str | None
    columns: list[LaneColumn]
    head_paths: dict
    manifest: dict = field(default_factory=dict)  # today's workflows, when the caller has them
    now: float | None = None  # when the report is made (None: the clock), for the stale mark


def gather_overview(reader: Reader, lanes: dict, commit: str, branch: str,
                    manifest: dict | None = None, now: float | None = None) -> OverviewData:
    entries = {lane_id: as_dict(lane) for lane_id, lane in as_dict(lanes.get("lanes")).items()}
    if not entries:
        raise ReportError("manifest/lanes.json lists no lanes")
    primary = lanes.get("primary")
    order = sorted(entries, key=lambda i: (i != primary, lanes_mod.is_retired(entries[i])))
    paths = {}
    for lane_id in order:
        prefix = lane_prefix(entries[lane_id])
        paths[lane_id] = (join_path(prefix, "runs", branch, commit, "summary.json"),
                          join_path(prefix.split("/")[0], "index", branch, f"{lane_id}.json"))
    # Every lane shares one results root, and so one index.
    registry_path = f"{lane_prefix(entries[order[0]]).split('/')[0]}/index/lanes.json"
    live = [i for i in order if not lanes_mod.is_retired(entries[i])]
    reader.prefetch([registry_path, *(p[0] for p in paths.values() if p[0]),
                     *(paths[i][1] for i in live if paths[i][1])])
    registry = as_dict((reader.read_json(registry_path) or {}).get("lanes"))
    columns = []
    for lane_id in order:
        summary_path, head_path = paths[lane_id]
        summary = reader.read_json(summary_path) if summary_path else None
        # A retired lane's history is read only when the lane has this commit: an
        # old summary is classified from its index entry, which also names the commit.
        head = (reader.read_json(head_path) if head_path and (lane_id in live or summary)
                else None)
        columns.append(LaneColumn(
            lane_id=lane_id, lane=entries[lane_id], summary=summary,
            head=head or {}, entries=entries_of(head),
            references=as_dict(as_dict(registry.get(lane_id)).get("goldens"))))
    return OverviewData(commit=commit, branch=branch, reader=reader, primary=primary,
                        columns=columns, head_paths={i: p[1] for i, p in paths.items()},
                        manifest=as_dict(manifest), now=now)


def disagreements(per_lane: dict[str, dict[str, str]], labels: dict[str, str]) -> list[str]:
    """Workflows green on one lane and red on another, of those two lanes ran."""
    out = []
    for wf in sorted({wf for states in per_lane.values() for wf in states}, key=str):
        ran = {lane: states[wf] for lane, states in per_lane.items() if wf in states}
        green = [labels[lane] for lane, s in ran.items() if s in GREEN]
        red = [labels[lane] for lane, s in ran.items() if s in RED]
        if len(ran) >= 2 and green and red:
            out.append(f"- {workflow_name(wf)} passes on {', '.join(green)}; fails on "
                       f"{', '.join(red)}")
    return out


# How old a lane's last full run may be before it is marked. The dashboard's
# banner uses the same two limits.
STALE_AFTER_S = {"nightly": 36 * 3600, "weekly": 8 * 24 * 3600}


def full_run(entries: list[dict], runs: set | None) -> tuple[dict, bool]:
    """(the index entry of a lane's newest full run, whether it is a full
    one). A full run holds a result for every workflow the lane runs today
    (`runs`), so a dispatch of one workflow is never shown as the lane's
    result. When no entry holds them all (a workflow was just added) the
    newest entry stands in, and it does too when `runs` is not known."""
    if not entries:
        return {}, False
    if not runs:
        return entries[0], True
    for e in entries:
        if runs <= set(as_dict(e.get("w"))):
            return e, True
    return entries[0], False


def age_text(seconds: float) -> str:
    hours = seconds / 3600
    return plural(round(hours / 24), "day") if hours >= 48 else plural(round(hours), "hour")


def long_lanes_block(columns: list[LaneColumn], runs: dict | None = None,
                     now: float | None = None) -> list[str]:
    """The "Nightly and weekly lanes" section of the overview: one row for
    each lane of `columns`, from the lane's newest full run. `runs` maps a
    lane id to the ids of the workflows the lane runs today. "Covers" answers
    what a contributor asks: is my commit in that run? A last full run older
    than STALE_AFTER_S is marked. A cron that GitHub dropped leaves no run and
    sends no mail, so that mark is its only sign."""
    now = time.time() if now is None else now
    rows = []
    for c in columns:
        name = lane_label(c.lane_id, c.lane)
        wanted = as_dict(runs).get(c.lane_id)
        wanted = wanted if isinstance(wanted, set) else None
        entry, full = full_run(c.entries, wanted)
        if not entry:
            rows.append([name, "no run yet", "", f"{DASH} no result", "", ""])
            continue
        when, url = utc(entry.get("t")), entry_run_url(entry)
        last = (f"[{when}]({url})" if url and when else when) or "time not recorded"
        cadence = c.lane.get("cadence")
        limit = STALE_AFTER_S.get(cadence) if isinstance(cadence, str) else None
        if limit and when and now - entry["t"] > limit:
            last += f" {WARN} no run for {age_text(now - entry['t'])}"
        commit, prev = entry["c"], entry.get("prev")
        n = as_int(as_dict(entry.get("range")).get("n"))
        if is_sha(commit) and is_sha(prev) and n != 1:
            covers = (f"[{tick(prev[:7])}..{tick(commit[:7])}"
                      + (f", {n} commits" if is_batch(n) else "")
                      + f"]({COMFY_URL}/compare/{prev}...{commit})")
        else:
            covers = sha_link(commit) + (", 1 commit" if n == 1 else "")
        states = entry_states(entry, c.references)
        result = rollup(states)
        if not full:
            result += f" · {len(wanted & set(states))} of {plural(len(wanted), 'workflow')}"
        untested = sum(as_dict(cell).get("v") == "infra_error"
                       for cell in as_dict(entry.get("w")).values())
        rows.append([name, last, covers, result, not_passing(states, 6, is_batch(n)),
                     str(untested)])
    if not rows:
        return []
    return ["### Nightly and weekly lanes", "",
            *table(["Lane", "Last full run", "Covers", "Result", "Not passing", "Not tested"],
                   rows)]


def render_overview(o: OverviewData, limits: Limits) -> str:
    labels = {c.lane_id: lane_label(c.lane_id, c.lane) for c in o.columns}
    blocks = [[f"## GPU regression · all lanes · {tick(o.commit[:8])}"]]
    seen: set[str] = set()  # the results the page shows: a smoke one adds its words
    if not o.reader.sha:
        blocks.append(["> [!WARNING]", "> The results branch could not be read, so there is "
                       "nothing to show for this commit."])

    meta = next((as_dict(c.summary.get("commit_meta")) for c in o.columns
                 if c.summary and c.summary.get("commit_meta")), None)
    if meta is None:
        m = next((as_dict(e.get("m")) for c in o.columns for e in c.entries
                  if e["c"] == o.commit and e.get("m")), {})
        meta = {"subject": m.get("s"), "author": m.get("a"), "committed_ts": m.get("ct"),
                "pr": m.get("pr"), "pr_title": m.get("pt"), "pr_author": m.get("pa")}
    blocks.append(table(["", ""], commit_rows(o.commit, meta)))

    rows, per_lane = [], {}
    for c in o.columns:
        if not (c.summary or c.active):
            continue
        role = " · ".join(x for x in (
            md(c.lane.get("role"), 30), "retired" if c.retired else
            "blocking" if c.lane.get("blocking") else "informational") if x)
        if not c.summary:
            rows.append([labels[c.lane_id], role, f"{DASH} no result", "", ""])
            continue
        states = c.states(o.commit)
        per_lane[c.lane_id] = states
        seen.update(states.values())
        when = utc(c.summary.get("run_ts"))
        url = checked_run_url(c.summary.get("run_url")) or run_url(c.summary.get("run_id"))
        rows.append([labels[c.lane_id], role, rollup(states),
                     f"[{when}]({url})" if url and when else when,
                     not_passing(states, 12, is_batch(c.covers(o.commit)))])
    if rows:
        blocks.append(["### This commit on every lane", "",
                       *table(["Lane", "Role", "Result", "Tested", "Not passing"], rows)])
    differ = disagreements(per_lane, labels)
    if differ:
        blocks.append(["### Lanes that disagree at this commit", "", *differ])

    active = [c for c in o.columns if c.active]
    # The nightly and weekly lanes: where their last run stands. They get no
    # column further down: a column would read "not run" on nearly every row.
    long = [c for c in o.columns if c.long_cadence]
    # What each lane runs today, when the manifest is at hand.
    today = {c.lane_id: lane_workflows(o.manifest, c.lane_id, c.lane) for c in active + long}
    blocks.append(long_lanes_block(long, today, o.now))
    for c in long:
        shown, _ = full_run(c.entries, today[c.lane_id])
        seen.update(entry_states(shown, c.references).values())

    failures = []
    for c in active + long:
        # Only for the workflows the lane runs today.
        runs = today[c.lane_id]
        lines = open_failures(c.head, c.entries, runs)
        failures += [f"{labels[c.lane_id]}:", "", *lines, ""] if lines else []
        if any(is_smoke_chain(fb) for wf, fb in as_dict(c.head.get("first_bad")).items()
               if runs is None or wf in runs):
            seen.add("smoke_fail")
    blocks.append(["### Open failures", "", *failures[:-1]] if failures
                  else ["### Open failures", "", "None on the active lanes."])

    by_commit: dict[str, dict[str, dict]] = {}
    for c in active:
        for e in c.entries:
            by_commit.setdefault(e["c"], {})[c.lane_id] = e
    newest = sorted(by_commit, key=lambda sha: max(entry_key(e) for e in by_commit[sha].values()),
                    reverse=True)[:limits.matrix_rows]
    if newest:
        matrix = []
        for sha in newest:
            per = by_commit[sha]
            m = next((as_dict(e.get("m")) for e in per.values() if e.get("m")), {})
            states = {c.lane_id: entry_states(per[c.lane_id], c.references)
                      for c in active if c.lane_id in per}
            seen.update(s for lane_states in states.values() for s in lane_states.values())
            cells = []
            for c in active:
                if c.lane_id not in per:
                    cells.append(label("not_run"))
                    continue
                text = rollup(states[c.lane_id])
                url = entry_run_url(per[c.lane_id])
                names = not_passing(states[c.lane_id], 4, is_batch(
                    as_dict(per[c.lane_id].get("range")).get("n")))
                cells.append((f"[{text}]({url})" if url else text)
                             + (f": {names}" if names else ""))
            mark = " **this report**" if sha == o.commit else ""
            matrix.append([sha_link(sha) + mark, md(m.get("s"), 80), pr_link(m.get("pr")),
                           md(m.get("a"), 40), *cells,
                           "lanes differ" if disagreements(states, labels) else ""])
        blocks.append(["### Recent commits", "",
                       *table(["Commit", "Subject", "PR", "Author",
                               *(labels[c.lane_id] for c in active), "Note"], matrix)])

    notes = []
    for c in o.columns:
        if c.retired:
            url = raw_url(o.reader, o.head_paths.get(c.lane_id))
            where = f"[its run index]({url})" if url else "its run index on the results branch"
            notes.append(f"- {labels[c.lane_id]} is retired: nothing runs on it any more, and "
                         f"its history is kept in {where}.")
        elif lanes_mod.is_long_cadence(c.lane) and not c.long_cadence:
            notes.append(f"- {labels[c.lane_id]} is not tested yet: it has no endpoint.")
        elif not (c.active or c.long_cadence):
            notes.append(f"- {labels[c.lane_id]} is not tested yet: it has no endpoint or is "
                         "not a per-commit lane.")
    if notes:
        blocks.append(["### Other lanes", "", *notes])
    smoke_level = bool(seen & {"smoke_pass", "smoke_fail"})
    if smoke_level:
        blocks[0] += ["", SMOKE_NOTE]
    footer = ["---", "", legend(smoke_level)]
    for extra in (shortened_note(limits), dashboard_line()):
        if extra:
            footer += ["", *extra]
    blocks.append(footer)
    return "\n\n".join("\n".join(b) for b in blocks if b) + "\n"


def overview_report(reader: Reader, lanes: dict, commit: str, branch: str,
                    manifest: dict | None = None, now: float | None = None) -> str | None:
    """The Markdown that puts every lane side by side, within the size limit.
    With `manifest`, today's workflows, a lane's open failures are listed only
    for the workflows it still runs. `now` is the time the stale mark of a
    nightly or weekly lane is worked out from (default: the clock)."""
    o = gather_overview(reader, lanes, commit, branch, manifest, now)
    return fit(lambda limits: render_overview(o, limits))


# -- command line ---------------------------------------------------------------------
def fallback_report(args, reason: str) -> str:
    """What the summary says when the report itself failed: the reason as a
    fixed word (an exception class, never its message) and the links that need
    no data."""
    commit = getattr(args, "commit", None)
    out = ["## GPU regression · report not available", "",
           f"The report could not be built ({reason}). The run's results are not affected."]
    links = []
    try:
        lane_id = getattr(args, "lane", None)
        lane = as_dict(as_dict(lanes_mod.load_lanes().get("lanes")).get(lane_id))
        branch = getattr(args, "branch", None)
        if lane and is_sha(commit) and is_branch(branch):
            path = join_path(lane_prefix(lane), "runs", branch, commit)
            links.append(f"[this run's files on the results branch](https://github.com/"
                         f"{results_repo()}/tree/{results_branch()}/{quote(path, safe='/')})")
    except Exception:  # noqa: BLE001 - the fallback must not fail
        pass
    if is_sha(commit):
        links.append(f"[the commit]({COMFY_URL}/commit/{commit})")
    if this_run_url():
        links.append(f"[this Actions run]({this_run_url()})")
    artifact = getattr(args, "artifact_url", None)
    if isinstance(artifact, str) and ARTIFACT_URL_RE.fullmatch(artifact):
        links.append(f"[the files attached to this run]({artifact})")
    if dashboard_url():
        links.append(f"[the dashboard]({dashboard_url()})")
    if links:
        out += ["", "Where to look instead: " + ", ".join(links) + "."]
    return "\n".join(out) + "\n"


def build(args, reader: Reader | None = None) -> str:
    if not is_sha(args.commit):
        raise ReportError("--commit is not a hex sha")
    if not is_branch(args.branch):
        raise ReportError("--branch is not a plain branch name")
    if reader is None:
        try:
            reader = GitReader(args.results_dir, fetch=not args.no_fetch)
        except (ReportError, OSError, subprocess.SubprocessError):
            reader = Reader()  # report what the runner holds; say the branch is unread
    lanes = lanes_mod.load_lanes()
    if args.cmd == "overview":
        text = overview_report(reader, lanes, args.commit, args.branch,
                               manifest=load_json_or_empty(lanes_mod.MANIFEST_FILE))
    else:
        text = lane_report(
            reader, lanes, args.commit, args.branch, args.lane, summary_file=args.summary,
            workdir=args.workdir, skipped=args.skipped, artifact_url=args.artifact_url,
            manifest=load_json_or_empty(lanes_mod.MANIFEST_FILE),
            packs=load_json_or_empty(lanes_mod.PACKS_FILE))
    if text is None:
        raise ReportError("it does not fit GitHub's size limit")
    return text


def emit(text: str, out: str | None) -> str:
    """Write the Markdown; returns a fixed status line. Never to stdout inside
    Actions: the text quotes logs."""
    step = os.environ.get("GITHUB_STEP_SUMMARY")
    wrote = []
    if step:
        with open(step, "a", encoding="utf-8", newline="\n") as f:
            f.write(text)
        wrote.append("the job summary")
    if out and out != "-":
        Path(out).write_text(text, encoding="utf-8", newline="\n")
        wrote.append("the --out file")
    elif out == "-" and not step:
        sys.stdout.buffer.write(text.encode("utf-8"))
        return ""
    if not wrote:
        return "run report: nowhere to write (no GITHUB_STEP_SUMMARY, no --out)"
    return f"run report: wrote {byte_size(text)} bytes to {' and '.join(wrote)}"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lane", "overview"):
        p = sub.add_parser(name)
        p.add_argument("--commit", required=True, help="the ComfyUI commit the report is about")
        p.add_argument("--branch", required=True)
        p.add_argument("--results-dir", default=".",
                       help="git checkout that can fetch the results branch")
        p.add_argument("--no-fetch", action="store_true",
                       help="read HEAD of --results-dir (a results checkout that is current)")
        p.add_argument("--out", default=None,
                       help="write the Markdown here when GITHUB_STEP_SUMMARY is not set "
                            "('-' is stdout)")
        if name == "lane":
            p.add_argument("--lane", required=True, help="lane id from manifest/lanes.json")
            p.add_argument("--summary", default=None,
                           help="the summary run_regression.py wrote on this runner")
            p.add_argument("--workdir", default=None,
                           help="run_regression.py's --workdir (<DIR>/<workflow>/run.json)")
            p.add_argument("--skipped", action="store_true",
                           help="this run did not test the commit: show what is published")
            p.add_argument("--artifact-url", default=None,
                           help="link to the files this run attached")
    return ap


def main(argv: list[str] | None = None, reader: Reader | None = None) -> int:
    try:
        args = parser().parse_args(argv)
    except SystemExit as e:
        if e.code:  # argparse has printed its usage; the job still must not fail
            _write(fallback_report(None, "its arguments were not accepted"), None)
        return 0
    try:
        text = build(args, reader)
    except ReportError as e:
        text = fallback_report(args, str(e))
    except Exception as e:  # noqa: BLE001 - a report is a courtesy, never a failure
        text = fallback_report(args, type(e).__name__)
    _write(text, args.out)
    return 0


def _write(text: str, out: str | None):
    try:
        status = emit(text, out)
    except Exception as e:  # noqa: BLE001
        status = f"run report: could not write the summary ({type(e).__name__})"
    if status:
        print(status)


if __name__ == "__main__":
    sys.exit(main())
