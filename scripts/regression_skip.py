"""Whether the regression cron tests a commit on a lane (again).

The cron polls ComfyUI master every 30 minutes and tests HEAD on every
per-commit lane. A commit counts as tested on a lane once a run published its
summary there, whatever the verdict: pass, fail and execution_error are all
results, and testing the same commit again only spends GPU time on the same
answer. The previous-run pointer (latest/<branch>.json) is no guide here: it
only advances on fully comparable runs, so while master HEAD hit an execution
error the cron used to re-run it on every tick.

Every run pushes a start marker (runs/<branch>/<SHA>/attempt.json, with its
attempt number) before it submits any GPU job, so attempts are counted even
when a run dies before publishing its summary: a failed push, a killed job, a
crash. Two cases get another attempt, up to MAX_ATTEMPTS per commit per lane:
a summary that records infra errors (RunPod, queue, checkout, model volume,
pack install: not a verdict on the commit), and a marker with no summary. A
dispatch that names a commit always runs.

  python regression_skip.py --commit SHA [--explicit] [--summary FILE]
                            [--attempt FILE] [--latest FILE]

FILE is the lane's published runs/<branch>/<SHA>/summary.json (or attempt.json,
or latest/<branch>.json); a missing or empty file means there is none, and
{"unreadable": true} means the results tree lists one that could not be read
(counted as published). Prints `skip=true` or `skip=false` for
$GITHUB_OUTPUT, and the reason on stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Standard library only, like lanes.py: run_regression.py imports it too.
MAX_ATTEMPTS = 2


def _count(record: dict | None) -> int:
    """The attempt a published summary or marker records (before the count
    existed: 1); 0 when there is none."""
    if not record:
        return 0
    try:
        return max(1, int(record.get("attempt") or 1))
    except (TypeError, ValueError):
        return 1


def attempts(summary: dict | None, marker: dict | None = None) -> int:
    """Attempts already spent on a commit: the latest of its summary and its
    start marker (a run that died after pushing its marker still counts)."""
    return max(_count(summary), _count(marker))


def next_attempt(summary: dict | None, marker: dict | None = None) -> int:
    """The attempt number a new run of a commit records, given what is published."""
    return attempts(summary, marker) + 1


def has_infra_error(summary: dict) -> bool:
    if summary.get("has_infra_error"):
        return True
    return any((wr or {}).get("verdict") == "infra_error"
               for wr in (summary.get("workflows") or {}).values())


def decide(commit: str, summary: dict | None = None, latest: dict | None = None,
           explicit: bool = False, marker: dict | None = None) -> tuple[bool, str]:
    """(run?, why) for testing `commit` on a lane, given what the lane published."""
    short = commit[:8]
    again = "dispatch it with commit= to test it again"
    if explicit:
        return True, f"{short} was requested explicitly"
    # The results tree lists a file whose content could not be fetched: do not
    # spend GPU time on a read error (the next tick reads it again).
    if summary and summary.get("unreadable"):
        return False, f"{short} has a published summary that could not be read; {again}"
    if marker and marker.get("unreadable"):
        return False, f"{short} has an attempt marker that could not be read; {again}"
    n = attempts(summary, marker)
    if summary:
        verdict = f"overall {summary.get('overall')}"
        if not has_infra_error(summary):
            return False, f"{short} already published on this lane ({verdict}); {again}"
        if n < MAX_ATTEMPTS:
            return True, (f"{short} attempt {n} had infra errors; retrying "
                          f"(attempt {n + 1} of {MAX_ATTEMPTS})")
        return False, (f"{short} already published ({verdict}, infra errors after {n} "
                       f"attempts); {again}")
    if latest and latest.get("commit") == commit:
        return False, f"{short} is the lane's previous-run pointer; already tested"
    if marker:
        if n < MAX_ATTEMPTS:
            return True, (f"{short} attempt {n} started but published no result; retrying "
                          f"(attempt {n + 1} of {MAX_ATTEMPTS})")
        return False, (f"{short} started {n} attempts on this lane and none published a "
                       f"result; {again}")
    return True, f"{short} has no published result on this lane yet"


def read_optional_json(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--explicit", action="store_true",
                    help="the commit was named by a dispatch: always test it")
    ap.add_argument("--summary", default=None, help="the lane's published summary.json for it")
    ap.add_argument("--attempt", default=None,
                    help="the lane's published attempt.json (start marker) for it")
    ap.add_argument("--latest", default=None, help="the lane's latest/<branch>.json")
    args = ap.parse_args(argv)
    run, why = decide(args.commit, read_optional_json(args.summary),
                      read_optional_json(args.latest), args.explicit,
                      read_optional_json(args.attempt))
    print(f"{'testing' if run else 'skipping'}: {why}", file=sys.stderr)
    print(f"skip={'false' if run else 'true'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
