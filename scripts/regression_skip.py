"""Whether the regression cron tests a commit on a lane (again).

The cron polls ComfyUI master every 5 minutes and tests HEAD on every
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

A run that names its workflows tests only those. When the commit had no result
of a full run yet, its summary lists them under `scope`. Such a summary is not
the lane's result of the commit, so it counts as no summary here: the cron
still tests the commit, and that run replaces the summary. Such a run pushes
no start marker either, so it is not counted as an attempt.

A nightly or weekly run pushes what it has every few workflows, with `partial`
in the summary. A summary that still says so belongs to a run that did not
finish. It counts as a start marker without a result: one more attempt.

An untested workflow is a reason for the retry only when another run could
test it. One whose models are not on the volume, whose model host refuses a
file, or that the run's deadline did not reach (NO_RETRY) would end the same
way, so a summary with only such workflows untested is left alone. So is one
whose job used up a time limit (`"reason": "timed_out"` in its cell: the
worker stopped its model download): another run could take as long again.

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
# Worker statuses of an untested workflow that another run right away would
# not change: its models are not on the volume, the model host refuses a file,
# or the run's deadline came before the workflow could start.
NO_RETRY = frozenset({"missing_model", "model_access_error", "not_reached"})
# `reason` in the cell of an untested workflow whose job used up a time limit
# the runner gave it (submit_and_poll.TIMED_OUT: this module imports nothing).
TIMED_OUT = "timed_out"


def _count(record: dict | None) -> int:
    """The attempt a published summary or marker records (before the count
    existed: 1); 0 when there is none."""
    if not record:
        return 0
    try:
        return max(1, int(record.get("attempt") or 1))
    except (TypeError, ValueError):
        return 1


def is_scoped(summary: dict | None) -> bool:
    """A summary of named workflows only (`scope`): not the lane's result of the commit."""
    return bool(summary) and "scope" in summary


def is_partial(summary: dict | None) -> bool:
    """A summary a run pushed before it had every result (`partial`). The
    run's final summary replaces it, so one that is still there belongs to a
    run that did not finish."""
    return bool(summary) and "partial" in summary


def attempts(summary: dict | None, marker: dict | None = None) -> int:
    """Attempts already spent on a commit: the latest of its summary and its
    start marker (a run that died after pushing its marker still counts, and
    so does one that left a partial summary). A summary of named workflows
    only is not an attempt."""
    return max(_count(None if is_scoped(summary) else summary), _count(marker))


def next_attempt(summary: dict | None, marker: dict | None = None) -> int:
    """The attempt number a new run of a commit records, given what is published."""
    return attempts(summary, marker) + 1


def untested(summary: dict) -> list[str]:
    """The worker status of each workflow of a summary that was not tested
    (None for a cell that names none)."""
    return [(wr or {}).get("worker_status")
            for wr in (summary.get("workflows") or {}).values()
            if (wr or {}).get("verdict") == "infra_error"]


def used_up_its_time(cell: dict | None) -> bool:
    """Whether the cell of an untested workflow says that its job used up a
    time limit (`reason`). Another run could test it, and could take as long."""
    return (cell or {}).get("reason") == TIMED_OUT


def no_retry(cell: dict | None) -> bool:
    """Whether an untested workflow is no reason to run its commit again:
    its status is in NO_RETRY, or its job used up a time limit."""
    status = (cell or {}).get("worker_status")
    return (isinstance(status, str) and status in NO_RETRY) or used_up_its_time(cell)


def _untested_cells(summary: dict) -> list[dict]:
    return [wr or {} for wr in (summary.get("workflows") or {}).values()
            if (wr or {}).get("verdict") == "infra_error"]


def has_infra_error(summary: dict) -> bool:
    """Whether a summary has an untested workflow that another run could
    test: the reason for the cron's retry. A workflow that `no_retry` names
    is not one. A summary that says it has infra errors and shows no such
    cell is taken at its word."""
    cells = _untested_cells(summary)
    if cells:
        return any(not no_retry(cell) for cell in cells)
    return bool(summary.get("has_infra_error"))


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
    scoped = is_scoped(summary)
    # A run that did not finish left what it had: counted as an attempt, like
    # a start marker, and no more a result than one. No summary is both: only
    # a full run pushes on its way, and a run of named workflows at such a
    # commit replaces what it finds.
    partial = is_partial(summary)
    n = attempts(summary, marker)
    if scoped or partial:
        summary = None
    if summary:
        verdict = f"overall {summary.get('overall')}"
        if not has_infra_error(summary):
            cells = _untested_cells(summary)
            held = sorted({cell.get("worker_status") for cell in cells
                           if isinstance(cell.get("worker_status"), str)
                           and not used_up_its_time(cell)})
            if held:
                verdict += (f", untested workflows that another run would not test either: "
                            f"{', '.join(held)}")
            spent = sum(used_up_its_time(cell) for cell in cells)
            if spent:
                verdict += (f", {spent} untested workflow{'' if spent == 1 else 's'} whose job "
                            "used up its time: another run can cost as much")
            return False, f"{short} already published on this lane ({verdict}); {again}"
        if n < MAX_ATTEMPTS:
            return True, (f"{short} attempt {n} had infra errors; retrying "
                          f"(attempt {n + 1} of {MAX_ATTEMPTS})")
        return False, (f"{short} already published ({verdict}, infra errors after {n} "
                       f"attempts); {again}")
    if latest and latest.get("commit") == commit:
        return False, f"{short} is the lane's previous-run pointer; already tested"
    if partial:
        if n < MAX_ATTEMPTS:
            return True, (f"{short} attempt {n} did not finish; retrying "
                          f"(attempt {n + 1} of {MAX_ATTEMPTS})")
        return False, f"{short} started {n} attempts on this lane and none finished; {again}"
    if marker:
        if n < MAX_ATTEMPTS:
            return True, (f"{short} attempt {n} started but published no result; retrying "
                          f"(attempt {n + 1} of {MAX_ATTEMPTS})")
        return False, (f"{short} started {n} attempts on this lane and none published a "
                       f"result; {again}")
    if scoped:
        return True, (f"{short} has a result for named workflows only on this lane; testing "
                      "every workflow of the lane")
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
