"""regression_skip: when the cron tests a commit on a lane, and when it leaves it alone."""
import json
import re
from pathlib import Path

import pytest

import regression_skip
from conftest import workflow_steps

SHA = "5f415e31abfd837870bbc733ba1e158de71ed985"
ROOT = Path(__file__).resolve().parents[2]


def _summary(overall="pass", infra=False, attempt=None, **workflows):
    s = {"schema_version": 2, "commit": SHA, "overall": overall, "has_infra_error": infra,
         "workflows": {wf: {"verdict": v} for wf, v in (workflows or {"sdxl_t2i": "pass"}).items()}}
    if attempt is not None:
        s["attempt"] = attempt
    return s


def decide(**kw):
    return regression_skip.decide(SHA, **kw)[0]


def test_an_untested_commit_runs():
    assert decide()
    assert decide(latest={"commit": "a" * 40})


@pytest.mark.parametrize("summary", [
    _summary("pass"),
    _summary("fail", sdxl_t2i="fail"),
    # The 2026-11-03/04 case: a timeout is an execution error, a result like any
    # other, even though the previous-run pointer does not advance on it.
    _summary("fail", z_image_turbo_t2i="execution_error", sdxl_t2i="pass"),
    _summary("pass", sdxl_t2i="no_baseline"),
])
def test_a_published_result_is_never_rerun_by_the_cron(summary):
    run, why = regression_skip.decide(SHA, summary, latest={"commit": "a" * 40})
    assert not run
    assert "already published" in why and "commit=" in why


def test_infra_errors_get_one_more_attempt():
    first = _summary("pass", infra=True, sdxl_t2i="infra_error")
    assert decide(summary=first)                      # no `attempt`: published by attempt 1
    assert decide(summary={**first, "attempt": 1})
    assert not decide(summary={**first, "attempt": 2})
    assert not decide(summary={**first, "attempt": 5})
    # Recognised from the verdicts too, for a summary without has_infra_error.
    bare = {"commit": SHA, "overall": "pass", "workflows": {"a": {"verdict": "infra_error"}}}
    assert decide(summary=bare)
    assert not decide(summary={**bare, "attempt": 2})


def test_an_explicit_commit_always_runs():
    for summary in (_summary("pass"), _summary("fail", z="execution_error"),
                    _summary("pass", infra=True, attempt=9, a="infra_error")):
        assert decide(summary=summary, explicit=True)
    assert decide(latest={"commit": SHA}, explicit=True)


def test_the_previous_run_pointer_still_counts_without_a_summary():
    assert not decide(latest={"commit": SHA})


def test_a_listed_but_unreadable_summary_counts_as_published():
    run, why = regression_skip.decide(SHA, {"unreadable": True})
    assert not run and "could not be read" in why
    assert decide(summary={"unreadable": True}, explicit=True)


def test_the_incident_replays_as_one_run_per_commit():
    """Ticks every 30 minutes over an execution-error streak: each commit runs
    once, however long master HEAD stays red."""
    published: dict[str, dict] = {}
    heads = ["783ab728", "783ab728", "783ab728", "7f09d3fa", "7f09d3fa", "5f415e31", "5f415e31"]
    runs = []
    for head in heads:
        if regression_skip.decide(head, published.get(head), {"commit": "ab08f7b2"})[0]:
            runs.append(head)
            published[head] = {"commit": head, "overall": "fail", "has_infra_error": False,
                               "attempt": regression_skip.next_attempt(published.get(head)),
                               "workflows": {"z_image_turbo_t2i": {"verdict": "execution_error"}}}
    assert runs == ["783ab728", "7f09d3fa", "5f415e31"]


def test_next_attempt_counts_from_what_is_published():
    assert regression_skip.next_attempt(None) == 1
    assert regression_skip.next_attempt({}) == 1
    assert regression_skip.next_attempt({"commit": SHA}) == 2
    assert regression_skip.next_attempt({"commit": SHA, "attempt": 2}) == 3
    assert regression_skip.next_attempt({"commit": SHA, "attempt": "junk"}) == 2
    # The start marker counts too, and the later of the two wins.
    assert regression_skip.next_attempt(None, {"attempt": 1}) == 2
    assert regression_skip.next_attempt({"attempt": 1}, {"attempt": 2}) == 3
    assert regression_skip.next_attempt({"attempt": 2}, {"attempt": 1}) == 3


def test_a_run_that_died_before_publishing_gets_one_more_attempt():
    """A marker without a summary: the run started its GPU jobs and never
    published (a failed push, a killed job). One retry, then the cron stops."""
    run, why = regression_skip.decide(SHA, marker={"attempt": 1})
    assert run and "published no result" in why and "attempt 2 of 2" in why
    run, why = regression_skip.decide(SHA, marker={"attempt": 2})
    assert not run and "none published a result" in why and "commit=" in why
    assert not decide(marker={"attempt": 7})
    assert not decide(marker={"unreadable": True})
    assert decide(marker={"attempt": 2}, explicit=True)
    # The previous-run pointer still settles it first.
    assert not decide(latest={"commit": SHA}, marker={"attempt": 1})


def test_a_retry_that_died_counts_against_the_infra_retry():
    """Attempt 1 published infra errors, attempt 2 pushed its marker and died:
    both attempts are spent."""
    first = _summary("pass", infra=True, attempt=1, sdxl_t2i="infra_error")
    assert decide(summary=first, marker={"attempt": 1})
    assert not decide(summary=first, marker={"attempt": 2})
    # A marker never reopens a commit whose result was not an infra error.
    assert not decide(summary=_summary("fail", z="execution_error"), marker={"attempt": 1})


def test_a_publish_that_keeps_failing_is_bounded():
    """The open end of the 2026-11 incident: every run of HEAD does its GPU
    work and then fails to push its results. Only the markers (pushed before
    the GPU jobs) land, and the cron stops after MAX_ATTEMPTS."""
    marker = None
    runs = 0
    for _tick in range(10):
        if not regression_skip.decide(SHA, None, {"commit": "a" * 40}, marker=marker)[0]:
            continue
        runs += 1
        marker = {"attempt": regression_skip.next_attempt(None, marker)}
    assert runs == regression_skip.MAX_ATTEMPTS


def test_cli_prints_the_step_output(tmp_path, capsys):
    summary, latest, empty = tmp_path / "s.json", tmp_path / "l.json", tmp_path / "e.json"
    summary.write_text(json.dumps(_summary("fail", z="execution_error")), encoding="utf-8")
    latest.write_text(json.dumps({"commit": "a" * 40}), encoding="utf-8")
    empty.write_text("", encoding="utf-8")

    def out(*argv):
        assert regression_skip.main(["--commit", SHA, *argv]) == 0
        cap = capsys.readouterr()
        return cap.out.strip(), cap.err

    assert out("--summary", str(summary), "--latest", str(latest))[0] == "skip=true"
    assert out("--summary", str(summary), "--explicit")[0] == "skip=false"
    # `git show` of a path that does not exist leaves an empty file.
    line, err = out("--summary", str(empty), "--latest", str(empty))
    assert line == "skip=false" and "no published result" in err
    assert out("--summary", str(tmp_path / "missing.json"))[0] == "skip=false"
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert out("--summary", str(tmp_path / "bad.json"))[0] == "skip=false"
    marker = tmp_path / "a.json"
    marker.write_text(json.dumps({"attempt": 2}), encoding="utf-8")
    assert out("--summary", str(empty), "--attempt", str(marker))[0] == "skip=true"
    assert out("--summary", str(empty), "--attempt", str(empty))[0] == "skip=false"


def test_the_workflow_decides_with_the_published_summary():
    regression = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert 'RUN_DIR="$PREFIX/runs/master/$COMMIT"' in regression
    assert 'read_published "$RUN_DIR/summary.json" "$SUMMARY"' in regression
    assert 'read_published "$RUN_DIR/attempt.json" "$ATTEMPT"' in regression
    assert 'git ls-tree --name-only FETCH_HEAD -- "$1"' in regression
    assert "echo '{\"unreadable\": true}' > \"$2\"" in regression
    assert '--summary "$SUMMARY" --attempt "$ATTEMPT" --latest "$LATEST"' in regression
    assert "python scripts/regression_skip.py" in regression
    assert 'if [ -n "$INPUT_COMMIT" ]; then args+=(--explicit); fi' in regression
    assert "steps.target.outputs.skip != 'true'" in regression


REPORT_ONLY = ("REPORT_ONLY: ${{ github.event_name == 'workflow_dispatch' && "
               "(inputs.report_only == true || inputs.report_only == 'true') }}")


def test_a_report_only_dispatch_starts_no_gpu_job():
    """report_only shows what is published. The skip step answers skip=true
    before it reads anything, and every step that tests, attaches or comments
    runs only when skip is not true."""
    steps = dict(workflow_steps(".github/workflows/regression.yml"))
    skip = steps["Skip if already tested on this lane"]
    assert REPORT_ONLY in skip
    block = re.search(r'\n( +)if \[ "\$REPORT_ONLY" = "true" \]; then\n(.*?)\n\1fi\n', skip, re.S)
    assert block, "the report-only early exit is gone"
    assert block.group(2).splitlines()[-2:] == [
        block.group(1) + '  echo "skip=true" >> "$GITHUB_OUTPUT"', block.group(1) + "  exit 0"]
    # Nothing is fetched or decided before it.
    for later in ("git fetch", "regression_skip.py", "read_published"):
        assert block.start() < skip.index(later)
    assert "id: target" in skip
    # The commit every lane reports on is the one resolved once for the run.
    assert "COMMIT: ${{ needs.lanes.outputs.commit }}" in skip
    for name, condition in (("Run regression suite", "if: steps.target.outputs.skip != 'true'"),
                            ("Attach the run's files",
                             "if: always() && steps.target.outputs.skip != 'true'"),
                            ("PR comment (PULSE)",
                             "if: always() && steps.target.outputs.skip != 'true'")):
        assert condition in steps[name], name
    assert "RUNPOD_API_KEY" in steps["Run regression suite"]
    assert not [name for name, text in steps.items()
                if "RUNPOD_API_KEY" in text and name != "Run regression suite"]


def test_a_report_only_dispatch_plans_every_lane_and_runs_beside_test_runs():
    """Reading results needs no endpoint and no cadence: a report-only run also
    shows a retired lane. It gets a concurrency group of its own, so it never
    waits for a test run or holds one up."""
    text = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    plan = dict(workflow_steps(".github/workflows/regression.yml"))["plan"]
    assert REPORT_ONLY in plan
    assert re.search(r'if \[ "\$REPORT_ONLY" = "true" \]; then\n +matrix=\$\(python lanes\.py '
                     r'report "\$\{only\[@\]\}"\)\n +else\n +matrix=\$\(python lanes\.py '
                     r'per-commit "\$\{only\[@\]\}"\)\n +fi', plan)
    assert ("  group: ${{ github.event_name == 'workflow_dispatch' && (inputs.report_only == true "
            "|| inputs.report_only == 'true') && format('gpu-regression-report-{0}', "
            "github.run_id) || 'gpu-regression' }}\n  cancel-in-progress: false") in text
    assert "report_only:\n" in text and "type: boolean\n        default: false" in text


def test_the_commit_is_resolved_once_to_a_full_sha():
    """Results are stored under 40 character directories. An abbreviated input
    is expanded before any lane uses it, and one that does not resolve stops
    the run."""
    step = dict(workflow_steps(".github/workflows/regression.yml"))["Resolve target commit"]
    assert "GH_TOKEN: ${{ github.token }}" in step
    assert 'FULL=$(gh api "repos/Comfy-Org/ComfyUI/commits/$COMMIT" --jq .sha) || FULL=""' in step
    assert ('if ! [[ "$FULL" =~ ^[0-9a-f]{40}$ ]] || [ "${FULL:0:${#COMMIT}}" != "$COMMIT" ]; '
            "then") in step
    assert 'if [ -n "$COMMIT" ] && [ "${#COMMIT}" -lt 40 ]; then' in step
    assert step.index("gh api") < step.index('echo "commit=$COMMIT" >> "$GITHUB_OUTPUT"')
    assert 'echo "short=${COMMIT:0:8}" >> "$GITHUB_OUTPUT"' in step
    text = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert "commit: ${{ steps.commit.outputs.commit }}" in text
    # No lane job reads the raw input as the commit to test or report.
    assert "COMMIT: ${{ inputs.commit }}" not in text.replace("INPUT_COMMIT: ", "")
