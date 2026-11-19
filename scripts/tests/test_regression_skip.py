"""regression_skip: when the cron tests a commit on a lane, and when it leaves it alone."""
import json
import re
import textwrap
from pathlib import Path

import pytest

import lanes as lanes_mod
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


def test_a_result_of_named_workflows_only_does_not_skip_the_full_run(tmp_path, capsys):
    """A dispatch that names a workflow and no commit tests master HEAD. Its
    summary holds that one workflow, under `scope`. The cron must still test
    the commit, or the one-workflow result would stand as the lane's own."""
    scoped = {**_summary("pass", attempt=1, flux_dev_t2i="pass"), "scope": ["flux_dev_t2i"]}
    run, why = regression_skip.decide(SHA, scoped, latest={"commit": "a" * 40})
    assert run and "named workflows only" in why and "every workflow of the lane" in why
    # Whatever it holds: a named workflow's failure or infra error is not the lane's result.
    assert decide(summary={**_summary("fail", a="execution_error"), "scope": ["a"]})
    assert decide(summary={**_summary("pass", infra=True, attempt=5, a="infra_error"),
                           "scope": ["a"]})
    # It is not an attempt: such a run pushes no start marker.
    assert regression_skip.attempts(scoped) == 0 and regression_skip.next_attempt(scoped) == 1
    assert regression_skip.next_attempt(scoped, {"attempt": 1}) == 2
    # A full run that died beside it is counted as it always was.
    run, why = regression_skip.decide(SHA, scoped, marker={"attempt": 1})
    assert run and "attempt 2 of 2" in why
    run, why = regression_skip.decide(SHA, scoped, marker={"attempt": 2})
    assert not run and "none published a result" in why
    # The full run replaces the summary, and from then on the commit is tested.
    assert not decide(summary=_summary("pass", attempt=1))
    # A re-test at a commit that has the lane's result adds `retests`, not
    # `scope`: the summary stays the lane's result.
    retested = {**_summary("fail", a="fail", b="pass"), "retests": [{"workflows": ["b"]}]}
    run, why = regression_skip.decide(SHA, retested)
    assert not run and "already published" in why
    # The step's own call.
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(scoped), encoding="utf-8")
    assert regression_skip.main(["--commit", SHA, "--summary", str(path)]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "skip=false" and "testing: " in cap.err


def test_a_summary_of_a_run_that_did_not_finish_gets_one_more_attempt(tmp_path, capsys):
    """A nightly or weekly run pushes what it has every few workflows, with
    `partial` in the summary. Its final summary has no such key. One that is
    still there belongs to a run that died: it counts as a start marker
    without a result."""
    partial = {**_summary("pass", attempt=1, a="pass", b="pass"),
               "partial": {"done": 2, "of": 55}}
    run, why = regression_skip.decide(SHA, partial, latest={"commit": "a" * 40})
    assert run and "attempt 1 did not finish" in why and "attempt 2 of 2" in why
    # Beside its start marker, which every full run pushes first.
    assert decide(summary=partial, marker={"attempt": 1})
    # It is an attempt: the next run is the second, and after that the cron stops.
    assert regression_skip.attempts(partial) == 1 and regression_skip.next_attempt(partial) == 2
    second = {**partial, "attempt": 2}
    run, why = regression_skip.decide(SHA, second, marker={"attempt": 2})
    assert not run and "started 2 attempts" in why and "none finished" in why and "commit=" in why
    # A second attempt that died before its first checkpoint left only its marker.
    assert not decide(summary=partial, marker={"attempt": 2})
    # What it holds so far is no result of the commit, whatever it is.
    for cells in ({"a": "execution_error"}, {"a": "fail"}, {"a": "infra_error"}):
        assert decide(summary={**_summary("fail", attempt=1, **cells),
                               "partial": {"done": 1, "of": 3}}), cells
    assert decide(summary=second, explicit=True)
    # A run that published before moved the pointer here: the commit has its result.
    assert not decide(summary=partial, latest={"commit": SHA})
    # The final summary of the same run is a result like any other.
    final = {key: value for key, value in partial.items() if key != "partial"}
    run, why = regression_skip.decide(SHA, final)
    assert not run and "already published" in why
    assert regression_skip.is_partial(partial)
    assert not regression_skip.is_partial(final) and not regression_skip.is_partial(None)
    # The step's own call.
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    assert regression_skip.main(["--commit", SHA, "--summary", str(path)]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "skip=false" and "did not finish" in cap.err


def _untested(*statuses, attempt=1):
    """A published summary with one passing workflow and one untested workflow
    for each worker status given."""
    cells = {"sdxl_t2i": {"verdict": "pass", "worker_status": "ok"}}
    cells.update({f"wf_{i}": {"verdict": "infra_error", "worker_status": status}
                  for i, status in enumerate(statuses)})
    return {"schema_version": 2, "commit": SHA, "overall": "pass", "has_infra_error": True,
            "attempt": attempt, "workflows": cells}


def test_a_workflow_that_another_run_would_not_test_either_is_no_reason_for_a_retry():
    """Models that are not on the volume yet, a model host that refuses a
    file, a workflow the run's deadline did not reach: a second run of the
    commit right away would end the same way, and cost a full run."""
    assert regression_skip.NO_RETRY == {"missing_model", "model_access_error", "not_reached"}
    for status in sorted(regression_skip.NO_RETRY):
        run, why = regression_skip.decide(SHA, _untested(status))
        assert not run and "already published" in why and "commit=" in why, status
        assert f"another run would not test either: {status})" in why
        assert not regression_skip.has_infra_error(_untested(status))
    assert not decide(summary=_untested("missing_model", "not_reached", "missing_model"))
    # One workflow that another run can test is reason enough, once.
    for status in ("worker_error", "infra_error", "model_fetch_error", "harness_error",
                   "no_outputs", None, ["missing_model"]):
        assert decide(summary=_untested("missing_model", status)), status
        assert not decide(summary=_untested("missing_model", status, attempt=2)), status
    # A dispatch that names the commit tests it whatever stands there.
    assert decide(summary=_untested("missing_model"), explicit=True)
    # A summary that says it has infra errors and shows no such cell is taken at its word.
    bare = {"commit": SHA, "overall": "pass", "has_infra_error": True, "workflows": {}}
    assert decide(summary=bare) and regression_skip.has_infra_error({"has_infra_error": True})
    assert regression_skip.untested(_untested("missing_model", None)) == ["missing_model", None]
    # A result with nothing untested says nothing about it.
    assert "another run" not in regression_skip.decide(SHA, _summary("pass"))[1]


def test_a_job_that_used_up_its_time_is_no_reason_for_a_retry():
    """A download that the worker stopped at its time limit: the cell says
    `"reason": "timed_out"`. Another run could test the workflow, and could
    download as long and cost as much, so the cron leaves the commit alone."""
    def stopped(status="model_fetch_error", **over):
        return {"verdict": "infra_error", "worker_status": status, "reason": "timed_out",
                "error": "the download was too slow: 12.4 of 56.6 GB in 600 s (20.7 MB/s)",
                **over}

    summary = _untested()
    summary["workflows"]["slow"] = stopped()
    assert not regression_skip.has_infra_error(summary)
    run, why = regression_skip.decide(SHA, summary)
    assert not run and "already published" in why and "commit=" in why
    assert ("(overall pass, 1 untested workflow whose job used up its time: another run can "
            "cost as much)") in why
    # Beside workflows that another run would not test either, each is named
    # in its own words.
    summary = _untested("missing_model")
    summary["workflows"].update(slow=stopped(), slower=stopped())
    run, why = regression_skip.decide(SHA, summary)
    assert not run and ("another run would not test either: missing_model, 2 untested "
                        "workflows whose job used up its time: another run can cost as much"
                        ) in why
    # One workflow that another run can test is reason enough, as ever.
    summary["workflows"]["flaky"] = {"verdict": "infra_error", "worker_status": "worker_error"}
    assert decide(summary=summary) and not decide(summary={**summary, "attempt": 2})
    # A download that failed for another reason is tried again, as it was.
    assert decide(summary=_untested("model_fetch_error"))
    for other in ("slow", None, True, ["timed_out"]):
        summary = _untested()
        summary["workflows"]["wf"] = stopped(reason=other)
        assert decide(summary=summary), other
    # The mark alone decides, whatever the status beside it. Only a cell of an
    # untested workflow is read.
    assert regression_skip.no_retry(stopped("infra_error"))
    assert regression_skip.no_retry({"worker_status": "missing_model"})
    assert not regression_skip.no_retry({"worker_status": "model_fetch_error"})
    assert not regression_skip.no_retry(None) and not regression_skip.used_up_its_time(None)
    assert regression_skip.TIMED_OUT == "timed_out"
    summary = _untested("worker_error")
    summary["workflows"]["sdxl_t2i"]["reason"] = "timed_out"
    assert decide(summary=summary)


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
# A scheduled run has no inputs, and an empty input reads as false: every run
# publishes, except a dispatch whose `publish` box is not ticked.
PUBLISHES = ("github.event_name != 'workflow_dispatch' || inputs.publish == true || "
             "inputs.publish == 'true'")


def test_a_dispatch_with_publish_off_tests_the_commit_and_writes_nothing():
    """A first look at a new workflow. The lane's result of the commit is no
    reason to skip it, nothing reaches the results branch, and no pull request
    gets a comment that would link to results nobody published."""
    text = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert re.search(r"\n      publish:\n        description: .+\n        type: boolean\n"
                     r"        default: true\n", text)
    steps = dict(workflow_steps(".github/workflows/regression.yml"))
    skip, suite = steps["Skip if already tested on this lane"], steps["Run regression suite"]
    for step in (skip, suite):
        assert "PUBLISH: ${{ " + PUBLISHES + " }}" in step
    off = 'if [ "$PUBLISH" != "true" ]; then args+='
    assert skip.index(off + "(--explicit); fi\n") < skip.index("python scripts/regression_skip.py")
    assert suite.index(off + "(--skip-publish); fi\n") < suite.index(
        'python scripts/run_regression.py "${args[@]}"\n')
    for arg in ('--commit "$COMMIT"', "--branch master", '--lane "$LANE"',
                '--workflows "$WORKFLOWS"'):
        assert arg in suite, arg
    assert ("if: always() && steps.target.outputs.skip != 'true' && (" + PUBLISHES + ")\n"
            ) in steps["PR comment (PULSE)"]
    # The report and the attached files do not depend on it.
    assert "inputs.publish" not in steps["Attach the run's files"]
    assert len([name for name, step in steps.items() if "inputs.publish" in step]) == 3


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


# -- the nightly and weekly workflow -----------------------------------------------------
PER_COMMIT_FILE = ".github/workflows/regression.yml"
LONG_FILE = ".github/workflows/nightly-weekly.yml"


def block(step: str, key: str) -> str:
    """What a step says under `key` (`run`, `env`, `with`): the rest of its
    line, or the indented lines below it, without their indentation."""
    lines = step.split("\n")
    at = next(i for i, line in enumerate(lines) if re.fullmatch(rf" +{key}:( .*)?", line))
    indent, value = re.fullmatch(rf"( +){key}:(?: (.*))?", lines[at]).groups()
    if value not in (None, "|"):
        return value
    body = []
    for line in lines[at + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip("\n")


def settings(step: str, key: str) -> str:
    """`block` without the comment lines: what the step sets, not why."""
    return "\n".join(line for line in block(step, key).split("\n")
                     if not line.lstrip().startswith("#"))


def lane_job_steps(path: str) -> list[tuple[str, str]]:
    """The named steps of the job that tests a lane, in order."""
    steps = workflow_steps(path)
    names = [name for name, _ in steps]
    first = names.index("Skip if already tested on this lane")
    return steps[first:names.index("Report", first) + 1]


def test_the_nightly_and_the_weekly_cron_are_told_apart_by_their_strings():
    """A cron run has no inputs: the plan knows the cadence only from the cron
    string that started the run, and the run's name in the Actions list from
    the same string. Three places, one test."""
    text = (ROOT / LONG_FILE).read_text(encoding="utf-8")
    crons = re.findall(r"^    - cron: '([^']+)'$", text, flags=re.M)
    assert crons == ["17 1 * * *", "17 3 * * 0"]
    plan = dict(workflow_steps(LONG_FILE))["plan"]
    assert block(plan, "env").split("\n") == [
        "CADENCE: ${{ inputs.cadence }}", "LANE: ${{ inputs.lane }}",
        "WORKFLOWS: ${{ inputs.workflows }}",
        "EVENT_SCHEDULE: ${{ github.event.schedule }}",
        f"NIGHTLY_CRON: '{crons[0]}'", f"WEEKLY_CRON: '{crons[1]}'"]
    script = block(plan, "run")
    nightly = 'if [ "$EVENT_SCHEDULE" = "$NIGHTLY_CRON" ]; then CADENCE=nightly; fi\n'
    weekly = 'if [ "$EVENT_SCHEDULE" = "$WEEKLY_CRON" ]; then CADENCE=weekly; fi\n'
    # The plan knows the workflows a dispatch names: a lane whose only workflow
    # is still switched off is tested when the dispatch names that workflow.
    matrix = ('matrix=$(python lanes.py cadence "${CADENCE:-nightly}" "${only[@]}" \\\n'
              '  --workflows "${WORKFLOWS:-all}")\n')
    assert script.index(nightly) < script.index(weekly) < script.index(matrix)
    # A cron string the plan does not know must not run as a nightly run by default.
    unknown = re.search(r'\nif \[ -n "\$EVENT_SCHEDULE" \] && \[ -z "\$CADENCE" \]; then\n'
                        r'  echo "::error::.+"\n  exit 1\nfi\n', script)
    assert unknown and script.index(weekly) < unknown.start() < script.index(matrix)
    assert 'if [ -n "$LANE" ]; then only=(--only "$LANE"); fi\n' in script
    assert script.endswith('echo "matrix=$matrix" >> "$GITHUB_OUTPUT"\n'
                           'echo "$matrix" | python -m json.tool')
    name = re.search(r"^run-name: (.+)$", text, flags=re.M).group(1)
    assert name == ("GPU regression · ${{ inputs.cadence || (github.event.schedule == "
                    f"'{crons[1]}' && 'weekly' || 'nightly') }}}}")
    # A dispatch says the cadence itself, and may only say one that has a matrix.
    assert re.search(r"\n      cadence:\n        description: .+\n        type: choice\n"
                     r"        options: \[nightly, weekly\]\n        default: nightly\n", text)
    assert lanes_mod.LONG_CADENCES == ("nightly", "weekly")
    # Neither is the per-commit cron, which stays where it is.
    per_commit = (ROOT / PER_COMMIT_FILE).read_text(encoding="utf-8")
    assert re.findall(r"^    - cron: '([^']+)'$", per_commit, flags=re.M) == ["*/5 * * * *"]
    assert "lanes.py cadence" not in per_commit and "lanes.py per-commit" not in text
    # The README tells people the same two times.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"| nightly | `{crons[0]}`, every day at 01:17 |" in readme
    assert f"| weekly | `{crons[1]}`, Sunday at 03:17 |" in readme


def test_nightly_and_weekly_runs_wait_in_a_group_of_their_own():
    """No per-commit tick replaces a waiting nightly or weekly run, the two
    cadences never run at once (they share one endpoint), and the run's page
    has its own name."""
    text = (ROOT / LONG_FILE).read_text(encoding="utf-8")
    assert "\nname: GPU regression (nightly and weekly)\n" in text
    assert ("\nconcurrency:\n  group: gpu-nightly-weekly\n  cancel-in-progress: false\n"
            "  queue: max\n") in text
    per_commit = (ROOT / PER_COMMIT_FILE).read_text(encoding="utf-8")
    assert "gpu-nightly-weekly" not in per_commit and "\nname: GPU regression\n" in per_commit
    assert "\npermissions:\n  contents: write\n" in text
    # The lane job: one per row of the plan, none for an empty plan, and each
    # with its lane's own time limit, which the plan's rows carry.
    assert ("  regression:\n    needs: lanes\n    if: needs.lanes.outputs.matrix != '[]'\n"
            "    runs-on: ubuntu-latest\n") in text
    assert "\n    timeout-minutes: ${{ matrix.timeout_minutes }}\n" in text
    assert "        include: ${{ fromJson(needs.lanes.outputs.matrix) }}\n" in text
    assert "    name: ${{ matrix.lane }} · ${{ needs.lanes.outputs.short }}\n" in text
    row = lanes_mod.cadence_matrix({"lanes": {"n": {"cadence": "nightly", "endpoint_id": "e"}}},
                                   "nightly")[0]
    for key in re.findall(r"\$\{\{ matrix\.(\w+) \}\}", text):
        assert key in row, key
    # No "All lanes" job and no PR comment: the per-commit pages carry both.
    assert re.findall(r"^  (\w+):$", text, flags=re.M) == ["schedule", "workflow_dispatch",
                                                           "lanes", "regression"]
    assert "notify_pr" not in text and "overview" not in text
    # Nor a report-only run: "GPU regression" shows any lane's published result.
    assert "inputs.report_only" not in text and "report_only:" not in text
    for name in ("commit", "workflows", "lane"):
        assert f"\n      {name}:\n        description: " in text
    assert re.search(r"\n      publish:\n        description: .+\n        type: boolean\n"
                     r"        default: true\n", text)
    # A lane report prints a command that dispatches this file to test one commit
    # of a range: the file's name and the inputs the command sets must exist.
    report = (ROOT / "scripts" / "run_report.py").read_text(encoding="utf-8")
    at = report.index(f"gh workflow run {LONG_FILE.rsplit('/', 1)[1]} --repo ")
    flags = re.findall(r"-f (\w+)=", report[at:at + 200])
    assert flags == ["cadence", "commit", "workflows"]
    for flag in flags:
        assert f"\n      {flag}:\n        description: " in text, flag
    # An edit of either workflow file alone runs these tests.
    tests = (ROOT / ".github" / "workflows" / "scripts-tests.yml").read_text(encoding="utf-8")
    for path in (PER_COMMIT_FILE, LONG_FILE):
        assert tests.count(f"'{path}'") == 2, path


def test_the_lane_job_of_the_nightly_and_weekly_workflow_is_that_of_the_per_commit_one():
    """The steps are copied, not shared through a reusable workflow (that
    would change the job names on every run page). Copies drift apart: this
    holds the two files to the same text."""
    ours, theirs = lane_job_steps(LONG_FILE), lane_job_steps(PER_COMMIT_FILE)
    assert [name for name, _ in theirs] == [
        "Skip if already tested on this lane", "Install the suite's packages",
        "Run regression suite", "Attach the run's files", "Report"]
    # One step more, before the suite and after the packages it needs.
    assert [name for name, _ in ours] == [
        "Skip if already tested on this lane", "Install the suite's packages",
        "Restore the endpoint's workers", "Run regression suite", "Attach the run's files",
        "Report"]
    ours, theirs = dict(ours), dict(theirs)
    for name in ("Skip if already tested on this lane", "Install the suite's packages", "Report"):
        assert block(ours[name], "run") == block(theirs[name], "run"), name
    for name in ("Install the suite's packages", "Report", "Attach the run's files"):
        for key in ("if", "env", "with", "id", "uses", "continue-on-error", "timeout-minutes"):
            if re.search(rf"\n +{key}:", theirs[name]):
                assert block(ours[name], key) == block(theirs[name], key), (name, key)
            else:
                assert not re.search(rf"\n +{key}:", ours[name]), (name, key)
    plan = dict(workflow_steps(LONG_FILE)), dict(workflow_steps(PER_COMMIT_FILE))
    for key in ("run", "env", "id"):
        assert block(plan[0]["Resolve target commit"], key) == block(
            plan[1]["Resolve target commit"], key), key
    # The skip step: the same but for report-only runs, which this file does not have.
    skip = "Skip if already tested on this lane"
    assert settings(ours[skip], "env") == settings(theirs[skip], "env").replace(
        REPORT_ONLY, "REPORT_ONLY: 'false'")
    assert "id: target" in ours[skip]
    # The suite: the script becomes the step's own process, so it is the one that
    # gets the signals GitHub sends when a run is stopped. Nothing may follow it.
    suite, base = block(ours["Run regression suite"], "run"), block(
        theirs["Run regression suite"], "run")
    command = 'python scripts/run_regression.py "${args[@]}"'
    assert suite == base.replace(command, "exec " + command) != base
    assert suite.endswith("\nexec " + command)
    # The lane names its endpoint: no secret names one for it.
    assert settings(ours["Run regression suite"], "env") == "\n".join(
        line for line in settings(theirs["Run regression suite"], "env").split("\n")
        if "RUNPOD_ENDPOINT_ID" not in line)
    assert "RUNPOD_ENDPOINT_ID" not in (ROOT / LONG_FILE).read_text(encoding="utf-8")


def test_the_workers_of_the_endpoint_are_restored_before_a_nightly_or_weekly_run():
    """RunPod sets an endpoint without requests to fewer workers, to none in
    the end. A week without a run would leave every job of the next run
    waiting in the queue."""
    steps = dict(lane_job_steps(LONG_FILE))
    restore = steps["Restore the endpoint's workers"]
    assert block(restore, "run") == (
        'python scripts/runpod_endpoints.py status --lane "$LANE" --restore-workers')
    assert block(restore, "env") == ("RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}\n"
                                     "LANE: ${{ matrix.lane }}")
    # Only a run that tests touches the endpoint or needs the key.
    tests = "steps.target.outputs.skip != 'true'"
    for name in ("Install the suite's packages", "Restore the endpoint's workers",
                 "Run regression suite"):
        assert block(steps[name], "if") == tests, name
    assert block(steps["Attach the run's files"], "if") == "always() && " + tests
    assert sorted(name for name, step in steps.items() if "RUNPOD_API_KEY" in step) == [
        "Restore the endpoint's workers", "Run regression suite"]
    # It is not a step that may fail quietly: a refused raise stops the run.
    assert "continue-on-error" not in restore
