"""notify_pr: owner gate, posting policy, sticky comment and modes, against a
mocked GitHub API (no network)."""
import base64
import json
import re

import pytest
import requests

import notify_pr

SHA = "c" * 40
API = "https://api.github.com/repos/Comfy-Org/ComfyUI"
CODEOWNERS = ("# owners\n* @comfyanonymous @guill @alexisrolland @rattus128 @kijai @Comfy-Org/core\n"
              "\n/.github/ @comfyanonymous\n")
LANE = "py313-torch2.14.1-cu130"  # the primary lane in manifest/lanes.json


def _wf(verdict, detail=None, prior=None, **kw):
    return {"verdict": verdict, "detail": detail, "prior_verdict": prior,
            "vs_golden": {"identical": False, "mean_mse": 3.442175, "mean_psnr_db": 42.76,
                          "mean_pct_pixels_changed": 69.986} if verdict == "fail" else
            ({"identical": True, "mean_mse": 0.0} if verdict == "pass" else None),
            "timings": {"prompt_exec_s": 41.23}, "vram_peak_mb": 21504.0,
            "gpu_name": "NVIDIA GeForce RTX 4090", "torch_version": "2.14.1+cu130",
            "python_version": "3.13.11 (main, Dec 9 2025) [Clang 21.1.4]", **kw}


def _summary(workflows, pr=16488, author="comfyanonymous", rng=None, branch="master", lane=LANE):
    return {"schema_version": 2, "branch": branch, "commit": SHA, "lane": lane,
            "overall": "fail", "tested_range": rng,
            "commit_meta": {"subject": "Port some optimizations (#16488)", "pr": pr,
                            "pr_author": author,
                            "pr_url": f"https://github.com/Comfy-Org/ComfyUI/pull/{pr}"},
            "workflows": workflows}


DRIFT = {"flux_dev_t2i": _wf("fail", "new_drift"), "sdxl_t2i": _wf("pass")}


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeGitHub:
    """Just enough of the REST API: CODEOWNERS, commit pulls, the token's
    user and one PR's issue comments (paginated)."""

    def __init__(self, login="marawan206", codeowners=CODEOWNERS, comments=None,
                 pulls=None, fail=None):
        self.login = login
        self.codeowners = codeowners
        self.comments = comments if comments is not None else []
        self.pulls = pulls or []
        self.fail = fail or ()
        self.calls = []
        self.next_id = 1000

    def writes(self):
        return [c for c in self.calls if c[0] in ("POST", "PATCH")]

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append((method, url, params, json, (headers or {}).get("Authorization")))
        if any(f in url for f in self.fail):
            raise requests.ConnectionError("network down")
        path = url.removeprefix(API)
        if method == "GET" and url == "https://api.github.com/user":
            return Resp(200, {"login": self.login}) if self.login else Resp(403, {})
        if method == "GET" and path.startswith("/contents/"):
            if path == "/contents/CODEOWNERS" and self.codeowners is not None:
                enc = base64.b64encode(self.codeowners.encode()).decode()
                return Resp(200, {"content": enc, "encoding": "base64"})
            return Resp(404, {"message": "Not Found"})
        if method == "GET" and path == f"/commits/{SHA}/pulls":
            return Resp(200, self.pulls)
        m = re.fullmatch(r"/issues/(\d+)/comments", path)
        if m and method == "GET":
            page, per = params["page"], params["per_page"]
            return Resp(200, self.comments[(page - 1) * per: page * per])
        if m and method == "POST":
            self.next_id += 1
            c = {"id": self.next_id, "body": json["body"], "user": {"login": self.login}}
            self.comments.append(c)
            return Resp(201, c)
        m = re.fullmatch(r"/issues/comments/(\d+)", path)
        if m and method == "PATCH":
            c = next(c for c in self.comments if c["id"] == int(m.group(1)))
            c["body"] = json["body"]
            return Resp(200, c)
        return Resp(404, {"message": "Not Found"})


@pytest.fixture
def gh(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(notify_pr.requests, "request", fake.request)
    return fake


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("PULSE_PR_COMMENTS", "PULSE_GH_TOKEN", "PULSE_ALLOWED_AUTHORS",
              "PULSE_DASHBOARD_URL", "GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    step = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    monkeypatch.setenv("GITHUB_TOKEN", "read-token")
    return step


def on(monkeypatch):
    monkeypatch.setenv("PULSE_PR_COMMENTS", "on")
    monkeypatch.setenv("PULSE_GH_TOKEN", "pat")


def run(tmp_path, summary):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary), encoding="utf-8")
    return notify_pr.run(p)


# -- owner gate ---------------------------------------------------------------------
def test_default_owners_reads_the_star_rule_and_ignores_teams():
    assert notify_pr.default_owners(CODEOWNERS) == [
        "comfyanonymous", "guill", "alexisrolland", "rattus128", "kijai"]
    assert notify_pr.default_owners("* @a\n*.py @b\n* @c @org/team  # later rule wins\n") == ["c"]
    assert notify_pr.default_owners("/docs/ @a\n") == []


def test_posts_for_a_code_owner(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    assert run(tmp_path, _summary(DRIFT)) == "created on #16488"
    (method, url, _, payload, auth), = gh.writes()
    assert method == "POST" and url == f"{API}/issues/16488/comments"
    assert auth == "Bearer pat"
    assert payload["body"].startswith("**(PULSE)**")


def test_skips_non_owners_and_team_names(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    assert "not a code owner" in run(tmp_path, _summary(DRIFT, author="someone"))
    # '@Comfy-Org/core' is a team entry, never a login.
    assert "not a code owner" in run(tmp_path, _summary(DRIFT, author="Comfy-Org/core"))
    assert gh.writes() == []


def test_allowed_authors_override_replaces_codeowners(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    monkeypatch.setenv("PULSE_ALLOWED_AUTHORS", "@someone, other")
    assert run(tmp_path, _summary(DRIFT, author="Someone")) == "created on #16488"
    assert "not a code owner" in run(tmp_path, _summary(DRIFT, author="comfyanonymous"))
    assert not any("/contents/" in c[1] for c in gh.calls)


def test_no_codeowners_and_no_override_never_posts(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    gh.codeowners = None
    assert run(tmp_path, _summary(DRIFT)).startswith("skip: CODEOWNERS unavailable")
    assert gh.writes() == []


# -- policy -------------------------------------------------------------------------
@pytest.mark.parametrize("workflows, expected", [
    ({"a": _wf("fail", "new_drift")}, "changed"),
    ({"a": _wf("execution_error", prior="pass")}, "changed"),
    ({"a": _wf("execution_error", prior="fail")}, "changed"),
    ({"a": _wf("execution_error", prior=None)}, None),               # new workflow
    ({"a": _wf("execution_error", prior="execution_error")}, None),   # standing error
    ({"a": _wf("fail", "inherited")}, None),
    ({"a": _wf("fail", None)}, None),                                 # unclassified
    ({"a": _wf("infra_error")}, None),
    ({"a": _wf("pass"), "b": _wf("infra_error")}, None),
    ({"a": _wf("pass"), "b": _wf("no_baseline")}, "clean"),
    ({}, None),
])
def test_decide(workflows, expected):
    assert notify_pr.decide({"workflows": workflows}) == expected


@pytest.mark.parametrize("workflows", [
    {"a": _wf("fail", "inherited")}, {"a": _wf("pass")}, {"a": _wf("infra_error")},
    {"a": _wf("no_baseline")}, {"a": _wf("execution_error", prior="execution_error")},
    {"a": _wf("execution_error", prior=None)}])
def test_quiet_runs_never_create_a_comment(tmp_path, monkeypatch, env, gh, workflows):
    on(monkeypatch)
    assert run(tmp_path, _summary(workflows)).startswith("skip")
    assert gh.writes() == []


@pytest.mark.parametrize("rng", [
    {"prev": "b" * 40, "commits_between": 0, "compare_url": "u"},      # older commit re-tested
    {"prev": "b" * 40, "commits_between": None, "compare_url": "u"}])  # compare unavailable
def test_drift_on_a_commit_behind_the_previous_run_is_not_reported(tmp_path, monkeypatch, env,
                                                                   gh, rng):
    on(monkeypatch)
    assert run(tmp_path, _summary(DRIFT, rng=rng)) == (
        "skip: the tested commit is not ahead of the previous run")
    assert gh.writes() == []
    ok = {**rng, "commits_between": 1}
    assert run(tmp_path, _summary(DRIFT, rng=ok)) == "created on #16488"


def test_drift_in_a_batch_of_commits_is_not_reported(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    rng = {"prev": "b" * 40, "commits_between": 3, "compare_url": "u"}
    assert run(tmp_path, _summary(DRIFT, rng=rng)) == (
        "skip: batch of 3 commits, not attributable to one PR")
    assert gh.writes() == []


def test_clean_rerun_resolves_an_existing_comment(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    run(tmp_path, _summary(DRIFT))
    clean = {"flux_dev_t2i": _wf("pass"), "sdxl_t2i": _wf("pass")}
    assert run(tmp_path, _summary(clean)) == "updated on #16488"
    assert len(gh.comments) == 1
    body = gh.comments[0]["body"]
    assert body.startswith("**(PULSE)** No GPU output changes vs golden")
    assert notify_pr.MARKER in body
    # Idempotent: the same clean result again changes nothing.
    assert run(tmp_path, _summary(clean)).startswith("unchanged")


def test_other_branches_and_lanes_are_skipped(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    assert "not master" in run(tmp_path, _summary(DRIFT, branch="release"))
    assert "primary lane" in run(tmp_path, _summary(DRIFT, lane="py313-torch2.15.0-cu132"))
    assert gh.calls == []


# -- sticky comment -----------------------------------------------------------------
def test_rerun_updates_instead_of_duplicating(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    assert run(tmp_path, _summary(DRIFT)) == "created on #16488"
    assert run(tmp_path, _summary(DRIFT)).startswith("unchanged")
    more = {**DRIFT, "sdxl_t2i": _wf("fail", "new_drift")}
    assert run(tmp_path, _summary(more)) == "updated on #16488"
    assert [c[0] for c in gh.writes()] == ["POST", "PATCH"]
    assert len(gh.comments) == 1


def test_only_the_token_owners_marker_comment_is_reused(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    gh.comments.append({"id": 1, "body": f"quoted {notify_pr.MARKER}",
                        "user": {"login": "someone-else"}})
    assert run(tmp_path, _summary(DRIFT)) == "created on #16488"
    assert gh.comments[0]["body"] == f"quoted {notify_pr.MARKER}"


def test_sticky_comment_is_found_past_the_first_page(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    gh.comments.extend({"id": i, "body": "lgtm", "user": {"login": "x"}} for i in range(1, 151))
    gh.comments.append({"id": 151, "body": f"old {notify_pr.MARKER}",
                        "user": {"login": "Marawan206"}})
    assert run(tmp_path, _summary(DRIFT)) == "updated on #16488"
    pages = [c[2]["page"] for c in gh.calls if c[0] == "GET" and c[1].endswith("/comments")]
    assert pages == [1, 2]
    assert gh.comments[-1]["body"].startswith("**(PULSE)**")


def test_pr_is_looked_up_when_the_summary_lacks_the_author(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    gh.pulls = [{"number": 1, "merged_at": None, "base": {"ref": "master"}, "user": {"login": "kijai"}},
                {"number": 77, "merged_at": "2026-10-01T00:00:00Z", "base": {"ref": "master"},
                 "merge_commit_sha": SHA, "user": {"login": "kijai"},
                 "html_url": "https://github.com/Comfy-Org/ComfyUI/pull/77"}]
    s = _summary(DRIFT, author=None)
    assert run(tmp_path, s) == "created on #77"
    gh.pulls = []
    assert run(tmp_path, s) == "skip: no merged PR for this commit"


# -- modes ------------------------------------------------------------------------
def test_dry_run_is_the_default_and_writes_only_the_step_summary(tmp_path, env, gh, capsys):
    assert run(tmp_path, _summary(DRIFT)) == "dry-run: would create on #16488"
    assert gh.writes() == []
    report = env.read_text(encoding="utf-8")
    assert "Would create a comment on [Comfy-Org/ComfyUI#16488]" in report
    assert "**(PULSE)** GPU output changed vs golden" in report
    assert "**(PULSE)**" in capsys.readouterr().out


def test_on_without_token_falls_back_to_dry_run(tmp_path, monkeypatch, env, gh, capsys):
    monkeypatch.setenv("PULSE_PR_COMMENTS", "on")
    assert run(tmp_path, _summary(DRIFT)).startswith("dry-run")
    assert gh.writes() == []
    assert "PULSE_GH_TOKEN is not set" in capsys.readouterr().out


def test_off_does_nothing(tmp_path, monkeypatch, env, gh):
    monkeypatch.setenv("PULSE_PR_COMMENTS", "off")
    assert run(tmp_path, _summary(DRIFT)) == "off"
    assert gh.calls == [] and not env.exists()


def test_network_errors_never_fail_the_job(tmp_path, monkeypatch, env, gh, capsys):
    on(monkeypatch)
    gh.fail = ("/issues/",)
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(_summary(DRIFT)), encoding="utf-8")
    assert notify_pr.main(["--summary", str(p)]) == 0
    assert "::warning::PULSE" in capsys.readouterr().out
    assert gh.writes() == []
    # A missing summary (the run step died early) is not an error either.
    assert notify_pr.main(["--summary", str(tmp_path / "missing.json")]) == 0


def test_unknown_token_owner_never_posts(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    gh.login = None
    assert notify_pr.main(["--summary", str(_write(tmp_path, _summary(DRIFT)))]) == 0
    assert gh.writes() == []


def _write(tmp_path, summary):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(summary), encoding="utf-8")
    return p


# -- body -------------------------------------------------------------------------
def test_body_shape():
    rng = {"prev": "b" * 40, "commits_between": 3,
           "compare_url": "https://github.com/Comfy-Org/ComfyUI/compare/bbb...ccc"}
    s = _summary({**DRIFT, "wan": _wf("execution_error", prior="pass", vs_golden=None)}, rng=rng)
    lanes = {"primary": LANE, "lanes": {LANE: {"cuda": "13.0", "gpu": "RTX 4090"}}}
    body = notify_pr.render_changed(s, lanes)
    lines = body.splitlines()
    assert lines[0] == f"**(PULSE)** GPU output changed vs golden · `{SHA[:7]}`"
    assert "| `flux_dev_t2i` | changed | new | 42.76 | 3.442 | 69.99% | 41.2 | 21.0 GB |" in lines
    assert "| `sdxl_t2i` | match | – | identical | 0 | – | 41.2 | 21.0 GB |" in lines
    assert "| `wan` | error | new | – | – | – | 41.2 | 21.0 GB |" in lines
    assert ("Lane: python 3.13.11 · torch 2.14.1+cu130 · CUDA 13.0 · NVIDIA GeForce RTX 4090"
            in lines)
    assert f"Appeared in a batch of 3 commits: [compare]({rng['compare_url']})" in lines
    assert f"[Run details](https://ci.comfy.org/regression/master/{SHA})" in lines
    assert "Non-blocking GPU regression check; this comment updates in place." in body
    assert body.rstrip().endswith(notify_pr.MARKER)
    assert len(body.encode()) < notify_pr.MAX_BODY
    for b in (body, notify_pr.render_resolved(s, lanes)):
        assert not re.search(r"claude|anthropic|\bAI\b|\bbot\b", b, re.I)
        # Neutral wording: "output changed", never "regression"/"broken" (the
        # footer, the dashboard path and the hidden marker aside).
        visible = b.replace("GPU regression check", "").replace(notify_pr.MARKER, "")
        assert not re.search(r"regression\b(?!/)|broken", visible, re.I)
    # A single tested commit has no batch line.
    assert "batch" not in notify_pr.render_changed(_summary(DRIFT), lanes)


def test_body_stays_bounded_with_many_workflows():
    wfs = {f"workflow_{i:02d}_with_a_long_name": _wf("pass") for i in range(40)}
    wfs["zz_changed"] = _wf("fail", "new_drift")
    body = notify_pr.render_changed(_summary(wfs), None)
    assert len(body) <= notify_pr.MAX_BODY
    assert "`zz_changed`" in body and "+40 more in the run details" in body
    assert body.startswith("**(PULSE)**")


def test_dashboard_link_names_a_non_primary_lane():
    lanes = {"primary": LANE, "lanes": {}}
    s = _summary(DRIFT, lane="py313")
    assert notify_pr.dashboard_url(s, lanes).endswith(f"/regression/master/{SHA}?lane=py313")
    assert notify_pr.dashboard_url(_summary(DRIFT), lanes).endswith(f"/regression/master/{SHA}")


RUN_URL = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs/43658104127"


def test_run_details_link_to_the_actions_run_that_tested_the_commit():
    """The run's page holds the full report, so the comment points there. The
    comment outlives the run (GitHub deletes it after 90 days): the commit's
    page on the dashboard is linked next to it."""
    lanes = {"primary": LANE, "lanes": {}}
    s = {**_summary(DRIFT), "run_url": RUN_URL, "run_id": 43658104127}
    page = f"https://ci.comfy.org/regression/master/{SHA}"
    for body in (notify_pr.render_changed(s, lanes), notify_pr.render_resolved(s, lanes)):
        assert f"[Run details]({RUN_URL}) · [results page]({page})" in body.splitlines()
        assert len(body) < notify_pr.MAX_BODY
    # A lane that is not the primary keeps its ?lane= on the page link.
    other = {**s, "lane": "py313"}
    assert f"· [results page]({page}?lane=py313)" in notify_pr.render_changed(other, lanes)


@pytest.mark.parametrize("run_url", [
    None, "", 42, "https://evil.example/actions/runs/1",
    # The right shape on another host or scheme: the host is pinned.
    "https://evil.example/a/b/actions/runs/1", "https://github.com.evil.example/a/b/actions/runs/1",
    "https://github.com@evil.example/a/b/actions/runs/1", "http://github.com/a/b/actions/runs/1",
    "https://github.com/a/b/actions/runs/1) [x](https://evil.example",
    "javascript:alert(1)"])
def test_run_details_fall_back_to_the_dashboard(run_url):
    """A summary from before the run link existed, or one whose link is not a
    plain Actions run URL, has the dashboard link alone."""
    lanes = {"primary": LANE, "lanes": {}}
    s = {**_summary(DRIFT), "run_url": run_url}
    assert notify_pr.details_url(s, lanes) == f"https://ci.comfy.org/regression/master/{SHA}"
    body = notify_pr.render_changed(s, lanes)
    assert f"[Run details](https://ci.comfy.org/regression/master/{SHA})" in body.splitlines()
    assert "evil" not in body and "results page" not in body


def test_posted_comment_carries_the_run_link(tmp_path, monkeypatch, env, gh):
    on(monkeypatch)
    assert run(tmp_path, {**_summary(DRIFT), "run_url": RUN_URL}) == "created on #16488"
    (_, _, _, payload, _), = gh.writes()
    assert f"[Run details]({RUN_URL})" in payload["body"]


# -- text workflows -------------------------------------------------------------------
# What a model could write: a mention, a link, markup. None of it may reach a comment.
NONE = notify_pr._fmt(None, ".1f")  # how the table shows a number a workflow does not have
HOSTILE = "@comfyanonymous see [this](https://evil.example) <b>now</b>"


def _text(verdict="fail", detail="new_drift", **vs_golden):
    vg = {"kind": "text", "files_compared": 1, "file_count_ref": 1, "file_count_cand": 1,
          "identical": False, "similarity": 0.864865, "similarity_basis": "words",
          "chars_ref": 208, "chars_cand": 209, "lines_ref": 3, "lines_cand": 3,
          "first_diff": {"file": "llm_00001.txt", "line": 2, "col": 22, "offset": 108,
                         "ref": "the reference line", "cand": HOSTILE}, **vs_golden}
    return _wf(verdict, detail, vs_golden=vg, output_kind="text", thresholds_used=None)


def test_a_new_text_change_adds_one_line_of_numbers_under_the_table():
    s = _summary({**DRIFT, "llm_qwen_t2t": _text()})
    lines = notify_pr.render_changed(s, None).splitlines()
    assert notify_pr.decide(s) == "changed"
    # The row is the one every workflow gets: a text has no PSNR, MSE or pixels.
    assert (f"| `llm_qwen_t2t` | changed | new | {NONE} | {NONE} | {NONE} | 41.2 | 21.0 GB |"
            in lines)
    note = ("- Text `llm_qwen_t2t`: 86.5% similar to the golden, first difference at line 2, "
            "column 22")
    at = lines.index(note)
    assert lines[at - 1] == "" and lines[at - 2].startswith("| `sdxl_t2i` |")
    assert lines[at + 1] == "" and lines[at + 2].startswith("Lane: ")
    # A text that matches reads like any match.
    same = _wf("pass", vs_golden={"kind": "text", "identical": True, "similarity": 1.0},
               output_kind="text")
    body = notify_pr.render_changed(_summary({**DRIFT, "llm_qwen_t2t": same}), None)
    assert (f"| `llm_qwen_t2t` | match | {NONE} | identical | {NONE} | {NONE} | 41.2 | 21.0 GB |"
            in body)
    assert "- Text" not in body


def test_generated_text_never_enters_a_comment(tmp_path, monkeypatch, env, gh):
    """The comment is posted on ComfyUI from a person's account, and model
    output could carry mentions or links: only numbers are taken from it."""
    hostile = _text(similarity=HOSTILE, error=None)
    hostile["vs_golden"]["first_diff"].update(line=HOSTILE, col={"x": HOSTILE}, file=HOSTILE)
    s = _summary({"llm_a": _text(), "llm_b": hostile})
    for body in (notify_pr.render_changed(s, None), notify_pr.render_resolved(s, None)):
        assert "evil" not in body and "comfyanonymous" not in body and "<b>" not in body
        assert "the reference line" not in body
    # Values that are not plain numbers are dropped, and a line with nothing left is too.
    lines = notify_pr.render_changed(s, None).splitlines()
    assert [line for line in lines if line.startswith("- Text")] == [
        "- Text `llm_a`: 86.5% similar to the golden, first difference at line 2, column 22"]
    for sim in (True, float("nan"), float("inf"), -1, 2, "0.5", None):
        odd = _text(similarity=sim)
        odd["vs_golden"]["first_diff"].update(line=0, col=True)
        assert notify_pr.text_notes({"x": odd}, {"x": "new_drift"}) == []
    on(monkeypatch)
    assert run(tmp_path, s) == "created on #16488"
    (_, _, _, payload, _), = gh.writes()
    assert "evil" not in payload["body"] and "comfyanonymous" not in payload["body"]


def test_which_text_changes_get_a_line():
    kinds = {"a": "new_drift", "b": "new_drift", "c": "new_drift", "d": "new_drift"}
    wfs = {wf: _text() for wf in kinds}
    # At most three lines: the table and the run report hold the rest.
    assert len(notify_pr.text_notes(wfs, kinds)) == 3
    assert notify_pr.text_notes(wfs, kinds)[0].startswith("- Text `a`: ")
    # Not for an inherited change, an execution error, a comparison that failed or an image.
    assert notify_pr.text_notes({"a": _text(detail="inherited")}, {"a": None}) == []
    assert notify_pr.text_notes({"a": _text()}, {"a": "new_error"}) == []
    assert notify_pr.text_notes({"a": _text(error="no text files to compare")},
                                {"a": "new_drift"}) == []
    assert notify_pr.text_notes(DRIFT, {"flux_dev_t2i": "new_drift", "sdxl_t2i": None}) == []
    assert notify_pr.text_notes({"a": {"vs_golden": "nope"}}, {"a": "new_drift"}) == []
    # A blank-only change never reads as fully alike; a missing position is left out.
    near = _text(similarity=0.999999)
    del near["vs_golden"]["first_diff"]
    assert notify_pr.text_notes({"a": near}, {"a": "new_drift"}) == [
        "- Text `a`: over 99.9% similar to the golden"]
    # A similarity that is not a share of words says what it is. The basis is read
    # from results: one that is not known adds nothing.
    assert notify_pr.text_notes({"a": _text(similarity_basis="lines")}, {"a": "new_drift"}) == [
        "- Text `a`: 86.5% similar to the golden by lines, first difference at line 2, column 22"]
    assert notify_pr.text_notes({"a": _text(similarity=0.0, similarity_basis="prefix")},
                                {"a": "new_drift"})[0].startswith(
        "- Text `a`: 0.0% similar to the golden by the shared start, first")
    for odd in (HOSTILE, ["lines"], None):
        assert notify_pr.text_notes({"a": _text(similarity_basis=odd)}, {"a": "new_drift"}) == [
            "- Text `a`: 86.5% similar to the golden, first difference at line 2, column 22"]
    # No file with the golden's name: nothing was compared, so no percentage is given.
    moved = _text(similarity=0.0, files_compared=0)
    del moved["vs_golden"]["first_diff"]
    assert notify_pr.text_notes({"a": moved}, {"a": "new_drift"}) == [
        "- Text `a`: its text file has another name than the golden's"]
    # An inherited text failure next to a new image change: the table says so, no line.
    body = notify_pr.render_changed(_summary({**DRIFT, "llm": _text(detail="inherited")}), None)
    assert f"| `llm` | changed | inherited | {NONE} | {NONE} | {NONE} | 41.2 | 21.0 GB |" in body
    assert "- Text" not in body


def test_a_summary_without_text_renders_the_body_it_always_did():
    """The tail of a comment starts with one empty line, as before text notes."""
    rng = {"prev": "b" * 40, "commits_between": 1, "compare_url": "u"}
    lanes = {"primary": LANE, "lanes": {LANE: {"cuda": "13.0", "gpu": "RTX 4090"}}}
    body = notify_pr.render_changed(_summary(DRIFT, rng=rng), lanes)
    assert body == "\n".join([
        f"**(PULSE)** GPU output changed vs golden · `{SHA[:7]}`",
        "",
        "| Workflow | Verdict | New / inherited | PSNR dB | Mean MSE | Pixels changed "
        "| Exec s | Peak VRAM |",
        "|---|---|---|---|---|---|---|---|",
        "| `flux_dev_t2i` | changed | new | 42.76 | 3.442 | 69.99% | 41.2 | 21.0 GB |",
        f"| `sdxl_t2i` | match | {NONE} | identical | 0 | {NONE} | 41.2 | 21.0 GB |",
        "",
        "Lane: python 3.13.11 · torch 2.14.1+cu130 · CUDA 13.0 · NVIDIA GeForce RTX 4090",
        f"[Run details](https://ci.comfy.org/regression/master/{SHA})",
        "",
        notify_pr.FOOTER,
        notify_pr.MARKER])


# -- smoke-level workflows ------------------------------------------------------------
def _smoke(verdict="fail", prior="pass", **kw):
    """The cell of a smoke-level workflow: nothing is compared, `smoke` holds
    what its checks found."""
    problems = [{"file": "music_00001.flac", "check": "silent",
                 "text": f"silent or constant {HOSTILE}"}] if verdict == "fail" else []
    return _wf(verdict, None, prior, vs_golden=None, level="smoke",
               smoke={"files": [{"name": HOSTILE, "type": "audio", "bytes": 11040}],
                      "problems": problems, "notes": [HOSTILE]}, **kw)


def test_a_new_smoke_failure_gives_one_row_that_says_smoke_check_failed():
    s = _summary({"music": _smoke(), "sdxl_t2i": _wf("pass"), "mesh": _smoke("pass")})
    assert notify_pr.change_kind(s["workflows"]["music"]) == "new_error"
    assert notify_pr.decide(s) == "changed"
    lines = notify_pr.render_changed(s, None).splitlines()
    # Neither a changed output nor an execution error: the headline says what it is.
    assert lines[0] == f"**(PULSE)** GPU workflow failed · `{SHA[:7]}`"
    # Nothing was compared, so the three metric columns hold the mark of a missing number.
    assert [line for line in lines if "music" in line] == [
        f"| `music` | smoke check failed | new | {NONE} | {NONE} | {NONE} | 41.2 | 21.0 GB |"]
    assert f"| `mesh` | smoke pass | {NONE} | {NONE} | {NONE} | {NONE} | 41.2 | 21.0 GB |" in lines
    assert f"| `sdxl_t2i` | match | {NONE} | identical | 0 | {NONE} | 41.2 | 21.0 GB |" in lines
    # File names and problem sentences come from the worker: none enters a comment.
    body = "\n".join(lines)
    assert "evil" not in body and "comfyanonymous" not in body and "silent" not in body
    assert "- Text" not in body


def test_which_smoke_results_are_worth_a_comment(tmp_path, monkeypatch, env, gh):
    kind = notify_pr.change_kind
    # Only a failed check where the checks passed the time before.
    assert kind(_smoke("fail", "pass")) == "new_error"
    for prior in ("fail", "execution_error", None):
        assert kind(_smoke("fail", prior)) is None
    assert kind(_smoke("pass", "fail")) is None and kind(_smoke("infra_error")) is None
    # `detail` is for compared outputs: a smoke cell that carried one would not count as drift.
    assert kind({**_smoke("fail", "fail"), "detail": "new_drift"}) is None
    # A smoke workflow that does not finish is an execution error like any other.
    assert kind(_smoke("execution_error", "pass")) == "new_error"
    assert notify_pr.decide(_summary({"music": _smoke("fail", "fail")})) is None
    assert notify_pr.decide(_summary({"music": _smoke("pass"), "x": _wf("pass")})) == "clean"
    # A standing smoke failure next to a new change reads "inherited", as a standing
    # reference failure does.
    body = notify_pr.render_changed(_summary({**DRIFT, "music": _smoke("fail", "fail")}), None)
    assert body.splitlines()[0] == f"**(PULSE)** GPU output changed vs golden · `{SHA[:7]}`"
    assert (f"| `music` | smoke check failed | inherited | {NONE} | {NONE} | {NONE} | 41.2 "
            "| 21.0 GB |") in body
    # Next to a new execution error of a reference workflow the headline covers both.
    both = {"music": _smoke(), "wan": _wf("execution_error", prior="pass", vs_golden=None)}
    assert notify_pr.render_changed(_summary(both), None).startswith(
        "**(PULSE)** GPU workflow failed · ")
    # A smoke workflow that merely passes, or already failed, leaves the headline alone.
    quiet = {"mesh": _smoke("pass"), "music": _smoke("fail", "fail"),
             "wan": _wf("execution_error", prior="pass", vs_golden=None)}
    assert notify_pr.render_changed(_summary(quiet), None).startswith(
        "**(PULSE)** GPU workflow execution error · ")
    assert f"| `wan` | error | new | {NONE} |" in notify_pr.render_changed(_summary(quiet), None)
    # Posted like any other change, and resolved by a clean run of the same commit.
    on(monkeypatch)
    assert run(tmp_path, _summary({"music": _smoke(), "sdxl_t2i": _wf("pass")})) == (
        "created on #16488")
    assert run(tmp_path, _summary({"music": _smoke("pass"), "sdxl_t2i": _wf("pass")})) == (
        "updated on #16488")
    body = gh.writes()[-1][3]["body"]
    assert (f"Re-checked at `{SHA[:7]}`: all 2 workflows pass: 1 compared with the golden (a "
            "match, or no golden yet) and 1 checked at smoke level. The earlier report on this "
            "PR no longer applies.") in body.splitlines()
    only = notify_pr.render_resolved(_summary({"a": _smoke("pass"), "b": _smoke("pass")}), None)
    assert f"Re-checked at `{SHA[:7]}`: all 2 workflows passed their smoke checks. The" in only


def test_a_body_without_smoke_cells_is_byte_for_byte_what_it_was():
    """Headline included: an execution error keeps its own, and a clean run
    its resolved note."""
    lanes = {"primary": LANE, "lanes": {LANE: {"cuda": "13.0", "gpu": "RTX 4090"}}}
    wfs = {"sdxl_t2i": _wf("pass"), "wan": _wf("execution_error", prior="pass", vs_golden=None),
           "old": _wf("fail", "inherited")}
    tail = ["",
            "Lane: python 3.13.11 · torch 2.14.1+cu130 · CUDA 13.0 · NVIDIA GeForce RTX 4090",
            f"[Run details](https://ci.comfy.org/regression/master/{SHA})", "", notify_pr.FOOTER,
            notify_pr.MARKER]
    assert notify_pr.render_changed(_summary(wfs), lanes) == "\n".join([
        f"**(PULSE)** GPU workflow execution error · `{SHA[:7]}`",
        "",
        "| Workflow | Verdict | New / inherited | PSNR dB | Mean MSE | Pixels changed "
        "| Exec s | Peak VRAM |",
        "|---|---|---|---|---|---|---|---|",
        "| `old` | changed | inherited | 42.76 | 3.442 | 69.99% | 41.2 | 21.0 GB |",
        f"| `sdxl_t2i` | match | {NONE} | identical | 0 | {NONE} | 41.2 | 21.0 GB |",
        f"| `wan` | error | new | {NONE} | {NONE} | {NONE} | 41.2 | 21.0 GB |",
        *tail])
    assert notify_pr.render_resolved(_summary({"sdxl_t2i": _wf("pass"), "x": _wf("no_baseline")}),
                                     lanes) == "\n".join([
        f"**(PULSE)** No GPU output changes vs golden · `{SHA[:7]}`",
        "",
        f"Re-checked at `{SHA[:7]}`: all 2 workflows match the golden (or have no golden yet). "
        "The earlier report on this PR no longer applies.",
        *tail])
    # A cell that names the reference level, or a level nobody knows, is no smoke cell.
    for level in ("reference", None, ["smoke"], "SMOKE"):
        cell = _wf("fail", "new_drift", level=level)
        assert notify_pr.change_kind(cell) == "new_drift"
        assert notify_pr._row("x", cell).startswith("| `x` | changed | new | 42.76 |")
