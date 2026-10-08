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
LANE = "py312-torch2.11.0-cu128"


def _wf(verdict, detail=None, prior=None, **kw):
    return {"verdict": verdict, "detail": detail, "prior_verdict": prior,
            "vs_golden": {"identical": False, "mean_mse": 3.442175, "mean_psnr_db": 42.76,
                          "mean_pct_pixels_changed": 69.986} if verdict == "fail" else
            ({"identical": True, "mean_mse": 0.0} if verdict == "pass" else None),
            "timings": {"prompt_exec_s": 41.23}, "vram_peak_mb": 21504.0,
            "gpu_name": "NVIDIA GeForce RTX 4090", "torch_version": "2.11.0+cu128",
            "python_version": "3.12.3 (main, Apr 10 2026) [GCC 13.2.0]", **kw}


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
    assert "primary lane" in run(tmp_path, _summary(DRIFT, lane="py313-torch2.12-cu130"))
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
    lanes = {"primary": LANE, "lanes": {LANE: {"cuda": "12.8", "gpu": "RTX 4090"}}}
    body = notify_pr.render_changed(s, lanes)
    lines = body.splitlines()
    assert lines[0] == f"**(PULSE)** GPU output changed vs golden · `{SHA[:7]}`"
    assert "| `flux_dev_t2i` | changed | new | 42.76 | 3.442 | 69.99% | 41.2 | 21.0 GB |" in lines
    assert "| `sdxl_t2i` | match | – | identical | 0 | – | 41.2 | 21.0 GB |" in lines
    assert "| `wan` | error | new | – | – | – | 41.2 | 21.0 GB |" in lines
    assert ("Lane: python 3.12.3 · torch 2.11.0+cu128 · CUDA 12.8 · NVIDIA GeForce RTX 4090"
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
    assert "`zz_changed`" in body and "+40 more on the dashboard" in body
    assert body.startswith("**(PULSE)**")


def test_dashboard_link_names_a_non_primary_lane():
    lanes = {"primary": LANE, "lanes": {}}
    s = _summary(DRIFT, lane="py313")
    assert notify_pr.dashboard_url(s, lanes).endswith(f"/regression/master/{SHA}?lane=py313")
    assert notify_pr.dashboard_url(_summary(DRIFT), lanes).endswith(f"/regression/master/{SHA}")
