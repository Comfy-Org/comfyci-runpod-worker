"""commit_meta: commit + pull request lookup against a mocked GitHub API."""
import requests

import commit_meta
import index_builder

SHA = "58b176f0c8427b1e3b1e0174ded3a7794641388b"
COMMIT = {"sha": SHA, "author": {"login": "kijai"},
          "commit": {"message": "Use PNG compress level 4 (#16867)\n\nbody",
                     "author": {"name": "Jukka"},
                     "committer": {"date": "2026-10-08T12:33:27Z"}},
          "parents": [{"sha": "p" * 40}]}


def _pr(number, merged_at="2026-10-08T12:33:27Z", base="master", merge_sha=SHA, **kw):
    return {"number": number, "title": f"PR {number}", "merged_at": merged_at,
            "base": {"ref": base}, "merge_commit_sha": merge_sha,
            "user": {"login": "kijai", "avatar_url": "https://avatars.example/u/1"},
            "labels": [{"name": "Core"}], "html_url": f"https://github.com/x/pull/{number}",
            **kw}


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.headers = {}

    def json(self):
        return self._body


def mock_api(monkeypatch, routes):
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(url)
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return Resp(404, {"message": "Not Found"})
    monkeypatch.setattr(commit_meta.requests, "get", get)
    return calls


def test_merged_pr_prefers_the_merge_commit_and_ignores_open_or_other_base():
    open_pr = _pr(1, merged_at=None)
    other_base = _pr(2, base="release")
    other_merge = _pr(3, merge_sha="x" * 40)
    exact = _pr(4)
    assert commit_meta.merged_pr([open_pr, other_base, other_merge, exact], SHA)["number"] == 4
    assert commit_meta.merged_pr([open_pr, other_merge], SHA)["number"] == 3
    assert commit_meta.merged_pr([open_pr, other_base], SHA) is None
    assert commit_meta.merged_pr(None, SHA) is None
    assert commit_meta.merged_pr({"message": "Not Found"}, SHA) is None


def test_fetch_commit_meta_adds_pr_details(monkeypatch):
    calls = mock_api(monkeypatch, {f"/commits/{SHA}": Resp(200, COMMIT),
                                   f"/commits/{SHA}/pulls": Resp(200, [_pr(16867)])})
    meta = commit_meta.fetch_commit_meta(SHA)
    assert calls[-1].endswith(f"/repos/Comfy-Org/ComfyUI/commits/{SHA}/pulls")
    assert meta["subject"] == "Use PNG compress level 4 (#16867)"
    assert meta["author"] == "kijai" and meta["parents"] == ["p" * 40]
    assert meta["pr"] == 16867 and meta["pr_title"] == "PR 16867"
    assert meta["pr_author"] == "kijai" and meta["pr_avatar"] == "https://avatars.example/u/1"
    assert meta["pr_labels"] == ["Core"]
    assert meta["pr_merged_at"] == meta["committed_ts"] == 1791462807
    assert meta["pr_url"] == "https://github.com/x/pull/16867"


def test_fetch_commit_meta_falls_back_to_the_subject_when_pulls_fail(monkeypatch, capsys):
    mock_api(monkeypatch, {f"/commits/{SHA}": Resp(200, COMMIT),
                           f"/commits/{SHA}/pulls": requests.ConnectionError("down")})
    meta = commit_meta.fetch_commit_meta(SHA)
    assert meta["pr"] == 16867
    assert meta["pr_title"] is meta["pr_author"] is meta["pr_url"] is None
    assert meta["pr_labels"] == []
    assert "::warning::" in capsys.readouterr().out

    # A direct push: no PR in the listing and no suffix in the subject.
    direct = {**COMMIT, "commit": {**COMMIT["commit"], "message": "Direct push"}}
    mock_api(monkeypatch, {f"/commits/{SHA}": Resp(200, direct),
                           f"/commits/{SHA}/pulls": Resp(200, [])})
    assert commit_meta.fetch_commit_meta(SHA)["pr"] is None


def test_fetch_commit_meta_is_none_when_the_commit_lookup_fails(monkeypatch):
    calls = mock_api(monkeypatch, {})
    assert commit_meta.fetch_commit_meta(SHA) is None
    assert len(calls) == 1  # no pulls lookup without the commit


def test_index_entry_carries_pr_title_author_and_avatar_only_when_known():
    base = {"subject": "x (#12)", "author": "me", "pr": 12, "committed_ts": 1}
    assert index_builder.compact_meta(base) == {"s": "x (#12)", "a": "me", "pr": 12, "ct": 1}
    full = {**base, "pr_title": "Fix x", "pr_author": "me", "pr_avatar": "https://a/1",
            "pr_labels": ["Core"], "pr_url": "https://github.com/x/pull/12"}
    assert index_builder.compact_meta(full) == {"s": "x (#12)", "a": "me", "pr": 12, "ct": 1,
                                                "pt": "Fix x", "pa": "me", "av": "https://a/1"}
    assert index_builder.compact_meta(None) is None
