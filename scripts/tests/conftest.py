import json
import subprocess
import sys
from pathlib import Path

import pytest

# Tests import the scripts as top-level modules, the same way CI runs them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout


def make_remote(tmp_path, files: dict[str, bytes | str | dict]):
    """Bare repo whose `results` branch holds `files` (path -> bytes/str/json)."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "results", str(bare)], check=True)
    git(bare, "config", "uploadpack.allowFilter", "true")
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "-b", "results", str(seed)], check=True)
    for path, content in files.items():
        p = seed / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        elif isinstance(content, str):
            p.write_text(content, encoding="utf-8")
        else:
            p.write_text(json.dumps(content, indent=1), encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed")
    git(seed, "push", "-q", str(bare), "HEAD:refs/heads/results")
    return bare


@pytest.fixture
def remote(tmp_path):
    """Results branch with one run and a latest pointer."""
    return make_remote(tmp_path, {
        "regression/runs/master/aaa/summary.json": {
            "commit": "aaa", "overall": "pass", "workflows": {"wf": {"verdict": "pass"}}},
        "regression/runs/master/aaa/wf/outputs/wf_00001_.png": b"\x89PNG seed",
        "regression/latest/master.json": {"commit": "aaa"},
    })
