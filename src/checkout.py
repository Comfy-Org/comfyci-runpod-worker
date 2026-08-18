"""Check out an arbitrary ComfyUI commit inside the pre-cloned repo.

The worker image ships with a clone of Comfy-Org/ComfyUI at /comfyui and its
requirements pre-installed. Per request we fetch the target sha (GitHub allows
fetching arbitrary reachable SHAs), hard-checkout, and re-run pip only when the
commit's requirements.txt differs from what the image baked in.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
# Written at image build time: sha256 of the requirements.txt that was pip-installed.
BAKED_REQ_HASH = Path(os.environ.get("BAKED_REQ_HASH_FILE", "/comfy-config/requirements.sha256"))
DEFAULT_REPO = "https://github.com/Comfy-Org/ComfyUI"


class CheckoutError(RuntimeError):
    pass


def _git(*args: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(COMFY_ROOT), capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise CheckoutError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:2000]}")
    return proc.stdout.strip()


def _req_hash() -> str:
    req = COMFY_ROOT / "requirements.txt"
    if not req.exists():
        return ""
    return hashlib.sha256(req.read_bytes()).hexdigest()


def checkout_commit(commit: str, repo_url: str | None = None) -> dict:
    """Fetch + hard checkout `commit`. Returns timing/pip info for the run record."""
    t0 = time.monotonic()
    repo_url = repo_url or DEFAULT_REPO
    current_remote = _git("remote", "get-url", "origin")
    if current_remote != repo_url:
        _git("remote", "set-url", "origin", repo_url)

    # Already on the right commit (warm worker re-used for the same push)?
    try:
        head = _git("rev-parse", "HEAD")
    except CheckoutError:
        head = ""
    if head != commit:
        try:
            _git("fetch", "--depth", "1", "origin", commit, timeout=600)
        except CheckoutError:
            # Unshallow fallback: some servers refuse SHA fetch on shallow clones.
            _git("fetch", "--unshallow", "origin", timeout=1200)
            _git("fetch", "origin", commit, timeout=600)
        _git("checkout", "-f", commit)
        _git("clean", "-fdq")
    checkout_s = round(time.monotonic() - t0, 2)

    pip_s = 0.0
    baked = BAKED_REQ_HASH.read_text().strip() if BAKED_REQ_HASH.exists() else ""
    now = _req_hash()
    if now and now != baked:
        t1 = time.monotonic()
        proc = subprocess.run(
            ["pip", "install", "-r", "requirements.txt", "--upgrade-strategy", "only-if-needed"],
            cwd=str(COMFY_ROOT), capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise CheckoutError(f"pip install failed: {proc.stderr.strip()[-2000:]}")
        pip_s = round(time.monotonic() - t1, 2)

    return {
        "commit_checked_out": _git("rev-parse", "HEAD"),
        "checkout_s": checkout_s,
        "pip_s": pip_s,
        "requirements_changed": bool(now and now != baked),
    }
