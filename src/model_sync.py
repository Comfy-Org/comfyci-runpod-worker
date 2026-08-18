"""Worker-side model sync: bring the network volume in line with the manifest.

Runs inside the serverless worker (mode=sync_models). Diffs the manifest's
model list against /runpod-volume/.sync-state.json and downloads what's
missing or changed. Files stream straight to their final path, so the state
file — updated only after a completed download — is the source of truth.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume"))
MODELS_ROOT = VOLUME_ROOT / "models"
STATE_FILE = VOLUME_ROOT / ".sync-state.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def _download(url: str, dest: Path, hf_token: str | None) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if hf_token and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {hf_token}")
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024 * 8)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def missing_models(models: list[dict]) -> list[str]:
    """Names from `models` not present-and-complete on the volume."""
    state = _load_state()
    out = []
    for m in models:
        key = f"{m['directory']}/{m['name']}"
        path = MODELS_ROOT / m["directory"] / m["name"]
        if key not in state or not path.exists():
            out.append(key)
    return out


def sync(models: list[dict], hf_token: str | None = None) -> dict:
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    state = _load_state()
    report = {"downloaded": [], "skipped": [], "failed": [], "deleted": []}
    for m in models:
        key = f"{m['directory']}/{m['name']}"
        path = MODELS_ROOT / m["directory"] / m["name"]
        entry = state.get(key)
        if entry and path.exists() and entry.get("url") == m["url"] and (
            not m.get("sha256") or entry.get("sha256") == m["sha256"]
        ):
            report["skipped"].append(key)
            continue
        t0 = time.monotonic()
        try:
            size, digest = _download(m["url"], path, hf_token)
        except Exception as e:
            report["failed"].append({"model": key, "error": str(e)[:500]})
            continue
        if m.get("sha256") and digest != m["sha256"]:
            path.unlink(missing_ok=True)
            report["failed"].append({"model": key, "error": f"sha256 mismatch: got {digest}"})
            continue
        state[key] = {"url": m["url"], "sha256": digest, "bytes": size,
                      "synced_at": int(time.time())}
        _save_state(state)
        report["downloaded"].append({"model": key, "bytes": size,
                                     "seconds": round(time.monotonic() - t0, 1)})
    return report
