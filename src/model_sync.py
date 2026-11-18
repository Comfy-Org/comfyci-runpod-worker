"""Worker-side models: the network volume, and what a job fetches for itself.

`sync` runs inside the serverless worker (mode=sync_models). It diffs the
manifest's model list against /runpod-volume/.sync-state.json and downloads
what's missing or changed. Files stream straight to their final path, so the
state file, updated only after a completed download, is the source of truth.

`fetch_local` runs at the start of a run job that says `fetch_missing`. It
downloads the models the volume does not hold to the worker's own disk, into
ComfyUI's models folder, which ComfyUI searches for every folder it knows. A
worker's disk is erased when the worker stops, so a file found again is a
bonus: every job checks what is there against its own list.

The Hugging Face token goes to https://huggingface.co and to no other host,
also not to the storage host a download is redirected to.
"""
from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume"))
MODELS_ROOT = VOLUME_ROOT / "models"
STATE_FILE = VOLUME_ROOT / ".sync-state.json"

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
# Where a job's own downloads go, and the list of what it put there.
LOCAL_ROOT = COMFY_ROOT / "models"
LOCAL_STATE = LOCAL_ROOT / ".comfyci-fetched.json"
# A download is written under this suffix and renamed when its sha256 matches.
PART = ".part"
# The one host the token is for.
TOKEN_HOST = "huggingface.co"
CHUNK_BYTES = 8 * 1024 * 1024
FETCH_STREAMS = 4
FETCH_TRIES = 3
# Seconds between two tries of a file, and the longest wait a rate limit gets.
RETRY_WAIT_S = 5
RATE_LIMIT_MAX_WAIT_S = 300
# The host says no, and another try would hear the same: no token, a licence
# that is not accepted, a file that moved.
NO_RETRY_HTTP = (401, 403, 404)
# Free space a job keeps beyond the files it downloads.
FREE_MARGIN_BYTES = 5 * 10**9
# What other downloaders leave on the worker's disk. It goes when room is short.
HF_CACHE = Path(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))


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


def token_for(url: str, hf_token: str | None) -> str | None:
    """The token, when a request to `url` may carry it: https, and the host is
    exactly huggingface.co. A host that only contains the name does not get it."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if (hf_token and parts.scheme == "https" and parts.hostname == TOKEN_HOST
            and port in (None, 443)):
        return hf_token
    return None


def _authorize(req: urllib.request.Request, hf_token: str | None) -> None:
    """Give a request the token its own URL may have, and no other. The header
    is one urllib does not copy to a redirect target by itself."""
    for name in [n for n in req.headers if n.lower() == "authorization"]:
        del req.headers[name]
    req.unredirected_hdrs.pop("Authorization", None)
    token = token_for(req.full_url, hf_token)
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")


class _TokenRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect, and decide about the token again for where it leads.
    urllib copies a request's headers to the redirect target, whatever its
    host. A file URL of huggingface.co redirects to a storage host."""

    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            _authorize(new, self.hf_token)
        return new


def _open(url: str, hf_token: str | None, timeout: float = 60):
    req = urllib.request.Request(url)
    _authorize(req, hf_token)
    return urllib.request.build_opener(_TokenRedirects(hf_token)).open(req, timeout=timeout)


class _Stopped(Exception):
    """A download gave up because another file of the job had failed."""


def _stream(url: str, dest: Path, hf_token: str | None,
            stop: threading.Event | None = None) -> tuple[int, str]:
    """Write `url` to `dest`, hashing while writing: (bytes, sha256). A
    connection that ends before the length the host announced is an error,
    not a short file."""
    h = hashlib.sha256()
    size = 0
    with _open(url, hf_token) as r, open(dest, "wb") as f:
        announced = r.headers.get("Content-Length")
        while True:
            if stop is not None and stop.is_set():
                raise _Stopped()
            chunk = r.read(CHUNK_BYTES)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)
    if announced and announced.isdigit() and size != int(announced):
        raise ConnectionError(f"the connection ended after {size} of {announced} bytes")
    return size, h.hexdigest()


def _download(url: str, dest: Path, hf_token: str | None) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _stream(url, dest, hf_token)


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


# -- a job's own models -----------------------------------------------------------------
class FetchError(Exception):
    """Why a file was not downloaded. `http_status` is the host's answer, when
    it gave one."""

    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


def _local_state() -> dict:
    try:
        state = json.loads(LOCAL_STATE.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    return {key: entry for key, entry in state.items() if isinstance(entry, dict)}


def _save_local_state(state: dict) -> None:
    tmp = LOCAL_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(LOCAL_STATE)


def _free_bytes() -> int:
    return shutil.disk_usage(LOCAL_ROOT).free


def _wait(stop: threading.Event, seconds: float) -> None:
    """Pause between two tries. It ends early when another file has failed."""
    stop.wait(seconds)


def _tracked() -> set[str] | None:
    """The files under LOCAL_ROOT that git tracks, as paths below it. ComfyUI
    keeps placeholders and configs there. None when git cannot say."""
    try:
        proc = subprocess.run(["git", "ls-files", "-z", "--", LOCAL_ROOT.name],
                              cwd=str(COMFY_ROOT), capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    prefix = LOCAL_ROOT.name + "/"
    names = (os.fsdecode(raw) for raw in proc.stdout.split(b"\0") if raw)
    return {name[len(prefix):] for name in names if name.startswith(prefix)}


def _clean_up(state: dict) -> list[str]:
    """Remove what no list knows, and return its names: every part file (a
    download that was killed left it), and every file that git does not track
    and the state does not list (an earlier workflow downloaded it by itself).
    Tracked files stay. When git cannot say what it tracks, only the part
    files go. An entry of the state whose file is gone is dropped."""
    tracked = _tracked()
    removed = []
    for folder, _dirs, names in os.walk(LOCAL_ROOT):
        for name in names:
            path = Path(folder) / name
            below = path.relative_to(LOCAL_ROOT).as_posix()
            if path == LOCAL_STATE:
                continue
            unknown = tracked is not None and below not in tracked and below not in state
            if name.endswith(PART) or unknown:
                try:
                    path.unlink()
                except OSError:
                    continue
                removed.append(below)
    for key in [k for k in state if not (LOCAL_ROOT / k).is_file()]:
        del state[key]
    return removed


def _local_path(model: dict) -> Path | None:
    """Where a model goes under LOCAL_ROOT. None for a name that leaves it."""
    root = LOCAL_ROOT.resolve()
    path = (root / str(model["directory"]) / str(model["name"])).resolve()
    return path if root in path.parents else None


def _reusable(entry: dict, model: dict, path: Path) -> bool:
    """Whether the file a state entry describes is the one the job asks for:
    the same url and sha256, and the size the entry recorded."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return (entry.get("url") == model["url"] and entry.get("bytes") == size
            and (not model.get("sha256") or entry.get("sha256") == model["sha256"]))


def _make_room(state: dict, keep: set[str], need: int, evicted: list[str]) -> int:
    """Free space until `need` bytes are free, and return the bytes free then.
    First the fetched files this job does not need, the one used longest ago
    first. Then what other downloaders cached."""
    free = _free_bytes()
    spare = sorted((key for key in state if key not in keep),
                   key=lambda key: state[key].get("used_at") or 0)
    for key in spare:
        if free >= need:
            break
        (LOCAL_ROOT / key).unlink(missing_ok=True)
        del state[key]
        evicted.append(key)
        free = _free_bytes()
    if evicted:
        _save_local_state(state)
    if free < need and HF_CACHE.exists():
        shutil.rmtree(HF_CACHE, ignore_errors=True)
        free = _free_bytes()
    return free


def rate_limit_wait_s(headers) -> int:
    """Seconds to wait after an HTTP 429: what the host's `RateLimit` header
    gives as the time to the reset (`"resolvers";r=0;t=55`), else its
    `Retry-After`, else RETRY_WAIT_S. Never more than RATE_LIMIT_MAX_WAIT_S."""
    found = re.search(r"(?:^|;)\s*t=(\d+)", headers.get("RateLimit") or "")
    seconds = int(found.group(1)) if found else None
    if seconds is None:
        after = (headers.get("Retry-After") or "").strip()
        seconds = int(after) if after.isdigit() else RETRY_WAIT_S
    return min(seconds, RATE_LIMIT_MAX_WAIT_S)


def repo_of(url: str) -> tuple[str | None, str | None]:
    """(host, repository) of a model URL, for the person who has to act on a
    refused file: the first two parts of the path, with `datasets/` or
    `spaces/` kept in front of them."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None, None
    names = [name for name in parts.path.split("/") if name]
    take = 3 if names[:1] in (["datasets"], ["spaces"]) else 2
    return parts.hostname, "/".join(names[:take]) if len(names) >= take else None


def _fetch_one(model: dict, path: Path, hf_token: str | None,
               stop: threading.Event) -> tuple[int, str]:
    """Download one model to `path` through its part file: (bytes, sha256).
    FETCH_TRIES tries. A host that answers 401, 403 or 404 gets no second
    one, nor does a file with the wrong sha256 or a full disk. FetchError when
    the file is not there in the end, and no part file is left."""
    part = path.with_name(path.name + PART)
    path.parent.mkdir(parents=True, exist_ok=True)
    error = FetchError("not tried")
    for attempt in range(1, FETCH_TRIES + 1):
        wait = RETRY_WAIT_S
        try:
            size, digest = _stream(model["url"], part, hf_token, stop)
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            e.close()
            error = FetchError(f"HTTP {e.code} {e.reason}"[:300], e.code)
            if e.code in NO_RETRY_HTTP:
                raise error
            if e.code == 429:
                wait = rate_limit_wait_s(e.headers)
        except _Stopped:
            part.unlink(missing_ok=True)
            raise
        except ValueError as e:  # a URL that urllib cannot open
            part.unlink(missing_ok=True)
            raise FetchError(f"bad url: {e}"[:300])
        except (OSError, http.client.HTTPException) as e:
            part.unlink(missing_ok=True)
            if getattr(e, "errno", None) == errno.ENOSPC:
                raise FetchError("disk full while writing")
            error = FetchError(f"{type(e).__name__}: {e}"[:300])
        else:
            if model.get("sha256") and digest != model["sha256"]:
                part.unlink(missing_ok=True)
                raise FetchError(f"sha256 mismatch: got {digest}")
            os.replace(part, path)
            return size, digest
        if attempt < FETCH_TRIES:
            _wait(stop, wait)
            if stop.is_set():
                raise _Stopped()
    raise error


def fetch_local(models: list[dict], hf_token: str | None = None,
                streams: int = FETCH_STREAMS) -> dict:
    """Bring `models` to the worker's own disk, under LOCAL_ROOT, and say what
    was done. The steps, in order:

    - Clean up: part files, and files that neither git nor the state knows.
    - Reuse: a file stays when the state has it with the same url and sha256
      and its size is the recorded one.
    - Make room: the files to download plus FREE_MARGIN_BYTES must be free.
      Fetched files this job does not need go first, then HF_CACHE. When that
      is not enough nothing is downloaded: "disk full: ...".
    - Download: up to `streams` files at once, each to a part file that is
      renamed when its sha256 matches. The first file that fails for good
      stops the others: the job is over then, and its time is billed.

    Returns `files` (downloaded), `reused` and `evicted` as lists of
    directory/name, `bytes` and `seconds` of the downloads with `mb_per_s`,
    `reused_bytes`, `streams`, `token` (whether there was a token to send to
    huggingface.co) and `failed`: one object per file that failed, with
    `model`, `error`, and for an answer of the host `http_status`, `host`
    and `repo`. A full disk is one object without `model`."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    streams = max(1, int(streams))
    report = {"files": [], "bytes": 0, "seconds": 0.0, "mb_per_s": None, "streams": streams,
              "token": bool(hf_token), "reused": [], "reused_bytes": 0, "evicted": [],
              "failed": []}
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    state = _local_state()
    _clean_up(state)

    wanted: list[tuple[str, dict, Path]] = []
    keep: set[str] = set()
    now = int(time.time())
    for m in models:
        key = f"{m['directory']}/{m['name']}"
        if key in keep:
            continue
        keep.add(key)
        path = _local_path(m)
        if path is None:
            report["failed"].append({"model": key, "error": "its name leaves the models folder"})
            continue
        entry = state.get(key)
        if entry and _reusable(entry, m, path):
            entry["used_at"] = now
            report["reused"].append(key)
            report["reused_bytes"] += entry["bytes"]
            continue
        # Another file under this name: it goes before the room is counted.
        if state.pop(key, None) is not None:
            path.unlink(missing_ok=True)
        wanted.append((key, m, path))
    _save_local_state(state)
    if report["failed"] or not wanted:
        return report

    need = sum(m.get("bytes") or 0 for _key, m, _path in wanted) + FREE_MARGIN_BYTES
    free = _make_room(state, keep, need, report["evicted"])
    if free < need:
        report["failed"].append({"error": f"disk full: {free / 1e9:.1f} GB free, "
                                          f"{need / 1e9:.1f} GB needed"})
        return report

    stop = threading.Event()
    lock = threading.Lock()
    done: dict[str, int] = {}

    def one(item: tuple[str, dict, Path]) -> None:
        key, m, path = item
        if stop.is_set():
            return
        try:
            size, digest = _fetch_one(m, path, hf_token, stop)
        except _Stopped:
            return
        except FetchError as e:
            stop.set()  # the job is over: the other downloads end too
            failure = {"model": key, "error": str(e)}
            if e.http_status is not None:
                host, repo = repo_of(m["url"])
                failure.update(http_status=e.http_status, host=host, repo=repo)
            with lock:
                report["failed"].append(failure)
            return
        except BaseException:
            stop.set()
            raise
        with lock:
            state[key] = {"url": m["url"], "sha256": digest, "bytes": size,
                          "used_at": int(time.time())}
            _save_local_state(state)
            done[key] = size

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(streams, len(wanted))) as pool:
        list(pool.map(one, wanted))
    seconds = time.monotonic() - t0
    report["files"] = [key for key, _m, _path in wanted if key in done]
    report["bytes"] = sum(done.values())
    report["seconds"] = round(seconds, 1)
    if report["bytes"] and seconds > 0:
        report["mb_per_s"] = round(report["bytes"] / 1e6 / seconds, 1)
    return report


def fetch_brief(report: dict) -> dict:
    """A fetch report as a run record carries it: each list of names as its
    length, and without the failures, which are the job's failure detail."""
    return {key: len(value) if isinstance(value, list) else value
            for key, value in report.items() if key != "failed"}


def fetch_failure(report: dict) -> tuple[str, dict] | None:
    """(worker status, failure detail) of a fetch that did not get every file,
    else None. `model_access_error` when the host refused a file (401, 403,
    404): a person has to act, and the detail names the host, the repository
    and the answer. Anything else is `model_fetch_error`, which another job
    can get past: the network, a rate limit, a wrong sha256, a full disk."""
    failed = report.get("failed") or []
    if not failed:
        return None
    refused = [f for f in failed if f.get("http_status") in NO_RETRY_HTTP]
    if refused:
        first = refused[0]
        return "model_access_error", {
            "message": (f"the model host answered {first['http_status']} for "
                        f"{first.get('repo') or first.get('model')}"),
            "host": first.get("host"), "repo": first.get("repo"),
            "http_status": first["http_status"], "failed": failed[:20]}
    first = failed[0]
    where = f"{first['model']}: " if first.get("model") else ""
    return "model_fetch_error", {"message": f"model download failed: {where}{first['error']}",
                                 "failed": failed[:20]}
