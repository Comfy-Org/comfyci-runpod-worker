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

A job can say how long its download may take (`timeout_s`). When that time is
up the download is stopped and the job answers with what it reached, so a slow
download does not run on until the runner stops waiting for it. A download
that does not end when it is stopped has its connection shut, so none stays
behind on a connection that only trickles. A write that finds the disk full
makes room once, from what the job does not need, and is tried once more:
what a worker's disk reports as free is not relied on.

The Hugging Face token goes to https://huggingface.co and to no other host,
also not to the storage host a download is redirected to. It is in no message
either: what a download says when it fails is published with the run.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import errno
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import socket
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
# What a Bearer header may carry (RFC 6750). A token of other characters is
# not sent: the HTTP library would refuse the header and quote it in its error.
TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/-]+=*")
# What a message says where the token stood.
TOKEN_MARK = "<token>"
# How a model URL may start.
URL_SCHEMES = ("http", "https")
CHUNK_BYTES = 8 * 1024 * 1024
# What a job's download takes from its connection at once, and the least a
# connection must bring in a second to be read that way. Between two pieces
# the download looks whether it has been stopped. A smaller piece costs
# speed: measured with four downloads at once, 1 MiB is as fast as the 8 MiB
# of before, and 64 KiB is far slower.
PIECE_BYTES = 1024 * 1024
FETCH_STREAMS = 4
FETCH_TRIES = 3
# Seconds between two tries of a file, and the longest wait a rate limit gets.
RETRY_WAIT_S = 5
RATE_LIMIT_MAX_WAIT_S = 300
# The host says no, and another try would hear the same: no token, a licence
# that is not accepted, a file that moved.
NO_RETRY_HTTP = (401, 403, 404)
# What the failure of a download says under `reason` when the job's time limit
# stopped it. It is the word a run record has for a job that used up its time
# (scripts/submit_and_poll.py), and the runner tries neither again.
TIMED_OUT = "timed_out"
# Seconds a job waits for its downloads to end once they are stopped: its time
# is up, or a file has failed. A download sees the stop between two pieces
# (`_stream`). One that is still reading after FETCH_CUT_AFTER_S of them waits
# for a piece that does not come: the job shuts its connection, which ends
# the read (`_Fetch.sever`). One that is still there at the end is still
# opening its connection. The job answers without it, so it answers this soon
# whatever a host does. Such a download ends by itself within the 60 s of
# `_open`, and changes no file (`_Fetch`).
FETCH_STOP_WAIT_S = 10
FETCH_CUT_AFTER_S = 8
# Seconds between two looks of a job at its downloads while it waits for
# them: a file that has failed ends the wait, also beside a download that is
# held up in a read.
FETCH_LOOK_S = 1
# How the threads of a job's downloads are named, for whoever looks for one
# that was left behind.
FETCH_THREADS = "model-fetch"
# What a write says when the disk has no room: no space, or a quota used up.
DISK_FULL_ERRNOS = tuple(code for code in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
                         if code is not None)
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


def clean_token(value) -> str | None:
    """A token as it is sent: without the white space a pasted secret has
    around it. None for no token, and for a value that a Bearer header cannot
    carry (a line break inside it, a character that is not ASCII)."""
    text = value.strip() if isinstance(value, str) else ""
    return text if TOKEN_RE.fullmatch(text) else None


def without_token(text: str, hf_token: str | None) -> str:
    """`text` with the token taken out of it, for a message that leaves the
    worker."""
    return text.replace(hf_token, TOKEN_MARK) if hf_token else text


def token_for(url: str, hf_token: str | None) -> str | None:
    """The token, when a request to `url` may carry it: https, and the host is
    exactly huggingface.co. A host that only contains the name does not get
    it, nor does a URL with a user name in front of the host: urllib would
    take the two together for the name to connect to."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if (hf_token and parts.scheme == "https" and parts.hostname == TOKEN_HOST
            and port in (None, 443) and "@" not in parts.netloc):
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
    """Open a model URL. ValueError for one that is not http or https: urllib
    would open a `file:` URL too, and that is a file of the worker itself."""
    if urllib.parse.urlsplit(url).scheme not in URL_SCHEMES:
        raise ValueError(f"not an http or https url: {url}")
    req = urllib.request.Request(url)
    _authorize(req, hf_token)
    return urllib.request.build_opener(_TokenRedirects(hf_token)).open(req, timeout=timeout)


class _Stopped(Exception):
    """A download gave up because another file of the job had failed, or
    because the job's time was up."""


def _stream(url: str, dest: Path, hf_token: str | None,
            stop: threading.Event | None = None) -> tuple[int, str]:
    """Write `url` to `dest`, hashing while writing: (bytes, sha256). A
    connection that ends before the length the host announced is an error,
    not a short file.

    The volume sync has no `stop` and reads CHUNK_BYTES at a time, as it
    always did. A job's download reads PIECE_BYTES at a time and looks at
    `stop` before each piece. A read waits until its piece is whole. So when
    a connection has brought less than a piece a second, the download takes
    what is there as it comes, until the connection is that fast again: it is
    then left with the next bytes the connection sends, however few. A read
    that began on a fast connection still waits for its whole piece, so the
    job can end it (`_Fetch.sever`): that is a stop too, not a failed try.
    The `stop` of a job (`_Fetch`) is told what has arrived, piece by piece,
    and opens the file itself: it does not for a download that has been
    stopped. A download that has every byte the host announced is not
    stopped: the one read it has left only finds the end."""
    h = hashlib.sha256()
    size = 0
    count = getattr(stop, "count", None)
    create = getattr(stop, "create", None) or (lambda path: open(path, "wb"))
    line = getattr(stop, "line", None) or (lambda path, response: contextlib.nullcontext())
    want = CHUNK_BYTES if stop is None else PIECE_BYTES
    if stop is not None and stop.is_set():
        raise _Stopped()
    if count:
        count(dest, 0)
    try:
        with _open(url, hf_token) as r, create(dest) as f, line(dest, r):
            announced = r.headers.get("Content-Length")
            total = int(announced) if announced and announced.isdecimal() else None
            # Whether the connection is a slow one, by what the last second
            # or more of it brought.
            slow, since, brought = False, time.monotonic(), 0
            while True:
                if stop is not None and stop.is_set() and size != total:
                    raise _Stopped()
                piece = r.read1(want) if slow else r.read(want)
                if not piece:
                    if size != total and getattr(stop, "severed", False):
                        raise _Stopped()  # the end the job made, not the file's
                    break
                f.write(piece)
                h.update(piece)
                size += len(piece)
                if count:
                    count(dest, size)
                if stop is not None:
                    brought += len(piece)
                    passed = time.monotonic() - since
                    if passed >= 1:
                        slow = brought < passed * want
                        since, brought = since + passed, 0
        if total is not None and size != total:
            raise ConnectionError(f"the connection ended after {size} of {announced} bytes")
    except _Stopped:
        raise  # what an unfinished file had received is part of what the job reached
    except BaseException as e:
        if (getattr(stop, "severed", False)
                and isinstance(e, (OSError, http.client.HTTPException))):
            # What a read says whose connection the job has shut.
            raise _Stopped() from None
        if count:
            count(dest, 0)  # this try is over, and its part file goes
        raise
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
    hf_token = clean_token(hf_token) or clean_token(os.environ.get("HF_TOKEN"))
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
            report["failed"].append({"model": key,
                                     "error": without_token(str(e), hf_token)[:500]})
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


class _DiskFull(FetchError):
    """A write found no room on the disk."""

    def __init__(self):
        super().__init__("disk full while writing")


def _twin(response) -> socket.socket | None:
    """A second handle on the connection `response` is read from, for a
    thread that is not the reader. None when there is no way to one."""
    try:
        return socket.fromfd(response.fileno(), socket.AF_INET, socket.SOCK_STREAM)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _shut(twin: socket.socket) -> None:
    """End a connection for whoever reads from it. On Linux, which the worker
    is, a read that waits for bytes comes back at once. On Windows it comes
    back with the next byte the connection brings, or after its 60 s."""
    try:
        twin.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # it has ended by itself


class _Fetch(threading.Event):
    """What the downloads of one job share.

    It is their stop signal: set when a file has failed for good, or when the
    job's time is up. It counts what each download has received (`received`,
    per part file), so a job that is stopped can say how far it came.

    It can end a read. A download looks at the stop between two pieces, and a
    piece can take as long as its connection likes. So a job that has waited
    FETCH_CUT_AFTER_S for its downloads shuts the connections that are still
    read from (`sever`). The handle for that is the job's own, made by the
    download before its first read (`line`): the response closes its own when
    the last byte is read, and that number can be another connection's by
    the time the job would look it up.

    And it is their one way to the files. A job waits FETCH_STOP_WAIT_S for
    its downloads and then goes on without those that are still there
    (`over`). Such a download can wake later, when the next job of the worker
    is writing a part file of the same name: from then on it removes nothing
    and puts nothing in place. `lock` makes that one step with what the job
    itself does."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.over = False
        self.received: dict[Path, int] = {}
        # None until the job has made room after a full disk, then whether
        # that removed anything.
        self.emptied: bool | None = None
        # Per part file, the job's handle on the connection it is read from.
        self.lines: dict[Path, socket.socket] = {}
        self.severed = False
        self.cutting = threading.Lock()

    def count(self, part: Path, size: int) -> None:
        self.received[part] = size

    @contextlib.contextmanager
    def line(self, part: Path, response):
        """For as long as a download reads `response`: the job can end that
        read (`sever`)."""
        twin = _twin(response)
        if twin is not None:
            with self.cutting:
                self.lines[part] = twin
                if self.severed:
                    _shut(twin)
        try:
            yield
        finally:
            if twin is not None:
                with self.cutting, contextlib.suppress(OSError):
                    self.lines.pop(part, None)
                    twin.close()

    def sever(self) -> None:
        """Shut the connection of every download that is still reading. Its
        read ends, and the download with it: stopped, not failed."""
        with self.cutting:
            self.severed = True
            for twin in self.lines.values():
                _shut(twin)

    def create(self, part: Path):
        """Open a part file for writing. Not for a download that has been
        stopped: opening would empty the file, and that can be the one the
        next job is writing."""
        with self.lock:
            if self.over or self.is_set():
                raise _Stopped()
            return open(part, "wb")

    def drop(self, part: Path) -> None:
        """Remove a part file, unless the job has gone on without its download."""
        with self.lock:
            if not self.over:
                part.unlink(missing_ok=True)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _below(key: str) -> Path | None:
    """The file of `key` (directory/name) under LOCAL_ROOT. None for a key that
    leaves the folder, and for one that is not the plain path of its file
    (`vae/../loras/x`): the list would know the file under another name than
    the disk does."""
    root = LOCAL_ROOT.resolve()
    try:
        path = (root / key).resolve()
    except (OSError, ValueError):
        return None
    if root not in path.parents or path.relative_to(root).as_posix() != key:
        return None
    return path


def _local_state() -> dict:
    """The worker's list of what it fetched: per directory/name the url, the
    sha256, the bytes and when a job last used the file. An entry that this
    module would not have written is left out, so its file is one that no
    list knows."""
    try:
        state = json.loads(LOCAL_STATE.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    return {key: entry for key, entry in state.items()
            if isinstance(entry, dict) and isinstance(entry.get("url"), str)
            and _is_number(entry.get("bytes")) and _is_number(entry.get("used_at"))
            and _below(key) is not None}


def _save_local_state(state: dict) -> None:
    tmp = LOCAL_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(LOCAL_STATE)


def _free_bytes() -> int:
    return shutil.disk_usage(LOCAL_ROOT).free


def _wait(stop: threading.Event, seconds: float) -> None:
    """Pause between two tries. It ends early when another file has failed,
    and when the job's time is up."""
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
    Tracked files stay, and so does a model of the state whose own name ends
    like a part file. When git cannot say what it tracks, only the part files
    go. An entry of the state whose file is gone is dropped."""
    tracked = _tracked()
    removed = []
    for folder, _dirs, names in os.walk(LOCAL_ROOT):
        for name in names:
            path = Path(folder) / name
            below = path.relative_to(LOCAL_ROOT).as_posix()
            if path == LOCAL_STATE or below in state:
                continue
            unknown = tracked is not None and below not in tracked
            if name.endswith(PART) or unknown:
                try:
                    path.unlink()
                except OSError:
                    continue
                removed.append(below)
    for key in [k for k in state if not (LOCAL_ROOT / k).is_file()]:
        del state[key]
    return removed


def _reusable(entry: dict, model: dict, path: Path) -> bool:
    """Whether the file a state entry describes is the one the job asks for:
    the same url and sha256, and the size the entry recorded."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return (entry.get("url") == model["url"] and entry.get("bytes") == size
            and (not model.get("sha256") or entry.get("sha256") == model["sha256"]))


def _device(path: Path) -> int:
    """The disk a path is on, as the system numbers it."""
    return path.stat().st_dev


def _spare_cache() -> Path | None:
    """HF_CACHE, when removing it gives the models room: a folder on the disk
    the models folder is on. None for any other. `HF_HOME` can name any
    folder, also one on the network volume, which is every worker's: a cache
    there is on another disk, and removing it would free nothing here. A
    folder that holds the volume or the models folder, or lies in one of
    them, is never removed."""
    try:
        cache, local, volume = HF_CACHE.resolve(), LOCAL_ROOT.resolve(), VOLUME_ROOT.resolve()
        if not cache.is_dir() or _device(cache) != _device(local):
            return None
    except OSError:
        return None
    for other in (local, volume):
        if cache == other or cache in other.parents or other in cache.parents:
            return None
    return cache


def _make_room(state: dict, keep: set[str], need: int, evicted: list[str]) -> int:
    """Free space until `need` bytes are free, and return the bytes free then.
    First the fetched files this job does not need, the one used longest ago
    first. Then what other downloaders cached (`_spare_cache`)."""
    free = _free_bytes()
    spare = sorted((key for key in state if key not in keep),
                   key=lambda key: state[key]["used_at"])
    for key in spare:
        if free >= need:
            break
        (LOCAL_ROOT / key).unlink(missing_ok=True)
        del state[key]
        evicted.append(key)
        free = _free_bytes()
    if evicted:
        _save_local_state(state)
    cache = _spare_cache() if free < need else None
    if cache is not None:
        shutil.rmtree(cache, ignore_errors=True)
        free = _free_bytes()
    return free


def _holds_a_file(folder: Path) -> bool:
    return any(names for _folder, _dirs, names in os.walk(folder))


def _empty(state: dict, keep: set[str], evicted: list[str]) -> bool:
    """Remove everything `_make_room` can remove, whatever the disk says is
    free, and say whether anything went: a fetched file, or a file of the
    cache. For a write that found the disk full: the free space the system
    reports may be that of another disk than the one the worker is held to,
    and then `_make_room` removed nothing."""
    before, cache = len(evicted), _spare_cache()
    cached = cache is not None and _holds_a_file(cache)
    _make_room(state, keep, math.inf, evicted)
    return len(evicted) > before or cached


def too_slow(received: int, wanted: int | None, seconds: float, files_done: int,
             files_wanted: int, timeout_s: float) -> dict:
    """The failure of a download that its job's time limit stopped: what it
    reached, as a sentence under `error` and as the numbers it is made of.
    `received` counts the unfinished files too. `wanted` is None when a file
    did not say its size. `seconds` is how long the downloads had: 0 when the
    time was up before the first of them began. The sizes are in GB, and in
    MB when they are under one."""
    speed = round(received / 1e6 / seconds, 1) if seconds > 0 else None
    unit, name = (1e9, "GB") if max(received, wanted or 0) >= 1e9 else (1e6, "MB")
    if seconds > 0:
        said = (f"the download was too slow: {received / unit:.1f}"
                + (f" of {wanted / unit:.1f}" if wanted else "")
                + f" {name} in {seconds:.0f} s ({speed:.1f} MB/s), "
                + f"{files_done} of {files_wanted} files complete")
    else:
        said = ("the download did not begin: cleaning up and making room took the "
                f"{timeout_s:g} s it was given")
    return {"error": said, "reason": TIMED_OUT, "timeout_s": timeout_s,
            "received_bytes": received, "wanted_bytes": wanted, "seconds": round(seconds, 1),
            "mb_per_s": speed, "files_done": files_done, "files_wanted": files_wanted}


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


def _mismatch(model: dict, size: int, digest: str) -> str | None:
    """Why a download is not the file its entry describes, else None: another
    sha256, or, for an entry without one, another size than its `bytes`."""
    if model.get("sha256"):
        return None if digest == model["sha256"] else f"sha256 mismatch: got {digest}"
    want = model.get("bytes")
    if _is_number(want) and size != want:
        return f"size mismatch: got {size} bytes, not {want}"
    return None


def _fetch_one(model: dict, path: Path, hf_token: str | None, stop: _Fetch, note) -> None:
    """Download one model to `path` through its part file. When the file is
    there under its name, `note(bytes, sha256)` has been called, in one step
    with the rename (`stop.lock`): a job that has gone on without this
    download (`stop.over`) gets neither.

    FETCH_TRIES tries. The model's own host answering 401, 403 or 404 gets no
    second one, nor does a file with the wrong sha256. The same answer from a
    host that a redirect led to is tried again like any failure: no person
    can act on it, and the model's host signs a new link for each try. A file
    without a sha256 must have the size its entry says. FetchError when the
    file is not there in the end, and no part file is left. `_DiskFull` when
    a write found no room, also the write of the list (`note`): a file that
    no list names would be removed by the next job, so it is removed here.
    _Stopped when `stop` ended it."""
    part = path.with_name(path.name + PART)
    own_host = repo_of(model["url"])[0]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FetchError(f"{type(e).__name__}: {e}")
    error = FetchError("not tried")
    for attempt in range(1, FETCH_TRIES + 1):
        wait = RETRY_WAIT_S
        try:
            size, digest = _stream(model["url"], part, hf_token, stop)
        except urllib.error.HTTPError as e:
            stop.drop(part)
            e.close()
            answered = repo_of(e.url or "")[0] or own_host
            if answered == own_host:
                error = FetchError(f"HTTP {e.code} {e.reason}", e.code)
                if e.code in NO_RETRY_HTTP:
                    raise error
            else:
                error = FetchError(f"HTTP {e.code} {e.reason} from {answered}")
            if e.code == 429:
                wait = rate_limit_wait_s(e.headers)
        except _Stopped:
            stop.drop(part)
            raise
        except ValueError as e:  # a URL that urllib cannot open
            stop.drop(part)
            raise FetchError(f"bad url: {e}")
        except (OSError, http.client.HTTPException) as e:
            stop.drop(part)
            if getattr(e, "errno", None) in DISK_FULL_ERRNOS:
                raise _DiskFull()
            error = FetchError(f"{type(e).__name__}: {e}")
        else:
            wrong = _mismatch(model, size, digest)
            if wrong is None:
                with stop.lock:
                    if stop.over:
                        raise _Stopped()
                    try:
                        os.replace(part, path)
                    except OSError as e:
                        wrong = f"{type(e).__name__}: {e}"
                    else:
                        try:
                            note(size, digest)
                        except OSError as e:
                            if e.errno not in DISK_FULL_ERRNOS:
                                raise
                            path.unlink(missing_ok=True)
                            raise _DiskFull() from None
                        return
            stop.drop(part)
            raise FetchError(wrong)
        if attempt < FETCH_TRIES:
            _wait(stop, wait)
            if stop.is_set():
                raise _Stopped()
    raise error


def fetch_local(models: list[dict], hf_token: str | None = None,
                streams: int = FETCH_STREAMS, timeout_s: float | None = None) -> dict:
    """Bring `models` to the worker's own disk, under LOCAL_ROOT, and say what
    was done. The steps, in order:

    - Clean up: part files, and files that neither git nor the state knows.
    - Reuse: a file stays when the state has it with the same url and sha256
      and its size is the recorded one.
    - Make room: the files to download plus FREE_MARGIN_BYTES must be free.
      Fetched files this job does not need go first, then HF_CACHE when it
      is on the worker's disk. When that is not enough nothing is downloaded:
      "disk full: ...".
    - Download: up to `streams` files at once, each to a part file that is
      renamed when its sha256 matches. The first file that fails for good
      stops the others: the job is over then, and its time is billed.

    A write that finds the disk full (`_DiskFull`) makes room once more, this
    time of everything the job does not need (`_empty`), and its file is
    tried once more. That happens once in a job, however many downloads find
    the disk full. When nothing could be removed, or the disk is full again,
    the file has failed: "disk full while writing".

    `timeout_s` is how long all of this may take, counted from the call (None:
    as long as it takes). When the time is up the downloads are stopped, and
    FETCH_STOP_WAIT_S later at the most the report is made, with or without
    them. The files that were complete stay and are in the state, and no part
    file is left. A file that is whole in those last seconds counts: when
    every file is, nothing was too slow and the job goes on. A file that
    fails for good in them is the failure, as it is at any other time.

    Returns `files` (downloaded), `reused` and `evicted` as lists of
    directory/name, `bytes` and `seconds` of the downloads with `mb_per_s`,
    `reused_bytes`, `streams`, `token` (whether there was a token to send to
    huggingface.co) and `failed`: one object per file that failed, with
    `model`, `error`, and for an answer of the model's own host `http_status`,
    `host` and `repo`. A full disk before any download is one object without
    `model`. So is a download that its time limit stopped (`too_slow`):
    `bytes` and `mb_per_s` then count what the unfinished files had received
    too, and `seconds` is the time up to the stop. No `error` holds the
    token."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    hf_token = clean_token(hf_token) or clean_token(os.environ.get("HF_TOKEN"))
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
        path = _below(key)
        if path is None or not isinstance(m.get("url"), str):
            why = ("it has no url" if path is not None else
                   "its name is not a plain path below the models folder")
            report["failed"].append({"model": key, "error": why})
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
    try:
        _save_local_state(state)
    except OSError as e:
        if e.errno not in DISK_FULL_ERRNOS:
            raise
        # No room for the list itself. What follows makes room or says "disk
        # full", and a list that is behind is put right by the next clean-up.
    if report["failed"] or not wanted:
        return report

    need = sum(m.get("bytes") or 0 for _key, m, _path in wanted) + FREE_MARGIN_BYTES
    free = _make_room(state, keep, need, report["evicted"])
    if free < need:
        report["failed"].append({"error": f"disk full: {free / 1e9:.1f} GB free, "
                                          f"{need / 1e9:.1f} GB needed"})
        return report

    sizes = [m.get("bytes") for _key, m, _path in wanted]
    total = sum(sizes) if all(_is_number(size) and size > 0 for size in sizes) else None
    if deadline is not None and time.monotonic() >= deadline:
        # The steps above used up the time: no download begins.
        report["failed"].append(too_slow(0, total, 0.0, 0, len(wanted), timeout_s))
        return report

    stop = _Fetch()
    done: dict[str, int] = {}

    def noted(key: str, m: dict):
        def note(size: int, digest: str) -> None:  # called with `stop.lock` held
            state[key] = {"url": m["url"], "sha256": digest, "bytes": size,
                          "used_at": int(time.time())}
            try:
                _save_local_state(state)
            except OSError:
                del state[key]  # the list on the disk does not have it
                raise
            done[key] = size
        return note

    def room_made() -> bool:
        """Make room after a full disk, once in the job, and say whether
        that removed anything. A download the job has left makes none."""
        with stop.lock:
            if stop.over:
                raise _Stopped()
            if stop.emptied is None:
                stop.emptied = _empty(state, keep, report["evicted"])
            return stop.emptied

    def one(item: tuple[str, dict, Path]) -> None:
        key, m, path = item
        if stop.is_set():
            return
        try:
            # A try that began after room was made has had its second chance.
            last = stop.emptied is not None
            try:
                _fetch_one(m, path, hf_token, stop, noted(key, m))
            except _DiskFull:
                if last or not room_made():
                    raise
                _fetch_one(m, path, hf_token, stop, noted(key, m))
        except _Stopped:
            return
        except FetchError as e:
            stop.set()  # the job is over: the other downloads end too
            failure = {"model": key, "error": without_token(str(e), hf_token)[:300]}
            if e.http_status is not None:
                host, repo = repo_of(m["url"])
                failure.update(http_status=e.http_status, host=host, repo=repo)
            with stop.lock:
                if not stop.over:
                    report["failed"].append(failure)
            return
        except BaseException:
            stop.set()
            raise

    def without(items) -> None:
        """Remove the part files of downloads the job goes on without. They
        touch no file any more themselves."""
        for _key, _m, path in items:
            try:
                path.with_name(path.name + PART).unlink(missing_ok=True)
            except OSError:
                pass  # the next job's clean-up removes it

    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(streams, len(wanted)),
                                                 thread_name_prefix=FETCH_THREADS)
    jobs: dict = {}
    try:
        for item in wanted:
            jobs[pool.submit(one, item)] = item
        # Until every download has ended, a file has failed, or the time is up.
        running = set(jobs)
        while running and not stop.is_set():
            look = FETCH_LOOK_S
            if deadline is not None:
                look = min(look, deadline - time.monotonic())
            if look <= 0:
                break
            running = concurrent.futures.wait(running, timeout=look).not_done
        # The time is up while a download is still going, and no failed file
        # has ended the job before: the downloads are stopped, and this is why.
        slow = bool(running) and not stop.is_set()
        stopped = time.monotonic()
        if running:
            stop.set()
            running = concurrent.futures.wait(
                running, timeout=min(FETCH_CUT_AFTER_S, FETCH_STOP_WAIT_S)).not_done
        if running:
            stop.sever()
            running = concurrent.futures.wait(
                running, timeout=max(FETCH_STOP_WAIT_S - FETCH_CUT_AFTER_S, 0)).not_done
    except BaseException:
        # The fetch itself broke. Its downloads do not run on behind the
        # answer of the job: they are ended here, and change no file.
        stop.set()
        stop.sever()
        with stop.lock:
            stop.over = True
        pool.shutdown(wait=False, cancel_futures=True)
        without(wanted)
        raise
    pool.shutdown(wait=False, cancel_futures=True)
    with stop.lock:
        stop.over = True
        ended = time.monotonic()
        received = sum(list(stop.received.values()))
    # A download that is still there is left behind, and its part file goes.
    without(jobs[job] for job in running)
    for job in jobs:
        if job.done() and not job.cancelled():
            job.result()  # what broke in a download breaks the fetch, as before
    # What the seconds after the stop brought decides too. A file that failed
    # for good in them is the failure, and the runner tries the job again as
    # it does after any such failure. When every file arrived in them, the
    # download is whole and the job goes on.
    slow = slow and len(done) < len(wanted) and not report["failed"]
    seconds = (stopped if slow else ended) - t0
    report["files"] = [key for key, _m, _path in wanted if key in done]
    report["bytes"] = sum(done.values())
    report["seconds"] = round(seconds, 1)
    if report["bytes"] and seconds > 0:
        report["mb_per_s"] = round(report["bytes"] / 1e6 / seconds, 1)
    if slow:
        reached = too_slow(received, total, seconds, len(done), len(wanted), timeout_s)
        report["failed"].append(reached)
        # The record of this download is its speed: all that arrived counts.
        report["bytes"], report["mb_per_s"] = received, reached["mb_per_s"]
    return report


def fetch_brief(report: dict) -> dict:
    """A fetch report as a run record carries it: each list of names as its
    length, and without the failures, which are the job's failure detail."""
    return {key: len(value) if isinstance(value, list) else value
            for key, value in report.items() if key != "failed"}


def fetch_failure(report: dict) -> tuple[str, dict] | None:
    """(worker status, failure detail) of a fetch that did not get every file,
    else None. `model_access_error` when the model's own host refused a file
    (401, 403, 404): a person has to act, and the detail names the host, the
    repository and the answer. Anything else is `model_fetch_error`, which
    another job can get past: the network, a rate limit, a wrong sha256, a
    full disk, a refusal by a host that a redirect led to.

    A download that the job's time limit stopped is a `model_fetch_error`
    too, with `"reason": "timed_out"` and what it reached beside the message
    (`too_slow`). Another job could get past it, and could take as long: the
    runner reads the mark and does not try that job again."""
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
    slow = [f for f in failed if f.get("reason") == TIMED_OUT]
    if slow:
        reached = {key: value for key, value in slow[0].items() if key != "error"}
        return "model_fetch_error", {"message": slow[0]["error"], **reached,
                                     "failed": failed[:20]}
    first = failed[0]
    where = f"{first['model']}: " if first.get("model") else ""
    return "model_fetch_error", {"message": f"model download failed: {where}{first['error']}",
                                 "failed": failed[:20]}
