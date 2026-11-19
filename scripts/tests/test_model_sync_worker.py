"""src/model_sync.py: what a job fetches to the worker's own disk
(`fetch_local`), against a local HTTP server that plays the model hosts, and
which host the Hugging Face token is sent to."""
import errno
import hashlib
import http.client
import http.server
import itertools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import model_sync  # noqa: E402

MIB = 1024 * 1024


class Host:
    """A local HTTP server. A path answers with what is queued for it, one
    answer per request and the last one for every request after it: (status,
    headers, body). It records each request with its Host and Authorization
    headers, and counts how many it is answering at once.

    Three headers of an answer make it slow, until the test lets go (`hold`).
    X-Slow: the first so many bytes, then nothing. X-Silent: nothing at all,
    not even the status line. X-Trickle: so many bytes a second, by the
    clock, after the X-First bytes it sends at once."""

    def __init__(self):
        self.routes: dict[str, list] = {}
        self.requests: list[dict] = []
        self.lock = threading.Condition()
        self.active = self.peak = 0
        self.together = 1   # a request is held until this many are in flight
        self.hold = threading.Event()   # set: a slow answer sends its second half
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                try:
                    owner.answer(self)
                except OSError:
                    pass  # the client went away

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def answer(self, request):
        with self.lock:
            self.requests.append({"host": request.headers.get("Host"), "path": request.path,
                                  "auth": request.headers.get("Authorization")})
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.lock.notify_all()
            self.lock.wait_for(lambda: self.peak >= self.together, timeout=10)
            queued = self.routes.get(request.path) or [(404, {}, b"not here")]
            status, headers, body = queued.pop(0) if len(queued) > 1 else queued[0]
        try:
            if headers.get("X-Silent"):
                self.hold.wait(60)
            request.send_response(status)
            for name, value in headers.items():
                request.send_header(name, value)
            if "Content-Length" not in headers:
                request.send_header("Content-Length", str(len(body)))
            request.end_headers()
            if headers.get("X-Slow"):
                # The first part, then nothing until the test lets go.
                half = int(headers["X-Slow"])
                request.wfile.write(body[:half])
                request.wfile.flush()
                self.hold.wait(60)
                body = body[half:]
            if headers.get("X-Trickle"):
                rate, began, sent = int(headers["X-Trickle"]), time.monotonic(), 0
                first = int(headers.get("X-First", 0))
                while sent < len(body) and not self.hold.is_set():
                    due = min(len(body), first + int((time.monotonic() - began) * rate))
                    request.wfile.write(body[sent:due])
                    request.wfile.flush()
                    sent = due
                    self.hold.wait(0.01)
                body = body[sent:]
            request.wfile.write(body)
        finally:
            with self.lock:
                self.active -= 1

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def as_https(self, monkeypatch):
        """Send every https request to this server, whatever host it names, so
        a test can use the real URLs. The request keeps its Host header."""
        port = self.port

        def https_open(handler, req):
            def connect(host, timeout=None, **kw):
                return http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            return handler.do_open(connect, req)

        monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_open)

    def close(self):
        self.hold.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def host():
    server = Host()
    yield server
    server.close()


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    """A ComfyUI checkout as the worker has it: `models/` is ignored by git,
    which still tracks a placeholder and a config in it. The module's paths
    point at it, the disk is large, and a retry does not wait."""
    root = tmp_path / "comfyui"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "configs").mkdir()
    (root / "models" / "checkpoints" / "put_checkpoints_here").write_text("")
    (root / "models" / "configs" / "v1-inference.yaml").write_text("model: {}\n")
    (root / ".gitignore").write_text("/models/\n")
    subprocess.run(["git", "init", "-q", "-b", "master", str(root)], check=True)
    git(root, "add", "-f", ".gitignore", "models")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    monkeypatch.setattr(model_sync, "COMFY_ROOT", root)
    monkeypatch.setattr(model_sync, "LOCAL_ROOT", root / "models")
    monkeypatch.setattr(model_sync, "LOCAL_STATE", root / "models" / ".comfyci-fetched.json")
    monkeypatch.setattr(model_sync, "HF_CACHE", tmp_path / "hf-cache")
    monkeypatch.setattr(model_sync, "RETRY_WAIT_S", 0)
    monkeypatch.setattr(model_sync, "_free_bytes", lambda: 10**12)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("no_proxy", "*")
    return root


def body_of(name: str, size: int = 2000) -> bytes:
    return (name.encode() * (size // len(name) + 1))[:size]


def model(host, name, body=None, directory="vae", **extra):
    """A manifest model entry whose file the host serves under /<name>."""
    body = body_of(name) if body is None else body
    host.routes.setdefault(f"/{name}", [(200, {}, body)])
    return {"directory": directory, "name": name, "url": host.url(f"/{name}"),
            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), **extra}


def state_of(comfy) -> dict:
    return json.loads((comfy / "models" / ".comfyci-fetched.json").read_text())


def test_a_job_downloads_its_models_into_comfyuis_own_folder(comfy, host):
    a, b = model(host, "a.safetensors"), model(host, "sub/b.safetensors", directory="loras")
    report = model_sync.fetch_local([a, b, a])
    assert (comfy / "models" / "vae" / "a.safetensors").read_bytes() == body_of("a.safetensors")
    assert (comfy / "models" / "loras" / "sub" / "b.safetensors").exists()
    assert report["files"] == ["vae/a.safetensors", "loras/sub/b.safetensors"]
    assert report["bytes"] == 4000 and report["seconds"] >= 0 and report["streams"] == 4
    assert (report["reused"], report["evicted"], report["failed"]) == ([], [], [])
    assert report["token"] is False and report["reused_bytes"] == 0
    # A file listed twice is one download, and no part file is left.
    assert sorted(host.paths()) == ["/a.safetensors", "/sub/b.safetensors"]
    assert not list((comfy / "models").rglob("*.part"))
    entry = state_of(comfy)["vae/a.safetensors"]
    assert (entry["url"], entry["sha256"], entry["bytes"]) == (a["url"], a["sha256"], 2000)
    assert model_sync.fetch_failure(report) is None
    assert model_sync.fetch_brief(report) == {
        "files": 2, "bytes": 4000, "seconds": report["seconds"], "mb_per_s": report["mb_per_s"],
        "streams": 4, "token": False, "reused": 0, "reused_bytes": 0, "evicted": 0}
    # What the folder holds beyond the models is what git tracks, and the list.
    assert git(comfy, "status", "--porcelain") == ""


def test_a_file_that_is_there_is_used_again(comfy, host):
    a, b = model(host, "a.safetensors"), model(host, "b.safetensors")
    model_sync.fetch_local([a])
    report = model_sync.fetch_local([a, b])
    assert report["reused"] == ["vae/a.safetensors"] and report["reused_bytes"] == 2000
    assert report["files"] == ["vae/b.safetensors"] and report["bytes"] == 2000
    assert host.paths() == ["/a.safetensors", "/b.safetensors"]
    # Everything is there: no request at all, and no speed to report.
    report = model_sync.fetch_local([a, b])
    assert (report["files"], report["bytes"], report["mb_per_s"]) == ([], 0, None)
    assert sorted(report["reused"]) == ["vae/a.safetensors", "vae/b.safetensors"]
    assert len(host.requests) == 2


def test_a_file_is_fetched_again_when_it_is_not_the_one_asked_for(comfy, host):
    a = model(host, "a.safetensors")
    model_sync.fetch_local([a])
    path = comfy / "models" / "vae" / "a.safetensors"
    # Another sha256 under the same name, another url, a file that was cut.
    new_body = body_of("other", 3000)
    host.routes["/a.safetensors"] = [(200, {}, new_body)]
    changed = {**a, "sha256": hashlib.sha256(new_body).hexdigest(), "bytes": 3000}
    assert model_sync.fetch_local([changed])["files"] == ["vae/a.safetensors"]
    assert path.read_bytes() == new_body
    host.routes["/moved"] = [(200, {}, new_body)]
    moved = {**changed, "url": host.url("/moved")}
    assert model_sync.fetch_local([moved])["files"] == ["vae/a.safetensors"]
    path.write_bytes(new_body[:100])
    assert model_sync.fetch_local([moved])["files"] == ["vae/a.safetensors"]
    assert path.read_bytes() == new_body
    assert host.paths() == ["/a.safetensors", "/a.safetensors", "/moved", "/moved"]


def test_a_wrong_sha256_is_deleted_and_not_tried_again(comfy, host):
    good, bad = model(host, "good.safetensors"), model(host, "bad.safetensors")
    bad["sha256"] = "0" * 64
    report = model_sync.fetch_local([good, bad], streams=1)
    got = hashlib.sha256(body_of("bad.safetensors")).hexdigest()
    assert report["failed"] == [{"model": "vae/bad.safetensors",
                                 "error": f"sha256 mismatch: got {got}"}]
    assert report["files"] == ["vae/good.safetensors"]
    folder = comfy / "models" / "vae"
    assert sorted(p.name for p in folder.iterdir()) == ["good.safetensors"]
    assert host.paths().count("/bad.safetensors") == 1
    assert "vae/bad.safetensors" not in state_of(comfy)
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error"
    assert failure["message"] == ("model download failed: vae/bad.safetensors: "
                                  f"sha256 mismatch: got {got}")
    assert failure["failed"] == report["failed"]


def test_a_rate_limit_waits_as_long_as_the_host_says(comfy, host, monkeypatch):
    waits = []
    monkeypatch.setattr(model_sync, "_wait", lambda stop, seconds: waits.append(seconds))
    a = model(host, "a.safetensors")
    host.routes["/a.safetensors"] = [
        (429, {"RateLimit": '"resolvers";r=0;t=55',
               "RateLimit-Policy": '"fixed window";"resolvers";q=3000;w=300'}, b"slow down"),
        (429, {"RateLimit": '"resolvers";r=0;t=900'}, b"slow down"),
        (200, {}, body_of("a.safetensors"))]
    report = model_sync.fetch_local([a])
    assert report["files"] == ["vae/a.safetensors"] and report["failed"] == []
    assert waits == [55, 300]  # the header's seconds, and never more than 300
    assert len(host.requests) == 3
    # The header in its other shapes.
    wait = model_sync.rate_limit_wait_s
    assert wait({"RateLimit": '"api";r=12;t=7'}) == 7
    assert wait({"RateLimit": "t=3"}) == 3
    assert wait({"Retry-After": "40"}) == 40
    monkeypatch.setattr(model_sync, "RETRY_WAIT_S", 5)
    assert wait({"RateLimit": '"api";r=0', "Retry-After": "soon"}) == 5
    assert wait({}) == 5


def test_a_rate_limit_that_does_not_end_is_a_fetch_error(comfy, host, monkeypatch):
    monkeypatch.setattr(model_sync, "_wait", lambda stop, seconds: None)
    a = model(host, "a.safetensors")
    host.routes["/a.safetensors"] = [(429, {"RateLimit": '"resolvers";r=0;t=5'}, b"")]
    report = model_sync.fetch_local([a])
    assert len(host.requests) == 3  # three tries a file
    assert report["failed"] == [{"model": "vae/a.safetensors",
                                 "error": "HTTP 429 Too Many Requests", "http_status": 429,
                                 "host": "127.0.0.1", "repo": None}]
    assert model_sync.fetch_failure(report)[0] == "model_fetch_error"


def test_a_server_error_and_a_cut_connection_are_tried_again(comfy, host):
    a, b = model(host, "a.safetensors"), model(host, "b.safetensors")
    host.routes["/a.safetensors"] = [(503, {}, b"busy"), (200, {}, body_of("a.safetensors"))]
    # The host announces 2000 bytes and ends after 700.
    host.routes["/b.safetensors"] = [(200, {"Content-Length": "2000"}, b"x" * 700),
                                     (200, {}, body_of("b.safetensors"))]
    report = model_sync.fetch_local([a, b], streams=1)
    assert report["failed"] == [] and len(report["files"]) == 2
    assert (comfy / "models" / "vae" / "b.safetensors").read_bytes() == body_of("b.safetensors")
    assert host.paths() == ["/a.safetensors"] * 2 + ["/b.safetensors"] * 2
    # A host that stays down: three tries, then the file has failed.
    c = model(host, "c.safetensors")
    host.routes["/c.safetensors"] = [(500, {}, b"down")]
    report = model_sync.fetch_local([c])
    assert host.paths().count("/c.safetensors") == 3
    assert report["failed"][0]["error"] == "HTTP 500 Internal Server Error"
    assert model_sync.fetch_failure(report)[0] == "model_fetch_error"
    assert not list((comfy / "models").rglob("*.part"))


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_refused_file_is_an_access_error_and_is_not_tried_again(comfy, host, status):
    path = "/Lightricks/LTX-2.5/resolve/main/ltx.safetensors"
    host.routes[path] = [(status, {}, b"no")]
    gated = {"directory": "checkpoints", "name": "ltx.safetensors", "url": host.url(path),
             "sha256": "1" * 64, "bytes": 10}
    report = model_sync.fetch_local([gated])
    assert host.paths() == [path]  # one request: no retry
    (failed,) = report["failed"]
    assert (failed["model"], failed["http_status"]) == ("checkpoints/ltx.safetensors", status)
    assert (failed["host"], failed["repo"]) == ("127.0.0.1", "Lightricks/LTX-2.5")
    kind, failure = model_sync.fetch_failure(report)
    assert kind == "model_access_error"
    assert failure == {"message": f"the model host answered {status} for Lightricks/LTX-2.5",
                       "host": "127.0.0.1", "repo": "Lightricks/LTX-2.5", "http_status": status,
                       "failed": report["failed"]}
    assert not (comfy / "models" / "checkpoints" / "ltx.safetensors").exists()
    assert not list((comfy / "models").rglob("*.part"))


def test_the_repository_of_a_model_url():
    repo = model_sync.repo_of
    assert repo("https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors") \
        == ("huggingface.co", "Comfy-Org/flux1-dev")
    assert repo("https://huggingface.co/datasets/org/set/resolve/main/x.bin") == (
        "huggingface.co", "datasets/org/set")
    assert repo("https://raw.githubusercontent.com/org/repo/abc/x.png") == (
        "raw.githubusercontent.com", "org/repo")
    assert repo("https://example.org/x.bin") == ("example.org", None)


def test_a_refused_file_stops_the_other_downloads(comfy, host, monkeypatch):
    """The job is over when one file cannot be had, and its time is billed:
    a download that has not finished ends, and one that has not started does
    not start."""
    big = body_of("big", 3 * model_sync.CHUNK_BYTES)
    slow = model(host, "slow.safetensors", big)
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": str(model_sync.CHUNK_BYTES)}, big)]
    host.routes["/gated"] = [(403, {}, b"no")]
    gated = {"directory": "vae", "name": "gated.safetensors", "url": host.url("/gated")}
    later = model(host, "later.safetensors")
    host.together = 2  # the refusal comes while the slow file is on its way
    repo_of = model_sync.repo_of

    def refused(url):
        host.hold.set()  # the failure is known: the slow file may go on now
        return repo_of(url)

    monkeypatch.setattr(model_sync, "repo_of", refused)
    report = model_sync.fetch_local([slow, gated, later], streams=2)
    assert [f["model"] for f in report["failed"]] == ["vae/gated.safetensors"]
    assert report["files"] == [] and "/later.safetensors" not in host.paths()
    assert not list((comfy / "models" / "vae").iterdir())
    assert model_sync.fetch_failure(report)[0] == "model_access_error"


def test_four_files_at_once(comfy, host):
    models = [model(host, f"m{i}.safetensors") for i in range(6)]
    host.together = 4
    report = model_sync.fetch_local(models)
    assert host.peak == 4 and report["streams"] == 4 and len(report["files"]) == 6
    # The files are listed in the order they were asked for.
    assert report["files"] == [f"vae/m{i}.safetensors" for i in range(6)]


def test_one_stream_downloads_one_file_at_a_time(comfy, host):
    models = [model(host, f"m{i}.safetensors") for i in range(4)]
    report = model_sync.fetch_local(models, streams=1)
    assert host.peak == 1 and report["streams"] == 1 and len(report["files"]) == 4
    assert host.paths() == [f"/m{i}.safetensors" for i in range(4)]


def _fake_disk(monkeypatch, comfy, tmp_path, capacity: int):
    """A disk of `capacity` bytes that holds the models folder and the cache."""
    def free() -> int:
        used = sum(p.stat().st_size for root in (comfy / "models", tmp_path / "hf-cache")
                   for p in root.rglob("*") if p.is_file() and p.suffix == ".safetensors")
        return capacity - used
    monkeypatch.setattr(model_sync, "_free_bytes", free)
    monkeypatch.setattr(model_sync, "FREE_MARGIN_BYTES", 500)


def test_room_is_made_from_the_files_this_job_does_not_need_oldest_first(comfy, host, tmp_path,
                                                                        monkeypatch):
    old = [model(host, f"old{i}.safetensors", body_of(f"old{i}", 3000)) for i in range(3)]
    kept = model(host, "kept.safetensors", body_of("kept", 3000))
    stamps = itertools.count(1000, 100)
    monkeypatch.setattr(model_sync, "time", types.SimpleNamespace(
        time=lambda: next(stamps), monotonic=time.monotonic))
    model_sync.fetch_local([old[1]])   # the order of use, not of the names
    model_sync.fetch_local([old[0]])
    model_sync.fetch_local([old[2]])
    model_sync.fetch_local([kept])
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    (cache / "blob.safetensors").write_bytes(b"c" * 2000)
    # 14,000 bytes are on a disk of 16,000. The job needs 5,000 and 500 to spare.
    _fake_disk(monkeypatch, comfy, tmp_path, 16000)
    new = model(host, "new.safetensors", body_of("new", 5000))
    report = model_sync.fetch_local([kept, new])
    assert report["evicted"] == ["vae/old1.safetensors", "vae/old0.safetensors"]
    assert report["files"] == ["vae/new.safetensors"] and report["failed"] == []
    assert report["reused"] == ["vae/kept.safetensors"]
    left = sorted(p.name for p in (comfy / "models" / "vae").iterdir())
    assert left == ["kept.safetensors", "new.safetensors", "old2.safetensors"]
    assert sorted(state_of(comfy)) == ["vae/kept.safetensors", "vae/new.safetensors",
                                       "vae/old2.safetensors"]
    assert (cache / "blob.safetensors").exists()  # the cache goes only when files are not enough
    assert model_sync.fetch_brief(report)["evicted"] == 2


def test_the_cache_of_other_downloaders_goes_when_the_files_are_not_enough(comfy, host, tmp_path,
                                                                          monkeypatch):
    old = model(host, "old.safetensors", body_of("old", 3000))
    model_sync.fetch_local([old])
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    (cache / "blob.safetensors").write_bytes(b"c" * 4000)
    _fake_disk(monkeypatch, comfy, tmp_path, 9000)  # 2,000 free
    new = model(host, "new.safetensors", body_of("new", 7000))
    report = model_sync.fetch_local([new])
    assert report["evicted"] == ["vae/old.safetensors"] and report["failed"] == []
    assert report["files"] == ["vae/new.safetensors"] and not (tmp_path / "hf-cache").exists()


def test_a_cache_that_is_not_on_the_workers_disk_is_never_removed(comfy, host, tmp_path,
                                                                  monkeypatch):
    """`HF_HOME` names the cache, and an endpoint may point it at the network
    volume to keep it between workers. Removing it there frees nothing on the
    worker's own disk, and takes the cache from every worker."""
    volume = tmp_path / "runpod-volume"
    kept = [volume / "hf" / "hub" / "blob.safetensors", volume / "models" / "vae" / "v.pt",
            comfy / "main.py", comfy / "models" / "checkpoints" / "put_checkpoints_here"]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"k" * 100)
    monkeypatch.setattr(model_sync, "VOLUME_ROOT", volume)
    monkeypatch.setattr(model_sync, "FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(model_sync, "_free_bytes", lambda: 1000)  # a disk that is nearly full
    big = {**model(host, "big.safetensors"), "bytes": 5000}
    # On the volume, the volume itself, a folder that holds the volume, and
    # one that holds the models folder.
    for cache in (volume / "hf", volume, tmp_path, comfy):
        monkeypatch.setattr(model_sync, "HF_CACHE", cache)
        report = model_sync.fetch_local([big])
        assert report["failed"] == [{"error": "disk full: 0.0 GB free, 0.0 GB needed"}]
        assert all(path.exists() for path in kept), cache
    # A cache on another disk gives the models no room either.
    cache = tmp_path / "hf-cache"
    (cache / "hub").mkdir(parents=True)
    monkeypatch.setattr(model_sync, "HF_CACHE", cache)
    monkeypatch.setattr(model_sync, "_device", lambda path: 7 if path.name == "hf-cache" else 1)
    model_sync.fetch_local([big])
    assert (cache / "hub").exists()
    # On the worker's own disk it goes, as before.
    monkeypatch.setattr(model_sync, "_device", lambda path: 1)
    model_sync.fetch_local([big])
    assert not cache.exists() and all(path.exists() for path in kept)
    assert host.requests == []


def test_a_full_disk_ends_before_any_download(comfy, host, tmp_path, monkeypatch):
    old = model(host, "old.safetensors", body_of("old", 3000))
    model_sync.fetch_local([old])
    seen = len(host.requests)
    monkeypatch.setattr(model_sync, "_free_bytes", lambda: 12_300_000_000)
    big = {**model(host, "big.safetensors"), "bytes": 60 * 10**9}
    small = model(host, "small.safetensors")
    report = model_sync.fetch_local([small, big])
    assert report["failed"] == [{"error": "disk full: 12.3 GB free, 65.0 GB needed"}]
    assert report["files"] == [] and len(host.requests) == seen  # nothing was asked for
    assert report["evicted"] == ["vae/old.safetensors"]  # it tried to make room first
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error"
    assert failure["message"] == "model download failed: disk full: 12.3 GB free, 65.0 GB needed"
    # A file the job needs and has is never removed for room.
    model_sync.fetch_local([small])
    report = model_sync.fetch_local([small, big])
    assert report["reused"] == ["vae/small.safetensors"] and report["evicted"] == []
    assert (comfy / "models" / "vae" / "small.safetensors").exists()


def test_what_nobody_knows_is_removed_and_what_git_tracks_is_kept(comfy, host):
    models = comfy / "models"
    known = model(host, "known.safetensors")
    model_sync.fetch_local([known])
    # An earlier workflow downloaded a file by itself, and a download was cut.
    (models / "checkpoints" / "stray.safetensors").write_bytes(b"s" * 100)
    (models / "sams").mkdir()
    (models / "sams" / "auto.pth").write_bytes(b"a" * 100)
    (models / "vae" / "half.safetensors.part").write_bytes(b"h" * 100)
    (models / "configs" / "v1-inference.yaml").write_text("model: {changed: true}\n")
    report = model_sync.fetch_local([known])
    assert report["reused"] == ["vae/known.safetensors"] and report["failed"] == []
    left = sorted(p.relative_to(models).as_posix() for p in models.rglob("*") if p.is_file())
    assert left == [".comfyci-fetched.json", "checkpoints/put_checkpoints_here",
                    "configs/v1-inference.yaml", "vae/known.safetensors"]
    # A tracked file stays as it is, also one that was changed.
    assert "changed" in (models / "configs" / "v1-inference.yaml").read_text()
    # A file the list names and the disk lost is no longer listed.
    (models / "vae" / "known.safetensors").unlink()
    model_sync.fetch_local([])
    assert state_of(comfy) == {}


def test_when_git_cannot_say_what_it_tracks_only_part_files_go(comfy, host, monkeypatch):
    models = comfy / "models"
    (models / "checkpoints" / "stray.safetensors").write_bytes(b"s" * 100)
    (models / "checkpoints" / "half.safetensors.part").write_bytes(b"h" * 100)

    def no_git(*args, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(model_sync.subprocess, "run", no_git)
    assert model_sync._tracked() is None
    report = model_sync.fetch_local([model(host, "a.safetensors")])
    assert report["files"] == ["vae/a.safetensors"]
    assert sorted(p.name for p in (models / "checkpoints").iterdir()) == [
        "put_checkpoints_here", "stray.safetensors"]


def test_a_name_that_leaves_the_models_folder_is_not_fetched(comfy, host):
    report = model_sync.fetch_local([{"directory": "..", "name": "main.py",
                                      "url": host.url("/main.py")},
                                     model(host, "a.safetensors")])
    plain = "its name is not a plain path below the models folder"
    assert report["failed"] == [{"model": "../main.py", "error": plain}]
    assert host.requests == [] and not (comfy / "main.py").exists()
    # A name that stays inside and is not the path of its file: the list would
    # know it as vae/../loras/x, the disk as loras/x, and every job would
    # remove it as unknown and download it again.
    for name in ("../loras/x.safetensors", "./x.safetensors", "sub//x.safetensors"):
        report = model_sync.fetch_local([{**model(host, "x.safetensors"), "name": name}])
        assert report["failed"] == [{"model": f"vae/{name}", "error": plain}], name
    assert host.requests == [] and not list((comfy / "models").rglob("x.safetensors"))


def test_a_model_whose_name_ends_like_a_part_file_is_kept(comfy, host):
    weights = model(host, "weights.part")
    assert model_sync.fetch_local([weights])["files"] == ["vae/weights.part"]
    (comfy / "models" / "vae" / "other.part").write_bytes(b"left by a killed download")
    report = model_sync.fetch_local([weights])
    assert report["reused"] == ["vae/weights.part"] and host.paths() == ["/weights.part"]
    assert sorted(p.name for p in (comfy / "models" / "vae").iterdir()) == ["weights.part"]


def test_a_url_that_is_not_http_is_not_opened(comfy, host, tmp_path):
    """urllib opens a `file:` URL too: a file of the worker itself would be
    copied into the models folder as if it were the model."""
    own = tmp_path / "own.txt"
    own.write_text("a file of the worker")
    for url in (own.as_uri(), "ftp://example.org/a.safetensors", "nonsense"):
        report = model_sync.fetch_local([{"directory": "vae", "name": "a.safetensors",
                                          "url": url}])
        assert report["failed"] == [{
            "model": "vae/a.safetensors",
            "error": f"bad url: not an http or https url: {url}"}], url
    # An entry without a url is a failed file too, not a job that breaks.
    report = model_sync.fetch_local([{"directory": "vae", "name": "a.safetensors"},
                                     model(host, "b.safetensors")])
    assert report["failed"] == [{"model": "vae/a.safetensors", "error": "it has no url"}]
    assert model_sync.fetch_failure(report)[0] == "model_fetch_error"
    assert host.requests == [] and not list((comfy / "models").rglob("*.safetensors"))
    # The volume sync opens its URLs the same way.
    with pytest.raises(ValueError, match="not an http or https url"):
        model_sync._open(own.as_uri(), None)


def test_a_file_without_a_sha256_must_have_the_size_its_entry_says(comfy, host):
    """A host that answers with a sign-in page says 200. With a sha256 the
    page fails the check. Without one, its size is all there is to go by."""
    host.routes["/f.safetensors"] = [(200, {}, b"<html>sign in</html>")]
    entry = {"directory": "vae", "name": "f.safetensors", "url": host.url("/f.safetensors")}
    report = model_sync.fetch_local([{**entry, "bytes": 5 * 10**9}])
    assert report["failed"] == [{"model": "vae/f.safetensors",
                                 "error": "size mismatch: got 20 bytes, not 5000000000"}]
    assert host.paths() == ["/f.safetensors"]  # the same bytes would come again: one try
    assert not (comfy / "models" / "vae" / "f.safetensors").exists()
    assert not list((comfy / "models").rglob("*.part"))
    # A sha256 that matches decides, whatever the entry says about the size.
    page = hashlib.sha256(b"<html>sign in</html>").hexdigest()
    assert model_sync.fetch_local([{**entry, "bytes": 7, "sha256": page}])["failed"] == []
    # An entry with neither is taken as it comes, as the volume sync takes it.
    bare = {**entry, "name": "g.safetensors"}
    assert model_sync.fetch_local([bare])["files"] == ["vae/g.safetensors"]


def test_a_file_that_cannot_be_put_in_place_is_a_failed_download(comfy, host):
    """A folder where the file goes, a file where its folder goes: the job
    ends as a failed download, not as a worker that broke with a part file
    left behind."""
    (comfy / "models" / "vae" / "d.safetensors").mkdir(parents=True)
    report = model_sync.fetch_local([model(host, "d.safetensors")])
    (failed,) = report["failed"]
    assert failed["model"] == "vae/d.safetensors"
    assert failed["error"].startswith(("IsADirectoryError", "PermissionError"))
    assert host.paths() == ["/d.safetensors"] and report["files"] == []
    assert not list((comfy / "models").rglob("*.part"))
    assert model_sync.fetch_failure(report)[0] == "model_fetch_error"
    # ComfyUI's own placeholder file, named as a folder.
    inside = model(host, "e.safetensors", directory="checkpoints/put_checkpoints_here")
    report = model_sync.fetch_local([inside])
    (failed,) = report["failed"]
    assert failed["model"] == "checkpoints/put_checkpoints_here/e.safetensors"
    assert failed["error"].startswith(("FileExistsError", "NotADirectoryError"))
    assert "/e.safetensors" not in host.paths()


def test_a_list_this_module_did_not_write_is_not_believed(comfy, host, tmp_path, monkeypatch):
    """The list decides which files may be removed for room. An entry that is
    damaged must not break every job of the worker, and one that names a file
    outside the models folder must not get that file removed."""
    a = model(host, "a.safetensors")
    model_sync.fetch_local([a])
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    whole = {"url": a["url"], "sha256": a["sha256"], "bytes": 2000, "used_at": 5}
    (comfy / "models" / ".comfyci-fetched.json").write_text(json.dumps({
        "vae/a.safetensors": {**whole, "bytes": "2000", "used_at": "then"},
        "vae/b.safetensors": {"used_at": 7}, "vae/c.safetensors": "junk",
        "../../outside.txt": whole, "vae/../../../outside.txt": whole}))
    monkeypatch.setattr(model_sync, "FREE_MARGIN_BYTES", 0)
    monkeypatch.setattr(model_sync, "_free_bytes", lambda: 0)  # every spare file would go
    report = model_sync.fetch_local([model(host, "new.safetensors")])
    assert report["failed"] == [{"error": "disk full: 0.0 GB free, 0.0 GB needed"}]
    assert report["evicted"] == [] and outside.read_text() == "keep me"
    # The file of the damaged entry is one that no list knows now: it went.
    assert not (comfy / "models" / "vae" / "a.safetensors").exists()
    assert state_of(comfy) == {}
    assert host.paths() == ["/a.safetensors"]


def test_a_download_that_was_killed_leaves_a_part_file_and_the_next_job_removes_it(
        comfy, host, tmp_path):
    """RunPod ends a job, the runner cancels it, a worker is replaced: the
    process dies in the middle of a file. The file was never under its final
    name, so nothing takes half a model for a whole one, and the next job
    removes what is left."""
    chunk = model_sync.CHUNK_BYTES
    body = body_of("slow", 2 * chunk + 1000)
    slow = model(host, "slow.safetensors", body)
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": str(chunk + 1000)}, body),
                                        (200, {}, body)]
    script = ("import json, sys\n"
              f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
              "import model_sync\n"
              "model_sync.FREE_MARGIN_BYTES = 0\n"
              "model_sync.fetch_local(json.loads(sys.argv[1]))\n")
    env = {**os.environ, "COMFY_ROOT": str(comfy), "no_proxy": "*", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("HF_TOKEN", None)
    proc = subprocess.Popen([sys.executable, "-c", script, json.dumps([slow])], env=env)
    part = comfy / "models" / "vae" / "slow.safetensors.part"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (part.exists()
                                                   and part.stat().st_size >= chunk):
            assert proc.poll() is None, "the download ended before it could be killed"
            time.sleep(0.05)
        assert part.exists() and part.stat().st_size >= chunk
    finally:
        proc.kill()
        proc.wait(30)
    assert part.exists() and not (comfy / "models" / "vae" / "slow.safetensors").exists()
    assert state_of(comfy) == {}  # the list never named it
    host.hold.set()
    report = model_sync.fetch_local([slow])
    assert report["files"] == ["vae/slow.safetensors"] and report["failed"] == []
    assert not part.exists()
    assert (comfy / "models" / "vae" / "slow.safetensors").read_bytes() == body


# -- the token -----------------------------------------------------------------------------
def test_which_url_may_carry_the_token():
    token = model_sync.token_for
    assert token("https://huggingface.co/org/repo/resolve/main/x.safetensors", "t") == "t"
    assert token("https://HuggingFace.co:443/org/repo/resolve/main/x", "t") == "t"
    for url in ("https://cdn-lfs.huggingface.co/repos/ab/cd",   # a storage host of its own
                "https://huggingface.co.example.org/org/repo/resolve/main/x",
                "https://example.org/huggingface.co/x",
                "https://nothuggingface.co/x", "https://cas-bridge.xethub.hf.co/x",
                "http://huggingface.co/org/repo/resolve/main/x",   # not encrypted
                "https://huggingface.co:8443/x", "https://huggingface.co@example.org/x",
                # urllib would connect to the whole of "example.org@huggingface.co".
                "https://example.org@huggingface.co/x", "https://u:p@huggingface.co/x",
                "https://example.org\\@huggingface.co/x",
                "https://raw.githubusercontent.com/org/repo/abc/x.png", "not a url",
                "https://[huggingface.co/x"):
        assert token(url, "t") is None, url
    assert token("https://huggingface.co/x", None) is None
    assert token("https://huggingface.co/x", "") is None
    # The report names the same host when it says what a refused file needs.
    import run_report
    assert run_report.TOKEN_HOST == model_sync.TOKEN_HOST == "huggingface.co"


def test_what_counts_as_a_token():
    clean = model_sync.clean_token
    assert clean("hf_AbC123") == "hf_AbC123"
    # A secret that was pasted with its line break, or with blanks around it.
    assert clean("hf_AbC123\n") == clean(" hf_AbC123\r\n") == clean("\thf_AbC123 ") == "hf_AbC123"
    assert clean("abc.DEF-123_~+/==") == "abc.DEF-123_~+/=="
    # What no Bearer header can carry is no token.
    for junk in (None, "", " \n", "hf_a b", "hf_a\nb", "\u200bhf_AbC123", "hf_\u00e9", "hf_'x'",
                 "Bearer hf_AbC123", 7, b"hf_AbC123"):
        assert clean(junk) is None, junk


def test_a_token_with_a_line_break_at_its_end_is_sent_without_it(comfy, host, monkeypatch):
    """The endpoint's HF_TOKEN is a secret someone pasted. With a line break
    at its end the HTTP library refused the header before any connection, and
    every job of the lane failed."""
    host.as_https(monkeypatch)
    body = body_of("weights")
    host.routes[HF_FILE] = [(200, {}, body)]
    m = {"directory": "vae", "name": "m.safetensors", "url": "https://huggingface.co" + HF_FILE,
         "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    monkeypatch.setenv("HF_TOKEN", "hf_secret\n")
    report = model_sync.fetch_local([m])
    assert report["failed"] == [] and report["token"] is True
    assert [r["auth"] for r in host.requests] == ["Bearer hf_secret"]
    # A value no header can carry is no token: the file is asked for without
    # one, and the report says that there was none.
    monkeypatch.setenv("HF_TOKEN", "\u200bhf_secret")
    report = model_sync.fetch_local([{**m, "name": "n.safetensors"}])
    assert report["failed"] == [] and report["token"] is False
    assert host.requests[-1]["auth"] is None
    # The token a caller hands over is cleaned the same way, and comes first.
    report = model_sync.fetch_local([{**m, "name": "o.safetensors"}], hf_token=" from_job\r\n")
    assert report["token"] is True and host.requests[-1]["auth"] == "Bearer from_job"


def test_no_failure_message_holds_the_token(comfy, host, tmp_path, monkeypatch):
    """What a download says when it fails goes into the job's answer, into
    run.json on the results branch and into the report. Whatever a library
    quotes in its error, the token is not in it."""
    token = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    said = []

    def refuse(url, dest, hf_token, stop=None):
        assert hf_token == token  # the line break is gone before anything is sent
        raise said[0](f"Invalid header value b'Bearer {hf_token}'")

    monkeypatch.setattr(model_sync, "_stream", refuse)
    m = {"directory": "vae", "name": "m.safetensors", "url": "https://huggingface.co" + HF_FILE}
    for kind, text in ((ValueError, "bad url: Invalid header value b'Bearer <token>'"),
                       (OSError, "OSError: Invalid header value b'Bearer <token>'")):
        said[:] = [kind]
        report = model_sync.fetch_local([m], hf_token=token + "\n")
        assert report["failed"] == [{"model": "vae/m.safetensors", "error": text}]
        assert token not in json.dumps([report, model_sync.fetch_failure(report)])
    # A long message is cut after the token is out of it, not before: a cut
    # through the token would leave its first part standing.
    said[:] = [lambda text: OSError("x" * 250 + text)]
    error = model_sync.fetch_local([m], hf_token=token)["failed"][0]["error"]
    assert error.endswith("b'Bearer <token>'") and token[:8] not in error
    assert len(f"OSError: {'x' * 250}Invalid header value b'Bearer {token}'") > 300
    # The volume sync, with the token from the job input as sync_models.py sends it.
    monkeypatch.setattr(model_sync, "VOLUME_ROOT", tmp_path / "volume")
    monkeypatch.setattr(model_sync, "MODELS_ROOT", tmp_path / "volume" / "models")
    monkeypatch.setattr(model_sync, "STATE_FILE", tmp_path / "volume" / ".sync-state.json")
    said[:] = [ValueError]
    report = model_sync.sync([m], token + "\n")
    assert report["failed"] == [{"model": "vae/m.safetensors",
                                 "error": "Invalid header value b'Bearer <token>'"}]
    assert token not in json.dumps(report)


HF_FILE = "/org/gated/resolve/main/m.safetensors"


def test_the_token_goes_to_huggingface_and_not_to_the_host_it_redirects_to(comfy, host,
                                                                         monkeypatch):
    host.as_https(monkeypatch)
    body = body_of("weights")
    host.routes[HF_FILE] = [(302, {"Location": "https://cdn-lfs.example.net/blob/abc"}, b"")]
    host.routes["/blob/abc"] = [(200, {}, body)]
    m = {"directory": "vae", "name": "m.safetensors", "url": "https://huggingface.co" + HF_FILE,
         "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    report = model_sync.fetch_local([m], hf_token="hf_secret")
    assert report["failed"] == [] and report["token"] is True
    assert [(r["host"], r["path"], r["auth"]) for r in host.requests] == [
        ("huggingface.co", HF_FILE, "Bearer hf_secret"),
        ("cdn-lfs.example.net", "/blob/abc", None)]
    assert (comfy / "models" / "vae" / "m.safetensors").read_bytes() == body


def test_the_token_follows_a_redirect_that_stays_on_huggingface(comfy, host, monkeypatch):
    """A repository that was renamed answers with a redirect to its new name."""
    host.as_https(monkeypatch)
    body = body_of("weights")
    renamed = "/org/new-name/resolve/main/m.safetensors"
    host.routes[HF_FILE] = [(307, {"Location": renamed}, b"")]
    host.routes[renamed] = [(302, {"Location": "https://huggingface.co.example.org/b"}, b"")]
    host.routes["/b"] = [(200, {}, body)]
    m = {"directory": "vae", "name": "m.safetensors", "url": "https://huggingface.co" + HF_FILE}
    monkeypatch.setenv("HF_TOKEN", "from_env")  # the endpoint's variable, when the job has none
    report = model_sync.fetch_local([m])
    assert report["failed"] == [] and report["token"] is True
    assert [(r["host"], r["auth"]) for r in host.requests] == [
        ("huggingface.co", "Bearer from_env"), ("huggingface.co", "Bearer from_env"),
        ("huggingface.co.example.org", None)]


def test_a_refusal_by_the_storage_host_is_tried_again(comfy, host, monkeypatch):
    """huggingface.co redirects a download to its storage host, with a link it
    signs for that request. A 403 there says nothing about a licence, and the
    next try gets a new link. It was taken for the model host's own refusal:
    one such answer ended the job for good, and the report told the owner to
    accept a licence."""
    host.as_https(monkeypatch)
    body = body_of("weights")
    host.routes[HF_FILE] = [(302, {"Location": "https://cdn.example.net/blob/abc?sig=1"}, b"")]
    host.routes["/blob/abc?sig=1"] = [(403, {}, b"Request has expired"), (200, {}, body)]
    m = {"directory": "vae", "name": "m.safetensors", "url": "https://huggingface.co" + HF_FILE,
         "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    report = model_sync.fetch_local([m], hf_token="hf_secret")
    assert report["failed"] == [] and report["files"] == ["vae/m.safetensors"]
    assert [r["host"] for r in host.requests] == ["huggingface.co", "cdn.example.net"] * 2
    # A storage host that keeps saying no: three tries, and a download that
    # failed. The message names the host, and another job may try again.
    host.routes["/blob/abc?sig=1"] = [(404, {}, b"gone")]
    report = model_sync.fetch_local([{**m, "name": "n.safetensors"}], hf_token="hf_secret")
    assert report["failed"] == [{"model": "vae/n.safetensors",
                                 "error": "HTTP 404 Not Found from cdn.example.net"}]
    assert len(host.requests) == 4 + 2 * 3
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error" and "http_status" not in failure
    # The model's own host saying no is final, also at the end of a redirect
    # that stays on it.
    renamed = "/org/new-name/resolve/main/m.safetensors"
    host.routes[HF_FILE] = [(307, {"Location": renamed}, b"")]
    host.routes[renamed] = [(403, {}, b"gated")]
    report = model_sync.fetch_local([{**m, "name": "o.safetensors"}], hf_token="hf_secret")
    assert len(host.requests) == 10 + 2
    assert model_sync.fetch_failure(report)[0] == "model_access_error"
    assert report["failed"][0]["host"] == "huggingface.co"


def test_a_file_of_another_host_never_gets_the_token(comfy, host, monkeypatch):
    host.as_https(monkeypatch)
    a = model(host, "a.safetensors")
    b = {**model(host, "b.png"), "url": "https://raw.githubusercontent.com/org/repo/abc/b.png"}
    host.routes["/org/repo/abc/b.png"] = host.routes["/b.png"]
    report = model_sync.fetch_local([a, b], hf_token="hf_secret")
    assert report["failed"] == [] and report["token"] is True  # there was one to send
    assert {r["auth"] for r in host.requests} == {None}


def test_the_volume_sync_holds_the_token_to_the_same_rule(tmp_path, host, monkeypatch):
    """The sync tested `"huggingface.co" in url`, and urllib copied the header
    to the host a download was redirected to."""
    monkeypatch.setattr(model_sync, "VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(model_sync, "MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(model_sync, "STATE_FILE", tmp_path / ".sync-state.json")
    monkeypatch.setenv("no_proxy", "*")
    host.as_https(monkeypatch)
    body = body_of("weights")
    host.routes[HF_FILE] = [(302, {"Location": "https://cdn-lfs.example.net/blob/abc"}, b"")]
    host.routes["/blob/abc"] = host.routes["/elsewhere"] = [(200, {}, body)]
    sha = hashlib.sha256(body).hexdigest()
    models = [{"directory": "vae", "name": "m.safetensors", "sha256": sha,
               "url": "https://huggingface.co" + HF_FILE},
              {"directory": "vae", "name": "n.safetensors", "sha256": sha,
               "url": "https://huggingface.co.example.org/elsewhere"}]
    report = model_sync.sync(models, "hf_secret")
    assert report["failed"] == [] and len(report["downloaded"]) == 2
    assert [(r["host"], r["auth"]) for r in host.requests] == [
        ("huggingface.co", "Bearer hf_secret"), ("cdn-lfs.example.net", None),
        ("huggingface.co.example.org", None)]
    assert model_sync.missing_models(models) == []
    assert (tmp_path / "models" / "vae" / "m.safetensors").read_bytes() == body
    # The sync still writes straight to the final path and skips what it has.
    assert model_sync.sync(models, "hf_secret")["skipped"] == ["vae/m.safetensors",
                                                               "vae/n.safetensors"]


# -- a time limit for the download ---------------------------------------------------------
def _left_behind() -> list:
    """The download threads that are still there."""
    return [t for t in threading.enumerate() if t.name.startswith(model_sync.FETCH_THREADS)]


def _gone(seconds: float = 20) -> bool:
    """Wait until no download thread is left, and say whether that came."""
    end = time.monotonic() + seconds
    while _left_behind() and time.monotonic() < end:
        time.sleep(0.02)
    return not _left_behind()


def _parts(comfy) -> list:
    return list((comfy / "models").rglob("*.part"))


def test_what_a_download_that_ran_out_of_time_says():
    said = model_sync.too_slow(12_400_000_000, 56_600_000_000, 600.2, 2, 5, 600)
    assert said == {
        "error": ("the download was too slow: 12.4 of 56.6 GB in 600 s (20.7 MB/s), "
                  "2 of 5 files complete"),
        "reason": "timed_out", "timeout_s": 600, "received_bytes": 12_400_000_000,
        "wanted_bytes": 56_600_000_000, "seconds": 600.2, "mb_per_s": 20.7, "files_done": 2,
        "files_wanted": 5}
    # A file that did not say its size: there is no total to name.
    assert model_sync.too_slow(12_400_000_000, None, 600.2, 2, 5, 600)["error"] == (
        "the download was too slow: 12.4 GB in 600 s (20.7 MB/s), 2 of 5 files complete")
    # Files of less than a GB are said in MB: "0.0 of 0.3 GB" says nothing.
    assert model_sync.too_slow(12_400_000, 334_600_000, 600.2, 0, 1, 600)["error"] == (
        "the download was too slow: 12.4 of 334.6 MB in 600 s (0.0 MB/s), 0 of 1 files complete")
    assert model_sync.too_slow(40_000_000, None, 80, 0, 1, 600)["error"] == (
        "the download was too slow: 40.0 MB in 80 s (0.5 MB/s), 0 of 1 files complete")
    # The time was up before the download began. It was not the download
    # that was slow, and there is no speed.
    nothing = model_sync.too_slow(0, 56_600_000_000, 0.0, 0, 5, 600)
    assert nothing["error"] == ("the download did not begin: cleaning up and making room took "
                                "the 600 s it was given")
    assert {key: value for key, value in nothing.items() if key != "error"} == {
        "reason": "timed_out", "timeout_s": 600, "received_bytes": 0,
        "wanted_bytes": 56_600_000_000, "seconds": 0.0, "mb_per_s": None, "files_done": 0,
        "files_wanted": 5}
    # The mark is the word the runner has for a job that used up its time.
    import submit_and_poll
    assert said["reason"] == model_sync.TIMED_OUT == submit_and_poll.TIMED_OUT == "timed_out"


def test_a_download_that_takes_too_long_is_stopped_and_says_what_it_reached(comfy, host,
                                                                           monkeypatch):
    """The runner allows a job so long for its download. A job that was still
    downloading then ended as the runner's plain timeout: no speed in its
    record, and nothing that said why. Now the worker stops and answers, also
    when a connection has gone silent in the middle of a file."""
    monkeypatch.setattr(model_sync, "FETCH_STOP_WAIT_S", 1.5)
    piece = model_sync.PIECE_BYTES
    fast = model(host, "fast.safetensors")
    body = body_of("slow", 3 * piece)
    slow = model(host, "slow.safetensors", body)
    # Two pieces and a little more, then nothing.
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": str(2 * piece + 1000)}, body)]
    started = time.monotonic()
    report = model_sync.fetch_local([fast, slow], timeout_s=0.6)
    took = time.monotonic() - started
    # The limit, the wait for the downloads to end, and no longer.
    assert 0.6 <= took < 0.6 + 1.5 + 2
    assert report["files"] == ["vae/fast.safetensors"]
    (failed,) = report["failed"]
    assert failed["error"].startswith("the download was too slow: 2.1 of 3.1 MB in ")
    assert failed["error"].endswith(", 1 of 2 files complete")
    # What the unfinished file had received counts, in whole pieces. The
    # start of the piece that was on its way can be among them, when the
    # read that waited for it has ended by then.
    received = failed["received_bytes"]
    assert 2000 + 2 * piece <= received <= 2000 + 2 * piece + 1000
    assert {key: failed[key] for key in failed if key not in ("error", "seconds", "mb_per_s")} == {
        "reason": "timed_out", "timeout_s": 0.6, "received_bytes": received,
        "wanted_bytes": 2000 + 3 * piece, "files_done": 1, "files_wanted": 2}
    # The seconds are those the downloads had, up to the stop: the wait after
    # it is no time in which anything was downloaded. The sentence then
    # agrees with the limit that a page prints beside it.
    assert 0 < failed["seconds"] <= 0.6 + 0.6
    # The report of this download is its speed: bytes and seconds are all
    # that arrived, in the time it took.
    assert (report["bytes"], report["seconds"], report["mb_per_s"]) == (
        received, failed["seconds"], failed["mb_per_s"])
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error"
    assert failure == {"message": failed["error"],
                       **{key: failed[key] for key in failed if key != "error"},
                       "failed": [failed]}
    assert model_sync.fetch_brief(report)["files"] == 1
    # The file that was complete stays, and is in the list for the next job.
    folder = comfy / "models" / "vae"
    assert (folder / "fast.safetensors").read_bytes() == body_of("fast.safetensors")
    assert sorted(state_of(comfy)) == ["vae/fast.safetensors"]
    # No part file is left when the job answers. (Only Linux, which the worker
    # is, removes a file that is still open, and ends a read on a connection
    # that is silent. Elsewhere that read ends with the connection's next
    # byte or after its 60 s.)
    assert not _parts(comfy) or os.name == "nt"
    # The host goes on. A download that was still there wakes, and the job
    # is over: it puts nothing in place and lists nothing.
    host.hold.set()
    assert _gone()
    assert not (folder / "slow.safetensors").exists()
    assert sorted(state_of(comfy)) == ["vae/fast.safetensors"]
    # The next job finds the complete file, and no part file anywhere.
    assert model_sync.fetch_local([fast])["reused"] == ["vae/fast.safetensors"]
    assert not _parts(comfy)


class Paced:
    """Stands in for a response and for the clock: each read takes the
    seconds and brings the bytes the test says, and is noted."""

    def __init__(self, reads):
        self.reads, self.now, self.calls, self.headers = list(reads), 100.0, [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def monotonic(self):
        return self.now

    def _next(self, name, want):
        if not self.reads:
            return b""
        seconds, size = self.reads.pop(0)
        self.now += seconds
        self.calls.append(name)
        assert size <= want
        return b"x" * size

    def read(self, want):
        return self._next("read", want)

    def read1(self, want):
        return self._next("read1", want)


def test_a_connection_that_got_slow_is_read_as_it_comes(tmp_path, monkeypatch):
    """A read waits until its piece is whole: on a slow connection that is
    long, and the download cannot see a stop meanwhile. So after a second or
    more that brought less than a piece a second, it takes what is there,
    however little, until a second or more has brought a piece a second."""
    piece = model_sync.PIECE_BYTES
    assert piece == 1024 * 1024
    paced = Paced([(0.01, piece)] * 3            # fast, and not yet a second
                  + [(5.0, piece)]               # this piece took 5 s
                  + [(0.3, 1000)] * 4            # 1.2 s of a little at a time
                  + [(0.1, 262144)] * 11         # fast again, in smaller parts: 1.1 s
                  + [(0.01, piece)] * 2)
    monkeypatch.setattr(model_sync, "_open", lambda url, hf_token, timeout=60: paced)
    monkeypatch.setattr(model_sync, "time", paced)
    seen = []
    stop = model_sync._Fetch()
    stop.count = lambda part, size: seen.append(size)
    size, _digest = model_sync._stream("https://example.org/x", tmp_path / "x.part", None, stop)
    assert paced.calls == ["read"] * 4 + ["read1"] * 15 + ["read"] * 2
    assert size == 6 * piece + 4 * 1000 + 11 * 262144 == seen[-1]
    assert (tmp_path / "x.part").stat().st_size == size
    # What has arrived is told after every read, so a job that is stopped
    # knows how far each download came.
    assert seen[:6] == [0, piece, 2 * piece, 3 * piece, 4 * piece, 4 * piece + 1000]
    # The volume sync has no stop to look at: it reads a chunk at a time, as
    # it always did, however slow.
    paced = Paced([(5.0, 1000), (5.0, 1000)])
    monkeypatch.setattr(model_sync, "_open", lambda url, hf_token, timeout=60: paced)
    monkeypatch.setattr(model_sync, "time", paced)
    assert model_sync._stream("https://example.org/x", tmp_path / "y", None)[0] == 2000
    assert paced.calls == ["read", "read"] and model_sync.CHUNK_BYTES == 8 * 1024 * 1024


def test_a_slow_connection_is_left_with_the_next_bytes_it_sends(comfy, host, monkeypatch):
    """A connection that still sends, 5,000 bytes a second. Its first piece
    of 6,500 bytes takes 1.3 s, and from then on the download takes what
    comes. When the time is up it ends by itself with the next bytes, long
    before the job would go on without it, and removes its own part file."""
    monkeypatch.setattr(model_sync, "PIECE_BYTES", 6500)
    body = body_of("drip", 400_000)
    drip = model(host, "drip.safetensors", body)
    host.routes["/drip.safetensors"] = [(200, {"X-Trickle": "5000"}, body)]
    started = time.monotonic()
    report = model_sync.fetch_local([drip], timeout_s=1.9)
    took = time.monotonic() - started
    # A wait can end one tick of the system's clock early: 16 ms on Windows.
    assert 1.9 - 0.05 <= took < 1.9 + 1.5
    assert (model_sync.FETCH_STOP_WAIT_S, model_sync.FETCH_CUT_AFTER_S) == (10, 8)
    (failed,) = report["failed"]
    # More than the first piece and less than two: it was read as it came.
    # The first piece is whole 1.3 s after the start, and a read of a whole
    # second one would have ended at 2.6 s.
    assert 6500 < failed["received_bytes"] < 13000 and failed["wanted_bytes"] == len(body)
    assert (failed["files_done"], failed["files_wanted"]) == (0, 1)
    assert report["files"] == [] and state_of(comfy) == {}
    assert not _parts(comfy) and _gone(5)


def test_a_host_that_does_not_answer_cannot_hold_the_job_back(comfy, host, monkeypatch):
    monkeypatch.setattr(model_sync, "FETCH_STOP_WAIT_S", 1)
    mute = model(host, "mute.safetensors")
    host.routes["/mute.safetensors"] = [(200, {"X-Silent": "1"}, body_of("mute.safetensors"))]
    started = time.monotonic()
    report = model_sync.fetch_local([mute], timeout_s=0.6)
    assert time.monotonic() - started < 0.6 + 1 + 2
    (failed,) = report["failed"]
    assert (failed["received_bytes"], failed["mb_per_s"], failed["files_done"]) == (0, 0.0, 0)
    assert (report["bytes"], report["mb_per_s"], report["files"]) == (0, 0.0, [])
    # The seconds it says are the limit's, not those and the wait after it.
    assert 0 < failed["seconds"] <= 0.6 + 0.4 and report["seconds"] == failed["seconds"]
    assert not _parts(comfy)
    # The host answers at last, with the whole file. The job is over: the
    # download opens no part file, and nothing is put in place.
    host.hold.set()
    assert _gone()
    assert not list((comfy / "models").rglob("mute*")) and state_of(comfy) == {}


@pytest.mark.parametrize("fast_pieces", [0, 1])
def test_a_read_that_waits_for_its_piece_is_ended_by_the_job(comfy, host, monkeypatch,
                                                             fast_pieces):
    """A read waits until its piece is whole. On a connection that sends 20
    bytes a second, from its first byte on or after a fast first piece, that
    is 14 hours for one piece, and nothing has judged the connection slow
    yet. Such a download held its thread, its connection and its open part
    file long after the job had answered. Now the job shuts the connection."""
    monkeypatch.setattr(model_sync, "FETCH_STOP_WAIT_S", 3)
    monkeypatch.setattr(model_sync, "FETCH_CUT_AFTER_S", 0.3)
    piece = model_sync.PIECE_BYTES
    body = body_of("drip", 3 * piece)
    drip = model(host, "drip.safetensors", body)
    host.routes["/drip.safetensors"] = [
        (200, {"X-Trickle": "20", "X-First": str(fast_pieces * piece + 500)}, body)]
    started = time.monotonic()
    report = model_sync.fetch_local([drip], timeout_s=0.6)
    # The limit and the wait until the connection is shut, not the 3 s after
    # which the job would go on without the download.
    assert time.monotonic() - started < 0.6 + 0.3 + 1.5
    (failed,) = report["failed"]
    assert (failed["reason"], failed["files_done"]) == ("timed_out", 0)
    # It was stopped, not failed: what it had received counts.
    assert fast_pieces * piece <= failed["received_bytes"] < fast_pieces * piece + 1000
    # The download is gone and so is its part file, while its host still sends.
    assert _gone(2) and not _parts(comfy) and not host.hold.is_set()
    assert report["files"] == [] and state_of(comfy) == {}


class Held:
    """Stands in for a response whose second read waits until its connection
    is shut, and for the job's handle on that connection. `ends` is what the
    read does then: the bytes it returns, or the error it raises. After that
    the connection has nothing more."""

    def __init__(self, ends):
        self.ends, self.headers = ends, {"Content-Length": str(10 * MIB)}
        self.pieces, self.closed = [b"x" * MIB], False
        self.reading, self.shut = threading.Event(), threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, want):
        if self.pieces:
            return self.pieces.pop()
        self.reading.set()
        assert self.shut.wait(10)
        ends, self.ends = self.ends, b""
        if isinstance(ends, Exception):
            raise ends
        return ends

    def shutdown(self, how):
        self.shut.set()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("ends", [
    b"x" * 700, b"", ConnectionAbortedError(10053, "aborted by the software in your host"),
    ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
    http.client.IncompleteRead(b"x" * 700, MIB - 700)],
    ids=["the bytes it has", "nothing", "an error", "an error of tls", "half a piece"])
def test_a_read_that_the_job_ended_is_a_stop_whatever_it_says(tmp_path, monkeypatch, ends):
    """What a read does when its connection is shut depends on the system
    and on the connection: it returns what it has, or nothing, or it raises.
    None of them is a failed try, a short file or a wrong sha256: the
    download was stopped, and what it had received still counts."""
    held = Held(ends)
    monkeypatch.setattr(model_sync, "_open", lambda url, hf_token, timeout=60: held)
    monkeypatch.setattr(model_sync, "_twin", lambda response: response)
    stop, part, raised = model_sync._Fetch(), tmp_path / "x.part", []

    def download():
        try:
            model_sync._stream("https://example.org/x", part, None, stop)
        except BaseException as e:  # noqa: BLE001
            raised.append(e)

    thread = threading.Thread(target=download)
    thread.start()
    assert held.reading.wait(10) and list(stop.lines) == [part]
    stop.set()
    stop.sever()
    thread.join(10)
    assert not thread.is_alive() and [type(e) for e in raised] == [model_sync._Stopped]
    assert MIB <= stop.received[part] <= MIB + 700
    # The job's handle is given back when the download ends.
    assert stop.lines == {} and held.closed
    # Without the job's doing, each of these is what it was: the end of a
    # connection before its file was whole, or the error itself.
    held, stop = Held(ends), model_sync._Fetch()
    held.shut.set()
    with pytest.raises((ConnectionError, OSError, http.client.HTTPException)) as broken:
        model_sync._stream("https://example.org/x", part, None, stop)
    assert not isinstance(broken.value, model_sync._Stopped) and stop.received[part] == 0


def test_the_jobs_handle_on_a_connection_is_closed_with_the_download(comfy, host):
    """To shut a connection the job keeps a handle of its own on it, for as
    long as the download reads. One that stayed open would be a connection
    the worker never gives back."""
    stop, seen = model_sync._Fetch(), []
    count = stop.count
    stop.count = lambda part, size: seen.append(dict(stop.lines)) or count(part, size)
    a = model(host, "a.safetensors")
    part = comfy / "models" / "a.part"
    assert model_sync._stream(a["url"], part, None, stop)[0] == 2000
    assert seen[0] == {} and list(seen[-1]) == [part]
    assert seen[-1][part].fileno() == -1 and stop.lines == {}
    # A response that has no connection to name is read as before.
    assert model_sync._twin(object()) is None


def test_a_file_that_failed_ends_the_job_beside_a_read_that_does_not_end(comfy, host,
                                                                        monkeypatch):
    """The first file that fails for good stops the others. One of them can
    be in a read that waits for its piece. A job without a time limit waited
    for that read, billed, for as long as the connection liked."""
    monkeypatch.setattr(model_sync, "FETCH_STOP_WAIT_S", 3)
    monkeypatch.setattr(model_sync, "FETCH_CUT_AFTER_S", 0.3)
    body = body_of("drip", 2 * MIB)
    drip = model(host, "drip.safetensors", body)
    host.routes["/drip.safetensors"] = [(200, {"X-Trickle": "20"}, body)]
    bad = {**model(host, "bad.safetensors"), "sha256": "0" * 64}
    real = model_sync._mismatch

    def late(entry, size, digest):
        time.sleep(0.5)  # the other download is in its read when this one fails
        return real(entry, size, digest)

    monkeypatch.setattr(model_sync, "_mismatch", late)
    # Should the job wait for the read after all, the test still ends.
    rescue = threading.Timer(8, host.hold.set)
    rescue.start()
    started = time.monotonic()
    report = model_sync.fetch_local([drip, bad], streams=2)
    took = time.monotonic() - started
    rescue.cancel()
    # The failure is seen when it comes, and the read is ended 0.3 s later.
    assert 0.5 <= took < 0.5 + 0.3 + 2
    (failed,) = report["failed"]
    assert failed["model"] == "vae/bad.safetensors" and "reason" not in failed
    assert failed["error"].startswith("sha256 mismatch: got ")
    assert _gone(2) and not _parts(comfy) and state_of(comfy) == {}


def test_what_arrives_while_the_downloads_are_stopped_counts(comfy, host, monkeypatch):
    """The time can be up between a file's last byte and its rename. That
    file is put in place and listed. So when it was the last one, the job has
    all its models: it was answered as a download that was too slow, with
    "1 of 1 files complete", and nothing tried it again."""
    real = model_sync._mismatch

    def late(entry, size, digest):
        time.sleep(1.0)  # the last byte is there, the rename is not done yet
        return real(entry, size, digest)

    monkeypatch.setattr(model_sync, "_mismatch", late)
    only = model(host, "only.safetensors")
    report = model_sync.fetch_local([only], timeout_s=0.6)
    assert report["failed"] == [] and report["files"] == ["vae/only.safetensors"]
    assert model_sync.fetch_failure(report) is None
    assert (report["bytes"], report["seconds"] >= 0.9) == (2000, True)
    assert sorted(state_of(comfy)) == ["vae/only.safetensors"]
    # A file that turns out wrong in those seconds is that failure, without
    # the mark of a job that used up its time: the runner tries it again.
    bad = {**model(host, "bad.safetensors"), "sha256": "0" * 64}
    report = model_sync.fetch_local([bad], timeout_s=0.6)
    (failed,) = report["failed"]
    assert failed["error"].startswith("sha256 mismatch: got ") and "reason" not in failed
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error" and "reason" not in failure


def test_a_download_that_has_every_byte_is_not_stopped(comfy, host):
    """The stop comes while the read of the last piece waits, and the piece
    then arrives whole. Every byte of the file is there: the download was
    thrown away, at "2097152 of 2097152 bytes"."""
    piece = model_sync.PIECE_BYTES
    body = body_of("two", 2 * piece)
    two = model(host, "two.safetensors", body)
    host.routes["/two.safetensors"] = [(200, {"X-Slow": str(piece)}, body)]
    threading.Timer(1.2, host.hold.set).start()
    started = time.monotonic()
    report = model_sync.fetch_local([two], timeout_s=0.6)
    assert time.monotonic() - started < 1.2 + 2
    assert report["failed"] == [] and report["files"] == ["vae/two.safetensors"]
    assert (comfy / "models" / "vae" / "two.safetensors").read_bytes() == body
    assert _gone(2) and not _parts(comfy)


def test_a_time_limit_that_is_up_before_the_download_begins(comfy, host, monkeypatch):
    """The limit counts from the call, so the clean-up is inside it. When
    that used it up, no request is made, and the answer does not blame a
    download: it said "too slow: 0.0 of 0.0 GB in 0 s"."""
    tracked = model_sync._tracked

    def slow_git():
        time.sleep(0.3)
        return tracked()

    monkeypatch.setattr(model_sync, "_tracked", slow_git)
    a = model(host, "a.safetensors")
    report = model_sync.fetch_local([a], timeout_s=0.1)
    (failed,) = report["failed"]
    assert failed == {
        "error": ("the download did not begin: cleaning up and making room took the 0.1 s "
                  "it was given"),
        "reason": "timed_out", "timeout_s": 0.1, "received_bytes": 0, "wanted_bytes": 2000,
        "seconds": 0.0, "mb_per_s": None, "files_done": 0, "files_wanted": 1}
    assert host.requests == [] and not _left_behind() and not _parts(comfy)
    # It is a job that used up its time, like any other the limit stopped.
    status, failure = model_sync.fetch_failure(report)
    assert (status, failure["reason"], failure["message"]) == (
        "model_fetch_error", "timed_out", failed["error"])


def test_a_time_limit_no_clock_can_count_to_is_no_limit(comfy, host):
    """A wait takes no more than some weeks of seconds. A larger limit broke
    the fetch after its downloads had begun, and they ran on behind the
    job's answer."""
    a = model(host, "a.safetensors")
    report = model_sync.fetch_local([a], timeout_s=1e12)
    assert report["files"] == ["vae/a.safetensors"] and report["failed"] == []
    assert threading.TIMEOUT_MAX < 1e12 and _gone(2)


def test_a_fetch_that_breaks_leaves_no_download_running(comfy, host, monkeypatch):
    """Whatever breaks the fetch while its downloads are going: they do not
    run on behind the answer of the job. One did, into the models folder and
    the list, when the wait for it would not take its number."""
    monkeypatch.setattr(model_sync, "FETCH_LOOK_S", 1e12)  # no wait takes that
    body = body_of("slow", 6000)
    slow = model(host, "slow.safetensors", body)
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": "3000"}, body)]
    with pytest.raises(OverflowError):
        model_sync.fetch_local([slow])
    # No part file is left. (Only Linux removes a file that is still open.)
    assert not _parts(comfy) or os.name == "nt"
    # The host sends the rest. The download ends, and puts nothing in place.
    host.hold.set()
    assert _gone()
    assert not (comfy / "models" / "vae" / "slow.safetensors").exists()
    assert state_of(comfy) == {}


def test_a_download_the_job_has_left_changes_no_file(comfy, host, monkeypatch):
    """A job whose time is up goes on without a download that does not end.
    That download can finish much later, when the worker's next job is writing
    a part file of the same name. It must not rename that file, remove it,
    empty it or list it."""
    a = model(host, "a.safetensors")
    path = comfy / "models" / "vae" / "a.safetensors"
    part = path.with_name("a.safetensors.part")
    left, noted = model_sync._Fetch(), []

    def the_job_goes_on(entry, size, digest):
        left.over = True  # just as the last byte has arrived
        part.write_bytes(b"what the next job has of it")
        return None

    monkeypatch.setattr(model_sync, "_mismatch", the_job_goes_on)
    with pytest.raises(model_sync._Stopped):
        model_sync._fetch_one(a, path, None, left, lambda size, digest: noted.append(size))
    assert noted == [] and not path.exists()
    assert part.read_bytes() == b"what the next job has of it"
    # One that had not opened its part file yet does not open it now.
    with pytest.raises(model_sync._Stopped):
        model_sync._fetch_one(a, path, None, left, lambda size, digest: noted.append(size))
    assert noted == [] and not path.exists() and len(host.requests) == 2
    assert part.read_bytes() == b"what the next job has of it"


def test_a_download_within_its_time_and_one_without_a_limit_are_what_they_were(comfy, host):
    body = body_of("slow", 6000)
    slow = model(host, "slow.safetensors", body)
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": "3000"}, body)]
    started = time.monotonic()
    threading.Timer(0.6, host.hold.set).start()
    # The job of a runner that names no limit: its download takes as long as
    # it takes.
    report = model_sync.fetch_local([slow])
    assert report["files"] == ["vae/slow.safetensors"] and report["failed"] == []
    assert time.monotonic() - started >= 0.5
    # With time to spare, the report of a download is the one it always was.
    limited = model_sync.fetch_local([model(host, "other.safetensors")], timeout_s=60)
    assert limited["files"] == ["vae/other.safetensors"] and limited["failed"] == []
    assert list(limited) == list(report) == [
        "files", "bytes", "seconds", "mb_per_s", "streams", "token", "reused", "reused_bytes",
        "evicted", "failed"]
    assert _gone(5) and not _parts(comfy)


def test_a_file_that_failed_is_the_failure_whatever_the_time_limit(comfy, host):
    bad = {**model(host, "bad.safetensors"), "sha256": "0" * 64}
    report = model_sync.fetch_local([bad], timeout_s=60)
    (failed,) = report["failed"]
    assert failed["error"].startswith("sha256 mismatch: got ") and "reason" not in failed
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error" and "reason" not in failure
    # A refused file decides the status, also beside a download that ran out
    # of time: a person has to act, and neither is tried again.
    slow = model_sync.too_slow(10, 20, 1.0, 0, 2, 5)
    refused = {"model": "vae/x", "error": "HTTP 403 Forbidden", "http_status": 403,
               "host": "huggingface.co", "repo": "a/b"}
    assert model_sync.fetch_failure({"failed": [slow, refused]})[0] == "model_access_error"
    # Beside any other failure the time limit decides.
    other = {"model": "vae/y", "error": "URLError: reset"}
    status, failure = model_sync.fetch_failure({"failed": [other, slow]})
    assert (status, failure["reason"], failure["message"]) == (
        "model_fetch_error", "timed_out", slow["error"])
    assert failure["failed"] == [other, slow]


def _handler(monkeypatch, ran):
    """src/handler.py around the real fetch: the checkout, the volume (which
    holds nothing), the pack install and the ComfyUI run are replaced."""
    if "handler" not in sys.modules:
        # The worker's SDK is not installed where the tests run, and the
        # handler starts its job loop when it is imported.
        monkeypatch.setitem(sys.modules, "runpod", types.SimpleNamespace(
            serverless=types.SimpleNamespace(start=lambda config: None)))
    import handler
    monkeypatch.setattr(handler, "image_env", lambda: {"image": "cu130-t2141"})
    monkeypatch.setattr(handler.checkout, "checkout_commit", lambda commit, repo=None: {
        "commit_checked_out": commit, "checkout_s": 0.1, "pip_s": 0.0})
    monkeypatch.setattr(model_sync, "missing_models", lambda models: [
        f"{m['directory']}/{m['name']}" for m in models])
    monkeypatch.setattr(handler.custom_nodes, "prepare", lambda requested: ran.append("packs"))
    monkeypatch.setattr(handler.comfy_runner, "execute",
                        lambda workflow, **kw: ran.append("workflow") or {"outputs": []})
    return handler


def test_a_job_whose_download_is_too_slow_answers_at_once(comfy, host, monkeypatch):
    """The whole job, as the worker runs it. With the limit the runner sends,
    the job answers when the limit is used up, with what the download reached,
    and the workflow does not run. Without it (an older runner) the job waits
    for its download as it always did."""
    monkeypatch.setattr(model_sync, "FETCH_STOP_WAIT_S", 0.4)
    ran = []
    handler = _handler(monkeypatch, ran)
    piece = model_sync.PIECE_BYTES
    fast = model(host, "fast.safetensors")
    body = body_of("slow", 3 * piece)
    slow = model(host, "slow.safetensors", body)
    host.routes["/slow.safetensors"] = [(200, {"X-Slow": str(2 * piece)}, body)]
    job = {"mode": "run", "comfy_commit": "c" * 40, "workflow_id": "wf", "lane": "weekly",
           "workflow": {"1": {"class_type": "SaveImage", "inputs": {}}}, "models": [fast, slow],
           "timeout_s": 600, "fetch_missing": True}
    started = time.monotonic()
    answer = handler.handler({"input": {**job, "fetch_timeout_s": 0.6}})
    assert time.monotonic() - started < 0.6 + 0.4 + 2 and ran == []
    assert answer["status"] == "model_fetch_error" and "error" not in answer
    failure = answer["failure"]
    assert failure["message"].startswith("the download was too slow: ")
    assert (failure["reason"], failure["timeout_s"], failure["received_bytes"],
            failure["wanted_bytes"], failure["files_done"], failure["files_wanted"]) == (
        "timed_out", 0.6, 2000 + 2 * piece, 2000 + 3 * piece, 1, 2)
    assert answer["model_fetch"]["files"] == 1
    assert answer["model_fetch"]["bytes"] == 2000 + 2 * piece
    assert 0.6 <= answer["timings"]["model_fetch_s"] < 0.6 + 0.4 + 2
    json.dumps(answer)  # it is what RunPod is given
    host.hold.set()
    assert _gone()

    # No limit in the job input: the same download is waited for.
    host.hold.clear()
    threading.Timer(0.8, host.hold.set).start()
    answer = handler.handler({"input": job})
    assert answer["status"] == "ok" and ran == ["packs", "workflow"]
    assert answer["model_fetch"]["files"] == 1 and answer["model_fetch"]["reused"] == 1
    assert answer["timings"]["model_fetch_s"] >= 0.6
    assert (comfy / "models" / "vae" / "slow.safetensors").read_bytes() == body


# -- a disk that is fuller than it says --------------------------------------------------
def _small_disk(monkeypatch, comfy, tmp_path, capacity: int) -> list:
    """A disk that holds `capacity` bytes and reports the free space of
    another one: `_free_bytes` stays at the fixture's 10**12, and a write that
    takes the models folder and the cache over the capacity fails with "no
    space left on device". Returns the list of the part files written, one
    name for each try."""
    real, tries = model_sync._stream, []

    def stream(url, dest, hf_token, stop=None):
        tries.append(dest.name)
        done = real(url, dest, hf_token, stop)
        used = sum(p.stat().st_size for root in (comfy / "models", tmp_path / "hf-cache")
                   for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".safetensors", model_sync.PART))
        if used > capacity:
            raise OSError(errno.ENOSPC, "No space left on device")
        return done

    monkeypatch.setattr(model_sync, "_stream", stream)
    return tries


def test_a_write_that_finds_the_disk_full_makes_room_and_is_tried_once_more(
        comfy, host, tmp_path, monkeypatch):
    """What a worker's disk reports as free can be the host's disk, not the
    120 GB the worker is held to. Then making room before the download removes
    nothing, the worker fills up, and a write fails in the middle of a job."""
    volume = tmp_path / "runpod-volume"
    (volume / "models" / "vae").mkdir(parents=True)
    (volume / "models" / "vae" / "on_volume.safetensors").write_bytes(b"v" * 9000)
    monkeypatch.setattr(model_sync, "VOLUME_ROOT", volume)
    old = [model(host, f"old{i}.safetensors", body_of(f"old{i}", 3000)) for i in range(2)]
    kept = model(host, "kept.safetensors", body_of("kept", 3000))
    stamps = itertools.count(1000, 100)
    monkeypatch.setattr(model_sync, "time", types.SimpleNamespace(
        time=lambda: next(stamps), monotonic=time.monotonic))
    for earlier in (old[1], old[0], kept):
        model_sync.fetch_local([earlier])
    cache = tmp_path / "hf-cache" / "hub"
    cache.mkdir(parents=True)
    (cache / "blob.safetensors").write_bytes(b"c" * 2000)
    # 11,000 bytes are there. The disk holds 14,000, and says it has room.
    tries = _small_disk(monkeypatch, comfy, tmp_path, 14000)
    first = model(host, "first.safetensors", body_of("first", 1000))
    new = model(host, "new.safetensors", body_of("new", 5000))
    report = model_sync.fetch_local([kept, first, new], streams=1)
    assert report["failed"] == []
    assert report["files"] == ["vae/first.safetensors", "vae/new.safetensors"]
    assert tries == ["first.safetensors.part", "new.safetensors.part", "new.safetensors.part"]
    # Everything the job does not need went, the oldest first, and the cache
    # of other downloaders. The report says so.
    assert report["evicted"] == ["vae/old1.safetensors", "vae/old0.safetensors"]
    assert model_sync.fetch_brief(report)["evicted"] == 2
    assert not (tmp_path / "hf-cache").exists()
    # What stays: the file the job found, the one it had finished, what git
    # tracks, and the volume.
    models = comfy / "models"
    left = sorted(p.relative_to(models).as_posix() for p in models.rglob("*") if p.is_file())
    assert left == [".comfyci-fetched.json", "checkpoints/put_checkpoints_here",
                    "configs/v1-inference.yaml", "vae/first.safetensors", "vae/kept.safetensors",
                    "vae/new.safetensors"]
    assert report["reused"] == ["vae/kept.safetensors"]
    assert sorted(state_of(comfy)) == ["vae/first.safetensors", "vae/kept.safetensors",
                                       "vae/new.safetensors"]
    assert (volume / "models" / "vae" / "on_volume.safetensors").stat().st_size == 9000


def test_a_disk_that_is_full_again_is_the_failure_it_was(comfy, host, tmp_path, monkeypatch):
    model_sync.fetch_local([model(host, "old.safetensors", body_of("old", 3000))])
    tries = _small_disk(monkeypatch, comfy, tmp_path, 4000)
    big = model(host, "big.safetensors", body_of("big", 5000))
    full = [{"model": "vae/big.safetensors", "error": "disk full while writing"}]
    report = model_sync.fetch_local([big])
    # Room was made, the file had its second try, and the disk was full again.
    assert report["evicted"] == ["vae/old.safetensors"] and len(tries) == 2
    assert report["failed"] == full and report["files"] == []
    status, failure = model_sync.fetch_failure(report)
    assert status == "model_fetch_error" and "reason" not in failure
    assert failure["message"] == ("model download failed: vae/big.safetensors: "
                                  "disk full while writing")
    assert not _parts(comfy)
    # Nothing is left to remove: another try would find the disk as full as
    # the first did, so there is none.
    del tries[:]
    report = model_sync.fetch_local([big])
    assert len(tries) == 1 and report["evicted"] == [] and report["failed"] == full
    # A quota that is used up is a full disk too.
    assert errno.ENOSPC in model_sync.DISK_FULL_ERRNOS
    assert getattr(errno, "EDQUOT", errno.ENOSPC) in model_sync.DISK_FULL_ERRNOS


def _full_at_first(monkeypatch) -> list:
    """The first write of every file finds the disk full, and no later one.
    Returns the list of the part files written, one name for each try."""
    real, tries = model_sync._stream, []

    def stream(url, dest, hf_token, stop=None):
        tries.append(dest.name)
        if tries.count(dest.name) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(url, dest, hf_token, stop)

    monkeypatch.setattr(model_sync, "_stream", stream)
    return tries


def test_downloads_that_find_the_disk_full_together_make_room_once(comfy, host, monkeypatch):
    model_sync.fetch_local([model(host, "old.safetensors")])
    emptied, empty = [], model_sync._empty
    monkeypatch.setattr(model_sync, "_empty",
                        lambda *args: emptied.append(args[1]) or empty(*args))
    tries = _full_at_first(monkeypatch)
    together = threading.Barrier(3)
    full_at_first = model_sync._stream

    def stream(url, dest, hf_token, stop=None):
        if dest.name not in tries:
            together.wait(10)  # all three are writing when the disk is full
        return full_at_first(url, dest, hf_token, stop)

    monkeypatch.setattr(model_sync, "_stream", stream)
    models = [model(host, f"m{i}.safetensors") for i in range(3)]
    report = model_sync.fetch_local(models, streams=3)
    assert len(emptied) == 1 and emptied[0] == {f"vae/m{i}.safetensors" for i in range(3)}
    assert report["evicted"] == ["vae/old.safetensors"]
    assert report["failed"] == [] and len(report["files"]) == 3 and len(tries) == 6


def test_a_download_that_began_after_room_was_made_gets_no_second_try(comfy, host, monkeypatch):
    """Room is made once in a job. A file whose download began after that
    has seen the disk as empty as the job can make it."""
    model_sync.fetch_local([model(host, "spare.safetensors")])
    tries = _full_at_first(monkeypatch)
    a, b = model(host, "a.safetensors"), model(host, "b.safetensors")
    report = model_sync.fetch_local([a, b], streams=1)
    assert tries == ["a.safetensors.part", "a.safetensors.part", "b.safetensors.part"]
    assert report["files"] == ["vae/a.safetensors"]
    assert report["evicted"] == ["vae/spare.safetensors"]
    assert report["failed"] == [{"model": "vae/b.safetensors",
                                 "error": "disk full while writing"}]


def test_a_full_disk_at_the_write_of_the_list_makes_room_too(comfy, host, monkeypatch):
    """The list of what the worker fetched is written on the disk the models
    are on, right after a file is put in place. A disk that is full at that
    moment broke the whole fetch (`worker_error`), no room was made, and the
    finished file was in no list."""
    model_sync.fetch_local([model(host, "old.safetensors")])
    save, refused = model_sync._save_local_state, []

    def full(state):
        raise OSError(errno.ENOSPC, "No space left on device")

    def full_once(state):
        if "vae/a.safetensors" in state and not refused:
            refused.append(sorted(state))
            full(state)
        return save(state)

    monkeypatch.setattr(model_sync, "_save_local_state", full_once)
    a = model(host, "a.safetensors")
    report = model_sync.fetch_local([a])
    # Room was made, and the file was fetched once more and listed.
    assert refused == [["vae/a.safetensors", "vae/old.safetensors"]]
    assert report["failed"] == [] and report["files"] == ["vae/a.safetensors"]
    assert report["evicted"] == ["vae/old.safetensors"]
    assert host.paths().count("/a.safetensors") == 2
    assert sorted(state_of(comfy)) == ["vae/a.safetensors"]
    assert (comfy / "models" / "vae" / "a.safetensors").read_bytes() == body_of("a.safetensors")

    # Nothing to remove: the file has failed like any that found the disk
    # full, and it is not left under its name without a line in the list.
    b = model(host, "b.safetensors")
    monkeypatch.setattr(model_sync, "_save_local_state", lambda state: (
        full(state) if "vae/b.safetensors" in state else save(state)))
    report = model_sync.fetch_local([a, b])
    assert report["failed"] == [{"model": "vae/b.safetensors",
                                 "error": "disk full while writing"}]
    assert report["reused"] == ["vae/a.safetensors"] and report["evicted"] == []
    assert not (comfy / "models" / "vae" / "b.safetensors").exists() and not _parts(comfy)
    assert sorted(state_of(comfy)) == ["vae/a.safetensors"]
    # Any other error of that write is a fetch that broke, as it was.

    def refuse(state):
        if "vae/b.safetensors" in state:
            raise PermissionError(errno.EACCES, "Permission denied")
        return save(state)

    monkeypatch.setattr(model_sync, "_save_local_state", refuse)
    with pytest.raises(PermissionError):
        model_sync.fetch_local([a, b])


def test_a_disk_too_full_for_the_list_before_the_download_is_a_full_disk(comfy, host,
                                                                        monkeypatch):
    """The list is written once before anything is downloaded. No room for
    it then is no reason to break: the steps after it make room, or say that
    the disk is full."""
    save, calls = model_sync._save_local_state, []

    def full_at_first(state):
        calls.append(sorted(state))
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return save(state)

    monkeypatch.setattr(model_sync, "_save_local_state", full_at_first)
    a = model(host, "a.safetensors")
    report = model_sync.fetch_local([a])
    assert calls == [[], ["vae/a.safetensors"]]
    assert report["failed"] == [] and report["files"] == ["vae/a.safetensors"]
    assert sorted(state_of(comfy)) == ["vae/a.safetensors"]


def test_an_empty_cache_folder_is_no_room_that_was_made(comfy, host, tmp_path, monkeypatch):
    """A second try after a full disk is for a disk that has more room now.
    A cache folder with nothing in it gave none, and the file was downloaded
    once more up to the point where the disk was full again."""
    (tmp_path / "hf-cache" / "hub" / "models--x").mkdir(parents=True)
    tries = _small_disk(monkeypatch, comfy, tmp_path, 4000)
    big = model(host, "big.safetensors", body_of("big", 5000))
    report = model_sync.fetch_local([big])
    assert len(tries) == 1 and host.paths() == ["/big.safetensors"]
    assert report["failed"] == [{"model": "vae/big.safetensors",
                                 "error": "disk full while writing"}]
    # One file in it is room: the cache goes, and the file has its second try.
    (tmp_path / "hf-cache" / "hub" / "models--x").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hf-cache" / "hub" / "models--x" / "blob.safetensors").write_bytes(b"c" * 3000)
    del tries[:]
    model_sync.fetch_local([big])
    assert len(tries) == 2 and not (tmp_path / "hf-cache").exists()
