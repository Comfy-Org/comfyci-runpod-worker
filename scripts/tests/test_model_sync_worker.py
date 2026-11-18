"""src/model_sync.py: what a job fetches to the worker's own disk
(`fetch_local`), against a local HTTP server that plays the model hosts, and
which host the Hugging Face token is sent to."""
import hashlib
import http.client
import http.server
import itertools
import json
import os
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
    headers, and counts how many it is answering at once."""

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
