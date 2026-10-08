"""Pluggable storage backends for regression results.

Both backends share the path layout documented in gcs_publish.py, so switching
is a config change (CI env var + dashboard base URL), not a data migration:

  github (current default): results live on an orphan branch of a GitHub repo
    and the dashboard reads them via raw.githubusercontent.com. Writes are a
    local checkout + one commit + push, authenticated by GITHUB_TOKEN in
    Actions or normal git credentials locally. Needs no cloud secrets.
  gcs: Google Cloud Storage (needs GCS_SERVICE_ACCOUNT_JSON / application
    default credentials). The long-term home once a service account exists.

Select with --storage or REGRESSION_STORAGE.

Derived files (indexes and other aggregates over many runs) are produced by
"derived writers" registered with register_derived(). They run inside
finalize(), and the GitHub backend re-runs them on top of the newest remote
tree whenever a push is rejected, so concurrent publishers never clobber each
other's aggregate files. A failing writer is reported and skipped: derived
files can be regenerated later, the run's primary results cannot.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable


class Storage:
    prefix: str = "regression"

    def __init__(self, prefix: str = "regression"):
        self.prefix = prefix
        self._derived: list[Callable[["Storage"], None]] = []

    # -- primitives every backend implements --------------------------------
    def download_json(self, path: str) -> dict | None:
        raise NotImplementedError

    def upload_json(self, path: str, obj: dict):
        raise NotImplementedError

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        raise NotImplementedError

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
        raise NotImplementedError

    def list_json(self, blob_prefix: str, name: str) -> list[str]:
        """Paths of every `<blob_prefix>/**/<name>` file, without materialising
        anything else (used by index builders)."""
        raise NotImplementedError

    def read_blob_json(self, path: str) -> dict | None:
        """Read one committed JSON file without materialising its directory."""
        raise NotImplementedError

    # -- derived writers -----------------------------------------------------
    def register_derived(self, fn: Callable[["Storage"], None]):
        """fn(store) regenerates aggregate files from the current tree. It must
        be idempotent: it may run several times per publish."""
        self._derived.append(fn)

    def _run_derived(self):
        for fn in self._derived:
            try:
                fn(self)
            except Exception as e:  # noqa: BLE001 - never lose primary results
                self._derived_failed(fn, e)

    def _derived_failed(self, fn, e: Exception):
        name = getattr(fn, "__name__", repr(fn))
        print(f"::warning::derived writer {name} failed, skipping it: {e!r}")

    def finalize(self, message: str):
        """Called once after all writes; git backend commits + pushes here."""
        self._run_derived()

    # -- layout helpers shared by all backends ------------------------------
    @property
    def root(self) -> str:
        """Top-level results root shared by every lane (e.g. 'regression')."""
        return self.prefix.split("/")[0]

    def golden_current(self, workflow_id: str) -> dict | None:
        return self.download_json(f"{self.prefix}/golden/{workflow_id}/current.json")

    def noise_floor(self, workflow_id: str, tag: str) -> dict | None:
        return self.download_json(f"{self.prefix}/golden/{workflow_id}/{tag}/noise_floor.json")

    def fetch_golden_outputs(self, workflow_id: str, tag: str, dest: Path) -> int:
        return self.download_dir(f"{self.prefix}/golden/{workflow_id}/{tag}/outputs", dest)

    def latest_pointer(self, branch: str) -> dict | None:
        return self.download_json(f"{self.prefix}/latest/{branch}.json")

    def run_summary(self, branch: str, commit: str) -> dict | None:
        return self.download_json(f"{self.prefix}/runs/{branch}/{commit}/summary.json")

    def fetch_run_outputs(self, branch: str, commit: str, workflow_id: str,
                          dest: Path) -> int:
        return self.download_dir(
            f"{self.prefix}/runs/{branch}/{commit}/{workflow_id}/outputs", dest)

    def publish_workflow_run(self, branch: str, commit: str, workflow_id: str,
                             local_dir: Path):
        self.upload_dir(f"{self.prefix}/runs/{branch}/{commit}/{workflow_id}", local_dir)

    def publish_summary(self, branch: str, commit: str, summary: dict):
        self.upload_json(f"{self.prefix}/runs/{branch}/{commit}/summary.json", summary)

    def update_latest(self, branch: str, pointer: dict):
        """Must land after all per-workflow results and summary.json."""
        self.upload_json(f"{self.prefix}/latest/{branch}.json", pointer)

    def snapshot_manifest(self, commit: str, manifest: dict):
        self.upload_json(f"{self.prefix}/manifest-snapshot/{commit}.json", manifest)


class GitHubStorage(Storage):
    """Results branch of a GitHub repo, served by raw.githubusercontent.com.

    The local checkout is a blobless partial clone with a cone sparse-checkout:
    only the small aggregate directories are materialised up front and every
    other path is added on first use, so publishing a run no longer writes the
    whole results tree (hundreds of MB of PNGs) to the runner's disk.
    """

    PUSH_ATTEMPTS = 8

    def __init__(self, repo_slug: str, branch: str = "results",
                 workdir: str | Path = "./results-checkout", prefix: str = "regression",
                 remote_url: str | None = None):
        super().__init__(prefix)
        self.repo_slug = repo_slug
        self.branch = branch
        self.workdir = Path(workdir).resolve()
        self.remote_url = remote_url
        self._written: list[str] = []
        self._derived_paths: set[str] = set()
        self._sparse: set[str] = set()
        self._lock = threading.Lock()
        self._clone()

    # -- git plumbing --------------------------------------------------------
    def _url(self, with_token: bool) -> str:
        if self.remote_url:
            return self.remote_url
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if with_token and token:
            return f"https://x-access-token:{token}@github.com/{self.repo_slug}.git"
        return f"https://github.com/{self.repo_slug}.git"

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.workdir), *args],
                              check=check, capture_output=True, text=True, encoding="utf-8")

    def _base_dirs(self) -> list[str]:
        return [f"{self.prefix}/latest", f"{self.prefix}/golden",
                f"{self.prefix}/manifest-snapshot", f"{self.root}/index"]

    def _clone(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not (self.workdir / ".git").exists():
            subprocess.run(["git", "init", "-q", "-b", self.branch, str(self.workdir)],
                           check=True, capture_output=True, text=True)
            self._git("remote", "add", "origin", self._url(False))
        # Blobless partial clone: trees come down with the fetch, blobs only
        # when a path is checked out. Idempotent on an existing checkout.
        self._git("config", "remote.origin.promisor", "true")
        self._git("config", "remote.origin.partialclonefilter", "blob:none")
        self._git("sparse-checkout", "init", "--cone")
        self._git("sparse-checkout", "set", *self._base_dirs())
        self._sparse = set(self._base_dirs())
        # Shallow-sync the results branch; if it doesn't exist yet the first
        # finalize() push creates it. Any other fetch failure is fatal: an
        # empty tree would publish bogus no-baseline verdicts.
        r = self._git("fetch", "--depth", "1", "--filter=blob:none", "origin", self.branch,
                      check=False)
        if r.returncode == 0:
            self._git("checkout", "-q", "-B", self.branch, "FETCH_HEAD")
        elif "couldn't find remote ref" not in r.stderr:
            raise RuntimeError(f"could not fetch {self.repo_slug}@{self.branch}: "
                               f"{r.stderr[-500:]}")

    def _p(self, path: str) -> Path:
        return self.workdir / path

    @staticmethod
    def _parent(path: str) -> str:
        return path.rsplit("/", 1)[0]

    def _ensure_dir(self, d: str):
        """Materialise directory `d` in the sparse checkout. Serialised: the
        orchestrator calls this from worker threads, and concurrent git
        invocations in one repository collide on .git/index.lock."""
        d = d.rstrip("/")
        with self._lock:
            for have in self._sparse:
                if d == have or d.startswith(have + "/"):
                    return
            self._git("sparse-checkout", "add", d)
            self._sparse.add(d)

    # -- primitives ----------------------------------------------------------
    def download_json(self, path: str) -> dict | None:
        self._ensure_dir(self._parent(path))
        f = self._p(path)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def upload_json(self, path: str, obj: dict):
        self._ensure_dir(self._parent(path))
        f = self._p(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        self._written.append(path)

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        self._ensure_dir(blob_prefix)
        local_dir = Path(local_dir)
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(local_dir).as_posix()
                dest = self._p(f"{blob_prefix}/{rel}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                self._written.append(f"{blob_prefix}/{rel}")

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
        self._ensure_dir(blob_prefix)
        src = self._p(blob_prefix)
        if not src.is_dir():
            return 0
        n = 0
        for p in sorted(src.rglob("*")):
            if p.is_file():
                dest = Path(local_dir) / p.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                n += 1
        return n

    def list_blobs(self, blob_prefix: str) -> dict[str, str]:
        """{path: blob id} for every committed file under `blob_prefix`, from the
        local trees only (no blob is fetched)."""
        r = self._git("ls-tree", "-r", "HEAD", f"{blob_prefix}/", check=False)
        if r.returncode != 0:
            return {}
        out = {}
        for line in r.stdout.splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                out[path] = parts[2]
        return out

    def list_json(self, blob_prefix: str, name: str) -> list[str]:
        return [p for p in self.list_blobs(blob_prefix) if p.endswith("/" + name)]

    def prefetch(self, paths: list[str]):
        """Fetch the blobs behind committed `paths` in ONE round trip. A partial
        clone otherwise lazily fetches one object per read, which takes seconds
        each against GitHub. Mirrors git's own batched lazy-fetch command."""
        wanted = set(paths)
        if not wanted:
            return
        # One tree listing over the common parent instead of one per directory.
        parents = [p.rsplit("/", 1)[0].split("/") for p in wanted]
        common = []
        for parts in zip(*parents):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        blobs = self.list_blobs("/".join(common)) if common else {}
        oids = [oid for path, oid in blobs.items() if path in wanted]
        if not oids:
            return
        # Bytes, not text: text mode would write CRLF on Windows and git would
        # reject the ids as refspecs.
        r = subprocess.run(
            ["git", "-C", str(self.workdir), "-c", "fetch.negotiationAlgorithm=noop",
             "fetch", "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no",
             "--filter=blob:none", "--stdin", "origin"],
            input=("\n".join(oids) + "\n").encode(), check=False, capture_output=True)
        if r.returncode != 0:  # reads fall back to per-object lazy fetches
            print(f"storage: prefetch failed: {r.stderr.decode(errors='replace')[-300:]}")

    def read_blob_json(self, path: str) -> dict | None:
        r = subprocess.run(["git", "-C", str(self.workdir), "cat-file", "-p", f"HEAD:{path}"],
                           check=False, capture_output=True)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.decode("utf-8"))

    # -- publish -------------------------------------------------------------
    def _run_derived(self):
        for fn in self._derived:
            before = len(self._written)
            try:
                fn(self)
                self._derived_paths.update(self._written[before:])
            except Exception as e:  # noqa: BLE001 - never lose primary results
                self._derived_failed(fn, e)
                self._discard(self._written[before:])
                del self._written[before:]

    def _discard(self, paths: list[str]):
        """Drop a failed writer's partial output from the working tree."""
        for path in paths:
            tracked = self._git("ls-files", "--error-unmatch", path, check=False).returncode == 0
            if tracked:
                self._git("checkout", "-q", "HEAD", "--", path, check=False)
            else:
                self._p(path).unlink(missing_ok=True)

    def _commit(self, message: str) -> bool:
        # --sparse: stage paths outside the cone instead of failing the publish.
        self._git("add", "-A", "--sparse")
        if not self._git("status", "--porcelain").stdout.strip():
            return False
        self._git("-c", "user.name=comfyci", "-c", "user.email=ci@comfy.org",
                  "commit", "-q", "-m", message)
        return True

    def finalize(self, message: str):
        self._run_derived()
        if not self._commit(message):
            print("storage: nothing new to publish")
            return
        url = self._url(True)
        for attempt in range(1, self.PUSH_ATTEMPTS + 1):
            r = self._git("push", "-q", url, f"HEAD:refs/heads/{self.branch}", check=False)
            if r.returncode == 0:
                print(f"storage: pushed results to {self.repo_slug}@{self.branch}")
                return
            if attempt == self.PUSH_ATTEMPTS:
                raise RuntimeError(f"could not push results: {r.stderr[-500:]}")
            # The remote advanced under us (concurrent publisher). Commit-scoped
            # files cannot conflict, so re-apply them on top of the new remote
            # head; derived files are regenerated from that head instead of
            # being restored, so the other publisher's updates survive.
            saved = {p: self._p(p).read_bytes() for p in self._written
                     if p not in self._derived_paths and self._p(p).exists()}
            self._git("fetch", "--depth", "1", "--filter=blob:none", "origin", self.branch)
            self._git("checkout", "-q", "-B", self.branch, "FETCH_HEAD")
            for path, data in saved.items():
                f = self._p(path)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
            self._run_derived()
            self._commit(message)
            time.sleep(2 + random.uniform(0, 3))


class GcsStorage(Storage):
    def __init__(self, bucket: str, prefix: str = "regression"):
        import gcs_publish  # deferred: google-cloud-storage only needed here
        super().__init__(prefix)
        self._g = gcs_publish
        self.bucket = bucket

    def download_json(self, path: str) -> dict | None:
        return self._g.download_json(self.bucket, path)

    def upload_json(self, path: str, obj: dict):
        self._g.upload_json(self.bucket, path, obj)

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        self._g.upload_dir(self.bucket, blob_prefix, Path(local_dir))

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
        return self._g.download_dir(self.bucket, blob_prefix, Path(local_dir))

    def list_json(self, blob_prefix: str, name: str) -> list[str]:
        return [p for p in self._g.list_paths(self.bucket, blob_prefix)
                if p.endswith("/" + name)]

    def read_blob_json(self, path: str) -> dict | None:
        return self.download_json(path)


def add_storage_args(ap):
    ap.add_argument("--storage", default=os.environ.get("REGRESSION_STORAGE", "github"),
                    choices=["github", "gcs"])
    ap.add_argument("--prefix", default="regression")
    ap.add_argument("--bucket", default=os.environ.get("GCS_BUCKET"),
                    help="GCS bucket (gcs storage only)")
    ap.add_argument("--results-repo",
                    default=os.environ.get("RESULTS_REPO", "Comfy-Org/comfyci-runpod-worker"))
    ap.add_argument("--results-branch", default=os.environ.get("RESULTS_BRANCH", "results"))
    ap.add_argument("--results-workdir", default="./results-checkout")


def from_args(args) -> Storage:
    if args.storage == "gcs":
        if not args.bucket:
            raise SystemExit("--bucket (or GCS_BUCKET) is required with --storage gcs")
        return GcsStorage(args.bucket, args.prefix)
    return GitHubStorage(args.results_repo, args.results_branch,
                         args.results_workdir, args.prefix)
