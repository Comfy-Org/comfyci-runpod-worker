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
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class Storage:
    prefix: str = "regression"

    # -- primitives every backend implements --------------------------------
    def download_json(self, path: str) -> dict | None:
        raise NotImplementedError

    def upload_json(self, path: str, obj: dict):
        raise NotImplementedError

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        raise NotImplementedError

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
        raise NotImplementedError

    def finalize(self, message: str):
        """Called once after all writes; git backend commits + pushes here."""

    # -- layout helpers shared by all backends ------------------------------
    def golden_current(self, workflow_id: str) -> dict | None:
        return self.download_json(f"{self.prefix}/golden/{workflow_id}/current.json")

    def noise_floor(self, workflow_id: str, tag: str) -> dict | None:
        return self.download_json(f"{self.prefix}/golden/{workflow_id}/{tag}/noise_floor.json")

    def fetch_golden_outputs(self, workflow_id: str, tag: str, dest: Path) -> int:
        return self.download_dir(f"{self.prefix}/golden/{workflow_id}/{tag}/outputs", dest)

    def latest_pointer(self, branch: str) -> dict | None:
        return self.download_json(f"{self.prefix}/latest/{branch}.json")

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
    """Results branch of a GitHub repo, served by raw.githubusercontent.com."""

    def __init__(self, repo_slug: str, branch: str = "results",
                 workdir: str | Path = "./results-checkout", prefix: str = "regression"):
        self.prefix = prefix
        self.repo_slug = repo_slug
        self.branch = branch
        self.workdir = Path(workdir).resolve()
        self._written: list[str] = []
        self._clone()

    def _url(self, with_token: bool) -> str:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if with_token and token:
            return f"https://x-access-token:{token}@github.com/{self.repo_slug}.git"
        return f"https://github.com/{self.repo_slug}.git"

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.workdir), *args],
                              check=check, capture_output=True, text=True)

    def _clone(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not (self.workdir / ".git").exists():
            subprocess.run(["git", "init", "-q", "-b", self.branch, str(self.workdir)],
                           check=True, capture_output=True, text=True)
            self._git("remote", "add", "origin", self._url(False))
        # Shallow-sync the results branch; if it doesn't exist yet the first
        # finalize() push creates it.
        r = self._git("fetch", "--depth", "1", "origin", self.branch, check=False)
        if r.returncode == 0:
            self._git("checkout", "-q", "-B", self.branch, "FETCH_HEAD")

    def _p(self, path: str) -> Path:
        return self.workdir / path

    def download_json(self, path: str) -> dict | None:
        f = self._p(path)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def upload_json(self, path: str, obj: dict):
        f = self._p(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        self._written.append(path)

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        local_dir = Path(local_dir)
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(local_dir).as_posix()
                dest = self._p(f"{blob_prefix}/{rel}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                self._written.append(f"{blob_prefix}/{rel}")

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
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

    def _commit(self, message: str) -> bool:
        self._git("add", "-A")
        if not self._git("status", "--porcelain").stdout.strip():
            return False
        self._git("-c", "user.name=comfyci", "-c", "user.email=ci@comfy.org",
                  "commit", "-q", "-m", message)
        return True

    def finalize(self, message: str):
        if not self._commit(message):
            print("storage: nothing new to publish")
            return
        url = self._url(True)
        for attempt in range(4):
            r = self._git("push", "-q", url, f"HEAD:refs/heads/{self.branch}", check=False)
            if r.returncode == 0:
                print(f"storage: pushed results to {self.repo_slug}@{self.branch}")
                return
            if attempt == 3:
                raise RuntimeError(f"could not push results: {r.stderr[-500:]}")
            # The remote advanced under us (concurrent publisher). Written paths
            # are commit-scoped so real conflicts can't happen: rebuild our
            # files on top of the new remote head and try again.
            saved = {p: self._p(p).read_bytes() for p in self._written
                     if self._p(p).exists()}
            self._git("fetch", "--depth", "1", "origin", self.branch)
            self._git("checkout", "-q", "-B", self.branch, "FETCH_HEAD")
            for path, data in saved.items():
                f = self._p(path)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)
            self._commit(message)
            time.sleep(2)


class GcsStorage(Storage):
    def __init__(self, bucket: str, prefix: str = "regression"):
        import gcs_publish  # deferred: google-cloud-storage only needed here
        self._g = gcs_publish
        self.bucket = bucket
        self.prefix = prefix

    def download_json(self, path: str) -> dict | None:
        return self._g.download_json(self.bucket, path)

    def upload_json(self, path: str, obj: dict):
        self._g.upload_json(self.bucket, path, obj)

    def upload_dir(self, blob_prefix: str, local_dir: Path):
        self._g.upload_dir(self.bucket, blob_prefix, Path(local_dir))

    def download_dir(self, blob_prefix: str, local_dir: Path) -> int:
        return self._g.download_dir(self.bucket, blob_prefix, Path(local_dir))


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
