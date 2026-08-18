"""GCS I/O for regression results.

Layout under gs://<bucket>/<prefix>/ (prefix defaults to "regression"):
  runs/<branch>/<commit>/<workflow_id>/{outputs/, run.json, comparison.json, figures/}
  runs/<branch>/<commit>/summary.json
  latest/<branch>.json                  -- pointer to the newest complete run, written LAST
  manifest-snapshot/<commit>.json
  golden/<workflow_id>/<tag>/{outputs/, run_r1.json, run_r2.json, noise_floor.json, blessed.json}
  golden/<workflow_id>/current.json     -- authoritative pointer to the active blessed tag
"""
from __future__ import annotations

import json
from pathlib import Path

from google.cloud import storage

_client = None


def client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def upload_json(bucket: str, blob_path: str, obj: dict):
    b = client().bucket(bucket).blob(blob_path)
    b.cache_control = "no-cache"
    b.upload_from_string(json.dumps(obj, indent=2), content_type="application/json")


def download_json(bucket: str, blob_path: str) -> dict | None:
    b = client().bucket(bucket).blob(blob_path)
    if not b.exists():
        return None
    return json.loads(b.download_as_bytes())


def upload_dir(bucket: str, blob_prefix: str, local_dir: Path):
    local_dir = Path(local_dir)
    bkt = client().bucket(bucket)
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            bkt.blob(f"{blob_prefix}/{rel}").upload_from_filename(str(p))


def download_dir(bucket: str, blob_prefix: str, local_dir: Path) -> int:
    """Download every blob under prefix into local_dir. Returns file count."""
    local_dir = Path(local_dir)
    n = 0
    for blob in client().list_blobs(bucket, prefix=blob_prefix.rstrip("/") + "/"):
        rel = blob.name[len(blob_prefix.rstrip("/")) + 1:]
        if not rel:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        n += 1
    return n


# -- regression-specific helpers ------------------------------------------------

def golden_current(bucket: str, prefix: str, workflow_id: str) -> dict | None:
    return download_json(bucket, f"{prefix}/golden/{workflow_id}/current.json")


def fetch_golden_outputs(bucket: str, prefix: str, workflow_id: str, tag: str,
                         dest: Path) -> int:
    return download_dir(bucket, f"{prefix}/golden/{workflow_id}/{tag}/outputs", dest)


def latest_pointer(bucket: str, prefix: str, branch: str) -> dict | None:
    return download_json(bucket, f"{prefix}/latest/{branch}.json")


def fetch_run_outputs(bucket: str, prefix: str, branch: str, commit: str,
                      workflow_id: str, dest: Path) -> int:
    return download_dir(bucket, f"{prefix}/runs/{branch}/{commit}/{workflow_id}/outputs", dest)


def publish_workflow_run(bucket: str, prefix: str, branch: str, commit: str,
                         workflow_id: str, local_dir: Path):
    upload_dir(bucket, f"{prefix}/runs/{branch}/{commit}/{workflow_id}", local_dir)


def publish_summary(bucket: str, prefix: str, branch: str, commit: str, summary: dict):
    upload_json(bucket, f"{prefix}/runs/{branch}/{commit}/summary.json", summary)


def update_latest(bucket: str, prefix: str, branch: str, pointer: dict):
    """Must be called after all per-workflow results and summary.json have landed."""
    upload_json(bucket, f"{prefix}/latest/{branch}.json", pointer)


def snapshot_manifest(bucket: str, prefix: str, commit: str, manifest: dict):
    upload_json(bucket, f"{prefix}/manifest-snapshot/{commit}.json", manifest)
