"""Keep the lanes' RunPod serverless endpoints in step with manifest/lanes.json.

  status [--lane ID]
      image, workers, GPU pool and CUDA floor of each provisioned lane's endpoint
  set-image (--lane ID | --image IMAGE) --tag TAG [--dry-run]
      point the lane's endpoint (or every non-legacy lane on IMAGE) at
      ghcr.io/comfy-org/comfyci-runpod-worker:TAG. TAG must start with the
      lane's image id, so a lane can never be handed another torch build.
  create --lane ID --tag TAG [--dry-run] [--write]
      create the lane's endpoint from `endpoint_defaults` in lanes.json plus the
      lane's own GPU/CUDA settings, print its id, and with --write record it as
      the lane's endpoint_id. Billable once workers run (nothing while idle at
      workers.min 0); run it by hand, once per lane.

Env: RUNPOD_API_KEY. Talks to the RunPod control-plane REST API v2
(https://api.runpod.io/v2, not the serverless job API on api.runpod.ai);
RUNPOD_REST_BASE overrides the base URL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

import lanes as lanes_mod
from index_builder import LANES_FILE, lane_config, load_lanes

REST_BASE = os.environ.get("RUNPOD_REST_BASE", "https://api.runpod.io/v2")
IMAGE_REPO = "ghcr.io/comfy-org/comfyci-runpod-worker"
# What a lane's endpoint is called on RunPod: the image it runs, so the console
# reads like the registry.
ENDPOINT_NAME = "comfyci-{image}"


def image_ref(tag: str) -> str:
    return f"{IMAGE_REPO}:{tag}"


def check_tag(image_id: str, tag: str) -> None:
    """A lane only ever runs its own image: the tag is `<image>` or `<image>-<sha>`."""
    if tag != image_id and not tag.startswith(f"{image_id}-"):
        raise SystemExit(f"tag {tag!r} is not a build of image {image_id!r}")


class Client:
    def __init__(self, api_key: str | None = None, base: str = REST_BASE):
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise SystemExit("RUNPOD_API_KEY is not set")
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def get(self, endpoint_id: str) -> dict:
        r = requests.get(f"{self.base}/serverless/{endpoint_id}", headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def patch(self, endpoint_id: str, body: dict) -> dict:
        r = requests.patch(f"{self.base}/serverless/{endpoint_id}", headers=self.headers,
                           json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def create(self, body: dict) -> dict:
        r = requests.post(f"{self.base}/serverless", headers=self.headers, json=body, timeout=60)
        if r.status_code >= 400:
            raise SystemExit(f"create failed ({r.status_code}): {r.text[:1500]}")
        return r.json()


def endpoint_body(lanes: dict, lane_id: str, tag: str) -> dict:
    """The REST v2 CreateEndpointRequest for a lane, from endpoint_defaults + the lane."""
    cfg = lane_config(lanes, lane_id)
    if not cfg:
        raise SystemExit(f"unknown lane {lane_id!r}")
    if cfg.get("endpoint_id"):
        raise SystemExit(f"lane {lane_id!r} already has endpoint {cfg['endpoint_id']}")
    image_id = cfg.get("image") or ""
    check_tag(image_id, tag)
    d = lanes.get("endpoint_defaults") or {}
    missing = [k for k in ("gpu_pools", "network_volumes", "data_center_ids", "registry") if not d.get(k)]
    if missing:
        raise SystemExit(f"endpoint_defaults is missing {', '.join(missing)}")
    workers = dict(d.get("workers") or {"min": 0, "max": 3, "idleTimeout": 20})
    if cfg.get("workers_max") is not None:
        workers["max"] = cfg["workers_max"]
    return {
        "name": ENDPOINT_NAME.format(image=image_id),
        "type": "QUEUE",
        "image": image_ref(tag),
        "registry": d["registry"],
        "gpu": {"pools": list(d["gpu_pools"]), "count": int(d.get("gpu_count", 1)),
                # The lane's CUDA channel is the floor the host driver must meet.
                "minCudaVersion": str(cfg.get("min_cuda_driver") or cfg.get("cuda"))},
        "workers": workers,
        "scaling": dict(d.get("scaling") or {"type": "QUEUE_DELAY", "queueDelay": 4}),
        "timeout": int(d.get("timeout_ms", 3600000)),
        "flashboot": d.get("flashboot", "FLASHBOOT"),
        "networkVolumes": list(d["network_volumes"]),
        "dataCenterIds": list(d["data_center_ids"]),
        "disk": int(d.get("disk_gb", 40)),
        "env": dict(d.get("env") or {}),
    }


def write_endpoint_id(lane_id: str, endpoint_id: str, path=LANES_FILE) -> None:
    """Record a freshly created endpoint in lanes.json, keeping the file's formatting."""
    text = path.read_text(encoding="utf-8")
    lanes = json.loads(text)
    lanes["lanes"][lane_id]["endpoint_id"] = endpoint_id
    path.write_text(json.dumps(lanes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create(lanes: dict, lane_id: str, tag: str, client: Client | None, dry_run: bool,
           write: bool) -> int:
    body = endpoint_body(lanes, lane_id, tag)
    if dry_run or client is None:
        print(json.dumps(body, indent=2))
        return 0
    created = client.create(body)
    endpoint_id = created.get("id")
    print(f"[{lane_id}] created endpoint {endpoint_id} ({body['name']}) on {body['image']}")
    if write and endpoint_id:
        write_endpoint_id(lane_id, endpoint_id)
        print(f"[{lane_id}] endpoint_id written to {LANES_FILE}")
    return 0


def plan_set_image(lanes: dict, tag: str, lane_id: str | None = None,
                   image_id: str | None = None) -> list[tuple[str, str, dict]]:
    """(lane id, endpoint id, PATCH body) for every endpoint the tag applies to."""
    if lane_id:
        cfg = lane_config(lanes, lane_id)
        if not cfg:
            raise SystemExit(f"unknown lane {lane_id!r}")
        targets = [lane_id]
        image_id = cfg.get("image")
    elif image_id:
        targets = lanes_mod.lanes_on_image(lanes, image_id)
    else:
        raise SystemExit("pass --lane or --image")
    check_tag(image_id or "", tag)
    plan = []
    for target in targets:
        cfg = lane_config(lanes, target)
        if cfg.get("image") != image_id:
            raise SystemExit(f"lane {target!r} runs image {cfg.get('image')!r}, not {image_id!r}")
        if not cfg.get("endpoint_id"):
            print(f"[{target}] no endpoint yet; skipping")
            continue
        plan.append((target, cfg["endpoint_id"], {"image": image_ref(tag)}))
    return plan


def set_image(lanes: dict, tag: str, lane_id: str | None, image_id: str | None,
              client: Client | None, dry_run: bool) -> int:
    plan = plan_set_image(lanes, tag, lane_id, image_id)
    if not plan:
        print("nothing to repoint")
        return 0
    for target, endpoint_id, body in plan:
        if dry_run or client is None:
            print(f"[{target}] would PATCH {endpoint_id}: {json.dumps(body)}")
            continue
        client.patch(endpoint_id, body)
        print(f"[{target}] endpoint {endpoint_id} -> {body['image']}")
    return 0


def status(lanes: dict, client: Client, lane_id: str | None) -> int:
    for target, cfg in (lanes.get("lanes") or {}).items():
        if lane_id and target != lane_id:
            continue
        if not cfg.get("endpoint_id"):
            print(f"{target}: not provisioned")
            continue
        ep = client.get(cfg["endpoint_id"])
        gpu = ep.get("gpu") or {}
        workers = ep.get("workers") or {}
        print(f"{target}: {cfg['endpoint_id']} image={ep.get('image')} "
              f"pools={gpu.get('pools')} minCuda={gpu.get('minCudaVersion')} "
              f"workers={workers.get('min')}-{workers.get('max')} idle={workers.get('idleTimeout')}s "
              f"volumes={ep.get('networkVolumes')} dc={ep.get('dataCenterIds')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status")
    p.add_argument("--lane", default=None)
    p = sub.add_parser("set-image")
    p.add_argument("--lane", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("create")
    p.add_argument("--lane", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--write", action="store_true", help="record the new id in manifest/lanes.json")
    args = ap.parse_args(argv)

    lanes = load_lanes()
    if args.cmd == "status":
        return status(lanes, Client(), args.lane)
    if args.cmd == "set-image":
        client = None if args.dry_run else Client()
        return set_image(lanes, args.tag, args.lane, args.image, client, args.dry_run)
    if args.cmd == "create":
        client = None if args.dry_run else Client()
        return create(lanes, args.lane, args.tag, client, args.dry_run, args.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
