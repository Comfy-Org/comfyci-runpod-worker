"""Keep the lanes' RunPod serverless endpoints in step with manifest/lanes.json.

  status [--lane ID]
      image, workers, GPU pool and CUDA floor of each provisioned lane's endpoint
  set-image (--lane ID | --image IMAGE) --tag TAG [--dry-run]
      point the lane's endpoint (or every non-legacy lane on IMAGE) at
      ghcr.io/comfy-org/comfyci-runpod-worker:TAG. TAG must start with the
      lane's image id, so a lane can never be handed another torch build.

Env: RUNPOD_API_KEY. Talks to the RunPod control-plane REST API v2
(https://api.runpod.io/v2, not the serverless job API on api.runpod.ai);
RUNPOD_REST_BASE overrides the base URL. Creating an endpoint stays a human
step (it is billable): create it from the image in the console or the API,
then commit its id as the lane's endpoint_id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

import lanes as lanes_mod
from index_builder import lane_config, load_lanes

REST_BASE = os.environ.get("RUNPOD_REST_BASE", "https://api.runpod.io/v2")
IMAGE_REPO = "ghcr.io/comfy-org/comfyci-runpod-worker"


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
    args = ap.parse_args(argv)

    lanes = load_lanes()
    if args.cmd == "status":
        return status(lanes, Client(), args.lane)
    if args.cmd == "set-image":
        client = None if args.dry_run else Client()
        return set_image(lanes, args.tag, args.lane, args.image, client, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
