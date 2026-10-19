"""The lane registry (manifest/lanes.json) as the workflows consume it.

A lane is an environment the corpus runs in: a worker image (python + torch build
+ CUDA channel), the RunPod endpoint running that image, a cadence and a storage
prefix. Images are declared once under `images` and lanes point at them, so the
build matrix and the per-commit matrix are both derived from this one file.

A workflow can restrict itself to lanes with a `lanes` list in
manifest/workflows.json (null = every lane); workflows that need custom-node
packs must, because only images CI builds (`build` not false) bake the packs'
requirements. `workflow_runs_on_lane` is the one rule run_regression.py,
golden_baseline.py and the per-commit plan share.

  python scripts/lanes.py images [--only a,b]      JSON list of images to build
  python scripts/lanes.py per-commit [--only a,b]  JSON list of provisioned per-commit lanes
                                                   (each with the workflow ids enabled on it)
  python scripts/lanes.py lanes-on-image <image>   JSON list of lanes an image repoint applies to
  python scripts/lanes.py validate                 exit 1 with the problems, if any
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Standard library only: the workflows' plan jobs run this before installing anything.
LANES_FILE = Path(__file__).resolve().parents[1] / "manifest" / "lanes.json"
MANIFEST_FILE = Path(__file__).resolve().parents[1] / "manifest" / "workflows.json"
PACKS_FILE = Path(__file__).resolve().parents[1] / "manifest" / "custom_nodes.json"


def load_lanes(path: Path | None = None) -> dict:
    return json.loads((path or LANES_FILE).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


CADENCES = ("per-commit", "nightly", "weekly")
ROLES = ("primary", "lane", "legacy")
IMAGE_FIELDS = ("python", "cuda", "torch_index", "torch_spec")


def _csv(value: str | None) -> set[str] | None:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def image_matrix(lanes: dict, only: str | None = None) -> list[dict]:
    """Images build-image.yml builds: every image not marked `"build": false`."""
    wanted = _csv(only)
    out = []
    for image_id, image in (lanes.get("images") or {}).items():
        if image.get("build") is False and not (wanted and image_id in wanted):
            continue
        if wanted and image_id not in wanted:
            continue
        out.append({"image": image_id, **{k: image.get(k) for k in IMAGE_FIELDS}})
    return out


def workflow_runs_on_lane(entry: dict, lane_id: str) -> bool:
    """The manifest's lane filter: null means every lane."""
    lanes = entry.get("lanes")
    return lanes is None or lane_id in lanes


def workflows_for_lane(manifest: dict, lane_id: str, enabled_only: bool = True) -> list[str]:
    """Workflow ids that run on `lane_id`, in manifest order."""
    return [wf_id for wf_id, entry in (manifest.get("workflows") or {}).items()
            if (entry.get("enabled") or not enabled_only) and workflow_runs_on_lane(entry, lane_id)]


def per_commit_matrix(lanes: dict, only: str | None = None,
                      manifest: dict | None = None) -> list[dict]:
    """Lanes regression.yml tests on every new commit: per-commit cadence and an
    endpoint. With the workflows manifest, each row also names the enabled
    workflows the lane runs, as a csv (the rows feed a matrix `include`, so the
    plan shows what the lane filter leaves without nesting a list)."""
    wanted = _csv(only)
    out = []
    for lane_id, lane in (lanes.get("lanes") or {}).items():
        if wanted and lane_id not in wanted:
            continue
        if lane.get("cadence") != "per-commit" or not lane.get("endpoint_id"):
            continue
        row = {"lane": lane_id, "prefix": lane.get("prefix", "regression"),
               "endpoint_id": lane["endpoint_id"]}
        if manifest is not None:
            row["workflows"] = ",".join(workflows_for_lane(manifest, lane_id))
        out.append(row)
    return out


def lanes_on_image(lanes: dict, image_id: str) -> list[str]:
    """Lanes a fresh build of `image_id` is rolled out to automatically.

    Provisioned lanes on the image, except lanes in the legacy role and lanes
    whose image is not built by CI (`"build": false`): those keep the sha tag
    they were blessed on until they retire, and are only moved by hand
    (`runpod_endpoints.py set-image --lane`).
    """
    image = (lanes.get("images") or {}).get(image_id) or {}
    if image.get("build") is False:
        return []
    return [lane_id for lane_id, lane in (lanes.get("lanes") or {}).items()
            if lane.get("image") == image_id and lane.get("endpoint_id")
            and lane.get("role") != "legacy"]


def validate(lanes: dict, manifest: dict | None = None, packs: dict | None = None) -> list[str]:
    """Problems with a registry, as messages; an empty list means it is consistent.

    With the workflows manifest, every `lanes` filter must name declared lanes,
    and a workflow with `custom_nodes` must be restricted to lanes whose image
    CI builds (the legacy image is never rebuilt, so it bakes no packs). With
    the pack registry, every pack a workflow names must be declared."""
    problems: list[str] = []
    images = lanes.get("images") or {}
    entries = lanes.get("lanes") or {}
    if lanes.get("primary") not in entries:
        problems.append(f"primary {lanes.get('primary')!r} is not a lane")
    for image_id, image in images.items():
        for field in IMAGE_FIELDS:
            if not image.get(field):
                problems.append(f"image {image_id}: missing {field}")
    prefixes: dict[str, str] = {}
    for lane_id, lane in entries.items():
        if lane.get("image") not in images:
            problems.append(f"lane {lane_id}: image {lane.get('image')!r} is not declared")
        if lane.get("cadence") not in CADENCES:
            problems.append(f"lane {lane_id}: cadence {lane.get('cadence')!r}")
        if lane.get("role") not in ROLES:
            problems.append(f"lane {lane_id}: role {lane.get('role')!r}")
        if "endpoint_id" not in lane:
            problems.append(f"lane {lane_id}: endpoint_id must be present (null until provisioned)")
        prefix = lane.get("prefix")
        layout = lane.get("layout")
        if layout == 1 and prefix != "regression":
            problems.append(f"lane {lane_id}: layout 1 must use prefix 'regression'")
        elif layout == 2 and prefix != f"regression/lanes/{lane_id}":
            problems.append(f"lane {lane_id}: layout 2 must use prefix 'regression/lanes/{lane_id}'")
        elif layout not in (1, 2):
            problems.append(f"lane {lane_id}: layout {layout!r}")
        if prefix in prefixes:
            problems.append(f"lane {lane_id}: prefix {prefix!r} already used by {prefixes[prefix]}")
        prefixes[prefix] = lane_id
        image = images.get(lane.get("image")) or {}
        for field in ("python", "cuda"):
            if image.get(field) and lane.get(field) and image[field] != lane[field]:
                problems.append(f"lane {lane_id}: {field} {lane[field]!r} disagrees with its "
                                f"image's {image[field]!r}")
    layout1 = [lane_id for lane_id, lane in entries.items() if lane.get("layout") == 1]
    if len(layout1) > 1:
        problems.append(f"only one lane may use the legacy layout 1: {', '.join(layout1)}")

    for wf_id, entry in ((manifest or {}).get("workflows") or {}).items():
        filt = entry.get("lanes")
        if filt is not None:
            if not isinstance(filt, list) or not filt:
                problems.append(f"workflow {wf_id}: lanes must be null or a non-empty list")
                filt = []
            for lane_id in filt:
                if lane_id not in entries:
                    problems.append(f"workflow {wf_id}: lanes names unknown lane {lane_id!r}")
        names = entry.get("custom_nodes") or []
        if names:
            if filt is None:
                problems.append(f"workflow {wf_id}: needs custom node packs, so it must name the "
                                "lanes whose image bakes them (lanes: [...])")
            for lane_id in filt or []:
                image_id = (entries.get(lane_id) or {}).get("image")
                if (images.get(image_id) or {}).get("build") is False:
                    problems.append(f"workflow {wf_id}: lane {lane_id} runs image {image_id}, which "
                                    "is never rebuilt and so bakes no custom node packs")
            if packs is not None:
                for name in names:
                    if name not in (packs.get("packs") or {}):
                        problems.append(f"workflow {wf_id}: custom node pack {name!r} is not in "
                                        "manifest/custom_nodes.json")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("images")
    p.add_argument("--only", default=None)
    p = sub.add_parser("per-commit")
    p.add_argument("--only", default=None)
    p = sub.add_parser("lanes-on-image")
    p.add_argument("image")
    sub.add_parser("validate")
    args = ap.parse_args(argv)

    lanes = load_lanes()
    if args.cmd == "images":
        print(json.dumps(image_matrix(lanes, args.only)))
    elif args.cmd == "per-commit":
        manifest = load_json(MANIFEST_FILE) if MANIFEST_FILE.exists() else None
        print(json.dumps(per_commit_matrix(lanes, args.only, manifest)))
    elif args.cmd == "lanes-on-image":
        print(json.dumps(lanes_on_image(lanes, args.image)))
    elif args.cmd == "validate":
        manifest = load_json(MANIFEST_FILE) if MANIFEST_FILE.exists() else None
        packs = load_json(PACKS_FILE) if PACKS_FILE.exists() else None
        problems = validate(lanes, manifest, packs)
        for problem in problems:
            print(f"::error::{problem}")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
