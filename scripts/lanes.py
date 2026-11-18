"""The lane registry (manifest/lanes.json) as the workflows consume it.

A lane is an environment the corpus runs in: a worker image (python + torch build
+ CUDA channel), the RunPod endpoint running that image, a cadence and a storage
prefix. Images are declared once under `images` and lanes point at them, so the
build matrix and the per-commit matrix are both derived from this one file.

A retired lane (cadence "retired") keeps its registry entry, storage prefix
and published history, but nothing dispatches GPU work to it any more: it is
left out of the per-commit matrix (even when named with --only), of image
repoints and of the build matrix, and run_regression.py / golden_baseline.py
refuse to run it.

Which lanes run a workflow is decided by two keys of its entry in
manifest/workflows.json. `lanes` is a list of lane ids (null = every lane);
workflows that need custom-node packs must use it, because only images CI
builds (`build` not false) bake the packs' requirements. `cadence` is
per-commit (the default), nightly or weekly, and a workflow runs on the lanes
of its own cadence. `why_not_on_lane` is the one rule run_regression.py,
golden_baseline.py, the reports and the plans of both workflow files share.

A workflow entry also says how it is checked. `level_of`, `cadence_of` and
`expect_of` read and validate its `level` (compared with a reference, or
smoke), its `cadence` and its `expect`.

Lanes can share one RunPod endpoint (the nightly and the weekly lane do).
`validate` holds such lanes to the same image and endpoint settings, with
what a lane leaves out filled in from `endpoint_defaults`.

A lane may name its own GPU pools, data centers and network volumes
(`gpu_pools`, `data_center_ids`, `network_volumes`). Each replaces the list of
`endpoint_defaults` (`placement_of`), and an empty list of volumes means no
volume (`without_volume`).

  python scripts/lanes.py images [--only a,b]      JSON list of images to build
                          [--since REV]            only the images changed after commit REV
  python scripts/lanes.py per-commit [--only a,b]  JSON list of provisioned per-commit lanes
                                                   (each with the workflow ids enabled on it)
  python scripts/lanes.py cadence <nightly|weekly> [--only a,b] [--workflows x,y]
                                                   JSON list of the provisioned lanes of that
                                                   cadence that have an enabled workflow, or
                                                   that run a workflow the dispatch names
  python scripts/lanes.py report [--only a,b]      JSON list of the lanes a report-only run shows:
                                                   every lane, retired and unprovisioned ones too
  python scripts/lanes.py lanes-on-image <image>   JSON list of lanes an image repoint applies to
  python scripts/lanes.py validate                 exit 1 with the problems, if any
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

# Standard library only: the workflows' plan jobs run this before installing anything.
REPO_ROOT = Path(__file__).resolve().parents[1]
LANES_FILE = Path(__file__).resolve().parents[1] / "manifest" / "lanes.json"
MANIFEST_FILE = Path(__file__).resolve().parents[1] / "manifest" / "workflows.json"
PACKS_FILE = Path(__file__).resolve().parents[1] / "manifest" / "custom_nodes.json"


def load_lanes(path: Path | None = None) -> dict:
    return json.loads((path or LANES_FILE).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


CADENCES = ("per-commit", "nightly", "weekly", "retired")
RETIRED = "retired"
ROLES = ("primary", "lane", "legacy")
IMAGE_FIELDS = ("python", "cuda", "torch_index", "torch_spec")
# What every worker image is built from, as paths in the repository, next to
# its own IMAGE_FIELDS in the registry. A change under one rebuilds them all.
IMAGE_SOURCES = ("Dockerfile", "src/", "docker/", "manifest/custom_nodes.json")
REGISTRY_PATH = "manifest/lanes.json"


def is_retired(lane: dict | None) -> bool:
    """A retired lane's history stays readable, but it runs nothing any more."""
    return (lane or {}).get("cadence") == RETIRED


LONG_CADENCES = ("nightly", "weekly")
# What a workflow entry of manifest/workflows.json may say about how it is checked.
LEVELS = ("reference", "smoke")
WORKFLOW_CADENCES = ("per-commit", "nightly", "weekly")
EXPECT_KEYS = ("width", "height", "frames", "seconds", "min_chars")
# Optional lane keys that count something: whole numbers above 0.
LANE_COUNTS = ("workers_max", "disk_gb", "idle_timeout_s", "max_in_flight", "timeout_minutes")
# Where a lane's models are: on the network volume, or fetched for each job.
MODEL_STORAGE = ("volume", "job")
# Lists a lane may name in place of the ones `endpoint_defaults` has: the GPU
# pools its workers run on, their data centers and their network volumes.
PLACEMENT_KEYS = ("gpu_pools", "data_center_ids", "network_volumes")
# What lanes on one endpoint must say alike: an endpoint has one of each.
ENDPOINT_KEYS = ("image", "endpoint_name", "workers_max", "disk_gb", "idle_timeout_s", "env",
                 *PLACEMENT_KEYS)
# The time limit of a nightly or weekly job whose lane names none: GitHub's own
# limit for a job on a hosted runner.
DEFAULT_TIMEOUT_MINUTES = 360
# The last minutes of such a lane's job belong to the publish and the report:
# the run's own deadline is the lane's `timeout_minutes` minus this.
DEADLINE_MARGIN_MIN = 25
# Seconds a job of such a run can take beyond its workflow's `timeout_s`: what
# the runner allows for a cold start and for the queue. It is
# submit_and_poll.poll_deadline_s(0, run_regression.LONG_QUEUE_DEADLINE_S),
# written as a number because this module imports nothing. A test holds the
# two alike.
LONG_JOB_ALLOWANCE_S = 2100
# What an endpoint gets when neither its lane nor `endpoint_defaults` says:
# its workers, the disk of a worker in GB, and its name.
DEFAULT_WORKERS = {"min": 0, "max": 3, "idleTimeout": 20}
DEFAULT_DISK_GB = 40
ENDPOINT_NAME = "comfyci-{image}"


def is_long_cadence(lane: dict | None) -> bool:
    """A lane that runs once a night or once a week, not on every commit."""
    return (lane or {}).get("cadence") in LONG_CADENCES


def deadline_window_s(lane: dict) -> int:
    """Seconds between the start of a nightly or weekly run and its deadline:
    the lane's `timeout_minutes` (DEFAULT_TIMEOUT_MINUTES when it names none)
    minus DEADLINE_MARGIN_MIN. A job whose worst case does not fit before the
    deadline is not started, so nothing starts on a lane where this is 0 or
    less."""
    return ((lane.get("timeout_minutes") or DEFAULT_TIMEOUT_MINUTES) - DEADLINE_MARGIN_MIN) * 60


def placement_of(lanes: dict, lane: dict | None, key: str) -> list | None:
    """A lane's GPU pools, data centers or network volumes (`key`, one of
    PLACEMENT_KEYS): its own list when it names one, else the list of
    `endpoint_defaults`. None when neither names one."""
    own = (lane or {}).get(key)
    return own if own is not None else (lanes.get("endpoint_defaults") or {}).get(key)


def without_volume(lanes: dict, lane: dict | None) -> bool:
    """Whether a lane's workers mount no network volume: its list of volumes
    (`placement_of`) is empty. Such a lane cannot read models from a volume."""
    return placement_of(lanes, lane, "network_volumes") == []


def on_volume(lanes: dict, lane: dict | None) -> bool:
    """Whether a lane reads its models from the network volume that the model
    sync fills: it is not retired, its workers mount a volume, and it does not
    say `"model_storage": "job"`."""
    return not (is_retired(lane) or (lane or {}).get("model_storage") == "job"
                or without_volume(lanes, lane))


def endpoint_settings_of(lanes: dict, lane: dict) -> dict:
    """What a lane asks of its endpoint, key by key of ENDPOINT_KEYS, with
    what the lane leaves out filled in from `endpoint_defaults`: what
    `runpod_endpoints.py create` sends for it. Lanes on one endpoint must
    agree on these, whichever of them writes a value out."""
    d = lanes.get("endpoint_defaults") or {}
    workers = d.get("workers") or DEFAULT_WORKERS

    def own(key: str, default):
        return lane[key] if lane.get(key) is not None else default

    env = [e if isinstance(e, dict) else {} for e in (d.get("env"), lane.get("env"))]
    return {"image": lane.get("image"),
            "endpoint_name": (lane.get("endpoint_name")
                              or ENDPOINT_NAME.format(image=lane.get("image") or "")),
            "workers_max": own("workers_max", workers.get("max")),
            "disk_gb": own("disk_gb", d.get("disk_gb", DEFAULT_DISK_GB)),
            "idle_timeout_s": own("idle_timeout_s", workers.get("idleTimeout")),
            "env": {**env[0], **env[1]},
            **{key: placement_of(lanes, lane, key) for key in PLACEMENT_KEYS}}


def _choice(entry: dict | None, key: str, known: tuple[str, ...]) -> str:
    """An entry's value for `key`, one of `known` (absent or null: the first)."""
    value = (entry or {}).get(key)
    if value is None:
        return known[0]
    if not isinstance(value, str) or value not in known:
        raise ValueError(f"unknown {key} {value!r} (known: {', '.join(known)})")
    return value


def level_of(entry: dict | None) -> str:
    """The check level a workflow entry declares. "reference" (absent or null):
    its output is compared with an approved reference. "smoke": it only has to
    finish and write usable files (smoke.py). ValueError for anything else."""
    return _choice(entry, "level", LEVELS)


def cadence_of(entry: dict | None) -> str:
    """How often a workflow entry asks to be run (absent or null: per-commit);
    ValueError for anything else than a known cadence."""
    return _choice(entry, "cadence", WORKFLOW_CADENCES)


def expect_of(entry: dict | None) -> dict:
    """What a smoke-level entry says its output must be: `expect`, an object
    with keys from EXPECT_KEYS and numbers above 0 ({} when absent or null).
    ValueError for anything else, and for an entry that is not smoke level:
    there the output is held to its reference."""
    expect = (entry or {}).get("expect")
    if expect is None:
        return {}
    if level_of(entry) != "smoke":
        raise ValueError("expect is for a smoke-level workflow only")
    if not isinstance(expect, dict):
        raise ValueError("expect must be an object")
    for key, value in expect.items():
        if key not in EXPECT_KEYS:
            raise ValueError(f"unknown expect key {key!r} (known: {', '.join(EXPECT_KEYS)})")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"expect.{key} must be a number above 0, not {value!r}")
    return expect


def retired_only_images(lanes: dict) -> set[str]:
    """Images that lanes run, but only retired ones: nothing would use a new build."""
    users: dict[str, list[dict]] = {}
    for lane in (lanes.get("lanes") or {}).values():
        users.setdefault(lane.get("image"), []).append(lane)
    return {image_id for image_id, on in users.items() if on and all(is_retired(l) for l in on)}


def _csv(value: str | None) -> set[str] | None:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def _git(repo: Path, *args: str) -> str | None:
    """What a git command prints in `repo`; None when it fails or there is no git."""
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def images_since(lanes: dict, rev: str | None, repo: Path | None = None
                 ) -> tuple[set[str] | None, str]:
    """(ids of the images the commits after `rev` changed, why). None for the
    ids means every image.

    Every image is built from IMAGE_SOURCES, so a change to one of those
    changes them all. A change to the registry alone changes only the images
    whose row of the build matrix differs from the one `rev` had, which is
    none when a lane was added or an endpoint id was written. An image also
    counts as changed when a build of it now repoints a lane that had its
    endpoint at `rev` and was not repointed with the image then (the lane
    moved to the image, left the legacy role or is no longer retired): the
    build is what moves that endpoint to the image. It compares with `rev`,
    the commit before a push, and not with the parent of HEAD: a push can
    hold several commits. When `rev` is not known here (all zeros for a new
    branch, gone after a forced push) every image is built."""
    repo = repo or REPO_ROOT
    if not re.fullmatch(r"[0-9a-f]{7,40}", rev or "") or not rev.strip("0"):
        return None, f"every image is built: no commit to compare with ({rev or 'none'})"
    # -z: git quotes a name with a byte above 127 or a quote in it otherwise,
    # and such a name would start with the quote, never with a source path.
    names = _git(repo, "diff", "--name-only", "--no-renames", "-z", rev, "HEAD")
    old_text = _git(repo, "show", f"{rev}:{REGISTRY_PATH}")
    if names is None or old_text is None:
        return None, (f"every image is built: {rev[:8]} or its {REGISTRY_PATH} is not in "
                      "this checkout")
    sources = [n for n in names.split("\0") if n.startswith(IMAGE_SOURCES)]
    if sources:
        return None, f"every image is built: {sources[0]} changed since {rev[:8]}"
    try:
        old = json.loads(old_text)
        before = {row["image"]: row for row in image_matrix(old)}
        had_endpoint = {lane_id for lane_id, lane in (old.get("lanes") or {}).items()
                        if lane.get("endpoint_id")}
        # Per image of today: the lanes with an endpoint that a build did not repoint then.
        waiting = {image_id: had_endpoint - set(lanes_on_image(old, image_id))
                   for image_id in (lanes.get("images") or {})}
    except (ValueError, AttributeError, TypeError):
        return None, f"every image is built: {REGISTRY_PATH} of {rev[:8]} could not be read"
    changed = {row["image"] for row in image_matrix(lanes)
               if before.get(row["image"]) != row
               or waiting[row["image"]] & set(lanes_on_image(lanes, row["image"]))}
    return changed, (f"images changed since {rev[:8]}: {', '.join(sorted(changed))}" if changed
                     else f"no image changed since {rev[:8]}: nothing is built")


def image_matrix(lanes: dict, only: str | None = None, since: str | None = None,
                 repo: Path | None = None) -> list[dict]:
    """Images build-image.yml builds: every image not marked `"build": false`
    (unless asked for by name), never one that only retired lanes run. With
    `since` (a commit), only the images changed after it (`images_since`)."""
    wanted = _csv(only)
    retired = retired_only_images(lanes)
    changed = images_since(lanes, since, repo)[0] if since is not None else None
    out = []
    for image_id, image in (lanes.get("images") or {}).items():
        if image_id in retired:
            continue
        if image.get("build") is False and not (wanted and image_id in wanted):
            continue
        if wanted and image_id not in wanted:
            continue
        if changed is not None and image_id not in changed:
            continue
        out.append({"image": image_id, **{k: image.get(k) for k in IMAGE_FIELDS}})
    return out


def why_not_on_lane(entry: dict, lane_id: str, lane: dict | None = None) -> str | None:
    """None when the workflow runs on the lane, else why not, in the words a
    log line prints. The manifest's `lanes` filter comes first (null means
    every lane), then the cadence: a workflow runs on the lanes of its own
    cadence. Without `lane` (the lane's entry in the registry), or for a
    retired lane, only the filter applies, so the references of a retired lane
    can still be blessed. ValueError for an entry with an unknown cadence."""
    names = entry.get("lanes")
    if names is not None and lane_id not in names:
        return f"manifest lanes: {', '.join(str(name) for name in names)}"
    if lane is None or is_retired(lane):
        return None
    cadence = cadence_of(entry)
    if cadence != lane.get("cadence"):
        return f"cadence {cadence}, the lane is {lane.get('cadence')}"
    return None


def workflow_runs_on_lane(entry: dict, lane_id: str, lane: dict | None = None) -> bool:
    """Whether a workflow entry runs on a lane (`why_not_on_lane`)."""
    return why_not_on_lane(entry, lane_id, lane) is None


def workflows_for_lane(manifest: dict, lane_id: str, enabled_only: bool = True,
                       lane: dict | None = None) -> list[str]:
    """Workflow ids that run on `lane_id`, in manifest order. With `lane`, the
    lane's entry in the registry, the cadence counts too. ValueError, naming
    the workflow as a log line does, for an entry with an unknown cadence."""
    out = []
    for wf_id, entry in (manifest.get("workflows") or {}).items():
        if enabled_only and not entry.get("enabled"):
            continue
        try:
            runs = workflow_runs_on_lane(entry, lane_id, lane)
        except ValueError as e:
            raise ValueError(f"[{wf_id}] {e}") from None
        if runs:
            out.append(wf_id)
    return out


def lanes_for_workflow(entry: dict, lanes: dict) -> list[str]:
    """Ids of the lanes that run a workflow entry and are not retired, in the
    order of the registry. A lane without an endpoint counts: it runs the
    workflow once it has one."""
    return [lane_id for lane_id, lane in (lanes.get("lanes") or {}).items()
            if not is_retired(lane) and workflow_runs_on_lane(entry, lane_id, lane)]


def volume_workflows(manifest: dict, lanes: dict) -> list[str]:
    """Ids of the workflows whose models the network volume has to hold, in
    manifest order: every workflow, switched on or not, that runs on a lane
    whose models are on the volume (`on_volume`). The model sync downloads the
    models of these and of no other, and the manifest lint adds up their
    sizes, so the two cannot disagree about what the volume holds. ValueError
    for an entry with an unknown cadence."""
    found = set()
    for lane_id, lane in (lanes.get("lanes") or {}).items():
        if on_volume(lanes, lane):
            found.update(workflows_for_lane(manifest, lane_id, enabled_only=False, lane=lane))
    return [wf_id for wf_id in (manifest.get("workflows") or {}) if wf_id in found]


def named_on_lane(manifest: dict, named: str | None, lane_id: str, lane: dict) -> list[str]:
    """The workflows a dispatch names (`named`, a csv of ids) that run on a
    lane, switched on or not: run_regression.py runs a workflow that is named,
    also one that is switched off. Empty when the dispatch names none ("all",
    or nothing)."""
    wanted = _csv(named) if named != "all" else None
    if not wanted:
        return []
    entries = {wf_id: entry for wf_id, entry in (manifest.get("workflows") or {}).items()
               if wf_id in wanted}
    return workflows_for_lane({"workflows": entries}, lane_id, enabled_only=False, lane=lane)


def per_commit_matrix(lanes: dict, only: str | None = None,
                      manifest: dict | None = None) -> list[dict]:
    """Lanes regression.yml tests on every new commit: per-commit cadence and an
    endpoint, also when `only` names a lane (a retired or unprovisioned lane
    stays out). With the workflows manifest, each row also names the enabled
    workflows the lane runs, as a csv (the rows feed a matrix `include`, so the
    plan shows what the lane filter leaves without nesting a list). A nightly
    or weekly workflow is not among them."""
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
            row["workflows"] = ",".join(workflows_for_lane(manifest, lane_id, lane=lane))
        out.append(row)
    return out


def cadence_matrix(lanes: dict, cadence: str, only: str | None = None,
                   manifest: dict | None = None, named: str | None = None) -> list[dict]:
    """Lanes nightly-weekly.yml tests in a run of `cadence` (nightly or
    weekly): that cadence and an endpoint. Each row carries the time limit of
    the lane's job. With the workflows manifest it also names the enabled
    workflows the lane runs, as a csv, and a lane that has none is left out:
    the run starts nothing for it. Unless the dispatch names a workflow that
    runs on the lane (`named`, its `workflows` input): a first look at a
    workflow that is still switched off."""
    if cadence not in LONG_CADENCES:
        raise ValueError(f"unknown cadence {cadence!r} (known: {', '.join(LONG_CADENCES)})")
    wanted = _csv(only)
    out = []
    for lane_id, lane in (lanes.get("lanes") or {}).items():
        if wanted and lane_id not in wanted:
            continue
        if lane.get("cadence") != cadence or not lane.get("endpoint_id"):
            continue
        row = {"lane": lane_id, "prefix": lane.get("prefix", "regression"),
               "endpoint_id": lane["endpoint_id"],
               "timeout_minutes": lane.get("timeout_minutes") or DEFAULT_TIMEOUT_MINUTES}
        if manifest is not None:
            row["workflows"] = ",".join(workflows_for_lane(manifest, lane_id, lane=lane))
            if not (row["workflows"] or named_on_lane(manifest, named, lane_id, lane)):
                continue
        out.append(row)
    return out


def cadence_left_out(lanes: dict, cadence: str, only: str | None = None,
                     manifest: dict | None = None, named: str | None = None) -> list[str]:
    """Why a lane is not in the matrix of a nightly or weekly run: each lane
    named in `only`, or without it each lane of the cadence."""
    entries = lanes.get("lanes") or {}
    wanted = _csv(only)
    names = sorted(wanted) if wanted else [lane_id for lane_id, lane in entries.items()
                                           if lane.get("cadence") == cadence]
    notes = []
    for lane_id in names:
        lane = entries.get(lane_id)
        if lane is None:
            notes.append(f"{lane_id}: not a lane in manifest/lanes.json")
        elif lane.get("cadence") != cadence:
            notes.append(f"{lane_id}: cadence {lane.get('cadence')!r}, not {cadence}")
        elif not lane.get("endpoint_id"):
            notes.append(f"{lane_id}: no endpoint_id yet")
        elif manifest is not None and not (workflows_for_lane(manifest, lane_id, lane=lane)
                                           or named_on_lane(manifest, named, lane_id, lane)):
            notes.append(f"{lane_id}: no enabled workflow with the {cadence} cadence" + (
                ", and the run names none that runs on the lane" if _csv(named) and named != "all"
                else ""))
    return notes


def report_matrix(lanes: dict, only: str | None = None) -> list[dict]:
    """Lanes a report-only run of regression.yml shows: every lane, whatever
    its cadence or endpoint. Such a run reads published results and starts no
    GPU job, so a retired lane's history can be shown like any other."""
    wanted = _csv(only)
    return [{"lane": lane_id, "prefix": lane.get("prefix", "regression")}
            for lane_id, lane in (lanes.get("lanes") or {}).items()
            if not wanted or lane_id in wanted]


def left_out(lanes: dict, only: str | None) -> list[str]:
    """Why each lane named in `only` is not in the per-commit matrix."""
    entries = lanes.get("lanes") or {}
    notes = []
    for lane_id in sorted(_csv(only) or ()):
        lane = entries.get(lane_id)
        if lane is None:
            notes.append(f"{lane_id}: not a lane in manifest/lanes.json")
        elif lane.get("cadence") != "per-commit":
            notes.append(f"{lane_id}: cadence {lane.get('cadence')!r}, not per-commit")
        elif not lane.get("endpoint_id"):
            notes.append(f"{lane_id}: no endpoint_id yet")
    return notes


def lanes_on_image(lanes: dict, image_id: str) -> list[str]:
    """Lanes a fresh build of `image_id` is rolled out to automatically.

    Provisioned lanes on the image, except lanes in the legacy role, retired
    lanes and lanes whose image is not built by CI (`"build": false`): those
    keep the sha tag they were blessed on. A legacy lane still in service is
    moved by hand (`runpod_endpoints.py set-image --lane`), a retired one not
    at all.
    """
    image = (lanes.get("images") or {}).get(image_id) or {}
    if image.get("build") is False:
        return []
    return [lane_id for lane_id, lane in (lanes.get("lanes") or {}).items()
            if lane.get("image") == image_id and lane.get("endpoint_id")
            and lane.get("role") != "legacy" and not is_retired(lane)]


def _is_count(value) -> bool:
    """A whole number above 0: what a lane key that counts something holds."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _lane_key_problems(lane_id: str, lane: dict) -> list[str]:
    """Problems with the optional keys of one lane: its counts, where its
    models are, the name and environment of its endpoint, and its own GPU
    pools, data centers and network volumes."""
    problems = []
    for key in LANE_COUNTS:
        value = lane.get(key)
        if value is not None and not _is_count(value):
            problems.append(f"lane {lane_id}: {key} must be a whole number above 0, "
                            f"not {value!r}")
    for key in PLACEMENT_KEYS:
        value = lane.get(key)
        ids = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        # Only the volumes may be an empty list: it means no volume.
        if value is None or (ids and (value or key == "network_volumes")):
            continue
        wanted = ("a list of volume ids, or [] for no volume" if key == "network_volumes"
                  else "a list of at least one id")
        problems.append(f"lane {lane_id}: {key} must be {wanted}, not {value!r}")
    where = lane.get("model_storage")
    if where is not None and not (isinstance(where, str) and where in MODEL_STORAGE):
        problems.append(f"lane {lane_id}: model_storage must be one of "
                        f"{', '.join(MODEL_STORAGE)}, not {where!r}")
    env = lane.get("env")
    if env is not None and not (isinstance(env, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        problems.append(f"lane {lane_id}: env must map variable names to strings")
    name = lane.get("endpoint_name")
    if name is not None and not (isinstance(name, str) and name):
        problems.append(f"lane {lane_id}: endpoint_name must be a name, not {name!r}")
    return problems


def _shared_endpoint_problems(lanes: dict) -> list[str]:
    """Problems with lanes that share an endpoint: the same `endpoint_name` or
    the same `endpoint_id`. An endpoint runs one image with one set of
    settings, so every lane on it must ask for the same, and one name cannot
    stand for two endpoints. What a lane asks for is compared with the
    defaults filled in (`endpoint_settings_of`): a lane that writes a default
    out agrees with one that leaves it out. A lane whose id is still null
    shares by its name. A retired lane runs nothing and is left out."""
    entries = lanes.get("lanes") or {}
    problems = []
    first: dict[tuple[str, str], str] = {}
    for lane_id, lane in entries.items():
        if is_retired(lane):
            continue
        others = []
        for key in ("endpoint_name", "endpoint_id"):
            value = lane.get(key)
            if isinstance(value, str) and value:
                other = first.setdefault((key, value), lane_id)
                if other != lane_id and other not in others:
                    others.append(other)
        for other in others:
            there = entries[other]
            asked, asked_there = (endpoint_settings_of(lanes, x) for x in (lane, there))
            for key in ENDPOINT_KEYS:
                if asked[key] != asked_there[key]:
                    problems.append(f"lane {lane_id}: {key} is {asked[key]!r}, but {other}, "
                                    f"which shares its endpoint, says {asked_there[key]!r}")
            mine, theirs = lane.get("endpoint_id"), there.get("endpoint_id")
            if mine and theirs and mine != theirs:
                problems.append(f"lane {lane_id}: endpoint_id is {mine!r}, but {other}, which "
                                f"has the same endpoint_name, says {theirs!r}")
    return problems


def _never_started(wf_id: str, timeout_s, entries: dict, on: list[str]) -> list[str]:
    """Problems with an enabled workflow on the nightly and weekly lanes that
    run it (`on`): a job that at its worst takes as long as a run has before
    its deadline is never submitted, so the workflow would be `not_reached`
    in every run. `timeout_s` is the workflow's own or the manifest's default.
    A lane whose time limit leaves no time at all is said once, for the lane."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        return []
    need = timeout_s + LONG_JOB_ALLOWANCE_S
    problems = []
    for lane_id in on:
        lane = entries[lane_id]
        minutes = lane.get("timeout_minutes")
        if not is_long_cadence(lane) or not (minutes is None or _is_count(minutes)):
            continue
        window = deadline_window_s(lane)
        if 0 < window <= need:
            problems.append(
                f"workflow {wf_id}: it can never start on lane {lane_id}: its job can take "
                f"{math.ceil(need / 60)} min (timeout_s {timeout_s:g} and {LONG_JOB_ALLOWANCE_S} s "
                f"for the queue and a cold start), and a run's deadline is {window // 60} min "
                "after its start. Raise the lane's timeout_minutes or lower timeout_s")
    return problems


def validate(lanes: dict, manifest: dict | None = None, packs: dict | None = None) -> list[str]:
    """Problems with a registry, as messages; an empty list means it is consistent.

    A lane without a network volume, retired lanes aside, must say
    `"model_storage": "job"`: there is no volume to read its models from. A
    nightly or weekly lane must have more than DEADLINE_MARGIN_MIN minutes.

    With the workflows manifest, every `lanes` filter must name declared lanes,
    a workflow with `custom_nodes` must be restricted to lanes whose image
    CI builds (the legacy image is never rebuilt, so it bakes no packs), and an
    enabled workflow must run on at least one lane that is not retired, with a
    job that fits before the deadline of a nightly or weekly lane's run. With
    the pack registry, every pack a workflow names must be declared."""
    problems: list[str] = []
    images = lanes.get("images") or {}
    entries = lanes.get("lanes") or {}
    primary_id = lanes.get("primary")
    if primary_id not in entries:
        problems.append(f"primary {primary_id!r} is not a lane")
    else:
        # The primary is the lane PULSE comments for and the dashboard opens on:
        # it has to be tested on every commit.
        primary = entries[primary_id]
        if primary.get("role") != "primary":
            problems.append(f"primary lane {primary_id}: role must be 'primary', "
                            f"not {primary.get('role')!r}")
        if primary.get("cadence") != "per-commit":
            problems.append(f"primary lane {primary_id}: cadence must be 'per-commit', "
                            f"not {primary.get('cadence')!r}")
    for lane_id, lane in entries.items():
        if lane_id != primary_id and lane.get("role") == "primary":
            problems.append(f"lane {lane_id}: role 'primary', but the primary lane is "
                            f"{primary_id!r}")
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
        problems += _lane_key_problems(lane_id, lane)
        # The model sync fills a network volume. A lane without one reads none of it.
        if (not is_retired(lane) and without_volume(lanes, lane)
                and lane.get("model_storage") != "job"):
            problems.append(f"lane {lane_id}: it has no network volume, so model_storage must "
                            f"be 'job', not {lane.get('model_storage')!r}")
        # A nightly or weekly run stops submitting jobs DEADLINE_MARGIN_MIN
        # minutes before the lane's time limit. A limit that is no longer than
        # that leaves every workflow of every run without a job.
        minutes = lane.get("timeout_minutes")
        if is_long_cadence(lane) and _is_count(minutes) and minutes <= DEADLINE_MARGIN_MIN:
            problems.append(f"lane {lane_id}: timeout_minutes {minutes} leaves no time for a "
                            f"job: a run's deadline is {DEADLINE_MARGIN_MIN} minutes before it")
    problems += _shared_endpoint_problems(lanes)
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
        # The two keys together can leave a workflow nowhere, and no run would
        # turn red for it: a pack workflow names its lanes, so a new cadence
        # alone takes it off them without putting it on another.
        if entry.get("enabled") and (entry.get("lanes") is None or filt):
            try:
                cadence = cadence_of(entry)
                on = lanes_for_workflow(entry, lanes)
                if not on:
                    problems.append(
                        f"workflow {wf_id}: cadence {cadence!r} and lanes "
                        f"{json.dumps(entry.get('lanes'))} leave it on no lane: "
                        + (f"add a {cadence} lane to its lanes list" if filt else
                           f"manifest/lanes.json has no {cadence} lane that is not retired"))
                timeout_s = entry.get("timeout_s")
                if timeout_s is None:
                    timeout_s = (manifest.get("defaults") or {}).get("timeout_s")
                problems += _never_started(wf_id, timeout_s, entries, on)
            except ValueError as e:
                problems.append(f"workflow {wf_id}: {e}")
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
    p.add_argument("--since", default=None,
                   help="a commit: list only the images changed after it (the commit before "
                        "a push). Every image when it is not known here")
    p = sub.add_parser("per-commit")
    p.add_argument("--only", default=None)
    p = sub.add_parser("cadence")
    p.add_argument("cadence", choices=list(LONG_CADENCES))
    p.add_argument("--only", default=None)
    p.add_argument("--workflows", default=None,
                   help="the `workflows` input of a dispatch: a lane with no enabled workflow "
                        "is planned when one of these runs on it")
    p = sub.add_parser("report")
    p.add_argument("--only", default=None)
    p = sub.add_parser("lanes-on-image")
    p.add_argument("image")
    sub.add_parser("validate")
    args = ap.parse_args(argv)

    lanes = load_lanes()
    if args.cmd == "images":
        if args.since is not None:
            print(f"::notice::{images_since(lanes, args.since)[1]}", file=sys.stderr)
        print(json.dumps(image_matrix(lanes, args.only, args.since)))
    elif args.cmd == "per-commit":
        manifest = load_json(MANIFEST_FILE) if MANIFEST_FILE.exists() else None
        for note in left_out(lanes, args.only):
            print(f"::notice::not tested: {note}", file=sys.stderr)
        print(json.dumps(per_commit_matrix(lanes, args.only, manifest)))
    elif args.cmd == "cadence":
        manifest = load_json(MANIFEST_FILE) if MANIFEST_FILE.exists() else None
        for note in cadence_left_out(lanes, args.cadence, args.only, manifest, args.workflows):
            print(f"::notice::not tested: {note}", file=sys.stderr)
        print(json.dumps(cadence_matrix(lanes, args.cadence, args.only, manifest,
                                        args.workflows)))
    elif args.cmd == "report":
        for lane_id in sorted((_csv(args.only) or set()) - set(lanes.get("lanes") or {})):
            print(f"::notice::not shown: {lane_id}: not a lane in manifest/lanes.json",
                  file=sys.stderr)
        print(json.dumps(report_matrix(lanes, args.only)))
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
