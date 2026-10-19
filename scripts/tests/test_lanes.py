"""The lane registry and what the workflows derive from it."""
import copy
import json
import re
from pathlib import Path

import pytest

import index_builder
import lanes as lanes_mod
import runpod_endpoints

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "manifest" / "lanes.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
PACKS = json.loads((ROOT / "manifest" / "custom_nodes.json").read_text(encoding="utf-8"))
# The committed registry provisions lanes over time; these tests want one lane that is
# declared but not yet provisioned, whatever the file says today.
NEW = "py313-torch2.14.1-cu130"
LEGACY = "py312-torch2.11.0-cu128"
PENDING = copy.deepcopy(REGISTRY)
PENDING["lanes"][NEW]["endpoint_id"] = None


def test_the_committed_registry_is_consistent():
    assert lanes_mod.validate(REGISTRY) == []
    assert lanes_mod.validate(REGISTRY, MANIFEST, PACKS) == []
    assert REGISTRY["schema_version"] == 2


def test_validate_names_each_problem():
    bad = copy.deepcopy(REGISTRY)
    bad["primary"] = "nope"
    lane = bad["lanes"]["py313-torch2.14.1-cu130"]
    lane["image"] = "ghost"
    lane["prefix"] = "regression/elsewhere"
    lane["cadence"] = "hourly"
    del lane["endpoint_id"]
    bad["images"]["cu130-t2141"].pop("torch_spec")
    problems = "\n".join(lanes_mod.validate(bad))
    for needle in ("primary 'nope'", "image 'ghost'", "regression/lanes/py313-torch2.14.1-cu130",
                   "cadence 'hourly'", "endpoint_id must be present", "missing torch_spec"):
        assert needle in problems


def test_validate_rejects_a_second_legacy_layout_and_prefix_reuse():
    bad = copy.deepcopy(REGISTRY)
    bad["lanes"]["py313-torch2.14.1-cu130"].update({"layout": 1, "prefix": "regression"})
    problems = "\n".join(lanes_mod.validate(bad))
    assert "only one lane may use the legacy layout 1" in problems
    assert "already used by" in problems


def test_validate_checks_lane_filters_and_the_pack_lane_rule():
    manifest = {"workflows": {
        "typo": {"enabled": True, "lanes": ["nope"]},
        "empty": {"enabled": True, "lanes": []},
        "unrestricted_pack": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"]},
        "legacy_pack": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"], "lanes": [LEGACY]},
        "ghost_pack": {"enabled": True, "custom_nodes": ["ComfyUI-Ghost"], "lanes": [NEW]},
        "fine": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"], "lanes": [NEW]},
        "plain": {"enabled": True},
    }}
    problems = lanes_mod.validate(REGISTRY, manifest, PACKS)
    text = "\n".join(problems)
    assert "workflow typo: lanes names unknown lane 'nope'" in text
    assert "workflow empty: lanes must be null or a non-empty list" in text
    assert "workflow unrestricted_pack: needs custom node packs" in text
    assert f"workflow legacy_pack: lane {LEGACY} runs image cu128-t2110, which is never rebuilt" in text
    assert "workflow ghost_pack: custom node pack 'ComfyUI-Ghost' is not in" in text
    assert not [p for p in problems if p.startswith("workflow fine") or p.startswith("workflow plain")]
    # Without the pack registry, unknown pack names are not checked here.
    assert "ComfyUI-Ghost" not in "\n".join(lanes_mod.validate(REGISTRY, manifest))


def test_workflow_lane_filter():
    assert lanes_mod.workflow_runs_on_lane({"lanes": None}, "any")
    assert lanes_mod.workflow_runs_on_lane({}, "any")
    assert lanes_mod.workflow_runs_on_lane({"lanes": [NEW]}, NEW)
    assert not lanes_mod.workflow_runs_on_lane({"lanes": [NEW]}, LEGACY)
    manifest = {"workflows": {
        "a": {"enabled": True},
        "b": {"enabled": True, "lanes": [NEW]},
        "c": {"enabled": False, "lanes": [NEW]},
        "d": {"enabled": True, "lanes": [LEGACY]},
    }}
    assert lanes_mod.workflows_for_lane(manifest, NEW) == ["a", "b"]
    assert lanes_mod.workflows_for_lane(manifest, LEGACY) == ["a", "d"]
    assert lanes_mod.workflows_for_lane(manifest, NEW, enabled_only=False) == ["a", "b", "c"]
    # The committed pack workflows are restricted to the baking lane.
    for wf_id, entry in MANIFEST["workflows"].items():
        if entry.get("custom_nodes"):
            assert not lanes_mod.workflow_runs_on_lane(entry, LEGACY), wf_id


def test_image_matrix_skips_unbuilt_images_unless_asked_for():
    rows = lanes_mod.image_matrix(REGISTRY)
    assert [r["image"] for r in rows] == ["cu130-t2141"]
    assert rows[0] == {"image": "cu130-t2141", "python": "3.13", "cuda": "13.0",
                       "torch_index": "https://download.pytorch.org/whl/cu130",
                       "torch_spec": "torch==2.14.1 torchvision==0.29.1"}
    # Asking for the legacy image explicitly overrides its build: false.
    assert [r["image"] for r in lanes_mod.image_matrix(REGISTRY, "cu128-t2110")] == ["cu128-t2110"]
    assert lanes_mod.image_matrix(REGISTRY, "nothing") == []


def test_per_commit_matrix_only_lists_provisioned_lanes():
    rows = lanes_mod.per_commit_matrix(PENDING)
    ids = [r["lane"] for r in rows]
    assert "py312-torch2.11.0-cu128" in ids and NEW not in ids
    for row in rows:
        assert row["endpoint_id"]
        assert row["prefix"]
    provisioned = copy.deepcopy(PENDING)
    provisioned["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = "abc123"
    rows = lanes_mod.per_commit_matrix(provisioned)
    assert {"lane": "py313-torch2.14.1-cu130",
            "prefix": "regression/lanes/py313-torch2.14.1-cu130",
            "endpoint_id": "abc123"} in rows
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(provisioned, "py313-torch2.14.1-cu130")] == [
        "py313-torch2.14.1-cu130"]


def test_per_commit_matrix_lists_each_lanes_workflows_with_the_manifest():
    manifest = {"workflows": {"a": {"enabled": True}, "b": {"enabled": True, "lanes": [NEW]},
                              "c": {"enabled": False}}}
    rows = {r["lane"]: r for r in lanes_mod.per_commit_matrix(REGISTRY, manifest=manifest)}
    assert rows[LEGACY]["workflows"] == "a"
    assert rows[NEW]["workflows"] == "a,b"


def test_lanes_on_image_never_moves_the_legacy_lane():
    assert lanes_mod.lanes_on_image(PENDING, "cu128-t2110") == []
    assert lanes_mod.lanes_on_image(PENDING, "cu130-t2141") == []  # not provisioned yet
    provisioned = copy.deepcopy(PENDING)
    provisioned["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = "abc123"
    assert lanes_mod.lanes_on_image(provisioned, "cu130-t2141") == ["py313-torch2.14.1-cu130"]


def test_lane_endpoint_never_falls_back_for_an_unprovisioned_lane():
    env = {"RUNPOD_ENDPOINT_ID": "fromenv"}
    reg = {"lanes": {"named": {"endpoint_id": "ep1"}, "pending": {"endpoint_id": None},
                     "old": {"prefix": "regression"}}}
    assert index_builder.lane_endpoint(reg, "named", env) == "ep1"
    assert index_builder.lane_endpoint(reg, "pending", env) is None
    assert index_builder.lane_endpoint(reg, "old", env) == "fromenv"
    assert index_builder.lane_endpoint(reg, "old", {}) is None
    assert index_builder.lane_endpoint(reg, "unknown", env) == "fromenv"


def test_set_image_plans_one_patch_per_provisioned_lane_and_refuses_foreign_tags():
    reg = copy.deepcopy(PENDING)
    reg["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = "abc123"
    plan = runpod_endpoints.plan_set_image(reg, "cu130-t2141-deadbeef", image_id="cu130-t2141")
    assert plan == [("py313-torch2.14.1-cu130", "abc123",
                     {"image": "ghcr.io/comfy-org/comfyci-runpod-worker:cu130-t2141-deadbeef"})]
    plan = runpod_endpoints.plan_set_image(reg, "cu130-t2141", lane_id="py313-torch2.14.1-cu130")
    assert plan[0][2]["image"].endswith(":cu130-t2141")
    with pytest.raises(SystemExit):
        runpod_endpoints.plan_set_image(reg, "cu128-t2110-deadbeef", lane_id="py313-torch2.14.1-cu130")
    with pytest.raises(SystemExit):
        runpod_endpoints.plan_set_image(reg, "cu130-t2141-deadbeef", lane_id="nope")
    # The legacy lane is pinned by hand: an image-wide repoint leaves it alone.
    assert runpod_endpoints.plan_set_image(reg, "cu128-t2110-deadbeef", image_id="cu128-t2110") == []


def test_endpoint_body_comes_from_defaults_plus_the_lane():
    body = runpod_endpoints.endpoint_body(PENDING, "py313-torch2.14.1-cu130", "cu130-t2141-abc")
    assert body["name"] == "comfyci-cu130-t2141"
    assert body["type"] == "QUEUE"
    assert body["image"] == "ghcr.io/comfy-org/comfyci-runpod-worker:cu130-t2141-abc"
    assert body["gpu"] == {"pools": ["ADA_24"], "count": 1, "minCudaVersion": "13.0"}
    assert body["workers"] == {"min": 0, "max": 3, "idleTimeout": 20}
    assert body["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4}
    assert body["timeout"] == 3600000 and body["flashboot"] == "FLASHBOOT" and body["disk"] == 40
    assert body["networkVolumes"] == ["n0xnkwnogx"] and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["registry"] == "cmszcugyp001tfk7ooe7rtp4d"
    assert body["env"] == {"RUNPOD_INIT_TIMEOUT": "800"}
    # An already provisioned lane is never created twice, and a foreign tag is refused.
    with pytest.raises(SystemExit, match="already has endpoint"):
        runpod_endpoints.endpoint_body(REGISTRY, "py312-torch2.11.0-cu128", "cu128-t2110-abc")
    with pytest.raises(SystemExit, match="not a build of image"):
        runpod_endpoints.endpoint_body(PENDING, "py313-torch2.14.1-cu130", "cu128-t2110-abc")
    lane = copy.deepcopy(PENDING)
    lane["lanes"]["py313-torch2.14.1-cu130"]["workers_max"] = 1
    assert runpod_endpoints.endpoint_body(lane, "py313-torch2.14.1-cu130", "cu130-t2141")["workers"]["max"] == 1


def test_write_endpoint_id_updates_only_that_lane(tmp_path):
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps(PENDING, indent=2), encoding="utf-8")
    runpod_endpoints.write_endpoint_id("py313-torch2.14.1-cu130", "ep999", path)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] == "ep999"
    after["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = None
    assert after == PENDING


def test_set_image_dry_run_sends_nothing(capsys):
    reg = copy.deepcopy(PENDING)
    reg["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = "abc123"
    assert runpod_endpoints.set_image(reg, "cu130-t2141-1", None, "cu130-t2141", None, True) == 0
    assert "would PATCH abc123" in capsys.readouterr().out


def test_dockerfile_takes_every_build_arg_the_workflow_passes():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    declared = set(re.findall(r"^ARG (\w+)", dockerfile, re.M))
    workflow = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    passed = set(re.findall(r"^\s+(\w+)=\$\{\{ (?:matrix|github)\.\w+ \}\}$", workflow, re.M))
    assert passed, "build-args block not found"
    assert passed <= declared, passed - declared
    # Every image field the matrix carries reaches the Dockerfile.
    for field, arg in (("python", "PYTHON_VERSION"), ("cuda", "CUDA_VERSION"),
                       ("torch_index", "TORCH_INDEX"), ("torch_spec", "TORCH_SPEC")):
        assert f"{arg}=${{{{ matrix.{field} }}}}" in workflow


def test_dockerfile_bakes_the_pack_registry():
    """The image carries the registry and the bake step, and a registry change
    (a pin bump) triggers a rebuild."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY manifest/custom_nodes.json /comfy-config/custom_nodes.json" in dockerfile
    assert "bake --manifest /comfy-config/custom_nodes.json" in dockerfile
    assert "libgl1" in dockerfile and "SAM2_BUILD_CUDA=0" in dockerfile
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    assert "- manifest/custom_nodes.json" in build


def test_workflows_derive_their_matrices_from_the_registry():
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    regression = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert "lanes.py images" in build
    assert "runpod_endpoints.py set-image" in build
    assert "lanes.py per-commit" in regression
    assert '--lane "$LANE"' in regression
    assert 'FETCH_HEAD:$PREFIX/latest/master.json' in regression
