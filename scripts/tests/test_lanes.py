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


def test_the_committed_registry_is_consistent():
    assert lanes_mod.validate(REGISTRY) == []
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
    rows = lanes_mod.per_commit_matrix(REGISTRY)
    ids = [r["lane"] for r in rows]
    assert "py312-torch2.11.0-cu128" in ids
    for row in rows:
        assert row["endpoint_id"]
        assert row["prefix"]
    provisioned = copy.deepcopy(REGISTRY)
    provisioned["lanes"]["py313-torch2.14.1-cu130"]["endpoint_id"] = "abc123"
    rows = lanes_mod.per_commit_matrix(provisioned)
    assert {"lane": "py313-torch2.14.1-cu130",
            "prefix": "regression/lanes/py313-torch2.14.1-cu130",
            "endpoint_id": "abc123"} in rows
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(provisioned, "py313-torch2.14.1-cu130")] == [
        "py313-torch2.14.1-cu130"]


def test_lanes_on_image_never_moves_the_legacy_lane():
    assert lanes_mod.lanes_on_image(REGISTRY, "cu128-t2110") == []
    assert lanes_mod.lanes_on_image(REGISTRY, "cu130-t2141") == []  # not provisioned yet
    provisioned = copy.deepcopy(REGISTRY)
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
    reg = copy.deepcopy(REGISTRY)
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


def test_set_image_dry_run_sends_nothing(capsys):
    reg = copy.deepcopy(REGISTRY)
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


def test_workflows_derive_their_matrices_from_the_registry():
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    regression = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert "lanes.py images" in build
    assert "runpod_endpoints.py set-image" in build
    assert "lanes.py per-commit" in regression
    assert '--lane "$LANE"' in regression
    assert 'FETCH_HEAD:$PREFIX/latest/master.json' in regression
