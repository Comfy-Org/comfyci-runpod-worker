"""Static checks on the real manifest and every workflow it references.

CPU-only: catches the mistakes that otherwise surface as a missing_model or
validation_error on a GPU worker (a model file the workflow loads but the
manifest never syncs, a seed override pointing at a renamed node, a directory
ComfyUI does not search, a custom-node pack nobody declared, a pack workflow
aimed at a lane whose image bakes no packs).
"""
import json
import re
from pathlib import Path

import pytest

import lanes as lanes_mod
import packs

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
WORKFLOWS = MANIFEST["workflows"]
LANES = lanes_mod.load_lanes()
PACKS = packs.load_registry()

ENTRY_KEYS = {"enabled", "workflow_path", "seed_overrides", "models",
              "gpu_type", "timeout_s", "thresholds", "golden_tag"}
OPTIONAL_ENTRY_KEYS = {"comfy_flags", "custom_nodes", "lanes"}
# Inputs whose string value names a file the worker must find on the volume.
FILE_INPUT = re.compile(r"^(ckpt|unet|clip|vae|lora)_name\d*$")
# Any other loader (control_net_name, upscale model_name, ...) is caught by extension.
FILE_EXT = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$", re.I)
# Nodes whose named input is a file from the volume's input/ folder (--input-directory).
INPUT_FILE_BY_NODE = {"LoadImage": "image", "VHS_LoadVideo": "video"}
# Loader values ComfyUI offers without a file behind them, so the volume has
# nothing to sync: nodes.VAELoader appends "pixel_space" to its combo for
# pixel-space models (PixelDiT), which have no VAE weights at all.
BUILTIN_FILES = {"pixel_space"}
# Output nodes a graph may end in: only SaveImage PNGs are compared, but a
# VHS_VideoCombine (gif encode path) is a legitimate second sink.
OUTPUT_NODES = {"SaveImage", "VHS_VideoCombine"}
INPUT_DIR = "/runpod-volume/models/input"
ENDPOINT_EXEC_TIMEOUT_S = 3600
# What each pack workflow must exercise (so a prompt edit cannot quietly drop
# the pack nodes); the Manager smoke has none by design.
PACK_NODES = {
    "kjnodes_sdxl_t2i": {"ImageResizeKJv2", "ColorMatchV2", "ImageConcanate"},
    "vhs_video_roundtrip": {"VHS_LoadVideo", "VHS_VideoCombine"},
    "impact_sdxl_detailer": {"UltralyticsDetectorProvider", "BboxDetectorSEGS", "DetailerForEach"},
    "manager_import_smoke": set(),
}


def _model_folders() -> set[str]:
    """Folder keys from docker/extra_model_paths.yaml (flat `key: value` lines)."""
    text = (ROOT / "docker" / "extra_model_paths.yaml").read_text(encoding="utf-8")
    keys = set(re.findall(r"^\s+([a-z_]+):\s*\S+", text, flags=re.M))
    keys.discard("base_path")
    return keys


def _load_workflow(entry: dict) -> dict:
    return json.loads((ROOT / entry["workflow_path"]).read_text(encoding="utf-8"))


def _referenced_files(wf: dict) -> set[str]:
    out = set()
    for node in wf.values():
        for name, value in node["inputs"].items():
            if not isinstance(value, str):
                continue
            if (FILE_INPUT.match(name) or FILE_EXT.search(value)
                    or INPUT_FILE_BY_NODE.get(node["class_type"]) == name):
                out.add(re.sub(r" \[(input|output|temp)\]$", "", value))
    return out


def _links(node: dict) -> list:
    return [v for v in node["inputs"].values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]


def _output_ancestors(wf: dict) -> set[str]:
    """Node ids that feed an output node (ComfyUI runs output nodes and their inputs only)."""
    seen, stack = set(), [nid for nid, n in wf.items() if n["class_type"] in OUTPUT_NODES]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        stack.extend(link[0] for link in _links(wf[nid]))
    return seen


def _has_seed(wf: dict) -> bool:
    return any("seed" in n["inputs"] or "noise_seed" in n["inputs"] for n in wf.values())


def test_model_folders_parsed():
    assert {"checkpoints", "diffusion_models", "text_encoders", "vae", "loras",
            "ultralytics", "ultralytics_bbox"} <= _model_folders()
    assert len(_model_folders()) == 15


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_entry_shape(wf_id):
    entry = WORKFLOWS[wf_id]
    assert ENTRY_KEYS <= set(entry), f"missing keys: {ENTRY_KEYS - set(entry)}"
    assert set(entry) <= ENTRY_KEYS | OPTIONAL_ENTRY_KEYS, \
        f"unknown keys: {set(entry) - ENTRY_KEYS - OPTIONAL_ENTRY_KEYS}"
    assert isinstance(entry["enabled"], bool)
    assert isinstance(entry.get("comfy_flags", []), list)
    assert isinstance(entry.get("custom_nodes", []), list)
    assert entry.get("lanes") is None or (isinstance(entry["lanes"], list) and entry["lanes"])
    timeout = entry["timeout_s"] or MANIFEST["defaults"]["timeout_s"]
    assert 0 < timeout < ENDPOINT_EXEC_TIMEOUT_S, "leave room for checkout and server start"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_workflow_is_api_format(wf_id):
    wf = _load_workflow(WORKFLOWS[wf_id])
    assert isinstance(wf, dict) and wf, "API format is a non-empty {node_id: node} object"
    for nid, node in wf.items():
        assert isinstance(node, dict) and isinstance(node.get("class_type"), str), nid
        assert isinstance(node.get("inputs"), dict), nid
        for link in _links(node):
            assert link[0] in wf, f"{nid} links to missing node {link[0]}"
            assert isinstance(link[1], int) and link[1] >= 0, f"{nid} link {link} has a bad output index"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_every_node_feeds_an_output(wf_id):
    """Leftover preview/aux output nodes still run (a PreviewAny on an LLM branch
    would load and sample the LLM), and dead nodes hide what the test exercises."""
    wf = _load_workflow(WORKFLOWS[wf_id])
    stray = sorted(f"{nid}:{wf[nid]['class_type']}" for nid in set(wf) - _output_ancestors(wf))
    assert not stray, f"nodes that do not feed an output node: {stray}"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_saves_png_under_its_own_prefix(wf_id):
    wf = _load_workflow(WORKFLOWS[wf_id])
    saves = [n for n in wf.values() if n["class_type"] == "SaveImage"]
    assert saves, "only SaveImage PNG outputs are compared"
    assert all(n["inputs"].get("filename_prefix") == wf_id for n in saves)
    # A second sink must not collide with the compared PNGs' file names.
    others = [n for n in wf.values() if n["class_type"] in OUTPUT_NODES - {"SaveImage"}]
    assert all(n["inputs"].get("filename_prefix") not in (wf_id, None) for n in others)


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_seed_overrides_hit_existing_inputs(wf_id):
    entry = WORKFLOWS[wf_id]
    wf = _load_workflow(entry)
    if _has_seed(wf):
        assert entry["seed_overrides"], "every workflow that samples pins its seed"
    else:
        assert entry["seed_overrides"] == {}, "no seed input to override"
    for nid, patch in entry["seed_overrides"].items():
        assert nid in wf, f"seed override targets missing node {nid}"
        for name in patch:
            assert name in wf[nid]["inputs"], f"node {nid} has no input {name}"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_every_loaded_file_is_synced(wf_id):
    entry = WORKFLOWS[wf_id]
    names = {m["name"] for m in entry["models"]}
    missing = _referenced_files(_load_workflow(entry)) - names - BUILTIN_FILES
    assert not missing, f"loaded by the workflow but not in its models list: {sorted(missing)}"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_every_synced_file_is_loaded(wf_id):
    """A models entry the workflow never loads is a dead download on the volume
    (e.g. a dropped switch branch's model left behind)."""
    entry = WORKFLOWS[wf_id]
    unused = {m["name"] for m in entry["models"]} - _referenced_files(_load_workflow(entry))
    assert not unused, f"in the models list but never loaded: {sorted(unused)}"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_model_entries(wf_id):
    entry = WORKFLOWS[wf_id]
    folders = _model_folders() | {"input"}
    for m in entry["models"]:
        assert {"name", "url", "directory"} <= set(m), m
        assert m["url"].startswith("https://"), m["url"]
        assert m["directory"] in folders, f"{m['name']}: ComfyUI does not search {m['directory']}/"
        assert not m["name"].startswith("/") and ".." not in m["name"].split("/"), m["name"]
        if "sha256" in m:
            assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), m["name"]
    if any(m["directory"] == "input" for m in entry["models"]):
        flags = entry.get("comfy_flags") or []
        assert "--input-directory" in flags, "input/ files need --input-directory"
        assert flags[flags.index("--input-directory") + 1] == INPUT_DIR


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_custom_nodes_name_declared_packs(wf_id):
    entry = WORKFLOWS[wf_id]
    for name in entry.get("custom_nodes") or []:
        assert name in PACKS["packs"], f"{name} is not declared in manifest/custom_nodes.json"
    # The records the job ships resolve without error.
    assert [r["name"] for r in packs.resolve(entry, PACKS)] == (entry.get("custom_nodes") or [])


@pytest.mark.parametrize("wf_id", sorted(PACK_NODES))
def test_pack_workflows_exercise_their_pack_nodes(wf_id):
    entry = WORKFLOWS[wf_id]
    classes = {n["class_type"] for n in _load_workflow(entry).values()}
    assert PACK_NODES[wf_id] <= classes, f"missing pack nodes: {PACK_NODES[wf_id] - classes}"
    assert entry.get("custom_nodes"), "a pack workflow names its packs"
    if not PACK_NODES[wf_id]:
        # An import smoke: core nodes only, so the only thing that can fail is the import.
        assert classes <= {"EmptyImage", "SaveImage"}


def test_lane_filters_and_the_pack_lane_rule():
    """Every `lanes` filter names declared lanes; a workflow with packs is
    restricted to lanes whose image CI builds (the legacy image bakes none)."""
    assert lanes_mod.validate(LANES, MANIFEST, PACKS) == []
    for wf_id, entry in WORKFLOWS.items():
        if entry.get("custom_nodes"):
            assert entry.get("lanes"), f"{wf_id} needs packs, so it must name its lanes"
            for lane_id in entry["lanes"]:
                image = LANES["images"][LANES["lanes"][lane_id]["image"]]
                assert image.get("build") is not False, f"{wf_id}: {lane_id}'s image is never rebuilt"


def test_pack_registry_is_consistent():
    assert packs.validate(PACKS) == []
    pins = {}
    for line in PACKS["constraints"]:
        name, version = line.split("==")
        pins[name] = version
    # Manager's startup fixer reinstalls whichever opencv variant lags the other.
    assert pins["opencv-python"] == pins["opencv-python-headless"]


def test_shared_model_keys_agree():
    """sync_models dedups on directory/name, so one key must mean one file."""
    seen: dict[str, tuple[str, dict]] = {}
    for wf_id, entry in WORKFLOWS.items():
        for m in entry["models"]:
            key = f"{m['directory']}/{m['name']}"
            if key in seen:
                other_id, other = seen[key]
                assert (m["url"], m.get("sha256")) == (other["url"], other.get("sha256")), \
                    f"{key} differs between {other_id} and {wf_id}"
            else:
                seen[key] = (wf_id, m)


def test_builtin_loader_values_need_no_model_entry():
    wf = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "pixel_space"}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["1", 0]}},
    }
    assert _referenced_files(wf) - BUILTIN_FILES == set()


def test_referenced_files_helper():
    wf = {
        "1": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "a.safetensors",
                                                          "clip_name2": "b.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"sampler_name": "euler"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "x.png [input]"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": ["9", 0]}},
        "5": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "cn.safetensors"}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["4", 0]}},
        "7": {"class_type": "VHS_LoadVideo", "inputs": {"video": "clip.mp4", "force_rate": 0}},
        "8": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face.pt"}},
    }
    assert _referenced_files(wf) == {"a.safetensors", "b.safetensors", "x.png", "cn.safetensors",
                                     "clip.mp4", "bbox/face.pt"}


def test_output_ancestors_helper():
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["1", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
        "5": {"class_type": "ImageScale", "inputs": {"image": ["1", 0]}},
        "6": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["5", 0], "filename_prefix": "y"}},
    }
    assert _output_ancestors(wf) == {"1", "3", "5", "6"}
