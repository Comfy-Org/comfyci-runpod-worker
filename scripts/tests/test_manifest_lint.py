"""Static checks on the real manifest and every workflow it references.

CPU-only: catches the mistakes that otherwise surface as a missing_model or
validation_error on a GPU worker (a model file the workflow loads but the
manifest never syncs, a seed override pointing at a renamed node, a directory
ComfyUI does not search).
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
WORKFLOWS = MANIFEST["workflows"]

ENTRY_KEYS = {"enabled", "workflow_path", "seed_overrides", "models",
              "gpu_type", "timeout_s", "thresholds", "golden_tag"}
OPTIONAL_ENTRY_KEYS = {"comfy_flags"}
# Inputs whose string value names a file the worker must find on the volume.
FILE_INPUT = re.compile(r"^(ckpt|unet|clip|vae|lora)_name\d*$")
# Any other loader (control_net_name, upscale model_name, ...) is caught by extension.
FILE_EXT = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$", re.I)
# Loader filenames that ship with ComfyUI itself rather than the volume.
BUILTIN_FILES: set[str] = set()
# LoadImage files come from the volume's input/ folder via --input-directory.
INPUT_DIR = "/runpod-volume/models/input"
ENDPOINT_EXEC_TIMEOUT_S = 3600


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
                    or (node["class_type"] == "LoadImage" and name == "image")):
                out.add(re.sub(r" \[(input|output|temp)\]$", "", value))
    return out


def _links(node: dict) -> list:
    return [v for v in node["inputs"].values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]


def _save_ancestors(wf: dict) -> set[str]:
    """Node ids that feed a SaveImage (ComfyUI runs output nodes and their inputs only)."""
    seen, stack = set(), [nid for nid, n in wf.items() if n["class_type"] == "SaveImage"]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        stack.extend(link[0] for link in _links(wf[nid]))
    return seen


def test_model_folders_parsed():
    assert {"checkpoints", "diffusion_models", "text_encoders", "vae", "loras"} <= _model_folders()
    assert len(_model_folders()) == 12


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_entry_shape(wf_id):
    entry = WORKFLOWS[wf_id]
    assert ENTRY_KEYS <= set(entry), f"missing keys: {ENTRY_KEYS - set(entry)}"
    assert set(entry) <= ENTRY_KEYS | OPTIONAL_ENTRY_KEYS, \
        f"unknown keys: {set(entry) - ENTRY_KEYS - OPTIONAL_ENTRY_KEYS}"
    assert isinstance(entry["enabled"], bool)
    assert isinstance(entry.get("comfy_flags", []), list)
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
def test_every_node_feeds_a_save(wf_id):
    """Leftover preview/aux output nodes still run (a PreviewAny on an LLM branch
    would load and sample the LLM), and dead nodes hide what the test exercises."""
    wf = _load_workflow(WORKFLOWS[wf_id])
    stray = sorted(f"{nid}:{wf[nid]['class_type']}" for nid in set(wf) - _save_ancestors(wf))
    assert not stray, f"nodes that do not feed a SaveImage: {stray}"


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_saves_png_under_its_own_prefix(wf_id):
    wf = _load_workflow(WORKFLOWS[wf_id])
    saves = [n for n in wf.values() if n["class_type"] == "SaveImage"]
    assert saves, "only SaveImage PNG outputs are compared"
    assert all(n["inputs"].get("filename_prefix") == wf_id for n in saves)


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_seed_overrides_hit_existing_inputs(wf_id):
    entry = WORKFLOWS[wf_id]
    wf = _load_workflow(entry)
    assert entry["seed_overrides"], "every workflow pins its seed"
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
        if "sha256" in m:
            assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), m["name"]
    if any(m["directory"] == "input" for m in entry["models"]):
        flags = entry.get("comfy_flags") or []
        assert "--input-directory" in flags, "input/ files need --input-directory"
        assert flags[flags.index("--input-directory") + 1] == INPUT_DIR


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


def test_referenced_files_helper():
    wf = {
        "1": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "a.safetensors",
                                                          "clip_name2": "b.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"sampler_name": "euler"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "x.png [input]"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": ["9", 0]}},
        "5": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "cn.safetensors"}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["4", 0]}},
    }
    assert _referenced_files(wf) == {"a.safetensors", "b.safetensors", "x.png", "cn.safetensors"}


def test_save_ancestors_helper():
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["1", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
    }
    assert _save_ancestors(wf) == {"1", "3"}
