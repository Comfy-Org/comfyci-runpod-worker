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
import output_kinds
import packs
import smoke

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
WORKFLOWS = MANIFEST["workflows"]
LANES = lanes_mod.load_lanes()
PACKS = packs.load_registry()

ENTRY_KEYS = {"enabled", "workflow_path", "seed_overrides", "models",
              "gpu_type", "timeout_s", "thresholds", "golden_tag"}
OPTIONAL_ENTRY_KEYS = {"comfy_flags", "custom_nodes", "lanes", "output_kind", "level", "cadence",
                       "expect"}
# The output kinds run_regression.py has a comparator for. A reference-level
# entry needs one of them: an audio workflow runs at smoke level only.
COMPARED_KINDS = {"image", "text"}
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
# Output nodes a graph may end in. The main sinks of the entry's `output_kind`
# write its main output under the workflow id: the SaveImage PNGs of an image
# workflow, the SaveText .txt of a text workflow, the FLAC of an audio
# workflow. Any other, like the SaveGLB of a 3D workflow, the SaveVideo of a
# video workflow or a VHS_VideoCombine (gif encode path), is a second sink
# with a prefix of its own. A 3D or a video workflow is an image workflow: its
# main output is a picture.
OUTPUT_NODES = {"SaveImage", "SaveText", "SaveAudio", "SaveAudioAdvanced", "SaveGLB", "SaveVideo",
                "VHS_VideoCombine"}
MAIN_SINKS = {"image": {"SaveImage"}, "text": {"SaveText"},
              "audio": {"SaveAudio", "SaveAudioAdvanced"}}
# Save nodes that write a file no smoke check can read.
UNCHECKED_SINKS = {"SaveAudioMP3", "SaveAudioOpus"}
INPUT_DIR = "/runpod-volume/models/input"
ENDPOINT_EXEC_TIMEOUT_S = 3600
# The network volume holds every model file of the manifest, and nothing ever
# deletes from it. The files may fill this share of `endpoint_defaults.volume_gb`
# (a GB counted as 10^9 bytes, the smaller reading).
VOLUME_FILL_PERCENT = 90
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


def _level_problems(entry: dict) -> list[str]:
    """What is wrong with how an entry asks to be checked: its `level`, its
    `cadence`, its `expect` and, for its level, its `output_kind`."""
    try:
        level = lanes_mod.level_of(entry)
        lanes_mod.cadence_of(entry)
        lanes_mod.expect_of(entry)  # smoke level only, known keys, numbers above 0
        kind = output_kinds.kind_of(entry)
    except ValueError as e:
        return [str(e)]
    problems = []
    if level == "reference" and kind not in COMPARED_KINDS:
        problems.append(f"no comparator for output_kind {kind!r} yet: it needs level smoke")
    if level == "smoke" and (entry.get("thresholds") is not None
                             or entry.get("golden_tag") is not None):
        problems.append("a smoke entry has no reference: thresholds and golden_tag are null")
    return problems


def _writes_mp4_h264(inputs: dict) -> bool:
    """Whether a SaveVideo node writes an MP4 with H.264. Its `format` is mp4
    or auto (auto is MP4 unless the codec asks for another container), and
    its codec, where it names one (`format.codec`, or the node's own `codec`),
    is h264 or auto. SaveVideo also writes mkv, webm and mov, and AV1 and
    ProRes: the smoke check looks for an .mp4, and the limit of its blank
    check was measured on H.264 (a moving AV1 clip measures under it)."""
    codecs = [inputs[key] for key in ("format.codec", "codec") if key in inputs]
    return (inputs.get("format", "auto") in ("mp4", "auto")
            and all(codec in ("h264", "auto") for codec in codecs))


def _sink_problems(wf_id: str, entry: dict, wf: dict) -> list[str]:
    """What is wrong with a workflow's output nodes, for its output kind."""
    kind = output_kinds.kind_of(entry)
    problems = []
    sinks = [n for n in wf.values() if n["class_type"] in OUTPUT_NODES]
    main = [n for n in sinks if n["class_type"] in MAIN_SINKS[kind]]
    named = [n for n in main if n["inputs"].get("filename_prefix") == wf_id]
    if kind == "text":
        # One SaveText named after the workflow writes the main .txt. Another
        # one (the model's reasoning, say) is a second sink.
        if len(named) != 1:
            problems.append("exactly one SaveText is prefixed with the workflow id")
        main = named
    elif kind == "audio":
        if not main:
            problems.append("an audio workflow saves its sound with SaveAudioAdvanced")
        if len(named) != len(main):
            problems.append("every SaveAudioAdvanced is prefixed with the workflow id")
    else:
        if not main:
            problems.append("only SaveImage PNG outputs are compared")
        if len(named) != len(main):
            problems.append("every SaveImage is prefixed with the workflow id")
    # A save node writes the format its check reads, whatever the workflow's kind.
    if any(n["class_type"] == "SaveText" and n["inputs"].get("format") != "txt" for n in sinks):
        problems.append("SaveText must write format txt: only .txt files are compared")
    if any(n["class_type"] == "SaveAudioAdvanced" and n["inputs"].get("format") != "flac"
           for n in sinks):
        problems.append("SaveAudioAdvanced must write format flac: only FLAC files are checked")
    if any(n["class_type"] == "SaveVideo" and not _writes_mp4_h264(n["inputs"]) for n in sinks):
        problems.append("SaveVideo must write mp4 with h264: only that is checked")
    for name in sorted({n["class_type"] for n in wf.values()} & UNCHECKED_SINKS):
        problems.append(f"{name} is not allowed: its file cannot be checked. Use "
                        "SaveAudioAdvanced with format flac")
    for n in sinks:
        if any(n is m for m in main):
            continue
        # A second sink must not collide with the file names of the main
        # output. A prefix with a folder publishes a nested file, which the
        # pages do not link.
        prefix = n["inputs"].get("filename_prefix")
        if not isinstance(prefix, str) or "/" in prefix or "\\" in prefix:
            problems.append(f"{n['class_type']} needs a flat filename_prefix, not {prefix!r}")
        elif not prefix.startswith(wf_id + "_"):
            problems.append(f"a second sink is prefixed '{wf_id}_...', not {prefix!r}")
    return problems


def _text_generation_problems(wf: dict) -> list[str]:
    """Settings that make generated text differ from run to run. A text
    workflow passes on an exact match, so it decodes greedily and without the
    multi-token-prediction head, whose `auto` mode adapts to measured timings."""
    problems = []
    for nid, n in wf.items():
        if not n["class_type"].startswith("TextGenerate"):
            continue
        if n["inputs"].get("sampling_mode") != "off":
            problems.append(f"{nid}: sampling_mode must be 'off' (greedy decoding)")
        if n["inputs"].get("mtp") in (None, "auto"):
            problems.append(f"{nid}: mtp must be set, and not to 'auto'")
    return problems


def _literal_dynamic_inputs(wf: dict) -> list[str]:
    """Literal `name.sub` inputs (the sub-inputs of a dynamic combo). The
    worker's validate_workflow knows only the top-level input names of
    /object_info and strips these before the prompt is queued."""
    return sorted(f"{nid}:{n['class_type']}.{name}" for nid, n in wf.items()
                  for name, value in n["inputs"].items()
                  if "." in name and not isinstance(value, list))


def _volume_files(manifest: dict, lanes: dict) -> dict[str, int]:
    """directory/name -> bytes of every file the volume has to hold: the models
    and input files of every workflow, enabled or not (the sync fetches both),
    that runs on a lane whose models are on the volume. Each file counts once.
    A workflow runs on the lanes of its own cadence, so one that only the
    weekly lane runs, whose models each job fetches, takes no room here. Nor
    does one that runs only on lanes without the volume (an empty
    `network_volumes`). Which workflows those are is `lanes.volume_workflows`,
    the rule sync_models.py downloads by: the lint counts what the sync sends."""
    files: dict[str, int] = {}
    for wf_id in lanes_mod.volume_workflows(manifest, lanes):
        for m in manifest["workflows"][wf_id]["models"]:
            files[f"{m['directory']}/{m['name']}"] = m["bytes"]
    return files


def _volume_problems(manifest: dict, lanes: dict) -> list[str]:
    """Why the model files would not fit the network volume, if they would not."""
    size_gb = (lanes.get("endpoint_defaults") or {}).get("volume_gb")
    if isinstance(size_gb, bool) or not isinstance(size_gb, int) or size_gb <= 0:
        return ["endpoint_defaults.volume_gb in manifest/lanes.json is the size of the network "
                f"volume in GB, a whole number above 0, not {size_gb!r}"]
    files = _volume_files(manifest, lanes)
    used, room = sum(files.values()), size_gb * 10**9 * VOLUME_FILL_PERCENT // 100
    if used <= room:
        return []
    largest = max(files, key=files.get)
    return [f"the model files need {used / 1e9:.1f} GB, and the {size_gb} GB volume may hold "
            f"{room / 1e9:.1f} GB ({VOLUME_FILL_PERCENT} %): {len(files)} files, the largest is "
            f"{largest} with {files[largest] / 1e9:.1f} GB"]


def _off_volume_problems(manifest: dict, lanes: dict) -> list[str]:
    """The enabled workflows that would end as `missing_model` in every run:
    no lane that reads its models from the volume runs them, so the model sync
    leaves their models out (`lanes.volume_workflows`), and they load a file
    that is not among the volume's. A workflow whose files another workflow
    puts on the volume is fine. One that no lane runs has no job at all, and
    `lanes.validate` names it.

    This rule goes when jobs can fetch models: `"model_storage": "job"` then
    is a way to run a workflow, and no longer a way to lose its models."""
    synced = set(lanes_mod.volume_workflows(manifest, lanes))
    files = _volume_files(manifest, lanes)
    problems = []
    for wf_id, entry in manifest["workflows"].items():
        if not entry.get("enabled") or wf_id in synced:
            continue
        if not lanes_mod.lanes_for_workflow(entry, lanes):
            continue
        missing = sorted({f"{m['directory']}/{m['name']}" for m in entry["models"]} - set(files))
        if missing:
            problems.append(f"{wf_id}: its models are on no volume: nothing fetches models for a "
                            f"job yet ({', '.join(missing)})")
    return problems


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
    assert _level_problems(entry) == []
    if output_kinds.kind_of(entry) == "text":  # kind_of refuses an unknown kind
        limits = entry["thresholds"] or {}
        assert set(limits) <= {"min_text_similarity"}, "a text workflow has one limit"
        assert all(isinstance(v, (int, float)) and 0 < v <= 1 for v in limits.values())


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
def test_saves_its_compared_output_under_its_own_prefix(wf_id):
    entry = WORKFLOWS[wf_id]
    assert _sink_problems(wf_id, entry, _load_workflow(entry)) == []


@pytest.mark.parametrize("wf_id", sorted(WORKFLOWS))
def test_no_input_the_worker_would_strip(wf_id):
    assert _literal_dynamic_inputs(_load_workflow(WORKFLOWS[wf_id])) == []


# The key itself, not kind_of(): an unknown kind is a failure of test_entry_shape
# for that entry, not an error that stops every test of this file from running.
@pytest.mark.parametrize("wf_id", sorted(w for w, e in WORKFLOWS.items()
                                          if e.get("output_kind") == "text"))
def test_text_workflows_generate_the_same_text_every_run(wf_id):
    assert _text_generation_problems(_load_workflow(WORKFLOWS[wf_id])) == []


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
        # The file's size, so the lint can add up what the volume has to hold.
        size = m.get("bytes")
        assert isinstance(size, int) and not isinstance(size, bool) and size > 0, \
            (f"{m['name']}: bytes must be the size of the download, not {size!r} "
             "(python scripts/import_template.py --fill-bytes adds a missing one)")
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
                assert m.get("bytes") == other.get("bytes"), \
                    f"{key} has another size in {other_id} than in {wf_id}"
            else:
                seen[key] = (wf_id, m)


def test_the_volume_has_room_for_every_model_file():
    """One edit can put a workflow on a lane that keeps its models on the
    volume. A full volume fails the sync, and nothing deletes from it."""
    assert _volume_problems(MANIFEST, LANES) == []
    files = _volume_files(MANIFEST, LANES)
    listed = {f"{m['directory']}/{m['name']}": m["bytes"]
              for entry in WORKFLOWS.values() for m in entry["models"]}
    assert files and files.items() <= listed.items()


def test_the_volume_rule_counts_each_file_once_and_stops_at_90_percent():
    def model(name, size):
        return {"name": name, "url": "https://x/" + name, "directory": "vae", "bytes": size}

    lanes = {"endpoint_defaults": {"volume_gb": 10}, "lanes": {
        "volume": {"cadence": "per-commit"}, "old": {"cadence": "retired"},
        "night": {"cadence": "nightly"},
        "fetch": {"cadence": "weekly", "model_storage": "job"}}}
    manifest = {"workflows": {
        "a": {"enabled": True, "models": [model("shared", 4 * 10**9), model("a", 2 * 10**9)]},
        "b": {"enabled": False, "models": [model("shared", 4 * 10**9), model("b", 2 * 10**9)]},
        # A nightly workflow: the nightly lane keeps its models on the volume too.
        "n": {"enabled": True, "cadence": "nightly", "models": [model("n", 10**9)]},
        # Fetched by each job, or only on a lane that runs nothing: not on the volume.
        "c": {"enabled": True, "cadence": "weekly", "lanes": ["fetch"],
              "models": [model("c", 50 * 10**9)]},
        "d": {"enabled": True, "lanes": ["old"], "models": [model("d", 50 * 10**9)]},
        # A weekly workflow names no lane: its cadence alone keeps it off the volume.
        "w": {"enabled": True, "cadence": "weekly", "models": [model("w", 50 * 10**9)]}}}
    assert _volume_files(manifest, lanes) == {"vae/shared": 4 * 10**9, "vae/a": 2 * 10**9,
                                              "vae/b": 2 * 10**9, "vae/n": 10**9}
    assert _volume_problems(manifest, lanes) == []  # 9.0 of 10 GB: at the limit
    # The weekly lane with its models on the volume: its workflows count.
    on_volume = {**lanes, "lanes": {**lanes["lanes"], "fetch": {"cadence": "weekly"}}}
    assert _volume_files(manifest, on_volume)["vae/w"] == 50 * 10**9
    manifest["workflows"]["b"]["models"][1]["bytes"] += 1
    assert _volume_problems(manifest, lanes) == [
        "the model files need 9.0 GB, and the 10 GB volume may hold 9.0 GB (90 %): 4 files, "
        "the largest is vae/shared with 4.0 GB"]
    for size in (None, 0, -5, "200", 200.0, True):
        said = _volume_problems(manifest, {**lanes, "endpoint_defaults": {"volume_gb": size}})
        assert len(said) == 1 and "volume_gb" in said[0], size
    assert "volume_gb" in _volume_problems(manifest, {"lanes": {}})[0]


def test_a_workflow_only_on_lanes_without_the_volume_does_not_count_against_it():
    """A lane on another card, in other data centers, mounts no volume
    (`"network_volumes": []`), so its workflows take no room on one."""
    def model(name, size):
        return {"name": name, "url": "https://x/" + name, "directory": "vae", "bytes": size}

    def lanes(**big):
        return {"endpoint_defaults": {"volume_gb": 10, "network_volumes": ["vol"]}, "lanes": {
            "volume": {"cadence": "per-commit"}, "big": {"cadence": "weekly", **big}}}

    manifest = {"workflows": {
        "a": {"enabled": True, "models": [model("a", 2 * 10**9)]},
        "huge": {"enabled": True, "cadence": "weekly", "models": [model("huge", 60 * 10**9)]}}}
    apart = lanes(network_volumes=[])
    assert _volume_files(manifest, apart) == {"vae/a": 2 * 10**9}
    assert _volume_problems(manifest, apart) == []
    # The same lane with the default volume, or with a volume it names: the
    # file counts. The lint knows one size, whatever the volume.
    for own in ({}, {"network_volumes": None}, {"network_volumes": ["vol2"]}):
        files = _volume_files(manifest, lanes(**own))
        assert files == {"vae/a": 2 * 10**9, "vae/huge": 60 * 10**9}, own
        assert len(_volume_problems(manifest, lanes(**own))) == 1, own
    # A workflow that a lane with the volume runs too still counts.
    two = lanes(network_volumes=[])
    two["lanes"]["week"] = {"cadence": "weekly"}
    assert "vae/huge" in _volume_files(manifest, two)
    manifest["workflows"]["huge"]["lanes"] = ["big"]
    assert _volume_files(manifest, two) == {"vae/a": 2 * 10**9}
    # Defaults without a volume: only a lane that names one counts.
    none = lanes(network_volumes=["vol"])
    none["endpoint_defaults"]["network_volumes"] = []
    assert _volume_files(manifest, none) == {"vae/huge": 60 * 10**9}


def test_every_enabled_workflow_finds_its_models_on_the_volume():
    assert _off_volume_problems(MANIFEST, LANES) == []


def test_a_workflow_whose_models_reach_no_volume_is_a_problem():
    """`validate` and the size rule accepted an enabled weekly workflow with
    models of its own while the weekly lane says `"model_storage": "job"`.
    The model sync left its models out, and nothing fetches models for a job
    yet: it would have ended as missing_model in every weekly run, with one
    line in the log of the sync job as the only word about it before."""
    def model(name, size=10**9):
        return {"name": name, "url": "https://x/" + name, "directory": "vae", "bytes": size}

    lanes = {"endpoint_defaults": {"volume_gb": 200, "network_volumes": ["vol"]}, "lanes": {
        "main": {"cadence": "per-commit"}, "old": {"cadence": "retired"},
        "week": {"cadence": "weekly", "model_storage": "job"}}}
    manifest = {"workflows": {
        "a": {"enabled": True, "models": [model("shared"), model("a")]},
        "big": {"enabled": True, "cadence": "weekly",
                "models": [model("shared"), model("big", 100 * 10**9), model("big2")]}}}
    assert "vae/big" not in _volume_files(manifest, lanes)
    assert _volume_problems(manifest, lanes) == []
    assert _off_volume_problems(manifest, lanes) == [
        "big: its models are on no volume: nothing fetches models for a job yet "
        "(vae/big, vae/big2)"]

    def problems(**entry):
        changed = {"workflows": {**manifest["workflows"],
                                 "big": {**manifest["workflows"]["big"], **entry}}}
        return _off_volume_problems(changed, lanes)

    # Every file it loads is on the volume for another workflow, or it loads none.
    assert problems(models=[model("shared"), model("a")]) == []
    assert problems(models=[]) == []
    # Switched off, it has no job. So has one that only a retired lane runs,
    # which `validate` names.
    assert problems(enabled=False) == []
    assert problems(cadence="per-commit", lanes=["old"]) == []
    # On a lane that reads its models from the volume, the sync sends them.
    assert problems(cadence="per-commit") == []
    volume = {**lanes, "lanes": {**lanes["lanes"], "week": {"cadence": "weekly"}}}
    assert _off_volume_problems(manifest, volume) == []
    # A lane without any volume is no better than one that leaves it unread.
    apart = {**lanes, "lanes": {**lanes["lanes"], "week": {
        "cadence": "weekly", "model_storage": "job", "network_volumes": []}}}
    assert len(_off_volume_problems(manifest, apart)) == 1


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


def _sink(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def test_sink_rules_by_output_kind():
    image = {"1": _sink("SaveImage", filename_prefix="wf", images=["9", 0])}
    second = "a second sink is prefixed 'wf_...', not 'wf'"
    assert _sink_problems("wf", {}, image) == []
    assert _sink_problems("wf", {}, {}) == ["only SaveImage PNG outputs are compared"]
    # A text sink in an image workflow is a second sink: it needs its own prefix.
    both = {**image, "2": _sink("SaveText", filename_prefix="wf", format="txt", text=["9", 0])}
    assert _sink_problems("wf", {}, both) == [second]
    both["2"] = _sink("SaveText", filename_prefix="wf_caption", format="txt", text=["9", 0])
    assert _sink_problems("wf", {}, both) == []

    text = {"1": _sink("SaveText", filename_prefix="wf", format="txt", text=["9", 0])}
    kind = {"output_kind": "text"}
    one_save, txt_only = ("exactly one SaveText is prefixed with the workflow id",
                          "SaveText must write format txt: only .txt files are compared")
    assert _sink_problems("wf", kind, text) == []
    # A picture in a text workflow is a second sink too.
    assert _sink_problems("wf", kind, image) == [one_save, second]
    # The format is written out: a default the worker fills in is not what the file says.
    for fmt in ("md", "json", None):
        other = {"1": _sink("SaveText", filename_prefix="wf", format=fmt)}
        assert _sink_problems("wf", kind, other) == [txt_only]
    assert _sink_problems("wf", kind, {"1": _sink("SaveText", filename_prefix="wf")}) == [txt_only]
    # A second text (the model's reasoning, say) is compared too, under a name of its own.
    second = {**text, "2": _sink("SaveText", filename_prefix="wf_thinking", format="txt")}
    assert _sink_problems("wf", kind, second) == []
    assert _sink_problems("wf", kind, {**text, "2": text["1"]}) == [one_save]
    for prefix, said in (("text/wf", "flat filename_prefix"), ("other", "prefixed 'wf_...'"),
                         (None, "flat filename_prefix"), ("wf\\x", "flat filename_prefix")):
        bad = {**text, "2": _sink("SaveImage", filename_prefix=prefix)}
        assert said in _sink_problems("wf", kind, bad)[0]
    # The id as a folder, as SaveText would write it without the lint.
    nested = {"1": _sink("SaveText", filename_prefix="text/wf", format="txt")}
    assert _sink_problems("wf", kind, nested) == [
        one_save, "SaveText needs a flat filename_prefix, not 'text/wf'"]
    with pytest.raises(ValueError, match="unknown output_kind"):
        _sink_problems("wf", {"output_kind": "video"}, text)


def test_second_sinks_get_a_prefix_of_their_own_whatever_they_write():
    """The mesh of a 3D workflow and the clip of a video workflow are second
    files of an image workflow: `<id>_mesh`, `<id>_video`."""
    image = {"1": _sink("SaveImage", filename_prefix="wf", images=["9", 0])}
    for node, good in (("SaveGLB", "wf_mesh"), ("SaveVideo", "wf_video"),
                       ("VHS_VideoCombine", "wf_gif"), ("SaveAudioAdvanced", "wf_audio")):
        extra = {"format": "flac"} if node == "SaveAudioAdvanced" else {}
        assert _sink_problems("wf", {}, {**image, "2": _sink(node, filename_prefix=good,
                                                              **extra)}) == []
        for bad, said in (("wf", "a second sink is prefixed 'wf_...', not 'wf'"),
                          ("mesh", "a second sink is prefixed 'wf_...', not 'mesh'"),
                          ("wfmesh", "a second sink is prefixed 'wf_...', not 'wfmesh'"),
                          ("3d/wf_mesh", f"{node} needs a flat filename_prefix, not '3d/wf_mesh'"),
                          (None, f"{node} needs a flat filename_prefix, not None")):
            wf = {**image, "2": _sink(node, filename_prefix=bad, **extra)}
            assert _sink_problems("wf", {}, wf) == [said], (node, bad)
    # Two pictures of one workflow share the workflow id: they are numbered.
    two = {**image, "2": _sink("SaveImage", filename_prefix="wf", images=["8", 0])}
    assert _sink_problems("wf", {}, two) == []
    two["2"] = _sink("SaveImage", filename_prefix="wf_second", images=["8", 0])
    assert _sink_problems("wf", {}, two) == ["every SaveImage is prefixed with the workflow id"]


def test_sink_rules_of_an_audio_workflow():
    kind = {"output_kind": "audio", "level": "smoke"}
    audio = {"1": _sink("SaveAudioAdvanced", filename_prefix="wf", format="flac", audio=["9", 0])}
    assert _sink_problems("wf", kind, audio) == []
    # The deprecated node writes FLAC only: it has no format to set.
    assert _sink_problems("wf", kind, {"1": _sink("SaveAudio", filename_prefix="wf")}) == []
    assert _sink_problems("wf", kind, {}) == [
        "an audio workflow saves its sound with SaveAudioAdvanced"]
    named = "every SaveAudioAdvanced is prefixed with the workflow id"
    flac = "SaveAudioAdvanced must write format flac: only FLAC files are checked"
    other = {"1": _sink("SaveAudioAdvanced", filename_prefix="audio/wf", format="flac")}
    assert _sink_problems("wf", kind, other) == [named]
    assert _sink_problems("wf", kind, {**audio, "2": _sink("SaveAudio",
                                                           filename_prefix="wf_b")}) == [named]
    # The format is written out, as for a text: a default is not what the file says.
    for fmt in ("mp3", "opus", None):
        wf = {"1": _sink("SaveAudioAdvanced", filename_prefix="wf", format=fmt)}
        assert _sink_problems("wf", kind, wf) == [flac]
    assert _sink_problems("wf", kind, {"1": _sink("SaveAudioAdvanced",
                                                  filename_prefix="wf")}) == [flac]
    # A cover picture next to the sound is a second sink.
    cover = {**audio, "2": _sink("SaveImage", filename_prefix="wf")}
    assert _sink_problems("wf", kind, cover) == ["a second sink is prefixed 'wf_...', not 'wf'"]
    cover["2"] = _sink("SaveImage", filename_prefix="wf_cover")
    assert _sink_problems("wf", kind, cover) == []
    # An MP3 or an Opus file cannot be checked, in a workflow of any kind.
    for node in ("SaveAudioMP3", "SaveAudioOpus"):
        said = f"{node} is not allowed: its file cannot be checked. Use SaveAudioAdvanced " \
               "with format flac"
        assert _sink_problems("wf", kind, {**audio, "2": _sink(node, filename_prefix="wf_x")}) == [
            said]
        image = {"1": _sink("SaveImage", filename_prefix="wf"), "2": _sink(node)}
        assert _sink_problems("wf", {}, image) == [said]
    # The sound of an image workflow is held to the format too.
    image = {"1": _sink("SaveImage", filename_prefix="wf"),
             "2": _sink("SaveAudioAdvanced", filename_prefix="wf_audio", format="mp3")}
    assert _sink_problems("wf", {}, image) == [flac]


def test_a_clip_is_saved_as_an_mp4_with_h264():
    """SaveVideo offers mkv, webm and mov, and AV1 and ProRes. A smoke entry
    that says webm passed the lint, and every run of it failed: "SaveVideo
    wrote no .mp4 file"."""
    image = {"1": _sink("SaveImage", filename_prefix="wf", images=["9", 0])}
    said = "SaveVideo must write mp4 with h264: only that is checked"

    def problems(**inputs):
        clip = _sink("SaveVideo", filename_prefix="wf_video", video=["8", 0], **inputs)
        return _sink_problems("wf", {}, {**image, "2": clip})

    for fine in ({}, {"format": "mp4"}, {"format": "auto"}, {"format": "auto", "codec": "auto"},
                 {"format": "mp4", "codec": "h264"}, {"format": "mp4", "format.codec": "h264"},
                 {"format": "auto", "format.codec": "auto"}):
        assert problems(**fine) == [], fine
    for other in ({"format": "webm"}, {"format": "mkv"}, {"format": "mov"}, {"format": "MP4"},
                  {"format": "mp4", "codec": "av1"}, {"format": "mp4", "format.codec": "av1"},
                  {"format": "auto", "codec": "prores"}, {"format": "auto", "format.codec": "av1"},
                  {"format": None}, {"format": ["4", 0]}, {"format": "mp4", "codec": ["4", 0]}):
        assert problems(**other) == [said], other
    # The check looks for the file this rule allows, and no other save node is held to it.
    assert smoke.SINKS["SaveVideo"] == (".mp4",)
    assert _sink_problems("wf", {}, {**image, "2": _sink(
        "VHS_VideoCombine", filename_prefix="wf_gif", format="video/webm")}) == []
    # A workflow of the manifest that saves a clip saves it that way.
    for wf_id, entry in WORKFLOWS.items():
        for node in _load_workflow(entry).values():
            if node["class_type"] == "SaveVideo":
                assert _writes_mp4_h264(node["inputs"]), wf_id


def test_the_lint_knows_the_save_nodes_the_smoke_checks_know():
    """A save node the smoke checks expect a file from is an output node here,
    so a graph may end in it, and every output kind has its main sinks."""
    assert OUTPUT_NODES == set(smoke.SINKS) | {"VHS_VideoCombine"}
    assert set(MAIN_SINKS) == set(output_kinds.KINDS)
    assert all(nodes <= OUTPUT_NODES for nodes in MAIN_SINKS.values())
    assert not UNCHECKED_SINKS & OUTPUT_NODES
    assert COMPARED_KINDS < set(output_kinds.KINDS)


def test_level_cadence_and_expect_of_an_entry():
    base = {"thresholds": None, "golden_tag": None}
    assert _level_problems(base) == []
    assert _level_problems({**base, "level": "reference", "cadence": "per-commit"}) == []
    assert _level_problems({**base, "level": "smoke", "cadence": "weekly",
                            "expect": {"width": 1024, "height": 1024, "frames": 25,
                                       "seconds": 30.5, "min_chars": 40}}) == []
    assert _level_problems({**base, "level": "main"}) == [
        "unknown level 'main' (known: reference, smoke)"]
    assert _level_problems({**base, "cadence": "daily"}) == [
        "unknown cadence 'daily' (known: per-commit, nightly, weekly)"]
    # A lane can be retired, a workflow cannot ask for it.
    assert _level_problems({**base, "cadence": "retired"}) != []
    # What the output must be is said at smoke level only: a reference says it elsewhere.
    assert _level_problems({**base, "expect": {"width": 1024}}) == [
        "expect is for a smoke-level workflow only"]
    smoke_level = {**base, "level": "smoke"}
    assert _level_problems({**smoke_level, "expect": {}}) == []
    assert _level_problems({**smoke_level, "expect": None}) == []
    for expect, said in (({"size": 3}, "unknown expect key 'size' (known: width, height, frames, "
                                       "seconds, min_chars)"),
                         ({"width": 0}, "expect.width must be a number above 0, not 0"),
                         ({"seconds": -1}, "expect.seconds must be a number above 0, not -1"),
                         ({"frames": "25"}, "expect.frames must be a number above 0, not '25'"),
                         ({"frames": True}, "expect.frames must be a number above 0, not True"),
                         ([["width", 1]], "expect must be an object")):
        assert _level_problems({**smoke_level, "expect": expect}) == [said]
    # Audio has no comparator yet: it runs at smoke level only.
    said = "no comparator for output_kind 'audio' yet: it needs level smoke"
    assert _level_problems({**base, "output_kind": "audio"}) == [said]
    assert _level_problems({**smoke_level, "output_kind": "audio"}) == []
    assert _level_problems({**smoke_level, "output_kind": "text"}) == []
    assert _level_problems({**smoke_level, "output_kind": "video"}) == [
        "unknown output_kind 'video' (known: image, text, audio)"]
    # A smoke entry has no reference, so it sets no limits and names no golden.
    said = "a smoke entry has no reference: thresholds and golden_tag are null"
    assert _level_problems({**smoke_level, "thresholds": {"max_mean_mse": 2.0}}) == [said]
    assert _level_problems({**smoke_level, "golden_tag": "v1"}) == [said]


def test_moving_a_workflow_to_smoke_level_is_one_edit():
    """Nothing here asks a smoke entry for a reference: with `"level": "smoke"`
    (and a cadence) added, an entry of today still passes its own rules."""
    plain = {w: e for w, e in WORKFLOWS.items()
             if e["thresholds"] is None and e["golden_tag"] is None}
    assert plain, "no entry without limits of its own to try this on"
    for wf_id, entry in plain.items():
        moved = {**entry, "level": "smoke", "cadence": "nightly"}
        assert set(moved) <= ENTRY_KEYS | OPTIONAL_ENTRY_KEYS
        assert _level_problems(moved) == [], wf_id
        assert _sink_problems(wf_id, moved, _load_workflow(moved)) == [], wf_id


def test_text_generation_and_dynamic_input_rules():
    greedy = {"1": {"class_type": "TextGenerateV2",
                    "inputs": {"sampling_mode": "off", "mtp": "off", "prompt": "x"}}}
    assert _text_generation_problems(greedy) == []
    sampled = {"1": {"class_type": "TextGenerateV2",
                     "inputs": {"sampling_mode": "on", "sampling_mode.seed": 1}}}
    assert _text_generation_problems(sampled) == [
        "1: sampling_mode must be 'off' (greedy decoding)", "1: mtp must be set, and not to 'auto'"]
    assert _literal_dynamic_inputs(sampled) == ["1:TextGenerateV2.sampling_mode.seed"]
    # A linked sub-input is not a literal: the worker keeps links.
    linked = {"1": {"class_type": "TextEncodeQwenImage21", "inputs": {"images.image_1": ["2", 0]}}}
    assert _literal_dynamic_inputs(linked) == []
    # `auto` follows measured timings; a fixed depth and the older node are held to the same.
    for mtp, said in (("auto", ["1: mtp must be set, and not to 'auto'"]), ("off", []), ("2", [])):
        node = {"class_type": "TextGenerate", "inputs": {"sampling_mode": "off", "mtp": mtp}}
        assert _text_generation_problems({"1": node}) == said
    assert _text_generation_problems({"1": {"class_type": "CLIPLoader", "inputs": {}}}) == []


def test_the_text_workflows_are_declared_as_text_and_write_their_own_text_file():
    """What the two staged workflows rely on, so an edit cannot drop it quietly.
    Another text workflow is held to the rules above only: it may set a
    similarity limit, save a second text or use a fixed mtp depth."""
    text = {w: e for w, e in WORKFLOWS.items() if output_kinds.kind_of(e) == "text"}
    assert {"llm_qwen38_27b_t2t", "llm_qwen3vl_4b_it2t"} <= set(text)
    for wf_id in ("llm_qwen38_27b_t2t", "llm_qwen3vl_4b_it2t"):
        entry = text[wf_id]
        wf = _load_workflow(entry)
        assert entry["thresholds"] is None, "exact match: no similarity limit to start with"
        assert entry["seed_overrides"] == {}, "greedy decoding has no seed"
        (save_id,) = [nid for nid, n in wf.items() if n["class_type"] == "SaveText"]
        (gen_id,) = [nid for nid, n in wf.items() if n["class_type"].startswith("TextGenerate")]
        save, gen = wf[save_id]["inputs"], wf[gen_id]["inputs"]
        assert (save["filename_prefix"], save["format"]) == (wf_id, "txt")
        assert save["text"] == [gen_id, 0], "output 0 is the generated text"
        assert (gen["sampling_mode"], gen["mtp"], gen["thinking"]) == ("off", "off", False)
        assert gen["use_default_template"] is True and 0 < gen["max_length"] <= 512
        assert not [n for n in wf.values() if n["class_type"].startswith("Preview")]
    # The 27B model loads 17 GB before it writes a token.
    assert WORKFLOWS["llm_qwen38_27b_t2t"]["timeout_s"] == 900
    assert WORKFLOWS["llm_qwen3vl_4b_it2t"]["timeout_s"] is None
    # An image entry never names its kind: absent means image.
    assert not [w for w, e in WORKFLOWS.items() if e.get("output_kind") == "image"]
