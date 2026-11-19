#!/usr/bin/env python3
"""Turn a workflow template into a corpus entry: one command, one pull request.

    python scripts/import_template.py audio_ace_step_1_5_checkpoint --id ace_step_15_t2a \\
        --server http://127.0.0.1:8188 --cadence nightly --timeout-s 900 \\
        --set 94.duration=30 --set 98.seconds=30 --expect seconds=30

The command converts the template to API format, makes its outputs the files
the suite can check, posts the prompt to the server to see that ComfyUI takes
it, writes ``manifest/workflows/<id>_api.json``, adds the entry to
``manifest/workflows.json`` and runs the manifest lint. With ``--dry-run`` it
prints the converted nodes, the entry and the result of the prompt check, and
writes nothing. That is where the node ids for ``--set`` come from.

What it does to the template, step by step:

* converts it the way the frontend's Export (API) does and picks the branch a
  constant switch selects (``template_api_export.py``)
* sets every ``TextGenerate`` node to greedy decoding. ``--keep-sampling``
  leaves a smoke-level workflow as the template has it
* names each output after the workflow id and adds the picture a video or a
  mesh is shown and compared by (``normalise_sinks``)
* drops the preview nodes
* applies the ``--set`` values
* reads the model files from the loader nodes (``properties.models``), with
  the sha256 and the size the host publishes, and lists the input files of
  ``LoadImage``, ``LoadAudio`` and ``LoadVideo`` at a pinned commit of the
  templates repository

``--server`` names a running ComfyUI of the current version. Its
``/object_info`` drives the conversion, and it gets the finished prompt. The
model and input files do not have to be there: a file input the server lists
nothing for is filled with the template's own file names, and "value not in
list" on such an input is no error. ``--object-info FILE`` takes a saved dump
where no server runs. The prompt check is then skipped.

The template comes from ``--templates-dir``, the ``templates/`` folder of a
workflow_templates checkout, or from GitHub at ``--templates-ref``.

It refuses and does not guess. A template whose widget layout lives on the
node instance (``CustomCombo``), one that needs nodes the server does not
have, or one that saves a file no check can read, is refused with the reason.

What the manifest lint would refuse in the new entry is refused before
anything is written, and a dry run says it too: a sampled text generation at
reference level, a model folder the worker does not search, a file the
manifest lists as another file, a seed nothing pins. A file the manifest
already lists is written the way the manifest has it. A rule that adds up
the whole manifest, like the room on the volume, is the lint's alone.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lanes as lanes_mod  # noqa: E402
from template_api_export import (  # noqa: E402
    ConversionError,
    Converter,
    FlatGraph,
    OUTPUT_PREFIXES,
    PREVIEW_PREFIXES,
    _entry_opts,
    _ordered_inputs,
)

ROOT = Path(__file__).resolve().parents[1]

SAVE_IMAGE = "SaveImage"
IMAGE_SAVES = ("SaveImage", "SaveImageAdvanced")
ANIMATED_SAVES = ("SaveAnimatedWEBP", "SaveAnimatedPNG")
AUDIO_SAVES = ("SaveAudioAdvanced", "SaveAudio", "SaveAudioMP3", "SaveAudioOpus")
# The save nodes a finished workflow may hold. Each has a check of its file.
KEPT_SAVES = ("SaveImage", "SaveText", "SaveAudioAdvanced", "SaveGLB", "SaveVideo")
# Nodes whose named input is a file from ComfyUI's input folder.
INPUT_FILE_BY_NODE = {"LoadImage": "image", "LoadAudio": "audio", "LoadVideo": "file"}
INPUT_DIR = "/runpod-volume/models/input"
MODEL_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# A file of a Hugging Face model repository: repository, revision, path.
HF_FILE = re.compile(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")
# The token of the environment goes to this host over this scheme, and
# nowhere else: over plain http anyone on the way reads it.
HF_HOST = "huggingface.co"
HF_SCHEME = "https"
TEMPLATES_REPO = "Comfy-Org/workflow_templates"
RAW = "https://raw.githubusercontent.com"
LEVELS = ("smoke", "reference")
RENDER_PX = 1024
HASH_CHUNK = 1024 * 1024
SEED_NAMES = ("seed", "noise_seed")
# What a request can end in besides an answer: no connection, or an answer
# that is cut short or is no HTTP.
NET_ERRORS = (OSError, http.client.HTTPException)
# A file name with the folder ComfyUI is told to read it from.
ANNOTATED = re.compile(r" \[(input|output|temp)\]$")
# Input kinds whose values come under names of their own (``images.image0``).
GROWING = ("COMFY_AUTOGROW_V3", "COMFY_DYNAMICGROUP_V3")

# ComfyUI's models/ subdirectory for each loader input, for a loader node that
# does not list its files in `properties.models`. Keyed by loader *and* input
# name, because the name alone is ambiguous: CLIPLoader.clip_name reads
# text_encoders/ while CLIPVisionLoader.clip_name reads clip_vision/.
DIRECTORY_BY_LOADER_INPUT = {
    ("CheckpointLoader", "ckpt_name"): "checkpoints",
    ("CheckpointLoaderSimple", "ckpt_name"): "checkpoints",
    ("unCLIPCheckpointLoader", "ckpt_name"): "checkpoints",
    ("UNETLoader", "unet_name"): "diffusion_models",
    ("CLIPLoader", "clip_name"): "text_encoders",
    ("DualCLIPLoader", "clip_name1"): "text_encoders",
    ("DualCLIPLoader", "clip_name2"): "text_encoders",
    ("TripleCLIPLoader", "clip_name1"): "text_encoders",
    ("TripleCLIPLoader", "clip_name2"): "text_encoders",
    ("TripleCLIPLoader", "clip_name3"): "text_encoders",
    ("QuadrupleCLIPLoader", "clip_name1"): "text_encoders",
    ("QuadrupleCLIPLoader", "clip_name2"): "text_encoders",
    ("QuadrupleCLIPLoader", "clip_name3"): "text_encoders",
    ("QuadrupleCLIPLoader", "clip_name4"): "text_encoders",
    ("CLIPVisionLoader", "clip_name"): "clip_vision",
    ("VAELoader", "vae_name"): "vae",
    ("LoraLoader", "lora_name"): "loras",
    ("LoraLoaderModelOnly", "lora_name"): "loras",
    ("ControlNetLoader", "control_net_name"): "controlnet",
    ("DiffControlNetLoader", "control_net_name"): "controlnet",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
    ("StyleModelLoader", "style_model_name"): "style_models",
    ("GLIGENLoader", "gligen_name"): "gligen",
    ("HypernetworkLoader", "hypernetwork_name"): "hypernetworks",
    ("PhotoMakerLoader", "photomaker_model_name"): "photomaker",
}


def is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _read_ids(prompt: dict[str, dict[str, Any]]) -> set[str]:
    """The ids of the nodes another node reads from."""
    return {v[0] for node in prompt.values() for v in node["inputs"].values() if is_link(v)}


def _new_id(prompt: dict[str, dict[str, Any]]) -> str:
    """A node id no node has: one more than the largest number an id starts with."""
    heads = [int(m.group()) for node_id in prompt if (m := re.match(r"\d+", node_id))]
    return str(max(heads, default=0) + 1)


# --------------------------------------------------------------------------- #
# file inputs the server lists nothing for
# --------------------------------------------------------------------------- #


def _optionless(entry: Any) -> bool:
    """Whether an input is a combo of the newer form with no option to choose."""
    opts = _entry_opts(entry)
    return (isinstance(entry, (list, tuple)) and bool(entry) and entry[0] == "COMBO"
            and not opts.get("options") and "default" not in opts)


def _own_files(node: dict[str, Any], input_name: str) -> list[str]:
    """The file names a template node holds for one of its inputs."""
    models = (node.get("properties") or {}).get("models") or []
    names = [m["name"] for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]
    if not names and INPUT_FILE_BY_NODE.get(node.get("type")) == input_name:
        values = node.get("widgets_values")
        first = values.get(input_name) if isinstance(values, dict) else (values or [None])[0]
        names = [first] if isinstance(first, str) and first else []
    return names


def fill_file_combos(object_info: dict[str, Any], template: dict[str, Any]) -> set[tuple[str, str]]:
    """Give each file input the server lists no file for the template's own names.

    A ComfyUI that does not have a model or an input file lists no option for
    the loader's combo, and the converter refuses a combo without options: it
    cannot tell it from one whose options live on the node instance. So the
    definition in ``object_info`` is filled, in memory, with what the template
    names for that node: a model loader's ``properties.models``, and the file
    name in the widgets of ``LoadImage``, ``LoadAudio`` and ``LoadVideo``.
    Returns the ``(class, input)`` pairs that were filled. ``CustomCombo`` has
    neither, so it stays refused: its options do live on the node.
    """
    filled: set[tuple[str, str]] = set()
    for node in FlatGraph(template).nodes.values():
        node_type = node.get("type")
        spec = object_info.get(node_type) if isinstance(node_type, str) else None
        if not isinstance(spec, dict):
            continue
        for name, entry in _ordered_inputs(spec):
            if (node_type, name) not in filled and not _optionless(entry):
                continue
            names = _own_files(node, name)
            if not names:
                continue
            if len(entry) < 2 or not isinstance(entry[1], dict):
                entry[1:] = [{}]
            options = entry[1].setdefault("options", [])
            options.extend(n for n in names if n not in options)
            filled.add((node_type, name))
    return filled


# --------------------------------------------------------------------------- #
# the prompt: text generation, outputs, previews, --set
# --------------------------------------------------------------------------- #


def greedy_text(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> None:
    """Make every text generation greedy: no sampling, no multi-token head.

    A sampled text can differ from run to run, and a reference-level output
    is compared with its reference. The sub-inputs that only sampling reads
    (``sampling_mode.<name>``) go with it. An input the node of this ComfyUI
    does not have is not added.
    """
    for node in prompt.values():
        if not node["class_type"].startswith("TextGenerate"):
            continue
        inputs = node["inputs"]
        spec = (object_info.get(node["class_type"]) or {}).get("input") or {}
        known = set(spec.get("required") or {}) | set(spec.get("optional") or {})
        for name in [n for n in inputs if n.startswith("sampling_mode.")]:
            del inputs[name]
        for name in ("sampling_mode", "mtp"):
            if name in known or name in inputs:
                inputs[name] = "off"


def collapse_previews(prompt: dict[str, dict[str, Any]]) -> None:
    """Drop preview nodes, forwarding the value of the ones that pass through.

    ``PreviewAny`` sits mid-graph in several templates, showing a string on its
    way to a text encoder. It is an output node, so leaving it in would make it
    a second thing the run executes and a node that feeds no save node.
    """
    for node_id, node in list(prompt.items()):
        if not node["class_type"].startswith(PREVIEW_PREFIXES):
            continue
        forwarded = next(iter(node["inputs"].values()), None)
        for other in prompt.values():
            for name, value in list(other["inputs"].items()):
                if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                    if forwarded is None:
                        raise ConversionError(
                            f"preview node {node_id} feeds {name} but has no input"
                        )
                    other["inputs"][name] = (
                        list(forwarded) if isinstance(forwarded, list) else forwarded
                    )
        del prompt[node_id]


def _need(object_info: dict[str, Any], *classes: str) -> None:
    missing = [c for c in classes if c not in object_info]
    if missing:
        raise ConversionError(
            f"this ComfyUI has no {', '.join(missing)} node: the importer adds it to save a "
            "picture of the output. Use a ComfyUI of the current version"
        )


def _plain_save(prompt: dict[str, dict[str, Any]], node_id: str, prefix: str) -> None:
    """Make an image save node a ``SaveImage`` PNG with the given prefix.

    Templates save through ``SaveImage`` or ``SaveImageAdvanced``, whose
    png/8-bit/sRGB default writes the same file. Rewriting the advanced node
    keeps every workflow in the corpus on one output shape.
    """
    node = prompt[node_id]
    images = node["inputs"].get("images")
    if images is None:
        raise ConversionError(f"save node {node_id} has no images input")
    written = (
        node["inputs"].get("format", "png"),
        node["inputs"].get("format.bit_depth", "8-bit"),
        node["inputs"].get("format.input_color_space", "sRGB"),
    )
    if written != ("png", "8-bit", "sRGB"):
        raise ConversionError(
            f"save node {node_id} writes {'/'.join(map(str, written))}, but "
            "comparison needs 8-bit PNG in sRGB: rewriting it to SaveImage would "
            "silently drop that setting and compare a file the template does not "
            "produce"
        )
    node["class_type"] = SAVE_IMAGE
    node["inputs"] = {"filename_prefix": prefix, "images": images}


def _save_last_frame(prompt: dict[str, dict[str, Any]], frames: list, prefix: str,
                     save_id: str | None = None) -> None:
    """Add the nodes that save the last picture of a batch as a PNG."""
    frame_id = _new_id(prompt)
    prompt[frame_id] = {
        "class_type": "ImageFromBatch",
        "inputs": {"image": list(frames), "batch_index": -1, "length": 1},
    }
    prompt[save_id or _new_id(prompt)] = {
        "class_type": SAVE_IMAGE,
        "inputs": {"filename_prefix": prefix, "images": [frame_id, 0]},
    }


def _keep_video(prompt: dict[str, dict[str, Any]], node_id: str, workflow_id: str) -> None:
    """Keep a ``SaveVideo`` as an H.264 MP4 and save its last frame as the picture."""
    node = prompt[node_id]
    video = node["inputs"].get("video")
    if not is_link(video):
        raise ConversionError(f"save node {node_id} has no video input")
    source = prompt.get(video[0], {})
    frames = source.get("inputs", {}).get("images")
    if source.get("class_type") != "CreateVideo" or not is_link(frames):
        parts_id = _new_id(prompt)
        prompt[parts_id] = {"class_type": "GetVideoComponents", "inputs": {"video": list(video)}}
        frames = [parts_id, 0]
    node["inputs"] = {
        "filename_prefix": f"{workflow_id}_video",
        "format": "mp4",
        "format.codec": "h264",
        "video": video,
    }
    _save_last_frame(prompt, frames, workflow_id)


def _feeds_a_mesh(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any],
                  link: Any) -> bool:
    """Whether a link carries a mesh, by the output type of the node it comes from."""
    if not is_link(link) or link[0] not in prompt:
        return False
    outputs = (object_info.get(prompt[link[0]]["class_type"]) or {}).get("output") or []
    return link[1] < len(outputs) and outputs[link[1]] == "MESH"


def _keep_mesh(prompt: dict[str, dict[str, Any]], node_id: str, workflow_id: str) -> None:
    """Keep a ``SaveGLB`` and save a picture of its mesh.

    ``RenderMesh`` without a camera frames the mesh by itself, from the front.
    """
    node = prompt[node_id]
    mesh = node["inputs"]["mesh"]
    node["inputs"] = {"filename_prefix": f"{workflow_id}_mesh", "mesh": mesh}
    render_id = _new_id(prompt)
    prompt[render_id] = {
        "class_type": "RenderMesh",
        "inputs": {"mesh": list(mesh), "mode": "auto", "width": RENDER_PX,
                   "height": RENDER_PX, "background": "#000000"},
    }
    prompt[_new_id(prompt)] = {
        "class_type": SAVE_IMAGE,
        "inputs": {"filename_prefix": workflow_id, "images": [render_id, 0]},
    }


def normalise_sinks(prompt: dict[str, dict[str, Any]], workflow_id: str,
                    object_info: dict[str, Any]) -> str:
    """Make the outputs the files the suite checks. Returns the output kind.

    The kind names the main output, the one saved under the workflow id:
    ``image`` (a PNG), ``audio`` (a FLAC) or ``text`` (a .txt). A video and a
    mesh are image workflows: the clip is saved as ``<id>_video`` and the mesh
    as ``<id>_mesh``, and a picture of each is added as the main output. With
    outputs of several kinds the first of video, mesh, audio, image, text
    names the kind. Every save node of that kind gets the workflow id as its
    prefix and every other one ``<id>_<type>``. Runs before the previews are
    dropped: a template that ends in a preview alone saves what it shows.
    """
    def of(*classes: str) -> list[str]:
        return [nid for nid, node in prompt.items() if node["class_type"] in classes]

    videos, meshes, audios = of("SaveVideo"), of("SaveGLB"), of(*AUDIO_SAVES)
    images, animated, texts = of(*IMAGE_SAVES), of(*ANIMATED_SAVES), of("SaveText")
    files_3d = of("Save3DAdvanced") + [
        nid for nid in meshes
        if not _feeds_a_mesh(prompt, object_info, prompt[nid]["inputs"].get("mesh"))
    ]
    if files_3d:
        raise ConversionError(
            f"no mesh link to render (node {', '.join(files_3d)} saves a 3D file, not a "
            "mesh): add a SaveGLB and a RenderMesh by hand"
        )
    read = _read_ids(prompt)
    saves = videos or meshes or audios or images or animated or texts
    # What a template without a save node shows: a picture, or a text.
    pictures = [] if saves else [nid for nid in of("PreviewImage") if nid not in read]
    answers = [] if saves or pictures else [
        nid for nid in of("PreviewAny")
        if nid not in read and is_link(prompt[nid]["inputs"].get("source"))
    ]

    if videos or meshes:
        kind = "image"
    elif audios:
        kind = "audio"
    elif images or animated or pictures:
        kind = "image"
    elif texts or answers:
        kind = "text"
    else:
        raise ConversionError(
            "the template saves nothing the suite can check: it has no image, video, "
            "mesh, audio or text output"
        )

    def prefix(sink: str) -> str:
        return workflow_id if sink == kind else f"{workflow_id}_{sink}"

    if videos:
        _need(object_info, "ImageFromBatch", "GetVideoComponents")
    if meshes:
        _need(object_info, "RenderMesh")
    if animated:
        _need(object_info, "ImageFromBatch")
    for node_id in videos:
        _keep_video(prompt, node_id, workflow_id)
    for node_id in meshes:
        _keep_mesh(prompt, node_id, workflow_id)
    for node_id in audios:
        audio = prompt[node_id]["inputs"].get("audio")
        if not is_link(audio):
            raise ConversionError(f"save node {node_id} has no audio input")
        prompt[node_id] = {
            "class_type": "SaveAudioAdvanced",
            "inputs": {"filename_prefix": prefix("audio"), "format": "flac", "audio": audio},
        }
    for node_id in images:
        _plain_save(prompt, node_id, prefix("image"))
    for node_id in animated:
        frames = prompt[node_id]["inputs"].get("images")
        if not is_link(frames):
            raise ConversionError(f"save node {node_id} has no images input")
        _save_last_frame(prompt, frames, prefix("image"), save_id=node_id)
    if pictures:
        shown = prompt[pictures[-1]]
        if not is_link(shown["inputs"].get("images")):
            raise ConversionError(f"preview node {pictures[-1]} has no images input")
        shown["class_type"] = SAVE_IMAGE
        shown["inputs"] = {"filename_prefix": workflow_id, "images": shown["inputs"]["images"]}
    if kind == "text":
        ends = texts or answers
        if len(ends) != 1:
            raise ConversionError(
                f"the template has {len(ends)} text outputs (nodes {', '.join(ends)}) and one "
                "is compared as the answer: finish it by hand, with one SaveText prefixed "
                f'"{workflow_id}" and any other prefixed "{workflow_id}_<name>"'
            )
    for node_id in texts or answers:
        inputs = prompt[node_id]["inputs"]
        text = inputs.get("text", inputs.get("source"))
        if not is_link(text):
            raise ConversionError(f"save node {node_id} has no text input")
        prompt[node_id] = {
            "class_type": "SaveText",
            "inputs": {"text": text, "filename_prefix": prefix("text"), "format": "txt"},
        }

    other = sorted({
        node["class_type"] for node in prompt.values()
        if _is_output(node["class_type"], object_info)
        and "Save" in node["class_type"] and node["class_type"] not in KEPT_SAVES
    })
    if other:
        raise ConversionError(
            f"the template also saves with {', '.join(other)}, and no check reads what "
            "that writes: export the workflow by hand and leave the node out"
        )
    return kind


def _is_output(class_type: str, object_info: dict[str, Any]) -> bool:
    return class_type.startswith(OUTPUT_PREFIXES) or bool(
        (object_info.get(class_type) or {}).get("output_node"))


def keep_what_is_saved(prompt: dict[str, dict[str, Any]]) -> None:
    """Drop every node that does not feed a save node.

    ComfyUI runs every output node of a prompt. A node that only shows
    something (``ImageCompare``, ``MaskPreview``) is one too, so it would run,
    with all that feeds it, for a result nobody reads.
    """
    keep: set[str] = set()
    stack = [nid for nid, node in prompt.items() if node["class_type"] in KEPT_SAVES]
    while stack:
        node_id = stack.pop()
        if node_id in keep or node_id not in prompt:
            continue
        keep.add(node_id)
        stack.extend(v[0] for v in prompt[node_id]["inputs"].values() if is_link(v))
    for node_id in list(prompt):
        if node_id not in keep:
            del prompt[node_id]


def describe(prompt: dict[str, dict[str, Any]]) -> str:
    """One line a node: its id, its class and its literal inputs."""
    def short(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= 60:
            return text
        return text[:56] + '..."' if text.startswith('"') else text[:57] + "..."

    width = max((len(node_id) for node_id in prompt), default=1)
    lines = []
    for node_id, node in prompt.items():
        literals = " ".join(f"{name}={short(value)}" for name, value in node["inputs"].items()
                            if not is_link(value))
        lines.append(f"  {node_id:>{width}}  {node['class_type']}  {literals}".rstrip())
    return "\n".join(lines)


def _finite(value: Any) -> bool:
    """Whether a value holds no number that JSON cannot write."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _json_value(raw: str) -> Any:
    """A value written as JSON. ValueError for anything else.

    Python reads ``NaN`` and ``Infinity`` and writes them back. They are not
    JSON, and a file that holds one is read by nothing else.
    """
    def refuse(word: str) -> None:
        raise ValueError(f"{word} is not JSON")

    value = json.loads(raw, parse_constant=refuse)
    if not _finite(value):
        raise ValueError("a number too large for JSON")
    return value


def apply_sets(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any],
               sets: list[str]) -> None:
    """Apply the ``--set <node>.<input>=<value>`` overrides. Values are JSON.

    An input name may hold a dot of its own (``format.codec``), a node id
    never does, so the first dot ends the id.
    """
    for text in sets:
        target, equals, raw = text.partition("=")
        node_id, dot, name = target.partition(".")
        if not (equals and dot and node_id and name):
            raise ConversionError(f"--set {text}: write <node>.<input>=<value>")
        try:
            value = _json_value(raw)
        except ValueError:
            raise ConversionError(
                f"--set {text}: the value is not JSON. A number is written 30, a "
                f"""string with quotes: '{target}="{raw}"'"""
            ) from None
        node = prompt.get(node_id)
        if node is None:
            raise ConversionError(f"--set {text}: the workflow has no node {node_id}")
        spec = (object_info.get(node["class_type"]) or {}).get("input") or {}
        known = set(node["inputs"]) | set(spec.get("required") or {}) | set(
            spec.get("optional") or {})
        if name not in known:
            raise ConversionError(
                f"--set {text}: node {node_id} ({node['class_type']}) has no input {name}")
        if is_link(node["inputs"].get(name)):
            raise ConversionError(
                f"--set {text}: {node_id}.{name} is fed by node {node['inputs'][name][0]}: "
                "set the value there"
            )
        node["inputs"][name] = value


def _node_inputs(object_info: dict[str, Any], class_type: str) -> dict[str, Any]:
    """The inputs a node class has in this ComfyUI, by name."""
    spec = (object_info.get(class_type) or {}).get("input") or {}
    return {**(spec.get("required") or {}), **(spec.get("optional") or {})}


def unknown_literals(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> list[str]:
    """The literal inputs a node of this ComfyUI does not have.

    The worker drops such an input before it queues the prompt, and ComfyUI
    would not read it either: the run would go on without the value the
    template has. An input named ``a.b`` is a sub-input of the input ``a``
    (the codec of a ``SaveVideo`` format, the seed of a sampling mode), and
    ``/object_info`` lists only ``a``: such an input is known when ``a`` is,
    which is the worker's rule too.
    """
    return sorted(
        f"{node_id}:{node['class_type']}.{name}" for node_id, node in prompt.items()
        if node["class_type"] in object_info
        for name, value in node["inputs"].items()
        if not isinstance(value, list)
        and name.split(".")[0] not in _node_inputs(object_info, node["class_type"]))


def open_required(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> list[str]:
    """The required inputs a node has neither a link nor a value for.

    ComfyUI refuses such a prompt: "Required input is missing". The converter
    reads the widgets whose type it knows, and the value another widget holds
    is left out without a word: the boxes of a ``CreateBoundingBoxes``, whose
    type is ``BOUNDING_BOXES``. An input that grows is not looked at: its
    values come under names of their own (``images.image0``).
    """
    found = []
    for node_id, node in prompt.items():
        spec = (object_info.get(node["class_type"]) or {}).get("input") or {}
        for name, entry in (spec.get("required") or {}).items():
            kind = entry[0] if isinstance(entry, (list, tuple)) and entry else None
            if name not in node["inputs"] and kind not in GROWING:
                found.append(f"{node_id}:{node['class_type']}.{name}"
                             + (f" ({kind})" if isinstance(kind, str) else ""))
    return sorted(found)


def refuse_stray_files(prompt: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> None:
    """Stop on a file input the entry cannot list.

    ``/object_info`` marks an input that takes an uploaded file. The entry
    lists the files of ``LoadImage``, ``LoadAudio`` and ``LoadVideo``, which
    the templates repository serves. A file of the temp or the output folder
    lived on the machine the template was made on, and no run has it.
    """
    for node_id, node in prompt.items():
        class_type = node["class_type"]
        entries = _node_inputs(object_info, class_type)
        for name, value in node["inputs"].items():
            if not isinstance(value, str) or not value.strip():
                continue
            if not any(key.endswith("_upload") for key in _entry_opts(entries.get(name))):
                continue
            folder = ANNOTATED.search(value)
            if folder and folder.group(1) != "input":
                raise ConversionError(
                    f"node {node_id} ({class_type}) reads {value}: a file of the "
                    f"{folder.group(1)} folder of the machine the template was made on, "
                    "which no run has"
                )
            if INPUT_FILE_BY_NODE.get(class_type) != name:
                raise ConversionError(
                    f"node {node_id} ({class_type}) reads the file {value} through {name}, "
                    "and the importer lists the input files of "
                    f"{', '.join(INPUT_FILE_BY_NODE)} only: write the workflow by hand"
                )


def finish_prompt(template: dict[str, Any], object_info: dict[str, Any], workflow_id: str,
                  sets: list[str] | None = None, keep_sampling: bool = False,
                  level: str = "smoke", shown: list | None = None,
                  ) -> tuple[dict, str, set[tuple[str, str]], list[str]]:
    """The prompt the workflow file holds, its output kind, the file inputs
    that were filled, and notes for the author.

    ``shown`` gets the prompt as it stood after each step that ended well, so
    a refusal can print the nodes it was looking at.
    """
    shown = [] if shown is None else shown
    filled = fill_file_combos(object_info, template)
    prompt = Converter(object_info).convert(template, resolve_switches=True)
    shown[:] = [copy.deepcopy(prompt)]
    if not keep_sampling:
        greedy_text(prompt, object_info)
    shows = {nid: node["class_type"] for nid, node in prompt.items()
             if _is_output(node["class_type"], object_info)
             and not node["class_type"].startswith(PREVIEW_PREFIXES)}
    kind = normalise_sinks(prompt, workflow_id, object_info)
    collapse_previews(prompt)
    keep_what_is_saved(prompt)
    shown[:] = [copy.deepcopy(prompt)]
    notes = [f"note: left out {class_type} (node {nid}): it saves nothing the suite checks"
             for nid, class_type in shows.items() if nid not in prompt]
    apply_sets(prompt, object_info, sets or [])
    shown[:] = [prompt]

    if kind == "audio" and level == "reference":
        raise ConversionError(
            "the template's output is audio, and nothing compares audio yet: "
            "use --level smoke"
        )
    partner = sorted({node["class_type"] for node in prompt.values()
                      if (object_info.get(node["class_type"]) or {}).get("api_node")})
    if partner:
        raise ConversionError(
            f"the template calls a paid service through {', '.join(partner)}: a partner "
            "node needs an account key and credits, and the worker has neither"
        )
    refuse_stray_files(prompt, object_info)
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if not _finite(value):
                raise ConversionError(
                    f"{node_id}.{name} is {value!r}, a number JSON cannot hold")
    unknown = unknown_literals(prompt, object_info)
    if unknown:
        raise ConversionError(
            "the worker drops these inputs before it queues the prompt, because the node "
            f"of this ComfyUI does not have them: {', '.join(unknown)}. The template was "
            "made for another version of the node"
        )
    left_open = open_required(prompt, object_info)
    if left_open:
        raise ConversionError(
            "ComfyUI refuses a prompt without these required inputs, and the converter "
            f"found neither a link nor a value for them: {', '.join(left_open)}. The "
            "template leaves the input open, or holds its value in a widget of a type the "
            "converter does not read. Give the value with --set, or write the workflow by "
            "hand"
        )
    return prompt, kind, filled, notes


# --------------------------------------------------------------------------- #
# model files and input files
# --------------------------------------------------------------------------- #


def note_urls(template: dict[str, Any]) -> dict[str, str]:
    """Download URLs keyed by file name, from the template's model-links notes."""
    notes: list[str] = []
    graphs = [template, *((template.get("definitions") or {}).get("subgraphs") or [])]
    for graph in graphs:
        for node in graph.get("nodes") or []:
            if node.get("type") not in ("MarkdownNote", "Note"):
                continue
            for value in node.get("widgets_values") or []:
                if isinstance(value, str):
                    notes.append(value)

    urls: dict[str, str] = {}
    for text in notes:
        for _label, url in MARKDOWN_LINK.findall(text):
            name = url.split("?")[0].rsplit("/", 1)[-1]
            urls.setdefault(name, url)
    return urls


def clean_url(url: str) -> str:
    """A Hugging Face URL without its query (``?download=true``).

    Another host keeps its query: on civitai it selects the file.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != HF_HOST:
        return url
    return urllib.parse.urlunsplit(parts._replace(query="", fragment=""))


def _file_name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _download_link(name: str, url: str) -> str:
    """The URL the entry gets for a model file. The manifest lint takes https only."""
    if not url.startswith("https://"):
        raise ConversionError(f"the download link of {name} is not https: {url}")
    return clean_url(url)


def model_files(prompt: dict[str, dict[str, Any]],
                template: dict[str, Any]) -> list[dict[str, str]]:
    """The model files the workflow loads: name, download URL, models/ folder.

    Read from ``properties.models`` of the template node behind each node of
    the prompt, so only the loaders that survived are asked. A loader that
    lists nothing falls back to the links in the template's notes and to
    ``DIRECTORY_BY_LOADER_INPUT``. In the order the prompt loads them.
    """
    nodes = FlatGraph(template).nodes
    notes: dict[str, str] | None = None
    found: dict[str, dict[str, str]] = {}
    for node_id, node in prompt.items():
        class_type = node["class_type"]
        listed = {
            m["name"]: m for m in ((nodes.get(node_id) or {}).get("properties") or {})
            .get("models") or []
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        }
        for name, value in node["inputs"].items():
            if not isinstance(value, str) or INPUT_FILE_BY_NODE.get(class_type) == name:
                continue
            own = listed.get(value) or listed.get(_file_name(value))
            if own and isinstance(own.get("url"), str) and isinstance(own.get("directory"), str):
                model = {"name": value, "url": _download_link(value, own["url"]),
                         "directory": own["directory"]}
            elif value.endswith(MODEL_SUFFIXES):
                directory = DIRECTORY_BY_LOADER_INPUT.get((class_type, name))
                if directory is None:
                    raise ConversionError(
                        f"{class_type}.{name} loads {value}, but which models/ "
                        "subdirectory it lives in is not known: add "
                        f"(\"{class_type}\", \"{name}\") to DIRECTORY_BY_LOADER_INPUT"
                    )
                notes = note_urls(template) if notes is None else notes
                url = notes.get(_file_name(value))
                if url is None:
                    raise ConversionError(
                        f"{value} is loaded by the workflow but the template has no "
                        "download link for it, on the loader node or in its notes: add "
                        "the models entry by hand"
                    )
                model = {"name": value, "url": _download_link(value, url),
                         "directory": directory}
            else:
                continue
            key = f"{model['directory']}/{model['name']}"
            if found.setdefault(key, model)["url"] != model["url"]:
                raise ConversionError(
                    f"the template has two download links for {key}: "
                    f"{found[key]['url']} and {model['url']}"
                )
    return list(found.values())


def input_names(prompt: dict[str, dict[str, Any]]) -> list[str]:
    """The files the workflow reads from ComfyUI's input folder, each once."""
    names: list[str] = []
    for node in prompt.values():
        value = node["inputs"].get(INPUT_FILE_BY_NODE.get(node["class_type"], ""))
        if isinstance(value, str):
            name = ANNOTATED.sub("", value)
            if name not in names:
                names.append(name)
    return names


def input_url(name: str, sha: str) -> str:
    """Where the templates repository serves an input file, at one commit."""
    return f"{RAW}/{TEMPLATES_REPO}/{sha}/input/{urllib.parse.quote(name)}"


# --------------------------------------------------------------------------- #
# what a host says about a file
# --------------------------------------------------------------------------- #


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face only sends X-Linked-ETag on the redirect, not on the CDN hit."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _takes_the_token(url: str) -> bool:
    """Whether a URL is Hugging Face itself, over a connection nobody reads."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == HF_SCHEME and parts.hostname == HF_HOST


class _TokenStaysHome(urllib.request.HTTPRedirectHandler):
    """Follow a redirect, and take the token off when it leaves Hugging Face.

    Python copies every header to the redirect target, and a file of a model
    repository redirects to a storage host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _takes_the_token(newurl):
            new.headers.pop("Authorization", None)
        return new


def _open(url: str, method: str, follow: bool):
    """Send a request. The Hugging Face token of the environment goes to
    huggingface.co only, and only over https: a gated repository answers
    nothing without it."""
    request = urllib.request.Request(url, method=method)
    token = os.environ.get("HF_TOKEN")
    if token and _takes_the_token(url):
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(_TokenStaysHome if follow else _NoRedirect)
    return opener.open(request, timeout=60)


def _head(url: str, follow: bool) -> tuple[int | None, Any]:
    """(status, headers) of a HEAD request, also for an error answer."""
    try:
        with _open(url, "HEAD", follow) as response:
            return response.status, response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.headers
    except NET_ERRORS:
        return None, {}


def _size(value: Any) -> int | None:
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def listed_bytes(url: str) -> int | None:
    """The size Hugging Face's file listing gives for a ``resolve`` URL."""
    match = HF_FILE.fullmatch(url.split("?")[0])
    if not match:
        return None
    repo, revision, path = match.groups()
    listing = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    if "/" in path:
        listing += "/" + urllib.parse.quote(path.rsplit("/", 1)[0])
    try:
        with urllib.request.urlopen(listing, timeout=60) as response:
            rows = json.load(response)
    except (*NET_ERRORS, ValueError):
        return None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("path") == path:
            size = row.get("size")
            return size if isinstance(size, int) and size > 0 else None
    return None


def _bytes_after(url: str, status: int | None, headers: Any) -> int | None:
    """The size of a download, given the answer to its first HEAD request."""
    size = _size(headers.get("X-Linked-Size"))
    if size is None and status is not None and 300 <= status < 400:
        status, headers = _head(url, follow=True)
    if size is None and status == 200:
        size = _size(headers.get("Content-Length"))
    return size if size is not None else listed_bytes(url)


def published_bytes(url: str) -> int | None:
    """The size of a download in bytes, or None when no host says it.

    Hugging Face names the size of an LFS file on its redirect
    (``X-Linked-Size``). Any other file answers with its ``Content-Length``,
    once the redirects are followed. A gated Hugging Face file answers neither
    without a token: its size comes from the repository's file listing, which
    is public.
    """
    return _bytes_after(url, *_head(url, follow=False))


def hash_download(url: str) -> tuple[str, int] | None:
    """(sha256, bytes) of a file, by downloading it once. None when that fails.

    A host that answers with a web page did not send the file: it sent its
    sign-in page, or the page of the file where the download link belongs.
    The model sync would fetch the same page, and its sha256 would match.
    A download that ends before the length the host named is no file either.
    """
    digest, size = hashlib.sha256(), 0
    try:
        with _open(url, "GET", follow=True) as response:
            if response.headers.get_content_type() == "text/html":
                raise ConversionError(
                    f"{url} answers with a web page, not with the file: a sign-in page, "
                    "or the page of the file where its download link belongs"
                )
            named = _size(response.headers.get("Content-Length"))
            while chunk := response.read(HASH_CHUNK):
                digest.update(chunk)
                size += len(chunk)
    except NET_ERRORS:
        return None
    if not size or named not in (None, size):
        return None
    return digest.hexdigest(), size


def _head_on_host(url: str) -> tuple[str, int | None, Any]:
    """HEAD a URL, following the redirects that stay on its host. (url, status, headers)

    A renamed Hugging Face repository answers with a redirect to its new name,
    and only the answer after that carries ``X-Linked-ETag``. The redirect to
    the storage host is not followed: the answer there has neither header.
    """
    for _ in range(5):
        status, headers = _head(url, follow=False)
        target = headers.get("Location")
        if headers.get("X-Linked-ETag") or status is None or not 300 <= status < 400 or not target:
            break
        moved = urllib.parse.urljoin(url, target)
        if urllib.parse.urlsplit(moved).hostname != urllib.parse.urlsplit(url).hostname:
            break
        url = moved
    return url, status, headers


def file_facts(url: str) -> tuple[str | None, int | None]:
    """(sha256, bytes) of a download. Either is None when the host keeps it back.

    One HEAD request gives both for a file Hugging Face stores as LFS
    (``X-Linked-ETag``, ``X-Linked-Size``), one more for each time its
    repository was renamed. Where the host publishes no sha256 (a file that is
    not stored as LFS, another host), the file is downloaded once and hashed.

    Only a gated file comes back without its sha256: the host answers 401 or
    403, and the download ends the same way. A file that is not there, and a
    host that is busy or does not answer, stop the import. Read as "no sha256
    published", either would end in an entry without one.
    """
    final, status, headers = _head_on_host(url)
    etag = (headers.get("X-Linked-ETag") or "").strip('"')
    if re.fullmatch(r"[0-9a-f]{64}", etag):
        return etag, _bytes_after(final, status, headers)
    if status == 404:
        raise ConversionError(
            f"{url} is not there: the host answers 404"
            + (". The templates repository has no such file at that commit: name a commit "
               "that is on GitHub with --templates-ref"
               if url.startswith(f"{RAW}/{TEMPLATES_REPO}/") else "")
        )
    if status is None or status == 429 or status >= 500:
        raise ConversionError(
            f"{urllib.parse.urlsplit(url).hostname} did not answer for {url} "
            f"({'no answer' if status is None else status}): try again"
        )
    print(f"{url} publishes no sha256: downloading it once to hash it", file=sys.stderr)
    hashed = hash_download(url)
    if hashed is None and status in (401, 403):
        return None, _bytes_after(final, status, headers)
    if hashed is None:
        raise ConversionError(f"{url} could not be downloaded to hash it: try again")
    return hashed


def fill_bytes(manifest_path: Path, lookup=None) -> list[str]:
    """Add ``bytes`` to every models entry of the manifest that has none.

    The file is rewritten with only those keys added. ``lookup`` gives the
    size of a URL (default: ``published_bytes``). Returns the URLs whose size
    could not be found: their entries stay as they are, and the manifest lint
    names them.
    """
    lookup = lookup or published_bytes
    _, manifest = _read_manifest(manifest_path)
    sizes: dict[str, int | None] = {}
    for entry in manifest["workflows"].values():
        for model in entry.get("models") or []:
            if "bytes" in model:
                continue
            url = model["url"]
            if url not in sizes:
                sizes[url] = lookup(url)
            if sizes[url] is not None:
                model["bytes"] = sizes[url]
    _write(manifest_path, _dump(manifest))
    return sorted(url for url, size in sizes.items() if size is None)


# --------------------------------------------------------------------------- #
# the entry and the manifest
# --------------------------------------------------------------------------- #


def _seed_source(prompt: dict[str, dict[str, Any]], node_id: str, name: str) -> tuple[str, str]:
    """Where the number of a seed that a link feeds is held: (node id, input).

    A ``PrimitiveInt`` holds it as ``value``. Any other node is taken when it
    has one whole number among its literal inputs: that one is the seed.
    """
    source_id = prompt[node_id]["inputs"][name][0]
    source = prompt.get(source_id) or {"class_type": "a missing node", "inputs": {}}
    numbers = [key for key, value in source["inputs"].items()
               if isinstance(value, int) and not isinstance(value, bool)]
    held = ["value"] if "value" in numbers else numbers
    if len(held) != 1:
        raise ConversionError(
            f"node {node_id} ({prompt[node_id]['class_type']}) gets its {name} from node "
            f"{source_id} ({source['class_type']}), and the importer cannot tell which input "
            "of that node holds the number. The lint asks every workflow with a seed for a "
            "seed override: write the entry by hand"
        )
    return source_id, held[0]


def is_seed(name: str) -> bool:
    """Whether an input is a seed, by its name. A sub-input goes by its last
    part: ``sampling_mode.seed`` is the seed of a sampled text generation."""
    return name.rsplit(".", 1)[-1] in SEED_NAMES


def seed_overrides(prompt: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The seed inputs a run must pin, with the values the template already has.

    A seed that a link feeds is pinned where its number is (``_seed_source``):
    the lint asks every workflow that has a seed input for an override. A
    sub-input is named whole, as the workflow file has it: the runner writes
    an override over the input of that name, dot or no dot.
    """
    overrides: dict[str, dict[str, Any]] = {}
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if not is_seed(name):
                continue
            if isinstance(value, int):
                overrides.setdefault(node_id, {})[name] = value
            elif is_link(value):
                source_id, held = _seed_source(prompt, node_id, name)
                overrides.setdefault(source_id, {})[held] = prompt[source_id]["inputs"][held]
    return overrides


def model_folders() -> set[str]:
    """The models/ folders the worker searches on the volume: the keys of
    ``docker/extra_model_paths.yaml`` (flat ``key: value`` lines)."""
    text = (ROOT / "docker" / "extra_model_paths.yaml").read_text(encoding="utf-8")
    keys = set(re.findall(r"^\s+([a-z_]+):\s*\S+", text, flags=re.M))
    keys.discard("base_path")
    return keys


def manifest_files(manifest: dict[str, Any],
                   but: str | None = None) -> dict[str, tuple[str, dict[str, Any]]]:
    """directory/name -> (workflow id, models entry) of each file the manifest
    lists, the files of the workflow ``but`` left out."""
    files: dict[str, tuple[str, dict[str, Any]]] = {}
    for workflow_id, entry in (manifest.get("workflows") or {}).items():
        for model in (entry.get("models") or []) if workflow_id != but else []:
            files.setdefault(f"{model['directory']}/{model['name']}", (workflow_id, model))
    return files


def _as_the_manifest_has_it(model: dict[str, Any], owner: str, listed: dict[str, Any],
                            notes: list[str]) -> dict[str, Any]:
    """The manifest's own entry for a file it already lists, when it is that file.

    The volume holds one file per folder and name, and the lint wants one URL
    and one sha256 for it. The gallery serves the same file from several
    repositories, and an input file is pinned to another commit on each
    import. The sha256 decides, and the size where the manifest has no sha256.
    """
    key = f"{model['directory']}/{model['name']}"
    if model.get("sha256") and listed.get("sha256"):
        same, by = model["sha256"] == listed["sha256"], "sha256"
    elif model.get("bytes") and listed.get("bytes"):
        same, by = model["bytes"] == listed["bytes"], "size"
    else:
        same, by = model["url"] == listed["url"], None
    if same and model["url"] != listed["url"]:
        notes.append(f"note: {key} is in the manifest for {owner}, and it is the same file "
                     f"by its {by}: the entry takes the URL it has there, {listed['url']}")
    if same:
        return dict(listed)
    if by is None:
        raise ConversionError(
            f"{key} is in the manifest for {owner} from {listed['url']}, and neither a "
            f"sha256 nor a size says whether {model['url']} is the same file: a volume "
            "holds one file per name"
        )
    if model["url"] == listed["url"]:
        raise ConversionError(
            f"{key} is in the manifest for {owner} with another {by}: the file at "
            f"{model['url']} changed since. Correct the manifest first"
        )
    raise ConversionError(
        f"{key} is another file in the manifest, for {owner}, by its {by}: a volume holds "
        f"one file per name. The manifest has {listed['url']}, the template "
        f"{model['url']}"
    )


def build_entry(
    workflow_id: str,
    prompt: dict[str, dict[str, Any]],
    template: dict[str, Any],
    *,
    cadence: str,
    kind: str = "image",
    level: str = "smoke",
    expect: dict[str, Any] | None = None,
    timeout_s: int | None = None,
    skip_sha: bool = False,
    pinned: Callable[[], str] | None = None,
    manifest: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """The manifest entry of a finished prompt.

    ``pinned`` gives the commit of the templates repository the input files
    are read at. It is asked only when the workflow has input files.
    ``manifest`` is the manifest the entry will join. What the lint would
    refuse in the new entry is refused here, before anything is written: a
    model folder the worker does not search, a file the manifest has as
    another file, a seed nothing pins. ``notes`` gets what the author should
    know.
    """
    notes = [] if notes is None else notes
    models = model_files(prompt, template)
    searched = model_folders()
    unmapped = [m for m in models if m["directory"] not in searched]
    if unmapped:
        raise ConversionError(
            "the worker does not search "
            + ", ".join(f"{m['directory']}/ ({m['name']})" for m in unmapped)
            + ": docker/extra_model_paths.yaml names the models/ folders it does"
        )
    inputs = input_names(prompt)
    if inputs:
        if pinned is None:
            raise ConversionError(
                f"the workflow reads {', '.join(inputs)} from the input folder, and no "
                "commit of the templates repository is known to pin them to"
            )
        sha = pinned()
        models += [{"name": name, "url": input_url(name, sha), "directory": "input"}
                   for name in inputs]
    seeds = seed_overrides(prompt)
    known = manifest_files(manifest or {}, but=workflow_id)
    for index, model in enumerate(models):
        key = f"{model['directory']}/{model['name']}"
        owner, listed = known.get(key, (None, None))
        if skip_sha:
            if listed and listed["url"] != model["url"]:
                notes.append(f"note: {key} is in the manifest for {owner} from another URL, "
                             "and --skip-sha looked nothing up to say whether it is the "
                             "same file")
            continue
        digest, size = file_facts(model["url"])
        if model["directory"] == "input" and digest is None:
            raise ConversionError(
                f"{model['name']} could not be downloaded from {model['url']}: the "
                "templates repository has no such input file at that commit. Name a "
                "commit that is on GitHub with --templates-ref"
            )
        if digest:
            model["sha256"] = digest
        if size:
            model["bytes"] = size
        if listed:
            models[index] = _as_the_manifest_has_it(model, owner, listed, notes)

    entry: dict[str, Any] = {"enabled": True, "level": level, "cadence": cadence}
    if kind != "image":
        entry["output_kind"] = kind
    if expect:
        entry["expect"] = expect
    entry["workflow_path"] = f"manifest/workflows/{workflow_id}_api.json"
    entry["seed_overrides"] = seeds
    entry["models"] = models
    if inputs:
        entry["comfy_flags"] = ["--input-directory", INPUT_DIR]
    entry.update(gpu_type=None, timeout_s=timeout_s, thresholds=None, golden_tag=None)
    return entry


def cadence_problem(cadence: str, lanes: dict[str, Any]) -> str | None:
    """Why no workflow can get this cadence today, read from the lane registry."""
    running = [lane for lane in (lanes.get("lanes") or {}).values()
               if lane.get("cadence") == cadence]
    if not running:
        return f"manifest/lanes.json has no {cadence} lane"
    if cadence == "weekly" and not any(lane.get("endpoint_id") for lane in running):
        return "the weekly lane is not switched on yet: use --cadence nightly"
    return None


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _line_end(path: Path) -> str:
    """The line end a file has: a rewrite that changed it would change every line."""
    with open(path, "rb") as handle:
        return "\r\n" if handle.readline().endswith(b"\r\n") else "\n"


def _write(path: Path, text: str, like: Path | None = None) -> None:
    """Write a file in one step, with the line ends of ``like`` (default: its own).

    The text goes to a file next to it, which then takes its place. A write
    that fails, on a full disk say, leaves the old file whole.
    """
    part = path.with_name(path.name + ".part")
    try:
        with open(part, "w", encoding="utf-8", newline=_line_end(like or path)) as handle:
            handle.write(text)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _layout_problem(manifest_path: Path, text: str, manifest: Any) -> str | None:
    """Why this command will not rewrite the manifest, if it will not."""
    if _dump(manifest) == text:
        return None
    return (f"{manifest_path} is not laid out the way this command writes it "
            "(2 spaces, one key a line): rewriting it would change more than the lines "
            "this command adds")


def _read_manifest(manifest_path: Path) -> tuple[str, dict[str, Any]]:
    """The manifest's text and what it holds, when writing it back changes no
    line but the new ones."""
    text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    problem = _layout_problem(manifest_path, text, manifest)
    if problem:
        raise SystemExit(problem)
    return text, manifest


def insert_entry(manifest_path: Path, workflow_id: str, entry: dict[str, Any],
                 seen: str | None = None) -> None:
    """Add the entry at the end of ``workflows``. No other line changes.

    ``seen`` is the manifest's text as the command read it when it started.
    The file is read again here. When it changed in between, another import
    wrote to it, and writing back what this command read would drop that
    entry: the file is left alone.
    """
    text, manifest = _read_manifest(manifest_path)
    if seen is not None and text != seen:
        raise SystemExit(
            f"{manifest_path.name} changed while this command ran, so it was left as it "
            "is: run the command again"
        )
    if workflow_id in manifest["workflows"]:
        raise SystemExit(f"{workflow_id} is already in {manifest_path.name}")
    manifest["workflows"][workflow_id] = entry
    _write(manifest_path, _dump(manifest))


def run_lint() -> int:
    """Run the whole manifest lint. The rules about lanes, shared model files
    and the volume are not tests of one workflow id, so no ``-k`` narrows it."""
    if importlib.util.find_spec("pytest") is None:
        print(
            "error: the manifest lint did not run, because pytest is not installed. "
            "Install pytest (pip install pytest), then run "
            "python -m pytest scripts/tests/test_manifest_lint.py",
            file=sys.stderr,
        )
        return 1
    lint = ROOT / "scripts" / "tests" / "test_manifest_lint.py"
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(lint), "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
    ).returncode


# --------------------------------------------------------------------------- #
# the server, and the templates repository
# --------------------------------------------------------------------------- #


def _get(url: str, accept: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _post(url: str, body: dict[str, Any]) -> tuple[int, bytes]:
    """(status, body) of a JSON POST, also for an error answer."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _missing_file(error: dict[str, Any], node: dict[str, Any], files: set[str],
                  filled: set[tuple[str, str]]) -> bool:
    """Whether a node error only says that the server does not have a file.

    A combo answers "value not in list". ``LoadImage``, ``LoadAudio`` and
    ``LoadVideo`` check their file themselves and answer "custom validation
    failed" for the same thing.
    """
    info = error.get("extra_info")
    name = info.get("input_name") if isinstance(info, dict) else None
    if not isinstance(name, str):
        return False
    class_type = node["class_type"]
    value = node["inputs"].get(name)
    named = isinstance(value, str) and ANNOTATED.sub("", value) in files
    if error.get("type") == "value_not_in_list":
        return named or (class_type, name) in filled
    if error.get("type") == "custom_validation_failed":
        return named and INPUT_FILE_BY_NODE.get(class_type) == name
    return False


def _take_out(server: str, prompt_id: str) -> str:
    """Take an accepted prompt out of a server's queue, and say how that went.

    Out of the queue first, then stopped in case it had already started. Both
    are asked, whatever the first one answers, and each answer is read: a
    prompt that stays would run on the author's machine.
    """
    failed = []
    for path, body in (("/queue", {"delete": [prompt_id]}),
                       ("/interrupt", {"prompt_id": prompt_id})):
        try:
            status = _post(server + path, body)[0]
        except NET_ERRORS as error:
            failed.append(f"{path}: {error}")
        else:
            if status != 200:
                failed.append(f"{path} answered {status}")
    if failed:
        return (f"prompt check: the server accepted the prompt as {prompt_id}, and taking "
                f"it out of the queue failed ({', '.join(failed)}): cancel it by hand")
    return "prompt check: the server accepted the prompt, and it was taken out of the queue"


def check_prompt(server: str, prompt: dict[str, dict[str, Any]], files: set[str],
                 filled: set[tuple[str, str]]) -> tuple[bool, list[str]]:
    """Post the prompt to a ComfyUI and read what it says. (passed, lines to print)

    It passes when ComfyUI names no node error, or only errors that say the
    server does not have a file of the entry. A prompt the server accepts is
    taken out of its queue at once, so nothing is generated there. An answer
    that is not what ComfyUI sends is a failed check, with what was sent.
    """
    server = server.rstrip("/")
    try:
        status, body = _post(f"{server}/prompt", {"prompt": prompt})
        answer = json.loads(body)
    except (*NET_ERRORS, ValueError) as error:
        return False, [f"prompt check: no answer from {server}/prompt ({error})"]
    if not isinstance(answer, dict):
        return False, [f"prompt check: {server}/prompt answered {status} with no object"]

    problems, absent = [], []
    node_errors = answer.get("node_errors") or {}
    if not isinstance(node_errors, dict):
        problems.append(f"  node_errors: {node_errors!r}")
        node_errors = {}
    for node_id, record in node_errors.items():
        errors = (record.get("errors") or []) if isinstance(record, dict) else None
        if not isinstance(errors, list):
            problems.append(f"  node {node_id}: {record!r}")
            continue
        node = prompt.get(node_id) or {"class_type": record.get("class_type"), "inputs": {}}
        for error in errors:
            if not isinstance(error, dict):
                problems.append(f"  node {node_id} ({node['class_type']}): {error!r}")
            elif _missing_file(error, node, files, filled):
                absent.append(str(node["inputs"].get(error["extra_info"]["input_name"])))
            else:
                problems.append(
                    f"  node {node_id} ({node['class_type']}): {error.get('message')}: "
                    f"{error.get('details')}"
                )
    lines = []
    prompt_id = answer.get("prompt_id")
    if prompt_id and not isinstance(prompt_id, str):
        problems.append(f"  {server}/prompt answered {status} with a prompt id that is no "
                        f"text ({prompt_id!r}): if the server queued it, cancel it by hand")
        prompt_id = None
    if prompt_id:
        lines.append(_take_out(server, prompt_id))
    if not prompt_id and not problems and not absent:
        # No node is named: an unknown node class, a prompt without outputs.
        refused = answer.get("error")
        if isinstance(refused, dict):
            problems.append(
                f"  {refused.get('message')}: {refused.get('details') or ''}".rstrip(": "))
        else:
            problems.append(f"  {server}/prompt answered {status}: {refused or 'no prompt id'}")
    if problems:
        return False, lines + ["prompt check: ComfyUI refuses the prompt"] + problems
    if absent:
        lines.append(
            f"prompt check: passed. The server does not have {', '.join(sorted(set(absent)))}, "
            "and names no other problem"
        )
    return True, lines or ["prompt check: passed"]


def resolve_ref(ref: str) -> str:
    """The full sha of a branch, tag or short sha of the templates repository."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    url = f"https://api.github.com/repos/{TEMPLATES_REPO}/commits/{urllib.parse.quote(ref)}"
    try:
        sha = _get(url, accept="application/vnd.github.sha").decode().strip()
    except (*NET_ERRORS, ValueError) as error:
        raise ConversionError(
            f"could not resolve {ref} in {TEMPLATES_REPO} ({error}): pass a full commit "
            "sha with --templates-ref"
        ) from None
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ConversionError(f"GitHub did not answer a commit sha for {ref}")
    return sha


def checkout_sha(templates_dir: Path) -> str | None:
    """The commit of a workflow_templates checkout, when the folder is its
    ``templates/`` folder. None for any other folder."""
    try:
        answer = subprocess.run(
            ["git", "-C", str(templates_dir), "rev-parse", "--show-prefix", "HEAD"],
            capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    if len(answer) == 2 and answer[0] == "templates/" and re.fullmatch(r"[0-9a-f]{40}", answer[1]):
        return answer[1]
    return None


def load_template(name: str, templates_dir: Path | None, pinned: Callable[[], str]) -> dict:
    """The template's graph, from a checkout or from GitHub at the pinned commit."""
    try:
        if templates_dir is not None:
            text = (templates_dir / f"{name}.json").read_text(encoding="utf-8")
        else:
            url = f"{RAW}/{TEMPLATES_REPO}/{pinned()}/templates/{urllib.parse.quote(name)}.json"
            text = _get(url).decode("utf-8")
        template = json.loads(text)
    except (*NET_ERRORS, ValueError) as error:
        raise ConversionError(f"could not read the template {name}: {error}") from None
    if not isinstance(template, dict) or not isinstance(template.get("nodes"), list):
        raise ConversionError(f"{name} is not a workflow template: it has no nodes")
    return template


def _defines_nodes(info: Any) -> bool:
    """Whether a value has the shape of ``/object_info``: an object of node
    classes, each with its inputs by name."""
    if not isinstance(info, dict):
        return False
    for spec in info.values():
        inputs = (spec.get("input") or {}) if isinstance(spec, dict) else None
        if not isinstance(inputs, dict) or not all(
                isinstance(inputs.get(part) or {}, dict) for part in ("required", "optional")):
            return False
    return True


def load_object_info(server: str | None, path: Path | None) -> dict[str, Any]:
    source = f"{server}/object_info" if server else path
    try:
        if server:
            info = json.loads(_get(f"{server.rstrip('/')}/object_info"))
        else:
            info = json.loads(path.read_text(encoding="utf-8"))
    except (*NET_ERRORS, ValueError) as error:
        raise ConversionError(
            f"could not read the node definitions from {source}: {error}") from None
    if not _defines_nodes(info):
        raise ConversionError(
            f"{source} holds no node definitions: /object_info is an object with one "
            "entry for each node class")
    return info


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #


def expectations(values: list[str], level: str) -> dict[str, Any]:
    """The entry's ``expect`` from the ``--expect key=number`` flags."""
    expect: dict[str, Any] = {}
    for text in values:
        key, _, raw = text.partition("=")
        try:
            expect[key] = _json_value(raw)
        except ValueError:
            raise ConversionError(f"--expect {text}: write <key>=<number>") from None
    try:
        lanes_mod.expect_of({"level": level, "expect": expect or None})
    except ValueError as error:
        raise ConversionError(f"--expect: {error}") from None
    return expect


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("template", nargs="?", help="Template name, e.g. image_pixeldit_t2i")
    parser.add_argument("--id", help="Workflow id in the manifest (default: the template name)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--server",
        help="A running ComfyUI, e.g. http://127.0.0.1:8188. Its /object_info drives the "
        "conversion and it checks the finished prompt. It needs no model or input file.",
    )
    source.add_argument(
        "--object-info",
        type=Path,
        help="A saved /object_info dump, where no server runs. The prompt check is skipped.",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        help="templates/ directory of a Comfy-Org/workflow_templates checkout "
        "(default: fetch the template from GitHub)",
    )
    parser.add_argument(
        "--templates-ref",
        help="Branch, tag or commit of the templates repository to read the template "
        "and pin its input files at (default: main, or the commit of --templates-dir)",
    )
    parser.add_argument(
        "--cadence",
        choices=lanes_mod.WORKFLOW_CADENCES,
        help="How often the workflow runs. Required: it decides what the workflow costs "
        "and where its models live.",
    )
    parser.add_argument(
        "--level",
        choices=LEVELS,
        default="smoke",
        help="smoke: it has to finish and write usable files. reference: its output is "
        "compared with an approved reference. (default: smoke)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NODE.INPUT=VALUE",
        help="Override a literal input, with a JSON value. For what a CI run makes "
        "smaller: 30 s of audio, 25 frames. Repeatable.",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="KEY=NUMBER",
        help=f"What a smoke-level output must be ({', '.join(lanes_mod.EXPECT_KEYS)}). "
        "Repeatable.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        help="The entry's timeout_s (default: null, the manifest default)",
    )
    parser.add_argument(
        "--keep-sampling",
        action="store_true",
        help="Leave text generation as the template has it, sampled. Smoke level only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the converted nodes, the entry and the result of the prompt check. "
        "Write nothing.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_entry",
        help="Write the workflow file and print the entry. Leave the manifest alone "
        "and run no lint.",
    )
    parser.add_argument(
        "--skip-sha",
        action="store_true",
        help="Do not look up sha256 checksums and sizes (no request to a model host).",
    )
    parser.add_argument(
        "--fill-bytes",
        action="store_true",
        help="Import nothing. Ask each download URL of manifest/workflows.json for its "
        "size and add it as `bytes` to every models entry that has none.",
    )
    args = parser.parse_args(argv)
    if args.fill_bytes:
        return args
    missing = [flag for flag, value in (
        ("template", args.template),
        ("--server or --object-info", args.server or args.object_info),
        ("--cadence", args.cadence),
    ) if not value]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def early_refusal(args: argparse.Namespace, workflow_id: str, seen: str,
                  manifest: dict[str, Any]) -> str | None:
    """What stops an import before a template is read.

    ``seen`` is the text of the manifest and ``manifest`` what it holds. What
    the command would find only when it writes is found here: a manifest it
    will not rewrite, and a file or an id that is taken. On Windows and macOS
    a name that differs by case only is the same file.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", workflow_id):
        return (f"the id {workflow_id!r} names the output files, so it is made of letters, "
                "digits, _ . and -: choose one with --id")
    problem = cadence_problem(args.cadence, lanes_mod.load_lanes(ROOT / "manifest" / "lanes.json"))
    if problem:
        return problem
    if args.timeout_s is not None and not 0 < args.timeout_s < 3600:
        return "--timeout-s is above 0 and under 3600, the endpoint's limit for a job"
    if args.keep_sampling and args.level != "smoke":
        return ("--keep-sampling is for a smoke-level workflow: a reference-level output is "
                "compared with its reference, so its text is generated without sampling")
    if not args.print_entry:
        problem = _layout_problem(ROOT / "manifest" / "workflows.json", seen, manifest)
        if problem:
            return problem
    if workflow_id in manifest["workflows"] and not args.print_entry:
        return f"{workflow_id} is already in manifest/workflows.json: choose another --id"
    twin = next((other for other in manifest["workflows"]
                 if other != workflow_id and other.casefold() == workflow_id.casefold()), None)
    if twin:
        return (f"{twin} is in manifest/workflows.json, and {workflow_id} differs from it "
                "by case only: on Windows and macOS both would be one workflow file. "
                "Choose another --id")
    name = f"{workflow_id}_api.json"
    folder = ROOT / "manifest" / "workflows"
    there = sorted(p.name for p in folder.iterdir()
                   if p.name.casefold() == name.casefold()) if folder.is_dir() else []
    if there and (there != [name] or not args.print_entry):
        return (f"manifest/workflows/{there[0]} is already there, and the command writes "
                "over no workflow file: choose another --id, or delete the file first")
    if not (args.dry_run or args.print_entry) and importlib.util.find_spec("pytest") is None:
        return ("the manifest lint is run with pytest, which is not installed: "
                "pip install pytest")
    return None


def main(argv: list[str] | None = None) -> int:
    # A prompt holds any text a template has. A console that cannot show a
    # character prints its escape and does not stop the command.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    if args.fill_bytes:
        unknown = fill_bytes(ROOT / "manifest" / "workflows.json")
        for url in unknown:
            print(
                f"error: no size found for {url}: add its bytes by hand",
                file=sys.stderr,
            )
        return 1 if unknown else 0

    workflow_id = args.id or args.template
    manifest_path = ROOT / "manifest" / "workflows.json"
    seen = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(seen)
    refusal = early_refusal(args, workflow_id, seen, manifest)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 1

    shas: list[str] = []

    def pinned() -> str:
        """The commit of the templates repository, found once and only when asked."""
        if not shas:
            local = args.templates_dir and not args.templates_ref and checkout_sha(
                args.templates_dir)
            shas.append(local or resolve_ref(args.templates_ref or "main"))
        return shas[0]

    # A template this importer cannot turn into a corpus entry is a refusal
    # with its reason, not a traceback. So is a file that is JSON and no
    # workflow: the converter reads what the frontend writes, and trips over
    # anything else.
    shown: list[dict] = []
    try:
        expect = expectations(args.expect, args.level)
        object_info = load_object_info(args.server, args.object_info)
        template = load_template(args.template, args.templates_dir, pinned)
        prompt, kind, filled, notes = finish_prompt(
            template, object_info, workflow_id, args.set, args.keep_sampling, args.level,
            shown)
        entry = build_entry(
            workflow_id, prompt, template, cadence=args.cadence, kind=kind, level=args.level,
            expect=expect, timeout_s=args.timeout_s, skip_sha=args.skip_sha, pinned=pinned,
            manifest=manifest, notes=notes)
    except (ConversionError, AttributeError, IndexError, KeyError, TypeError, ValueError,
            RecursionError) as error:
        said = str(error) if isinstance(error, ConversionError) else (
            "the template is not a workflow the converter can read "
            f"({type(error).__name__}: {error})")
        print(f"error: {said}", file=sys.stderr)
        if shown:
            print(f"nodes ({len(shown[0])}):\n{describe(shown[0])}", file=sys.stderr)
        return 1

    listed = manifest_files(manifest, but=workflow_id)
    for model in entry["models"]:
        lacks = [key for key in ("sha256", "bytes") if key not in model]
        if listed.get(f"{model['directory']}/{model['name']}", ("", None))[1] == model:
            continue  # written as the manifest has it, with what the manifest knows
        if lacks and not args.skip_sha:
            notes.append(
                f"note: no {' and no '.join(lacks)} found for {model['url']}"
                + ("" if os.environ.get("HF_TOKEN") else
                   " (a gated repository answers only with HF_TOKEN set)"))
    lanes = lanes_mod.load_lanes(ROOT / "manifest" / "lanes.json")["lanes"]
    if not any(lane.get("cadence") == args.cadence and lane.get("endpoint_id")
               for lane in lanes.values()):
        notes.append(f"note: no {args.cadence} lane has an endpoint yet, so nothing runs the "
                     "workflow until one has")
    if args.server:
        passed, lines = check_prompt(
            args.server, prompt, {model["name"] for model in entry["models"]}, filled)
    else:
        passed, lines = True, ["prompt check: skipped, there is no --server to ask"]

    if args.dry_run:
        print(f"nodes ({len(prompt)}):\n{describe(prompt)}")
        print(f"entry:\n{json.dumps({workflow_id: entry}, indent=2, ensure_ascii=False)}")
        print("\n".join(notes + lines))
        print("dry run: nothing was written")
        return 0 if passed else 1
    print("\n".join(notes + lines), file=sys.stderr)
    if not passed:
        return 1

    path = ROOT / "manifest" / "workflows" / f"{workflow_id}_api.json"
    try:
        _write(path, _dump(prompt), like=manifest_path)
        if not args.print_entry:
            try:
                insert_entry(manifest_path, workflow_id, entry, seen)
            except BaseException:
                path.unlink(missing_ok=True)  # an entry that did not go in leaves no file
                raise
    except (OSError, SystemExit) as error:
        print(f"error: nothing was written: {error}", file=sys.stderr)
        return 1
    print(f"wrote {path.relative_to(ROOT).as_posix()} ({len(prompt)} nodes)", file=sys.stderr)
    if args.print_entry:
        print(json.dumps({workflow_id: entry}, indent=2, ensure_ascii=False))
        return 0
    print(f"added {workflow_id} to manifest/workflows.json", file=sys.stderr)
    failed = run_lint()
    if failed:
        print(
            "error: the manifest lint does not pass with the new entry. Fix the entry or "
            f"the workflow file, or take both back: remove {workflow_id} from "
            f"manifest/workflows.json and delete {path.relative_to(ROOT).as_posix()}",
            file=sys.stderr,
        )
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
