#!/usr/bin/env python3
"""Turn a workflow template into a corpus entry: one workflow file, one manifest row.

Does the chores the README's "Adding a workflow" section used to spell out by
hand: convert the template to API format, pick the branch a constant switch
selects, collapse the preview nodes a corpus run has no use for, normalise the
output to a ``SaveImage`` PNG named after the workflow id, and read the model
download URLs out of the template's own model-links note (ComfyUI publishes a
sha256 for most of them, so the entry can pin one).

    python scripts/import_template.py image_pixeldit_t2i \
        --templates-dir ../workflow_templates/templates \
        --object-info object_info.json \
        --id pixeldit_t2i

Writes ``manifest/workflows/<id>_api.json`` and prints the manifest entry to
paste under ``workflows`` in ``manifest/workflows.json``. It does not edit the
manifest: the entry still needs a ``timeout_s`` chosen for the model, and lands
``"enabled": false`` until a golden exists (see the README).

It writes image workflows only. A template whose result is text (it ends in
``PreviewAny`` or ``SaveText``) is refused with what to do by hand: the README's
"Adding a workflow" section has the rules for a text workflow.

Get the ``/object_info`` dump from the ComfyUI version the corpus runs against,
with the workflow's models present so their combo values resolve:

    curl -s http://127.0.0.1:8188/object_info > object_info.json

The import is only as good as its verification. After writing the workflow,
submit it to a local ComfyUI to confirm the server accepts the prompt:

    curl -sS -X POST http://127.0.0.1:8188/prompt \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\": $(cat manifest/workflows/<id>_api.json)}"

A 200 means every node, input name, widget value and link survived the
conversion; the run itself can then fail on missing weights without telling you
anything new.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from template_api_export import (  # noqa: E402
    ConversionError,
    Converter,
    OUTPUT_PREFIXES,
    PREVIEW_PREFIXES,
)

ROOT = Path(__file__).resolve().parents[1]

SAVE_IMAGE = "SaveImage"
# What a text template ends in: a preview that shows the text (and writes no
# file) or a node that saves it.
TEXT_SINKS = ("PreviewAny", "SaveText")
MODEL_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# A file of a Hugging Face model repository: repository, revision, path.
HF_FILE = re.compile(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)")

# ComfyUI's models/ subdirectory for each loader input, so the entry says where
# the worker must put the file. Keyed by loader *and* input name, because the
# name alone is ambiguous: CLIPLoader.clip_name reads text_encoders/ while
# CLIPVisionLoader.clip_name reads clip_vision/.
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


def refuse_text_output(prompt: dict[str, dict[str, Any]], workflow_id: str) -> None:
    """Stop on a template whose result is text, before its previews are dropped.

    Such a template saves no image, video, audio or 3D file and ends in a
    ``PreviewAny`` nothing reads, or in a ``SaveText``. Collapsing the previews
    would leave it with no output at all, and "expected exactly one SaveImage"
    would not say why. A text workflow is compared as text (``"output_kind":
    "text"``), which this importer does not write. A video or audio template
    with a text preview on the side is not a text template: it keeps the
    ``SaveImage`` refusal.
    """
    if any(node["class_type"].startswith(OUTPUT_PREFIXES) for node in prompt.values()):
        return
    read = {
        value[0]
        for node in prompt.values()
        for value in node["inputs"].values()
        if isinstance(value, list) and len(value) == 2
    }
    sinks = [
        f"{node['class_type']} node {node_id}"
        for node_id, node in prompt.items()
        if node["class_type"] in TEXT_SINKS and node_id not in read
    ]
    if sinks:
        raise ConversionError(
            f"the template's output is text ({', '.join(sinks)}), and this importer "
            "only writes image workflows. Convert it with template_api_export.py and "
            "finish the file by hand: end it in SaveText with filename_prefix "
            f'"{workflow_id}" and format "txt", make text generation greedy '
            '(sampling_mode "off", mtp "off"), and add "output_kind": "text" to its '
            'manifest entry. See "Adding a workflow" in the README.'
        )


def collapse_previews(prompt: dict[str, dict[str, Any]]) -> None:
    """Drop preview nodes, forwarding the value of the ones that pass through.

    ``PreviewAny`` sits mid-graph in several templates, showing a string on its
    way to a text encoder. It is an output node, so leaving it in would make it
    a second thing the run executes and a node that feeds no SaveImage.
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


def normalise_save(prompt: dict[str, dict[str, Any]], workflow_id: str) -> None:
    """Make the single output a ``SaveImage`` PNG prefixed with the workflow id.

    Comparison reads PNGs saved under the workflow id, and templates reach that
    through either ``SaveImage`` or ``SaveImageAdvanced`` (whose png/8-bit/sRGB
    default writes the same file). Rewriting the advanced node keeps every
    workflow in the corpus on one output shape.
    """
    saves = [
        node_id
        for node_id, node in prompt.items()
        if node["class_type"].startswith("SaveImage")
    ]
    if len(saves) != 1:
        raise ConversionError(
            f"expected exactly one SaveImage output, found {len(saves)}: {saves}"
        )
    node = prompt[saves[0]]
    images = node["inputs"].get("images")
    if images is None:
        raise ConversionError(f"save node {saves[0]} has no images input")
    written = (
        node["inputs"].get("format", "png"),
        node["inputs"].get("format.bit_depth", "8-bit"),
        node["inputs"].get("format.input_color_space", "sRGB"),
    )
    if written != ("png", "8-bit", "sRGB"):
        raise ConversionError(
            f"save node {saves[0]} writes {'/'.join(map(str, written))}, but "
            "comparison needs 8-bit PNG in sRGB: rewriting it to SaveImage would "
            "silently drop that setting and compare a file the template does not "
            "produce"
        )
    node["class_type"] = SAVE_IMAGE
    node["inputs"] = {"filename_prefix": workflow_id, "images": images}


def loaded_files(prompt: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Model file name -> the models/ subdirectory it belongs in."""
    found: dict[str, str] = {}
    for node in prompt.values():
        class_type = node["class_type"]
        for name, value in node["inputs"].items():
            if not isinstance(value, str) or not value.endswith(MODEL_SUFFIXES):
                continue
            directory = DIRECTORY_BY_LOADER_INPUT.get((class_type, name))
            if directory is None:
                raise ConversionError(
                    f"{class_type}.{name} loads {value}, but which models/ "
                    "subdirectory it lives in is not known: add "
                    f"(\"{class_type}\", \"{name}\") to DIRECTORY_BY_LOADER_INPUT"
                )
            found[value] = directory
    return found


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face only sends X-Linked-ETag on the redirect, not on the CDN hit."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def published_sha256(url: str) -> str | None:
    """The sha256 Hugging Face publishes for an LFS file, when it publishes one."""
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(request, timeout=60) as response:
            etag = (response.headers.get("X-Linked-ETag") or "").strip('"')
    except urllib.error.HTTPError as error:
        etag = (error.headers.get("X-Linked-ETag") or "").strip('"')
    except OSError:
        return None
    return etag if re.fullmatch(r"[0-9a-f]{64}", etag) else None


def _head(url: str, follow: bool) -> tuple[int | None, Any]:
    """(status, headers) of a HEAD request, also for an error answer."""
    opener = urllib.request.build_opener(*([] if follow else [_NoRedirect]))
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(request, timeout=60) as response:
            return response.status, response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.headers
    except OSError:
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
    except (OSError, ValueError):
        return None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("path") == path:
            size = row.get("size")
            return size if isinstance(size, int) and size > 0 else None
    return None


def published_bytes(url: str) -> int | None:
    """The size of a download in bytes, or None when no host says it.

    Hugging Face names the size of an LFS file on its redirect
    (``X-Linked-Size``). Any other file answers with its ``Content-Length``,
    once the redirects are followed. A gated Hugging Face file answers neither
    without a token: its size comes from the repository's file listing, which
    is public.
    """
    status, headers = _head(url, follow=False)
    size = _size(headers.get("X-Linked-Size"))
    if size is None and status is not None and 300 <= status < 400:
        status, headers = _head(url, follow=True)
    if size is None and status == 200:
        size = _size(headers.get("Content-Length"))
    return size if size is not None else listed_bytes(url)


def fill_bytes(manifest_path: Path, lookup=None) -> list[str]:
    """Add ``bytes`` to every models entry of the manifest that has none.

    The file is rewritten with only those keys added. ``lookup`` gives the
    size of a URL (default: ``published_bytes``). Returns the URLs whose size
    could not be found: their entries stay as they are, and the manifest lint
    names them.
    """
    lookup = lookup or published_bytes
    text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    if json.dumps(manifest, indent=2, ensure_ascii=False) + "\n" != text:
        raise SystemExit(
            f"{manifest_path} is not laid out the way this command writes it "
            "(2 spaces, one key a line): rewriting it would change more than the sizes"
        )
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sorted(url for url, size in sizes.items() if size is None)


def seed_overrides(prompt: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The seed inputs a run must pin, with the values the template already has."""
    overrides: dict[str, dict[str, Any]] = {}
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if name in ("seed", "noise_seed") and isinstance(value, int):
                overrides.setdefault(node_id, {})[name] = value
    return overrides


def build_entry(
    workflow_id: str,
    prompt: dict[str, dict[str, Any]],
    template: dict[str, Any],
    skip_sha: bool,
) -> dict[str, Any]:
    urls = note_urls(template)
    models = []
    for name, directory in sorted(loaded_files(prompt).items()):
        url = urls.get(name)
        if url is None:
            raise ConversionError(
                f"{name} is loaded by the workflow but no download link for it "
                "appears in the template's notes: add the models entry by hand"
            )
        model: dict[str, Any] = {"name": name, "url": url, "directory": directory}
        if not skip_sha:
            sha = published_sha256(url)
            if sha:
                model["sha256"] = sha
            size = published_bytes(url)
            if size:
                model["bytes"] = size
        models.append(model)

    return {
        "enabled": False,
        "workflow_path": f"manifest/workflows/{workflow_id}_api.json",
        "seed_overrides": seed_overrides(prompt),
        "models": models,
        "gpu_type": None,
        "timeout_s": None,
        "thresholds": None,
        "golden_tag": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("template", nargs="?", help="Template name, e.g. image_pixeldit_t2i")
    parser.add_argument(
        "--templates-dir",
        type=Path,
        help="templates/ directory of a Comfy-Org/workflow_templates checkout",
    )
    parser.add_argument(
        "--object-info",
        type=Path,
        help="ComfyUI /object_info dump taken with the workflow's models present",
    )
    parser.add_argument(
        "--id",
        help="Workflow id in the manifest (default: the template name)",
    )
    parser.add_argument(
        "--skip-sha",
        action="store_true",
        help="Do not look up published sha256 checksums and sizes (no network).",
    )
    parser.add_argument(
        "--fill-bytes",
        action="store_true",
        help="Import nothing. Ask each download URL of manifest/workflows.json for its "
        "size and add it as `bytes` to every models entry that has none.",
    )
    args = parser.parse_args()

    if args.fill_bytes:
        unknown = fill_bytes(ROOT / "manifest" / "workflows.json")
        for url in unknown:
            print(
                f"error: no size found for {url}: add its bytes by hand",
                file=sys.stderr,
            )
        return 1 if unknown else 0
    if not (args.template and args.templates_dir and args.object_info):
        parser.error(
            "the following arguments are required: template, --templates-dir, "
            "--object-info"
        )

    workflow_id = args.id or args.template
    template = json.loads(
        (args.templates_dir / f"{args.template}.json").read_text(encoding="utf-8")
    )
    object_info = json.loads(args.object_info.read_text(encoding="utf-8"))

    # A template this importer cannot turn into a corpus entry is a refusal
    # with its reason, not a traceback.
    try:
        prompt = Converter(object_info).convert(template, resolve_switches=True)
        refuse_text_output(prompt, workflow_id)
        collapse_previews(prompt)
        normalise_save(prompt, workflow_id)
        Converter(object_info).prune(prompt)
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    path = ROOT / "manifest" / "workflows" / f"{workflow_id}_api.json"
    path.write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)} ({len(prompt)} nodes)", file=sys.stderr)
    try:
        entry = build_entry(workflow_id, prompt, template, args.skip_sha)
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({workflow_id: entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
