"""The corpus-specific steps on top of the template -> API conversion.

No test asks a host on the network. What the importer asks of a ComfyUI server
or of a model host is answered by a small HTTP server on this machine.
"""
import copy
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import import_template
from import_template import (
    apply_sets,
    build_entry,
    cadence_problem,
    check_prompt,
    clean_url,
    collapse_previews,
    describe,
    fill_file_combos,
    finish_prompt,
    greedy_text,
    input_names,
    insert_entry,
    keep_what_is_saved,
    model_files,
    normalise_sinks,
    note_urls,
    seed_overrides,
)
from template_api_export import ConversionError, Converter

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "importer"
SHA = "85984c2b1ee8c0aa39179bf122a7d01fa1b21c09"


def spec(required=None, optional=None, output=(), output_node=False, **more):
    """A node definition the way /object_info gives it."""
    required, optional = required or {}, optional or {}
    return {"input": {"required": required, "optional": optional},
            "input_order": {"required": list(required), "optional": list(optional)},
            "output": list(output), "output_node": output_node, **more}


PREFIX = {"filename_prefix": ["STRING", {"default": "ComfyUI"}]}
# The save and preview nodes of the sink table, and what feeds them.
INFO = {
    "KSampler": spec({"seed": ["INT", {"default": 0}]}, output=["LATENT"]),
    "VAEDecode": spec({"samples": ["LATENT"]}, output=["IMAGE"]),
    "VAEDecodeAudio": spec({"samples": ["LATENT"]}, output=["AUDIO"]),
    "VoxelToMesh": spec({"voxel": ["VOXEL"]}, output=["MESH"]),
    "Hunyuan3DFile": spec({"image": ["IMAGE"]}, output=["FILE_3D_GLB"]),
    "TextGenerate": spec({"prompt": ["STRING", {}],
                          "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": []}]},
                         {"mtp": ["COMBO", {"options": ["auto", "off"]}]}, output=["STRING"]),
    "CLIPTextEncode": spec({"text": ["STRING", {}]}, output=["CONDITIONING"]),
    "SaveImage": spec({"images": ["IMAGE"], **PREFIX}, output_node=True),
    "SaveImageAdvanced": spec({"images": ["IMAGE"], **PREFIX}, output_node=True),
    "PreviewImage": spec({"images": ["IMAGE"]}, output_node=True),
    "PreviewAny": spec({"source": ["*", {}]}, output=["STRING"], output_node=True),
    "ImageCompare": spec({"image_a": ["IMAGE"], "image_b": ["IMAGE"]}, output_node=True),
    "SaveAnimatedWEBP": spec({"images": ["IMAGE"], **PREFIX}, output_node=True),
    "SaveAnimatedPNG": spec({"images": ["IMAGE"], **PREFIX}, output_node=True),
    "SaveWEBM": spec({"images": ["IMAGE"], **PREFIX}, output_node=True),
    "ImageFromBatch": spec({"image": ["IMAGE"], "batch_index": ["INT", {}],
                            "length": ["INT", {}]}, output=["IMAGE"]),
    "CreateVideo": spec({"images": ["IMAGE"], "fps": ["FLOAT", {}]}, output=["VIDEO"]),
    "LoadVideo": spec({"file": ["COMBO", {"options": []}]}, output=["VIDEO"]),
    "GetVideoComponents": spec({"video": ["VIDEO"]}, output=["IMAGE", "AUDIO", "FLOAT"]),
    "SaveVideo": spec({"video": ["VIDEO"], **PREFIX,
                       "format": ["COMFY_DYNAMICCOMBO_V3", {"options": []}]}, output_node=True),
    "SaveGLB": spec({"mesh": ["MESH,FILE_3D_GLB"], **PREFIX}, output_node=True),
    "Save3DAdvanced": spec({"model_3d": ["FILE_3D_GLB"], **PREFIX}, output_node=True),
    "RenderMesh": spec({"mesh": ["MESH"], "mode": ["COMBO", {"options": ["auto"]}],
                        "width": ["INT", {}], "height": ["INT", {}],
                        "background": ["COLOR", {"default": "#000000"}]}, output=["IMAGE"]),
    "SaveAudioAdvanced": spec({"audio": ["AUDIO"], **PREFIX,
                               "format": ["COMFY_DYNAMICCOMBO_V3", {"options": []}]},
                              output_node=True),
    "SaveAudio": spec({"audio": ["AUDIO"], **PREFIX}, output_node=True),
    "SaveAudioMP3": spec({"audio": ["AUDIO"], **PREFIX}, output_node=True),
    "SaveAudioOpus": spec({"audio": ["AUDIO"], **PREFIX}, output_node=True),
    "SaveText": spec({"text": ["STRING", {"forceInput": True}], **PREFIX,
                      "format": ["COMBO", {"options": ["txt", "md"]}]}, output_node=True),
    "PartnerImageNode": spec({"prompt": ["STRING", {}]}, output=["IMAGE"], api_node=True),
}


def n(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def graph(**sinks):
    """A sampler, what decodes it to each kind of output, and the given sinks."""
    return {
        "1": n("KSampler", seed=5),
        "2": n("VAEDecode", samples=["1", 0]),
        "3": n("VAEDecodeAudio", samples=["1", 0]),
        "4": n("VoxelToMesh", voxel=["1", 0]),
        "5": n("TextGenerate", prompt="hi"),
        **sinks,
    }


def finished(prompt, workflow_id="wf"):
    """The sinks made ready, the previews dropped, what feeds no save node gone."""
    kind = normalise_sinks(prompt, workflow_id, INFO)
    collapse_previews(prompt)
    keep_what_is_saved(prompt)
    return kind


# --------------------------------------------------------------------------- #
# sinks, row by row
# --------------------------------------------------------------------------- #


def test_every_image_save_becomes_a_png_save_named_after_the_workflow():
    prompt = graph(**{
        "8": n("SaveImage", filename_prefix="PixelDiT", images=["2", 0]),
        "9": n("SaveImageAdvanced", **{
            "filename_prefix": "Qwen", "format": "png", "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB", "images": ["2", 0]}),
    })
    assert finished(prompt, "qwen_t2i") == "image"
    assert prompt["8"] == prompt["9"] == n("SaveImage", filename_prefix="qwen_t2i",
                                           images=["2", 0])
    assert sorted(prompt) == ["1", "2", "8", "9"]  # what fed no save node is gone


@pytest.mark.parametrize(
    "inputs, match",
    [
        ({"format": "exr", "images": ["2", 0]}, "8-bit PNG"),
        ({"format": "png", "format.bit_depth": "16-bit", "images": ["2", 0]}, "8-bit PNG"),
        ({"format": "png", "format.bit_depth": "8-bit", "format.input_color_space": "HDR",
          "images": ["2", 0]}, "8-bit PNG"),
        ({"format": "png", "format.bit_depth": "8-bit"}, "no images input"),
    ],
)
def test_a_save_comparison_cannot_read_is_refused(inputs, match):
    with pytest.raises(ConversionError, match=match):
        normalise_sinks(graph(**{"9": n("SaveImageAdvanced", **inputs)}), "wf", INFO)


def test_a_template_that_only_previews_saves_the_last_picture_nothing_reads():
    prompt = graph(**{
        "7": n("PreviewImage", images=["2", 0]),
        "8": n("ImageFromBatch", image=["2", 0], batch_index=0, length=1),
        "9": n("PreviewImage", images=["8", 0]),
        "10": n("PreviewAny", source=["5", 0]),
    })
    assert finished(prompt) == "image"
    assert prompt["9"] == n("SaveImage", filename_prefix="wf", images=["8", 0])
    assert sorted(prompt) == ["1", "2", "8", "9"]
    # Next to a save node a preview is only a preview.
    prompt = graph(**{"7": n("SaveText", text=["5", 0], filename_prefix="t", format="md"),
                      "9": n("PreviewImage", images=["2", 0])})
    assert finished(prompt) == "text" and sorted(prompt) == ["5", "7"]


@pytest.mark.parametrize("node", ["SaveAnimatedWEBP", "SaveAnimatedPNG"])
def test_an_animated_save_becomes_a_picture_of_its_last_frame(node):
    prompt = graph(**{"9": n(node, images=["2", 0], filename_prefix="ComfyUI", fps=6.0)})
    assert finished(prompt) == "image"
    assert prompt["10"] == n("ImageFromBatch", image=["2", 0], batch_index=-1, length=1)
    assert prompt["9"] == n("SaveImage", filename_prefix="wf", images=["10", 0])
    assert node not in {m["class_type"] for m in prompt.values()}


def test_a_clip_is_kept_as_mp4_and_its_last_frame_is_the_picture():
    prompt = graph(**{
        "6": n("CreateVideo", images=["2", 0], fps=24.0),
        "9": n("SaveVideo", video=["6", 0], filename_prefix="video/ComfyUI", format="auto",
               **{"format.codec": "auto", "codec": "auto"}),
    })
    assert finished(prompt) == "image"
    assert prompt["9"] == n("SaveVideo", filename_prefix="wf_video", format="mp4",
                            **{"format.codec": "h264"}, video=["6", 0])
    # The frames are the ones the clip was made of.
    assert prompt["10"] == n("ImageFromBatch", image=["2", 0], batch_index=-1, length=1)
    assert prompt["11"] == n("SaveImage", filename_prefix="wf", images=["10", 0])


def test_a_clip_that_no_create_video_made_is_taken_apart_for_its_frames():
    prompt = {"6": n("LoadVideo", file="clip.mp4"),
              "9": n("SaveVideo", video=["6", 0], filename_prefix="x", format="auto")}
    assert finished(prompt) == "image"
    assert prompt["10"] == n("GetVideoComponents", video=["6", 0])
    assert prompt["11"] == n("ImageFromBatch", image=["10", 0], batch_index=-1, length=1)
    assert prompt["12"] == n("SaveImage", filename_prefix="wf", images=["11", 0])
    with pytest.raises(ConversionError, match="save node 9 has no video input"):
        normalise_sinks({"9": n("SaveVideo", filename_prefix="x", format="auto")}, "wf", INFO)
    older = {k: v for k, v in INFO.items() if k != "GetVideoComponents"}
    with pytest.raises(ConversionError, match="this ComfyUI has no GetVideoComponents node"):
        normalise_sinks({"6": n("LoadVideo", file="clip.mp4"),
                         "9": n("SaveVideo", video=["6", 0])}, "wf", older)


def test_a_mesh_is_kept_and_a_picture_of_it_is_the_main_output():
    prompt = graph(**{"9": n("SaveGLB", mesh=["4", 0], filename_prefix="3d/ComfyUI")})
    assert finished(prompt) == "image"
    assert prompt["9"] == n("SaveGLB", filename_prefix="wf_mesh", mesh=["4", 0])
    assert prompt["10"] == n("RenderMesh", mesh=["4", 0], mode="auto", width=1024,
                             height=1024, background="#000000")
    assert prompt["11"] == n("SaveImage", filename_prefix="wf", images=["10", 0])


def test_a_3d_file_has_no_mesh_to_render_and_is_refused():
    said = "no mesh link to render .node 9 saves a 3D file, not a mesh.: add a SaveGLB and a " \
           "RenderMesh by hand"
    file_fed = graph(**{"6": n("Hunyuan3DFile", image=["2", 0]),
                        "9": n("SaveGLB", mesh=["6", 0], filename_prefix="3d/x")})
    advanced = graph(**{"6": n("Hunyuan3DFile", image=["2", 0]),
                        "9": n("Save3DAdvanced", model_3d=["6", 0], filename_prefix="3d/x")})
    for prompt in (file_fed, advanced):
        with pytest.raises(ConversionError, match=said):
            normalise_sinks(prompt, "wf", INFO)


@pytest.mark.parametrize("node", ["SaveAudioAdvanced", "SaveAudio", "SaveAudioMP3",
                                  "SaveAudioOpus"])
def test_sound_is_saved_as_flac_under_the_workflow_id(node):
    more = {"format": "mp3", "format.quality": "V0"} if node == "SaveAudioAdvanced" else {}
    prompt = graph(**{"9": n(node, audio=["3", 0], filename_prefix="audio/ComfyUI", **more)})
    assert finished(prompt) == "audio"
    assert prompt["9"] == n("SaveAudioAdvanced", filename_prefix="wf", format="flac",
                            audio=["3", 0])
    assert sorted(prompt) == ["1", "3", "9"]


def test_a_text_template_ends_in_one_save_text():
    preview = graph(**{"9": n("PreviewAny", source=["5", 0])})
    saved = graph(**{"9": n("SaveText", text=["5", 0], filename_prefix="x", format="md"),
                     "10": n("PreviewAny", source=["5", 0])})
    for prompt in (preview, saved):
        assert finished(prompt, "llm_x") == "text"
        assert prompt == {"5": n("TextGenerate", prompt="hi"),
                          "9": n("SaveText", text=["5", 0], filename_prefix="llm_x",
                                 format="txt")}
    # Two answers: the importer cannot tell which one is compared.
    two = graph(**{"9": n("PreviewAny", source=["5", 0]), "10": n("PreviewAny", source=["5", 0])})
    with pytest.raises(ConversionError, match="has 2 text outputs .nodes 9, 10. and one is"):
        normalise_sinks(two, "llm_x", INFO)
    # A preview on the way to an encoder is no answer, and one that shows a
    # typed-in text has nothing to save.
    passing = graph(**{"9": n("PreviewAny", source=["5", 0]),
                       "10": n("CLIPTextEncode", text=["9", 0])})
    for prompt in (passing, {"9": n("PreviewAny", source="a cat")}, graph()):
        with pytest.raises(ConversionError, match="saves nothing the suite can check"):
            normalise_sinks(prompt, "wf", INFO)


def test_the_first_kind_of_video_mesh_audio_image_text_names_the_workflow():
    sinks = {
        "video": ("20", n("SaveVideo", video=["19", 0], filename_prefix="v")),
        "mesh": ("21", n("SaveGLB", mesh=["4", 0], filename_prefix="m")),
        "audio": ("22", n("SaveAudio", audio=["3", 0], filename_prefix="a")),
        "image": ("23", n("SaveImage", images=["2", 0], filename_prefix="i")),
        "text": ("24", n("SaveText", text=["5", 0], filename_prefix="t", format="md")),
    }
    order = list(sinks)
    for first in range(len(order)):
        prompt = graph(**{"19": n("CreateVideo", images=["2", 0], fps=24.0)},
                       **copy.deepcopy(dict(sinks[name] for name in order[first:])))
        kind = normalise_sinks(prompt, "wf", INFO)
        assert kind == {"video": "image", "mesh": "image"}.get(order[first], order[first])
        prefixes = {name: prompt[sinks[name][0]]["inputs"]["filename_prefix"]
                    for name in order[first:]}
        # A clip and a mesh are never the main output: their picture is.
        wanted = {name: "wf" if name == kind else f"wf_{name}" for name in order[first:]}
        assert prefixes == wanted, order[first]
        formats = {m["inputs"]["format"] for m in prompt.values()
                   if m["class_type"] in ("SaveAudioAdvanced", "SaveText")}
        assert formats <= {"flac", "txt"}
        pictures = [m for m in prompt.values() if m["class_type"] == "SaveImage"]
        assert {m["inputs"]["filename_prefix"] for m in pictures} == (
            {"wf"} if kind == "image" else {"wf_image"} if first < 4 else set())


def test_what_the_importer_writes_passes_the_sink_rules_of_the_lint():
    """One rule for prefixes, the same here and in the manifest lint."""
    from test_manifest_lint import OUTPUT_NODES, _sink_problems

    assert set(import_template.KEPT_SAVES) <= OUTPUT_NODES
    cases = [
        ({"9": n("SaveImage", images=["2", 0]), "10": n("SaveImage", images=["2", 0])}, {}),
        ({"9": n("SaveAnimatedWEBP", images=["2", 0])}, {}),
        ({"6": n("CreateVideo", images=["2", 0]), "9": n("SaveVideo", video=["6", 0]),
          "10": n("SaveAudioMP3", audio=["3", 0])}, {}),
        ({"9": n("SaveGLB", mesh=["4", 0]), "10": n("SaveImage", images=["2", 0])}, {}),
        ({"9": n("SaveAudioOpus", audio=["3", 0]), "10": n("SaveImage", images=["2", 0]),
          "11": n("SaveText", text=["5", 0])}, {"output_kind": "audio"}),
        ({"9": n("PreviewAny", source=["5", 0])}, {"output_kind": "text"}),
    ]
    for sinks, entry in cases:
        prompt = graph(**sinks)
        assert finished(prompt) == entry.get("output_kind", "image")
        assert _sink_problems("wf", entry, prompt) == [], sinks


def test_a_save_node_no_check_reads_is_refused_and_a_node_that_only_shows_is_left_out():
    with pytest.raises(ConversionError, match="also saves with SaveWEBM, and no check reads"):
        normalise_sinks(graph(**{"8": n("SaveImage", images=["2", 0]),
                                 "9": n("SaveWEBM", images=["2", 0])}), "wf", INFO)
    prompt = graph(**{"8": n("SaveImage", images=["2", 0]),
                      "9": n("ImageCompare", image_a=["2", 0], image_b=["2", 0])})
    assert finished(prompt) == "image" and sorted(prompt) == ["1", "2", "8"]


def test_a_preview_in_the_middle_of_the_graph_forwards_its_value_and_goes():
    """PreviewAny shows a string on its way to a text encoder. Left in, it is a
    second output node and a node that feeds no save node."""
    prompt = {
        "1": {"class_type": "PreviewAny", "inputs": {"source": "a cat"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0]}},
    }
    collapse_previews(prompt)
    assert prompt == {"2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}}}


def test_a_preview_forwards_a_link_as_a_link():
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
    }
    collapse_previews(prompt)
    assert prompt["3"]["inputs"]["model"] == ["1", 0]


def test_a_terminal_preview_is_simply_dropped():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }
    collapse_previews(prompt)
    assert sorted(prompt) == ["1"]


# --------------------------------------------------------------------------- #
# file inputs, text generation, --set
# --------------------------------------------------------------------------- #

LOADERS = {
    "UpscaleModelLoader": spec({"model_name": ["COMBO", {"multiselect": False, "options": []}]},
                               output=["UPSCALE_MODEL"]),
    "LoadAudio": spec({"audio": ["COMBO", {"options": [], "audio_upload": True}]},
                      output=["AUDIO"]),
    "LoadVideo": spec({"file": ["COMBO", {"options": [], "video_upload": True}]},
                      output=["VIDEO"]),
    "LoadImage": spec({"image": [["example.png"], {"image_upload": True}]}, output=["IMAGE"]),
    "VAELoader": spec({"vae_name": ["COMBO", {"options": ["ae.safetensors"]}]}, output=["VAE"]),
    "CustomCombo": spec({"choice": ["COMBO", {"multiselect": False, "options": []}]},
                        output=["STRING"]),
    "Mix": spec({"model": ["UPSCALE_MODEL"], "audio": ["AUDIO"], "video": ["VIDEO"],
                 "image": ["IMAGE"]}, output_node=True),
}


def ui(node_id, node_type, widgets, links=(), models=None, inputs=()):
    """A template node with one output, read by the given links."""
    return {"id": node_id, "type": node_type, "mode": 0, "widgets_values": widgets,
            "inputs": list(inputs), "outputs": [{"name": "out", "type": "*",
                                                 "links": list(links)}],
            "properties": {"models": models} if models else {}}


def loaders_template():
    hf = "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main"
    return {
        "nodes": [
            ui(1, "UpscaleModelLoader", ["RealESRGAN_x4plus.safetensors"], [1], models=[
                {"name": "RealESRGAN_x4plus.safetensors", "directory": "upscale_models",
                 "url": f"{hf}/RealESRGAN_x4plus.safetensors?download=true"}]),
            ui(2, "LoadAudio", ["voice_demo.mp3", None, None], [2]),
            ui(3, "LoadVideo", ["warrior.mp4", "image"], [3]),
            ui(4, "LoadImage", ["cat.png", "image"], [4]),
            ui(5, "LoadAudio", ["second.flac", None, ""]),
            ui(9, "Mix", [], inputs=[
                {"name": "model", "type": "UPSCALE_MODEL", "link": 1},
                {"name": "audio", "type": "AUDIO", "link": 2},
                {"name": "video", "type": "VIDEO", "link": 3},
                {"name": "image", "type": "IMAGE", "link": 4}]),
        ],
        "links": [[1, 1, 0, 9, 0, "UPSCALE_MODEL"], [2, 2, 0, 9, 1, "AUDIO"],
                  [3, 3, 0, 9, 2, "VIDEO"], [4, 4, 0, 9, 3, "IMAGE"]],
    }


def test_a_file_input_the_server_lists_nothing_for_is_filled_from_the_template():
    """A server without the files lists no option, and the converter refuses a
    combo without options. 17 templates load a sound and 38 a clip."""
    info, template = copy.deepcopy(LOADERS), loaders_template()
    with pytest.raises(ConversionError, match="node 1 .UpscaleModelLoader.: combo whose options"):
        Converter(copy.deepcopy(info)).convert(template)

    assert fill_file_combos(info, template) == {
        ("UpscaleModelLoader", "model_name"), ("LoadAudio", "audio"), ("LoadVideo", "file")}

    def options(node, name):
        return info[node]["input"]["required"][name][1]["options"]

    assert options("UpscaleModelLoader", "model_name") == ["RealESRGAN_x4plus.safetensors"]
    # Every node of a class adds its file, also one nothing reads.
    assert options("LoadAudio", "audio") == ["voice_demo.mp3", "second.flac"]
    assert options("LoadVideo", "file") == ["warrior.mp4"]
    # An input that has its options is left as the server gave it.
    assert info["LoadImage"] == LOADERS["LoadImage"] and info["VAELoader"] == LOADERS["VAELoader"]
    prompt = Converter(info).convert(template)
    assert {k: v["inputs"] for k, v in prompt.items() if k != "9"} == {
        "1": {"model_name": "RealESRGAN_x4plus.safetensors"},
        "2": {"audio": "voice_demo.mp3"}, "3": {"file": "warrior.mp4"},
        "4": {"image": "cat.png"}}


def test_a_combo_whose_options_live_on_the_node_stays_refused():
    """CustomCombo keeps its options on the node instance. Nothing in the
    template says which widget value is which, so nothing is filled."""
    info = copy.deepcopy(LOADERS)
    template = {"nodes": [ui(1, "CustomCombo", ["Music", 0, "Music", "SFX", ""], [1]),
                          ui(9, "Mix", [], inputs=[{"name": "model", "type": "*", "link": 1}])],
                "links": [[1, 1, 0, 9, 0, "STRING"]]}
    assert fill_file_combos(info, template) == set()
    assert info == LOADERS
    with pytest.raises(ConversionError) as refusal:
        finish_prompt(template, info, "wf")
    assert str(refusal.value) == (
        "node 1 (CustomCombo): combo whose options are defined on the node instance, not "
        "the server: the widget layout cannot be derived from /object_info")


def test_text_generation_is_made_greedy():
    sampled = {"prompt": "hi", "sampling_mode": "on", "sampling_mode.temperature": 0.7,
               "sampling_mode.seed": 0, "thinking": False, "mtp": "auto"}
    prompt = {"1": n("TextGenerate", **sampled), "2": n("TextGenerateV2", prompt="x"),
              "3": n("CLIPTextEncode", text="sampling_mode.x")}
    greedy_text(prompt, INFO)
    assert prompt["1"]["inputs"] == {"prompt": "hi", "sampling_mode": "off", "thinking": False,
                                     "mtp": "off"}
    # A node this ComfyUI does not define gets no input it may not have.
    assert prompt["2"]["inputs"] == {"prompt": "x"}
    assert prompt["3"]["inputs"] == {"text": "sampling_mode.x"}


def test_set_overrides_a_literal_and_names_what_it_cannot_find():
    def fresh():
        return {"3": n("KSampler", seed=5, steps=20, model=["4", 0]),
                "37:13": n("TextEncode", max_duration=60, **{"mode.sub": 1}),
                "9": n("SaveImage", images=["3", 0], filename_prefix="wf")}

    info = {**INFO, "KSampler": spec({"seed": ["INT", {}], "steps": ["INT", {}],
                                       "model": ["MODEL"]}, {"cfg": ["FLOAT", {}]})}
    prompt = fresh()
    apply_sets(prompt, info, ["3.steps=4", "37:13.max_duration=30", "3.cfg=1.5",
                              '9.filename_prefix="other"', "37:13.mode.sub=[1, 2]"])
    assert prompt["3"]["inputs"] == {"seed": 5, "steps": 4, "model": ["4", 0], "cfg": 1.5}
    assert prompt["37:13"]["inputs"] == {"max_duration": 30, "mode.sub": [1, 2]}
    assert prompt["9"]["inputs"]["filename_prefix"] == "other"
    for bad, said in (
        ("7.steps=4", "--set 7.steps=4: the workflow has no node 7"),
        ("3.stepz=4", "--set 3.stepz=4: node 3 (KSampler) has no input stepz"),
        ("3.model=1", "--set 3.model=1: 3.model is fed by node 4: set the value there"),
        ("3.steps=four", "--set 3.steps=four: the value is not JSON. A number is written 30, "
                         "a string with quotes: '3.steps=\"four\"'"),
        ("3.steps", "--set 3.steps: write <node>.<input>=<value>"),
        ("steps=4", "--set steps=4: write <node>.<input>=<value>"),
    ):
        prompt = fresh()
        with pytest.raises(ConversionError) as refusal:
            apply_sets(prompt, info, ["3.seed=6", bad])
        assert str(refusal.value) == said


def test_the_node_list_shows_each_id_class_and_literal_input():
    prompt = {"3": n("KSampler", seed=5, cfg=1.5, model=["4", 0], name="euler"),
              "37:13": n("TextEncode", text="a" * 80, flags=[1, 2], on=True),
              "9": n("SaveImage", images=["3", 0])}
    assert describe(prompt).splitlines() == [
        '      3  KSampler  seed=5 cfg=1.5 name="euler"',
        '  37:13  TextEncode  text="' + "a" * 55 + '..." flags=[1, 2] on=true',
        "      9  SaveImage"]


# --------------------------------------------------------------------------- #
# model files, input files, what a host says
# --------------------------------------------------------------------------- #


def test_download_links_are_read_from_notes_at_both_graph_levels():
    template = {
        "nodes": [
            {
                "type": "MarkdownNote",
                "widgets_values": [
                    "**vae**\n- [v.safetensors](https://example.test/a/v.safetensors) (242 MB)"
                ],
            }
        ],
        "definitions": {
            "subgraphs": [
                {
                    "nodes": [
                        {
                            "type": "Note",
                            "widgets_values": [
                                "[u.safetensors](https://example.test/b/u.safetensors)"],
                        }
                    ]
                }
            ]
        },
    }
    assert note_urls(template) == {
        "v.safetensors": "https://example.test/a/v.safetensors",
        "u.safetensors": "https://example.test/b/u.safetensors",
    }


def test_the_first_link_for_a_file_name_wins():
    template = {
        "nodes": [
            {
                "type": "MarkdownNote",
                "widgets_values": [
                    "[v.safetensors](https://example.test/first/v.safetensors)\n"
                    "[v.safetensors](https://example.test/second/v.safetensors)"
                ],
            }
        ]
    }
    assert note_urls(template)["v.safetensors"] == "https://example.test/first/v.safetensors"


HF = "https://huggingface.co/Comfy-Org/x/resolve/main"


def models_template():
    """Loaders at the top level and in a subgraph, each with its own files."""
    def listed(name, folder, query=""):
        return [{"name": name, "url": f"{HF}/{folder}/{name}{query}", "directory": folder}]

    return {
        "nodes": [
            ui(4, "UNETLoader", ["dit.safetensors", "default"],
               models=listed("dit.safetensors", "diffusion_models", "?download=true")),
            ui(5, "VAELoader", ["pixel_space"], models=listed("ae.safetensors", "vae")),
            ui(6, "UNETLoader", ["dropped.safetensors", "default"],
               models=listed("dropped.safetensors", "diffusion_models")),
            ui(7, "CLIPLoader", ["old/t5.safetensors", "sd3"]),
            ui(40, "UNETLoader", ["dit.safetensors", "default"],
               models=listed("dit.safetensors", "diffusion_models")),
            ui(8, "LoraLoaderModelOnly", ["style.safetensors", 1.0], models=[{
                "name": "style.safetensors", "directory": "loras",
                "url": "https://civitai.com/api/download/models/353362?type=Model&fp=fp16"}]),
            {"id": 30, "type": "MarkdownNote",
             "widgets_values": ["[t5](https://example.test/enc/t5.safetensors)"]},
            {"id": 37, "type": "sub-1", "mode": 0, "inputs": [], "outputs": []},
        ],
        "definitions": {"subgraphs": [{"id": "sub-1", "nodes": [
            ui(6, "DualCLIPLoader", ["a.safetensors", "b.safetensors", "flux"], models=[
                *listed("a.safetensors", "text_encoders"),
                *listed("b.safetensors", "text_encoders")])]}]},
    }


def test_model_files_come_from_the_loader_nodes_that_survive():
    prompt = {
        "37:6": n("DualCLIPLoader", clip_name1="a.safetensors", clip_name2="b.safetensors",
                  type="flux"),
        "4": n("UNETLoader", unet_name="dit.safetensors", weight_dtype="default"),
        "5": n("VAELoader", vae_name="pixel_space"),
        "7": n("CLIPLoader", clip_name="old/t5.safetensors", type="sd3"),
        "8": n("LoraLoaderModelOnly", lora_name="style.safetensors", strength_model=1.0),
        "40": n("UNETLoader", unet_name="dit.safetensors"),
        "9": n("SaveImage", images=["4", 0], filename_prefix="wf"),
    }
    assert model_files(prompt, models_template()) == [
        # In the order the prompt loads them. A subgraph's loader is asked too.
        {"name": "a.safetensors", "url": f"{HF}/text_encoders/a.safetensors",
         "directory": "text_encoders"},
        {"name": "b.safetensors", "url": f"{HF}/text_encoders/b.safetensors",
         "directory": "text_encoders"},
        # A query is cut off a Hugging Face URL. The loader that was dropped,
        # and the file a loader lists and does not load, are not in the list.
        # Two loaders of one file give one entry.
        {"name": "dit.safetensors", "url": f"{HF}/diffusion_models/dit.safetensors",
         "directory": "diffusion_models"},
        # A loader that lists nothing: the link of the notes, the folder of the table.
        {"name": "old/t5.safetensors", "url": "https://example.test/enc/t5.safetensors",
         "directory": "text_encoders"},
        # On civitai the query selects the file.
        {"name": "style.safetensors", "directory": "loras",
         "url": "https://civitai.com/api/download/models/353362?type=Model&fp=fp16"},
    ]


def test_a_url_with_a_query():
    assert clean_url(f"{HF}/vae/ae.safetensors?download=true#x") == f"{HF}/vae/ae.safetensors"
    assert clean_url(f"{HF}/vae/ae.safetensors") == f"{HF}/vae/ae.safetensors"
    for other in ("https://civitai.com/api/download/models/1?type=Model&format=SafeTensor",
                  "https://example.test/huggingface.co/a?b=1"):
        assert clean_url(other) == other


def test_a_loaded_file_without_a_link_or_a_folder_is_refused():
    template = {"nodes": [ui(1, "UNETLoader", ["u.safetensors"])]}
    with pytest.raises(ConversionError, match="u.safetensors is loaded by the workflow but the "
                                              "template has no download link for it"):
        model_files({"1": n("UNETLoader", unet_name="u.safetensors")}, template)
    with pytest.raises(ConversionError, match="DIRECTORY_BY_LOADER_INPUT"):
        model_files({"1": n("SomeLoader", mystery_name="m.safetensors")}, template)
    # The same input name on another loader has its own folder.
    notes = {"nodes": [{"id": 5, "type": "Note", "widgets_values": [
        "[t](https://h.test/t.safetensors) [v](https://h.test/v.safetensors)"]}]}
    prompt = {"1": n("CLIPLoader", clip_name="t.safetensors"),
              "2": n("CLIPVisionLoader", clip_name="v.safetensors")}
    assert [m["directory"] for m in model_files(prompt, notes)] == ["text_encoders",
                                                                    "clip_vision"]
    two = {"nodes": [
        ui(1, "UNETLoader", ["u.safetensors"], models=[
            {"name": "u.safetensors", "url": "https://a.test/u", "directory": "unet"}]),
        ui(2, "UNETLoader", ["u.safetensors"], models=[
            {"name": "u.safetensors", "url": "https://b.test/u", "directory": "unet"}])]}
    with pytest.raises(ConversionError, match="two download links for unet/u.safetensors"):
        model_files({"1": n("UNETLoader", unet_name="u.safetensors"),
                     "2": n("UNETLoader", unet_name="u.safetensors")}, two)


def test_every_seed_input_is_pinned_at_the_value_the_template_ships():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}},
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
    }
    assert seed_overrides(prompt) == {"1": {"seed": 42}, "2": {"noise_seed": 7}}


def test_a_seed_that_a_link_feeds_is_pinned_where_its_number_is():
    """audio_ace_step1_5_xl_turbo feeds both seeds from one PrimitiveInt. The
    lint asks every workflow with a seed input for an override, so the entry
    without one was written and then failed the lint."""
    prompt = {
        "109": n("PrimitiveInt", value=0),
        "3": n("KSampler", seed=["109", 0], steps=8),
        "94": n("TextEncodeAceStepAudio1.5", seed=["109", 0], bpm=120),
        "5": n("RandomNoise", noise_seed=["6", 0]),
        "6": n("SeedSource", number=31, label="x", fixed=True),
    }
    assert seed_overrides(prompt) == {"109": {"value": 0}, "6": {"number": 31}}
    # The override hits an input that is there, which is what the lint asks.
    assert all(name in prompt[node_id]["inputs"]
               for node_id, patch in seed_overrides(prompt).items() for name in patch)
    # A node with two numbers, or with none: the importer does not guess.
    for source in (n("Add", a=1, b=2), n("Seeder", seed=["7", 0]), n("Seeder")):
        with pytest.raises(ConversionError) as refusal:
            seed_overrides({"1": n("KSampler", seed=["2", 0]), "2": source})
        assert str(refusal.value) == (
            f"node 1 (KSampler) gets its seed from node 2 ({source['class_type']}), and the "
            "importer cannot tell which input of that node holds the number. The lint asks "
            "every workflow with a seed for a seed override: write the entry by hand")


def test_the_seed_of_a_sampled_text_generation_is_pinned_under_its_whole_name():
    """`sampling_mode.seed` is a sub-input of the sampling mode. The runner
    writes a seed override over the input the entry names, dot or no dot, so
    the entry names it as the workflow file has it."""
    import run_regression

    prompt = {
        "5": n("TextGenerate", prompt="hi", sampling_mode="on",
               **{"sampling_mode.temperature": 0.7, "sampling_mode.seed": 7}),
        "6": n("TextGenerateLTX2Prompt", sampling_mode="on",
               **{"sampling_mode.seed": ["109", 0]}),
        "109": n("PrimitiveInt", value=3),
        # The last part of a name decides: `seed.mode` and `reseed` are no seeds.
        "7": n("KSampler", seed=5, **{"seed.mode": 1, "sampling_mode.reseed": 2}),
        "8": n("TextGenerate", prompt="hi", sampling_mode="off"),
    }
    pins = seed_overrides(prompt)
    assert pins == {"5": {"sampling_mode.seed": 7}, "109": {"value": 3}, "7": {"seed": 5}}
    # What a run queues when the entry says another number: the inputs the
    # entry names have it, and the prompt has no input more than before.
    entry = {node_id: {name: 99 for name in patch} for node_id, patch in pins.items()}
    queued = run_regression.apply_seed_overrides(prompt, entry)
    assert queued["5"]["inputs"] == {"prompt": "hi", "sampling_mode": "on",
                                     "sampling_mode.temperature": 0.7, "sampling_mode.seed": 99}
    assert (queued["109"]["inputs"], queued["7"]["inputs"]["seed"]) == ({"value": 99}, 99)
    assert {node_id: list(node["inputs"]) for node_id, node in queued.items()} == {
        node_id: list(node["inputs"]) for node_id, node in prompt.items()}
    assert prompt["5"]["inputs"]["sampling_mode.seed"] == 7


def loads_inputs():
    return {
        "1": n("UNETLoader", unet_name="u.safetensors"),
        "2": n("LoadImage", image="cat.png [input]"),
        "3": n("LoadAudio", audio="voice demo.mp3"),
        "4": n("LoadVideo", file="clip.mp4"),
        "5": n("LoadImage", image="cat.png"),
        "6": n("KSampler", seed=5, image=["2", 0]),
        "9": n("SaveImage", images=["6", 0], filename_prefix="wf"),
    }


UNET_TEMPLATE = {"nodes": [ui(1, "UNETLoader", ["u.safetensors"], models=[
    {"name": "u.safetensors", "url": "https://h.test/u.safetensors?download=true",
     "directory": "diffusion_models"}])]}


def test_input_files_become_models_entries_pinned_to_a_commit(monkeypatch):
    assert input_names(loads_inputs()) == ["cat.png", "voice demo.mp3", "clip.mp4"]
    raw = f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{SHA}/input"
    asked = []
    monkeypatch.setattr(import_template, "file_facts",
                        lambda url: asked.append(url) or ("ab" * 32, len(url)))
    entry = build_entry("wf", loads_inputs(), UNET_TEMPLATE, cadence="nightly",
                        pinned=lambda: SHA)
    assert entry["models"] == [
        {"name": "u.safetensors", "url": "https://h.test/u.safetensors?download=true",
         "directory": "diffusion_models", "sha256": "ab" * 32, "bytes": 42},
        {"name": "cat.png", "url": f"{raw}/cat.png", "directory": "input",
         "sha256": "ab" * 32, "bytes": len(f"{raw}/cat.png")},
        {"name": "voice demo.mp3", "url": f"{raw}/voice%20demo.mp3", "directory": "input",
         "sha256": "ab" * 32, "bytes": len(f"{raw}/voice%20demo.mp3")},
        {"name": "clip.mp4", "url": f"{raw}/clip.mp4", "directory": "input",
         "sha256": "ab" * 32, "bytes": len(f"{raw}/clip.mp4")},
    ]
    assert asked == [m["url"] for m in entry["models"]]
    assert entry["comfy_flags"] == ["--input-directory", "/runpod-volume/models/input"]
    assert list(entry) == ["enabled", "level", "cadence", "workflow_path", "seed_overrides",
                           "models", "comfy_flags", "gpu_type", "timeout_s", "thresholds",
                           "golden_tag"]
    # An input file the repository does not have at that commit stops the import.
    monkeypatch.setattr(import_template, "file_facts", lambda url: (None, None))
    with pytest.raises(ConversionError, match=f"cat.png could not be downloaded from {raw}/cat"):
        build_entry("wf", loads_inputs(), UNET_TEMPLATE, cadence="nightly", pinned=lambda: SHA)
    with pytest.raises(ConversionError, match="no commit of the templates repository is known"):
        build_entry("wf", loads_inputs(), UNET_TEMPLATE, cadence="nightly", skip_sha=True)


def test_the_entry_is_switched_on_and_says_how_it_is_checked(monkeypatch):
    prompt = {"1": n("UNETLoader", unet_name="u.safetensors"), "2": n("KSampler", seed=5)}
    monkeypatch.setattr(import_template, "file_facts", lambda url: pytest.fail("no lookup"))
    entry = build_entry("my_wf", prompt, UNET_TEMPLATE, cadence="nightly", skip_sha=True)
    assert entry == {
        "enabled": True, "level": "smoke", "cadence": "nightly",
        "workflow_path": "manifest/workflows/my_wf_api.json",
        "seed_overrides": {"2": {"seed": 5}},
        "models": [{"name": "u.safetensors", "directory": "diffusion_models",
                    "url": "https://h.test/u.safetensors?download=true"}],
        "gpu_type": None, "timeout_s": None, "thresholds": None, "golden_tag": None}
    assert "pinned" not in entry and "comfy_flags" not in entry
    full = build_entry("my_wf", prompt, UNET_TEMPLATE, cadence="per-commit", kind="audio",
                       level="reference", expect={"seconds": 30}, timeout_s=900, skip_sha=True)
    assert list(full)[:5] == ["enabled", "level", "cadence", "output_kind", "expect"]
    assert (full["level"], full["cadence"], full["output_kind"], full["expect"],
            full["timeout_s"]) == ("reference", "per-commit", "audio", {"seconds": 30}, 900)
    # A host that says nothing leaves the keys out, and the lint asks for them.
    monkeypatch.setattr(import_template, "file_facts", lambda url: (None, None))
    assert list(build_entry("my_wf", prompt, UNET_TEMPLATE, cadence="nightly")["models"][0]) == [
        "name", "url", "directory"]


LFS = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
GATED = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"
MOVED = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/96309bd/input/shirt.png"


def test_a_download_size_comes_from_the_redirect_the_file_or_the_listing(monkeypatch):
    """The four answers the manifest's own URLs gave on 2026-11-18."""
    answers = {
        (LFS, False): (302, {"X-Linked-Size": "253806246", "Content-Length": "1084"}),
        (MOVED, False): (307, {"Content-Length": "132"}),
        (MOVED, True): (200, {"Content-Length": "335304388"}),
        (RAW, False): (200, {"Content-Length": "1366572"}),
        (GATED, False): (401, {"Content-Length": "136"}),
        ("https://h.test/gone", False): (404, {"Content-Length": "9"}),
        ("https://h.test/down", False): (None, {}),
        ("https://h.test/empty", False): (200, {"Content-Length": "0"}),
    }
    asked, listed = [], []
    monkeypatch.setattr(import_template, "_head",
                        lambda url, follow: asked.append((url, follow)) or answers[(url, follow)])
    monkeypatch.setattr(import_template, "listed_bytes",
                        lambda url: listed.append(url) or {GATED: 335304388}.get(url))
    size = import_template.published_bytes
    assert size(LFS) == 253806246      # the redirect names it: the length there is the page's
    assert size(MOVED) == 335304388    # a renamed repository: only the file itself says it
    assert size(RAW) == 1366572
    assert size(GATED) == 335304388    # no token: the length of the refusal is not a size
    assert [size(f"https://h.test/{name}") for name in ("gone", "down", "empty")] == [None] * 3
    assert asked == [(LFS, False), (MOVED, False), (MOVED, True), (RAW, False), (GATED, False),
                     ("https://h.test/gone", False), ("https://h.test/down", False),
                     ("https://h.test/empty", False)]
    assert listed == [GATED, "https://h.test/gone", "https://h.test/down", "https://h.test/empty"]


def test_one_request_gives_the_sha_and_the_size_of_an_lfs_file(monkeypatch):
    """`ace_step_1.5_turbo_aio.safetensors` on 2026-11-18: its repository was
    renamed, so the first answer is a redirect on the same host without the
    headers. Hashing that 10 GB file by download would have taken an hour."""
    old = "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/c/a.safetensors"
    new = "https://huggingface.co/Comfy-Org/ACE-Step-1.5/resolve/main/c/a.safetensors"
    linked = {"X-Linked-ETag": '"' + "67" * 32 + '"', "X-Linked-Size": "10025478736",
              "Location": "https://cas-bridge.xethub.hf.co/x", "Content-Length": "1084"}
    answers = {LFS: (302, linked),
               old: (307, {"Location": "/Comfy-Org/ACE-Step-1.5/resolve/main/c/a.safetensors"}),
               new: (302, linked),
               RAW: (200, {"Content-Length": "9"}),
               GATED: (401, {}),
               "https://h.test/away": (302, {"Location": "https://cdn.test/file"})}
    asked, downloads = [], []
    monkeypatch.setattr(import_template, "_head",
                        lambda url, follow: asked.append((url, follow)) or answers[url])
    monkeypatch.setattr(import_template, "listed_bytes", lambda url: 335304388)
    monkeypatch.setattr(import_template, "hash_download", lambda url: downloads.append(url) or {
        RAW: ("ab" * 32, 9), "https://h.test/away": ("cd" * 32, 7)}.get(url))
    facts = import_template.file_facts
    assert facts(LFS) == ("67" * 32, 10025478736) and asked == [(LFS, False)]
    assert facts(old) == ("67" * 32, 10025478736)
    assert asked[1:] == [(old, False), (new, False)] and downloads == []
    # No sha256 published: the file is downloaded once. A gated one cannot be.
    assert facts(RAW) == ("ab" * 32, 9)
    assert facts(GATED) == (None, 335304388)
    # A redirect to another host is not followed for headers: the file is hashed.
    assert facts("https://h.test/away") == ("cd" * 32, 7)
    assert downloads == [RAW, GATED, "https://h.test/away"]


def test_a_host_that_is_busy_or_lacks_the_file_stops_the_import(monkeypatch, capsys):
    """Every answer without a sha256 was read as "the host publishes none". A
    host that answered 503 to each request ended in an entry without its
    sha256, with exit code 0 and a note that blamed a gated repository."""
    url = "https://huggingface.co/Comfy-Org/x/resolve/main/vae/ae.safetensors"
    raw = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/0f0f/input/cat.png"
    answer = {}
    monkeypatch.setattr(import_template, "_head", lambda u, follow: answer["head"])
    monkeypatch.setattr(import_template, "listed_bytes", lambda u: 335304388)
    monkeypatch.setattr(import_template, "hash_download", lambda u: answer.get("download"))
    for status, said in ((503, "huggingface.co did not answer for {url} (503): try again"),
                         (500, "huggingface.co did not answer for {url} (500): try again"),
                         (429, "huggingface.co did not answer for {url} (429): try again"),
                         (None, "huggingface.co did not answer for {url} (no answer): try again"),
                         (404, "{url} is not there: the host answers 404"),
                         (200, "{url} could not be downloaded to hash it: try again")):
        answer["head"] = (status, {})
        with pytest.raises(ConversionError) as refusal:
            import_template.file_facts(url)
        assert str(refusal.value) == said.format(url=url), status
    # An input file that is not at the pinned commit says what to do about it.
    answer["head"] = (404, {})
    with pytest.raises(ConversionError) as refusal:
        import_template.file_facts(raw)
    assert str(refusal.value) == (
        f"{raw} is not there: the host answers 404. The templates repository has no such "
        "file at that commit: name a commit that is on GitHub with --templates-ref")
    # Only a gated file comes back without its sha256: its download fails too.
    for status in (401, 403):
        answer["head"] = (status, {})
        assert import_template.file_facts(url) == (None, 335304388)
    # A host that refuses the HEAD request and sends the file is hashed, as before.
    capsys.readouterr()
    for status in (403, 405, 200):
        answer.update(head=(status, {}), download=("ab" * 32, 9))
        assert import_template.file_facts(url) == ("ab" * 32, 9)
    assert capsys.readouterr().err == (
        f"{url} publishes no sha256: downloading it once to hash it\n" * 3)


def test_the_listing_is_asked_for_the_files_own_folder(monkeypatch):
    class Answer:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            import io
            return io.BytesIO(json.dumps(self.rows).encode())

        def __exit__(self, *exc):
            return False

    asked = []

    def urlopen(url, timeout=None):
        asked.append(url)
        if "missing" in url:
            raise OSError("404")
        return Answer([{"type": "directory", "path": "split_files/vae/old"},
                       {"type": "file", "path": "ae.safetensors", "size": 335304388},
                       {"type": "file", "path": "split_files/vae/ae.safetensors", "size": 77,
                        "lfs": {"size": 77}}])

    monkeypatch.setattr(import_template.urllib.request, "urlopen", urlopen)
    assert import_template.listed_bytes(GATED + "?download=true") == 335304388
    assert import_template.listed_bytes(MOVED) == 77
    assert asked == [
        "https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev/tree/main",
        "https://huggingface.co/api/models/Comfy-Org/z_image_turbo/tree/main/split_files/vae"]
    # A file the listing does not have, a listing that fails, a host that has none.
    assert import_template.listed_bytes(GATED.replace("ae.", "other.")) is None
    assert import_template.listed_bytes(GATED.replace("FLUX.1-dev", "missing")) is None
    assert import_template.listed_bytes(RAW) is None and len(asked) == 4


@pytest.fixture
def server():
    """An HTTP server on this machine. A test says what each path answers
    (`answer`) and reads what was sent (`seen`)."""
    state = SimpleNamespace(seen=[], answers={})

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            state.seen.append((self.command, self.path, body, dict(self.headers)))
            status, headers, payload = state.answers.get((self.command, self.path),
                                                         (404, {}, b"not here"))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        do_GET = do_POST = do_HEAD = _serve

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_port}"

    def answer(method, path, payload=b"", status=200, **headers):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode()
        state.answers[(method, path)] = (status, headers, payload)

    state.answer = answer
    state.sent = lambda: [(method, path, body) for method, path, body, _ in state.seen]
    yield state
    httpd.shutdown()
    httpd.server_close()


def test_the_token_goes_to_hugging_face_and_to_no_other_host(server, monkeypatch):
    """A gated file answers only with the token, and its download redirects to
    a storage host, which must not get it."""
    data = b"weights" * 1000
    server.answer("GET", "/org/old/resolve/main/a.bin", status=307,
                  Location="/org/new/resolve/main/a.bin")
    server.answer("GET", "/org/new/resolve/main/a.bin", data)
    server.answer("HEAD", "/org/new/resolve/main/a.bin", status=302,
                  Location="https://storage.test/a.bin")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    # Any other host: no token, whatever the environment holds.
    assert import_template.hash_download(f"{server.url}/org/new/resolve/main/a.bin") == (
        hashlib.sha256(data).hexdigest(), 7000)
    assert server.seen[-1][3].get("Authorization") is None
    # Hugging Face, played by the server on this machine: the token goes with
    # the request and with a redirect that stays there (a renamed repository).
    monkeypatch.setattr(import_template, "HF_HOST", "127.0.0.1")
    monkeypatch.setattr(import_template, "HF_SCHEME", "http")
    server.seen.clear()
    assert import_template.hash_download(f"{server.url}/org/old/resolve/main/a.bin")[1] == 7000
    assert import_template._head(f"{server.url}/org/new/resolve/main/a.bin", follow=False)[0] \
        == 302
    assert [(path, headers.get("Authorization")) for _, path, _, headers in server.seen] == [
        ("/org/old/resolve/main/a.bin", "Bearer hf_secret"),
        ("/org/new/resolve/main/a.bin", "Bearer hf_secret"),
        ("/org/new/resolve/main/a.bin", "Bearer hf_secret")]
    # A redirect that leaves the host takes the token off. Python would copy it.
    request = import_template.urllib.request.Request(
        f"{server.url}/org/new/resolve/main/a.bin", headers={"Authorization": "Bearer hf_secret"})
    follow = import_template._TokenStaysHome().redirect_request

    def token_after(target):
        return follow(request, None, 302, "Found", {}, target).headers.get("Authorization")

    assert token_after("https://storage.test/a.bin") is None
    assert token_after(f"{server.url}/elsewhere/a.bin") == "Bearer hf_secret"
    # Without a token nothing is sent, and a file that is not there has no hash.
    monkeypatch.delenv("HF_TOKEN")
    assert import_template.hash_download(f"{server.url}/gone") is None
    assert server.seen[-1][3].get("Authorization") is None


def test_the_token_never_goes_out_over_plain_http(monkeypatch):
    """A template may name `http://huggingface.co/...`, on a loader or in a
    note. The host matched, so the token went out unencrypted, for anyone on
    the way to read. No request leaves this test: the opener only records."""
    sent = []

    class Recorder:
        def open(self, request, timeout=None):
            sent.append((request.full_url, request.get_header("Authorization")))
            raise OSError("not sent")

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(import_template.urllib.request, "build_opener", lambda *a: Recorder())
    path = "/a/b/resolve/main/x.safetensors"
    urls = [f"https://huggingface.co{path}", f"http://huggingface.co{path}",
            f"https://HuggingFace.co:443{path}", f"https://huggingface.co.example.org{path}",
            f"https://huggingface.co@example.org{path}", "https://cdn-lfs.huggingface.co/x",
            "https://civitai.com/api/download/models/1"]
    for url in urls:
        assert import_template._head(url, follow=False) == (None, {})
        assert import_template.hash_download(url) is None
    assert [url for url, token in sent if token] == [urls[0]] * 2 + [urls[2]] * 2
    assert all(token == "Bearer hf_secret" for _, token in sent if token)
    # A redirect to the same host over http takes the token off too.
    request = import_template.urllib.request.Request(
        urls[0], headers={"Authorization": "Bearer hf_secret"})
    follow = import_template._TokenStaysHome().redirect_request

    def token_after(target):
        return follow(request, None, 302, "Found", {}, target).headers.get("Authorization")

    assert token_after("http://huggingface.co/a/b/resolve/main/y") is None
    assert token_after("https://huggingface.co/a/b/resolve/main/y") == "Bearer hf_secret"
    # A link that is not https is refused before any request. The lint takes
    # https only, so such an entry failed it after the files were written.
    for where in ("loader", "note"):
        link = "http://huggingface.co/a/b/resolve/main/u.safetensors"
        listed = [{"name": "u.safetensors", "url": link, "directory": "diffusion_models"}]
        template = {"nodes": [
            ui(1, "UNETLoader", ["u.safetensors"], models=listed if where == "loader" else None),
            {"id": 5, "type": "Note", "widgets_values": [f"[u.safetensors]({link})"]}]}
        with pytest.raises(ConversionError) as refusal:
            model_files({"1": n("UNETLoader", unet_name="u.safetensors")}, template)
        assert str(refusal.value) == f"the download link of u.safetensors is not https: {link}"


def _sized_manifest():
    def model(name, **more):
        return {"name": name, "url": f"https://h.test/{name}", "directory": "vae", **more}

    return {"schema_version": 1, "workflows": {
        "a": {"models": [model("x.safetensors", sha256="0" * 64), model("y.png", bytes=7)]},
        "b": {"models": [model("x.safetensors", sha256="0" * 64), model("z.safetensors")]},
        "c": {"models": []}}}


def test_fill_bytes_adds_the_missing_sizes_and_nothing_else(tmp_path):
    path = tmp_path / "workflows.json"
    before = json.dumps(_sized_manifest(), indent=2) + "\n"
    path.write_text(before, encoding="utf-8")
    asked = []
    unknown = import_template.fill_bytes(
        path, lambda url: asked.append(url) or {"https://h.test/x.safetensors": 123}.get(url))
    # Each URL once. One that has its size is not asked, and one nobody knows is named.
    assert asked == ["https://h.test/x.safetensors", "https://h.test/z.safetensors"]
    assert unknown == ["https://h.test/z.safetensors"]
    after = path.read_text(encoding="utf-8")
    models = {wf: entry["models"] for wf, entry in json.loads(after)["workflows"].items()}
    assert list(models["a"][0]) == ["name", "url", "directory", "sha256", "bytes"]
    assert models["a"][0]["bytes"] == 123 == models["b"][0]["bytes"]
    assert models["a"][1]["bytes"] == 7 and "bytes" not in models["b"][1]
    # The file differs by the two new lines and the comma each of them needs.
    added = [line.strip() for line in after.splitlines() if line not in before.splitlines()]
    assert sorted(added) == ['"bytes": 123'] * 2 + ['"sha256": "' + "0" * 64 + '",'] * 2
    assert len(after.splitlines()) == len(before.splitlines()) + 2
    # Run again with every size known: nothing changes.
    assert import_template.fill_bytes(path, lambda url: 5) == []
    assert json.loads(path.read_text(encoding="utf-8"))["workflows"]["b"]["models"][1]["bytes"] == 5


def test_fill_bytes_leaves_a_manifest_in_another_layout_alone(tmp_path):
    path = tmp_path / "workflows.json"
    compact = json.dumps(_sized_manifest())
    path.write_text(compact, encoding="utf-8")
    with pytest.raises(SystemExit, match="not laid out the way this command writes it"):
        import_template.fill_bytes(path, lambda url: pytest.fail("nothing is asked"))
    assert path.read_text(encoding="utf-8") == compact


def test_the_fill_bytes_command_needs_no_template(tmp_path, monkeypatch, capsys):
    (tmp_path / "manifest").mkdir()
    path = tmp_path / "manifest" / "workflows.json"
    path.write_text(json.dumps(_sized_manifest(), indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(import_template, "ROOT", tmp_path)
    monkeypatch.setattr(import_template, "published_bytes",
                        lambda url: None if url.endswith("z.safetensors") else 99)
    assert import_template.main(["--fill-bytes"]) == 1
    assert capsys.readouterr().err == (
        "error: no size found for https://h.test/z.safetensors: add its bytes by hand\n")
    written = json.loads(path.read_text(encoding="utf-8"))["workflows"]
    assert written["a"]["models"][0]["bytes"] == 99 and "bytes" not in written["b"]["models"][1]
    monkeypatch.setattr(import_template, "published_bytes", lambda url: 99)
    assert import_template.main(["--fill-bytes"]) == 0
    # Without the flag the command asks for what an import needs. The cadence
    # has no default: where a workflow runs decides what it costs.
    for argv, missing in (
        (["--skip-sha"], "template, --server or --object-info, --cadence"),
        (["some_template", "--object-info", "x.json"], "--cadence"),
        (["some_template", "--cadence", "nightly"], "--server or --object-info"),
    ):
        with pytest.raises(SystemExit):
            import_template.main(argv)
        assert f"the following arguments are required: {missing}\n" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the prompt check
# --------------------------------------------------------------------------- #

CHECKED = {
    "4": n("CheckpointLoaderSimple", ckpt_name="sd.safetensors"),
    "10": n("LoadImage", image="cat.png"),
    "11": n("LoadAudio", audio="voice.mp3"),
    "3": n("KSampler", seed=7, steps=20, model=["4", 0], image=["10", 0]),
    "9": n("SaveImage", images=["3", 0], filename_prefix="wf"),
}
FILES = {"sd.safetensors", "cat.png", "voice.mp3"}


def node_error(class_type, *errors):
    return {"class_type": class_type, "dependent_outputs": ["9"], "errors": [
        {"type": kind, "message": message, "details": details,
         "extra_info": {"input_name": name}} for kind, message, name, details in errors]}


NOT_IN_LIST = ("value_not_in_list", "Value not in list", "ckpt_name",
               "ckpt_name: 'sd.safetensors' not in []")
NO_IMAGE = ("custom_validation_failed", "Custom validation failed for node", "image",
            "image - Invalid image file: cat.png")
NO_SOUND = ("value_not_in_list", "Value not in list", "audio", "audio: 'voice.mp3' not in []")
REFUSED = {"type": "prompt_outputs_failed_validation", "details": "",
           "message": "Prompt outputs failed validation"}


def test_the_prompt_check_passes_when_the_server_only_lacks_the_files(server):
    """ComfyUI rejects a prompt that names a file it does not have. That is all
    a server without the models can say against a good prompt."""
    server.answer("POST", "/prompt", status=400, payload={"error": REFUSED, "node_errors": {
        "4": node_error("CheckpointLoaderSimple", NOT_IN_LIST),
        "10": node_error("LoadImage", NO_IMAGE),
        "11": node_error("LoadAudio", NO_SOUND)}})
    filled = {("LoadAudio", "audio")}
    assert check_prompt(server.url + "/", CHECKED, {"sd.safetensors", "cat.png"}, filled) == (
        True, ["prompt check: passed. The server does not have cat.png, sd.safetensors, "
               "voice.mp3, and names no other problem"])
    # The whole prompt was sent, once, and nothing else was asked.
    assert server.sent() == [("POST", "/prompt", {"prompt": CHECKED})]
    # The same errors on an input that names no file of the entry are errors.
    passed, lines = check_prompt(server.url, CHECKED, set(), set())
    assert not passed and lines == [
        "prompt check: ComfyUI refuses the prompt",
        "  node 4 (CheckpointLoaderSimple): Value not in list: ckpt_name: 'sd.safetensors' "
        "not in []",
        "  node 10 (LoadImage): Custom validation failed for node: image - Invalid image "
        "file: cat.png",
        "  node 11 (LoadAudio): Value not in list: audio: 'voice.mp3' not in []"]


def test_the_prompt_check_stops_on_any_other_error(server, monkeypatch):
    too_big = ("value_bigger_than_max", "Value 20000 bigger than max of 10000", "steps", "steps")
    other_node = ("custom_validation_failed", "Custom validation failed for node", "seed",
                  "seed - not today")
    server.answer("POST", "/prompt", status=400, payload={"error": REFUSED, "node_errors": {
        "4": node_error("CheckpointLoaderSimple", NOT_IN_LIST),
        "3": node_error("KSampler", too_big, other_node)}})
    passed, lines = check_prompt(server.url, CHECKED, FILES, set())
    assert not passed and lines == [
        "prompt check: ComfyUI refuses the prompt",
        "  node 3 (KSampler): Value 20000 bigger than max of 10000: steps",
        "  node 3 (KSampler): Custom validation failed for node: seed - not today"]
    # A prompt ComfyUI turns down as a whole names no node.
    server.answer("POST", "/prompt", status=400, payload={"node_errors": {}, "error": {
        "type": "missing_node_type", "message": "Node 'Sampler' not found.",
        "details": "Node ID '#3'"}})
    assert check_prompt(server.url, CHECKED, FILES, set()) == (False, [
        "prompt check: ComfyUI refuses the prompt", "  Node 'Sampler' not found.: Node ID '#3'"])
    for payload, said in ((b"<html>busy</html>", "prompt check: no answer from"),
                          (b"[]", "answered 500 with no object"),
                          (b"{}", "/prompt answered 500: no prompt id")):
        server.answer("POST", "/prompt", payload, status=500)
        passed, lines = check_prompt(server.url, CHECKED, FILES, set())
        assert not passed and said in " ".join(lines)
    monkeypatch.setattr(import_template, "_post", lambda url, body: (_ for _ in ()).throw(
        ConnectionRefusedError("nobody listens")))
    assert check_prompt("http://127.0.0.1:8188", CHECKED, FILES, set()) == (False, [
        "prompt check: no answer from http://127.0.0.1:8188/prompt (nobody listens)"])


def test_a_prompt_the_server_accepts_is_taken_out_of_its_queue_at_once(server):
    """A server that has the files would start to generate on the author's machine."""
    server.answer("POST", "/prompt", {"prompt_id": "abc-1", "number": 3, "node_errors": {}})
    server.answer("POST", "/queue")
    server.answer("POST", "/interrupt")
    assert check_prompt(server.url, CHECKED, FILES, set()) == (True, [
        "prompt check: the server accepted the prompt, and it was taken out of the queue"])
    assert server.sent() == [("POST", "/prompt", {"prompt": CHECKED}),
                             ("POST", "/queue", {"delete": ["abc-1"]}),
                             ("POST", "/interrupt", {"prompt_id": "abc-1"})]
    # Accepted for one output while another branch is wrong: removed, and refused.
    server.seen.clear()
    wrong = ("required_input_missing", "Required input is missing", "images", "images")
    server.answer("POST", "/prompt", {"prompt_id": "abc-2", "number": 4, "node_errors": {
        "9": node_error("SaveImage", wrong)}})
    passed, lines = check_prompt(server.url, CHECKED, FILES, set())
    assert not passed and lines[1:] == [
        "prompt check: ComfyUI refuses the prompt",
        "  node 9 (SaveImage): Required input is missing: images"]
    assert [path for _, path, _ in server.sent()] == ["/prompt", "/queue", "/interrupt"]


# --------------------------------------------------------------------------- #
# the manifest, the lanes, the lint
# --------------------------------------------------------------------------- #


def test_the_entry_goes_to_the_end_of_the_manifest_and_no_other_line_changes(tmp_path):
    path = tmp_path / "workflows.json"
    shutil.copy(ROOT / "manifest" / "workflows.json", path)
    before = path.read_text(encoding="utf-8")
    entry = {"enabled": True, "level": "smoke", "models": [], "timeout_s": None}
    insert_entry(path, "importer_selftest", entry)
    after = path.read_text(encoding="utf-8")
    manifest = json.loads(after)
    assert list(manifest["workflows"])[-1] == "importer_selftest"
    assert manifest["workflows"]["importer_selftest"] == entry
    old, new = before.splitlines(), after.splitlines()
    # Every old line is still there, in order. The one before the new entry got its comma.
    assert new[:len(old) - 3] == old[:-3] and new[len(old) - 3] == old[-3] + ","
    assert new[-2:] == old[-2:] and len(new) == len(old) + 6
    with pytest.raises(SystemExit, match="importer_selftest is already in workflows.json"):
        insert_entry(path, "importer_selftest", entry)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="not laid out the way this command writes it"):
        insert_entry(path, "another", entry)


def lanes_registry(**weekly):
    return {"lanes": {
        "old": {"cadence": "retired", "endpoint_id": "e0"},
        "main": {"cadence": "per-commit", "endpoint_id": "e1"},
        "night": {"cadence": "nightly", "endpoint_id": None},
        "week": {"cadence": "weekly", "endpoint_id": None, **weekly}}}


def test_weekly_is_refused_while_no_weekly_lane_has_an_endpoint():
    assert cadence_problem("weekly", lanes_registry()) == (
        "the weekly lane is not switched on yet: use --cadence nightly")
    assert cadence_problem("weekly", lanes_registry(endpoint_id="e9")) is None
    # A nightly lane without its endpoint takes a workflow: it runs once the lane has one.
    assert cadence_problem("nightly", lanes_registry()) is None
    assert cadence_problem("per-commit", lanes_registry()) is None
    only = {"lanes": {"main": {"cadence": "per-commit", "endpoint_id": "e1"}}}
    assert cadence_problem("nightly", only) == "manifest/lanes.json has no nightly lane"
    # The registry of this repository names a lane for each cadence a workflow may ask for.
    real = json.loads((ROOT / "manifest" / "lanes.json").read_text(encoding="utf-8"))
    for cadence in ("per-commit", "nightly", "weekly"):
        assert cadence_problem(cadence, real) in (
            None, "the weekly lane is not switched on yet: use --cadence nightly")


def test_the_lint_is_run_whole_and_a_missing_pytest_is_said(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(import_template.subprocess, "run",
                        lambda cmd, cwd: ran.append((cmd, cwd)) or SimpleNamespace(returncode=3))
    assert import_template.run_lint() == 3
    lint = ROOT / "scripts" / "tests" / "test_manifest_lint.py"
    assert ran == [([sys.executable, "-m", "pytest", str(lint), "-q", "-p",
                     "no:cacheprovider"], ROOT)]
    assert "-k" not in ran[0][0], "rules about lanes and the volume are no test of one id"
    monkeypatch.setattr(import_template.importlib.util, "find_spec", lambda name: None)
    assert import_template.run_lint() == 1 and len(ran) == 1
    assert "pytest is not installed. Install pytest (pip install pytest)" in capsys.readouterr().err


def test_a_sub_input_goes_by_the_option_its_combo_chooses():
    """/object_info lists `sampling_mode` and, under each of its options, the
    inputs that option brings. ComfyUI reads `sampling_mode.<name>` when the
    chosen option has <name>, asks for the required ones ("Required input is
    missing") and passes over any other without a word. The importer went by
    `sampling_mode` alone, so it took a text generation switched to sampling
    with none of its values, and one with a value no option has."""
    sampling = ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "on", "inputs": {"required": {"temperature": ["FLOAT", {"default": 0.7}],
                                              "seed": ["INT", {"default": 0}]},
                                 "optional": {"presence_penalty": ["FLOAT", {"default": 0.0}]}}},
        {"key": "off", "inputs": {"required": {}}}]}]
    info = {"TextGenerate": spec({"prompt": ["STRING", {}], "sampling_mode": sampling},
                                 {"mtp": ["COMBO", {"options": ["auto", "off"]}]}),
            "SaveVideo": SERVER_INFO["SaveVideo"],
            "EmptyImage": spec({"width": ["INT", {"default": 512}]}),
            "Maths": spec({"expression": ["STRING", {}], "values": ["COMFY_AUTOGROW_V3", {}]})}

    def checked(class_type, **inputs):
        prompt = {"5": n(class_type, **inputs)}
        return (import_template.unknown_literals(prompt, info),
                import_template.open_required(prompt, info))

    def text(**inputs):
        return checked("TextGenerate", prompt="hi", **inputs)

    on = {"sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.seed": 7}
    assert text(**on) == ([], [])
    assert text(**on, **{"sampling_mode.presence_penalty": 1.5}) == ([], [])
    assert text(sampling_mode="off") == ([], [])
    # Switched on, with none or some of what the option requires.
    assert text(sampling_mode="on") == ([], ["5:TextGenerate.sampling_mode.seed (INT)",
                                            "5:TextGenerate.sampling_mode.temperature (FLOAT)"])
    assert text(sampling_mode="on", **{"sampling_mode.seed": ["4", 0]}) == (
        [], ["5:TextGenerate.sampling_mode.temperature (FLOAT)"])
    # A value no option has, values left over next to another option, and a
    # dot on an input that is no combo: ComfyUI reads none of them.
    assert text(**on, **{"sampling_mode.typical_p": 0.9}) == (
        ["5:TextGenerate.sampling_mode.typical_p"], [])
    assert text(**{**on, "sampling_mode": "off"}) == (
        ["5:TextGenerate.sampling_mode.seed", "5:TextGenerate.sampling_mode.temperature"], [])
    assert checked("EmptyImage", width=64, **{"width.px": 4096}) == (["5:EmptyImage.width.px"], [])
    # Nothing is chosen by a link or by a value that is no option. The combo
    # itself is asked for where the class has it.
    for value in (["4", 0], "always"):
        assert text(sampling_mode=value, **{"sampling_mode.seed": 7}) == (
            ["5:TextGenerate.sampling_mode.seed"], [])
    assert text() == ([], ["5:TextGenerate.sampling_mode (COMFY_DYNAMICCOMBO_V3)"])

    def clip(**inputs):
        return checked("SaveVideo", video=["3", 0], filename_prefix="x", format="mp4", **inputs)

    # An option may hold a combo of its own. ComfyUI does not ask for that
    # one (a SaveVideo without its codec picks one itself), and it asks for
    # what the option chosen there requires.
    assert clip() == clip(**{"format.codec": "h264"}) == ([], [])
    assert clip(**{"format.codec": "h264", "format.codec.encoding": "auto"}) == ([], [])
    assert clip(**{"format.codec": "h264", "format.codec.encoding": "re-encode"}) == (
        [], ["5:SaveVideo.format.codec.encoding.crf (FLOAT)"])
    assert clip(**{"format.codec": "h264", "format.codec.encoding": "re-encode",
                   "format.codec.encoding.crf": 30.0}) == ([], [])
    assert clip(**{"format.codec": "auto", "format.codec.encoding": "auto",
                   "format.profile": "high"}) == (
        ["5:SaveVideo.format.codec.encoding", "5:SaveVideo.format.profile"], [])
    # An input that grows takes its values under names of its own.
    assert checked("Maths", expression="a + b", **{"values.a": 32, "values.b": ["4", 0]}) == (
        [], [])
    required, optional, growing = import_template.chosen_inputs(
        info, n("SaveVideo", format="mp4", **{"format.codec": "h264"}))
    assert (list(required), list(optional), growing) == (
        ["video", "filename_prefix", "format"], ["format.codec", "format.codec.encoding"], [])
    assert import_template.chosen_inputs(info, n("Maths"))[2] == ["values"]
    assert import_template.chosen_inputs(info, n("Unknown", a=1)) == ({}, {}, [])


def test_the_importer_names_the_inputs_the_worker_drops(monkeypatch):
    """The worker drops a literal input whose name, or whose part before the
    first dot, is no input of the class, and keeps every other `a.b`: the
    codec of a video format, the seed of a sampling mode. The importer names
    what the worker drops, and the sub-inputs the worker keeps and ComfyUI
    does not read. It reads nothing under `src/` for it, so this test asks
    the worker's own check about the same prompt."""
    info = {"KSampler": spec({"seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}]}),
            "TextGenerate": SERVER_INFO["TextGenerate"], "SaveVideo": SERVER_INFO["SaveVideo"]}
    prompt = {"3": n("KSampler", seed=5, steps=4, bogus="x", model=["4", 0]),
              "5": n("TextGenerate", prompt="hi", sampling_mode="on",
                     **{"sampling_mode.temperature": 0.7, "sampling_mode.seed": 7,
                        "sampling_mode.typical_p": 0.9, "sampling.seed": 1}),
              "9": n("SaveVideo", format="mp4", video=["3", 0], filename_prefix="x",
                     **{"format.codec": "h264", "format.codec.encoding": "auto",
                        "format.frames": ["3", 1], "old.sub": 1, "filename_prefix.dir": "y"})}
    before = copy.deepcopy(prompt)
    named = import_template.unknown_literals(prompt, info)
    assert named == ["3:KSampler.bogus", "5:TextGenerate.sampling.seed",
                     "5:TextGenerate.sampling_mode.typical_p",
                     "9:SaveVideo.filename_prefix.dir", "9:SaveVideo.old.sub"]
    assert prompt == before
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    import comfy_runner
    monkeypatch.setattr(comfy_runner, "http_get_json", lambda url, timeout=5.0: info)
    report = comfy_runner.validate_workflow(8199, prompt)
    assert report["stripped_inputs"] == ["KSampler.bogus='x'", "TextGenerate.sampling.seed=1",
                                         "SaveVideo.old.sub=1"]
    # The worker queues the rest. What the importer names of it is a
    # sub-input that the chosen option, or an input that is no combo, lacks.
    dropped = {f"{node_id}:{node['class_type']}.{name}" for node_id, node in before.items()
               for name in node["inputs"] if name not in prompt[node_id]["inputs"]}
    assert len(dropped) == 3 and set(named) - dropped == {
        "5:TextGenerate.sampling_mode.typical_p", "9:SaveVideo.filename_prefix.dir"}
    assert {node_id: list(node["inputs"]) for node_id, node in prompt.items()} == {
        node_id: [name for name in node["inputs"]
                  if f"{node_id}:{node['class_type']}.{name}" not in dropped]
        for node_id, node in before.items()}
    # A class nobody knows is the converter's to refuse.
    assert import_template.unknown_literals({"8": n("Unknown", a=1)}, info) == []
    assert import_template.unknown_literals({"3": n("KSampler", seed=5)}, info) == []
    # The importer itself does not read `src/`: what is there is the rule of
    # the next image, and an import is for the one that runs.
    source = Path(import_template.__file__).read_text(encoding="utf-8")
    assert "comfy_runner" not in source and '"src"' not in source


# --------------------------------------------------------------------------- #
# the command, from the template to the manifest
# --------------------------------------------------------------------------- #

SERVER_INFO = {
    "CheckpointLoaderSimple": spec({"ckpt_name": ["COMBO", {"options": []}]},
                                   output=["MODEL", "CLIP", "VAE"]),
    "LoadImage": spec({"image": [["example.png"], {"image_upload": True}]},
                      output=["IMAGE", "MASK"]),
    "Sampler": spec({"model": ["MODEL"], "image": ["IMAGE"],
                     "seed": ["INT", {"default": 0, "control_after_generate": True}],
                     "steps": ["INT", {"default": 20}]}, output=["IMAGE"]),
    "TextGenerate": spec({"prompt": ["STRING", {"default": ""}], "sampling_mode": [
        "COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "on", "inputs": {"required": {"temperature": ["FLOAT", {"default": 0.7}],
                                                  "seed": ["INT", {"default": 0}]}}},
            {"key": "off", "inputs": {"required": {}}}]}]},
        {"mtp": ["COMBO", {"default": "auto", "options": ["auto", "off"]}]}, output=["STRING"]),
    "CreateVideo": INFO["CreateVideo"], "ImageFromBatch": INFO["ImageFromBatch"],
    "GetVideoComponents": INFO["GetVideoComponents"],
    # Every format of the real node asks for a codec, a combo of its own, and
    # h264 may be encoded again at a quality the workflow names.
    "SaveVideo": spec({"video": ["VIDEO"], **PREFIX, "format": ["COMFY_DYNAMICCOMBO_V3", {
        "options": [{"key": key, "inputs": {"required": {"codec": ["COMFY_DYNAMICCOMBO_V3", {
            "options": [{"key": "auto", "inputs": {"required": {}}}, {"key": "h264", "inputs": {
                "required": {}, "optional": {"encoding": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                    {"key": "auto", "inputs": {"required": {}}},
                    {"key": "re-encode", "inputs": {"required": {
                        "crf": ["FLOAT", {"default": 18.0}]}}}]}]}}}],
        }]}}} for key in ("auto", "mp4")]}]}, output_node=True),
    "SaveImage": INFO["SaveImage"], "PreviewImage": INFO["PreviewImage"],
    "PreviewAny": INFO["PreviewAny"], "SaveText": INFO["SaveText"],
    "SaveAudio": INFO["SaveAudio"], "PartnerImageNode": INFO["PartnerImageNode"],
    "ToSound": spec({"image": ["IMAGE"]}, output=["AUDIO"]),
}
MODEL_URL = "https://huggingface.co/Comfy-Org/sd/resolve/main/checkpoints/sd.safetensors"


def small_template(save="SaveImage"):
    """A checkpoint, an input picture, a sampler, a save node and a preview."""
    return {
        "nodes": [
            ui(4, "CheckpointLoaderSimple", ["sd.safetensors"], [1], models=[
                {"name": "sd.safetensors", "url": MODEL_URL + "?download=true",
                 "directory": "checkpoints"}]),
            ui(10, "LoadImage", ["cat.png", "image"], [2]),
            ui(3, "Sampler", [7, "fixed", 20], [3, 4], inputs=[
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "image", "type": "IMAGE", "link": 2}]),
            ui(9, save, ["ComfyUI"], inputs=[{"name": "images", "type": "IMAGE", "link": 3}]),
            ui(11, "PreviewImage", [], inputs=[{"name": "images", "type": "IMAGE", "link": 4}]),
            {"id": 12, "type": "MarkdownNote", "widgets_values": ["a note"]},
        ],
        "links": [[1, 4, 0, 3, 0, "MODEL"], [2, 10, 0, 3, 1, "IMAGE"],
                  [3, 3, 0, 9, 0, "IMAGE"], [4, 3, 0, 11, 0, "IMAGE"]],
    }


MANIFEST = {"schema_version": 1, "defaults": {"timeout_s": 600}, "workflows": {
    "old_wf": {"enabled": True, "models": []}}}
FACTS = {MODEL_URL: ("11" * 32, 2000),
         f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{SHA}/input/cat.png":
             ("22" * 32, 300)}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository the command may write to: a manifest, the lanes, a
    templates folder, and a saved /object_info. The lint is not run: `lint`
    counts the calls. No host is asked for a file."""
    (tmp_path / "manifest" / "workflows").mkdir(parents=True)
    (tmp_path / "templates").mkdir()
    (tmp_path / "docker").mkdir()
    shutil.copy(ROOT / "docker" / "extra_model_paths.yaml", tmp_path / "docker")
    (tmp_path / "manifest" / "workflows.json").write_text(
        json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "manifest" / "lanes.json").write_text(
        json.dumps(lanes_registry()), encoding="utf-8")
    (tmp_path / "templates" / "small.json").write_text(
        json.dumps(small_template()), encoding="utf-8")
    (tmp_path / "object_info.json").write_text(json.dumps(SERVER_INFO), encoding="utf-8")
    monkeypatch.setattr(import_template, "ROOT", tmp_path)
    monkeypatch.setattr(import_template, "file_facts", lambda url: FACTS[url])
    lint = []
    monkeypatch.setattr(import_template, "run_lint", lambda: lint.append("ran") or 0)
    return SimpleNamespace(
        root=tmp_path, lint=lint,
        base=["small", "--id", "small_i2i", "--templates-dir", str(tmp_path / "templates"),
              "--templates-ref", SHA, "--cadence", "nightly"],
        offline=["--object-info", str(tmp_path / "object_info.json")],
        files=lambda: sorted(p.relative_to(tmp_path).as_posix()
                             for p in tmp_path.rglob("*") if p.is_file()),
        manifest=lambda: (tmp_path / "manifest" / "workflows.json").read_text(encoding="utf-8"))


SMALL_PROMPT = {
    "4": n("CheckpointLoaderSimple", ckpt_name="sd.safetensors"),
    "10": n("LoadImage", image="cat.png"),
    "3": n("Sampler", seed=7, steps=4, model=["4", 0], image=["10", 0]),
    "9": n("SaveImage", filename_prefix="small_i2i", images=["3", 0]),
}
SMALL_ENTRY = {
    "enabled": True, "level": "smoke", "cadence": "nightly", "expect": {"width": 512},
    "workflow_path": "manifest/workflows/small_i2i_api.json",
    "seed_overrides": {"3": {"seed": 7}},
    "models": [
        {"name": "sd.safetensors", "url": MODEL_URL, "directory": "checkpoints",
         "sha256": "11" * 32, "bytes": 2000},
        {"name": "cat.png",
         "url": f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{SHA}/input/"
                "cat.png", "directory": "input", "sha256": "22" * 32, "bytes": 300}],
    "comfy_flags": ["--input-directory", "/runpod-volume/models/input"],
    "gpu_type": None, "timeout_s": 900, "thresholds": None, "golden_tag": None}
MORE = ["--timeout-s", "900", "--set", "3.steps=4", "--expect", "width=512"]


def test_one_command_writes_the_workflow_the_entry_and_runs_the_lint(repo, server, capsys):
    server.answer("GET", "/object_info", SERVER_INFO)
    server.answer("POST", "/prompt", status=400, payload={"error": REFUSED, "node_errors": {
        "4": node_error("CheckpointLoaderSimple", NOT_IN_LIST),
        "10": node_error("LoadImage", NO_IMAGE)}})
    before = repo.manifest()
    assert import_template.main([*repo.base, "--server", server.url, *MORE]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert written.read_text(encoding="utf-8") == json.dumps(SMALL_PROMPT, indent=2) + "\n"
    manifest = json.loads(repo.manifest())
    assert list(manifest["workflows"]) == ["old_wf", "small_i2i"]
    assert manifest["workflows"]["small_i2i"] == SMALL_ENTRY
    assert json.dumps(manifest["workflows"]["small_i2i"]) == json.dumps(SMALL_ENTRY), "key order"
    assert repo.manifest().startswith(before[:before.rindex("}\n  }")])
    assert repo.lint == ["ran"]
    # The server was asked for its definitions and got the finished prompt.
    assert server.sent() == [("GET", "/object_info", None),
                             ("POST", "/prompt", {"prompt": SMALL_PROMPT})]
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.splitlines() == [
        "note: no nightly lane has an endpoint yet, so nothing runs the workflow until one has",
        "prompt check: passed. The server does not have cat.png, sd.safetensors, and names "
        "no other problem",
        "wrote manifest/workflows/small_i2i_api.json (4 nodes)",
        "added small_i2i to manifest/workflows.json"]


def test_a_dry_run_prints_the_nodes_the_entry_and_the_check_and_writes_nothing(
        repo, server, capsys):
    server.answer("GET", "/object_info", SERVER_INFO)
    server.answer("POST", "/prompt", {"prompt_id": "p-1", "node_errors": {}})
    server.answer("POST", "/queue")
    server.answer("POST", "/interrupt")
    files, manifest = repo.files(), repo.manifest()
    assert import_template.main([*repo.base, "--server", server.url, *MORE, "--dry-run"]) == 0
    assert repo.files() == files and repo.manifest() == manifest and repo.lint == []
    out = capsys.readouterr().out
    assert out.startswith(
        "nodes (4):\n"
        '   4  CheckpointLoaderSimple  ckpt_name="sd.safetensors"\n'
        '  10  LoadImage  image="cat.png"\n'
        "   3  Sampler  seed=7 steps=4\n"
        '   9  SaveImage  filename_prefix="small_i2i"\n'
        "entry:\n{\n"
        '  "small_i2i": {\n')
    entry, _, rest = out.partition("entry:\n")[2].partition("\nnote: ")
    assert json.loads(entry) == {"small_i2i": SMALL_ENTRY}
    assert rest.splitlines()[1:] == [
        "prompt check: the server accepted the prompt, and it was taken out of the queue",
        "dry run: nothing was written"]
    assert [path for _, path, _ in server.sent()] == ["/object_info", "/prompt", "/queue",
                                                      "/interrupt"]


def test_an_import_stops_when_comfyui_refuses_the_prompt(repo, server, capsys):
    server.answer("GET", "/object_info", SERVER_INFO)
    wrong = ("value_bigger_than_max", "Value 400 bigger than max of 100", "steps", "steps")
    server.answer("POST", "/prompt", status=400, payload={"error": REFUSED, "node_errors": {
        "3": node_error("Sampler", wrong)}})
    files, manifest = repo.files(), repo.manifest()
    for more in ([], ["--dry-run"]):
        assert import_template.main([*repo.base, "--server", server.url, *more]) == 1
        printed = capsys.readouterr()
        assert "prompt check: ComfyUI refuses the prompt\n  node 3 (Sampler): Value 400 bigger " \
               "than max of 100: steps\n" in printed.out + printed.err
        assert repo.files() == files and repo.manifest() == manifest and repo.lint == []


def test_without_a_server_the_prompt_check_is_skipped_and_said(repo, capsys):
    assert import_template.main([*repo.base, *repo.offline, *MORE]) == 0
    assert "prompt check: skipped, there is no --server to ask\n" in capsys.readouterr().err
    assert json.loads(repo.manifest())["workflows"]["small_i2i"] == SMALL_ENTRY
    assert repo.lint == ["ran"]


def test_print_writes_the_workflow_file_and_leaves_the_manifest_alone(repo, capsys):
    manifest = repo.manifest()
    assert import_template.main([*repo.base, *repo.offline, *MORE, "--print"]) == 0
    assert json.loads(capsys.readouterr().out) == {"small_i2i": SMALL_ENTRY}
    assert repo.manifest() == manifest and repo.lint == []
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert json.loads(written.read_text(encoding="utf-8")) == SMALL_PROMPT
    # Without lookups the entry has no sha256 and no size, and no host is asked.
    assert import_template.main([*repo.base, *repo.offline, "--print", "--skip-sha"]) == 0
    models = json.loads(capsys.readouterr().out)["small_i2i"]["models"]
    assert [list(m) for m in models] == [["name", "url", "directory"]] * 2


def test_a_lint_that_fails_fails_the_command_and_says_how_to_take_it_back(
        repo, monkeypatch, capsys):
    monkeypatch.setattr(import_template, "run_lint", lambda: 1)
    assert import_template.main([*repo.base, *repo.offline]) == 1
    assert capsys.readouterr().err.endswith(
        "error: the manifest lint does not pass with the new entry. Fix the entry or the "
        "workflow file, or take both back: remove small_i2i from manifest/workflows.json and "
        "delete manifest/workflows/small_i2i_api.json\n")
    assert "small_i2i" in json.loads(repo.manifest())["workflows"]


def test_what_stops_an_import_before_a_template_is_read(repo, monkeypatch, capsys):
    monkeypatch.setattr(import_template, "load_template", lambda *a: pytest.fail("not read"))
    files = repo.files()

    def refused(*more, base=None):
        assert import_template.main([*(base or repo.base), *repo.offline, *more]) == 1
        printed = capsys.readouterr()
        assert printed.out == "" and repo.files() == files and repo.lint == []
        return printed.err

    weekly = [*repo.base[:-1], "weekly"]
    assert refused(base=weekly) == (
        "error: the weekly lane is not switched on yet: use --cadence nightly\n")
    assert refused("--dry-run", base=weekly).startswith("error: the weekly lane is not")
    assert refused("--timeout-s", "3600") == (
        "error: --timeout-s is above 0 and under 3600, the endpoint's limit for a job\n")
    assert refused("--timeout-s", "0").startswith("error: --timeout-s is above 0")
    assert refused(base=["small", "--id", "old_wf", *repo.base[3:]]) == (
        "error: old_wf is already in manifest/workflows.json: choose another --id\n")
    assert refused(base=["small", "--id", "3d/mesh one", *repo.base[3:]]).startswith(
        "error: the id '3d/mesh one' names the output files")
    assert refused("--keep-sampling", "--level", "reference") == (
        "error: --keep-sampling is for a smoke-level workflow: a reference-level output is "
        "compared with its reference, so its text is generated without sampling\n")
    for expect, said in ((["--expect", "size=3"], "error: --expect: unknown expect key 'size'"),
                         (["--expect", "width=0"], "error: --expect: expect.width must be a "),
                         (["--expect", "width"], "error: --expect width: write <key>=<number>"),
                         (["--expect", "width=512", "--level", "reference"],
                          "error: --expect: expect is for a smoke-level workflow only")):
        assert refused(*expect).startswith(said)
    # A weekly lane with an endpoint takes the workflow.
    (repo.root / "manifest" / "lanes.json").write_text(
        json.dumps(lanes_registry(endpoint_id="e9")), encoding="utf-8")
    monkeypatch.undo()
    monkeypatch.setattr(import_template, "ROOT", repo.root)
    assert import_template.main([*weekly, *repo.offline, "--dry-run", "--skip-sha"]) == 0
    assert '"cadence": "weekly"' in capsys.readouterr().out


def test_a_missing_pytest_stops_the_command_before_it_writes(repo, monkeypatch, capsys):
    """pytest is not in scripts/requirements.txt, and the lint is run with it."""
    monkeypatch.setattr(import_template.importlib.util, "find_spec", lambda name: None)
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline]) == 1
    assert capsys.readouterr().err == (
        "error: the manifest lint is run with pytest, which is not installed: "
        "pip install pytest\n")
    assert repo.files() == files
    # A dry run and --print run no lint, so they need none.
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 0
    assert import_template.main([*repo.base, *repo.offline, "--print", "--skip-sha"]) == 0
    assert repo.lint == []


def write_template(repo, template):
    (repo.root / "templates" / "small.json").write_text(json.dumps(template), encoding="utf-8")


# --------------------------------------------------------------------------- #
# what is written, when, and over what
# --------------------------------------------------------------------------- #


def test_a_file_or_an_id_that_is_taken_is_found_before_anything_is_written(repo, capsys):
    """Each of these was found after the workflow file was written, or not at
    all: a workflow file that is already there was written over, and so was
    the file of an id that differs by case only, on Windows."""
    workflows = repo.root / "manifest" / "workflows"

    def refused(*more, base=None):
        files = {name: (repo.root / name).read_bytes() for name in repo.files()}
        assert import_template.main([*(base or repo.base), *repo.offline, *more]) == 1
        assert {name: (repo.root / name).read_bytes() for name in repo.files()} == files
        assert repo.lint == []
        return capsys.readouterr().err

    # A file no entry names. This repository has one: ltxv_t2v_api.json.
    (workflows / "small_i2i_api.json").write_text('{"mine": true}\n', encoding="utf-8")
    said = ("error: manifest/workflows/small_i2i_api.json is already there, and the command "
            "writes over no workflow file: choose another --id, or delete the file first\n")
    assert refused() == said and refused("--dry-run") == said
    # --print writes the file of its own id, as it did.
    assert import_template.main([*repo.base, *repo.offline, "--print", "--skip-sha"]) == 0
    capsys.readouterr()
    (workflows / "small_i2i_api.json").unlink()
    # The same id in another case is the same file on Windows and macOS.
    other_case = ["small", "--id", "OLD_WF", *repo.base[3:]]
    assert refused(base=other_case) == (
        "error: old_wf is in manifest/workflows.json, and OLD_WF differs from it by case "
        "only: on Windows and macOS both would be one workflow file. Choose another --id\n")
    assert refused("--print", base=other_case).startswith("error: old_wf is in manifest/")
    (workflows / "Small_I2I_api.json").write_text("{}\n", encoding="utf-8")
    for more in ([], ["--print"]):
        assert refused(*more).startswith(
            "error: manifest/workflows/Small_I2I_api.json is already there")
    (workflows / "Small_I2I_api.json").unlink()
    # A manifest in another layout: the command would not rewrite it.
    path = repo.root / "manifest" / "workflows.json"
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    assert refused() == (
        f"error: {path} is not laid out the way this command writes it (2 spaces, one key a "
        "line): rewriting it would change more than the lines this command adds\n")
    # --print leaves the manifest alone, so its layout is not asked about.
    assert import_template.main([*repo.base, *repo.offline, "--print", "--skip-sha"]) == 0


@pytest.mark.parametrize("line_end", [b"\n", b"\r\n"])
def test_the_files_keep_the_line_ends_the_manifest_has(repo, line_end):
    """On Windows a manifest with LF line ends came back with CRLF: every line
    of it was in the diff. The workflow file gets the manifest's line ends."""
    def laid_out(manifest):
        return (json.dumps(manifest, indent=2) + "\n").encode().replace(b"\n", line_end)

    def held(data):
        return data.count(line_end) == data.count(b"\n") > 5 and (
            line_end == b"\r\n" or b"\r" not in data)

    path = repo.root / "manifest" / "workflows.json"
    path.write_bytes(laid_out(MANIFEST))
    before = path.read_bytes()
    assert import_template.main([*repo.base, *repo.offline]) == 0
    after = path.read_bytes()
    assert held(after) and held((repo.root / "manifest" / "workflows"
                                 / "small_i2i_api.json").read_bytes())
    # No line that was there changed, but the one that got its comma.
    assert after.startswith(before[:before.rindex(b"}" + line_end + b"  }")])
    assert "small_i2i" in json.loads(after)["workflows"]
    # --fill-bytes rewrites a manifest the same way.
    sized = repo.root / "sized.json"
    sized.write_bytes(laid_out(_sized_manifest()))
    assert import_template.fill_bytes(sized, lambda url: 5) == []
    assert held(sized.read_bytes()) and b'"bytes": 5' in sized.read_bytes()


def test_a_write_that_fails_leaves_the_manifest_whole_and_no_file_behind(
        repo, monkeypatch, capsys):
    """The manifest was rewritten in place. On a full disk the write ended
    in the middle: 64 KiB of a file that was no longer JSON."""
    replace = import_template.os.replace

    def full_disk(source, target):
        if Path(target).name == "workflows.json":
            raise OSError(28, "No space left on device")
        replace(source, target)

    monkeypatch.setattr(import_template.os, "replace", full_disk)
    files, manifest = repo.files(), repo.manifest()
    assert import_template.main([*repo.base, *repo.offline]) == 1
    # The workflow file went in first and is taken back, and no part file stays.
    assert repo.files() == files and repo.manifest() == manifest and repo.lint == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: nothing was written: [Errno 28] No space left on device")
    monkeypatch.undo()
    # The text goes to a file next to the manifest, which then takes its place.
    path = repo.root / "manifest" / "workflows.json"
    seen = []
    monkeypatch.setattr(import_template.os, "replace", lambda source, target: seen.append(
        (Path(source).name, Path(target).name, Path(source).read_text(encoding="utf-8"))))
    insert_entry(path, "wf_one", {"enabled": True})
    assert [(source, target) for source, target, _ in seen] == [
        ("workflows.json.part", "workflows.json")]
    assert list(json.loads(seen[0][2])["workflows"]) == ["old_wf", "wf_one"]
    assert repo.manifest() == manifest and repo.files() == files


def test_a_manifest_that_changed_while_the_command_ran_is_left_alone(
        repo, monkeypatch, capsys):
    """Two imports at once: the second had read the manifest before the first
    wrote its entry, and wrote back what it had read. The first entry was gone."""
    path = repo.root / "manifest" / "workflows.json"

    def meanwhile(url):
        if "other_wf" not in json.loads(repo.manifest())["workflows"]:
            insert_entry(path, "other_wf", {"enabled": True, "models": []})
        return FACTS[url]

    monkeypatch.setattr(import_template, "file_facts", meanwhile)
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline]) == 1
    assert list(json.loads(repo.manifest())["workflows"]) == ["old_wf", "other_wf"]
    assert repo.files() == files and repo.lint == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        "error: nothing was written: workflows.json changed while this command ran, so it "
        "was left as it is: run the command again")
    # Run again, the entry goes in after the other one.
    assert import_template.main([*repo.base, *repo.offline]) == 0
    assert list(json.loads(repo.manifest())["workflows"]) == ["old_wf", "other_wf", "small_i2i"]


@pytest.mark.parametrize("flag, value, said", [
    ("--expect", "seconds=Infinity", "error: --expect seconds=Infinity: write <key>=<number>"),
    ("--expect", "seconds=1e999", "error: --expect seconds=1e999: write <key>=<number>"),
    ("--set", "3.steps=NaN", "error: --set 3.steps=NaN: the value is not JSON."),
    ("--set", "3.steps=-Infinity", "error: --set 3.steps=-Infinity: the value is not JSON."),
    ("--set", "3.steps=[1, 1e999]", "error: --set 3.steps=[1, 1e999]: the value is not JSON."),
])
def test_a_number_that_is_not_json_is_refused(repo, capsys, flag, value, said):
    """Python reads NaN and Infinity and writes them back: `"seconds": Infinity`
    went into the manifest with exit code 0, and nothing but Python reads that."""
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline, flag, value]) == 1
    assert capsys.readouterr().err.startswith(said)
    assert repo.files() == files
    # A template can hold one too.
    template = small_template()
    template["nodes"][2]["widgets_values"] = [7, "fixed", float("nan")]
    write_template(repo, template)
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 1
    assert capsys.readouterr().err.startswith(
        "error: 3.steps is nan, a number JSON cannot hold\nnodes (4):\n")


# --------------------------------------------------------------------------- #
# what the lint would say of the entry, said before the write
# --------------------------------------------------------------------------- #

OTHER_URL = "https://huggingface.co/first/home/resolve/main/sd.safetensors"
OLD_CAT = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/0f0f0f0/input/cat.png"


def test_a_file_the_manifest_already_has_is_written_as_the_manifest_has_it(repo, capsys):
    """One folder and name is one file on the volume, and the lint wants one
    URL and one sha256 for it. The gallery serves the same file from several
    repositories, and an input file is pinned to another commit on each
    import: 66 of the 159 templates the dry run took ended in a red lint,
    after their files were written."""
    run = [*repo.base, *repo.offline]
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"

    def manifest_has(**model):
        manifest = copy.deepcopy(MANIFEST)
        manifest["workflows"]["old_wf"]["models"] = [
            {"name": "sd.safetensors", "url": OTHER_URL, "directory": "checkpoints", **model},
            {"name": "cat.png", "url": OLD_CAT, "directory": "input", "sha256": "22" * 32,
             "bytes": 300}]
        (repo.root / "manifest" / "workflows.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written.unlink(missing_ok=True)
        capsys.readouterr()
        return manifest["workflows"]["old_wf"]["models"]

    def refused(*more):
        files, manifest = repo.files(), repo.manifest()
        assert import_template.main([*run, *more]) == 1
        assert repo.files() == files and repo.manifest() == manifest
        return capsys.readouterr().err.splitlines()[0]

    # The same file from another repository, and the picture at another commit.
    theirs = manifest_has(sha256="11" * 32, bytes=2000)
    assert import_template.main([*run, "--dry-run"]) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("note: ")]
    assert notes[:2] == [
        "note: checkpoints/sd.safetensors is in the manifest for old_wf, and it is the same "
        f"file by its sha256: the entry takes the URL it has there, {OTHER_URL}",
        "note: input/cat.png is in the manifest for old_wf, and it is the same file by its "
        f"sha256: the entry takes the URL it has there, {OLD_CAT}"]
    assert import_template.main(run) == 0
    assert json.loads(repo.manifest())["workflows"]["small_i2i"]["models"] == theirs
    # Another file under the same name: the volume cannot hold both.
    manifest_has(sha256="33" * 32, bytes=2000)
    assert refused() == refused("--dry-run") == (
        "error: checkpoints/sd.safetensors is another file in the manifest, for old_wf, by "
        "its sha256: a volume holds one file per name. The manifest has "
        f"{OTHER_URL}, the template {MODEL_URL}")
    # Where the manifest has no sha256, the size decides.
    theirs = manifest_has(bytes=2000)
    assert import_template.main(run) == 0
    said = capsys.readouterr().err
    # The entry has no sha256 because the manifest has none: no lookup failed.
    assert "is the same file by its size" in said and "no sha256 found" not in said
    assert json.loads(repo.manifest())["workflows"]["small_i2i"]["models"] == theirs
    manifest_has(bytes=1999)
    assert "is another file in the manifest, for old_wf, by its size" in refused()
    # The manifest's own URL, and the file there is no longer the one it lists.
    manifest_has(url=MODEL_URL, sha256="33" * 32, bytes=2000)
    assert refused() == (
        "error: checkpoints/sd.safetensors is in the manifest for old_wf with another "
        f"sha256: the file at {MODEL_URL} changed since. Correct the manifest first")
    # Without lookups nothing can be said, and that is said.
    manifest_has(sha256="33" * 32, bytes=2000)
    assert import_template.main([*run, "--dry-run", "--skip-sha"]) == 0
    assert ("note: checkpoints/sd.safetensors is in the manifest for old_wf from another URL, "
            "and --skip-sha looked nothing up to say whether it is the same file\n"
            ) in capsys.readouterr().out
    # --print for an id the manifest has: its own old entry is not in the way.
    again = ["small", "--id", "old_wf", *repo.base[3:], *repo.offline, "--print"]
    assert import_template.main(again) == 0
    assert json.loads(capsys.readouterr().out)["old_wf"]["models"][0]["url"] == MODEL_URL


def test_a_model_folder_the_worker_does_not_search_is_refused(repo, capsys):
    """11 templates name background_removal, geometry_estimation and four more
    folders. The lint refuses a folder that docker/extra_model_paths.yaml does
    not map, and the importer said nothing before it wrote."""
    assert import_template.model_folders() >= {"checkpoints", "vae", "text_encoders", "loras"}
    assert "base_path" not in import_template.model_folders()
    # A folder no yaml maps, so the test holds whatever folders the yaml gains.
    template = small_template()
    template["nodes"][0]["properties"]["models"][0]["directory"] = "holograms"
    write_template(repo, template)
    files = repo.files()
    for more in ([], ["--dry-run"], ["--print"], ["--dry-run", "--skip-sha"]):
        assert import_template.main([*repo.base, *repo.offline, *more]) == 1
        printed = capsys.readouterr()
        assert printed.out == "" and printed.err.startswith(
            "error: the worker does not search holograms/ (sd.safetensors): "
            "docker/extra_model_paths.yaml names the models/ folders it does\nnodes (4):\n")
    assert repo.files() == files
    # The yaml of the checkout decides: a folder it gains is taken.
    yaml = repo.root / "docker" / "extra_model_paths.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8") + "    holograms: holograms\n",
                    encoding="utf-8")
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 0


def test_an_imported_seed_that_a_link_feeds_has_its_override(repo, capsys):
    info = {**SERVER_INFO, "PrimitiveInt": spec(
        {"value": ["INT", {"default": 0, "control_after_generate": True}]}, output=["INT"])}
    template = small_template()
    template["nodes"][2]["inputs"].append(
        {"name": "seed", "type": "INT", "link": 5, "widget": {"name": "seed"}})
    template["nodes"].append(ui(20, "PrimitiveInt", [31, "fixed"], [5]))
    template["links"].append([5, 20, 0, 3, 2, "INT"])
    write_template(repo, template)
    (repo.root / "object_info.json").write_text(json.dumps(info), encoding="utf-8")
    assert import_template.main([*repo.base, *repo.offline]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    prompt = json.loads(written.read_text(encoding="utf-8"))
    assert prompt["3"]["inputs"]["seed"] == ["20", 0] and prompt["20"]["inputs"] == {"value": 31}
    entry = json.loads(repo.manifest())["workflows"]["small_i2i"]
    assert entry["seed_overrides"] == {"20": {"value": 31}}


def test_a_file_input_the_entry_cannot_list_is_refused(repo, capsys):
    """image_anima_lllite_image_inpainting paints its mask in the browser:
    Painter.mask names a file of the temp folder of one session. The dry run
    said nothing, and the run would have ended where Painter opens the file."""
    info = {"Painter": spec({"mask": ["STRING", {"image_upload": True, "default": ""}]},
                            {"image": ["IMAGE"]}),
            "LoadImage": LOADERS["LoadImage"], "LoadVideo": LOADERS["LoadVideo"],
            "CLIPTextEncode": INFO["CLIPTextEncode"]}
    fine = {"1": n("LoadImage", image="cat.png [input]"), "2": n("LoadVideo", file="clip.mp4"),
            "3": n("Painter", mask="", image=["1", 0]),
            "4": n("CLIPTextEncode", text="a note [temp]"), "5": n("Unknown", image="x [temp]")}
    import_template.refuse_stray_files(fine, info)
    for node, said in (
        (n("Painter", mask="painter/p-694.png [temp]"),
         "node 3 (Painter) reads painter/p-694.png [temp]: a file of the temp folder of the "
         "machine the template was made on, which no run has"),
        (n("LoadImage", image="ComfyUI_00001_.png [output]"),
         "node 3 (LoadImage) reads ComfyUI_00001_.png [output]: a file of the output folder "
         "of the machine the template was made on, which no run has"),
        (n("Painter", mask="mask.png"),
         "node 3 (Painter) reads the file mask.png through mask, and the importer lists the "
         "input files of LoadImage, LoadAudio, LoadVideo only: write the workflow by hand"),
    ):
        with pytest.raises(ConversionError) as refusal:
            import_template.refuse_stray_files({**fine, "3": node}, info)
        assert str(refusal.value) == said
    # The command says it before it writes, and --set can name a file that is there.
    template = small_template()
    template["nodes"][1]["widgets_values"] = ["clipspace/cat.png [temp]", "image"]
    write_template(repo, template)
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 1
    assert capsys.readouterr().err.startswith(
        "error: node 10 (LoadImage) reads clipspace/cat.png [temp]: a file of the temp "
        "folder of the machine the template was made on, which no run has\nnodes (4):\n")
    assert import_template.main([*repo.base, *repo.offline, "--set", '10.image="cat.png"']) == 0
    assert sorted(set(repo.files()) - set(files)) == ["manifest/workflows/small_i2i_api.json"]


def test_a_required_input_with_neither_a_link_nor_a_value_is_refused(repo, capsys):
    """image_moss_image_t2i_bboxes draws its boxes on a CreateBoundingBoxes.
    The widget that holds them has the type BOUNDING_BOXES, which the
    converter does not read, so the boxes were left out without a word. A
    server refused the prompt: "Required input is missing: editor_state".
    With a saved /object_info the command wrote the file and ended with 0."""
    boxes = spec({"width": ["INT", {"default": 1024}],
                  "editor_state": ["BOUNDING_BOXES", {"default": [], "socketless": False}]},
                 {"last_incoming": ["BOUNDING_BOXES", {"default": [], "socketless": True}]},
                 output=["IMAGE"])
    info = {**SERVER_INFO, "Boxes": boxes, "Batch": spec(
        {"images": ["COMFY_AUTOGROW_V3", {}], "mode": [["a", "b"], {}]}, output=["IMAGE"])}
    prompt = {"1": n("Boxes", width=512), "2": n("Batch", **{"images.image0": ["1", 0]}),
              "3": n("Sampler", seed=7, model=["4", 0]), "8": n("Unknown")}
    before = copy.deepcopy(prompt)
    # An optional input is not asked for, and neither is one that grows: its
    # values have names of their own. A class nobody knows is the converter's.
    assert import_template.open_required(prompt, info) == [
        "1:Boxes.editor_state (BOUNDING_BOXES)", "2:Batch.mode",
        "3:Sampler.image (IMAGE)", "3:Sampler.steps (INT)"]
    assert prompt == before
    # A link fills an input, and so does a value: a list goes the way the frontend sends it.
    whole = {"1": n("Boxes", width=512, editor_state={"__value__": []}),
             "2": n("Batch", mode="a"), "3": n("Sampler", seed=7, steps=4, model=["4", 0],
                                                image=["1", 0])}
    assert import_template.open_required(whole, info) == []
    # The command says it before it writes, and with no server to ask.
    template = small_template()
    template["nodes"][1] = ui(10, "Boxes", [512, [{"x": 0, "y": 8}], []], [2])
    write_template(repo, template)
    (repo.root / "object_info.json").write_text(json.dumps(info), encoding="utf-8")
    files = repo.files()
    for more in ([], ["--dry-run"], ["--print"], ["--dry-run", "--skip-sha"]):
        assert import_template.main([*repo.base, *repo.offline, *more]) == 1
        printed = capsys.readouterr()
        assert printed.out == "" and printed.err.startswith(
            "error: ComfyUI refuses a prompt without these required inputs, and the converter "
            "found neither a link nor a value for them: 10:Boxes.editor_state "
            "(BOUNDING_BOXES). The template leaves the input open, holds its value in a "
            "widget of a type the converter does not read, or does not choose the option "
            "the input belongs to. Give the value with --set, or write the workflow by "
            "hand\nnodes (4):\n")
    assert repo.files() == files and repo.lint == []
    # --set gives the value the template holds.
    given = '10.editor_state={"__value__": [{"x": 0, "y": 8}]}'
    assert import_template.main([*repo.base, *repo.offline, "--set", given]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert json.loads(written.read_text(encoding="utf-8"))["10"] == n(
        "Boxes", width=512, editor_state={"__value__": [{"x": 0, "y": 8}]})


# --------------------------------------------------------------------------- #
# files and answers that are not what they should be
# --------------------------------------------------------------------------- #


def test_a_web_page_is_not_hashed_as_the_file(server):
    """A host that answers a download with its sign-in page sends 200 and
    HTML. Its sha256 went into the entry, and the model sync would have
    fetched the same page and found the sha256 right."""
    page = b"<!doctype html><html><body>Sign in to download</body></html>"
    html = {"Content-Type": "text/html; charset=utf-8"}
    server.answer("HEAD", "/api/download/models/1", page, **html)
    server.answer("GET", "/api/download/models/1", page, **html)
    url = f"{server.url}/api/download/models/1"
    for ask in (import_template.hash_download, import_template.file_facts):
        with pytest.raises(ConversionError) as refusal:
            ask(url)
        assert str(refusal.value) == (
            f"{url} answers with a web page, not with the file: a sign-in page, or the page "
            "of the file where its download link belongs")
    # Anything else a 200 sends is the file.
    data = b"weights" * 10
    for method in ("HEAD", "GET"):
        server.answer(method, "/x.safetensors", data, **{"Content-Type": "binary/octet-stream"})
    assert import_template.file_facts(f"{server.url}/x.safetensors") == (
        hashlib.sha256(data).hexdigest(), 70)


def test_a_prompt_that_stays_in_the_queue_is_not_called_removed(server, monkeypatch):
    """The answers to /queue and /interrupt were not read: a server that took
    the prompt and refused both was said to have had it taken out."""
    server.answer("POST", "/prompt", {"prompt_id": "abc-1", "node_errors": {}})
    server.answer("POST", "/queue", b"no", status=500)
    server.answer("POST", "/interrupt", b"no", status=403)
    start = "prompt check: the server accepted the prompt as abc-1, and taking it out of the "
    assert check_prompt(server.url, CHECKED, FILES, set()) == (True, [
        start + "queue failed (/queue answered 500, /interrupt answered 403): cancel it by hand"])
    # Both are asked, whatever the first one answers.
    assert [path for _, path, _ in server.sent()] == ["/prompt", "/queue", "/interrupt"]
    server.answer("POST", "/queue")
    assert check_prompt(server.url, CHECKED, FILES, set()) == (True, [
        start + "queue failed (/interrupt answered 403): cancel it by hand"])
    post = import_template._post
    monkeypatch.setattr(import_template, "_post", lambda url, body: post(url, body) if url.endswith(
        "/prompt") else (_ for _ in ()).throw(ConnectionResetError("gone")))
    assert check_prompt(server.url, CHECKED, FILES, set()) == (True, [
        start + "queue failed (/queue: gone, /interrupt: gone): cancel it by hand"])


@pytest.mark.parametrize("answer, said", [
    ({"node_errors": ["x"]}, "  node_errors: ['x']"),
    ({"node_errors": {"3": "boom"}}, "  node 3: 'boom'"),
    ({"node_errors": {"3": {"errors": "boom"}}}, "  node 3: {'errors': 'boom'}"),
    ({"node_errors": {"3": {"errors": ["boom"]}}}, "  node 3 (KSampler): 'boom'"),
    ({"node_errors": {"4": {"errors": [{"type": "value_not_in_list", "message": "Value not in "
                                        "list", "extra_info": "ckpt_name"}]}}},
     "  node 4 (CheckpointLoaderSimple): Value not in list: None"),
    ({"node_errors": {"4": {"errors": [{"type": "value_not_in_list", "message": "odd",
                                        "extra_info": {"input_name": ["ckpt_name"]}}]}}},
     "  node 4 (CheckpointLoaderSimple): odd: None"),
])
def test_an_answer_that_is_not_what_comfyui_sends_is_a_failed_check(server, answer, said):
    """Each of these ended the command in a traceback."""
    server.answer("POST", "/prompt", answer, status=400)
    assert check_prompt(server.url, CHECKED, FILES, set()) == (
        False, ["prompt check: ComfyUI refuses the prompt", said])


def test_a_prompt_id_that_is_no_text_cannot_be_taken_out(server):
    server.answer("POST", "/prompt", {"prompt_id": {"a": 1}, "node_errors": {}})
    passed, lines = check_prompt(server.url, CHECKED, FILES, set())
    assert not passed and lines == ["prompt check: ComfyUI refuses the prompt", (
        f"  {server.url}/prompt answered 200 with a prompt id that is no text ({{'a': 1}}): "
        "if the server queued it, cancel it by hand")]
    assert [path for _, path, _ in server.sent()] == ["/prompt"]


@pytest.fixture
def raw_server():
    """A server that answers every request with the same bytes and hangs up."""
    sockets = []

    def serve(payload):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(5)
        sockets.append(sock)

        def answer():
            while True:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    return
                conn.settimeout(2)
                try:
                    conn.recv(65536)
                    conn.sendall(payload)
                except OSError:
                    pass
                conn.close()

        threading.Thread(target=answer, daemon=True).start()
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield serve
    for sock in sockets:
        sock.close()


CUT_SHORT = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
             b"Content-Length: 500\r\n\r\n{\"a\":")
NOT_HTTP = b"\x00\x01garbage\r\n\r\n"


@pytest.mark.parametrize("payload", [CUT_SHORT, NOT_HTTP], ids=["cut short", "not http"])
def test_an_answer_that_is_no_http_is_an_error_line(repo, raw_server, capsys, payload):
    """http.client has errors of its own that are no OSError: an answer that
    is cut short, a TLS port asked over http."""
    url = raw_server(payload)
    passed, lines = check_prompt(url, CHECKED, FILES, set())
    assert not passed and len(lines) == 1
    assert lines[0].startswith(f"prompt check: no answer from {url}/prompt (")
    assert import_template.main([*repo.base, "--server", url, "--dry-run"]) == 1
    assert capsys.readouterr().err.startswith(
        f"error: could not read the node definitions from {url}/object_info: ")
    assert import_template.hash_download(f"{url}/x") is None
    if payload is NOT_HTTP:
        assert import_template._head(f"{url}/x", follow=False) == (None, {})


@pytest.mark.parametrize("text", ["[]", '"text"', "7", "null", '{"Sampler": 5}',
                                  '{"Sampler": {"input": [1]}}', '{"Sampler": {"input": 5}}',
                                  '{"Sampler": {"input": {"required": [1]}}}'])
def test_a_file_that_is_no_object_info_is_an_error_line(repo, capsys, text):
    """The templates' own index.json is a list, and is JSON."""
    path = repo.root / "object_info.json"
    path.write_text(text, encoding="utf-8")
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 1
    assert capsys.readouterr().err == (
        f"error: {path} holds no node definitions: /object_info is an object with one entry "
        "for each node class\n")


@pytest.mark.parametrize("text", ['{"nodes": [{}]}', '{"nodes": [1]}', '{"nodes": [null]}',
                                  '{"nodes": [{"id": 1, "type": "SaveImage", "inputs": 5}]}',
                                  '{"nodes": [{"id": 1, "type": "SaveImage"}], "links": [[1]]}'])
def test_a_template_that_is_json_and_no_workflow_is_refused_with_a_reason(repo, capsys, text):
    (repo.root / "templates" / "small.json").write_text(text, encoding="utf-8")
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and "Traceback" not in printed.err
    assert printed.err.startswith("error: ") and repo.files() == files
    if "the converter can read" in printed.err:
        assert re.match(r"error: the template is not a workflow the converter can read "
                        r"\((Attribute|Index|Key|Type|Value)Error: .+\)\n", printed.err)


def test_a_ref_github_answers_with_no_text_is_an_error_line(monkeypatch):
    monkeypatch.setattr(import_template, "_get", lambda url, accept=None: b"\xff\xfe\x00junk")
    with pytest.raises(ConversionError, match="could not resolve main in Comfy-Org/workflow_"
                                              "templates .*: pass a full commit sha"):
        import_template.resolve_ref("main")


def test_a_refusal_names_its_reason_prints_the_nodes_and_writes_nothing(repo, capsys):
    files, manifest = repo.files(), repo.manifest()

    def refused(*more):
        assert import_template.main([*repo.base, *repo.offline, *more]) == 1
        printed = capsys.readouterr()
        assert "Traceback" not in printed.err and printed.out == ""
        assert repo.files() == files and repo.manifest() == manifest and repo.lint == []
        return printed.err

    # An unknown node or input of --set prints the nodes whose ids it may name.
    assert refused("--set", "7.steps=4") == (
        "error: --set 7.steps=4: the workflow has no node 7\n"
        "nodes (4):\n"
        '   4  CheckpointLoaderSimple  ckpt_name="sd.safetensors"\n'
        '  10  LoadImage  image="cat.png"\n'
        "   3  Sampler  seed=7 steps=20\n"
        '   9  SaveImage  filename_prefix="small_i2i"\n')
    assert refused("--set", "3.cfg=1", "--dry-run").startswith(
        "error: --set 3.cfg=1: node 3 (Sampler) has no input cfg\nnodes (4):\n")
    # A template that is not there, and one this ComfyUI lacks a node for.
    assert refused("--templates-dir", str(repo.root)).startswith(
        "error: could not read the template small: ")
    template = small_template()
    template["nodes"][2]["type"] = "SamplerFromAPack"
    write_template(repo, template)
    assert refused() == "error: node 3 has unknown type 'SamplerFromAPack'\n"
    # A save node no check reads: the nodes as the template has them are shown.
    write_template(repo, small_template(save="SaveWEBM"))
    info = {**SERVER_INFO, "SaveWEBM": INFO["SaveWEBM"]}
    (repo.root / "object_info.json").write_text(json.dumps(info), encoding="utf-8")
    said = refused()
    assert said.startswith("error: the template also saves with SaveWEBM, and no check reads "
                           "what that writes: export the workflow by hand and leave the node "
                           "out\nnodes (5):\n")
    assert '   9  SaveWEBM  filename_prefix="ComfyUI"\n  11  PreviewImage\n' in said


def test_a_partner_node_is_refused(repo, capsys):
    """It calls a paid service, and the worker has no account to pay with."""
    template = small_template()
    template["nodes"][2] = ui(3, "PartnerImageNode", ["a cat"], [3, 4])
    write_template(repo, template)
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 1
    assert capsys.readouterr().err.startswith(
        "error: the template calls a paid service through PartnerImageNode: a partner node "
        "needs an account key and credits, and the worker has neither\nnodes (2):\n")


def text_template(save):
    return {"nodes": [ui(5, "TextGenerate", ["hi", "on", 0.7, 7, "randomize", "auto"], [1]),
                      ui(7, save, [], inputs=[{"name": "source", "type": "*", "link": 1}])],
            "links": [[1, 5, 0, 7, 0, "STRING"]]}


def test_a_text_template_is_imported_greedy(repo, capsys):
    write_template(repo, text_template("PreviewAny"))
    assert import_template.main([*repo.base, *repo.offline, "--level", "reference",
                                 "--cadence", "per-commit"]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "5": n("TextGenerate", prompt="hi", sampling_mode="off", mtp="off"),
        "7": n("SaveText", text=["5", 0], filename_prefix="small_i2i", format="txt")}
    entry = json.loads(repo.manifest())["workflows"]["small_i2i"]
    assert (entry["level"], entry["cadence"], entry["output_kind"]) == (
        "reference", "per-commit", "text")
    assert entry["seed_overrides"] == {} and entry["models"] == []
    assert "comfy_flags" not in entry and "expect" not in entry


def test_keep_sampling_leaves_a_smoke_level_text_generation_as_the_template_has_it(
        repo, capsys):
    """A sampled text generation has `sampling_mode.temperature` and
    `sampling_mode.seed`, sub-inputs of its sampling mode. While the worker
    dropped such inputs the flag was refused. Now the workflow file has what
    the template has, and the entry pins the seed under its whole name."""
    write_template(repo, text_template("PreviewAny"))
    sampled = {"prompt": "hi", "sampling_mode": "on", "sampling_mode.temperature": 0.7,
               "sampling_mode.seed": 7, "mtp": "auto"}
    files = repo.files()
    assert import_template.main([*repo.base, *repo.offline, "--keep-sampling", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "nodes (2):\n"
        '  5  TextGenerate  prompt="hi" sampling_mode="on" sampling_mode.temperature=0.7 '
        'sampling_mode.seed=7 mtp="auto"\n'
        '  7  SaveText  filename_prefix="small_i2i" format="txt"\n')
    assert '"seed_overrides": {\n      "5": {\n        "sampling_mode.seed": 7\n' in out
    assert repo.files() == files and repo.lint == []
    assert import_template.main([*repo.base, *repo.offline, "--keep-sampling"]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "5": n("TextGenerate", **sampled),
        "7": n("SaveText", text=["5", 0], filename_prefix="small_i2i", format="txt")}
    entry = json.loads(repo.manifest())["workflows"]["small_i2i"]
    assert (entry["level"], entry["output_kind"]) == ("smoke", "text")
    assert entry["seed_overrides"] == {"5": {"sampling_mode.seed": 7}}
    assert repo.lint == ["ran"]
    # Without the flag a smoke-level text is greedy too, and has no seed to pin.
    assert import_template.main([*repo.base[:2], "greedy", *repo.base[3:], *repo.offline]) == 0
    written = repo.root / "manifest" / "workflows" / "greedy_api.json"
    assert json.loads(written.read_text(encoding="utf-8"))["5"] == n(
        "TextGenerate", prompt="hi", sampling_mode="off", mtp="off")
    assert json.loads(repo.manifest())["workflows"]["greedy"]["seed_overrides"] == {}


def test_a_text_generation_switched_to_sampling_needs_what_sampling_requires(repo, capsys):
    """`--set 5.sampling_mode="on"` on a text generation the command had made
    greedy left it with none of the values sampling requires. With a saved
    /object_info the command wrote that workflow and the lint passed it. A
    server refuses the prompt: "Required input is missing: temperature"."""
    write_template(repo, text_template("PreviewAny"))
    files = repo.files()
    on = ["--set", '5.sampling_mode="on"']
    for more in ([], ["--dry-run"]):
        assert import_template.main([*repo.base, *repo.offline, *on, *more]) == 1
        printed = capsys.readouterr()
        assert printed.out == "" and printed.err.startswith(
            "error: ComfyUI refuses a prompt without these required inputs, and the converter "
            "found neither a link nor a value for them: 5:TextGenerate.sampling_mode.seed "
            "(INT), 5:TextGenerate.sampling_mode.temperature (FLOAT). The template leaves the "
            "input open, holds its value in a widget of a type the converter does not read, "
            "or does not choose the option the input belongs to. Give the value with --set, "
            "or write the workflow by hand\nnodes (2):\n"
            '  5  TextGenerate  prompt="hi" sampling_mode="on" mtp="off"\n')
    # --set gives the values, after the one that chooses their option.
    values = ["--set", "5.sampling_mode.temperature=0.7", "--set", "5.sampling_mode.seed=7"]
    assert import_template.main([*repo.base, *repo.offline, *values, *on]) == 1
    assert capsys.readouterr().err.startswith(
        "error: --set 5.sampling_mode.temperature=0.7: node 5 (TextGenerate) has no input "
        "sampling_mode.temperature\n")
    # At reference level the text stays greedy, whatever asks for another:
    # the lint would refuse the entry after it was written.
    for more, said in (([*on, *values], "5: sampling_mode must be 'off' (greedy decoding)"),
                       (["--set", '5.mtp="auto"'], "5: mtp must be set, and not to 'auto'")):
        for dry in ([], ["--dry-run"]):
            assert import_template.main([*repo.base, *repo.offline, "--level", "reference",
                                         *more, *dry]) == 1
            assert capsys.readouterr().err.startswith(
                "error: a reference-level output is compared with its reference, so its text "
                f"is generated the same on every run: {said}. The lint asks this of every "
                "reference-level workflow: use --level smoke for one that samples\nnodes (2):\n")
    # Sampling values next to a mode that reads none of them say nothing to
    # ComfyUI, and the run would go on without them.
    assert import_template.main([*repo.base, *repo.offline, "--keep-sampling", "--set",
                                 '5.sampling_mode="off"']) == 1
    assert capsys.readouterr().err.startswith(
        "error: the node of this ComfyUI does not have these inputs, so a run would go on "
        "without the values they hold: 5:TextGenerate.sampling_mode.seed, "
        "5:TextGenerate.sampling_mode.temperature. The template was made for another version "
        "of the node, or a --set chose an option that does not take them\nnodes (2):\n")
    assert repo.files() == files and repo.lint == []
    assert import_template.main([*repo.base, *repo.offline, *on, *values]) == 0
    written = repo.root / "manifest" / "workflows" / "small_i2i_api.json"
    assert json.loads(written.read_text(encoding="utf-8"))["5"] == n(
        "TextGenerate", prompt="hi", sampling_mode="on", mtp="off",
        **{"sampling_mode.temperature": 0.7, "sampling_mode.seed": 7})
    entry = json.loads(repo.manifest())["workflows"]["small_i2i"]
    assert (entry["level"], entry["seed_overrides"]) == ("smoke", {"5": {"sampling_mode.seed": 7}})


def clip_template():
    """The small template with a clip made of its picture, and saved."""
    template = small_template()
    template["nodes"][3] = ui(9, "SaveVideo", ["video/ComfyUI", "auto", "auto"], inputs=[
        {"name": "video", "type": "VIDEO", "link": 5}])
    template["nodes"].append(ui(13, "CreateVideo", [24.0], [5], inputs=[
        {"name": "images", "type": "IMAGE", "link": 3}]))
    template["links"][2] = [3, 3, 0, 13, 0, "IMAGE"]
    template["links"].append([5, 13, 0, 9, 0, "VIDEO"])
    return template


def test_a_template_that_saves_a_clip_is_imported_with_its_codec(repo, capsys):
    """`SaveVideo` takes its codec as `format.codec`, a sub-input of its
    format. While the worker dropped such inputs every template that saves a
    clip was refused."""
    write_template(repo, clip_template())
    assert import_template.main([*repo.base, *repo.offline]) == 0
    written = json.loads((repo.root / "manifest" / "workflows" / "small_i2i_api.json")
                         .read_text(encoding="utf-8"))
    assert written["9"] == n("SaveVideo", filename_prefix="small_i2i_video", format="mp4",
                             **{"format.codec": "h264"}, video=["13", 0])
    # The picture of the run is the clip's last frame.
    assert written["14"] == n("ImageFromBatch", image=["3", 0], batch_index=-1, length=1)
    assert written["15"] == n("SaveImage", filename_prefix="small_i2i", images=["14", 0])
    entry = json.loads(repo.manifest())["workflows"]["small_i2i"]
    assert "output_kind" not in entry and entry["seed_overrides"] == {"3": {"seed": 7}}
    said = capsys.readouterr().err
    assert repo.lint == ["ran"] and "error" not in said
    # The limit a smoke-level clip is held to was measured on made-up clips
    # only, and the README asks for clips of the model first. While every
    # template with a clip was refused, nobody had to be told.
    note = ("note: node 9 saves a clip, and the limit the smoke check holds a clip to rests "
            "on made-up clips: measure clips of this model against it before the workflow "
            "is merged (README, Smoke level)\n")
    assert note in said
    assert import_template.main([*repo.base[:2], "compared", *repo.base[3:], *repo.offline,
                                 "--level", "reference"]) == 0
    assert "saves a clip" not in capsys.readouterr().err


def test_audio_has_no_reference_level_yet(repo, capsys):
    template = small_template(save="SaveAudio")
    template["nodes"][3]["inputs"][0] = {"name": "audio", "type": "AUDIO", "link": 5}
    template["nodes"].append(ui(13, "ToSound", [], [5], inputs=[
        {"name": "image", "type": "IMAGE", "link": 3}]))
    template["links"][2] = [3, 3, 0, 13, 0, "IMAGE"]
    template["links"].append([5, 13, 0, 9, 0, "AUDIO"])
    write_template(repo, template)
    assert import_template.main([*repo.base, *repo.offline, "--level", "reference"]) == 1
    assert capsys.readouterr().err.startswith(
        "error: the template's output is audio, and nothing compares audio yet: use --level "
        "smoke\nnodes (5):\n")
    assert import_template.main([*repo.base, *repo.offline, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"output_kind": "audio"' in out
    assert '   9  SaveAudioAdvanced  filename_prefix="small_i2i" format="flac"\n' in out


def test_the_template_comes_from_github_at_a_pinned_commit(repo, monkeypatch, capsys):
    """Without a checkout the template is fetched, and the branch is resolved
    to a commit first, so the input URLs the entry gets do not move."""
    asked = []

    def get(url, accept=None):
        asked.append((url, accept))
        if url.endswith("/commits/main"):
            return (SHA + "\n").encode()
        if url.endswith("/templates/small.json"):
            return json.dumps(small_template()).encode()
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(import_template, "_get", get)
    base = ["small", "--id", "small_i2i", "--cadence", "nightly", *repo.offline, "--dry-run"]
    assert import_template.main(base) == 0
    raw = f"https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{SHA}"
    assert asked == [
        ("https://api.github.com/repos/Comfy-Org/workflow_templates/commits/main",
         "application/vnd.github.sha"),
        (f"{raw}/templates/small.json", None)]
    assert f'"url": "{raw}/input/cat.png"' in capsys.readouterr().out
    # A full sha is taken as it is. A name GitHub does not know stops the import.
    asked.clear()
    assert import_template.main([*base, "--templates-ref", SHA]) == 0
    assert [url for url, _ in asked] == [f"{raw}/templates/small.json"]
    capsys.readouterr()
    assert import_template.main([*base, "--templates-ref", "no-such-branch"]) == 1
    assert capsys.readouterr().err.startswith(
        "error: could not resolve no-such-branch in Comfy-Org/workflow_templates (HTTP Error "
        "404: Not Found): pass a full commit sha with --templates-ref\n")


def test_a_checkout_pins_its_input_files_to_its_own_commit(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    (tmp_path / "templates" / "more").mkdir(parents=True)
    (tmp_path / "templates" / "a.json").write_text("{}", encoding="utf-8")
    assert import_template.checkout_sha(tmp_path / "templates") is None  # no repository
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "templates")
    assert import_template.checkout_sha(tmp_path / "templates") == git("rev-parse", "HEAD")
    # Any other folder of a repository is not the templates folder of a checkout.
    assert import_template.checkout_sha(tmp_path) is None
    assert import_template.checkout_sha(tmp_path / "templates" / "more") is None
    assert import_template.checkout_sha(tmp_path / "missing") is None


# --------------------------------------------------------------------------- #
# the two audio workflows, and the real lint
# --------------------------------------------------------------------------- #

AUDIO_FACTS = {m["url"]: (m["sha256"], m["bytes"]) for entry in json.loads(
    (FIXTURES / "expected_entries.json").read_text(encoding="utf-8")).values()
    for m in entry["models"]}


@pytest.mark.parametrize("template, workflow_id, more", [
    ("audio_ace_step_1_5_checkpoint", "ace_step_15_t2a",
     ["--cadence", "nightly", "--timeout-s", "900", "--set", "94.duration=30",
      "--set", "98.seconds=30", "--expect", "seconds=30"]),
    ("audio_minimax_music_3", "minimax_music3_t2a",
     ["--cadence", "weekly", "--timeout-s", "1200", "--set", "37:13.max_duration=30"]),
])
def test_the_two_audio_templates_import_as_the_files_made_by_hand(
        repo, monkeypatch, capsys, template, workflow_id, more):
    """`ace_step_15_t2a` and `minimax_music3_t2a` were written by hand from
    the converter's output. The command gives the same workflow file, byte for
    byte, and the same entry."""
    (repo.root / "manifest" / "lanes.json").write_text(
        json.dumps(lanes_registry(endpoint_id="e9")), encoding="utf-8")
    monkeypatch.setattr(import_template, "file_facts", lambda url: AUDIO_FACTS[url])
    argv = [template, "--id", workflow_id, "--templates-dir", str(FIXTURES),
            "--object-info", str(FIXTURES / "object_info.json"), *more]
    assert import_template.main(argv) == 0
    written = repo.root / "manifest" / "workflows" / f"{workflow_id}_api.json"
    by_hand = FIXTURES / f"{workflow_id}_api.json"
    assert written.read_text(encoding="utf-8") == by_hand.read_text(encoding="utf-8")
    entry = json.loads(repo.manifest())["workflows"][workflow_id]
    expected = json.loads((FIXTURES / "expected_entries.json").read_text(encoding="utf-8"))
    assert entry == expected[workflow_id]
    assert list(entry) == list(expected[workflow_id]), "the keys in the same order"
    # A dry run prints the same entry and writes nothing more.
    files = repo.files()
    capsys.readouterr()
    assert import_template.main([*argv, "--id", f"{workflow_id}_2", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out.partition("entry:\n")[2].partition("\nnote")[0]
                         .partition("\nprompt check")[0])
    assert printed[f"{workflow_id}_2"]["models"] == entry["models"] and repo.files() == files


def test_an_imported_workflow_passes_the_real_manifest_lint(tmp_path, monkeypatch):
    """The command on a copy of this repository, with the lint it really runs."""
    for folder in ("scripts", "manifest", "docker"):
        shutil.copytree(ROOT / folder, tmp_path / folder,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "small.json").write_text(
        json.dumps(small_template()), encoding="utf-8")
    (tmp_path / "object_info.json").write_text(json.dumps(SERVER_INFO), encoding="utf-8")
    monkeypatch.setattr(import_template, "ROOT", tmp_path)
    monkeypatch.setattr(import_template, "file_facts", lambda url: FACTS[url])
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    before = (tmp_path / "manifest" / "workflows.json").read_text(encoding="utf-8")
    argv = ["small", "--id", "importer_selftest", "--templates-dir", str(tmp_path / "templates"),
            "--templates-ref", SHA, "--object-info", str(tmp_path / "object_info.json"),
            "--cadence", "nightly", "--timeout-s", "900", "--expect", "width=512"]
    assert import_template.main(argv) == 0, "the lint of the copy passes with the new entry"
    after = (tmp_path / "manifest" / "workflows.json").read_text(encoding="utf-8")
    assert list(json.loads(after)["workflows"])[-1] == "importer_selftest"
    assert not [line for line in before.splitlines()[:-3] if line not in after.splitlines()]
    # The lint does look at the new entry: with a wrong prefix the command fails.
    path = tmp_path / "manifest" / "workflows" / "importer_selftest_api.json"
    path.write_text(path.read_text(encoding="utf-8").replace(
        '"importer_selftest"', '"other"'), encoding="utf-8")
    assert import_template.run_lint() == 1


def test_a_clip_and_a_sampled_text_pass_the_real_manifest_lint_at_smoke_level(
        tmp_path, monkeypatch, capfd):
    """The lint refused every literal sub-input while the worker dropped
    them. Now the codec of a clip passes, and a sampled text generation
    passes at smoke level, with its seed pinned. At reference level the lint
    still asks for a greedy one, and says which node is not."""
    for folder in ("scripts", "manifest", "docker"):
        shutil.copytree(ROOT / folder, tmp_path / folder,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (tmp_path / "templates").mkdir()
    for name, template in (("clip", clip_template()), ("text", text_template("PreviewAny"))):
        (tmp_path / "templates" / f"{name}.json").write_text(
            json.dumps(template), encoding="utf-8")
    (tmp_path / "object_info.json").write_text(json.dumps(SERVER_INFO), encoding="utf-8")
    monkeypatch.setattr(import_template, "ROOT", tmp_path)
    monkeypatch.setattr(import_template, "file_facts", lambda url: FACTS[url])
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    more = ["--templates-dir", str(tmp_path / "templates"), "--templates-ref", SHA,
            "--object-info", str(tmp_path / "object_info.json"), "--cadence", "nightly"]
    assert import_template.main(["clip", "--id", "selftest_clip", *more]) == 0
    assert import_template.main(["text", "--id", "selftest_text", *more, "--keep-sampling"]) == 0
    manifest_path = tmp_path / "manifest" / "workflows.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = json.loads((tmp_path / "manifest" / "workflows" / "selftest_clip_api.json")
                      .read_text(encoding="utf-8"))
    assert clip["9"]["inputs"]["format.codec"] == "h264"
    entry = manifest["workflows"]["selftest_text"]
    assert (entry["level"], entry["seed_overrides"]) == ("smoke", {"5": {"sampling_mode.seed": 7}})
    # The same entry without its level is at reference level.
    del entry["level"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    capfd.readouterr()
    assert import_template.run_lint() == 1
    said = capfd.readouterr().out
    assert "test_text_workflows_generate_the_same_text_every_run[selftest_text]" in said
    assert "5: sampling_mode must be 'off' (greedy decoding)" in said
    assert "1 failed" in said
