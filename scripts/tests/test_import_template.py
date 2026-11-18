"""The corpus-specific steps on top of the template -> API conversion."""
import json
import sys

import pytest

import import_template
from import_template import (
    build_entry,
    collapse_previews,
    loaded_files,
    normalise_save,
    note_urls,
    refuse_text_output,
    seed_overrides,
)
from template_api_export import ConversionError


def test_a_preview_in_the_middle_of_the_graph_forwards_its_value_and_goes():
    """PreviewAny shows a string on its way to a text encoder. Left in, it is a
    second output node and a node that feeds no SaveImage."""
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


def test_an_advanced_save_is_rewritten_to_a_plain_png_save_named_after_the_workflow():
    prompt = {
        "9": {
            "class_type": "SaveImageAdvanced",
            "inputs": {
                "filename_prefix": "Qwen_image_2.1",
                "format": "png",
                "format.bit_depth": "8-bit",
                "format.input_color_space": "sRGB",
                "images": ["8", 0],
            },
        }
    }
    normalise_save(prompt, "qwen_image_2_1_t2i")
    assert prompt["9"] == {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "qwen_image_2_1_t2i", "images": ["8", 0]},
    }


def test_a_plain_save_just_gets_the_workflow_id_as_its_prefix():
    prompt = {"9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "PixelDiT", "images": ["8", 0]}}}
    normalise_save(prompt, "pixeldit_t2i")
    assert prompt["9"]["inputs"]["filename_prefix"] == "pixeldit_t2i"


@pytest.mark.parametrize(
    "inputs, match",
    [
        ({"format": "exr", "images": ["8", 0]}, "8-bit PNG"),
        ({"format": "png", "format.bit_depth": "16-bit", "images": ["8", 0]}, "8-bit PNG"),
        (
            {
                "format": "png",
                "format.bit_depth": "8-bit",
                "format.input_color_space": "HDR",
                "images": ["8", 0],
            },
            "8-bit PNG",
        ),
        ({"format": "png", "format.bit_depth": "8-bit"}, "no images input"),
    ],
)
def test_a_save_comparison_cannot_read_is_refused(inputs, match):
    prompt = {"9": {"class_type": "SaveImageAdvanced", "inputs": inputs}}
    with pytest.raises(ConversionError, match=match):
        normalise_save(prompt, "wf")


def test_more_than_one_save_is_refused():
    prompt = {
        "8": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    with pytest.raises(ConversionError, match="exactly one SaveImage"):
        normalise_save(prompt, "wf")


def test_each_loaded_file_is_mapped_to_the_models_subdirectory_comfyui_searches():
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "l.safetensors"}},
        "5": {"class_type": "KSampler", "inputs": {"sampler_name": "euler", "seed": 1}},
    }
    assert loaded_files(prompt) == {
        "u.safetensors": "diffusion_models",
        "c.safetensors": "text_encoders",
        "v.safetensors": "vae",
        "l.safetensors": "loras",
    }


def test_a_loader_input_with_no_known_directory_is_refused_rather_than_guessed():
    prompt = {"1": {"class_type": "SomeLoader", "inputs": {"mystery_name": "m.safetensors"}}}
    with pytest.raises(ConversionError, match="DIRECTORY_BY_LOADER_INPUT"):
        loaded_files(prompt)


def test_the_same_input_name_on_a_different_loader_gets_its_own_directory():
    """CLIPLoader.clip_name reads text_encoders/, CLIPVisionLoader.clip_name
    reads clip_vision/: keying on the name alone would sync one to the other."""
    prompt = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t.safetensors"}},
        "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "v.safetensors"}},
    }
    assert loaded_files(prompt) == {
        "t.safetensors": "text_encoders",
        "v.safetensors": "clip_vision",
    }


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
                            "widgets_values": ["[u.safetensors](https://example.test/b/u.safetensors)"],
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


def test_every_seed_input_is_pinned_at_the_value_the_template_ships():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}},
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
    }
    assert seed_overrides(prompt) == {"1": {"seed": 42}, "2": {"noise_seed": 7}}


def test_a_seed_driven_by_a_link_is_not_pinned_as_a_literal():
    prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": ["2", 0]}}}
    assert seed_overrides(prompt) == {}


def test_the_entry_lands_disabled_and_points_at_its_own_workflow_file():
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 5}},
    }
    template = {
        "nodes": [
            {"type": "MarkdownNote", "widgets_values": ["[u.safetensors](https://example.test/u.safetensors)"]}
        ]
    }
    entry = build_entry("my_wf", prompt, template, skip_sha=True)
    assert entry["enabled"] is False
    assert entry["workflow_path"] == "manifest/workflows/my_wf_api.json"
    assert entry["seed_overrides"] == {"2": {"seed": 5}}
    assert entry["models"] == [
        {
            "name": "u.safetensors",
            "url": "https://example.test/u.safetensors",
            "directory": "diffusion_models",
        }
    ]


def test_a_loaded_file_the_template_never_links_is_refused():
    prompt = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "u.safetensors"}}}
    with pytest.raises(ConversionError, match="no download link"):
        build_entry("my_wf", prompt, {"nodes": []}, skip_sha=True)


def _text_prompt(sink):
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "q.safetensors"}},
        "3": {"class_type": "TextGenerateV2", "inputs": {"prompt": "hi", "clip": ["1", 0]}},
        "4": sink,
    }


@pytest.mark.parametrize(
    "sink, named",
    [
        ({"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}, "PreviewAny node 4"),
        (
            {"class_type": "SaveText", "inputs": {"text": ["3", 0], "filename_prefix": "x"}},
            "SaveText node 4",
        ),
    ],
)
def test_a_template_whose_output_is_text_is_refused_with_what_to_do(sink, named):
    """Dropping the preview would leave no output, and the SaveImage error
    that follows would not say that the template is a text one."""
    with pytest.raises(ConversionError) as refusal:
        refuse_text_output(_text_prompt(sink), "llm_x")
    message = str(refusal.value)
    assert f"the template's output is text ({named})" in message
    assert "only writes image workflows" in message
    assert 'SaveText with filename_prefix "llm_x" and format "txt"' in message
    assert 'sampling_mode "off", mtp "off"' in message and '"output_kind": "text"' in message


def test_an_image_template_with_text_previews_is_not_taken_for_a_text_one():
    # A preview on the way to a text encoder, and one nothing reads, next to a save.
    prompt = {
        **_text_prompt({"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}),
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
        "9": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["8", 0]}},
    }
    refuse_text_output(prompt, "wf")
    collapse_previews(prompt)
    assert sorted(prompt) == ["1", "3", "6", "9"] and prompt["6"]["inputs"]["text"] == ["3", 0]
    # A video or audio template with a text preview on the side saves a file
    # of its own: its refusal is the SaveImage one, not advice to compare the preview.
    for save in ("SaveVideo", "SaveAudioAdvanced", "SaveGLB", "SaveWEBM", "SaveAnimatedWEBP"):
        media = {
            **_text_prompt({"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}}),
            "9": {"class_type": save, "inputs": {"video": ["8", 0]}},
        }
        refuse_text_output(media, "wf")
        collapse_previews(media)
        with pytest.raises(ConversionError, match="expected exactly one SaveImage output, found 0"):
            normalise_save(media, "wf")
    # No save and no text sink either: the usual refusal, further down, says so.
    refuse_text_output({"1": {"class_type": "KSampler", "inputs": {}}}, "wf")
    refuse_text_output({"1": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
                        "2": {"class_type": "KSampler", "inputs": {}}}, "wf")


def test_the_command_refuses_a_text_template_without_a_traceback(tmp_path, monkeypatch, capsys):
    """llm_qwen3vl_text_gen in short: a loader, a generator and a preview."""
    template = {
        "nodes": [
            {"id": 5, "type": "CLIPLoader", "mode": 0, "inputs": [],
             "outputs": [{"name": "CLIP", "type": "CLIP", "links": [5]}],
             "widgets_values": ["q.safetensors"]},
            {"id": 10, "type": "TextGenerateV2", "mode": 0,
             "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
             "outputs": [{"name": "generated_text", "type": "STRING", "links": [7]}],
             "widgets_values": ["what's in the images"]},
            {"id": 7, "type": "PreviewAny", "mode": 0,
             "inputs": [{"name": "source", "type": "*", "link": 7}], "outputs": [],
             "widgets_values": []},
        ],
        "links": [[5, 5, 0, 10, 0, "CLIP"], [7, 10, 0, 7, 0, "STRING"]],
    }
    object_info = {
        "CLIPLoader": {"input": {"required": {"clip_name": [["q.safetensors"]]}},
                       "input_order": {"required": ["clip_name"]}},
        "TextGenerateV2": {"input": {"required": {"clip": ["CLIP", {}],
                                                  "prompt": ["STRING", {"default": ""}]}},
                           "input_order": {"required": ["clip", "prompt"]}},
        "PreviewAny": {"input": {"required": {"source": ["*", {}]}},
                       "input_order": {"required": ["source"]}, "output_node": True},
    }
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "llm_x.json").write_text(json.dumps(template), encoding="utf-8")
    (tmp_path / "object_info.json").write_text(json.dumps(object_info), encoding="utf-8")
    (tmp_path / "manifest" / "workflows").mkdir(parents=True)
    monkeypatch.setattr(import_template, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "import_template.py", "llm_x", "--templates-dir", str(tmp_path / "templates"),
        "--object-info", str(tmp_path / "object_info.json"), "--skip-sha"])
    assert import_template.main() == 1
    printed = capsys.readouterr()
    assert printed.err.startswith("error: the template's output is text (PreviewAny node 7), ")
    assert "Traceback" not in printed.err and printed.out == ""
    assert list((tmp_path / "manifest" / "workflows").iterdir()) == []  # nothing was written
