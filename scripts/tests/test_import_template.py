"""The corpus-specific steps on top of the template -> API conversion."""
import pytest

from import_template import (
    build_entry,
    collapse_previews,
    loaded_files,
    normalise_save,
    note_urls,
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
