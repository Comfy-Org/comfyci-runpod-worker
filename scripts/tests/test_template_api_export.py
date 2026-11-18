"""Rules the template -> API conversion has to get right.

Each test pins one rule the frontend's Export (API) applies, because a rule
applied wrongly does not fail loudly: it produces a prompt the server accepts
with a value from the wrong widget, and the corpus then compares the wrong
image for months.
"""
import pytest

from template_api_export import ConversionError, Converter, FlatGraph

# A node definition set small enough to read, covering every input kind the
# conversion has to handle.
OBJECT_INFO = {
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "ddim"], {}],
                "latent_image": ["LATENT"],
                "denoise": ["FLOAT", {"default": 1.0}],
            }
        },
        "input_order": {
            "required": [
                "model",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "latent_image",
                "denoise",
            ]
        },
    },
    "UNETLoader": {
        "input": {"required": {"unet_name": [["a.safetensors", "b.safetensors"], {}]}},
        "input_order": {"required": ["unet_name"]},
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        },
        "input_order": {"required": ["width", "height", "batch_size"]},
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "input_order": {"required": ["images", "filename_prefix"]},
    },
    "SaveImageAdvanced": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                "format": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "png",
                                "inputs": {
                                    "required": {
                                        "bit_depth": [
                                            "COMBO",
                                            {"default": "8-bit", "options": ["8-bit", "16-bit"]},
                                        ]
                                    }
                                },
                            },
                            {"key": "exr", "inputs": {"required": {}}},
                        ]
                    },
                ],
            },
            "optional": {
                "quality": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {"hidden": True, "options": [{"key": "auto", "inputs": {"required": {}}}]},
                ]
            },
        },
        "input_order": {
            "required": ["images", "filename_prefix", "format"],
            "optional": ["quality"],
        },
    },
    "PreviewAny": {
        "input": {"required": {"source": ["*"]}},
        "input_order": {"required": ["source"]},
        "output_node": True,
    },
    "ComfySwitchNode": {
        "input": {
            "required": {"switch": ["BOOLEAN", {}]},
            "optional": {
                "on_false": ["COMFY_MATCHTYPE_V3", {}],
                "on_true": ["COMFY_MATCHTYPE_V3", {}],
            },
        },
        "input_order": {"required": ["switch"], "optional": ["on_false", "on_true"]},
    },
    "MathExpression": {
        "input": {
            "required": {
                "expression": ["STRING", {"default": "a + b"}],
                "values": ["COMFY_AUTOGROW_V3", {"template": {"names": ["a", "b"]}}],
            }
        },
        "input_order": {"required": ["expression", "values"]},
    },
    "PrimitiveInt": {
        "input": {"required": {"value": ["INT", {"default": 0}]}},
        "input_order": {"required": ["value"]},
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"default": ""}], "clip": ["CLIP"]}},
        "input_order": {"required": ["text", "clip"]},
    },
}


def convert(workflow, **kwargs):
    return Converter(OBJECT_INFO).convert(workflow, **kwargs)


def node(node_id, node_type, *, widgets=None, inputs=None, outputs=None, mode=0):
    return {
        "id": node_id,
        "type": node_type,
        "mode": mode,
        "widgets_values": [] if widgets is None else widgets,
        "inputs": inputs or [],
        "outputs": outputs or [],
    }


def save(node_id=99, link=1, prefix="out"):
    return node(
        node_id,
        "SaveImage",
        widgets=[prefix],
        inputs=[{"name": "images", "type": "IMAGE", "link": link}],
    )


# --------------------------------------------------------------------------- #
# widget values -> named inputs
# --------------------------------------------------------------------------- #


def test_widget_values_map_onto_named_inputs_skipping_the_seed_control():
    """[seed, control, steps, ...] is one array; the control widget is UI-only."""
    workflow = {
        "nodes": [
            node(
                1,
                "KSampler",
                widgets=[42, "randomize", 8, 1.5, "ddim", 0.9],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}],
            ),
            save(link=1),
        ],
        "links": [[1, 1, 0, 99, 0, "IMAGE"]],
    }
    assert convert(workflow)["1"]["inputs"] == {
        "seed": 42,
        "steps": 8,
        "cfg": 1.5,
        "sampler_name": "ddim",
        "denoise": 0.9,
    }


def test_missing_trailing_widget_values_fall_back_to_the_definition_default():
    workflow = {
        "nodes": [
            node(
                1,
                "EmptyLatentImage",
                widgets=[768],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}],
            ),
            save(link=1),
        ],
        "links": [[1, 1, 0, 99, 0, "IMAGE"]],
    }
    assert convert(workflow)["1"]["inputs"] == {
        "width": 768,
        "height": 512,
        "batch_size": 1,
    }


def test_values_left_over_from_a_widget_the_node_no_longer_shows_are_ignored():
    workflow = {
        "nodes": [
            node(
                1,
                "EmptyLatentImage",
                widgets=[768, 768, 1, "stale", "stale"],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}],
            ),
            save(link=1),
        ],
        "links": [[1, 1, 0, 99, 0, "IMAGE"]],
    }
    assert convert(workflow)["1"]["inputs"]["batch_size"] == 1


def test_dynamic_combo_expands_into_the_inputs_its_choice_unlocks():
    """png adds bit_depth; the hidden quality widget takes its default."""
    workflow = {
        "nodes": [
            node(
                1,
                "SaveImageAdvanced",
                widgets=["pic", "png", "16-bit"],
                inputs=[{"name": "images", "type": "IMAGE", "link": 1}],
            ),
            node(2, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
        ],
        "links": [[1, 2, 0, 1, 0, "IMAGE"]],
    }
    inputs = convert(workflow)["1"]["inputs"]
    assert inputs["format"] == "png"
    assert inputs["format.bit_depth"] == "16-bit"
    assert inputs["quality"] == "auto"


def test_dynamic_combo_choice_without_sub_inputs_adds_none():
    workflow = {
        "nodes": [
            node(
                1,
                "SaveImageAdvanced",
                widgets=["pic", "exr"],
                inputs=[{"name": "images", "type": "IMAGE", "link": 1}],
            ),
            node(2, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
        ],
        "links": [[1, 2, 0, 1, 0, "IMAGE"]],
    }
    inputs = convert(workflow)["1"]["inputs"]
    assert inputs["format"] == "exr"
    assert not [key for key in inputs if key.startswith("format.")]


# TextGenerateV2 as ComfyUI defines it (tooltips and the long autogrow name
# lists left out): its seed is a sub-input of a dynamic combo and does not set
# control_after_generate.
TEXT_NODES = {
    "TextGenerateV2": {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "prompt": ["STRING", {"default": "", "multiline": True}],
                "max_length": ["INT", {"default": 512, "min": 1, "max": 32768}],
                "sampling_mode": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "on",
                                "inputs": {
                                    "required": {
                                        "temperature": ["FLOAT", {"default": 0.7}],
                                        "top_k": ["INT", {"default": 64}],
                                        "top_p": ["FLOAT", {"default": 0.95}],
                                        "min_p": ["FLOAT", {"default": 0.05}],
                                        "repetition_penalty": ["FLOAT", {"default": 1.05}],
                                        "presence_penalty": ["FLOAT", {"default": 0.0}],
                                        "seed": ["INT", {"default": 0, "min": 0}],
                                    }
                                },
                            },
                            {"key": "off", "inputs": {"required": {}}},
                        ]
                    },
                ],
            },
            "optional": {
                "images": ["COMFY_AUTOGROW_V3", {"template": {"names": ["image1", "image2"]}}],
                "thinking": ["BOOLEAN", {"default": False}],
                "use_default_template": ["BOOLEAN", {"default": True, "advanced": True}],
                "mtp": ["COMBO", {"default": "auto", "options": ["auto", "off", "2", "tree2s1"]}],
                "system_prompt": ["STRING", {"forceInput": True}],
            },
        },
        "input_order": {
            "required": ["clip", "prompt", "max_length", "sampling_mode"],
            "optional": ["images", "thinking", "use_default_template", "mtp", "system_prompt"],
        },
    },
    "CLIPLoader": {
        "input": {
            "required": {
                "clip_name": [["qwen3vl_4b_fp8_scaled.safetensors"]],
                "type": [["stable_diffusion", "ideogram4"]],
            },
            "optional": {"device": [["default", "cpu"], {"advanced": True}]},
        },
        "input_order": {"required": ["clip_name", "type"], "optional": ["device"]},
    },
    "PreviewAny": OBJECT_INFO["PreviewAny"],
}


def text_template(widgets):
    """The loader, generator and preview of a text template, with the
    generator's ``widgets_values`` as given."""
    return {
        "nodes": [
            node(
                5,
                "CLIPLoader",
                widgets=["qwen3vl_4b_fp8_scaled.safetensors", "ideogram4", "default"],
                outputs=[{"name": "CLIP", "type": "CLIP", "links": [5]}],
            ),
            node(7, "PreviewAny", inputs=[{"name": "source", "type": "*", "link": 7}]),
            node(
                10,
                "TextGenerateV2",
                widgets=widgets,
                inputs=[
                    {"name": "clip", "type": "CLIP", "link": 5},
                    {"name": "images.image1", "type": "IMAGE", "link": None},
                    {"name": "system_prompt", "type": "STRING", "link": None},
                ],
                outputs=[{"name": "generated_text", "type": "STRING", "links": [7]}],
            ),
        ],
        "links": [[5, 5, 0, 10, 0, "CLIP"], [7, 10, 0, 7, 0, "STRING"]],
    }


def test_a_control_word_stored_after_a_seed_that_asked_for_none_is_skipped():
    """llm_qwen3vl_text_gen stores "randomize" after sampling_mode.seed although
    the definition has no control_after_generate. Read as a value, it would
    shift every widget after it: thinking "randomize", mtp true."""
    stored = ["what's in the images", 512, "on", 0.7, 64, 0.95, 0.05, 1.05, 0, 0,
              "randomize", False, True, "auto"]
    inputs = Converter(TEXT_NODES).convert(text_template(stored))["10"]["inputs"]
    assert inputs == {
        "prompt": "what's in the images",
        "max_length": 512,
        "sampling_mode": "on",
        "sampling_mode.temperature": 0.7,
        "sampling_mode.top_k": 64,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.05,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.presence_penalty": 0,
        "sampling_mode.seed": 0,
        "thinking": False,
        "use_default_template": True,
        "mtp": "auto",
        "clip": ["5", 0],
    }


def test_a_seed_without_a_stored_control_word_reads_the_same_values():
    """image_ming_image_01_design_t2i stores the same node without the word."""
    stored = ["", 4096, "on", 0.7, 64, 0.95, 0.05, 1.05, 0, 0, False, True, "auto"]
    inputs = Converter(TEXT_NODES).convert(text_template(stored))["10"]["inputs"]
    assert (inputs["sampling_mode.seed"], inputs["thinking"]) == (0, False)
    assert (inputs["use_default_template"], inputs["mtp"]) == (True, "auto")
    # Greedy decoding has no seed at all, so nothing follows it.
    greedy = Converter(TEXT_NODES).convert(text_template(["hi", 256, "off", False, True, "off"]))
    assert greedy["10"]["inputs"] == {
        "prompt": "hi",
        "max_length": 256,
        "sampling_mode": "off",
        "thinking": False,
        "use_default_template": True,
        "mtp": "off",
        "clip": ["5", 0],
    }


def test_a_control_word_the_next_widget_can_hold_is_its_value():
    """Only a word the next widget cannot hold is taken for the stored control."""
    object_info = {
        "Sampler": {
            "input": {
                "required": {
                    "seed": ["INT", {"default": 0}],
                    "mode": [["fixed", "randomize"], {}],
                    "label": ["STRING", {"default": ""}],
                    "steps": ["INT", {"default": 20}],
                }
            },
            "input_order": {"required": ["seed", "mode", "label", "steps"]},
            "output_node": True,
        }
    }

    def values(widgets):
        workflow = {"nodes": [node(1, "Sampler", widgets=widgets)], "links": []}
        return Converter(object_info).convert(workflow)["1"]["inputs"]

    assert values([7, "randomize", "fixed", 30]) == {
        "seed": 7, "mode": "randomize", "label": "fixed", "steps": 30}
    # After a widget that is no seed, a control word is never dropped: it stays
    # in place and the mismatch shows, as before.
    assert values([7, "fixed", "note", "increment"])["steps"] == "increment"


def test_a_combo_whose_options_live_on_the_node_is_refused():
    """Its widget layout is not in /object_info, so the array cannot be mapped."""
    object_info = {
        "InstanceCombo": {
            "input": {"required": {"choice": ["COMBO", {"options": []}]}},
            "input_order": {"required": ["choice"]},
        }
    }
    workflow = {"nodes": [node(1, "InstanceCombo", widgets=[0, "a", "b"])], "links": []}
    with pytest.raises(ConversionError, match="options are defined on the node"):
        Converter(object_info).convert(workflow)


def test_an_unknown_node_type_is_refused():
    workflow = {"nodes": [node(1, "SomeCustomNode", widgets=[1])], "links": []}
    with pytest.raises(ConversionError, match="unknown type"):
        convert(workflow)


# ImageCropV2, ImageCropToMask and a node with a box in the middle, as ComfyUI
# defines the first two (tooltips left out).
BOX_NODES = {
    "ImageCropV2": {
        "input": {"required": {
            "image": ["IMAGE", {}],
            "crop_region": ["BOUNDING_BOX", {
                "default": {"x": 0, "y": 0, "width": 512, "height": 512},
                "socketless": True, "component": "ImageCrop"}],
        }},
        "input_order": {"required": ["image", "crop_region"]},
    },
    "ImageCropToMask": {
        "input": {"required": {
            "images": ["IMAGE", {}],
            "width": ["INT", {"default": 1024}],
            "height": ["INT", {"default": 1024}],
            "pad_factor": ["FLOAT", {"default": 1.1}],
            "grow_mask": ["INT", {"default": 0}],
            "background": ["COLOR", {"default": "#000000", "socketless": True}],
        }},
        "input_order": {"required": ["images", "width", "height", "pad_factor", "grow_mask",
                                     "background"]},
    },
    "BoxThenText": {
        "input": {"required": {
            "region": ["BOUNDING_BOX", {"default": {"x": 0, "y": 0, "width": 8, "height": 8}}],
            "label": ["STRING", {"default": ""}],
            "boxes": ["BOUNDING_BOX", {"forceInput": True}],
        }},
        "input_order": {"required": ["region", "label", "boxes"]},
        "output_node": True,
    },
    "SaveImage": OBJECT_INFO["SaveImage"],
}


def box_workflow(crop_widgets, mask_widgets):
    return {
        "nodes": [
            node(1, "ImageCropV2", widgets=crop_widgets,
                 outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1]}]),
            node(2, "ImageCropToMask", widgets=mask_widgets,
                 inputs=[{"name": "images", "type": "IMAGE", "link": 1}],
                 outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2]}]),
            save(link=2),
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 99, 0, "IMAGE"]],
    }


def test_a_colour_and_a_bounding_box_are_widgets_with_a_value():
    """3d_pixal3d_multi_views crops four views out of one sheet. Both values
    were dropped, and the worker filled the default: a 512 px crop at 0, 0."""
    front = {"x": 1652, "y": 0, "width": 1492, "height": 2688}
    prompt = Converter(BOX_NODES).convert(
        box_workflow([front, 0, 0, 512, 512], [1024, 768, 1.1, 0, "#848484"]))
    assert prompt["1"]["inputs"] == {"crop_region": front}
    assert prompt["2"]["inputs"] == {
        "width": 1024, "height": 768, "pad_factor": 1.1, "grow_mask": 0,
        "background": "#848484", "images": ["1", 0]}
    # A node saved before it had the widgets takes the definition's values.
    prompt = Converter(BOX_NODES).convert(box_workflow([], [1024, 768]))
    assert prompt["1"]["inputs"] == {"crop_region": {"x": 0, "y": 0, "width": 512, "height": 512}}
    assert prompt["2"]["inputs"]["background"] == "#000000"


def test_the_four_numbers_stored_after_a_box_are_not_the_next_widgets():
    """The frontend stores x, y, width and height once more after the box."""
    box = {"x": 4, "y": 2, "width": 64, "height": 32}

    def inputs(widgets):
        workflow = {"nodes": [node(1, "BoxThenText", widgets=widgets)], "links": []}
        return Converter(BOX_NODES).convert(workflow)["1"]["inputs"]

    assert inputs([box, 4, 2, 64, 32, "front"]) == {"region": box, "label": "front"}
    # Stored without the four numbers, the next value is the next widget's.
    assert inputs([box, "front"]) == {"region": box, "label": "front"}
    # A box that only a link can fill is no widget and takes no value.
    assert "boxes" not in inputs([box, 4, 2, 64, 32, "front"])
    for bad in ("front", 512, None, {"x": 0, "y": 0}):
        with pytest.raises(ConversionError, match="region holds .* not a box with x, y, width"):
            inputs([bad, "front"])


# --------------------------------------------------------------------------- #
# subgraphs
# --------------------------------------------------------------------------- #

SUBGRAPH_ID = "11111111-2222-3333-4444-555555555555"


def subgraph_workflow(instance_widgets, instance_inputs, instance_links=()):
    """One subgraph: an EmptyLatentImage feeding out, driven by two promoted widgets."""
    return {
        "nodes": [
            node(
                7,
                SUBGRAPH_ID,
                widgets=instance_widgets,
                inputs=instance_inputs,
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}],
            ),
            save(link=1),
        ],
        "links": [[1, 7, 0, 99, 0, "IMAGE"], *instance_links],
        "definitions": {
            "subgraphs": [
                {
                    "id": SUBGRAPH_ID,
                    "name": "Latent",
                    "inputs": [
                        {"name": "width", "type": "INT"},
                        {"name": "height", "type": "INT"},
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT"}],
                    "nodes": [
                        node(
                            3,
                            "EmptyLatentImage",
                            widgets=[512, 512, 1],
                            inputs=[
                                {
                                    "name": "width",
                                    "type": "INT",
                                    "widget": {"name": "width"},
                                    "link": 20,
                                },
                                {
                                    "name": "height",
                                    "type": "INT",
                                    "widget": {"name": "height"},
                                    "link": 21,
                                },
                            ],
                            outputs=[{"name": "LATENT", "type": "LATENT", "links": [22]}],
                        )
                    ],
                    "links": [
                        {"id": 20, "origin_id": -10, "origin_slot": 0, "target_id": 3, "target_slot": 0},
                        {"id": 21, "origin_id": -10, "origin_slot": 1, "target_id": 3, "target_slot": 1},
                        {"id": 22, "origin_id": 3, "origin_slot": 0, "target_id": -20, "target_slot": 0},
                    ],
                }
            ]
        },
    }


def test_subgraph_nodes_flatten_under_the_instance_id():
    prompt = convert(subgraph_workflow([], []))
    assert sorted(prompt) == ["7:3", "99"]
    assert prompt["99"]["inputs"]["images"] == ["7:3", 0]


def test_the_instance_promoted_widget_values_override_the_inner_node_values():
    prompt = convert(subgraph_workflow([1024, 768], []))
    assert prompt["7:3"]["inputs"]["width"] == 1024
    assert prompt["7:3"]["inputs"]["height"] == 768


def test_an_instance_with_no_stored_values_leaves_the_inner_values_alone():
    prompt = convert(subgraph_workflow([], []))
    assert prompt["7:3"]["inputs"]["width"] == 512


def test_a_promoted_widget_wired_from_outside_becomes_a_link():
    workflow = subgraph_workflow(
        [1024, 768],
        [{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 5}],
        instance_links=[[5, 8, 0, 7, 0, "INT"]],
    )
    workflow["nodes"].append(
        node(8, "PrimitiveInt", widgets=[640], outputs=[{"name": "INT", "type": "INT", "links": [5]}])
    )
    prompt = convert(workflow)
    assert prompt["7:3"]["inputs"]["width"] == ["8", 0]
    assert prompt["7:3"]["inputs"]["height"] == 768


def test_a_muted_subgraph_instance_does_not_run_the_nodes_inside_it():
    """Mode lives on the instance, not on its inner nodes: without checking it,
    a disabled refiner group runs anyway and pulls its models into the entry."""
    workflow = subgraph_workflow([1024, 768], [])
    workflow["nodes"][0]["mode"] = 2
    prompt = convert(workflow)
    assert sorted(prompt) == ["99"]
    assert "images" not in prompt["99"]["inputs"]


def test_a_bypassed_subgraph_instance_forwards_instead_of_running():
    workflow = subgraph_workflow(
        [1024, 768],
        [{"name": "passthrough", "type": "LATENT", "link": 5}],
        instance_links=[[5, 8, 0, 7, 0, "LATENT"]],
    )
    workflow["nodes"][0]["mode"] = 4
    workflow["nodes"][0]["outputs"] = [
        {"name": "LATENT", "type": "LATENT", "links": [1]}
    ]
    workflow["nodes"].append(
        node(8, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [5]}])
    )
    prompt = convert(workflow)
    assert "7:3" not in prompt
    assert prompt["99"]["inputs"]["images"] == ["8", 0]


def test_a_node_inside_a_muted_subgraph_instance_is_not_emitted():
    """Even when something else still references it, the instance's mode wins."""
    workflow = subgraph_workflow([1024, 768], [])
    workflow["nodes"][0]["mode"] = 2
    graph = FlatGraph(workflow)
    assert graph.inside_inactive_instance("7:3")
    assert not graph.inside_inactive_instance("99")


def test_socket_typed_subgraph_inputs_do_not_consume_a_widget_value():
    """An instance stores one value per widget input, so an IMAGE input in the
    middle of the definition must not shift the ones after it."""
    workflow = subgraph_workflow([1024, 768], [])
    definition = workflow["definitions"]["subgraphs"][0]
    definition["inputs"].insert(0, {"name": "reference", "type": "IMAGE"})
    for link in definition["links"]:
        if link["origin_id"] == -10:
            link["origin_slot"] += 1
    prompt = convert(workflow)
    assert prompt["7:3"]["inputs"]["width"] == 1024
    assert prompt["7:3"]["inputs"]["height"] == 768


def test_an_instance_stores_a_value_for_each_input_it_shows_as_a_widget():
    """Not for each input of a widget type. utility_image_segment_sam3 has two
    STRING inputs wired to sockets that only a link can fill: counted as
    widgets, they moved every value after them (threshold false,
    refine_iterations "sam3.1_multiplex_fp16.safetensors")."""
    workflow = subgraph_workflow([1024, 768], [])
    definition = workflow["definitions"]["subgraphs"][0]
    definition["inputs"].insert(0, {"name": "coords", "type": "STRING"})
    for link in definition["links"]:
        if link["origin_id"] == -10:
            link["origin_slot"] += 1
    definition["nodes"].append(
        node(4, "PreviewAny", inputs=[{"name": "source", "type": "*", "link": 23}]))
    definition["links"].append(
        {"id": 23, "origin_id": -10, "origin_slot": 0, "target_id": 4, "target_slot": 0})
    prompt = convert(workflow)
    assert (prompt["7:3"]["inputs"]["width"], prompt["7:3"]["inputs"]["height"]) == (1024, 768)
    assert prompt["7:4"]["inputs"] == {}
    # An input of type * that feeds a widget is one, and a null leaves the
    # inner node's value: templates_product_scene_transformation stores
    # [null, 720, ...] for two such inputs.
    definition["inputs"][1]["type"] = "*"
    workflow["nodes"][0]["widgets_values"] = [None, 768]
    prompt = convert(workflow)
    assert (prompt["7:3"]["inputs"]["width"], prompt["7:3"]["inputs"]["height"]) == (512, 768)


def test_an_instance_with_proxy_widgets_stores_its_values_in_their_order():
    """Image_capybara_v0_1_text_to_image lists width and height before the
    model names, the subgraph lists them after: read in the order of the
    inputs, the latent got a model name for its height."""
    workflow = subgraph_workflow([None, 768, 1024], [])
    workflow["nodes"][0]["properties"] = {
        "proxyWidgets": [["3", "batch_size"], ["-1", "height"], ["-1", "width"]]}
    inputs = convert(workflow)["7:3"]["inputs"]
    assert inputs == {"width": 1024, "height": 768, "batch_size": 1}
    # Without stored values the inner nodes hold them.
    workflow["nodes"][0]["widgets_values"] = []
    assert convert(workflow)["7:3"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
    # The entries at the end of the list that store nothing leave no value:
    # image_flux2_klein_text_to_image has 6 values for 8 entries.
    workflow["nodes"][0]["widgets_values"] = [None, 768]
    assert convert(workflow)["7:3"]["inputs"] == {"width": 512, "height": 768, "batch_size": 1}
    # More values than widgets: no telling which is whose.
    workflow["nodes"][0]["widgets_values"] = [None, 768, 1024, 2]
    with pytest.raises(ConversionError, match="instance 7 stores 4 values for 3 promoted"):
        convert(workflow)
    # Unless nothing has to be read from them: both inputs are wired.
    workflow["nodes"][0]["inputs"] = [
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 5},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": 6}]
    workflow["links"] += [[5, 8, 0, 7, 0, "INT"], [6, 8, 0, 7, 1, "INT"]]
    workflow["nodes"].append(node(8, "PrimitiveInt", widgets=[640], outputs=[
        {"name": "INT", "type": "INT", "links": [5, 6]}]))
    assert convert(workflow)["7:3"]["inputs"]["height"] == ["8", 0]


def test_a_socket_inside_a_subgraph_gets_the_value_of_the_widget_it_shares_an_input_with():
    """image_ernie_image shows its width in a PreviewAny. The subgraph input
    feeds the latent's width widget, which holds the value, and the preview's
    socket, which got no input at all."""
    def workflow(instance_widgets=(), inner_widgets=(640, 512, 1)):
        made = subgraph_workflow(list(instance_widgets), [])
        definition = made["definitions"]["subgraphs"][0]
        definition["nodes"][0]["widgets_values"] = list(inner_widgets)
        definition["nodes"].append(
            node(4, "PreviewAny", inputs=[{"name": "source", "type": "*", "link": 23}]))
        definition["links"].append(
            {"id": 23, "origin_id": -10, "origin_slot": 0, "target_id": 4, "target_slot": 0})
        return made

    prompt = convert(workflow())
    assert prompt["7:4"]["inputs"] == {"source": 640}
    assert prompt["7:3"]["inputs"]["width"] == 640
    # A value the instance stores is the value of both.
    prompt = convert(workflow(instance_widgets=[1024, 768]))
    assert (prompt["7:4"]["inputs"]["source"], prompt["7:3"]["inputs"]["width"]) == (1024, 1024)
    # Wired from outside, both read the wire.
    wired = workflow()
    wired["nodes"][0]["inputs"] = [
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 5}]
    wired["links"].append([5, 8, 0, 7, 0, "INT"])
    wired["nodes"].append(node(8, "PrimitiveInt", widgets=[320],
                               outputs=[{"name": "INT", "type": "INT", "links": [5]}]))
    prompt = convert(wired)
    assert prompt["7:4"]["inputs"]["source"] == prompt["7:3"]["inputs"]["width"] == ["8", 0]
    # An input that no inner widget holds gives the socket nothing.
    alone = workflow()
    alone["definitions"]["subgraphs"][0]["links"][0]["origin_slot"] = 1
    assert convert(alone)["7:4"]["inputs"] == {}


# --------------------------------------------------------------------------- #
# collapsing
# --------------------------------------------------------------------------- #


def test_a_reroute_is_followed_through_to_the_real_producer():
    workflow = {
        "nodes": [
            node(1, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
            node(
                2,
                "Reroute",
                inputs=[{"name": "", "type": "*", "link": 1}],
                outputs=[{"name": "", "type": "*", "links": [2]}],
            ),
            save(link=2),
        ],
        "links": [[1, 1, 0, 2, 0, "LATENT"], [2, 2, 0, 99, 0, "IMAGE"]],
    }
    prompt = convert(workflow)
    assert "2" not in prompt
    assert prompt["99"]["inputs"]["images"] == ["1", 0]


def test_a_legacy_primitive_node_is_inlined_as_the_value_it_holds():
    workflow = {
        "nodes": [
            node(1, "PrimitiveNode", widgets=[1024, "fixed"], outputs=[{"name": "INT", "type": "INT", "links": [1]}]),
            node(
                2,
                "EmptyLatentImage",
                widgets=[512, 512, 1],
                inputs=[{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 1}],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [2]}],
            ),
            save(link=2),
        ],
        "links": [[1, 1, 0, 2, 0, "INT"], [2, 2, 0, 99, 0, "IMAGE"]],
    }
    prompt = convert(workflow)
    assert "1" not in prompt
    assert prompt["2"]["inputs"]["width"] == 1024


def test_a_muted_node_drops_and_leaves_its_consumer_input_unset():
    workflow = {
        "nodes": [
            node(1, "KSampler", mode=2, outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
            save(link=1),
        ],
        "links": [[1, 1, 0, 99, 0, "IMAGE"]],
    }
    prompt = convert(workflow)
    assert "1" not in prompt
    assert "images" not in prompt["99"]["inputs"]


def test_a_bypassed_node_forwards_its_output_to_its_first_input_of_that_type():
    workflow = {
        "nodes": [
            node(1, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
            node(
                2,
                "KSampler",
                mode=4,
                inputs=[
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "latent_image", "type": "LATENT", "link": 1},
                ],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [2]}],
            ),
            save(link=2),
        ],
        "links": [[1, 1, 0, 2, 1, "LATENT"], [2, 2, 0, 99, 0, "IMAGE"]],
    }
    prompt = convert(workflow)
    assert "2" not in prompt
    assert prompt["99"]["inputs"]["images"] == ["1", 0]


def test_a_bypassed_node_forwards_as_the_frontend_does():
    """image_ming_image_01_design_image_edit bypasses a ResizeImageMaskNode,
    whose input is typed IMAGE,MASK: no input had exactly the output's type,
    and the node after it lost its picture."""
    def forwarded(inputs, outputs, read=0):
        workflow = {
            "nodes": [
                node(1, "KSampler", outputs=[{"name": "a", "type": "IMAGE", "links": [1]}]),
                node(3, "KSampler", outputs=[{"name": "b", "type": "IMAGE", "links": [3]}]),
                node(2, "Resize", mode=4, inputs=inputs, outputs=outputs),
                save(link=2),
            ],
            # Link 1 comes from node 1 and link 3 from node 3, into the slot that names it.
            "links": [*([s["link"], s["link"], 0, 2, at, "IMAGE"]
                        for at, s in enumerate(inputs) if s["link"]),
                      [2, 2, read, 99, 0, "IMAGE"]],
        }
        return convert(workflow)["99"]["inputs"].get("images")

    def slot(name, kind, link=None):
        return {"name": name, "type": kind, "link": link}

    image = [slot("resized", "IMAGE")]
    # An input that lists the type takes it, and so does one of any type.
    assert forwarded([slot("input", "IMAGE,MASK", 1)], image) == ["1", 0]
    assert forwarded([slot("mask", "MASK"), slot("any", "*", 3)], image) == ["3", 0]
    # An input of exactly the type goes before one that only takes it.
    assert forwarded([slot("mask", "MASK"), slot("either", "IMAGE,MASK", 1),
                      slot("image", "IMAGE", 3)], image) == ["3", 0]
    # The input at the output's own position goes before the first of the type.
    two = [slot("left", "IMAGE"), slot("right", "IMAGE")]
    pair = [slot("left", "IMAGE", 1), slot("right", "IMAGE", 3)]
    assert forwarded(pair, two, read=0) == ["1", 0]
    assert forwarded(pair, two, read=1) == ["3", 0]
    # No input takes the type: nothing is forwarded.
    assert forwarded([slot("mask", "MASK", 1)], image) is None


def test_autogrow_sockets_are_read_off_the_node_not_the_definition():
    """values.a / values.b exist on the node instance only."""
    workflow = {
        "nodes": [
            node(1, "PrimitiveInt", widgets=[3], outputs=[{"name": "INT", "type": "INT", "links": [1]}]),
            node(
                2,
                "MathExpression",
                widgets=["a * 2"],
                inputs=[{"name": "values.a", "type": "FLOAT,INT,BOOLEAN", "link": 1}],
                outputs=[{"name": "INT", "type": "INT", "links": [2]}],
            ),
            node(
                3,
                "EmptyLatentImage",
                widgets=[512, 512, 1],
                inputs=[{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 2}],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [3]}],
            ),
            save(link=3),
        ],
        "links": [
            [1, 1, 0, 2, 0, "INT"],
            [2, 2, 0, 3, 0, "INT"],
            [3, 3, 0, 99, 0, "IMAGE"],
        ],
    }
    prompt = convert(workflow)
    assert prompt["2"]["inputs"] == {"expression": "a * 2", "values.a": ["1", 0]}


# --------------------------------------------------------------------------- #
# pruning and switches
# --------------------------------------------------------------------------- #


def test_nodes_that_feed_no_output_are_dropped():
    workflow = {
        "nodes": [
            node(1, "KSampler", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}]),
            node(2, "UNETLoader", widgets=["a.safetensors"], outputs=[{"name": "MODEL", "type": "MODEL", "links": None}]),
            save(link=1),
        ],
        "links": [[1, 1, 0, 99, 0, "IMAGE"]],
    }
    assert sorted(convert(workflow)) == ["1", "99"]


def test_a_workflow_with_no_output_node_is_refused():
    workflow = {"nodes": [node(1, "UNETLoader", widgets=["a.safetensors"])], "links": []}
    with pytest.raises(ConversionError, match="no output node"):
        convert(workflow)


def switch_workflow(switch_value):
    return {
        "nodes": [
            node(1, "PrimitiveInt", widgets=[10], outputs=[{"name": "INT", "type": "INT", "links": [1]}]),
            node(2, "PrimitiveInt", widgets=[20], outputs=[{"name": "INT", "type": "INT", "links": [2]}]),
            node(
                3,
                "ComfySwitchNode",
                widgets=[switch_value],
                inputs=[
                    {"name": "on_false", "type": "INT", "link": 1},
                    {"name": "on_true", "type": "INT", "link": 2},
                ],
                outputs=[{"name": "output", "type": "INT", "links": [3]}],
            ),
            node(
                4,
                "EmptyLatentImage",
                widgets=[512, 512, 1],
                inputs=[{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 3}],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [4]}],
            ),
            save(link=4),
        ],
        "links": [
            [1, 1, 0, 3, 0, "INT"],
            [2, 2, 0, 3, 1, "INT"],
            [3, 3, 0, 4, 0, "INT"],
            [4, 4, 0, 99, 0, "IMAGE"],
        ],
    }


def test_a_switch_is_left_alone_unless_resolution_is_asked_for():
    prompt = convert(switch_workflow(False))
    assert prompt["3"]["class_type"] == "ComfySwitchNode"
    assert sorted(prompt) == ["1", "2", "3", "4", "99"]


@pytest.mark.parametrize(
    "switch_value, kept, dropped",
    [(False, "1", "2"), (True, "2", "1")],
)
def test_resolving_a_switch_keeps_one_branch_and_drops_the_other(switch_value, kept, dropped):
    """The dropped branch is why: its models never have to be synced."""
    prompt = convert(switch_workflow(switch_value), resolve_switches=True)
    assert prompt["4"]["inputs"]["width"] == [kept, 0]
    assert "3" not in prompt
    assert dropped not in prompt


def test_a_switch_driven_by_a_node_cannot_be_resolved_and_stays():
    workflow = switch_workflow(False)
    workflow["nodes"][2]["inputs"].append(
        {"name": "switch", "type": "BOOLEAN", "widget": {"name": "switch"}, "link": 5}
    )
    workflow["nodes"].append(
        node(5, "PrimitiveInt", widgets=[1], outputs=[{"name": "INT", "type": "INT", "links": [5]}])
    )
    workflow["links"].append([5, 5, 0, 3, 2, "BOOLEAN"])
    prompt = convert(workflow, resolve_switches=True)
    assert prompt["3"]["class_type"] == "ComfySwitchNode"
