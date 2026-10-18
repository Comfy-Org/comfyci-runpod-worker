"""Rules the template -> API conversion has to get right.

Each test pins one rule the frontend's Export (API) applies, because a rule
applied wrongly does not fail loudly: it produces a prompt the server accepts
with a value from the wrong widget, and the corpus then compares the wrong
image for months.
"""
import pytest

from template_api_export import ConversionError, Converter

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
