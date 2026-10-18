#!/usr/bin/env python3
"""Turn a ComfyUI workflow template into the API prompt format a run submits.

Templates ship as UI graphs (``templates/<name>.json`` in
Comfy-Org/workflow_templates): subgraphs, promoted widgets, reroutes, switches.
A run submits the API format: one flat ``{node id: {class_type, inputs}}``
object. The frontend does that conversion in Export (API); this is the same
conversion offline, driven by a ComfyUI ``/object_info`` dump, so a template can
enter the corpus without hand-editing JSON.

What it reproduces, rule by rule:

* subgraph instances flatten to ``<instance id>:<inner id>`` keys, with the
  instance's promoted widget values overriding the inner nodes' own values
* ``widgets_values`` arrays map onto named inputs in ``input_order``, skipping
  the UI-only widget that follows a ``control_after_generate`` seed and
  expanding dynamic combos into their ``<name>.<sub>`` inputs
* reroutes, legacy primitives, muted and bypassed nodes collapse away
* nodes that do not feed an output node are dropped

Verified against the API exports in the templates repo's ``benchmarks/``
directory, which the frontend produced: it reproduces 8 of the 10 it can
convert byte for byte, and the other two differ only in widget values that were
edited after the export was taken.

It refuses rather than guess: a node whose widget layout lives on the instance
(a server-side combo with no options) raises instead of producing a prompt with
shifted values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FRONTEND_ONLY = {"MarkdownNote", "Note", "PrimitiveNode"}
PASSTHROUGH = {"Reroute"}
# The legacy virtual primitive: it holds a widget value and pushes it into the
# widget it is wired to, so it never reaches the backend as a node.
VIRTUAL_VALUE = {"PrimitiveNode"}

MODE_ALWAYS, MODE_NEVER, MODE_BYPASS = 0, 2, 4

SWITCH_NODE = "ComfySwitchNode"
# Preview nodes are output nodes, but a corpus run compares saved files; keeping
# them as roots would pull whole unused branches (and their models) into the run.
PREVIEW_PREFIXES = ("Preview", "SaveImageWebsocket")

OUTPUT_PREFIXES = (
    "SaveImage",
    "SaveVideo",
    "SaveAudio",
    "SaveWEBM",
    "SaveGLB",
    "SaveAnimated",
)

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
AUTOGROW = "COMFY_AUTOGROW_V3"
MATCHTYPE = "COMFY_MATCHTYPE_V3"
# Socket-only V3 input kinds: the slots exist on the node, never as widgets.
SOCKET_DYNAMIC = {AUTOGROW, MATCHTYPE}
# Subgraph input types that a parent instance serialises as a widget value.
PROMOTED_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class ConversionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# node definitions
# --------------------------------------------------------------------------- #


def _ordered_inputs(spec: dict[str, Any]) -> list[tuple[str, Any]]:
    inputs = spec.get("input") or {}
    order = spec.get("input_order") or {}
    out: list[tuple[str, Any]] = []
    for section in ("required", "optional"):
        section_inputs = inputs.get(section) or {}
        for name in order.get(section) or list(section_inputs):
            if name in section_inputs:
                out.append((name, section_inputs[name]))
    return out


def _is_widget(entry: Any) -> bool:
    if not isinstance(entry, (list, tuple)) or not entry:
        return False
    type_ = entry[0]
    if isinstance(type_, list):
        return True
    opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    if opts.get("forceInput"):
        return False
    return type_ in WIDGET_TYPES or type_ == DYNAMIC_COMBO


def _entry_opts(entry: Any) -> dict[str, Any]:
    if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
        return entry[1]
    return {}


def _is_socket_dynamic(entry: Any) -> bool:
    return isinstance(entry, (list, tuple)) and bool(entry) and entry[0] in SOCKET_DYNAMIC


def dynamic_combo_subinputs(entry: Any, key: Any) -> list[tuple[str, Any]]:
    """Sub-inputs a dynamic combo contributes once ``key`` is chosen."""
    opts = _entry_opts(entry)
    for option in opts.get("options") or []:
        if not isinstance(option, dict) or option.get("key") != key:
            continue
        sub = option.get("inputs") or {}
        out: list[tuple[str, Any]] = []
        for section in ("required", "optional"):
            for sub_name, sub_entry in (sub.get(section) or {}).items():
                out.append((sub_name, sub_entry))
        return out
    return []


def default_of(entry: Any) -> Any:
    opts = _entry_opts(entry)
    if "default" in opts:
        return opts["default"]
    if isinstance(entry, (list, tuple)) and isinstance(entry[0], list) and entry[0]:
        return entry[0][0]
    options = opts.get("options") or []
    if isinstance(entry, (list, tuple)) and entry[0] == DYNAMIC_COMBO:
        if not options:
            raise ConversionError("dynamic combo with no options")
        return options[0].get("key")
    if isinstance(entry, (list, tuple)) and entry[0] == "COMBO":
        if not options:
            raise ConversionError(
                "combo with no server-side options: its values live in the UI only"
            )
        return options[0]
    return None


# --------------------------------------------------------------------------- #
# graph flattening
# --------------------------------------------------------------------------- #


def _normalise_links(graph: dict[str, Any]) -> list[tuple[Any, Any, int, Any, int]]:
    links = []
    for link in graph.get("links") or []:
        if isinstance(link, dict):
            links.append(
                (
                    link.get("id"),
                    link.get("origin_id"),
                    link.get("origin_slot"),
                    link.get("target_id"),
                    link.get("target_slot"),
                )
            )
        elif isinstance(link, (list, tuple)) and len(link) >= 5:
            links.append((link[0], link[1], link[2], link[3], link[4]))
    return links


class FlatGraph:
    """A template graph with every subgraph instance expanded."""

    def __init__(self, workflow: dict[str, Any]) -> None:
        self.subgraphs = {
            s["id"]: s for s in (workflow.get("definitions") or {}).get("subgraphs") or []
        }
        self.nodes: dict[str, dict[str, Any]] = {}
        self.instances: dict[str, str] = {}  # flat id -> subgraph uuid
        self.in_edge: dict[tuple[str, int], tuple[str, int]] = {}
        self._walk(workflow, prefix="")

    def _flat(self, prefix: str, node_id: Any) -> str:
        return f"{prefix}{node_id}"

    def _walk(self, graph: dict[str, Any], prefix: str) -> None:
        for node in graph.get("nodes") or []:
            flat_id = self._flat(prefix, node["id"])
            self.nodes[flat_id] = node
            uuid = node.get("type")
            if uuid in self.subgraphs:
                self.instances[flat_id] = uuid
                self._walk(self.subgraphs[uuid], prefix=f"{flat_id}:")

        for _lid, origin, oslot, target, tslot in _normalise_links(graph):
            self.in_edge[(self._flat(prefix, target), tslot)] = (
                self._flat(prefix, origin),
                oslot,
            )

    # -- resolution -------------------------------------------------------- #

    def _instance_slot_for_input(self, instance_id: str, index: int) -> int | None:
        """Index into the instance node's own input list for subgraph input ``index``."""
        defs = self._subgraph_inputs(instance_id)
        if index >= len(defs):
            return None
        name = defs[index].get("name")
        for i, slot in enumerate(self.nodes[instance_id].get("inputs") or []):
            if slot.get("name") == name:
                return i
        return None

    def _subgraph_inputs(self, instance_id: str) -> list[dict[str, Any]]:
        return self.subgraphs[self.instances[instance_id]].get("inputs") or []

    def _promoted_widget_index(self, instance_id: str, index: int) -> int | None:
        """Position of subgraph input ``index`` in the instance's widget array.

        Only widget-typed subgraph inputs are serialised as instance widget
        values; socket inputs (IMAGE, AUDIO, MODEL, ...) are skipped.
        """
        position = 0
        for i, slot in enumerate(self._subgraph_inputs(instance_id)):
            if slot.get("type") not in PROMOTED_WIDGET_TYPES:
                continue
            if i == index:
                return position
            position += 1
        return None

    def _promoted_value(self, instance_id: str, index: int) -> tuple[bool, Any]:
        """The value an instance node holds for its subgraph input ``index``.

        A subgraph instance serialises one widget value per widget-typed
        subgraph input, in definition order, whether or not that input is
        exposed as a slot on the instance. An instance saved against an older
        definition may carry fewer values, in which case the inner node's own
        widget value stands.
        """
        position = self._promoted_widget_index(instance_id, index)
        if position is None:
            return False, None
        values = self.nodes[instance_id].get("widgets_values") or []
        if position < len(values):
            return True, values[position]
        return False, None

    def resolve(self, node_id: str, slot: int) -> tuple[str, Any] | None:
        """Follow a node input back to its real producer.

        Returns ``("link", (node id, output slot))`` when a node drives the
        input, ``("value", v)`` when a subgraph instance supplies a literal, or
        ``None`` when nothing is connected.
        """
        seen: set[tuple[str, int]] = set()
        cursor: tuple[str, int] | None = (node_id, slot)
        while cursor is not None:
            if cursor in seen:
                raise ConversionError(f"cycle resolving input {cursor}")
            seen.add(cursor)
            edge = self.in_edge.get(cursor)
            if edge is None:
                return None
            origin_id, origin_slot = edge
            tail = origin_id.rsplit(":", 1)[-1]

            if tail == "-10":  # subgraph input boundary, step out
                instance_id = origin_id[: -len(":-10")] if ":" in origin_id else ""
                if not instance_id or instance_id not in self.instances:
                    return None
                outer = self._instance_slot_for_input(instance_id, origin_slot)
                if outer is None or (self.nodes[instance_id].get("inputs") or [])[
                    outer
                ].get("link") is None:
                    # Not wired from outside: the instance's own promoted widget
                    # value wins, falling back to the inner node's value.
                    has_value, value = self._promoted_value(instance_id, origin_slot)
                    return ("value", value) if has_value else None
                cursor = (instance_id, outer)
                continue

            if origin_id in self.instances:  # subgraph output boundary, step in
                cursor = (f"{origin_id}:-20", origin_slot)
                continue

            node = self.nodes.get(origin_id)
            if node is None:
                return None
            if node.get("type") in PASSTHROUGH:
                cursor = (origin_id, 0)
                continue
            if node.get("type") in VIRTUAL_VALUE:
                values = node.get("widgets_values") or []
                if not values:
                    return None
                return ("value", values[0])
            if node.get("mode") == MODE_BYPASS:
                cursor = self._bypass_input(origin_id, origin_slot)
                if cursor is None:
                    return None
                continue
            if node.get("mode") == MODE_NEVER:
                return None
            return ("link", (origin_id, origin_slot))
        return None

    def _bypass_input(self, node_id: str, output_slot: int) -> tuple[str, int] | None:
        """A bypassed node forwards each output to its first input of that type."""
        node = self.nodes[node_id]
        outputs = node.get("outputs") or []
        if output_slot >= len(outputs):
            return None
        wanted = outputs[output_slot].get("type")
        for index, slot in enumerate(node.get("inputs") or []):
            if slot.get("type") == wanted:
                return (node_id, index)
        return None


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #


class Converter:
    def __init__(self, object_info: dict[str, Any]) -> None:
        self.object_info = object_info
        self._plans: dict[str, WidgetPlan] = {}

    def plan(self, node_type: str) -> WidgetPlan:
        if node_type not in self._plans:
            self._plans[node_type] = WidgetPlan(self.object_info[node_type])
        return self._plans[node_type]

    def convert(
        self, workflow: dict[str, Any], resolve_switches: bool = False
    ) -> dict[str, Any]:
        graph = FlatGraph(workflow)
        prompt: dict[str, Any] = {}

        for flat_id, node in graph.nodes.items():
            node_type = node.get("type")
            if flat_id in graph.instances:
                continue
            if node_type in FRONTEND_ONLY or node_type in PASSTHROUGH:
                continue
            if node.get("mode") in (MODE_NEVER, MODE_BYPASS):
                continue
            if node_type not in self.object_info:
                raise ConversionError(f"node {flat_id} has unknown type {node_type!r}")
            prompt[flat_id] = {
                "class_type": node_type,
                "inputs": self._inputs(graph, flat_id, node),
            }

        if resolve_switches:
            self._resolve_switches(prompt)
        self.prune(prompt)
        return prompt

    def _resolve_switches(self, prompt: dict[str, Any]) -> None:
        """Replace each switch whose condition is a constant with its branch.

        A corpus run wants one deterministic path through the graph: the unused
        branch is dropped so its models never have to be synced.
        """
        progress = True
        while progress:
            progress = False
            for node_id, node in list(prompt.items()):
                if node["class_type"] != SWITCH_NODE:
                    continue
                switch = node["inputs"].get("switch")
                if not isinstance(switch, bool):
                    continue
                branch = "on_true" if switch else "on_false"
                if branch not in node["inputs"]:
                    raise ConversionError(
                        f"switch {node_id} selects {branch}, which is not connected"
                    )
                chosen = node["inputs"][branch]
                for other in prompt.values():
                    for name, value in list(other["inputs"].items()):
                        if (
                            isinstance(value, list)
                            and len(value) == 2
                            and value[0] == node_id
                        ):
                            other["inputs"][name] = (
                                list(chosen) if isinstance(chosen, list) else chosen
                            )
                del prompt[node_id]
                progress = True

    # -- widget values --------------------------------------------------- #

    def _consume(
        self,
        name: str,
        entry: Any,
        values: list[Any],
        cursor: int,
        out: dict[str, Any],
    ) -> int:
        """Read one widget (and any inputs it unlocks) off ``widgets_values``."""
        opts = _entry_opts(entry)
        if (
            isinstance(entry, (list, tuple))
            and entry[0] == "COMBO"
            and not opts.get("options")
            and "default" not in opts
        ):
            raise ConversionError(
                "combo whose options are defined on the node instance, not the "
                "server: the widget layout cannot be derived from /object_info"
            )
        if opts.get("hidden"):
            value = default_of(entry)
        elif cursor < len(values):
            value = values[cursor]
            cursor += 1
            if opts.get("control_after_generate") and cursor < len(values):
                cursor += 1  # the UI-only control widget that follows a seed
        else:
            value = default_of(entry)
        out[name] = value

        if isinstance(entry, (list, tuple)) and entry[0] == DYNAMIC_COMBO:
            for sub_name, sub_entry in dynamic_combo_subinputs(entry, value):
                cursor = self._consume(
                    f"{name}.{sub_name}", sub_entry, values, cursor, out
                )
        return cursor

    def _widget_values(self, node: dict[str, Any], node_type: str) -> dict[str, Any]:
        spec = self.object_info[node_type]
        values = node.get("widgets_values")
        if isinstance(values, dict):
            return dict(values)
        values = list(values or [])
        resolved: dict[str, Any] = {}
        cursor = 0
        for name, entry in _ordered_inputs(spec):
            if not _is_widget(entry):
                continue
            cursor = self._consume(name, entry, values, cursor, resolved)
        # Trailing values left over from a widget the node no longer shows are
        # stale serialisation; the frontend ignores them and so do we.
        return resolved

    # -- inputs ------------------------------------------------------------ #

    def _inputs(
        self, graph: FlatGraph, flat_id: str, node: dict[str, Any]
    ) -> dict[str, Any]:
        node_type = node["type"]
        spec = self.object_info[node_type]
        widget_values = self._widget_values(node, node_type)

        links: dict[str, tuple[str, int]] = {}
        overrides: dict[str, Any] = {}
        slot_names: list[str] = []
        for index, slot in enumerate(node.get("inputs") or []):
            name = (slot.get("widget") or {}).get("name") or slot.get("name")
            slot_names.append(name)
            resolved = graph.resolve(flat_id, index)
            if resolved is None:
                continue
            kind, payload = resolved
            if kind == "link":
                links[name] = payload
            else:
                overrides[name] = payload

        def pick(name: str, fallback: dict[str, Any]) -> tuple[bool, Any]:
            if name in links:
                return True, list(links[name])
            if name in overrides:
                return True, overrides[name]
            if name in fallback:
                return True, fallback[name]
            return False, None

        inputs: dict[str, Any] = {}
        # Widgets keep their declared position, whether or not they were
        # converted to an input socket.
        for name, entry in _ordered_inputs(spec):
            if not _is_widget(entry):
                continue
            found, value = pick(name, widget_values)
            if not found:
                continue
            inputs[name] = value
            prefix = f"{name}."
            for key in [k for k in widget_values if k.startswith(prefix)]:
                sub_found, sub_value = pick(key, widget_values)
                if sub_found:
                    inputs[key] = sub_value
        # Then the socket inputs, in the order the node lists them. Dynamic
        # sockets (autogrow, matchtype) only exist on the node, not in the spec.
        for name in slot_names:
            if name in inputs:
                continue
            found, value = pick(name, {})
            if found:
                inputs[name] = value
        return inputs

    def prune(self, prompt: dict[str, Any]) -> None:
        """Drop every node that does not feed an output node."""
        outputs = [
            node_id
            for node_id, node in prompt.items()
            if self._is_output(node["class_type"])
        ]
        if not outputs:
            raise ConversionError("workflow has no output node")
        keep: set[str] = set()
        stack = list(outputs)
        while stack:
            node_id = stack.pop()
            if node_id in keep or node_id not in prompt:
                continue
            keep.add(node_id)
            for value in prompt[node_id]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    stack.append(value[0])
        for node_id in list(prompt):
            if node_id not in keep:
                del prompt[node_id]

    def _is_output(self, node_type: str) -> bool:
        if node_type.startswith(PREVIEW_PREFIXES):
            return False
        if node_type.startswith(OUTPUT_PREFIXES):
            return True
        return bool(self.object_info.get(node_type, {}).get("output_node"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--object-info", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resolve-switches",
        action="store_true",
        help="Collapse constant-condition switches to the branch they select.",
    )
    args = parser.parse_args()

    object_info = json.loads(args.object_info.read_text())
    workflow = json.loads(args.template.read_text())
    prompt = Converter(object_info).convert(
        workflow, resolve_switches=args.resolve_switches
    )
    text = json.dumps(prompt, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
