"""The custom-node pack registry (manifest/custom_nodes.json) as the
orchestration scripts consume it.

A pack is keyed by its directory name under ComfyUI/custom_nodes and pinned
to a commit; a workflow entry lists the packs it needs under `custom_nodes`.
The records a job ships to the worker (src/custom_nodes.py) are built here,
so the worker needs no copy of the manifest and a pin bump is one PR to this
repo. Pins follow the frontend's custom-node gate
(ComfyUI_frontend browser_tests/fixtures/data/customNodeManifest.core.json)
so both CI systems test the same pack commits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

REGISTRY_FILE = Path(__file__).resolve().parents[1] / "manifest" / "custom_nodes.json"
# The fields a job record carries, straight from the registry entry.
JOB_FIELDS = ("repo", "sha", "kind", "requirements", "pip_args", "comfy_flags", "files",
              "log_ok", "log_missing")
KINDS = ("git", "pip")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+==\S+$")


def load_registry(path: Path | None = None) -> dict:
    return json.loads(Path(path or REGISTRY_FILE).read_text(encoding="utf-8"))


def resolve(entry: dict, registry: dict) -> list[dict]:
    """Job records for a workflow entry's `custom_nodes`, in manifest order."""
    out = []
    for name in entry.get("custom_nodes") or []:
        pack = (registry.get("packs") or {}).get(name)
        if pack is None:
            raise SystemExit(f"workflow lists unknown custom node pack {name!r}; "
                             "declare it in manifest/custom_nodes.json")
        rec = {"name": name, **{k: pack[k] for k in JOB_FIELDS if k in pack}}
        rec.setdefault("kind", "git")
        rec.setdefault("requirements", True)
        out.append(rec)
    return out


def records_for(entry: dict, repo_root: Path | None = None) -> list[dict]:
    """resolve() against the registry next to the workflows manifest; [] for a
    workflow without packs, for which the registry need not even exist."""
    if not entry.get("custom_nodes"):
        return []
    path = (Path(repo_root) / "manifest" / "custom_nodes.json") if repo_root else REGISTRY_FILE
    return resolve(entry, load_registry(path))


def validate(registry: dict) -> list[str]:
    """Problems with a registry, as messages; an empty list means it is consistent."""
    problems: list[str] = []
    packs = registry.get("packs") or {}
    if not packs:
        problems.append("registry declares no packs")
    for name, pack in packs.items():
        if not NAME_RE.match(name):
            problems.append(f"pack {name!r}: not a safe custom_nodes directory name")
        if not SHA_RE.match(str(pack.get("sha", ""))):
            problems.append(f"pack {name}: sha must be a full 40-hex commit")
        if not str(pack.get("repo", "")).startswith("https://github.com/"):
            problems.append(f"pack {name}: repo must be an https://github.com/ url")
        if not pack.get("label"):
            problems.append(f"pack {name}: missing label")
        kind = pack.get("kind", "git")
        if kind not in KINDS:
            problems.append(f"pack {name}: kind {kind!r}")
        if not isinstance(pack.get("requirements", True), bool):
            problems.append(f"pack {name}: requirements must be a boolean")
        if kind == "pip" and not pack.get("log_ok"):
            problems.append(f"pack {name}: a pip pack needs log_ok")
        for key in ("pip_args", "comfy_flags"):
            if key in pack and not (isinstance(pack[key], list)
                                    and all(isinstance(v, str) for v in pack[key])):
                problems.append(f"pack {name}: {key} must be a list of strings")
        for rel in (pack.get("files") or {}):
            posix, windows = PurePosixPath(rel), PureWindowsPath(rel)
            if (posix.is_absolute() or windows.is_absolute() or windows.drive
                    or ".." in posix.parts or ".." in windows.parts):
                problems.append(f"pack {name}: files entry {rel!r} must stay inside the checkout")
    for line in registry.get("constraints") or []:
        if not PIN_RE.match(line):
            problems.append(f"constraint {line!r} is not a name==version pin")
    return problems
