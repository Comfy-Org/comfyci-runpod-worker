"""Custom-node packs: install them at pinned commits for one job and read
their import status back from the ComfyUI server log.

A job names the packs it needs as records from manifest/custom_nodes.json
(shipped in the job input, so the worker needs no manifest copy). Each `git`
pack is a depth-1 fetch of its pinned sha into a local cache
(CUSTOM_NODES_CACHE/<name>/<sha>, filled at image build time by `bake`),
copied into ComfyUI/custom_nodes/<name> with a pin marker. Its requirements
are pip-installed under the lane's constraints only when the file's hash
differs from what the image baked, so a pin bump without a rebuild still
works. Packs installed by an earlier job that this job does not request are
removed, so a job without packs runs a bare custom_nodes/. ComfyUI's
.gitignore covers custom_nodes/ and user/, so checkout.py's `git clean`
never touches either. A pack's install.py is never run: dependencies come
from its requirements.txt, models from the workflow manifest.

A `pip` pack (ComfyUI-Manager 4.x is a Python package that ComfyUI loads
with --enable-manager, not a custom_nodes directory) is installed from its
repo at the sha when the image is built; the pin that was baked is recorded
and a different sha is installed at job time.

Import status comes from ComfyUI's own summary, logged once every pack was
loaded (nodes.init_external_custom_nodes):

    Import times for custom nodes:
       0.0 seconds: /comfyui/custom_nodes/websocket_image_save.py
       1.3 seconds (IMPORT FAILED): /comfyui/custom_nodes/ComfyUI-KJNodes

ComfyUI's console formatter (app/logger.py, ColoredFormatter, since May
2026) prefixes every record with a coloured level tag, "\\x1b[32m[INFO]\\x1b[0m "
for these lines, whether or not the output is a terminal, so the log is read
with ANSI codes and the tag stripped; older commits print the bare lines.

A pip pack has no line there; its record names the log line that proves it
loaded (`log_ok`) and the one ComfyUI prints when the package is absent
(`log_missing`).

    python custom_nodes.py bake --manifest manifest/custom_nodes.json
        (Dockerfile) fetch every pack into the cache, install requirements,
        record the hashes and pins the image baked.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
# Pristine checkouts per pack and sha; the image ships them for the pinned shas.
CACHE_ROOT = Path(os.environ.get("CUSTOM_NODES_CACHE", "/comfy-cache/custom_nodes"))
# Written at image build time: <name>.requirements.sha256 / <name>.pin per pack.
BAKED_DIR = Path(os.environ.get("BAKED_CUSTOM_NODES_DIR", "/comfy-config/custom_nodes"))
# The lane's torch pins, and the same plus the registry's own pins (opencv) for pack installs.
CONSTRAINTS = Path(os.environ.get("COMFY_CONSTRAINTS_FILE", "/comfy-config/constraints.txt"))
PACK_CONSTRAINTS = Path(os.environ.get("PACK_CONSTRAINTS_FILE", "/comfy-config/constraints-packs.txt"))

PIN_MARKER = ".comfyci-pin"
REQ_MARKER = ".comfyci-requirements.sha256"
KINDS = ("git", "pip")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUMMARY_HEADER = "Import times for custom nodes:"
SUMMARY_LINE = re.compile(r"^\s*(?P<seconds>\d+\.\d+) seconds(?P<failed> \(IMPORT FAILED\))?: (?P<path>.+?)\s*$")
# ANSI escape sequences (colour, bold, reset) and the "[INFO] " / "[WARNING] "
# tag ComfyUI's console formatter puts in front of every record (optional, so
# the bare lines of older commits and leading whitespace normalise the same).
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
LEVEL_TAG_RE = re.compile(r"^\s*(?:\[[A-Z]+\]\s*)?")


class InstallError(RuntimeError):
    pass


# -- records -------------------------------------------------------------------
def safe_relative_path(rel) -> bool:
    """A path that stays inside the checkout on any platform: relative on both
    POSIX and Windows rules, no parent references."""
    if not isinstance(rel, str) or not rel:
        return False
    posix, windows = PurePosixPath(rel), PureWindowsPath(rel)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or rel.startswith(("/", "\\")):
        return False
    return ".." not in posix.parts and ".." not in windows.parts


def validate_record(rec: dict) -> dict:
    """A normalised pack record, or InstallError naming what is wrong with it."""
    if not isinstance(rec, dict):
        raise InstallError(f"pack record is not an object: {rec!r}")
    name = rec.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise InstallError(f"unsafe pack name: {name!r}")
    sha = rec.get("sha")
    if not isinstance(sha, str) or not SHA_RE.match(sha):
        raise InstallError(f"{name}: sha must be a full 40-hex commit, got {sha!r}")
    repo = rec.get("repo")
    if not isinstance(repo, str) or not repo.startswith("https://"):
        raise InstallError(f"{name}: repo must be an https:// url, got {repo!r}")
    kind = rec.get("kind", "git")
    if kind not in KINDS:
        raise InstallError(f"{name}: kind {kind!r} (expected one of {', '.join(KINDS)})")
    out = {"name": name, "repo": repo, "sha": sha, "kind": kind,
           "requirements": bool(rec.get("requirements", True))}
    for key in ("pip_args", "comfy_flags"):
        value = rec.get(key) or []
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise InstallError(f"{name}: {key} must be a list of strings")
        out[key] = list(value)
    files = rec.get("files") or {}
    if not isinstance(files, dict):
        raise InstallError(f"{name}: files must map relative paths to text")
    for rel, content in files.items():
        if not safe_relative_path(rel) or not isinstance(content, str):
            raise InstallError(f"{name}: files entry {rel!r} must be a relative path with text content")
    out["files"] = dict(files)
    if kind == "pip":
        if not isinstance(rec.get("log_ok"), str) or not rec["log_ok"]:
            raise InstallError(f"{name}: a pip pack needs log_ok, the log line that proves it loaded")
        out["log_ok"] = rec["log_ok"]
        out["log_missing"] = rec.get("log_missing") if isinstance(rec.get("log_missing"), str) else None
    return out


def comfy_flags(requested: list[dict]) -> list[str]:
    """Server flags the packs contribute (e.g. --enable-manager), in order, deduplicated."""
    out: list[str] = []
    for rec in requested:
        for flag in rec.get("comfy_flags") or []:
            if flag not in out:
                out.append(flag)
    return out


def merge_flags(job_flags: list[str], requested: list[dict]) -> list[str]:
    out = list(job_flags)
    for flag in comfy_flags(requested):
        if flag not in out:
            out.append(flag)
    return out


# -- git / pip ------------------------------------------------------------------
def _git(args: list[str], cwd: Path, timeout: int = 300) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise InstallError(f"git {' '.join(args)} failed: {proc.stderr.strip()[-1500:]}")
    return proc.stdout.strip()


def _rmtree(path: Path):
    """rmtree that also clears read-only bits (git pack files on Windows)."""
    def onerror(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def _run_pip(args: list[str], cwd: Path | None, log=print) -> None:
    """pip in the lane's venv; the image's ENV (SAM2_BUILD_CUDA etc.) applies."""
    cmd = ["pip", "install", *args]
    log(f"[custom_nodes] {' '.join(cmd)}" + (f" (in {cwd})" if cwd else ""))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                          timeout=3600)
    if proc.returncode != 0:
        raise InstallError(f"pip install failed: {proc.stderr.strip()[-2000:]}")


def constraints_args() -> list[str]:
    for path in (PACK_CONSTRAINTS, CONSTRAINTS):
        if path.exists():
            return ["-c", str(path)]
    return []


def fetch_pack(rec: dict, dest: Path, log=print, attempts: int = 3) -> None:
    """Depth-1 fetch of the pinned sha into `dest` (the frontend gate's recipe:
    init, remote add, fetch --depth 1 <sha>, checkout FETCH_HEAD), .git dropped."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        _git(["init", "-q", "-b", "main", "."], tmp)
        _git(["remote", "add", "origin", rec["repo"]], tmp)
        last = None
        for attempt in range(1, attempts + 1):
            try:
                _git(["fetch", "--depth", "1", "origin", rec["sha"]], tmp, timeout=600)
                break
            except InstallError as e:
                last = e
                log(f"[custom_nodes] {rec['name']}: fetch attempt {attempt} failed: {e}")
                if attempt < attempts:
                    time.sleep(2 * attempt)
        else:
            raise InstallError(f"{rec['name']}: pinned commit {rec['sha']} could not be fetched "
                               f"from {rec['repo']}: {last}")
        _git(["checkout", "-q", "FETCH_HEAD"], tmp)
        _rmtree(tmp / ".git")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            _rmtree(tmp)


def cached_source(rec: dict, log=print) -> tuple[Path, bool]:
    """The pristine checkout for (name, sha); fetched into the cache when absent."""
    path = CACHE_ROOT / rec["name"] / rec["sha"]
    if path.exists():
        return path, False
    log(f"[custom_nodes] fetching {rec['name']}@{rec['sha'][:12]} from {rec['repo']}")
    fetch_pack(rec, path, log)
    return path, True


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def baked_requirements_hash(name: str) -> str:
    return _read(BAKED_DIR / f"{name}.requirements.sha256")


def baked_pin(name: str) -> str:
    return _read(BAKED_DIR / f"{name}.pin")


def _pip_requirements(src: Path, rec: dict, log=print) -> float:
    t0 = time.monotonic()
    _run_pip(["-r", "requirements.txt", "--upgrade-strategy", "only-if-needed",
              *constraints_args(), *rec.get("pip_args", [])], src, log)
    return round(time.monotonic() - t0, 2)


def _pip_package(rec: dict, log=print) -> float:
    t0 = time.monotonic()
    _run_pip([f"git+{rec['repo']}@{rec['sha']}", *constraints_args(), *rec.get("pip_args", [])],
             None, log)
    return round(time.monotonic() - t0, 2)


def _write_files(rec: dict) -> None:
    """Pack-provided files relative to the ComfyUI checkout (a Manager config
    that keeps its startup offline, say); written before the server starts."""
    for rel, content in (rec.get("files") or {}).items():
        path = COMFY_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


# -- per job --------------------------------------------------------------------
def prepare(requested: list[dict], log=print) -> dict:
    """Bring ComfyUI/custom_nodes in line with `requested`: remove packs this
    job did not ask for, install the ones it did at their shas, pip the
    requirements the image did not bake. Returns per-pack install info."""
    t0 = time.monotonic()
    records = [validate_record(r) for r in requested]
    names = {r["name"] for r in records}
    cn_dir = COMFY_ROOT / "custom_nodes"
    cn_dir.mkdir(parents=True, exist_ok=True)

    removed = []
    for entry in sorted(cn_dir.iterdir()):
        # Only directories this module installed carry the marker; ComfyUI's own
        # tracked files (websocket_image_save.py, the example) are never touched.
        if entry.is_dir() and (entry / PIN_MARKER).exists() and entry.name not in names:
            _rmtree(entry)
            removed.append(entry.name)
            log(f"[custom_nodes] removed {entry.name} (not requested by this job)")

    packs: dict[str, dict] = {}
    pip_total = 0.0
    for rec in records:
        t1 = time.monotonic()
        info: dict = {"sha": rec["sha"], "kind": rec["kind"], "source": None, "pip_s": 0.0}
        if rec["kind"] == "git":
            target = cn_dir / rec["name"]
            if _read(target / PIN_MARKER) == rec["sha"]:
                info["source"] = "installed"  # warm worker, same pin
            else:
                if target.exists():
                    _rmtree(target)
                src, fetched = cached_source(rec, log)
                shutil.copytree(src, target)
                (target / PIN_MARKER).write_text(rec["sha"], encoding="utf-8")
                info["source"] = "fetched" if fetched else "cache"
            if rec["requirements"]:
                now = _sha256_file(target / "requirements.txt")
                baked = baked_requirements_hash(rec["name"])
                if now and now != baked and now != _read(target / REQ_MARKER):
                    log(f"[custom_nodes] {rec['name']}: requirements differ from the image's "
                        f"(baked {baked[:12] or 'none'}), installing")
                    info["pip_s"] = _pip_requirements(target, rec, log)
                    info["requirements_changed"] = True
                    (target / REQ_MARKER).write_text(now, encoding="utf-8")
        else:
            if baked_pin(rec["name"]) == rec["sha"]:
                info["source"] = "baked"
            else:
                log(f"[custom_nodes] {rec['name']}: image baked pin "
                    f"{baked_pin(rec['name'])[:12] or 'none'}, installing {rec['sha'][:12]}")
                info["pip_s"] = _pip_package(rec, log)
                info["source"] = "pip"
        _write_files(rec)
        pip_total += info["pip_s"]
        # install_s, not seconds: merge_status() later adds the import time under
        # "seconds", and the run record should keep both.
        info["install_s"] = round(time.monotonic() - t1, 2)
        packs[rec["name"]] = info

    return {"packs": packs, "removed": removed,
            "seconds": round(time.monotonic() - t0, 2), "pip_s": round(pip_total, 2)}


# -- import status --------------------------------------------------------------
def plain_text(text: str) -> str:
    """The log without ANSI escape sequences, so markers match whatever colour
    ComfyUI's console formatter wrapped them in."""
    return ANSI_RE.sub("", text)


def plain_line(line: str) -> str:
    """One log line without escape sequences, its leading level tag and
    leading whitespace: the record's message as ComfyUI logged it."""
    return LEVEL_TAG_RE.sub("", ANSI_RE.sub("", line), count=1)


def parse_import_summary(text: str) -> dict[str, dict] | None:
    """{basename: {"seconds", "failed"}} from the last import summary in the
    log, or None when ComfyUI has not printed one. Lines are read with the
    colour codes and the "[INFO] " tag stripped (bare lines parse the same)."""
    text = plain_text(text)
    idx = text.rfind(SUMMARY_HEADER)
    if idx < 0:
        return None
    out: dict[str, dict] = {}
    for line in text[idx + len(SUMMARY_HEADER):].splitlines():
        m = SUMMARY_LINE.match(plain_line(line))
        if m:
            name = os.path.basename(m.group("path").rstrip("/\\"))
            out[name] = {"seconds": float(m.group("seconds")), "failed": bool(m.group("failed"))}
    return out


def import_status(text: str, requested: list[dict]) -> dict[str, dict]:
    """Per requested pack: import "ok", "failed" (ComfyUI logged IMPORT FAILED)
    or "missing" (no line in the summary; for a pip pack, ComfyUI said the
    package is absent) plus the import time when the summary has one."""
    text = plain_text(text)
    summary = parse_import_summary(text)
    status: dict[str, dict] = {}
    for rec in requested:
        name = rec["name"]
        if rec.get("kind", "git") == "pip":
            ok, missing = rec.get("log_ok"), rec.get("log_missing")
            if ok and ok in text:
                st = "ok"
            elif missing and missing in text:
                st = "missing"
            else:
                st = "failed"
            status[name] = {"import": st, "seconds": None}
        else:
            row = (summary or {}).get(name)
            if row is None:
                st = "missing"
            else:
                st = "failed" if row["failed"] else "ok"
            status[name] = {"import": st, "seconds": row["seconds"] if row else None}
    return status


def check_imports(log_path: Path, requested: list[dict], wait_s: float = 5.0) -> dict[str, dict]:
    """import_status over the server log once it is ready; waits briefly for
    the summary block when a git pack is requested, since the log is flushed
    per record and the block is written just before the server binds."""
    deadline = time.monotonic() + wait_s
    needs_summary = any(rec.get("kind", "git") == "git" for rec in requested)
    while True:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if not needs_summary or parse_import_summary(text) is not None or time.monotonic() >= deadline:
            return import_status(text, requested)
        time.sleep(0.25)


def failed_packs(status: dict[str, dict]) -> list[str]:
    return [name for name, info in status.items() if info.get("import") != "ok"]


def startup_blame(text: str, requested: list[dict], status: dict[str, dict] | None = None) -> list[str]:
    """Requested packs the log proves would have failed the import check, for
    a server that died or never answered. Once ComfyUI has printed its import
    summary, pack loading was reached and every requested pack that is not ok
    there is to blame, exactly as with a server that came up. Before the
    summary only a pip pack that ComfyUI reported absent (its `log_missing`
    line) is: a log without the summary otherwise means the server fell over
    earlier (CUDA init, an OOM kill, a core import error on the commit under
    test, a cold start past the wait), which is the infrastructure's problem,
    not the pack's, and keeps its own status."""
    status = import_status(text, requested) if status is None else status
    reached = parse_import_summary(text) is not None
    blamed = []
    for rec in requested:
        name = rec["name"]
        st = (status.get(name) or {}).get("import")
        if st == "ok":
            continue
        if reached or (rec.get("kind", "git") == "pip" and st == "missing"):
            blamed.append(name)
    return blamed


def describe(status: dict[str, dict], names: list[str] | None = None) -> str:
    """One line for the failure record: which requested packs did not load
    (all that are not ok, or just `names`)."""
    parts = []
    for name in failed_packs(status) if names is None else names:
        st = status[name].get("import")
        parts.append(f"{name}: {'IMPORT FAILED' if st == 'failed' else 'not loaded (' + str(st) + ')'}")
    return "custom node pack(s) requested by the workflow did not load: " + "; ".join(parts)


def merge_status(installed: dict | None, imports: dict | None) -> dict:
    """Install info from prepare() joined with the import status, per pack."""
    out: dict[str, dict] = {}
    for name in list(installed or {}) + [n for n in (imports or {}) if n not in (installed or {})]:
        out[name] = {**((installed or {}).get(name) or {}), **((imports or {}).get(name) or {})}
    return out


# -- image build ----------------------------------------------------------------
def bake(manifest_path: Path, log=print) -> None:
    """Fetch every registry pack into the cache and install its requirements
    under the lane's pins (plus the registry's own), recording per pack the
    requirements hash or pip pin the image baked."""
    import json
    registry = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    BAKED_DIR.mkdir(parents=True, exist_ok=True)
    lines = CONSTRAINTS.read_text(encoding="utf-8").splitlines() if CONSTRAINTS.exists() else []
    lines += list(registry.get("constraints") or [])
    PACK_CONSTRAINTS.parent.mkdir(parents=True, exist_ok=True)
    PACK_CONSTRAINTS.write_text("".join(f"{line}\n" for line in lines if line.strip()), encoding="utf-8")
    for name, pack in (registry.get("packs") or {}).items():
        rec = validate_record({"name": name, **pack})
        if rec["kind"] == "git":
            src, _ = cached_source(rec, log)
            digest = ""
            if rec["requirements"] and (src / "requirements.txt").exists():
                _pip_requirements(src, rec, log)
                digest = _sha256_file(src / "requirements.txt")
            (BAKED_DIR / f"{name}.requirements.sha256").write_text(digest or "none", encoding="utf-8")
        else:
            _pip_package(rec, log)
            (BAKED_DIR / f"{name}.pin").write_text(rec["sha"], encoding="utf-8")
        log(f"[custom_nodes] baked {name}@{rec['sha'][:12]} ({rec['kind']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bake")
    p.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "bake":
        bake(Path(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
