"""Drive a ComfyUI server through one workflow execution.

Ported from the pr15056 QA harness (benchmark/pr15056/runner.py): spawn the
server with an isolated output directory, wait for /system_stats, validate the
workflow against /object_info (recording every adaptation so schema drift is
visible), POST /prompt, poll /history until completed or errored, inventory the
PNG outputs with sha256 hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
EXTRA_MODEL_PATHS = os.environ.get("EXTRA_MODEL_PATHS", "/comfy-config/extra_model_paths.yaml")


class ExecutionFailure(RuntimeError):
    def __init__(self, status: str, detail: dict | None = None):
        super().__init__(status)
        self.status = status
        self.detail = detail or {}


def http_get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_server_ready(port: int, proc: subprocess.Popen, timeout: float = 300.0) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ExecutionFailure("server_died", {"returncode": proc.returncode})
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as r:
                if r.status == 200:
                    return time.monotonic()
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, socket.timeout):
            pass
        time.sleep(0.2)
    raise ExecutionFailure("server_start_timeout", {"timeout_s": timeout})


def validate_workflow(port: int, wf: dict) -> dict:
    """Check the workflow against /object_info and adapt it in place.

    Unknown primitive inputs are stripped (older commit than the workflow's
    schema), missing required widgets are filled from the server's defaults.
    Every adaptation is recorded; unknown node classes fail validation.
    """
    report = {"missing_nodes": [], "stripped_inputs": [], "filled_defaults": [], "ok": True}
    try:
        info = http_get_json(f"http://127.0.0.1:{port}/object_info", timeout=120)
    except Exception as e:
        report["ok"] = False
        report["error"] = f"object_info fetch failed: {e}"
        return report
    for nid, node in wf.items():
        ct = node["class_type"]
        if ct not in info:
            report["missing_nodes"].append(ct)
            continue
        spec = info[ct].get("input", {})
        required = spec.get("required", {})
        known = set(required) | set(spec.get("optional", {}))
        for inp in list(node.get("inputs", {})):
            if inp not in known and not isinstance(node["inputs"][inp], list):
                report["stripped_inputs"].append(f"{ct}.{inp}={node['inputs'][inp]!r}")
                del node["inputs"][inp]
        for name, rspec in required.items():
            if name in node["inputs"]:
                continue
            extra = rspec[1] if len(rspec) > 1 and isinstance(rspec[1], dict) else {}
            if "default" in extra:
                node["inputs"][name] = extra["default"]
                report["filled_defaults"].append(f"{ct}.{name}={extra['default']!r}")
    if report["missing_nodes"]:
        report["ok"] = False
    return report


def run_prompt(port: int, wf: dict, timeout_s: float) -> dict:
    client_id = f"comfyci-{uuid.uuid4().hex[:8]}"
    body = json.dumps({"prompt": wf, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/prompt", data=body,
        headers={"Content-Type": "application/json"},
    )
    t_post = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:4000]
        raise ExecutionFailure("prompt_rejected", {"detail": detail})
    prompt_id = resp["prompt_id"]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            hist = http_get_json(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=5)
            if prompt_id in hist:
                st = hist[prompt_id].get("status", {})
                if st.get("completed") or st.get("status_str") == "error":
                    entry = hist[prompt_id]
                    entry["_exec_seconds"] = round(time.monotonic() - t_post, 2)
                    return entry
        except Exception:
            pass
        time.sleep(0.5)
    raise ExecutionFailure("timeout", {"timeout_s": timeout_s, "prompt_id": prompt_id})


def start_comfy(port: int, output_dir: Path, log_path: Path, extra_flags: list[str]):
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(port),
        "--disable-auto-launch",
        "--output-directory", str(output_dir),
        *(("--extra-model-paths-config", EXTRA_MODEL_PATHS)
          if Path(EXTRA_MODEL_PATHS).exists() else ()),
        *extra_flags,
    ]
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    log_f.write(f"# cmd: {' '.join(cmd)}\n# ts: {datetime.now(timezone.utc).isoformat()}\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(COMFY_ROOT), stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def kill_comfy(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_vram_peak_mb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2, 1)
    except Exception:
        return None


def read_rss_peak_mb(pid: int) -> float | None:
    """Kernel-tracked resident-set high-water mark (VmHWM), so no sampling loop."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


def collect_outputs(output_dir: Path) -> list[dict]:
    out = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            out.append({
                "filename": str(p.relative_to(output_dir)).replace(os.sep, "/"),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "path": str(p),
            })
    return out


def execute(workflow: dict, timeout_s: float = 600, extra_flags: list[str] | None = None,
            port: int = 8199) -> dict:
    """Run one workflow on a fresh server. Returns the run record; outputs stay on disk."""
    run_uuid = uuid.uuid4().hex[:12]
    output_dir = Path(f"/tmp/comfyci-out-{run_uuid}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/tmp/comfyci-log-{run_uuid}.log")

    record: dict = {"output_dir": str(output_dir), "log_path": str(log_path)}
    t_spawn = time.monotonic()
    proc, log_f = start_comfy(port, output_dir, log_path, extra_flags or [])
    try:
        t_ready = wait_server_ready(port, proc)
        record["server_start_s"] = round(t_ready - t_spawn, 2)
        try:
            stats = http_get_json(f"http://127.0.0.1:{port}/system_stats", timeout=10)
            sysinfo = stats.get("system", {})
            record["comfy_version"] = sysinfo.get("comfyui_version")
            record["torch_version"] = sysinfo.get("pytorch_version")
            record["python_version"] = sysinfo.get("python_version")
            devices = stats.get("devices", [])
            if devices:
                record["gpu_name"] = devices[0].get("name")
        except Exception:
            pass

        record["validation"] = validate_workflow(port, workflow)
        if not record["validation"]["ok"]:
            raise ExecutionFailure("validation_error", record["validation"])

        hist = run_prompt(port, workflow, timeout_s)
        record["prompt_exec_s"] = hist.get("_exec_seconds")
        st = hist.get("status", {})
        if not st.get("completed"):
            msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            err = msgs[0][1] if msgs else {}
            raise ExecutionFailure("execution_error", {
                "node_id": err.get("node_id"),
                "node_type": err.get("node_type"),
                "message": err.get("exception_message"),
                "traceback": (err.get("traceback") or [])[-10:],
            })
        record["vram_peak_mb"] = read_vram_peak_mb()
        record["rss_peak_mb"] = read_rss_peak_mb(proc.pid)
    finally:
        kill_comfy(proc)
        try:
            log_f.close()
        except Exception:
            pass

    record["outputs"] = collect_outputs(output_dir)
    if not record["outputs"]:
        raise ExecutionFailure("no_outputs", {"log_tail": log_path.read_text(errors="replace")[-3000:]})
    return record
