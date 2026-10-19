"""src/custom_nodes.py: the worker's pack install bookkeeping (cache, pin
markers, reconcile, requirements fallback) against a local git upstream and a
fake pip, the import status it reads from ComfyUI's server log, and the
runner's startup-failure attribution built on it."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import comfy_runner  # noqa: E402
import custom_nodes  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40
# ComfyUI's console formatter (app/logger.py ColoredFormatter, commit 63bcaec5,
# 2026-05-25) prefixes every record with a coloured level tag, terminal or not;
# the Manager's own lines are plain prints. The order is main.py's: Manager
# prestartup, torch, Manager start, then the custom-node imports and summary.
INFO = "\x1b[32m[INFO]\x1b[0m "
WARN = "\x1b[1m\x1b[33m[WARNING]\x1b[0m "
LOG = (
    "# cmd: python main.py --listen 127.0.0.1\n"
    "[START] Security scan\n"
    "[DONE] Security scan\n"
    "[PRE] ComfyUI-Manager\n"
    f"{INFO}Total VRAM 24217 MB, total RAM 128000 MB\n"
    "[START] ComfyUI-Manager\n"
    f"{INFO}[Impact Pack/Subpack] Using folder_paths to determine whitelist path: /comfyui/user/x\n"
    f"{WARN}Traceback (most recent call last):\n"
    '  File "/comfyui/custom_nodes/ComfyUI-KJNodes/__init__.py", line 1, in <module>\n'
    "ModuleNotFoundError: No module named 'color_matcher'\n"
    f"{WARN}Cannot import /comfyui/custom_nodes/ComfyUI-KJNodes module for custom nodes: "
    "No module named 'color_matcher'\n"
    f"{INFO}\n"
    "Import times for custom nodes:\n"
    f"{INFO}   0.0 seconds: /comfyui/custom_nodes/websocket_image_save.py\n"
    f"{INFO}   0.4 seconds: /comfyui/custom_nodes/ComfyUI-VideoHelperSuite\n"
    f"{INFO}   1.3 seconds (IMPORT FAILED): /comfyui/custom_nodes/ComfyUI-KJNodes\n"
    f"{INFO}   2.7 seconds: /comfyui/custom_nodes/ComfyUI-Impact-Pack\n"
    f"{INFO}\n"
    f"{INFO}Starting server\n"
)
# The same log as commits before the coloured formatter print it.
PLAIN_LOG = custom_nodes.plain_text(LOG).replace("[INFO] ", "").replace("[WARNING] ", "")
assert "\x1b" not in PLAIN_LOG and "[INFO]" not in PLAIN_LOG and "[START] ComfyUI-Manager" in PLAIN_LOG


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def rec(name="ComfyUI-KJNodes", sha=SHA_A, **extra):
    return {"name": name, "repo": f"https://github.com/example/{name}", "sha": sha, **extra}


@pytest.fixture
def upstream(tmp_path):
    """A bare repo standing in for GitHub (arbitrary shas fetchable, as on
    github.com) whose single commit ships an __init__.py and requirements.txt."""
    bare = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    (seed / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    (seed / "requirements.txt").write_text("color-matcher\n")
    git(seed, "add", "-A")
    git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    sha = git(seed, "rev-parse", "HEAD")
    git(seed, "push", "-q", str(bare), "HEAD:refs/heads/main")
    return bare, sha


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """Module paths pointed at a scratch ComfyUI checkout, cache and baked
    directory, with pip replaced by a recorder."""
    comfy = tmp_path / "comfyui"
    (comfy / "custom_nodes").mkdir(parents=True)
    (comfy / "custom_nodes" / "websocket_image_save.py").write_text("# tracked by ComfyUI\n")
    monkeypatch.setattr(custom_nodes, "COMFY_ROOT", comfy)
    monkeypatch.setattr(custom_nodes, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(custom_nodes, "BAKED_DIR", tmp_path / "baked")
    monkeypatch.setattr(custom_nodes, "CONSTRAINTS", tmp_path / "constraints.txt")
    monkeypatch.setattr(custom_nodes, "PACK_CONSTRAINTS", tmp_path / "constraints-packs.txt")
    (tmp_path / "constraints.txt").write_text("torch==2.14.1\ntorchvision==0.29.1\n")
    calls = []
    monkeypatch.setattr(custom_nodes, "_run_pip",
                        lambda args, cwd, log=print: calls.append((list(args), cwd)))
    return comfy, calls


def fill_cache(name, sha, files):
    d = custom_nodes.CACHE_ROOT / name / sha
    d.mkdir(parents=True)
    for rel, text in files.items():
        (d / rel).write_bytes(text.encode())  # bytes: no newline translation on Windows
    return d


# -- fetch ---------------------------------------------------------------------
def test_fetch_pack_checks_out_the_pinned_sha_without_a_git_dir(upstream, tmp_path):
    bare, sha = upstream
    dest = tmp_path / "cache" / "Pack" / sha
    custom_nodes.fetch_pack({"name": "Pack", "repo": str(bare), "sha": sha}, dest, log=lambda *_: None)
    assert (dest / "__init__.py").read_text() == "NODE_CLASS_MAPPINGS = {}\n"
    assert (dest / "requirements.txt").exists()
    assert not (dest / ".git").exists()
    assert [p.name for p in dest.parent.iterdir()] == [sha]  # no temp dirs left behind


def test_fetch_pack_names_an_unreachable_pin(upstream, tmp_path):
    bare, _ = upstream
    dest = tmp_path / "cache" / "Pack" / SHA_A
    with pytest.raises(custom_nodes.InstallError, match="pinned commit a{40} could not be fetched"):
        custom_nodes.fetch_pack({"name": "Pack", "repo": str(bare), "sha": SHA_A}, dest,
                                log=lambda *_: None, attempts=1)
    assert not dest.exists() and not list(dest.parent.iterdir())


# -- prepare -------------------------------------------------------------------
def test_prepare_installs_from_the_cache_and_pins(worker):
    comfy, calls = worker
    fill_cache("ComfyUI-KJNodes", SHA_A, {"__init__.py": "x", "requirements.txt": "color-matcher\n"})
    report = custom_nodes.prepare([rec()], log=lambda *_: None)
    target = comfy / "custom_nodes" / "ComfyUI-KJNodes"
    assert (target / "__init__.py").read_text() == "x"
    assert (target / custom_nodes.PIN_MARKER).read_text() == SHA_A
    info = report["packs"]["ComfyUI-KJNodes"]
    assert info["source"] == "cache" and info["kind"] == "git" and info["sha"] == SHA_A
    # Nothing baked: the requirements are installed under the constraints, once.
    assert info["requirements_changed"] is True
    assert calls == [(["-r", "requirements.txt", "--upgrade-strategy", "only-if-needed",
                       "-c", str(custom_nodes.CONSTRAINTS)], target)]
    assert report["removed"] == [] and report["pip_s"] == info["pip_s"]

    # A warm worker with the same pin: nothing copied, nothing pip-installed.
    again = custom_nodes.prepare([rec()], log=lambda *_: None)
    assert again["packs"]["ComfyUI-KJNodes"]["source"] == "installed"
    assert "requirements_changed" not in again["packs"]["ComfyUI-KJNodes"]
    assert len(calls) == 1


def test_prepare_skips_pip_when_the_image_baked_the_same_requirements(worker):
    comfy, calls = worker
    req = "color-matcher\n"
    fill_cache("ComfyUI-KJNodes", SHA_A, {"__init__.py": "x", "requirements.txt": req})
    custom_nodes.BAKED_DIR.mkdir()
    (custom_nodes.BAKED_DIR / "ComfyUI-KJNodes.requirements.sha256").write_text(
        hashlib.sha256(req.encode()).hexdigest())
    report = custom_nodes.prepare([rec()], log=lambda *_: None)
    assert calls == []
    assert "requirements_changed" not in report["packs"]["ComfyUI-KJNodes"]
    # A pack without requirements, or with the flag off, never pips either.
    fill_cache("Plain", SHA_B, {"__init__.py": "x"})
    fill_cache("Off", SHA_B, {"__init__.py": "x", "requirements.txt": "numpy\n"})
    custom_nodes.prepare([rec("Plain", SHA_B), rec("Off", SHA_B, requirements=False)],
                         log=lambda *_: None)
    assert calls == []


def test_prepare_uses_the_pack_constraints_when_the_image_wrote_them(worker):
    comfy, calls = worker
    custom_nodes.PACK_CONSTRAINTS.write_text("torch==2.14.1\nopencv-python==4.12.0.88\n")
    fill_cache("ComfyUI-Impact-Pack", SHA_A, {"__init__.py": "x", "requirements.txt": "sam2\n"})
    custom_nodes.prepare([rec("ComfyUI-Impact-Pack", pip_args=["--no-build-isolation"])],
                         log=lambda *_: None)
    assert calls[0][0][-3:] == ["-c", str(custom_nodes.PACK_CONSTRAINTS), "--no-build-isolation"]


def test_prepare_removes_only_the_packs_it_installed(worker):
    comfy, calls = worker
    cn = comfy / "custom_nodes"
    (cn / "Stale").mkdir()
    (cn / "Stale" / custom_nodes.PIN_MARKER).write_text(SHA_B)
    (cn / "Stale" / "__init__.py").write_text("x")
    (cn / "SomebodyElses").mkdir()
    (cn / "SomebodyElses" / "__init__.py").write_text("x")
    report = custom_nodes.prepare([], log=lambda *_: None)
    assert report == {"packs": {}, "removed": ["Stale"], "seconds": report["seconds"], "pip_s": 0.0}
    assert sorted(p.name for p in cn.iterdir()) == ["SomebodyElses", "websocket_image_save.py"]
    assert calls == []


def test_prepare_replaces_a_pack_whose_pin_changed(worker):
    comfy, calls = worker
    fill_cache("ComfyUI-KJNodes", SHA_A, {"__init__.py": "old", "stale.py": "x"})
    fill_cache("ComfyUI-KJNodes", SHA_B, {"__init__.py": "new"})
    custom_nodes.prepare([rec(sha=SHA_A)], log=lambda *_: None)
    report = custom_nodes.prepare([rec(sha=SHA_B)], log=lambda *_: None)
    target = comfy / "custom_nodes" / "ComfyUI-KJNodes"
    assert report["packs"]["ComfyUI-KJNodes"]["source"] == "cache"
    assert (target / "__init__.py").read_text() == "new"
    assert not (target / "stale.py").exists()
    assert (target / custom_nodes.PIN_MARKER).read_text() == SHA_B


def test_prepare_fetches_a_pin_the_cache_does_not_hold(worker, monkeypatch):
    comfy, calls = worker
    fetched = []

    def fake_fetch(record, dest, log=print, attempts=3):
        fetched.append((record["name"], record["sha"], record["repo"]))
        dest.mkdir(parents=True)
        (dest / "__init__.py").write_text("fetched")

    monkeypatch.setattr(custom_nodes, "fetch_pack", fake_fetch)
    report = custom_nodes.prepare([rec()], log=lambda *_: None)
    assert fetched == [("ComfyUI-KJNodes", SHA_A, "https://github.com/example/ComfyUI-KJNodes")]
    assert report["packs"]["ComfyUI-KJNodes"]["source"] == "fetched"
    assert (comfy / "custom_nodes" / "ComfyUI-KJNodes" / "__init__.py").read_text() == "fetched"
    assert (custom_nodes.CACHE_ROOT / "ComfyUI-KJNodes" / SHA_A / "__init__.py").exists()


def test_pip_pack_uses_the_baked_pin_or_installs_and_writes_its_files(worker):
    comfy, calls = worker
    manager = rec("ComfyUI-Manager", kind="pip", requirements=False,
                  comfy_flags=["--enable-manager"], log_ok="[START] ComfyUI-Manager",
                  files={"user/__manager/config.ini": "[default]\nnetwork_mode = offline\n"})
    custom_nodes.BAKED_DIR.mkdir()
    (custom_nodes.BAKED_DIR / "ComfyUI-Manager.pin").write_text(SHA_A)
    report = custom_nodes.prepare([manager], log=lambda *_: None)
    assert report["packs"]["ComfyUI-Manager"]["source"] == "baked" and calls == []
    assert (comfy / "user" / "__manager" / "config.ini").read_text() == "[default]\nnetwork_mode = offline\n"
    assert not (comfy / "custom_nodes" / "ComfyUI-Manager").exists()

    (custom_nodes.BAKED_DIR / "ComfyUI-Manager.pin").write_text(SHA_B)
    report = custom_nodes.prepare([manager], log=lambda *_: None)
    assert report["packs"]["ComfyUI-Manager"]["source"] == "pip"
    assert calls == [([f"git+https://github.com/example/ComfyUI-Manager@{SHA_A}",
                       "-c", str(custom_nodes.CONSTRAINTS)], None)]


def test_validate_record_rejects_what_the_worker_must_not_run():
    for bad, needle in [
        (rec("../escape"), "unsafe pack name"),
        (rec(sha="e27a505"), "full 40-hex"),
        ({**rec(), "repo": "git@github.com:x/y.git"}, "https://"),
        (rec(kind="zip"), "kind 'zip'"),
        (rec("ComfyUI-Manager", kind="pip"), "needs log_ok"),
        (rec(files={"/etc/passwd": "x"}), "relative path"),
        (rec(files={"C:\\x": "x"}), "relative path"),
        (rec(files={"../x": "y"}), "relative path"),
        (rec(files={"user\\..\\x": "y"}), "relative path"),
        (rec(comfy_flags="--enable-manager"), "list of strings"),
        ("ComfyUI-KJNodes", "not an object"),
    ]:
        with pytest.raises(custom_nodes.InstallError, match=needle):
            custom_nodes.validate_record(bad)
    ok = custom_nodes.validate_record(rec())
    assert ok["kind"] == "git" and ok["requirements"] is True and ok["files"] == {}


def test_flags_merge_without_duplicates():
    packs = [rec("A", comfy_flags=["--enable-manager", "--x"]), rec("B", comfy_flags=["--x", "--y"])]
    assert custom_nodes.comfy_flags(packs) == ["--enable-manager", "--x", "--y"]
    assert custom_nodes.merge_flags(["--input-directory", "/in", "--x"], packs) == [
        "--input-directory", "/in", "--x", "--enable-manager", "--y"]
    assert custom_nodes.merge_flags(["--a"], []) == ["--a"]


# -- import status -------------------------------------------------------------
MANAGER = rec("ComfyUI-Manager", kind="pip", log_ok="[START] ComfyUI-Manager",
              log_missing="must be installed first")
ABSENT = ("\x1b[1m\x1b[31m[ERROR]\x1b[0m \n\nTo use the `--enable-manager` feature, "
          "the `comfyui-manager` package must be installed first.\ncommand:\n\tpip install -r x\n")


@pytest.mark.parametrize("log", [LOG, PLAIN_LOG], ids=["colored", "plain"])
def test_import_status_from_comfyuis_summary(log):
    requested = [rec("ComfyUI-KJNodes"), rec("ComfyUI-VideoHelperSuite"), rec("ComfyUI-Impact-Pack"),
                 rec("ComfyUI-Impact-Subpack"), MANAGER]
    status = custom_nodes.import_status(log, requested)
    assert status == {
        "ComfyUI-KJNodes": {"import": "failed", "seconds": 1.3},
        "ComfyUI-VideoHelperSuite": {"import": "ok", "seconds": 0.4},
        "ComfyUI-Impact-Pack": {"import": "ok", "seconds": 2.7},
        "ComfyUI-Impact-Subpack": {"import": "missing", "seconds": None},
        "ComfyUI-Manager": {"import": "ok", "seconds": None},
    }
    assert custom_nodes.failed_packs(status) == ["ComfyUI-KJNodes", "ComfyUI-Impact-Subpack"]
    text = custom_nodes.describe(status)
    assert text.startswith("custom node pack(s) requested by the workflow did not load: ")
    assert "ComfyUI-KJNodes: IMPORT FAILED" in text and "ComfyUI-Impact-Subpack: not loaded (missing)" in text
    assert custom_nodes.describe(status, ["ComfyUI-KJNodes"]).endswith(": ComfyUI-KJNodes: IMPORT FAILED")

    # A pip pack: absent package vs a crash before its start marker.
    assert custom_nodes.import_status(ABSENT, [MANAGER])["ComfyUI-Manager"]["import"] == "missing"
    assert custom_nodes.import_status("[PRE] ComfyUI-Manager\nTraceback", [MANAGER])["ComfyUI-Manager"]["import"] == "failed"
    # No summary at all (server died before loading packs): every git pack is missing.
    assert custom_nodes.parse_import_summary("Total VRAM 24217 MB") is None
    assert custom_nodes.import_status("", requested[:1]) == {"ComfyUI-KJNodes": {"import": "missing", "seconds": None}}


def test_summary_lines_parse_under_the_colored_formatter():
    """The real console format: colour codes and a level tag on every record,
    which an anchored match of the bare line would miss entirely."""
    block = (f"{INFO}\nImport times for custom nodes:\n"
             f"{INFO}   0.4 seconds: /comfyui/custom_nodes/ComfyUI-VideoHelperSuite\n"
             f"{INFO}   1.3 seconds (IMPORT FAILED): /comfyui/custom_nodes/ComfyUI-KJNodes\n"
             # A record with its own line colour is wrapped in colour and reset.
             f"{INFO}\x1b[36m 112.9 seconds: /comfyui/custom_nodes/ComfyUI-Impact-Pack\x1b[0m\n"
             f"{INFO}\n")
    assert custom_nodes.parse_import_summary(block) == {
        "ComfyUI-VideoHelperSuite": {"seconds": 0.4, "failed": False},
        "ComfyUI-KJNodes": {"seconds": 1.3, "failed": True},
        "ComfyUI-Impact-Pack": {"seconds": 112.9, "failed": False},
    }
    assert custom_nodes.plain_line(f"{INFO}   0.4 seconds: /x") == "0.4 seconds: /x"
    assert custom_nodes.plain_line("   0.4 seconds: /x") == "0.4 seconds: /x"
    # Markers of a pip pack match through the colouring too.
    colored_ok = f"{INFO}\x1b[32m[START] ComfyUI-Manager\x1b[0m\n"
    assert custom_nodes.import_status(colored_ok, [MANAGER])["ComfyUI-Manager"]["import"] == "ok"


@pytest.mark.parametrize("log", [LOG, PLAIN_LOG], ids=["colored", "plain"])
def test_parse_import_summary_reads_the_last_block_and_tolerates_interleaved_lines(log):
    text = log + f"\nImport times for custom nodes:\n[Impact Pack] wildcards loaded\n{INFO}   5.0 seconds: /comfyui/custom_nodes/ComfyUI-KJNodes\n"
    summary = custom_nodes.parse_import_summary(text)
    assert summary == {"ComfyUI-KJNodes": {"seconds": 5.0, "failed": False}}
    assert custom_nodes.parse_import_summary(log)["websocket_image_save.py"] == {"seconds": 0.0, "failed": False}


def test_startup_blame_needs_proof_that_pack_loading_was_reached():
    kj, vhs = rec("ComfyUI-KJNodes"), rec("ComfyUI-VideoHelperSuite")
    # No summary: the server fell over before loading packs (CUDA init, OOM, a
    # core import error); the git packs read as missing but nobody is blamed.
    early = f"{INFO}Total VRAM 24217 MB\nRuntimeError: CUDA driver initialization failed\n"
    assert custom_nodes.import_status(early, [kj])["ComfyUI-KJNodes"]["import"] == "missing"
    assert custom_nodes.startup_blame(early, [kj, vhs, MANAGER]) == []
    assert custom_nodes.startup_blame("", [kj, MANAGER]) == []
    # The summary exists: a pack that failed or has no line there is blamed, as
    # it would be once the server answered; the ok ones are not.
    assert custom_nodes.startup_blame(LOG, [kj, vhs, rec("ComfyUI-Impact-Subpack"), MANAGER]) == [
        "ComfyUI-KJNodes", "ComfyUI-Impact-Subpack"]
    assert custom_nodes.startup_blame(LOG, [vhs, MANAGER]) == []
    # A pip pack ComfyUI reported absent is blamed even before the summary;
    # one whose start marker is merely missing from a truncated log is not.
    assert custom_nodes.startup_blame(ABSENT + early, [kj, MANAGER]) == ["ComfyUI-Manager"]
    assert custom_nodes.startup_blame("[PRE] ComfyUI-Manager\n" + early, [MANAGER]) == []
    # A precomputed status is honoured.
    status = custom_nodes.import_status(LOG, [kj])
    assert custom_nodes.startup_blame(LOG, [kj], status) == ["ComfyUI-KJNodes"]


def test_runner_reclassifies_a_startup_failure_only_when_a_pack_is_proven_at_fault(tmp_path):
    log = tmp_path / "server.log"
    required = [rec("ComfyUI-KJNodes"), rec("ComfyUI-VideoHelperSuite"), MANAGER]

    # Died before loading packs: infra status stands, packs attached as detail.
    log.write_text(f"{INFO}Total VRAM 24217 MB\nRuntimeError: CUDA driver initialization failed\n")
    e = comfy_runner.ExecutionFailure("server_died", {"returncode": 1})
    comfy_runner.explain_failure(e, log, required)
    assert e.status == "server_died" and e.detail["returncode"] == 1
    assert e.detail["custom_nodes"]["ComfyUI-KJNodes"]["import"] == "missing"
    assert "startup_failure" not in e.detail and "message" not in e.detail
    assert e.detail["log_tail"].endswith("CUDA driver initialization failed\n")

    # Never answered, but the summary shows a pack failed: that is the verdict.
    log.write_text(LOG)
    e = comfy_runner.ExecutionFailure("server_start_timeout", {"timeout_s": 300})
    comfy_runner.explain_failure(e, log, required)
    assert e.status == "custom_node_error" and e.detail["startup_failure"] == "server_start_timeout"
    assert e.detail["message"] == ("custom node pack(s) requested by the workflow did not load: "
                                   "ComfyUI-KJNodes: IMPORT FAILED")
    assert e.detail["custom_nodes"]["ComfyUI-VideoHelperSuite"]["import"] == "ok"

    # Summary present and every requested pack ok: the death is not theirs.
    e = comfy_runner.ExecutionFailure("server_died", {"returncode": -9})
    comfy_runner.explain_failure(e, log, [rec("ComfyUI-VideoHelperSuite"), MANAGER])
    assert e.status == "server_died" and "startup_failure" not in e.detail

    # The pip package ComfyUI reported absent is blamed even without a summary;
    # the git pack, unproven, is left out of the message.
    log.write_text(ABSENT + f"{INFO}Total VRAM 24217 MB\n")
    e = comfy_runner.ExecutionFailure("server_died", {"returncode": 1})
    comfy_runner.explain_failure(e, log, required)
    assert e.status == "custom_node_error" and e.detail["startup_failure"] == "server_died"
    assert e.detail["message"].endswith(": ComfyUI-Manager: not loaded (missing)")

    # A failure past startup keeps its status; one that already carries the
    # packs' status (the import check itself) is left alone.
    log.write_text(LOG)
    e = comfy_runner.ExecutionFailure("execution_error", {"message": "boom"})
    comfy_runner.explain_failure(e, log, required)
    assert e.status == "execution_error" and e.detail["custom_nodes"]["ComfyUI-KJNodes"]["import"] == "failed"
    e = comfy_runner.ExecutionFailure("custom_node_error", {"custom_nodes": {"X": {"import": "failed"}}})
    comfy_runner.explain_failure(e, log, required)
    assert e.detail["custom_nodes"] == {"X": {"import": "failed"}} and "log_tail" in e.detail
    # Nothing requested: only the tail is attached.
    e = comfy_runner.ExecutionFailure("server_died", {})
    comfy_runner.explain_failure(e, log, [])
    assert e.status == "server_died" and set(e.detail) == {"log_tail"}


def test_check_imports_waits_briefly_for_the_summary(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(f"{INFO}Total VRAM 24217 MB\n")
    t0 = time.monotonic()
    status = custom_nodes.check_imports(log, [rec("ComfyUI-KJNodes")], wait_s=0.4)
    assert status["ComfyUI-KJNodes"]["import"] == "missing"
    assert time.monotonic() - t0 >= 0.3
    log.write_text(LOG)
    t0 = time.monotonic()
    status = custom_nodes.check_imports(log, [rec("ComfyUI-VideoHelperSuite")], wait_s=5)
    assert status["ComfyUI-VideoHelperSuite"]["import"] == "ok" and time.monotonic() - t0 < 1
    # A pip-only request needs no summary, so it never waits.
    manager = rec("ComfyUI-Manager", kind="pip", log_ok="[START] ComfyUI-Manager")
    log.write_text("")
    t0 = time.monotonic()
    assert custom_nodes.check_imports(log, [manager], wait_s=5)["ComfyUI-Manager"]["import"] == "failed"
    assert time.monotonic() - t0 < 1


def test_merge_status_joins_install_info_with_imports():
    installed = {"A": {"sha": SHA_A, "source": "cache", "pip_s": 0.0}}
    imports = {"A": {"import": "ok", "seconds": 0.4}, "B": {"import": "missing", "seconds": None}}
    assert custom_nodes.merge_status(installed, imports) == {
        "A": {"sha": SHA_A, "source": "cache", "pip_s": 0.0, "import": "ok", "seconds": 0.4},
        "B": {"import": "missing", "seconds": None}}
    assert custom_nodes.merge_status(None, None) == {}
    assert custom_nodes.merge_status(installed, None) == installed


# -- image build ---------------------------------------------------------------
def test_bake_records_hashes_pins_and_pack_constraints(worker, tmp_path, monkeypatch):
    comfy, calls = worker

    def fake_fetch(record, dest, log=print, attempts=3):
        dest.mkdir(parents=True)
        (dest / "__init__.py").write_text("x")
        if record["name"] != "Plain":
            (dest / "requirements.txt").write_bytes(f"dep-for-{record['name']}\n".encode())

    monkeypatch.setattr(custom_nodes, "fetch_pack", fake_fetch)
    registry = {"constraints": ["opencv-python==4.12.0.88", "opencv-python-headless==4.12.0.88"],
                "packs": {
                    "ComfyUI-KJNodes": {"repo": "https://github.com/k/KJ", "sha": SHA_A, "label": "KJ"},
                    "Plain": {"repo": "https://github.com/k/Plain", "sha": SHA_B, "label": "P"},
                    "ComfyUI-Manager": {"repo": "https://github.com/c/M", "sha": SHA_B, "label": "M",
                                        "kind": "pip", "requirements": False, "log_ok": "[START]"}}}
    manifest = tmp_path / "custom_nodes.json"
    manifest.write_text(json.dumps(registry))
    custom_nodes.bake(manifest, log=lambda *_: None)

    assert custom_nodes.PACK_CONSTRAINTS.read_text() == (
        "torch==2.14.1\ntorchvision==0.29.1\nopencv-python==4.12.0.88\nopencv-python-headless==4.12.0.88\n")
    baked = custom_nodes.BAKED_DIR
    assert (baked / "ComfyUI-KJNodes.requirements.sha256").read_text() == \
        hashlib.sha256(b"dep-for-ComfyUI-KJNodes\n").hexdigest()
    assert (baked / "Plain.requirements.sha256").read_text() == "none"
    assert (baked / "ComfyUI-Manager.pin").read_text() == SHA_B
    kj_src = custom_nodes.CACHE_ROOT / "ComfyUI-KJNodes" / SHA_A
    assert calls == [
        (["-r", "requirements.txt", "--upgrade-strategy", "only-if-needed",
          "-c", str(custom_nodes.PACK_CONSTRAINTS)], kj_src),
        ([f"git+https://github.com/c/M@{SHA_B}", "-c", str(custom_nodes.PACK_CONSTRAINTS)], None),
    ]
    # What the bake recorded is what prepare() compares against on the worker.
    assert custom_nodes.baked_requirements_hash("ComfyUI-KJNodes") == \
        hashlib.sha256(b"dep-for-ComfyUI-KJNodes\n").hexdigest()
    assert custom_nodes.baked_pin("ComfyUI-Manager") == SHA_B
    assert custom_nodes.baked_pin("nope") == ""
