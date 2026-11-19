"""src/handler.py and src/comfy_runner.py: what a run job answers. A job that
does not ask for a fetch answers what it always did, byte for byte. One that
asks has the models the volume lacks fetched first, for no longer than the
job says. Output files are sent smallest first up to the size limit and
listed by name, and a dotted sub-input stays in the prompt. The two image
files the worker reads its environment from are held to what these rely on."""
import base64
import hashlib
import http.client
import json
import re
import sys
import types
from pathlib import Path

import pytest

import run_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# The worker's SDK is not installed where the tests run, and the handler starts
# its job loop when it is imported. A stand-in takes that call.
sys.modules["runpod"] = types.SimpleNamespace(
    serverless=types.SimpleNamespace(start=lambda config: None))

import comfy_runner  # noqa: E402
import handler  # noqa: E402
import model_sync  # noqa: E402

FETCH_LOCAL = model_sync.fetch_local  # the Worker below puts a stand-in in its place
COMMIT = "c" * 40
MODELS = [{"directory": "vae", "name": "ae.safetensors", "url": "https://huggingface.co/a/b/x",
           "sha256": "1" * 64, "bytes": 300},
          {"directory": "text_encoders", "name": "t5.safetensors",
           "url": "https://huggingface.co/a/c/y", "sha256": "2" * 64, "bytes": 500},
          {"directory": "input", "name": "cat.png", "url": "https://example.org/cat.png",
           "bytes": 20}]
# What a per-commit job sends today (scripts/submit_and_poll.py).
JOB = {"mode": "run", "comfy_commit": COMMIT, "comfy_repo": None, "workflow_id": "wf",
       "workflow": {"1": {"class_type": "SaveImage", "inputs": {}}}, "models": MODELS,
       "timeout_s": 600, "comfy_flags": [], "lane": "py313-torch2.14.1-cu130", "custom_nodes": []}
# The answer of the handler before jobs could fetch models, to the job above
# on the worker below: every key, value and position of it.
ANSWER_BEFORE = (
    '{"status": "ok", "workflow_id": "wf", "timings": {"checkout_s": 1.9, "pip_s": 0.0, '
    '"server_start_s": 5.1, "prompt_exec_s": 12.3}, "lane": "py313-torch2.14.1-cu130", '
    '"env": {"image": "cu130-t2141", "image_sha": "abc", "torch": "2.14.1+cu130", '
    '"python": "3.13.16", "runtime_backends": ["comfy-kitchen backend: cuda"]}, '
    '"commit_checked_out": "cccccccccccccccccccccccccccccccccccccccc", '
    '"comfy_version": "0.45.0", "torch_version": "2.14.1+cu130", "python_version": "3.13.16", '
    '"gpu_name": "NVIDIA GeForce RTX 4090", "vram_peak_mb": 21480.0, "rss_peak_mb": 30110.5, '
    '"validation": {"missing_nodes": [], "stripped_inputs": [], "filled_defaults": [], '
    '"ok": true}, "outputs": [{"filename": "wf_00001_.png", "bytes": 8, '
    '"sha256": "0f910e785fbcc9a739ddc7dce0deb2b4c15bb889c25613a6658e58973ec84af0", '
    '"data_b64": "iVBORyBvbmU="}, {"filename": "wf_mesh_00001_.glb", "bytes": 12, '
    '"sha256": "faa1107e9d18ad7f3640cf2c68600e48ce8974b9880a1209cb9a488610ebd167", '
    '"data_b64": "Z2xURiBtZXNoIDEy"}]}')


class Worker:
    """The handler's surroundings, replaced: the checkout, the volume, the
    pack install, the ComfyUI run and the fetch. It records what was called."""

    def __init__(self, tmp_path, monkeypatch, missing=(), report=None):
        self.calls: list = []
        self.files = []
        for name, body in (("wf_00001_.png", b"\x89PNG one"), ("wf_mesh_00001_.glb",
                                                               b"glTF mesh 12")):
            path = tmp_path / name
            path.write_bytes(body)
            self.files.append({"filename": name, "bytes": len(body),
                               "sha256": hashlib.sha256(body).hexdigest(), "path": str(path)})
        self.report = report
        monkeypatch.setattr(handler, "image_env",
                            lambda: {"image": "cu130-t2141", "image_sha": "abc"})
        monkeypatch.setattr(handler.checkout, "checkout_commit", lambda commit, repo=None: {
            "commit_checked_out": commit, "checkout_s": 1.9, "pip_s": 0.0})
        monkeypatch.setattr(handler.model_sync, "missing_models", lambda models: list(missing))
        monkeypatch.setattr(handler.model_sync, "fetch_local", self.fetch, raising=False)
        monkeypatch.setattr(handler.custom_nodes, "prepare", lambda requested: {
            "packs": {}, "seconds": 0.0, "pip_s": 0.0})
        monkeypatch.setattr(handler.comfy_runner, "execute", self.execute)

    def fetch(self, models, hf_token=None, streams=4):
        self.calls.append(("fetch", [f"{m['directory']}/{m['name']}" for m in models], streams))
        if isinstance(self.report, Exception):
            raise self.report
        return self.report

    def execute(self, workflow, timeout_s=600, extra_flags=None, required_custom_nodes=None):
        self.calls.append(("execute", timeout_s))
        return {"comfy_version": "0.45.0", "torch_version": "2.14.1+cu130",
                "python_version": "3.13.16", "gpu_name": "NVIDIA GeForce RTX 4090",
                "vram_peak_mb": 21480.0, "rss_peak_mb": 30110.5,
                "validation": {"missing_nodes": [], "stripped_inputs": [], "filled_defaults": [],
                               "ok": True},
                "server_start_s": 5.1, "prompt_exec_s": 12.3,
                "runtime_backends": ["comfy-kitchen backend: cuda"], "outputs": self.files}


def fetched(**over) -> dict:
    """A report of model_sync.fetch_local."""
    return {"files": ["vae/ae.safetensors", "text_encoders/t5.safetensors"], "bytes": 800,
            "seconds": 4.0, "mb_per_s": 0.0, "streams": 4, "token": True, "reused": [],
            "reused_bytes": 0, "evicted": [], "failed": [], **over}


def test_a_job_that_asks_for_no_fetch_answers_what_it_always_did(tmp_path, monkeypatch):
    worker = Worker(tmp_path, monkeypatch)
    answer = handler.handler({"input": dict(JOB)})
    assert json.dumps(answer) == ANSWER_BEFORE
    assert worker.calls == [("execute", 600.0)]
    # Models that are not on the volume end such a job as they always did.
    worker = Worker(tmp_path, monkeypatch, missing=["vae/ae.safetensors"])
    answer = handler.handler({"input": dict(JOB)})
    assert json.dumps(answer) == (
        '{"status": "missing_model", "workflow_id": "wf", "timings": {"checkout_s": 1.9, '
        '"pip_s": 0.0}, "lane": "py313-torch2.14.1-cu130", "env": {"image": "cu130-t2141", '
        '"image_sha": "abc"}, "commit_checked_out": "cccccccccccccccccccccccccccccccccccccccc", '
        '"failure": {"message": "models absent from volume", "missing": ["vae/ae.safetensors"]}}')
    assert worker.calls == []
    # `fetch_missing` must be true itself: no other value starts a download.
    for value in (False, None, "true", 1):
        worker = Worker(tmp_path, monkeypatch, missing=["vae/ae.safetensors"])
        answer = handler.handler({"input": {**JOB, "fetch_missing": value}})
        assert answer["status"] == "missing_model" and worker.calls == [], value


def test_a_job_that_asks_fetches_the_models_the_volume_does_not_hold(tmp_path, monkeypatch):
    missing = ["vae/ae.safetensors", "text_encoders/t5.safetensors"]
    worker = Worker(tmp_path, monkeypatch, missing=missing, report=fetched())
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert worker.calls == [("fetch", missing, 4), ("execute", 600.0)]
    assert answer["status"] == "ok"
    assert answer["model_fetch"] == {"files": 2, "bytes": 800, "seconds": 4.0, "mb_per_s": 0.0,
                                     "streams": 4, "token": True, "reused": 0, "reused_bytes": 0,
                                     "evicted": 0}
    assert answer["timings"]["model_fetch_s"] >= 0
    # The fetch is timed apart: the workflow's own seconds are what they were.
    assert answer["timings"]["prompt_exec_s"] == 12.3
    # Everything else is the answer of a job without a fetch.
    del answer["model_fetch"], answer["timings"]["model_fetch_s"]
    assert json.dumps(answer) == ANSWER_BEFORE


def test_only_what_is_missing_is_fetched_and_never_an_input_file(tmp_path, monkeypatch):
    # One model is on the volume: the other is fetched.
    worker = Worker(tmp_path, monkeypatch, missing=["text_encoders/t5.safetensors"],
                    report=fetched(files=["text_encoders/t5.safetensors"], bytes=500))
    answer = handler.handler({"input": {**JOB, "fetch_missing": True, "fetch_streams": 1}})
    assert worker.calls == [("fetch", ["text_encoders/t5.safetensors"], 1), ("execute", 600.0)]
    assert answer["model_fetch"]["files"] == 1
    # Nothing is missing: no fetch, and the answer names none.
    worker = Worker(tmp_path, monkeypatch)
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert worker.calls == [("execute", 600.0)] and json.dumps(answer) == ANSWER_BEFORE
    # An input file stays on the volume. One that is absent ends the job
    # before anything is downloaded.
    worker = Worker(tmp_path, monkeypatch, missing=["vae/ae.safetensors", "input/cat.png"])
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert answer["status"] == "missing_model" and worker.calls == []
    assert answer["failure"] == {"message": "models absent from volume",
                                 "missing": ["input/cat.png"]}
    assert "model_fetch" not in answer


def test_how_many_files_a_job_downloads_at_once():
    assert model_sync.FETCH_STREAMS == 4
    assert handler.fetch_streams({}) == 4
    assert [handler.fetch_streams({"fetch_streams": n}) for n in (1, 4, 16)] == [1, 4, 16]
    for junk in (0, -1, 17, 2.0, "2", True, None, [2]):
        assert handler.fetch_streams({"fetch_streams": junk}) == 4, junk
    # The runner refuses a number the worker would answer with its own 4.
    most = run_regression.FETCH_STREAMS_MAX
    assert handler.fetch_streams({"fetch_streams": most}) == most
    assert handler.fetch_streams({"fetch_streams": most + 1}) == 4


def test_no_answer_holds_the_token(tmp_path, monkeypatch):
    """The endpoint's HF_TOKEN with a line break at its end, as a pasted secret
    has it. The HTTP library refused the header and quoted it, the quote was
    the job's failure, and the failure goes into run.json on a public branch."""
    token = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    dialled = []

    def offline(connection):
        dialled.append(connection.host)
        raise OSError("offline")

    monkeypatch.setattr(http.client.HTTPSConnection, "connect", offline)
    root = tmp_path / "comfyui"
    for name, value in (("COMFY_ROOT", root), ("LOCAL_ROOT", root / "models"),
                        ("LOCAL_STATE", root / "models" / ".comfyci-fetched.json"),
                        ("HF_CACHE", tmp_path / "hf-cache"), ("RETRY_WAIT_S", 0),
                        ("VOLUME_ROOT", tmp_path / "volume"),
                        ("MODELS_ROOT", tmp_path / "volume" / "models"),
                        ("STATE_FILE", tmp_path / "volume" / ".sync-state.json"),
                        ("_tracked", lambda: None), ("_free_bytes", lambda: 10**12)):
        monkeypatch.setattr(model_sync, name, value)
    monkeypatch.setenv("HF_TOKEN", token + "\n")
    monkeypatch.setenv("no_proxy", "*")
    Worker(tmp_path, monkeypatch, missing=["vae/ae.safetensors"])
    monkeypatch.setattr(handler.model_sync, "fetch_local", FETCH_LOCAL)
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert token not in json.dumps(answer)
    # The token was good enough to send: the request got as far as the host.
    assert dialled == ["huggingface.co"] * 3 and answer["model_fetch"]["token"] is True
    assert answer["status"] == "model_fetch_error"
    assert answer["failure"]["failed"] == [{"model": "vae/ae.safetensors",
                                            "error": "URLError: <urlopen error offline>"}]
    # The volume sync gets the token in its job input.
    monkeypatch.delenv("HF_TOKEN")
    answer = handler.handler({"input": {"mode": "sync_models", "models": MODELS[:1],
                                        "hf_token": token + "\n"}})
    assert token not in json.dumps(answer) and len(dialled) == 4
    assert answer == {"status": "sync_partial", "sync": {
        "downloaded": [], "skipped": [], "deleted": [],
        "failed": [{"model": "vae/ae.safetensors", "error": "<urlopen error offline>"}]}}


def test_a_fetch_that_fails_ends_the_job_before_the_workflow_runs(tmp_path, monkeypatch):
    missing = ["vae/ae.safetensors", "text_encoders/t5.safetensors"]
    failed = [{"model": "vae/ae.safetensors", "error": "sha256 mismatch: got abc"}]
    worker = Worker(tmp_path, monkeypatch, missing=missing,
                    report=fetched(files=["text_encoders/t5.safetensors"], bytes=500,
                                   failed=failed))
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert answer["status"] == "model_fetch_error"
    assert answer["failure"] == {
        "message": "model download failed: vae/ae.safetensors: sha256 mismatch: got abc",
        "failed": failed}
    assert worker.calls == [("fetch", missing, 4)]  # the workflow did not run
    # What was downloaded before is still said: it is the speed measurement.
    assert answer["model_fetch"]["files"] == 1 and answer["model_fetch"]["bytes"] == 500
    assert "failed" not in answer["model_fetch"] and answer["timings"]["model_fetch_s"] >= 0
    assert "outputs" not in answer and "error" not in answer

    # The host refused a file: a person has to act, and the answer says where.
    refused = [{"model": "vae/ae.safetensors", "error": "HTTP 403 Forbidden", "http_status": 403,
                "host": "huggingface.co", "repo": "Lightricks/LTX-2.5"}]
    worker = Worker(tmp_path, monkeypatch, missing=missing,
                    report=fetched(files=[], bytes=0, failed=refused))
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert answer["status"] == "model_access_error"
    assert answer["failure"] == {
        "message": "the model host answered 403 for Lightricks/LTX-2.5",
        "host": "huggingface.co", "repo": "Lightricks/LTX-2.5", "http_status": 403,
        "failed": refused}
    assert worker.calls == [("fetch", missing, 4)]
    # A refusal decides the status, also beside a failure of another kind.
    worker = Worker(tmp_path, monkeypatch, missing=missing,
                    report=fetched(files=[], bytes=0, failed=failed + refused))
    assert handler.handler({"input": {**JOB, "fetch_missing": True}})["status"] == (
        "model_access_error")
    # A fetch that breaks is a worker error, like any other step that does.
    worker = Worker(tmp_path, monkeypatch, missing=missing, report=RuntimeError("no disk"))
    answer = handler.handler({"input": {**JOB, "fetch_missing": True}})
    assert answer["status"] == "worker_error" and answer["failure"]["message"] == "no disk"
    assert worker.calls == [("fetch", missing, 4)]


class Timed(Worker):
    """A Worker whose fetch records what else it was told."""

    def fetch(self, models, hf_token=None, streams=4, **told):
        self.calls.append(("told", told))
        return super().fetch(models, hf_token, streams)


def test_a_job_that_names_a_time_limit_hands_it_to_the_fetch(tmp_path, monkeypatch):
    missing = ["vae/ae.safetensors", "text_encoders/t5.safetensors"]

    def told(job_input):
        worker = Timed(tmp_path, monkeypatch, missing=missing, report=fetched())
        assert handler.handler({"input": job_input})["status"] == "ok"
        assert worker.calls[1:] == [("fetch", missing, 4), ("execute", 600.0)]
        return worker.calls[0][1]

    asked = {**JOB, "fetch_missing": True}
    assert told({**asked, "fetch_timeout_s": 2401}) == {"timeout_s": 2401}
    assert told({**asked, "fetch_timeout_s": 540.5}) == {"timeout_s": 540.5}
    # The job of a runner from before the key: the fetch is called as it was,
    # with no limit. The stand-in of the tests above takes no such argument.
    assert told(asked) == {} and handler.fetch_timeout_s(asked) is None
    worker = Worker(tmp_path, monkeypatch, missing=missing, report=fetched())
    assert handler.handler({"input": asked})["status"] == "ok"
    # Anything but a number above 0 is no limit either.
    for junk in (0, -5, "600", True, None, [600], float("inf"), float("nan")):
        assert told({**asked, "fetch_timeout_s": junk}) == {}, junk
    # The limit is for a job that fetches: without `fetch_missing` nothing reads it.
    worker = Timed(tmp_path, monkeypatch, missing=missing, report=fetched())
    answer = handler.handler({"input": {**JOB, "fetch_timeout_s": 600}})
    assert answer["status"] == "missing_model" and worker.calls == []


def test_a_download_that_was_stopped_at_its_limit_answers_what_it_reached(tmp_path,
                                                                         monkeypatch):
    missing = ["vae/ae.safetensors", "text_encoders/t5.safetensors"]
    slow = model_sync.too_slow(12_400_000_000, 56_600_000_000, 600.2, 1, 2, 600)
    worker = Timed(tmp_path, monkeypatch, missing=missing, report=fetched(
        files=["vae/ae.safetensors"], bytes=12_400_000_000, seconds=600.2, mb_per_s=20.7,
        failed=[slow]))
    answer = handler.handler({"input": {**JOB, "fetch_missing": True, "fetch_timeout_s": 600}})
    # The status is the one of any download that failed. `reason` is what
    # keeps the runner from trying the job again.
    assert answer["status"] == "model_fetch_error"
    assert answer["failure"] == {
        "message": ("the download was too slow: 12.4 of 56.6 GB in 600 s (20.7 MB/s), "
                    "1 of 2 files complete"),
        "reason": "timed_out", "timeout_s": 600, "received_bytes": 12_400_000_000,
        "wanted_bytes": 56_600_000_000, "seconds": 600.2, "mb_per_s": 20.7, "files_done": 1,
        "files_wanted": 2, "failed": [slow]}
    assert worker.calls == [("told", {"timeout_s": 600}), ("fetch", missing, 4)]
    # The record of the download says the speed, which is why the job answers.
    assert answer["model_fetch"] == {"files": 1, "bytes": 12_400_000_000, "seconds": 600.2,
                                     "mb_per_s": 20.7, "streams": 4, "token": True, "reused": 0,
                                     "reused_bytes": 0, "evicted": 0}
    assert "outputs" not in answer and "error" not in answer


# -- the output files of an answer -----------------------------------------------------
def answer_before(files: list[dict], limit: int) -> list[dict]:
    """The `outputs` list as the handler built it before: in name order, each
    file sent while the sizes so far, sent or not, stay within the limit."""
    total, outputs = 0, []
    for f in files:
        entry = {"filename": f["filename"], "bytes": f["bytes"], "sha256": f["sha256"]}
        total += f["bytes"]
        if total <= limit:
            entry["data_b64"] = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        else:
            entry["truncated"] = True
        outputs.append(entry)
    return outputs


def output_files(tmp_path, sizes: dict[str, int]) -> list[dict]:
    """Output files as comfy_runner.collect_outputs lists them: by name."""
    files = []
    for name in sorted(sizes):
        path = tmp_path / name
        path.write_bytes(name.encode()[:1] * sizes[name])
        files.append({"filename": name, "bytes": sizes[name], "path": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return files


def test_the_answer_of_a_run_under_the_limit_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(handler, "MAX_INLINE_BYTES", 1000)
    for sizes in ({"wf_00001_.png": 400}, {"wf_00001_.png": 400, "wf_00002_.png": 600},
                  {"b.png": 1, "a.glb": 900, "c.txt": 0, "d.mp4": 99},
                  {"z.png": 1000}, {}):
        files = output_files(tmp_path, sizes)
        outputs = handler.inline_outputs(files)
        assert json.dumps(outputs) == json.dumps(answer_before(files, 1000)), sizes
        assert all("data_b64" in o and "truncated" not in o for o in outputs)
        assert [o["filename"] for o in outputs] == sorted(sizes)


def test_files_are_sent_smallest_first_and_listed_in_name_order(tmp_path, monkeypatch):
    """A mesh or a clip that sorts first used to take the room, and the
    picture after it was listed and not sent."""
    monkeypatch.setattr(handler, "MAX_INLINE_BYTES", 1000)
    files = output_files(tmp_path, {"wf_00001_.glb": 950, "wf_00002_.png": 300,
                                    "wf_00003_.txt": 40, "wf_00004_.mp4": 700})
    outputs = handler.inline_outputs(files)
    assert [o["filename"] for o in outputs] == ["wf_00001_.glb", "wf_00002_.png",
                                                "wf_00003_.txt", "wf_00004_.mp4"]
    assert [o["filename"] for o in outputs if "data_b64" in o] == ["wf_00002_.png",
                                                                  "wf_00003_.txt"]
    assert [o for o in outputs if "data_b64" not in o] == [
        {"filename": name, "bytes": size, "sha256": files[i]["sha256"], "truncated": True}
        for i, name, size in ((0, "wf_00001_.glb", 950), (3, "wf_00004_.mp4", 700))]
    assert base64.b64decode(outputs[1]["data_b64"]) == b"w" * 300
    # Before, the first file by name was sent and nothing after it.
    assert [o["filename"] for o in answer_before(files, 1000) if "data_b64" in o] == [
        "wf_00001_.glb"]
    # Up to the limit and not a byte over. Files of one size go in name order.
    assert handler.sent_inline([{"bytes": 500}, {"bytes": 500}, {"bytes": 1}]) == {0, 2}
    assert handler.sent_inline([{"bytes": 500}, {"bytes": 500}]) == {0, 1}
    assert handler.sent_inline([{"bytes": 1001}, {"bytes": 1000}]) == {1}
    assert handler.sent_inline([{"bytes": 1001}]) == set()
    assert handler.sent_inline([]) == set()


def test_the_files_of_an_answer_fit_runpods_limit():
    """RunPod allows the answer of a /run job 10 MB. The files travel as
    base64, four characters for three bytes."""
    assert handler.MAX_INLINE_BYTES == 7 * 1024 * 1024
    as_base64 = len(base64.b64encode(b"\0" * handler.MAX_INLINE_BYTES))
    assert as_base64 == 9_786_712 and as_base64 < 10_000_000


# -- the prompt the worker sends ---------------------------------------------------------
def test_a_dotted_sub_input_stays_when_its_input_is_known(monkeypatch):
    """SaveVideo takes its codec as `format.codec`, sampled text generation
    its seed as `sampling_mode.seed`. /object_info lists `format` and
    `sampling_mode` only, so the worker removed them as unknown, and ComfyUI
    then rejected the prompt."""
    info = {"SaveVideo": {"input": {"required": {
                "video": ["VIDEO"], "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
                "format": ["COMFY_DYNAMICCOMBO_V3", {"options": []}]}}},
            "TextGenerate": {"input": {"required": {"prompt": ["STRING"]},
                                       "optional": {"sampling_mode": ["COMBO"]}}}}
    monkeypatch.setattr(comfy_runner, "http_get_json", lambda url, timeout=5.0: info)
    wf = {"1": {"class_type": "SaveVideo", "inputs": {
              "video": ["2", 0], "format": "mp4", "format.codec": "h264",
              "format.codec.profile": "high", "old_widget": 3, "old.sub": 1,
              "gone.codec": ["2", 1]}},
          "2": {"class_type": "TextGenerate", "inputs": {
              "prompt": "a cat", "sampling_mode": "on", "sampling_mode.seed": 7,
              "sampling_mode.temperature": 0.7, ".seed": 1, "sampling_modes.seed": 2}}}
    report = comfy_runner.validate_workflow(8199, wf)
    assert report["ok"] is True
    # A sub-input of a known input stays, at any depth. An unknown input goes,
    # and so does a dotted name whose first part is not an input. A link stays.
    assert wf["1"]["inputs"] == {"video": ["2", 0], "format": "mp4", "format.codec": "h264",
                                 "format.codec.profile": "high", "gone.codec": ["2", 1],
                                 "filename_prefix": "video/ComfyUI"}
    assert wf["2"]["inputs"] == {"prompt": "a cat", "sampling_mode": "on",
                                 "sampling_mode.seed": 7, "sampling_mode.temperature": 0.7}
    assert report["stripped_inputs"] == ["SaveVideo.old_widget=3", "SaveVideo.old.sub=1",
                                         "TextGenerate..seed=1",
                                         "TextGenerate.sampling_modes.seed=2"]
    assert report["filled_defaults"] == ["SaveVideo.filename_prefix='video/ComfyUI'"]


def test_a_workflow_without_dotted_inputs_is_validated_as_before(monkeypatch):
    info = {"KSampler": {"input": {"required": {"seed": ["INT", {"default": 0}],
                                                "steps": ["INT", {"default": 20}]},
                                   "optional": {"denoise": ["FLOAT"]}}}}
    monkeypatch.setattr(comfy_runner, "http_get_json", lambda url, timeout=5.0: info)
    wf = {"3": {"class_type": "KSampler", "inputs": {"seed": 5, "denoise": 1.0, "cfg_old": 8,
                                                     "model": ["4", 0]}},
          "9": {"class_type": "Gone", "inputs": {}}}
    report = comfy_runner.validate_workflow(8199, wf)
    assert report == {"missing_nodes": ["Gone"], "stripped_inputs": ["KSampler.cfg_old=8"],
                      "filled_defaults": ["KSampler.steps=20"], "ok": False}
    assert wf["3"]["inputs"] == {"seed": 5, "denoise": 1.0, "model": ["4", 0], "steps": 20}


# -- the image ---------------------------------------------------------------------------
def test_a_change_to_the_worker_or_its_image_runs_these_tests():
    """The worker's code is tested here, in the suite of the scripts. A pull
    request that only edits src/ or the Dockerfile has to start it."""
    workflow = (ROOT / ".github" / "workflows" / "scripts-tests.yml").read_text(encoding="utf-8")
    for path in ("src/**", "Dockerfile", "docker/extra_model_paths.yaml", "scripts/**"):
        assert workflow.count(f"'{path}'") == 2, path  # on a pull request and on a push
    assert "python -m pytest scripts/tests -q" in workflow


# The model folders ComfyUI knows (folder_paths.py, 25 of them).
COMFY_MODEL_FOLDERS = (
    "checkpoints", "configs", "loras", "vae", "text_encoders", "diffusion_models", "clip_vision",
    "style_models", "embeddings", "diffusers", "vae_approx", "controlnet", "gligen",
    "upscale_models", "latent_upscale_models", "hypernetworks", "photomaker", "classifiers",
    "model_patches", "audio_encoders", "background_removal", "frame_interpolation",
    "geometry_estimation", "optical_flow", "detection")


def test_the_volume_is_searched_for_every_model_folder_comfyui_knows():
    text = (ROOT / "docker" / "extra_model_paths.yaml").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()[1:] if line.strip()]
    mapped = dict(line.split(": ", 1) for line in lines)
    assert mapped.pop("base_path") == "/runpod-volume/models"
    assert len(COMFY_MODEL_FOLDERS) == 25 and set(COMFY_MODEL_FOLDERS) <= set(mapped)
    # The five others: two older folder names and the detector folders of a pack.
    assert sorted(set(mapped) - set(COMFY_MODEL_FOLDERS)) == [
        "clip", "ultralytics", "ultralytics_bbox", "ultralytics_segm", "unet"]
    assert len(mapped) == 30
    assert all(folder == name for name, folder in mapped.items()
               if not name.startswith("ultralytics_"))
    # ComfyUI logs each folder it is told to search. None of those lines reads
    # as one of the backend lines a run records in its `env` block.
    for name, folder in mapped.items():
        line = f"Adding extra search path {name} /runpod-volume/models/{folder}"
        assert not comfy_runner.BACKEND_LINE.search(line), line


def test_the_build_sha_does_not_undo_the_cache_of_the_installs():
    """Docker counts a build argument that changed as a change to every RUN
    below its ARG line. IMAGE_SHA is new on every build: declared at the top,
    it made each rebuild install torch and ComfyUI's requirements afresh."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = dockerfile.splitlines()
    (declared,) = [i for i, line in enumerate(lines) if line.startswith("ARG IMAGE_SHA")]
    runs = [i for i, line in enumerate(lines) if line.startswith("RUN ")]
    after = [i for i in runs if i > declared]
    # One RUN follows it: the one that writes the sha into image.json.
    assert len(after) == 1 and len(runs) > 4
    assert "${IMAGE_SHA}" in "\n".join(lines[after[0]:after[0] + 4])
    assert "> ${IMAGE_INFO_FILE}" in "\n".join(lines[after[0]:after[0] + 4])
    # Nothing above its declaration reads it, and every install is above.
    above = "\n".join(lines[:declared])
    assert "IMAGE_SHA" not in above
    for install in ("pip install ${TORCH_SPEC}", "pip install -r ${COMFY_ROOT}/requirements.txt",
                    "pip install runpod==", "bake_custom_nodes.py bake"):
        assert install in above, install
    # The other build arguments are read by the installs: they stay at the top.
    first_run = runs[0]
    for arg in ("IMAGE_ID", "PYTHON_VERSION", "CUDA_VERSION", "TORCH_INDEX", "TORCH_SPEC"):
        assert re.search(rf"^ARG {arg}=", "\n".join(lines[:first_run]), re.M), arg
    # The worker's code is copied in last, so a change to it rebuilds nothing else.
    assert lines.index("COPY src/ /worker/src/") > after[0]
