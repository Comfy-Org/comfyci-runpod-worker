"""sync_models: which endpoint the model-sync job runs on."""
import json
import sys
from pathlib import Path

import pytest

import index_builder
import lanes as lanes_mod
import sync_models

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "manifest" / "lanes.json").read_text(encoding="utf-8"))


def _lanes(primary="new", **endpoints):
    return {"primary": primary,
            "lanes": {lane: {"endpoint_id": ep, "cadence": "per-commit"}
                      for lane, ep in endpoints.items()}}


def test_the_sync_follows_the_primary_lane():
    env = {"RUNPOD_ENDPOINT_ID": "from-secret"}
    lanes = _lanes(new="ep-new", old="ep-old")
    assert sync_models.sync_endpoint(lanes, env) == ("ep-new", "primary lane new")
    assert sync_models.sync_endpoint(_lanes("old", new="ep-new", old="ep-old"), env)[0] == "ep-old"


def test_the_secret_is_only_a_fallback():
    env = {"RUNPOD_ENDPOINT_ID": "from-secret"}
    for lanes in (_lanes(new=None, old="ep-old"), _lanes("ghost", new="ep-new"), {}, None):
        endpoint, source = sync_models.sync_endpoint(lanes, env)
        assert endpoint == "from-secret" and source.startswith("RUNPOD_ENDPOINT_ID")
    with pytest.raises(SystemExit, match="no endpoint to sync on"):
        sync_models.sync_endpoint(_lanes(new=None), {})


def test_the_committed_registry_syncs_on_its_primary():
    primary = REGISTRY["primary"]
    endpoint = REGISTRY["lanes"][primary]["endpoint_id"]
    assert endpoint, "the primary lane must be provisioned"
    assert sync_models.sync_endpoint(REGISTRY, {"RUNPOD_ENDPOINT_ID": "x"}) == (
        endpoint, f"primary lane {primary}")


def test_main_submits_the_job_to_the_primary_endpoint(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "workflows.json"
    manifest.write_text(json.dumps({"defaults": {}, "workflows": {
        "a": {"enabled": True, "models": [{"directory": "vae", "name": "x.safetensors", "url": "u"}]},
        "b": {"enabled": False, "models": [{"directory": "vae", "name": "x.safetensors", "url": "u"}]},
    }}), encoding="utf-8")
    monkeypatch.setattr(index_builder, "load_lanes",
                        lambda path=None: _lanes(new="ep-new", old="ep-old"))
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "from-secret")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sent = {}

    class Client:
        def __init__(self, endpoint_id):
            sent["endpoint"] = endpoint_id

        def run(self, job_input):
            sent["input"] = job_input
            return "job1"

    monkeypatch.setattr(sync_models, "RunPodClient", Client)
    monkeypatch.setattr(sync_models, "_poll",
                        lambda client, job_id, timeout: {"status": "COMPLETED", "output": {"status": "ok"}})
    monkeypatch.setattr(sys, "argv", ["sync_models.py", "--manifest", str(manifest)])
    assert sync_models.main() == 0
    assert sent["endpoint"] == "ep-new"
    assert sent["input"] == {"mode": "sync_models",
                             "models": [{"directory": "vae", "name": "x.safetensors", "url": "u"}]}
    assert "via endpoint ep-new (primary lane new)" in capsys.readouterr().out


def _model(name, size=10):
    return {"directory": "vae", "name": name, "url": "u", "bytes": size}


def test_the_sync_downloads_what_the_lint_counts(tmp_path, monkeypatch, capsys):
    """The lint adds up the models of the workflows that a lane reads from the
    volume, and fails a manifest that would fill it. The sync used to send
    every workflow's models: 100 GB of a weekly workflow went to the volume
    uncounted, because the weekly lane fetches its models for each job."""
    lanes = {"primary": "main", "endpoint_defaults": {"network_volumes": ["vol"]}, "lanes": {
        "main": {"cadence": "per-commit", "endpoint_id": "ep-main"},
        "week": {"cadence": "weekly", "endpoint_id": None, "model_storage": "job"},
        "bare": {"cadence": "nightly", "endpoint_id": None, "network_volumes": [],
                 "model_storage": "job"},
        "old": {"cadence": "retired", "endpoint_id": "ep-old"}}}
    workflows = {
        "image": {"enabled": True, "models": [_model("a")]},
        "off": {"enabled": False, "models": [_model("b")]},
        "weekly_big": {"enabled": True, "cadence": "weekly", "models": [_model("big", 10**11)]},
        "weekly_shared": {"enabled": True, "cadence": "weekly", "models": [_model("a")]},
        "nightly_bare": {"enabled": True, "cadence": "nightly", "models": [_model("c")]},
        "retired_only": {"enabled": True, "lanes": ["old"], "models": [_model("d")]},
        "no_models": {"enabled": True, "cadence": "weekly", "models": []}}
    manifest = tmp_path / "workflows.json"
    manifest.write_text(json.dumps({"defaults": {}, "workflows": workflows}), encoding="utf-8")
    assert lanes_mod.volume_workflows({"workflows": workflows}, lanes) == ["image", "off"]
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: lanes)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sent = []

    class Client:
        def __init__(self, endpoint_id):
            pass

        def run(self, job_input):
            sent.append([m["name"] for m in job_input["models"]])
            return "job1"

    monkeypatch.setattr(sync_models, "RunPodClient", Client)
    monkeypatch.setattr(sync_models, "_poll", lambda client, job_id, timeout: {
        "status": "COMPLETED", "output": {"status": "ok"}})

    def sync(*argv):
        monkeypatch.setattr(sys, "argv", ["sync_models.py", "--manifest", str(manifest), *argv])
        sent.clear()
        return sync_models.main(), capsys.readouterr().out

    code, out = sync()
    assert code == 0 and sent == [["a", "b"]]
    assert ("not synced, no lane reads their models from the volume: weekly_big, weekly_shared, "
            "nightly_bare, retired_only\n") in out
    # Asked for by name, such a workflow is still not synced, and the run says so.
    code, out = sync("--workflows", "weekly_big")
    assert code == 0 and sent == [] and "not synced" in out and "no models in manifest" in out
    # Once the lane keeps its models on the volume, they are synced, and the lint counts them.
    del lanes["lanes"]["week"]["model_storage"]
    assert sync()[0] == 0 and sent == [["a", "b", "big"]]
    # Without a registry every workflow is synced, as before lanes said where models are.
    monkeypatch.setattr(index_builder, "load_lanes", lambda path=None: json.loads("not json"))
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "from-secret")
    assert sync()[0] == 0 and sent == [["a", "b", "big", "c", "d"]]
    # The committed files: the sync sends the models of every workflow, as it did.
    committed = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
    assert lanes_mod.volume_workflows(committed, REGISTRY) == list(committed["workflows"])


def test_the_workers_serve_the_sync_job():
    """Every lane image ships src/, whose handler serves mode sync_models, and
    the workflow serialises syncs so one endpoint writes the volume at a time."""
    handler = (ROOT / "src" / "handler.py").read_text(encoding="utf-8")
    assert 'if mode == "sync_models":' in handler and "model_sync.sync(" in handler
    assert "COPY src/ /worker/src/" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "sync-models.yml").read_text(encoding="utf-8")
    assert "group: sync-models" in workflow and "cancel-in-progress: false" in workflow
