"""scripts/packs.py: the custom-node registry and the job records a workflow
entry resolves to."""
import copy

import pytest

import packs

REGISTRY = packs.load_registry()


def test_the_committed_registry_is_consistent():
    assert packs.validate(REGISTRY) == []
    assert REGISTRY["schema_version"] == 1
    assert set(REGISTRY["packs"]) == {"ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite",
                                      "ComfyUI-Impact-Pack", "ComfyUI-Impact-Subpack",
                                      "ComfyUI-Manager"}


def test_resolve_builds_job_records_in_manifest_order():
    entry = {"custom_nodes": ["ComfyUI-Impact-Subpack", "ComfyUI-Impact-Pack"]}
    recs = packs.resolve(entry, REGISTRY)
    assert [r["name"] for r in recs] == ["ComfyUI-Impact-Subpack", "ComfyUI-Impact-Pack"]
    impact = recs[1]
    assert impact["kind"] == "git" and impact["requirements"] is True
    assert impact["repo"] == "https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    assert impact["sha"] == "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
    assert impact["pip_args"] == ["--no-build-isolation"]
    # Labels and notes stay in the registry; the worker gets only what it installs from.
    assert "label" not in impact and "note" not in impact
    assert set(recs[0]) == {"name", "repo", "sha", "kind", "requirements"}


def test_resolve_manager_carries_flags_files_and_log_markers():
    rec = packs.resolve({"custom_nodes": ["ComfyUI-Manager"]}, REGISTRY)[0]
    assert rec["kind"] == "pip" and rec["requirements"] is False
    assert rec["comfy_flags"] == ["--enable-manager"]
    assert "network_mode = offline" in rec["files"]["user/__manager/config.ini"]
    assert rec["log_ok"] == "[START] ComfyUI-Manager"
    assert rec["log_missing"] == "must be installed first"


def test_resolve_refuses_unknown_packs():
    with pytest.raises(SystemExit, match="unknown custom node pack 'ComfyUI-Ghost'"):
        packs.resolve({"custom_nodes": ["ComfyUI-Ghost"]}, REGISTRY)


def test_records_for_needs_no_registry_without_packs(tmp_path):
    assert packs.records_for({}, tmp_path) == []
    assert packs.records_for({"custom_nodes": []}, tmp_path) == []
    with pytest.raises(FileNotFoundError):
        packs.records_for({"custom_nodes": ["ComfyUI-KJNodes"]}, tmp_path)
    assert packs.records_for({"custom_nodes": ["ComfyUI-KJNodes"]})[0]["name"] == "ComfyUI-KJNodes"


def test_validate_names_each_problem():
    bad = copy.deepcopy(REGISTRY)
    bad["packs"]["ComfyUI-KJNodes"]["sha"] = "e27a505"
    bad["packs"]["ComfyUI-VideoHelperSuite"]["repo"] = "git@github.com:Kosinkadink/x.git"
    bad["packs"]["ComfyUI-Impact-Pack"]["kind"] = "zip"
    bad["packs"]["ComfyUI-Impact-Pack"]["pip_args"] = "--no-build-isolation"
    del bad["packs"]["ComfyUI-Impact-Subpack"]["label"]
    del bad["packs"]["ComfyUI-Manager"]["log_ok"]
    bad["packs"]["ComfyUI-Manager"]["files"] = {"../etc/passwd": "x", "/etc/passwd": "y"}
    bad["packs"]["../escape"] = dict(REGISTRY["packs"]["ComfyUI-KJNodes"])
    bad["constraints"].append("opencv-python")
    text = "\n".join(packs.validate(bad))
    for needle in ("pack ComfyUI-KJNodes: sha must be a full 40-hex commit",
                   "pack ComfyUI-VideoHelperSuite: repo must be an https://github.com/ url",
                   "pack ComfyUI-Impact-Pack: kind 'zip'",
                   "pack ComfyUI-Impact-Pack: pip_args must be a list of strings",
                   "pack ComfyUI-Impact-Subpack: missing label",
                   "pack ComfyUI-Manager: a pip pack needs log_ok",
                   "pack ComfyUI-Manager: files entry '../etc/passwd'",
                   "pack ComfyUI-Manager: files entry '/etc/passwd'",
                   "pack '../escape': not a safe custom_nodes directory name",
                   "constraint 'opencv-python' is not a name==version pin"):
        assert needle in text, needle
    assert packs.validate({"packs": {}}) == ["registry declares no packs"]
