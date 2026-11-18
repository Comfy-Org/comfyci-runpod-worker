"""The lane registry and what the workflows derive from it."""
import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

import index_builder
import lanes as lanes_mod
import runpod_endpoints
from conftest import git, workflow_steps

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "manifest" / "lanes.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "manifest" / "workflows.json").read_text(encoding="utf-8"))
PACKS = json.loads((ROOT / "manifest" / "custom_nodes.json").read_text(encoding="utf-8"))
# The committed registry provisions lanes over time; these tests want one lane that is
# declared but not yet provisioned, whatever the file says today.
NEW = "py313-torch2.15.0-cu132"
PRIMARY = "py313-torch2.14.1-cu130"
LEGACY = "py312-torch2.11.0-cu128"  # retired
PENDING = copy.deepcopy(REGISTRY)
PENDING["lanes"][NEW]["endpoint_id"] = None


def test_the_committed_registry_is_consistent():
    assert lanes_mod.validate(REGISTRY) == []
    assert lanes_mod.validate(REGISTRY, MANIFEST, PACKS) == []
    assert REGISTRY["schema_version"] == 2


def test_the_committed_lanes():
    """cu130 is the primary, cu128 is retired but keeps its history, cu132 runs per commit."""
    lanes = REGISTRY["lanes"]
    assert REGISTRY["primary"] == PRIMARY
    assert {k: lanes[PRIMARY][k] for k in ("role", "blocking", "cadence")} == {
        "role": "primary", "blocking": True, "cadence": "per-commit"}
    legacy = lanes[LEGACY]
    assert lanes_mod.is_retired(legacy) and legacy["role"] == "legacy" and not legacy["blocking"]
    # Its published history stays where the dashboard reads it.
    assert (legacy["layout"], legacy["prefix"], legacy["image"]) == (1, "regression", "cu128-t2110")
    assert legacy["endpoint_id"]
    assert REGISTRY["images"]["cu128-t2110"]["build"] is False
    new = lanes[NEW]
    assert {k: new[k] for k in ("python", "torch", "cuda", "gpu", "cadence", "blocking", "role",
                                "layout", "prefix", "image", "min_cuda_driver")} == {
        "python": "3.13", "torch": "2.15.0", "cuda": "13.2", "gpu": "RTX 4090",
        "cadence": "per-commit", "blocking": False, "role": "lane", "layout": 2,
        "prefix": f"regression/lanes/{NEW}", "image": "cu132-t2150", "min_cuda_driver": "13.2"}
    assert {k: REGISTRY["images"]["cu132-t2150"][k] for k in lanes_mod.IMAGE_FIELDS} == {
        "python": "3.13", "cuda": "13.2", "torch_index": "https://download.pytorch.org/whl/cu132",
        "torch_spec": "torch==2.15.0 torchvision==0.29.2"}
    # The pack workflows run on both lanes whose images bake the packs.
    for wf_id, entry in MANIFEST["workflows"].items():
        if entry.get("custom_nodes"):
            assert entry["lanes"] == [PRIMARY, NEW], wf_id


def test_validate_names_each_problem():
    bad = copy.deepcopy(REGISTRY)
    bad["primary"] = "nope"
    lane = bad["lanes"]["py313-torch2.14.1-cu130"]
    lane["image"] = "ghost"
    lane["prefix"] = "regression/elsewhere"
    lane["cadence"] = "hourly"
    del lane["endpoint_id"]
    bad["images"]["cu130-t2141"].pop("torch_spec")
    problems = "\n".join(lanes_mod.validate(bad))
    for needle in ("primary 'nope'", "image 'ghost'", "regression/lanes/py313-torch2.14.1-cu130",
                   "cadence 'hourly'", "endpoint_id must be present", "missing torch_spec"):
        assert needle in problems


def test_validate_rejects_a_second_legacy_layout_and_prefix_reuse():
    bad = copy.deepcopy(REGISTRY)
    bad["lanes"]["py313-torch2.14.1-cu130"].update({"layout": 1, "prefix": "regression"})
    problems = "\n".join(lanes_mod.validate(bad))
    assert "only one lane may use the legacy layout 1" in problems
    assert "already used by" in problems


def test_validate_checks_lane_filters_and_the_pack_lane_rule():
    manifest = {"workflows": {
        "typo": {"enabled": True, "lanes": ["nope"]},
        "empty": {"enabled": True, "lanes": []},
        "unrestricted_pack": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"]},
        "legacy_pack": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"], "lanes": [LEGACY]},
        "ghost_pack": {"enabled": True, "custom_nodes": ["ComfyUI-Ghost"], "lanes": [NEW]},
        "fine": {"enabled": True, "custom_nodes": ["ComfyUI-KJNodes"], "lanes": [NEW]},
        "plain": {"enabled": True},
    }}
    problems = lanes_mod.validate(REGISTRY, manifest, PACKS)
    text = "\n".join(problems)
    assert "workflow typo: lanes names unknown lane 'nope'" in text
    assert "workflow empty: lanes must be null or a non-empty list" in text
    assert "workflow unrestricted_pack: needs custom node packs" in text
    assert f"workflow legacy_pack: lane {LEGACY} runs image cu128-t2110, which is never rebuilt" in text
    assert "workflow ghost_pack: custom node pack 'ComfyUI-Ghost' is not in" in text
    assert not [p for p in problems if p.startswith("workflow fine") or p.startswith("workflow plain")]
    # Without the pack registry, unknown pack names are not checked here.
    assert "ComfyUI-Ghost" not in "\n".join(lanes_mod.validate(REGISTRY, manifest))


def test_workflow_lane_filter():
    assert lanes_mod.workflow_runs_on_lane({"lanes": None}, "any")
    assert lanes_mod.workflow_runs_on_lane({}, "any")
    assert lanes_mod.workflow_runs_on_lane({"lanes": [NEW]}, NEW)
    assert not lanes_mod.workflow_runs_on_lane({"lanes": [NEW]}, LEGACY)
    manifest = {"workflows": {
        "a": {"enabled": True},
        "b": {"enabled": True, "lanes": [NEW]},
        "c": {"enabled": False, "lanes": [NEW]},
        "d": {"enabled": True, "lanes": [LEGACY]},
    }}
    assert lanes_mod.workflows_for_lane(manifest, NEW) == ["a", "b"]
    assert lanes_mod.workflows_for_lane(manifest, LEGACY) == ["a", "d"]
    assert lanes_mod.workflows_for_lane(manifest, NEW, enabled_only=False) == ["a", "b", "c"]
    # The committed pack workflows are restricted to the baking lane.
    for wf_id, entry in MANIFEST["workflows"].items():
        if entry.get("custom_nodes"):
            assert not lanes_mod.workflow_runs_on_lane(entry, LEGACY), wf_id


def test_the_level_and_the_cadence_of_a_workflow_entry():
    """Absent or null means today's behaviour: compared with a reference, on
    every commit. Anything that is not a known value is a mistake."""
    assert lanes_mod.LEVELS == ("reference", "smoke")
    assert lanes_mod.WORKFLOW_CADENCES == ("per-commit", "nightly", "weekly")
    for entry in (None, {}, {"level": None, "cadence": None}):
        assert lanes_mod.level_of(entry) == "reference"
        assert lanes_mod.cadence_of(entry) == "per-commit"
    assert lanes_mod.level_of({"level": "smoke"}) == "smoke"
    assert lanes_mod.level_of({"level": "reference"}) == "reference"
    assert [lanes_mod.cadence_of({"cadence": c}) for c in ("per-commit", "nightly", "weekly")] == [
        "per-commit", "nightly", "weekly"]
    for odd in ("", "Smoke", "main", False, 0, ["smoke"]):
        with pytest.raises(ValueError, match="unknown level .* \\(known: reference, smoke\\)"):
            lanes_mod.level_of({"level": odd})
    # A lane can be retired. A workflow cannot ask for that: it is switched off instead.
    for odd in ("", "retired", "daily", True, 7):
        with pytest.raises(ValueError, match="unknown cadence .* \\(known: per-commit, nightly, "
                                             "weekly\\)"):
            lanes_mod.cadence_of({"cadence": odd})
    # The cadence is validated only so far: the lanes list alone decides where a workflow runs.
    assert lanes_mod.workflow_runs_on_lane({"cadence": "weekly"}, PRIMARY)
    # No committed entry uses a value the functions do not know.
    for wf_id, entry in MANIFEST["workflows"].items():
        assert lanes_mod.level_of(entry) in lanes_mod.LEVELS, wf_id
        assert lanes_mod.cadence_of(entry) in lanes_mod.WORKFLOW_CADENCES, wf_id
        assert isinstance(lanes_mod.expect_of(entry), dict), wf_id


def test_what_a_smoke_entry_expects_of_its_output():
    assert lanes_mod.EXPECT_KEYS == ("width", "height", "frames", "seconds", "min_chars")
    for entry in (None, {}, {"expect": None}, {"level": "smoke"}, {"level": "smoke", "expect": {}}):
        assert lanes_mod.expect_of(entry) == {}
    expect = {"width": 1024, "height": 1024, "frames": 25, "seconds": 30.5, "min_chars": 40}
    assert lanes_mod.expect_of({"level": "smoke", "expect": expect}) == expect
    # At reference level the output is held to its reference, not to an expectation.
    for entry in ({"expect": {"width": 64}}, {"level": "reference", "expect": {}}):
        with pytest.raises(ValueError, match="expect is for a smoke-level workflow only"):
            lanes_mod.expect_of(entry)
    with pytest.raises(ValueError, match="unknown level"):
        lanes_mod.expect_of({"level": "main", "expect": {"width": 64}})
    for odd in ([], "width=64", 64, True):
        with pytest.raises(ValueError, match="expect must be an object"):
            lanes_mod.expect_of({"level": "smoke", "expect": odd})
    with pytest.raises(ValueError, match="unknown expect key 'size' \\(known: width, height, "
                                         "frames, seconds, min_chars\\)"):
        lanes_mod.expect_of({"level": "smoke", "expect": {"width": 64, "size": 3}})
    for value in (0, -1, 0.0, "64", None, True, False, [64], float("nan")):
        with pytest.raises(ValueError, match="expect.width must be a number above 0, not "):
            lanes_mod.expect_of({"level": "smoke", "expect": {"width": value}})


def test_a_nightly_or_weekly_lane_is_a_long_cadence_lane():
    assert lanes_mod.LONG_CADENCES == ("nightly", "weekly")
    assert set(lanes_mod.LONG_CADENCES) < set(lanes_mod.CADENCES)
    assert lanes_mod.is_long_cadence({"cadence": "nightly"})
    assert lanes_mod.is_long_cadence({"cadence": "weekly"})
    for lane in ({"cadence": "per-commit"}, {"cadence": "retired"}, {}, None):
        assert not lanes_mod.is_long_cadence(lane)


def test_image_matrix_skips_unbuilt_images_unless_asked_for():
    rows = lanes_mod.image_matrix(REGISTRY)
    assert [r["image"] for r in rows] == ["cu130-t2141", "cu132-t2150"]
    assert rows[0] == {"image": "cu130-t2141", "python": "3.13", "cuda": "13.0",
                       "torch_index": "https://download.pytorch.org/whl/cu130",
                       "torch_spec": "torch==2.14.1 torchvision==0.29.1"}
    assert [r["image"] for r in lanes_mod.image_matrix(REGISTRY, "cu132-t2150")] == ["cu132-t2150"]
    # The retired lane's image is never built again, even when asked for...
    assert lanes_mod.image_matrix(REGISTRY, "cu128-t2110") == []
    # ...while a legacy image still in service can be, by name, despite build: false.
    serving = copy.deepcopy(REGISTRY)
    serving["lanes"][LEGACY]["cadence"] = "per-commit"
    assert [r["image"] for r in lanes_mod.image_matrix(serving)] == ["cu130-t2141", "cu132-t2150"]
    assert [r["image"] for r in lanes_mod.image_matrix(serving, "cu128-t2110")] == ["cu128-t2110"]
    assert lanes_mod.image_matrix(REGISTRY, "nothing") == []


class PushRepo:
    """A repository like this one, cut to what an image is built from: each
    commit() is one commit of a push, and built() is the plan for a push that
    started after the commit `since`."""

    def __init__(self, root, registry):
        self.root, self.registry = root, copy.deepcopy(registry)
        subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
        self.first = self.commit(**{"Dockerfile": "FROM scratch\n", "src/handler.py": "x = 1\n",
                                    "README.md": "one\n"})

    def commit(self, **files) -> str:
        files["manifest/lanes.json"] = json.dumps(self.registry, indent=2) + "\n"
        for path, text in files.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text, encoding="utf-8")
        git(self.root, "add", "-A")
        git(self.root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
            "--allow-empty", "-m", "change")
        return git(self.root, "rev-parse", "HEAD").strip()

    def built(self, since, only=None) -> list[str]:
        rows = lanes_mod.image_matrix(self.registry, only, since=since, repo=self.root)
        return [row["image"] for row in rows]

    def why(self, since) -> str:
        return lanes_mod.images_since(self.registry, since, self.root)[1]


EVERY_IMAGE = ["cu130-t2141", "cu132-t2150"]


def test_a_push_that_changes_no_image_builds_none(tmp_path):
    """Adding a lane, writing an endpoint id or changing a default is an edit
    of lanes.json. It used to rebuild every image and roll it out."""
    repo = PushRepo(tmp_path, REGISTRY)
    assert repo.built(None) == EVERY_IMAGE  # no --since: a dispatch builds as before
    repo.registry["lanes"][NEW]["endpoint_id"] = "abc123"
    repo.registry["lanes"]["nightly"] = {**REGISTRY["lanes"][PRIMARY], "cadence": "nightly",
                                         "prefix": "regression/lanes/nightly"}
    repo.registry["endpoint_defaults"]["volume_gb"] = 500
    repo.registry["images"]["cu130-t2141"]["note"] = "no image is built from a note"
    lane_edit = repo.commit(**{"README.md": "two\n", ".github/workflows/build-image.yml": "x\n"})
    assert repo.built(repo.first) == []
    assert repo.why(repo.first) == f"no image changed since {repo.first[:8]}: nothing is built"
    # An image's own entry: that image, and no other.
    repo.registry["images"]["cu132-t2150"]["torch_spec"] = "torch==2.15.1 torchvision==0.29.2"
    image_edit = repo.commit()
    assert repo.built(lane_edit) == ["cu132-t2150"] == repo.built(repo.first)
    assert repo.why(lane_edit) == f"images changed since {lane_edit[:8]}: cu132-t2150"
    assert repo.built(lane_edit, only="cu130-t2141") == []
    assert repo.built(image_edit) == []
    # An image that was not built before the push: a new one, and one whose
    # only lane was retired until now.
    repo.registry["images"]["cu140-t2160"] = {**REGISTRY["images"]["cu132-t2150"], "cuda": "14.0"}
    repo.registry["lanes"]["cu140"] = {**REGISTRY["lanes"][NEW], "image": "cu140-t2160",
                                       "cuda": "14.0", "prefix": "regression/lanes/cu140"}
    del repo.registry["images"]["cu128-t2110"]["build"]
    repo.registry["lanes"][LEGACY]["cadence"] = "per-commit"
    repo.commit()
    assert repo.built(image_edit) == ["cu128-t2110", "cu140-t2160"]


def test_a_push_that_changes_what_images_are_built_from_builds_them_all(tmp_path):
    repo = PushRepo(tmp_path, REGISTRY)
    for path in ("src/handler.py", "docker/extra_model_paths.yaml", "manifest/custom_nodes.json",
                 "Dockerfile"):
        before = git(repo.root, "rev-parse", "HEAD").strip()
        repo.commit(**{path: f"changed {path}\n"})
        assert repo.built(before) == EVERY_IMAGE, path
        assert repo.why(before) == f"every image is built: {path} changed since {before[:8]}"
    # A push of two commits, src/ first and lanes.json last. Compared with the
    # last commit alone it would build nothing, and nobody would be told.
    before = repo.commit(**{"README.md": "three\n"})
    repo.commit(**{"src/handler.py": "x = 2\n"})
    repo.registry["lanes"][NEW]["endpoint_id"] = "abc123"
    last = repo.commit()
    assert repo.built(before) == EVERY_IMAGE
    assert git(repo.root, "diff", "--name-only", "HEAD~1", "HEAD").split() == [
        "manifest/lanes.json"]
    # A file moved out of src/ is a change to src/.
    git(repo.root, "mv", "src/handler.py", "handler.py")
    repo.commit()
    assert repo.built(last) == EVERY_IMAGE


def test_a_push_without_a_known_commit_before_it_builds_every_image(tmp_path, monkeypatch, capsys):
    """A new branch has all zeros there, and a forced push a commit the
    checkout cannot fetch."""
    repo = PushRepo(tmp_path, REGISTRY)
    repo.registry["lanes"][NEW]["endpoint_id"] = "abc123"
    repo.commit()
    for since in ("0" * 40, "f" * 40, "", "HEAD~1", "--output=x"):
        assert repo.built(since) == EVERY_IMAGE, since
    assert repo.why("0" * 40) == f"every image is built: no commit to compare with ({'0' * 40})"
    assert repo.why("f" * 40) == ("every image is built: ffffffff or its manifest/lanes.json is "
                                  "not in this checkout")
    assert not (repo.root / "x").exists()  # only a commit id ever reaches git
    # A checkout that is no repository at all (a source archive).
    assert lanes_mod.image_matrix(REGISTRY, since=repo.first, repo=tmp_path.parent) \
        == lanes_mod.image_matrix(REGISTRY)
    # The command: the plan on stdout, the reason on stderr.
    monkeypatch.setattr(lanes_mod, "load_lanes", lambda path=None: repo.registry)
    monkeypatch.setattr(lanes_mod, "REPO_ROOT", repo.root)
    assert lanes_mod.main(["images", "--since", repo.first]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == []
    assert out.err.strip() == f"::notice::no image changed since {repo.first[:8]}: nothing is built"
    assert lanes_mod.main(["images", "--since", "0" * 40]) == 0
    out = capsys.readouterr()
    assert [row["image"] for row in json.loads(out.out)] == EVERY_IMAGE
    assert out.err.startswith("::notice::every image is built: no commit to compare with")
    assert lanes_mod.main(["images"]) == 0
    out = capsys.readouterr()
    assert [row["image"] for row in json.loads(out.out)] == EVERY_IMAGE and out.err == ""


def test_the_build_plans_a_push_against_the_commit_before_it():
    """Not against HEAD~1: a push can hold several commits. The checkout holds
    only HEAD, so the commit before the push is fetched, and when that fails
    lanes.py builds every image."""
    plan = dict(workflow_steps(".github/workflows/build-image.yml"))["plan"]
    assert "EVENT: ${{ github.event_name }}" in plan
    assert "BEFORE: ${{ github.event.before }}" in plan
    assert re.search(r'\n( +)elif \[ "\$EVENT" = "push" \]; then\n'
                     r'\1  git fetch --quiet --no-tags --depth=1 origin "\$BEFORE" \|\| true\n'
                     r'\1  matrix=\$\(python lanes\.py images --since "\$BEFORE"\)\n', plan)
    # A dispatch names its images, or builds every one.
    assert re.search(r'if \[ -n "\$IMAGES" \]; then\n +matrix=\$\(python lanes\.py images '
                     r'--only "\$IMAGES"\)\n', plan)
    assert re.search(r"\n +else\n +matrix=\$\(python lanes\.py images\)\n +fi\n", plan)
    # Every path that starts the workflow on a push is one the plan knows.
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    paths = re.search(r"\n    paths:\n((?:      - .+\n)+)", build).group(1)
    paths = [line.strip()[2:].strip("'") for line in paths.splitlines()]
    assert sorted(paths) == sorted(["Dockerfile", "src/**", "docker/**", "manifest/lanes.json",
                                    "manifest/custom_nodes.json",
                                    ".github/workflows/build-image.yml"])
    for source in lanes_mod.IMAGE_SOURCES:
        assert source in paths or source + "**" in paths, source
    assert lanes_mod.REGISTRY_PATH in paths


def test_per_commit_matrix_only_lists_provisioned_lanes():
    rows = lanes_mod.per_commit_matrix(PENDING)
    ids = [r["lane"] for r in rows]
    assert ids == [PRIMARY]  # cu128 is retired, cu132 has no endpoint yet
    for row in rows:
        assert row["endpoint_id"]
        assert row["prefix"]
    assert lanes_mod.per_commit_matrix(PENDING, NEW) == []
    provisioned = copy.deepcopy(PENDING)
    provisioned["lanes"][NEW]["endpoint_id"] = "abc123"
    rows = lanes_mod.per_commit_matrix(provisioned)
    assert {"lane": NEW, "prefix": f"regression/lanes/{NEW}", "endpoint_id": "abc123"} in rows
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(provisioned, NEW)] == [NEW]


def test_the_retired_lane_stays_out_of_the_matrix_even_when_named():
    assert lanes_mod.per_commit_matrix(REGISTRY, LEGACY) == []
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(REGISTRY, f"{LEGACY},{PRIMARY}")] == [PRIMARY]
    assert lanes_mod.left_out(PENDING, f"{LEGACY},{NEW}") == [
        f"{LEGACY}: cadence 'retired', not per-commit", f"{NEW}: no endpoint_id yet"]
    assert lanes_mod.lanes_on_image(REGISTRY, "cu128-t2110") == []


def test_per_commit_matrix_lists_each_lanes_workflows_with_the_manifest():
    manifest = {"workflows": {"a": {"enabled": True}, "b": {"enabled": True, "lanes": [NEW]},
                              "c": {"enabled": False}}}
    provisioned = copy.deepcopy(PENDING)
    provisioned["lanes"][NEW]["endpoint_id"] = "abc123"
    rows = {r["lane"]: r for r in lanes_mod.per_commit_matrix(provisioned, manifest=manifest)}
    assert rows[PRIMARY]["workflows"] == "a"
    assert rows[NEW]["workflows"] == "a,b"


def test_lanes_on_image_never_moves_the_legacy_lane():
    assert lanes_mod.lanes_on_image(PENDING, "cu128-t2110") == []
    assert lanes_mod.lanes_on_image(PENDING, "cu130-t2141") == [PRIMARY]
    assert lanes_mod.lanes_on_image(PENDING, "cu132-t2150") == []  # not provisioned yet
    provisioned = copy.deepcopy(PENDING)
    provisioned["lanes"][NEW]["endpoint_id"] = "abc123"
    assert lanes_mod.lanes_on_image(provisioned, "cu132-t2150") == [NEW]


def test_lane_endpoint_never_falls_back_for_an_unprovisioned_lane():
    env = {"RUNPOD_ENDPOINT_ID": "fromenv"}
    reg = {"lanes": {"named": {"endpoint_id": "ep1"}, "pending": {"endpoint_id": None},
                     "old": {"prefix": "regression"}}}
    assert index_builder.lane_endpoint(reg, "named", env) == "ep1"
    assert index_builder.lane_endpoint(reg, "pending", env) is None
    assert index_builder.lane_endpoint(reg, "old", env) == "fromenv"
    assert index_builder.lane_endpoint(reg, "old", {}) is None
    assert index_builder.lane_endpoint(reg, "unknown", env) == "fromenv"


def test_set_image_plans_one_patch_per_provisioned_lane_and_refuses_foreign_tags(capsys):
    reg = copy.deepcopy(PENDING)
    # A lane without an endpoint is skipped: build-image's repoint is a no-op for it.
    assert runpod_endpoints.plan_set_image(reg, "cu132-t2150-deadbeef", image_id="cu132-t2150") == []
    assert runpod_endpoints.plan_set_image(reg, "cu132-t2150-deadbeef", lane_id=NEW) == []
    assert f"[{NEW}] no endpoint yet; skipping" in capsys.readouterr().out
    assert runpod_endpoints.set_image(reg, "cu132-t2150-1", None, "cu132-t2150", None, True) == 0
    assert "nothing to repoint" in capsys.readouterr().out
    reg["lanes"][NEW]["endpoint_id"] = "abc123"
    plan = runpod_endpoints.plan_set_image(reg, "cu132-t2150-deadbeef", image_id="cu132-t2150")
    assert plan == [(NEW, "abc123",
                     {"image": "ghcr.io/comfy-org/comfyci-runpod-worker:cu132-t2150-deadbeef"})]
    plan = runpod_endpoints.plan_set_image(reg, "cu132-t2150", lane_id=NEW)
    assert plan[0][2]["image"].endswith(":cu132-t2150")
    with pytest.raises(SystemExit):
        runpod_endpoints.plan_set_image(reg, "cu130-t2141-deadbeef", lane_id=NEW)
    with pytest.raises(SystemExit):
        runpod_endpoints.plan_set_image(reg, "cu132-t2150-deadbeef", lane_id="nope")
    # The retired lane is never moved: not by an image-wide repoint, not by name.
    assert runpod_endpoints.plan_set_image(reg, "cu128-t2110-deadbeef", image_id="cu128-t2110") == []
    with pytest.raises(SystemExit, match="is retired"):
        runpod_endpoints.plan_set_image(reg, "cu128-t2110-deadbeef", lane_id=LEGACY)


def test_endpoint_body_comes_from_defaults_plus_the_lane():
    body = runpod_endpoints.endpoint_body(PENDING, NEW, "cu132-t2150-abc")
    assert body["name"] == "comfyci-cu132-t2150"
    assert body["type"] == "QUEUE"
    assert body["image"] == "ghcr.io/comfy-org/comfyci-runpod-worker:cu132-t2150-abc"
    assert body["gpu"] == {"pools": ["ADA_24"], "count": 1, "minCudaVersion": "13.2"}
    assert body["workers"] == {"min": 0, "max": 3, "idleTimeout": 20}
    assert body["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4}
    assert body["timeout"] == 3600000 and body["flashboot"] == "FLASHBOOT" and body["disk"] == 40
    assert body["networkVolumes"] == ["n0xnkwnogx"] and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["registry"] == "cmszcugyp001tfk7ooe7rtp4d"
    assert body["env"] == {"RUNPOD_INIT_TIMEOUT": "800"}
    # An already provisioned lane is never created twice, a retired one never
    # gets one, and a foreign tag is refused.
    with pytest.raises(SystemExit, match="already has endpoint"):
        runpod_endpoints.endpoint_body(REGISTRY, PRIMARY, "cu130-t2141-abc")
    with pytest.raises(SystemExit, match="is retired"):
        runpod_endpoints.endpoint_body(REGISTRY, LEGACY, "cu128-t2110-abc")
    with pytest.raises(SystemExit, match="not a build of image"):
        runpod_endpoints.endpoint_body(PENDING, NEW, "cu130-t2141-abc")
    lane = copy.deepcopy(PENDING)
    lane["lanes"][NEW]["workers_max"] = 1
    assert runpod_endpoints.endpoint_body(lane, NEW, "cu132-t2150")["workers"]["max"] == 1


def test_write_endpoint_id_updates_only_that_lane(tmp_path):
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps(PENDING, indent=2), encoding="utf-8")
    runpod_endpoints.write_endpoint_id(NEW, "ep999", path)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["lanes"][NEW]["endpoint_id"] == "ep999"
    after["lanes"][NEW]["endpoint_id"] = None
    assert after == PENDING


def test_write_endpoint_id_keeps_the_committed_formatting(tmp_path):
    """create --write re-serialises lanes.json: the diff it leaves is the id alone."""
    text = (ROOT / "manifest" / "lanes.json").read_text(encoding="utf-8")
    path = tmp_path / "lanes.json"
    path.write_text(text, encoding="utf-8")
    current = json.loads(text)["lanes"][NEW]["endpoint_id"]
    runpod_endpoints.write_endpoint_id(NEW, current, path)
    assert path.read_text(encoding="utf-8") == text


def test_set_image_dry_run_sends_nothing(capsys):
    reg = copy.deepcopy(PENDING)
    reg["lanes"][NEW]["endpoint_id"] = "abc123"
    assert runpod_endpoints.set_image(reg, "cu132-t2150-1", None, "cu132-t2150", None, True) == 0
    assert "would PATCH abc123" in capsys.readouterr().out


class FakeEndpoints:
    """Stands in for the RunPod client: live endpoints by id, and which were read."""

    def __init__(self, live):
        self.live, self.read = live, []

    def get(self, endpoint_id):
        self.read.append(endpoint_id)
        return self.live[endpoint_id]


def _live(workers_max=3, idle=20, disk=40):
    """An endpoint as RunPod's GET answers it, cut to the keys `status` reads."""
    return {"image": "ghcr.io/comfy-org/comfyci-runpod-worker:img-abc", "disk": disk,
            "gpu": {"pools": ["ADA_24"], "count": 1, "minCudaVersion": "13.0"},
            "workers": {"idleTimeout": idle, "max": workers_max, "min": 0},
            "networkVolumes": ["n0xnkwnogx"], "dataCenterIds": ["EU-RO-1"]}


def _status_registry():
    reg = _registry(old={"cadence": "retired", "role": "legacy", "image": "old"},
                    main={"role": "primary"}, new={"endpoint_id": None})
    reg["endpoint_defaults"] = copy.deepcopy(REGISTRY["endpoint_defaults"])
    return reg


def test_status_prints_every_difference_between_runpod_and_the_manifest(capsys):
    """RunPod lowers the workers of an endpoint nobody uses, and a retired
    endpoint keeps its share of the worker quota until someone sets it to 0."""
    reg = _status_registry()
    client = FakeEndpoints({"ep-main": _live(workers_max=2), "ep-old": _live(disk=30)})
    assert runpod_endpoints.status(reg, client, None) == 0  # it reports, it does not fail
    out = capsys.readouterr().out
    assert "main: differs: workers.max is 2 on RunPod, the manifest says 3\n" in out
    assert ("old: differs: workers.max is 3 on RunPod, the manifest says 0 "
            "(the lane is retired)\n") in out
    assert "new: not provisioned" in out
    assert "2 of the settings above differ from manifest/lanes.json" in out
    # Only the endpoints the manifest names are read, and nothing else is called.
    assert client.read == ["ep-old", "ep-main"]
    # --strict turns a difference into a failure. One lane can be asked for.
    assert runpod_endpoints.status(reg, client, None, strict=True) == 1
    assert runpod_endpoints.status(reg, client, "new", strict=True) == 0
    capsys.readouterr()
    assert runpod_endpoints.status(reg, client, "main", strict=True) == 1
    out = capsys.readouterr().out
    assert out.count("differs") == 1 and "1 of the settings above differ" in out


def test_status_is_quiet_when_runpod_and_the_manifest_agree(capsys):
    reg = _status_registry()
    # A retired endpoint is held to 0 workers only: its disk and idle time do not matter.
    client = FakeEndpoints({"ep-main": _live(), "ep-old": _live(workers_max=0, idle=5, disk=30)})
    assert runpod_endpoints.status(reg, client, None, strict=True) == 0
    out = capsys.readouterr().out
    assert "differ" not in out
    assert "main: ep-main image=ghcr.io/comfy-org/comfyci-runpod-worker:img-abc" in out
    assert "workers=0-3 idle=20s" in out


def test_status_compares_workers_idle_time_and_disk_with_what_create_asks_for():
    reg = _status_registry()
    diff = runpod_endpoints.differences
    assert diff(reg, "main", _live(idle=5, disk=120)) == [
        "workers.idleTimeout is 5 on RunPod, the manifest says 20",
        "disk is 120 on RunPod, the manifest says 40"]
    # RunPod sets an endpoint without requests for 7 days to 0 workers.
    assert diff(reg, "main", _live(workers_max=0)) == [
        "workers.max is 0 on RunPod, the manifest says 3"]
    assert diff(reg, "main", {}) == [
        "workers.max is None on RunPod, the manifest says 3",
        "workers.idleTimeout is None on RunPod, the manifest says 20",
        "disk is None on RunPod, the manifest says 40"]
    # The lane's own largest number of workers, as `create` reads it.
    reg["lanes"]["main"]["workers_max"] = 2
    assert diff(reg, "main", _live(workers_max=2)) == []
    reg["lanes"]["new"]["workers_max"] = 2
    body = runpod_endpoints.endpoint_body(reg, "new", "img-abc")
    assert (body["workers"], body["disk"]) == ({"min": 0, "max": 2, "idleTimeout": 20}, 40)
    assert runpod_endpoints.endpoint_settings(reg, reg["lanes"]["new"]) == (body["workers"], 40)


def test_status_strict_is_a_flag_of_the_command(monkeypatch, capsys):
    reg = _status_registry()
    client = FakeEndpoints({"ep-main": _live(workers_max=2), "ep-old": _live(workers_max=0)})
    monkeypatch.setattr(runpod_endpoints, "load_lanes", lambda: reg)
    monkeypatch.setattr(runpod_endpoints, "Client", lambda: client)
    assert runpod_endpoints.main(["status"]) == 0
    assert runpod_endpoints.main(["status", "--strict"]) == 1
    assert runpod_endpoints.main(["status", "--strict", "--lane", "old"]) == 0
    assert "main: differs: workers.max is 2 on RunPod" in capsys.readouterr().out


def test_dockerfile_takes_every_build_arg_the_workflow_passes():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    declared = set(re.findall(r"^ARG (\w+)", dockerfile, re.M))
    workflow = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    passed = set(re.findall(r"^\s+(\w+)=\$\{\{ (?:matrix|github)\.\w+ \}\}$", workflow, re.M))
    assert passed, "build-args block not found"
    assert passed <= declared, passed - declared
    # Every image field the matrix carries reaches the Dockerfile.
    for field, arg in (("python", "PYTHON_VERSION"), ("cuda", "CUDA_VERSION"),
                       ("torch_index", "TORCH_INDEX"), ("torch_spec", "TORCH_SPEC")):
        assert f"{arg}=${{{{ matrix.{field} }}}}" in workflow


def test_dockerfile_bakes_the_pack_registry():
    """The image carries the registry and the bake step, and a registry change
    (a pin bump) triggers a rebuild."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY manifest/custom_nodes.json /comfy-config/custom_nodes.json" in dockerfile
    assert "bake --manifest /comfy-config/custom_nodes.json" in dockerfile
    assert "libgl1" in dockerfile and "SAM2_BUILD_CUDA=0" in dockerfile
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    assert "- manifest/custom_nodes.json" in build


def test_workflows_derive_their_matrices_from_the_registry():
    build = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    regression = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert "lanes.py images" in build
    assert "runpod_endpoints.py set-image" in build
    assert "lanes.py per-commit" in regression
    assert '--lane "$LANE"' in regression
    assert 'FETCH_HEAD:$PREFIX/latest/master.json' in regression


def _registry(**lanes):
    """A self-contained registry: lane id -> overrides of a provisioned lane."""
    base = {"label": "x", "python": "3.13", "torch": "2.14.1", "cuda": "13.0", "gpu": "RTX 4090",
            "cadence": "per-commit", "blocking": False, "role": "lane", "image": "img"}
    out = {"schema_version": 2, "primary": "main",
           "images": {"img": {"python": "3.13", "cuda": "13.0", "torch_index": "i", "torch_spec": "s"},
                      "old": {"python": "3.13", "cuda": "13.0", "torch_index": "i", "torch_spec": "s"}},
           "lanes": {}}
    for n, (lane_id, over) in enumerate(lanes.items()):
        layout = 1 if n == 0 else 2
        out["lanes"][lane_id] = {**base, "layout": layout,
                                 "prefix": "regression" if layout == 1 else f"regression/lanes/{lane_id}",
                                 "endpoint_id": f"ep-{lane_id}", **over}
    return out


def test_validate_holds_the_primary_to_the_primary_role():
    ok = _registry(old={"cadence": "retired", "role": "legacy", "image": "old"},
                   main={"role": "primary", "blocking": True})
    assert lanes_mod.validate(ok) == []
    for over, needle in (({"role": "lane"}, "role must be 'primary', not 'lane'"),
                         ({"role": "legacy"}, "role must be 'primary', not 'legacy'"),
                         ({"cadence": "retired"}, "cadence must be 'per-commit', not 'retired'"),
                         ({"cadence": "nightly"}, "cadence must be 'per-commit', not 'nightly'")):
        bad = copy.deepcopy(ok)
        bad["lanes"]["main"].update(over)
        problems = lanes_mod.validate(bad)
        assert any(p.startswith("primary lane main: ") and needle in p for p in problems), problems
    # Exactly one lane carries the primary role: the one `primary` names.
    bad = copy.deepcopy(ok)
    bad["lanes"]["old"]["role"] = "primary"
    assert "lane old: role 'primary', but the primary lane is 'main'" in lanes_mod.validate(bad)


def test_a_retired_lane_is_never_dispatched():
    reg = _registry(old={"cadence": "retired", "role": "legacy", "image": "old"},
                    main={"role": "primary"})
    # Per-commit matrix: out, even when named explicitly, and the plan says why.
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(reg)] == ["main"]
    assert lanes_mod.per_commit_matrix(reg, "old") == []
    assert lanes_mod.per_commit_matrix(reg, "old,main")[0]["lane"] == "main"
    assert lanes_mod.left_out(reg, "old,main,ghost") == [
        "ghost: not a lane in manifest/lanes.json", "old: cadence 'retired', not per-commit"]
    # Builds and repoints: an image only retired lanes run is never built, even
    # when asked for, and a retired lane is never repointed, whatever its role.
    assert [r["image"] for r in lanes_mod.image_matrix(reg)] == ["img"]
    assert lanes_mod.image_matrix(reg, "old") == []
    assert lanes_mod.lanes_on_image(reg, "old") == []
    shared = copy.deepcopy(reg)
    shared["lanes"]["old"].update({"image": "img", "role": "lane"})
    assert lanes_mod.lanes_on_image(shared, "img") == ["main"]
    assert runpod_endpoints.plan_set_image(shared, "img-1", image_id="img") == [
        ("main", "ep-main", {"image": "ghcr.io/comfy-org/comfyci-runpod-worker:img-1"})]
    with pytest.raises(SystemExit, match="is retired"):
        runpod_endpoints.plan_set_image(reg, "old-1", lane_id="old")
    unprovisioned = copy.deepcopy(reg)
    unprovisioned["lanes"]["old"]["endpoint_id"] = None
    unprovisioned["endpoint_defaults"] = REGISTRY["endpoint_defaults"]
    with pytest.raises(SystemExit, match="is retired"):
        runpod_endpoints.endpoint_body(unprovisioned, "old", "old-1")


def test_per_commit_plan_explains_lanes_it_leaves_out(monkeypatch, capsys):
    reg = _registry(old={"cadence": "retired", "role": "legacy", "image": "old"},
                    main={"role": "primary"})
    monkeypatch.setattr(lanes_mod, "load_lanes", lambda path=None: reg)
    assert lanes_mod.main(["per-commit", "--only", "old"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == []
    assert "not tested: old: cadence 'retired', not per-commit" in out.err


def test_a_report_only_plan_lists_every_lane(monkeypatch, capsys):
    """A report-only run reads published results and starts no GPU job, so a
    retired lane, an unprovisioned one and a nightly one are shown as well."""
    reg = _registry(old={"cadence": "retired", "role": "legacy", "image": "old"},
                    main={"role": "primary"}, new={"endpoint_id": None},
                    slow={"cadence": "nightly"})
    assert lanes_mod.report_matrix(reg) == [
        {"lane": "old", "prefix": "regression"},
        {"lane": "main", "prefix": "regression/lanes/main"},
        {"lane": "new", "prefix": "regression/lanes/new"},
        {"lane": "slow", "prefix": "regression/lanes/slow"}]
    assert [row["lane"] for row in lanes_mod.report_matrix(reg, "old, slow")] == ["old", "slow"]
    assert lanes_mod.report_matrix(reg, "ghost") == []
    # The test plan of the same registry: the one provisioned per-commit lane.
    assert [r["lane"] for r in lanes_mod.per_commit_matrix(reg)] == ["main"]
    monkeypatch.setattr(lanes_mod, "load_lanes", lambda path=None: reg)
    assert lanes_mod.main(["report", "--only", "old,ghost"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == [{"lane": "old", "prefix": "regression"}]
    assert "::notice::not shown: ghost: not a lane in manifest/lanes.json" in out.err
    assert lanes_mod.main(["report"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4
    # The committed registry: the retired lane is listed where its history is.
    assert {"lane": LEGACY, "prefix": "regression"} in lanes_mod.report_matrix(REGISTRY)
    regression = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert 'lanes.py report "${only[@]}"' in regression
