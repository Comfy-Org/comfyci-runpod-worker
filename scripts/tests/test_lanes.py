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
    # Without the lane's entry in the registry the rule cannot know the lane's
    # cadence: the lanes list alone decides, as it did before workflows had a cadence.
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


def test_a_number_a_smoke_entry_expects_is_finite():
    """Python's json reads Infinity and NaN from a file as numbers, and a
    number too large for a float as Infinity. A typed Infinity passed as "a
    number above 0", and no output can meet it."""
    for typed, said in (("Infinity", "Infinity"), ("1e999", "Infinity"),
                        ("-Infinity", "-Infinity"), ("NaN", "NaN")):
        entry = json.loads('{"level": "smoke", "expect": {"width": 64, "frames": %s}}' % typed)
        with pytest.raises(ValueError) as refused:
            lanes_mod.expect_of(entry)
        assert str(refused.value) == (f"expect.frames must be a number above 0, not {said}: "
                                      "it has to be finite"), typed
    # A large number that is one is a number like any other.
    large = {"frames": 1e300, "min_chars": 10**300}
    assert lanes_mod.expect_of({"level": "smoke", "expect": large}) == large
    # The size of 1e999 written out in digits is a whole number for Python.
    # It was taken, and the smoke check that met it broke on the comparison
    # and said of the output file that it "could not be read".
    entry = json.loads('{"level": "smoke", "expect": {"min_chars": 1%s}}' % ("0" * 400))
    with pytest.raises(ValueError) as refused:
        lanes_mod.expect_of(entry)
    assert str(refused.value) == ("expect.min_chars must be a number above 0, not a whole number "
                                  "beyond 1e308: no check can compare with it")


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
    moved = repo.commit()
    assert repo.built(last) == EVERY_IMAGE
    # A name that git prints in quotes unless it is asked not to: a letter
    # outside ASCII, and a blank.
    for name in ("src/café_nodes.py", "docker/two words.yaml"):
        before = git(repo.root, "rev-parse", "HEAD").strip()
        repo.commit(**{name: "x = 1\n"})
        assert repo.built(before) == EVERY_IMAGE, name
        assert repo.why(before) == f"every image is built: {name} changed since {before[:8]}"
    assert repo.built(moved) == EVERY_IMAGE


def test_a_push_that_puts_an_endpoint_under_an_image_builds_that_image(tmp_path):
    """The build is what points an endpoint at an image. A lane that has its
    endpoint and moves to another image, leaves the legacy role or is no
    longer retired is repointed only when that image is built, so the edit
    builds it. Each such edit used to build and repoint nothing."""
    registry = copy.deepcopy(REGISTRY)
    registry["lanes"][NEW]["endpoint_id"] = "ep-new"
    registry["lanes"][NIGHTLY].update({"endpoint_id": "ep-night", "role": "legacy"})
    registry["lanes"][WEEKLY].update({"endpoint_id": "ep-week", "cadence": "retired"})
    repo = PushRepo(tmp_path / "one", registry)
    assert lanes_mod.lanes_on_image(repo.registry, "cu130-t2141") == [PRIMARY]
    # A lane moves to another image: that image is built, the one it left is not.
    repo.registry["lanes"][NEW]["image"] = "cu130-t2141"
    moved = repo.commit()
    assert repo.built(repo.first) == ["cu130-t2141"]
    assert repo.why(repo.first) == f"images changed since {repo.first[:8]}: cu130-t2141"
    assert lanes_mod.lanes_on_image(repo.registry, "cu130-t2141") == [PRIMARY, NEW]
    # A legacy lane becomes one that builds repoint.
    repo.registry["lanes"][NIGHTLY]["role"] = "lane"
    legacy = repo.commit()
    assert repo.built(moved) == ["cu130-t2141"] and repo.built(legacy) == []
    # A retired lane runs again.
    repo.registry["lanes"][WEEKLY]["cadence"] = "weekly"
    back = repo.commit()
    assert repo.built(legacy) == ["cu130-t2141"] and repo.built(back) == []
    # Not a lane that gets its first endpoint, a new lane, another endpoint id,
    # or a lane that leaves the lanes a build repoints: its endpoint is created
    # from a tag by hand, or is no longer moved at all.
    repo.registry["lanes"]["fresh"] = {**REGISTRY["lanes"][PRIMARY], "endpoint_id": "ep-fresh",
                                       "prefix": "regression/lanes/fresh"}
    repo.registry["lanes"][PRIMARY]["endpoint_id"] = "ep-other"
    repo.registry["lanes"][NEW]["role"] = "legacy"
    repo.commit()
    assert repo.built(back) == []
    # A registry of before that lists no lanes at all is read as one without endpoints.
    bare = PushRepo(tmp_path / "two", {"images": REGISTRY["images"]})
    bare.registry = copy.deepcopy(REGISTRY)
    bare.commit()
    assert bare.built(bare.first) == []


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


def test_the_build_plans_a_push_against_the_commit_the_images_were_built_from():
    """The head of the last push run of the workflow that succeeded: compared
    with the commit before the push alone, the push after a failed build
    would build nothing and the images would stay behind. That commit stands
    in when no such run is known (not HEAD~1: a push can hold several
    commits). The checkout holds only HEAD, so the commit is fetched, and
    when that fails lanes.py builds every image."""
    plan = dict(workflow_steps(".github/workflows/build-image.yml"))["plan"]
    assert "EVENT: ${{ github.event_name }}" in plan
    assert "BEFORE: ${{ github.event.before }}" in plan
    assert "GH_TOKEN: ${{ github.token }}" in plan
    assert re.search(r'\n( +)elif \[ "\$EVENT" = "push" \]; then\n'
                     r'\1  built=\$\(gh run list --repo "\$GITHUB_REPOSITORY" '
                     r'--workflow build-image\.yml \\\n'
                     r'\1    --branch main --event push --status success --limit 1 \\\n'
                     r"\1    --json headSha --jq '\.\[0\]\.headSha // empty' \|\| true\)\n"
                     r'\1  since="\$\{built:-\$BEFORE\}"\n'
                     r'\1  git fetch --quiet --no-tags --depth=1 origin "\$since" \|\| true\n'
                     r'\1  matrix=\$\(python lanes\.py images --since "\$since"\)\n', plan)
    # The plan job may read the workflow's runs, and the branch it asks about
    # is the one whose pushes start the workflow.
    whole = (ROOT / ".github" / "workflows" / "build-image.yml").read_text(encoding="utf-8")
    assert ("  plan:\n    runs-on: ubuntu-latest\n" in whole
            and "    permissions:\n      contents: read\n      actions: read\n" in whole)
    assert "\n  push:\n    branches: [main]\n" in whole
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
    """Stands in for the RunPod client: live endpoints by id, which were read
    and what was sent to change them. A change of the workers is applied, so
    the next read sees it. `refuse` is what RunPod answers to a change it
    does not allow."""

    def __init__(self, live, refuse=None):
        self.live, self.read, self.patched, self.refuse = live, [], [], refuse

    def get(self, endpoint_id):
        self.read.append(endpoint_id)
        return copy.deepcopy(self.live[endpoint_id])

    def patch(self, endpoint_id, body):
        self.patched.append((endpoint_id, body))
        if self.refuse:
            raise SystemExit(f"update of endpoint {endpoint_id} failed (400): {self.refuse}")
        self.live[endpoint_id]["workers"].update(body.get("workers") or {})
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


def test_status_of_a_lane_that_is_no_lane_ends_with_an_error(monkeypatch, capsys):
    """The README has people restore the workers of an idle lane by hand, with
    `--lane`. A mistyped id printed nothing and ended well, as if nothing
    differed and the workers were back."""
    reg = _status_registry()
    client = FakeEndpoints({"ep-main": _live(workers_max=0), "ep-old": _live()})
    for flags in ({}, {"strict": True}, {"restore": True}, {"strict": True, "restore": True}):
        with pytest.raises(SystemExit) as stop:
            runpod_endpoints.status(reg, client, "mian", **flags)
        assert str(stop.value) == "unknown lane 'mian'", flags
    assert client.read == [] and client.patched == [] and capsys.readouterr().out == ""
    # The command, as `create` and `set-image` answer the same mistake.
    monkeypatch.setattr(runpod_endpoints, "load_lanes", lambda: reg)
    monkeypatch.setattr(runpod_endpoints, "Client", lambda: client)
    with pytest.raises(SystemExit, match="^unknown lane 'mian'$"):
        runpod_endpoints.main(["status", "--lane", "mian", "--restore-workers", "--strict"])
    with pytest.raises(SystemExit, match="^unknown lane 'mian'$"):
        runpod_endpoints.main(["set-image", "--lane", "mian", "--tag", "img-abc", "--dry-run"])
    assert client.read == [] and client.patched == []


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
        "disk is None on RunPod, the manifest says 40",
        "gpu.pools is [] on RunPod, the manifest says ['ADA_24']",
        "dataCenterIds is [] on RunPod, the manifest says ['EU-RO-1']",
        "networkVolumes is [] on RunPod, the manifest says ['n0xnkwnogx']"]
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


# -- lanes whose jobs fetch their models ------------------------------------------------
def test_which_lanes_fetch_and_what_a_job_fetches():
    fetches = lanes_mod.fetches_models
    assert fetches({"cadence": "weekly", "model_storage": "job"})
    assert fetches({"cadence": "per-commit", "model_storage": "job", "network_volumes": []})
    for lane in (None, {}, {"cadence": "weekly"}, {"model_storage": "volume"},
                 {"model_storage": None}, {"cadence": "retired", "model_storage": "job"}):
        assert not fetches(lane), lane
    # The committed registry: the weekly lane fetches, and no other lane does.
    assert [lane_id for lane_id, lane in REGISTRY["lanes"].items() if fetches(lane)] == [
        "py313-torch2.14.1-cu130-weekly"]

    def model(name, size, directory="vae"):
        return {"name": name, "directory": directory, "bytes": size}

    entry = {"models": [model("a", 60 * 10**9), model("cat.png", 10**9, "input"),
                        model("b", 1, "loras"), "junk"]}
    assert [m["name"] for m in lanes_mod.fetched_models(entry)] == ["a", "b"]
    assert lanes_mod.fetch_bytes(entry) == 60 * 10**9 + 1
    # 25 MB a second, rounded up: the runner waits that long, and the lint plans with it.
    assert lanes_mod.FETCH_PLAN_BYTES_PER_S == 25_000_000
    assert lanes_mod.fetch_allowance_s(entry) == 2401
    assert lanes_mod.fetch_allowance_s({"models": [model("a", 25_000_000)]}) == 1
    for nothing in (None, {}, {"models": None}, {"models": []},
                    {"models": [model("cat.png", 10**9, "input")]},
                    {"models": [model("a", None), model("b", "9"), model("c", True),
                                model("d", -5)]}):
        assert lanes_mod.fetched_models(nothing) is not None
        assert lanes_mod.fetch_allowance_s(nothing) == 0, nothing


def test_validate_counts_the_download_when_it_asks_whether_a_job_can_ever_start():
    """A run starts no job that could still be running at its deadline. On a
    lane whose jobs fetch their models the download is part of that time."""
    lanes = {"primary": "main", "images": {"img": {
        "python": "3.13", "cuda": "13.0", "torch_index": "i", "torch_spec": "t"}}, "lanes": {
        "main": {"cadence": "per-commit", "role": "primary", "layout": 2, "image": "img",
                 "prefix": "regression/lanes/main", "endpoint_id": "ep"},
        "week": {"cadence": "weekly", "role": "lane", "layout": 2, "image": "img",
                 "prefix": "regression/lanes/week", "endpoint_id": None,
                 "endpoint_name": "long", "timeout_minutes": 120, "model_storage": "job"}}}

    def problems(size, **lane):
        registry = {**lanes, "lanes": {**lanes["lanes"], "week": {**lanes["lanes"]["week"],
                                                                 **lane}}}
        manifest = {"defaults": {"timeout_s": 600}, "workflows": {"big": {
            "enabled": True, "cadence": "weekly",
            "models": [{"name": "m", "directory": "vae", "bytes": size}]}}}
        return lanes_mod.validate(registry, manifest)

    # 95 minutes to the deadline. 600 s and 2,100 s leave 3,000 s for the download.
    assert problems(74 * 10**9) == []            # 2,960 s
    assert problems(75 * 10**9) == [
        "workflow big: it can never start on lane week: its job can take 95 min (timeout_s 600 "
        "and 2100 s for the queue and a cold start, and 3000 s to download its models), and a "
        "run's deadline is 95 min after its start. Raise the lane's timeout_minutes or lower "
        "timeout_s"]
    # The same file on a lane that reads it from the volume takes no time to download.
    assert problems(75 * 10**9, model_storage="volume") == []
    assert problems(75 * 10**9, model_storage=None) == []
    assert problems(75 * 10**9, timeout_minutes=121) == []


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


# -- cadences: nightly and weekly lanes ------------------------------------------------
NIGHTLY = "py313-torch2.14.1-cu130-nightly"
WEEKLY = "py313-torch2.14.1-cu130-weekly"
RETIRED_LANE = {"cadence": "retired", "role": "legacy", "image": "old"}


def _cadence_registry(**week):
    """A retired lane, the per-commit primary, a nightly lane with an endpoint
    and a weekly lane without one (`week` overrides it)."""
    return _registry(old=RETIRED_LANE, main={"role": "primary"},
                     night={"cadence": "nightly", "timeout_minutes": 120},
                     week={"cadence": "weekly", "endpoint_id": None, "timeout_minutes": 350,
                           **week})


CADENCE_MANIFEST = {"workflows": {
    "image": {"enabled": True},
    "pack": {"enabled": True, "lanes": ["main"]},
    "mesh": {"enabled": True, "cadence": "nightly"},
    "music": {"enabled": True, "cadence": "weekly"},
    "off": {"enabled": False, "cadence": "nightly"}}}


def test_the_committed_nightly_and_weekly_lanes():
    """They follow the primary lane's image and share one endpoint, which is
    their own: a nightly or weekly run never takes a worker from a per-commit
    run. The label names the cadence, because the dashboard prints only the
    label of a lane."""
    lanes = REGISTRY["lanes"]
    for lane_id, cadence, minutes in ((NIGHTLY, "nightly", 120), (WEEKLY, "weekly", 350)):
        lane = lanes[lane_id]
        assert {k: v for k, v in lane.items() if k not in ("endpoint_id", "model_storage")} == {
            "label": f"{cadence} · py3.13 / torch 2.14.1 / cu130 / RTX 4090",
            "python": "3.13", "torch": "2.14.1", "cuda": "13.0", "gpu": "RTX 4090",
            "cadence": cadence, "blocking": False, "role": "lane", "layout": 2,
            "prefix": f"regression/lanes/{lane_id}", "image": "cu130-t2141",
            "min_cuda_driver": "13.0", "endpoint_name": "comfyci-cu130-nightly-weekly",
            "workers_max": 3, "disk_gb": 120, "idle_timeout_s": 60,
            "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}"}, "max_in_flight": 2,
            "timeout_minutes": minutes}, lane_id
        assert lanes_mod.is_long_cadence(lane) and "endpoint_id" in lane
        assert lane["image"] == lanes[PRIMARY]["image"]
        assert lane["endpoint_name"] != runpod_endpoints.endpoint_name(lanes[PRIMARY])
        # Never the endpoint of the per-commit lane on the same image.
        assert lane["endpoint_id"] != lanes[PRIMARY]["endpoint_id"]
    assert "model_storage" not in lanes[NIGHTLY] and lanes[WEEKLY]["model_storage"] == "job"
    for key in lanes_mod.ENDPOINT_KEYS:
        assert lanes[NIGHTLY].get(key) == lanes[WEEKLY].get(key), key
    # More workers than jobs at once: a worker whose host is busy cannot start.
    assert lanes[NIGHTLY]["workers_max"] > lanes[NIGHTLY]["max_in_flight"]
    # Neither is a per-commit lane, whatever endpoint it has.
    ready = copy.deepcopy(REGISTRY)
    ready["lanes"][NIGHTLY]["endpoint_id"] = ready["lanes"][WEEKLY]["endpoint_id"] = "ep-nw"
    assert lanes_mod.validate(ready, MANIFEST, PACKS) == []
    assert not {NIGHTLY, WEEKLY} & {r["lane"] for r in lanes_mod.per_commit_matrix(ready)}
    assert lanes_mod.per_commit_matrix(ready, f"{NIGHTLY},{WEEKLY}") == []
    # They add no image to the build.
    assert lanes_mod.image_matrix(ready) == lanes_mod.image_matrix(PENDING)
    # The README's table of lanes names every lane of the registry with its image.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for lane_id, lane in lanes.items():
        assert f"\n| `{lane_id}` | `{lane['image']}`" in readme, lane_id


def test_a_workflow_runs_on_the_lanes_of_its_own_cadence():
    """Each pair of workflow cadence and lane cadence, with and without a
    `lanes` list."""
    why, runs = lanes_mod.why_not_on_lane, lanes_mod.workflow_runs_on_lane
    lanes = {"commit": {"cadence": "per-commit"}, "night": {"cadence": "nightly"},
             "week": {"cadence": "weekly"}}
    for said in (None, "per-commit", "nightly", "weekly"):
        entry = {} if said is None else {"cadence": said}
        own = said or "per-commit"
        for lane_id, lane in lanes.items():
            same = lane["cadence"] == own
            reason = None if same else f"cadence {own}, the lane is {lane['cadence']}"
            # No list, or null: every lane of the cadence.
            assert why(entry, lane_id, lane) == reason
            assert why({**entry, "lanes": None}, lane_id, lane) == reason
            assert runs(entry, lane_id, lane) is same
            # A list that names the lane does not put the workflow on a lane of another cadence.
            assert why({**entry, "lanes": [lane_id, "x"]}, lane_id, lane) == reason
            # A list that does not name it keeps the workflow off, and is said first.
            assert why({**entry, "lanes": ["x", "y"]}, lane_id, lane) == "manifest lanes: x, y"
            assert not runs({**entry, "lanes": ["x", "y"]}, lane_id, lane)
    # A sentence for a mismatch with `lanes: null`, where joining the list would fail.
    assert why({"lanes": None, "cadence": "weekly"}, "commit", lanes["commit"]) == (
        "cadence weekly, the lane is per-commit")
    # A retired lane runs nothing, and its references of every cadence can still be
    # blessed: only the list counts. The same without the lane's entry in the registry.
    for lane in ({"cadence": "retired"}, None):
        for entry in ({}, {"cadence": "nightly"}, {"cadence": "weekly", "lanes": ["old"]}):
            assert why(entry, "old", lane) is None
        assert why({"cadence": "weekly", "lanes": ["x"]}, "old", lane) == "manifest lanes: x"
    # A cadence nobody knows is an error where the cadence decides, not a quiet "no".
    with pytest.raises(ValueError, match="unknown cadence 'daily'"):
        why({"cadence": "daily"}, "commit", lanes["commit"])
    assert why({"cadence": "daily", "lanes": ["x"]}, "commit", lanes["commit"]) == (
        "manifest lanes: x")


def test_the_workflows_of_a_lane_and_the_lanes_of_a_workflow():
    reg, manifest = _cadence_registry(), copy.deepcopy(CADENCE_MANIFEST)
    lanes, entries = reg["lanes"], manifest["workflows"]

    def on(lane_id, **kw):
        return lanes_mod.workflows_for_lane(manifest, lane_id, lane=lanes[lane_id], **kw)

    assert on("main") == ["image", "pack"]
    assert on("night") == ["mesh"] and on("night", enabled_only=False) == ["mesh", "off"]
    assert on("week") == ["music"]
    # A retired lane, and a call without the lane's entry: the list alone, as before.
    assert on("old") == ["image", "mesh", "music"]
    assert lanes_mod.workflows_for_lane(manifest, "night") == ["image", "mesh", "music"]
    # The other way round. A lane without an endpoint counts, a retired lane does not.
    assert lanes_mod.lanes_for_workflow(entries["image"], reg) == ["main"]
    assert lanes_mod.lanes_for_workflow(entries["pack"], reg) == ["main"]
    assert lanes_mod.lanes_for_workflow(entries["mesh"], reg) == ["night"]
    assert lanes_mod.lanes_for_workflow(entries["music"], reg) == ["week"]
    assert lanes_mod.lanes_for_workflow({"lanes": ["old"]}, reg) == []
    assert lanes_mod.lanes_for_workflow({"cadence": "nightly", "lanes": ["main"]}, reg) == []
    assert lanes_mod.lanes_for_workflow({}, {}) == []
    # An unknown cadence names its workflow, as a log line does.
    entries["odd"] = {"enabled": True, "cadence": "daily"}
    with pytest.raises(ValueError, match=r"^\[odd\] unknown cadence 'daily' \(known: "):
        on("main")
    assert "odd" in lanes_mod.workflows_for_lane(manifest, "main")


def test_the_per_commit_matrix_is_what_it_was_before_workflows_had_a_cadence():
    """For today's manifest every row lists what the `lanes` list alone gave
    it. A nightly or weekly workflow is left out of every row."""
    provisioned = copy.deepcopy(REGISTRY)
    provisioned["lanes"][NEW]["endpoint_id"] = "abc123"
    rows = lanes_mod.per_commit_matrix(provisioned, manifest=MANIFEST)
    assert [row["lane"] for row in rows] == [PRIMARY, NEW]
    for row in rows:
        assert list(row) == ["lane", "prefix", "endpoint_id", "workflows"]
        by_list = [wf_id for wf_id, entry in MANIFEST["workflows"].items()
                   if entry.get("enabled") and (entry.get("lanes") is None
                                                or row["lane"] in entry["lanes"])
                   and lanes_mod.cadence_of(entry) == "per-commit"]
        assert row["workflows"] == ",".join(by_list) and by_list, row["lane"]
    # An entry that names no cadence runs where it ran before.
    for wf_id, entry in MANIFEST["workflows"].items():
        if "cadence" not in entry:
            for lane_id in (PRIMARY, NEW):
                assert lanes_mod.workflow_runs_on_lane(
                    entry, lane_id, REGISTRY["lanes"][lane_id]) is (
                    entry.get("lanes") is None or lane_id in entry["lanes"]), wf_id
    reg = _cadence_registry()
    assert lanes_mod.per_commit_matrix(reg, manifest=CADENCE_MANIFEST) == [
        {"lane": "main", "prefix": "regression/lanes/main", "endpoint_id": "ep-main",
         "workflows": "image,pack"}]


def test_the_matrix_of_a_nightly_or_weekly_run():
    """The lanes of the cadence that have an endpoint and an enabled workflow.
    With none the matrix is empty and the run ends after its plan, so a lane
    can be declared before its endpoint exists."""
    reg, manifest = _cadence_registry(), copy.deepcopy(CADENCE_MANIFEST)
    night = {"lane": "night", "prefix": "regression/lanes/night", "endpoint_id": "ep-night",
             "timeout_minutes": 120, "workflows": "mesh"}
    rows = lanes_mod.cadence_matrix(reg, "nightly", manifest=manifest)
    assert rows == [night] and list(rows[0]) == list(night)
    assert lanes_mod.cadence_matrix(reg, "nightly", "night,main", manifest) == [night]
    assert lanes_mod.cadence_matrix(reg, "nightly", "main", manifest) == []
    # Without an endpoint: left out. With one: its own row and its own time limit.
    assert lanes_mod.cadence_matrix(reg, "weekly", manifest=manifest) == []
    assert lanes_mod.cadence_left_out(reg, "weekly", None, manifest) == [
        "week: no endpoint_id yet"]
    assert lanes_mod.cadence_matrix(_cadence_registry(endpoint_id="ep-night"), "weekly",
                                    manifest=manifest) == [
        {"lane": "week", "prefix": "regression/lanes/week", "endpoint_id": "ep-night",
         "timeout_minutes": 350, "workflows": "music"}]
    # With no enabled workflow: left out, and the plan says so.
    manifest["workflows"]["mesh"]["enabled"] = False
    assert lanes_mod.cadence_matrix(reg, "nightly", manifest=manifest) == []
    assert lanes_mod.cadence_left_out(reg, "nightly", None, manifest) == [
        "night: no enabled workflow with the nightly cadence"]
    # Unless the dispatch names a workflow that runs on the lane: the first look
    # at one that is still switched off. The runner runs a workflow that is named.
    for named in ("mesh", "off", "ghost,off"):
        assert lanes_mod.cadence_matrix(reg, "nightly", manifest=manifest, named=named) == [
            {**night, "workflows": ""}], named
        assert lanes_mod.cadence_left_out(reg, "nightly", None, manifest, named) == []
    assert lanes_mod.named_on_lane(manifest, "music,off,mesh", "night",
                                   reg["lanes"]["night"]) == ["mesh", "off"]
    # Only the entries that are named are read: another one's mistake is not this run's.
    odd = {"workflows": {**manifest["workflows"], "odd": {"enabled": False, "cadence": "daily"}}}
    assert lanes_mod.named_on_lane(odd, "off", "night", reg["lanes"]["night"]) == ["off"]
    with pytest.raises(ValueError, match=r"^\[odd\] unknown cadence 'daily'"):
        lanes_mod.named_on_lane(odd, "odd", "night", reg["lanes"]["night"])
    # A workflow of another cadence or lane does not, nor does a run that names none.
    for named in ("music", "image", "ghost"):
        assert lanes_mod.cadence_matrix(reg, "nightly", manifest=manifest, named=named) == []
        assert lanes_mod.cadence_left_out(reg, "nightly", None, manifest, named) == [
            "night: no enabled workflow with the nightly cadence, and the run names none that "
            "runs on the lane"]
    for named in ("all", "", None):
        assert lanes_mod.cadence_matrix(reg, "nightly", manifest=manifest, named=named) == []
        assert lanes_mod.cadence_left_out(reg, "nightly", None, manifest, named) == [
            "night: no enabled workflow with the nightly cadence"]
    # Without the manifest the lanes are listed as they are.
    assert lanes_mod.cadence_matrix(reg, "nightly") == [
        {k: v for k, v in night.items() if k != "workflows"}]
    # A lane that names no time limit gets GitHub's own.
    del reg["lanes"]["night"]["timeout_minutes"]
    assert lanes_mod.cadence_matrix(reg, "nightly")[0]["timeout_minutes"] == 360
    # Why a named lane is not in the plan.
    assert lanes_mod.cadence_left_out(reg, "nightly", "main,ghost,week,old") == [
        "ghost: not a lane in manifest/lanes.json", "main: cadence 'per-commit', not nightly",
        "old: cadence 'retired', not nightly", "week: cadence 'weekly', not nightly"]
    assert lanes_mod.cadence_left_out(reg, "nightly", None, CADENCE_MANIFEST) == []
    # Only the two long cadences have such a matrix.
    for cadence in ("per-commit", "retired", "daily"):
        with pytest.raises(ValueError, match="unknown cadence"):
            lanes_mod.cadence_matrix(reg, cadence)


def test_the_cadence_command_prints_the_matrix_and_why_a_lane_is_left_out(monkeypatch, capsys,
                                                                         tmp_path):
    reg = _cadence_registry()
    manifest = tmp_path / "workflows.json"
    manifest.write_text(json.dumps(CADENCE_MANIFEST), encoding="utf-8")
    monkeypatch.setattr(lanes_mod, "load_lanes", lambda path=None: reg)
    monkeypatch.setattr(lanes_mod, "MANIFEST_FILE", manifest)
    assert lanes_mod.main(["cadence", "nightly"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == [
        {"lane": "night", "prefix": "regression/lanes/night", "endpoint_id": "ep-night",
         "timeout_minutes": 120, "workflows": "mesh"}]
    assert out.out.count("\n") == 1 and out.err == ""
    assert lanes_mod.main(["cadence", "weekly"]) == 0
    out = capsys.readouterr()
    assert out.out == "[]\n"
    assert out.err == "::notice::not tested: week: no endpoint_id yet\n"
    assert lanes_mod.main(["cadence", "weekly", "--only", "night"]) == 0
    out = capsys.readouterr()
    assert out.out == "[]\n" and "not tested: night: cadence 'nightly', not weekly" in out.err
    # --only names the lanes to plan: a lane of the cadence that is not named stays out.
    assert lanes_mod.main(["cadence", "nightly", "--only", "main,night"]) == 0
    assert [row["lane"] for row in json.loads(capsys.readouterr().out)] == ["night"]
    assert lanes_mod.main(["cadence", "nightly", "--only", "main"]) == 0
    out = capsys.readouterr()
    assert out.out == "[]\n"
    assert out.err == "::notice::not tested: main: cadence 'per-commit', not nightly\n"
    with pytest.raises(SystemExit):
        lanes_mod.main(["cadence", "per-commit"])
    capsys.readouterr()
    # The `workflows` input of a dispatch: "all" changes nothing, and a workflow
    # that is switched off keeps its lane in the plan when it is named.
    assert lanes_mod.main(["cadence", "nightly", "--workflows", "all"]) == 0
    assert [row["workflows"] for row in json.loads(capsys.readouterr().out)] == ["mesh"]
    off = copy.deepcopy(CADENCE_MANIFEST)
    off["workflows"]["mesh"]["enabled"] = False
    manifest.write_text(json.dumps(off), encoding="utf-8")
    assert lanes_mod.main(["cadence", "nightly", "--workflows", "all"]) == 0
    out = capsys.readouterr()
    assert out.out == "[]\n"
    assert out.err == "::notice::not tested: night: no enabled workflow with the nightly cadence\n"
    assert lanes_mod.main(["cadence", "nightly", "--workflows", "mesh"]) == 0
    out = capsys.readouterr()
    assert [row["lane"] for row in json.loads(out.out)] == ["night"] and out.err == ""
    # The committed files: every row has what nightly-weekly.yml reads from it.
    monkeypatch.undo()
    for cadence in lanes_mod.LONG_CADENCES:
        assert lanes_mod.main(["cadence", cadence]) == 0
        for row in json.loads(capsys.readouterr().out):
            assert set(row) == {"lane", "prefix", "endpoint_id", "timeout_minutes", "workflows"}
            assert REGISTRY["lanes"][row["lane"]]["cadence"] == cadence and row["workflows"]


def test_validate_names_a_workflow_that_its_cadence_and_lanes_leave_on_no_lane():
    """A pack workflow names its lanes. Give it another cadence and no lane
    takes it: the per-commit lanes drop it for its cadence, the nightly lane
    for its list. No run would turn red for a workflow that never runs."""
    reg = _cadence_registry()

    def problems(**entry):
        return lanes_mod.validate(reg, {"workflows": {"wf": {"enabled": True, **entry}}})

    assert problems(lanes=["main"], cadence="nightly") == [
        "workflow wf: cadence 'nightly' and lanes [\"main\"] leave it on no lane: add a nightly "
        "lane to its lanes list"]
    assert problems(lanes=["main", "night"], cadence="nightly") == []
    assert problems(lanes=["main"]) == [] and problems() == [] and problems(cadence="weekly") == []
    # Only a retired lane is no lane.
    assert problems(lanes=["old"]) == [
        "workflow wf: cadence 'per-commit' and lanes [\"old\"] leave it on no lane: add a "
        "per-commit lane to its lanes list"]
    # No list, and no lane of the cadence.
    del reg["lanes"]["week"]
    assert problems(cadence="weekly") == [
        "workflow wf: cadence 'weekly' and lanes null leave it on no lane: manifest/lanes.json "
        "has no weekly lane that is not retired"]
    # A workflow that is switched off is held to nothing, and a list that is no list
    # or a cadence nobody knows is said once.
    assert lanes_mod.validate(reg, {"workflows": {"wf": {"enabled": False, "cadence": "weekly"}}}) \
        == []
    assert problems(lanes=[]) == ["workflow wf: lanes must be null or a non-empty list"]
    assert problems(cadence="daily") == [
        "workflow wf: unknown cadence 'daily' (known: per-commit, nightly, weekly)"]
    # The committed files: the same mistake on a real pack workflow.
    manifest = copy.deepcopy(MANIFEST)
    pack = next(wf_id for wf_id, e in MANIFEST["workflows"].items()
                if e.get("enabled") and e.get("custom_nodes"))
    manifest["workflows"][pack]["cadence"] = "nightly"
    said = lanes_mod.validate(REGISTRY, manifest, PACKS)
    assert len(said) == 1 and said[0].startswith(
        f"workflow {pack}: cadence 'nightly' and lanes [\"{PRIMARY}\", ")
    assert said[0].endswith("leave it on no lane: add a nightly lane to its lanes list")
    manifest["workflows"][pack]["lanes"].append(NIGHTLY)
    assert lanes_mod.validate(REGISTRY, manifest, PACKS) == []


SHARED = {"endpoint_name": "comfyci-nw", "workers_max": 3, "disk_gb": 120, "idle_timeout_s": 60,
          "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}"}}


def _shared_registry():
    """A nightly and a weekly lane on one endpoint. Only the nightly has its id yet."""
    return _registry(old=RETIRED_LANE, main={"role": "primary"},
                     night={"cadence": "nightly", **SHARED, "endpoint_id": "ep-nw"},
                     week={"cadence": "weekly", **SHARED, "endpoint_id": None})


def test_validate_holds_lanes_that_share_an_endpoint_to_the_same_settings():
    """An endpoint runs one image with one number of workers, one disk, one
    idle time and one environment, whatever number of lanes use it."""
    ok = _shared_registry()
    assert lanes_mod.validate(ok) == []
    ok["lanes"]["week"]["endpoint_id"] = "ep-nw"  # the weekly lane gets the id later
    assert lanes_mod.validate(ok) == []
    for key, value in (("workers_max", 2), ("disk_gb", 80), ("idle_timeout_s", 20),
                       ("env", {}), ("image", "old"), ("endpoint_name", "comfyci-other")):
        bad = copy.deepcopy(ok)
        bad["lanes"]["week"][key] = value
        assert lanes_mod.validate(bad) == [
            f"lane week: {key} is {value!r}, but night, which shares its endpoint, says "
            f"{ok['lanes']['night'][key]!r}"], key
    # What a lane leaves out is the default, so a default that only one of them
    # writes out is no difference. Any other value that only one names is one.
    same = copy.deepcopy(ok)
    del same["lanes"]["week"]["workers_max"]  # 3, the default
    assert lanes_mod.validate(same) == []
    bad = copy.deepcopy(ok)
    del bad["lanes"]["week"]["disk_gb"]
    assert lanes_mod.validate(bad) == [
        "lane week: disk_gb is 40, but night, which shares its endpoint, says 120"]
    # The defaults of the registry come first, as they do for `create`.
    own = copy.deepcopy(ok)
    own["endpoint_defaults"] = {"workers": {"min": 0, "max": 5, "idleTimeout": 60},
                                "disk_gb": 120, "env": dict(SHARED["env"])}
    for key in ("disk_gb", "idle_timeout_s", "env"):
        del own["lanes"]["week"][key]
    assert lanes_mod.validate(own) == []
    del own["lanes"]["week"]["workers_max"]
    assert lanes_mod.validate(own) == [
        "lane week: workers_max is 5, but night, which shares its endpoint, says 3"]
    # One name cannot stand for two endpoints.
    bad = copy.deepcopy(ok)
    bad["lanes"]["week"]["endpoint_id"] = "ep-other"
    assert lanes_mod.validate(bad) == [
        "lane week: endpoint_id is 'ep-other', but night, which has the same endpoint_name, "
        "says 'ep-nw'"]
    # Lanes that share by the id alone are held to it as well.
    by_id = _registry(old=RETIRED_LANE, main={"role": "primary"},
                      night={"cadence": "nightly", "endpoint_id": "ep-main", "workers_max": 2})
    assert lanes_mod.validate(by_id) == [
        "lane night: workers_max is 2, but main, which shares its endpoint, says 3"]
    by_id["lanes"]["night"].update({"workers_max": 3, "endpoint_name": "comfyci-img"})
    assert lanes_mod.validate(by_id) == []  # the name its image gives it anyway
    # A retired lane runs nothing: its settings are held to nothing.
    retired = copy.deepcopy(ok)
    retired["lanes"]["old"].update({**SHARED, "workers_max": 1, "endpoint_id": "ep-nw"})
    assert lanes_mod.validate(retired) == []
    # Lanes with their own endpoints may differ in everything.
    apart = copy.deepcopy(ok)
    apart["lanes"]["week"].update({"endpoint_name": "comfyci-w", "workers_max": 1,
                                   "endpoint_id": "ep-w"})
    assert lanes_mod.validate(apart) == []


def test_validate_checks_the_optional_keys_of_a_lane():
    ok = _registry(old=RETIRED_LANE, main={
        "role": "primary", "workers_max": 2, "disk_gb": 40, "idle_timeout_s": 20,
        "max_in_flight": 2, "timeout_minutes": 120, "model_storage": "volume",
        "env": {"A": "b"}, "endpoint_name": "comfyci-main"})
    assert lanes_mod.validate(ok) == []

    def problems(key, value):
        bad = copy.deepcopy(ok)
        bad["lanes"]["main"][key] = value
        return lanes_mod.validate(bad)

    assert lanes_mod.LANE_COUNTS == ("workers_max", "disk_gb", "idle_timeout_s", "max_in_flight",
                                     "timeout_minutes")
    for key in lanes_mod.LANE_COUNTS:
        for value in (0, -1, 2.5, "3", True, [3]):
            assert problems(key, value) == [
                f"lane main: {key} must be a whole number above 0, not {value!r}"], (key, value)
        assert problems(key, None) == [] and problems(key, 1) == []
    assert problems("model_storage", "job") == [] and problems("model_storage", None) == []
    for value in ("disk", "", 1, True, ["job"]):
        assert problems("model_storage", value) == [
            f"lane main: model_storage must be one of volume, job, not {value!r}"]
    assert problems("env", {}) == [] and problems("env", None) == []
    for value in ({"A": 1}, {"A": None}, {1: "b"}, ["A=b"], "A=b"):
        assert problems("env", value) == ["lane main: env must map variable names to strings"]
    for value in ("", 3, ["n"]):
        assert problems("endpoint_name", value) == [
            f"lane main: endpoint_name must be a name, not {value!r}"]


def test_validate_fills_in_the_defaults_that_create_sends():
    """What `validate` compares for a lane is what `create` asks RunPod for,
    default by default: the two read the same numbers and the same name."""
    assert runpod_endpoints.ENDPOINT_NAME is lanes_mod.ENDPOINT_NAME
    for reg in (REGISTRY, _shared_registry()):
        for lane_id, lane in reg["lanes"].items():
            asked = lanes_mod.endpoint_settings_of(reg, lane)
            assert list(asked) == list(lanes_mod.ENDPOINT_KEYS)
            workers, disk = runpod_endpoints.endpoint_settings(reg, lane)
            assert (asked["workers_max"], asked["idle_timeout_s"], asked["disk_gb"]) == (
                workers["max"], workers["idleTimeout"], disk), lane_id
            assert asked["endpoint_name"] == runpod_endpoints.endpoint_name(lane), lane_id
    blank = copy.deepcopy(REGISTRY)
    for lane in blank["lanes"].values():
        lane["endpoint_id"] = None
    for lane_id in (PRIMARY, NIGHTLY):
        lane = blank["lanes"][lane_id]
        asked = lanes_mod.endpoint_settings_of(blank, lane)
        body = runpod_endpoints.endpoint_body(blank, lane_id, f"{lane['image']}-abc")
        assert (body["name"], body["env"], body["gpu"]["pools"], body["dataCenterIds"],
                body["networkVolumes"]) == (
            asked["endpoint_name"], asked["env"], asked["gpu_pools"], asked["data_center_ids"],
            asked["network_volumes"]), lane_id
    # An environment that is no object is validate's finding, not a crash here.
    odd = lanes_mod.endpoint_settings_of({"endpoint_defaults": {"env": {"A": "b"}}},
                                         {"env": "A=c"})
    assert odd["env"] == {"A": "b"}


def test_validate_refuses_a_lane_and_a_workflow_that_leave_no_time_for_a_job():
    """A nightly or weekly run stops submitting 25 minutes before the lane's
    time limit, and starts no job whose worst case passes that deadline. A
    lane with no time left, and a workflow whose job never fits, would be a
    red run every night that nothing explains before it happens."""
    assert (lanes_mod.DEADLINE_MARGIN_MIN, lanes_mod.LONG_JOB_ALLOWANCE_S) == (25, 2100)
    reg = _cadence_registry()
    assert lanes_mod.deadline_window_s(reg["lanes"]["night"]) == 95 * 60
    assert lanes_mod.deadline_window_s({}) == (lanes_mod.DEFAULT_TIMEOUT_MINUTES - 25) * 60

    def lane_problems(lane_id, minutes):
        bad = copy.deepcopy(reg)
        bad["lanes"][lane_id]["timeout_minutes"] = minutes
        return lanes_mod.validate(bad)

    for minutes in (1, 24, 25):
        assert lane_problems("night", minutes) == [
            f"lane night: timeout_minutes {minutes} leaves no time for a job: a run's deadline "
            "is 25 minutes before it"]
        assert lane_problems("week", minutes) == [
            f"lane week: timeout_minutes {minutes} leaves no time for a job: a run's deadline "
            "is 25 minutes before it"]
    assert lane_problems("night", 26) == [] and lane_problems("night", None) == []
    # A per-commit lane has no deadline, and a retired lane runs nothing.
    assert lane_problems("main", 1) == [] and lane_problems("old", 1) == []

    def problems(defaults=600, registry=reg, **entry):
        manifest = {"defaults": {"timeout_s": defaults},
                    "workflows": {"wf": {"enabled": True, "cadence": "nightly", **entry}}}
        return lanes_mod.validate(registry, manifest)

    never = ("workflow wf: it can never start on lane {}: its job can take {} min (timeout_s {} "
             "and 2100 s for the queue and a cold start), and a run's deadline is {} min after "
             "its start. Raise the lane's timeout_minutes or lower timeout_s")
    # The nightly lane: 120 minutes, so the deadline is 95 minutes after the start.
    assert problems() == [] and problems(timeout_s=3599) == []
    assert problems(timeout_s=3600) == [never.format("night", 95, 3600, 95)]
    assert problems(timeout_s=3601) == [never.format("night", 96, 3601, 95)]
    assert problems(defaults=7200) == [never.format("night", 155, 7200, 95)]
    assert problems(defaults=7200, timeout_s=900) == []
    # The weekly lane, which has no endpoint yet: 350 minutes.
    assert problems(cadence="weekly", timeout_s=17399) == []
    assert problems(cadence="weekly", timeout_s=17400) == [never.format("week", 325, 17400, 325)]
    # A workflow that is switched off, one of the per-commit lane, and a manifest
    # that names no time are held to nothing.
    assert problems(timeout_s=3600, enabled=False) == []
    assert problems(timeout_s=99999, cadence="per-commit") == []
    assert lanes_mod.validate(reg, {"workflows": {"wf": {"enabled": True,
                                                         "cadence": "nightly"}}}) == []
    # A lane that names no limit has GitHub's 360 minutes.
    free = copy.deepcopy(reg)
    del free["lanes"]["night"]["timeout_minutes"]
    assert problems(registry=free, timeout_s=17999) == []
    assert problems(registry=free, timeout_s=18000) == [never.format("night", 335, 18000, 335)]
    # A lane without any time is said once, for the lane, not for each workflow.
    short = copy.deepcopy(reg)
    short["lanes"]["night"]["timeout_minutes"] = 20
    assert problems(registry=short) == [
        "lane night: timeout_minutes 20 leaves no time for a job: a run's deadline is 25 "
        "minutes before it"]
    short["lanes"]["night"]["timeout_minutes"] = 60  # 35 minutes: under the default job's 45
    assert problems(registry=short) == [never.format("night", 45, 600, 35)]
    # The committed files: every workflow fits its lanes, and one that would not is named.
    manifest = copy.deepcopy(MANIFEST)
    assert manifest["defaults"]["timeout_s"] + 2100 < lanes_mod.deadline_window_s(
        REGISTRY["lanes"][NIGHTLY])
    wf_id = next(iter(manifest["workflows"]))
    manifest["workflows"][wf_id].update({"enabled": True, "cadence": "nightly", "lanes": None,
                                         "timeout_s": 3600})
    manifest["workflows"][wf_id].pop("custom_nodes", None)
    assert lanes_mod.validate(REGISTRY, manifest, PACKS) == [
        never.format(NIGHTLY, 95, 3600, 95).replace("workflow wf", f"workflow {wf_id}")]


def test_the_endpoint_of_a_lane_is_created_from_the_lanes_own_settings():
    """The nightly and weekly endpoint: its own name, a larger disk, a longer
    idle time and one more variable. `status` holds the live endpoint to the
    same numbers."""
    reg = copy.deepcopy(REGISTRY)
    reg["lanes"][NIGHTLY]["endpoint_id"] = reg["lanes"][WEEKLY]["endpoint_id"] = None
    body = runpod_endpoints.endpoint_body(reg, NIGHTLY, "cu130-t2141-abc")
    assert body["name"] == "comfyci-cu130-nightly-weekly"
    assert body["image"] == "ghcr.io/comfy-org/comfyci-runpod-worker:cu130-t2141-abc"
    assert body["workers"] == {"min": 0, "max": 3, "idleTimeout": 60} and body["disk"] == 120
    assert body["env"] == {"RUNPOD_INIT_TIMEOUT": "800",
                           "HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}"}
    assert body["gpu"] == {"pools": ["ADA_24"], "count": 1, "minCudaVersion": "13.0"}
    # Everything else is what every endpoint gets, and the defaults stay as they were.
    plain = runpod_endpoints.endpoint_body(PENDING, NEW, "cu132-t2150-abc")
    for key in ("type", "registry", "scaling", "timeout", "flashboot", "networkVolumes",
                "dataCenterIds"):
        assert body[key] == plain[key], key
    assert reg["endpoint_defaults"] == REGISTRY["endpoint_defaults"]
    assert plain["env"] == {"RUNPOD_INIT_TIMEOUT": "800"} and plain["disk"] == 40
    # A variable of the lane wins over a default of the same name.
    reg["lanes"][NIGHTLY]["env"] = {"RUNPOD_INIT_TIMEOUT": "900"}
    assert runpod_endpoints.endpoint_body(reg, NIGHTLY, "cu130-t2141")["env"] == {
        "RUNPOD_INIT_TIMEOUT": "900"}
    assert runpod_endpoints.endpoint_settings(reg, reg["lanes"][NIGHTLY]) == (body["workers"], 120)
    assert runpod_endpoints.differences(reg, NIGHTLY, _live(idle=60, disk=120)) == []
    assert runpod_endpoints.differences(reg, NIGHTLY, _live()) == [
        "workers.idleTimeout is 20 on RunPod, the manifest says 60",
        "disk is 40 on RunPod, the manifest says 120"]
    # One endpoint is created once: the weekly lane gets the id by hand.
    reg["lanes"][NIGHTLY]["endpoint_id"] = "ep-nw"
    with pytest.raises(SystemExit, match="shares the endpoint 'comfyci-cu130-nightly-weekly' "
                                         f"with '{NIGHTLY}', which already has ep-nw"):
        runpod_endpoints.endpoint_body(reg, WEEKLY, "cu130-t2141-abc")
    # A retired lane that keeps the name and an id stands in nobody's way: it
    # runs nothing, and its endpoint is not one to write into another lane.
    reg["lanes"][NIGHTLY].update({"cadence": "retired", "endpoint_id": "ep-old"})
    assert runpod_endpoints.endpoint_body(reg, WEEKLY, "cu130-t2141-abc")["name"] == (
        "comfyci-cu130-nightly-weekly")


def test_set_image_sends_one_patch_to_an_endpoint_that_lanes_share(capsys):
    reg = copy.deepcopy(REGISTRY)
    reg["lanes"][NIGHTLY]["endpoint_id"] = reg["lanes"][WEEKLY]["endpoint_id"] = "ep-nw"
    main = reg["lanes"][PRIMARY]["endpoint_id"]
    assert lanes_mod.lanes_on_image(reg, "cu130-t2141") == [PRIMARY, NIGHTLY, WEEKLY]
    image = {"image": "ghcr.io/comfy-org/comfyci-runpod-worker:cu130-t2141-abc"}
    plan = runpod_endpoints.plan_set_image(reg, "cu130-t2141-abc", image_id="cu130-t2141")
    assert plan == [(PRIMARY, main, image), (NIGHTLY, "ep-nw", image)]
    assert f"[{WEEKLY}] shares endpoint ep-nw with {NIGHTLY}; repointed once\n" in (
        capsys.readouterr().out)
    # A lane asked for by name is repointed, shared endpoint or not.
    assert runpod_endpoints.plan_set_image(reg, "cu130-t2141-abc", lane_id=WEEKLY) == [
        (WEEKLY, "ep-nw", image)]
    client = FakeEndpoints({main: _live(), "ep-nw": _live()})
    assert runpod_endpoints.set_image(reg, "cu130-t2141-abc", None, "cu130-t2141", client,
                                      False) == 0
    assert client.patched == [(main, image), ("ep-nw", image)]


def test_status_restores_the_workers_runpod_took_from_an_idle_endpoint(capsys):
    """RunPod lowers the largest number of workers of an endpoint without
    requests, to 2 after some days and to 0 in the end. The jobs of the next
    run would then only wait in the queue."""
    reg = _status_registry()
    client = FakeEndpoints({"ep-main": _live(workers_max=2), "ep-old": _live(workers_max=0)})
    # Without the flag the command only reads.
    assert runpod_endpoints.status(reg, client, "main") == 0
    assert "main: differs: workers.max is 2 on RunPod, the manifest says 3\n" in (
        capsys.readouterr().out)
    assert client.patched == []
    # With it: one PATCH, which names the two numbers it must not change.
    assert runpod_endpoints.status(reg, client, "main", strict=True, restore=True) == 0
    assert client.patched == [("ep-main", {"workers": {"min": 0, "idleTimeout": 20, "max": 3}})]
    out = capsys.readouterr().out
    assert "main: workers.max was 2 on RunPod, raised to 3\n" in out
    assert "workers=0-3 idle=20s" in out and "differ" not in out
    # The endpoint is as the manifest says now: the next run sends nothing.
    assert runpod_endpoints.status(reg, client, None, strict=True, restore=True) == 0
    assert len(client.patched) == 1 and client.live["ep-old"]["workers"]["max"] == 0
    # From 0, and to the lane's own number.
    reg["lanes"]["main"]["workers_max"] = 2
    for have in (0, 1):
        client = FakeEndpoints({"ep-main": _live(workers_max=have)})
        assert runpod_endpoints.status(reg, client, "main", restore=True) == 0
        assert client.patched == [("ep-main", {"workers": {"min": 0, "idleTimeout": 20,
                                                           "max": 2}})]
    # Never downwards, never for a retired lane, and nothing else is changed:
    # those stay differences that the command reports.
    client = FakeEndpoints({"ep-main": _live(workers_max=5, idle=5, disk=30),
                            "ep-old": _live(workers_max=3)})
    capsys.readouterr()
    assert runpod_endpoints.status(reg, client, None, strict=True, restore=True) == 1
    assert client.patched == []
    out = capsys.readouterr().out
    assert "main: differs: workers.max is 5 on RunPod, the manifest says 2\n" in out
    assert "old: differs: workers.max is 3 on RunPod, the manifest says 0" in out
    assert "4 of the settings above differ" in out
    # An answer without the number, or with something that is no number, sends
    # nothing either and raises nothing.
    for value in (None, "1", True, 1.5):
        odd = _live(workers_max=value)
        if value is None:
            del odd["workers"]["max"]
        client = FakeEndpoints({"ep-main": odd})
        assert runpod_endpoints.status(reg, client, "main", restore=True) == 0
        assert client.patched == [], value


def test_a_refused_restore_ends_the_step_with_runpods_message(monkeypatch, capsys):
    """The account has a worker quota. A raise it refuses must stop the run
    before it submits jobs that no worker would take."""
    reg, real_client = _status_registry(), runpod_endpoints.Client
    client = FakeEndpoints({"ep-main": _live(workers_max=0)},
                           refuse='{"error": "max workers quota exceeded"}')
    monkeypatch.setattr(runpod_endpoints, "load_lanes", lambda: reg)
    monkeypatch.setattr(runpod_endpoints, "Client", lambda: client)
    with pytest.raises(SystemExit, match=r"update of endpoint ep-main failed \(400\): .*quota"):
        runpod_endpoints.main(["status", "--lane", "main", "--restore-workers"])
    assert len(client.patched) == 1 and "raised" not in capsys.readouterr().out
    # The flag belongs to the command, and without it nothing is sent.
    assert runpod_endpoints.main(["status", "--lane", "main"]) == 0
    assert len(client.patched) == 1

    # The real client quotes RunPod's answer, for a repoint as for a restore.
    class Refused:
        status_code, text = 400, '{"error": "max workers quota exceeded"}'

    monkeypatch.setattr(runpod_endpoints.requests, "patch", lambda *a, **kw: Refused())
    with pytest.raises(SystemExit) as stop:
        real_client("key", "https://runpod.invalid").patch("ep-1", {"workers": {}})
    assert str(stop.value) == ('update of endpoint ep-1 failed (400): {"error": "max workers '
                               'quota exceeded"}')


def _create_request(name, tag, cuda, idle=20, disk=40, env=None):
    """The create request of one of today's lanes, key by key in the order it
    is sent: the 24 GB pool, the one data center and the model volume."""
    return {"name": name, "type": "QUEUE",
            "image": f"ghcr.io/comfy-org/comfyci-runpod-worker:{tag}",
            "registry": "cmszcugyp001tfk7ooe7rtp4d",
            "gpu": {"pools": ["ADA_24"], "count": 1, "minCudaVersion": cuda},
            "workers": {"min": 0, "max": 3, "idleTimeout": idle},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
            "timeout": 3600000, "flashboot": "FLASHBOOT",
            "networkVolumes": ["n0xnkwnogx"], "dataCenterIds": ["EU-RO-1"], "disk": disk,
            "env": {"RUNPOD_INIT_TIMEOUT": "800", **(env or {})}}


def test_the_create_requests_of_todays_lanes_are_what_they_were():
    """No lane of today names GPU pools, data centers or volumes of its own,
    so each request is the one it was before a lane could. The text is
    compared, so the order of the keys counts too."""
    blank = copy.deepcopy(REGISTRY)
    for lane in blank["lanes"].values():
        lane["endpoint_id"] = None
    shared = _create_request("comfyci-cu130-nightly-weekly", "cu130-t2141-abc", "13.0", idle=60,
                             disk=120, env={"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}"})
    wanted = {PRIMARY: _create_request("comfyci-cu130-t2141", "cu130-t2141-abc", "13.0"),
              NEW: _create_request("comfyci-cu132-t2150", "cu132-t2150-abc", "13.2"),
              NIGHTLY: shared, WEEKLY: shared}
    for lane_id, want in wanted.items():
        tag = f"{REGISTRY['lanes'][lane_id]['image']}-abc"
        body = runpod_endpoints.endpoint_body(blank, lane_id, tag)
        assert json.dumps(body, indent=2) == json.dumps(want, indent=2), lane_id


# A weekly lane for models that need more than 24 GB: the 96 GB pool, in the
# data centers that have it, and no model volume.
OTHER_CARD = {"gpu": "RTX PRO 6000", "gpu_pools": ["BLACKWELL_96"],
              "data_center_ids": ["US-NC-1", "US-CA-2"], "network_volumes": [],
              "model_storage": "job"}


def test_a_lane_may_name_its_own_gpu_pools_data_centers_and_volumes():
    plain = runpod_endpoints.endpoint_body(PENDING, NEW, "cu132-t2150-abc")
    reg = copy.deepcopy(PENDING)
    reg["lanes"][NEW].update(OTHER_CARD)
    assert lanes_mod.validate(reg, MANIFEST, PACKS) == []
    body = runpod_endpoints.endpoint_body(reg, NEW, "cu132-t2150-abc")
    assert body["gpu"] == {"pools": ["BLACKWELL_96"], "count": 1, "minCudaVersion": "13.2"}
    assert body["dataCenterIds"] == ["US-NC-1", "US-CA-2"]
    assert body["networkVolumes"] == []  # an empty list: an endpoint without a volume
    # Nothing else of the request changes, and the defaults stay as they were.
    assert list(body) == list(plain)
    assert {k for k in body if body[k] != plain[k]} == {"gpu", "dataCenterIds", "networkVolumes"}
    assert reg["endpoint_defaults"] == REGISTRY["endpoint_defaults"]
    # The `gpu` key is the label people read. It selects nothing.
    reg["lanes"][NEW]["gpu"] = "RTX 4090"
    assert runpod_endpoints.endpoint_body(reg, NEW, "cu132-t2150-abc") == body
    # Each key replaces its own default and no other, with a list of its own.
    for key, value, sent in (("gpu_pools", ["ADA_48_PRO", "AMPERE_48"], "gpu"),
                             ("data_center_ids", ["US-IL-1"], "dataCenterIds"),
                             ("network_volumes", ["vol2"], "networkVolumes")):
        one = copy.deepcopy(PENDING)
        one["lanes"][NEW][key] = value
        assert lanes_mod.placement_of(one, one["lanes"][NEW], key) == value
        assert lanes_mod.placement_of(one, one["lanes"][PRIMARY], key) == (
            REGISTRY["endpoint_defaults"][key])
        body = runpod_endpoints.endpoint_body(one, NEW, "cu132-t2150-abc")
        assert {k for k in body if body[k] != plain[k]} == {sent}, key
        got = body["gpu"]["pools"] if key == "gpu_pools" else body[sent]
        assert got == value and got is not value
        # null is the same as absent: the default.
        one["lanes"][NEW][key] = None
        assert runpod_endpoints.endpoint_body(one, NEW, "cu132-t2150-abc") == plain
    assert lanes_mod.placement_of({}, None, "gpu_pools") is None
    # A list that neither the lane nor the defaults name is refused.
    bare = copy.deepcopy(PENDING)
    del bare["endpoint_defaults"]["gpu_pools"], bare["endpoint_defaults"]["network_volumes"]
    with pytest.raises(SystemExit) as stop:
        runpod_endpoints.endpoint_body(bare, NEW, "cu132-t2150-abc")
    assert str(stop.value) == (f"lane {NEW!r} has no gpu_pools, network_volumes: neither the "
                               "lane nor endpoint_defaults names one")
    bare["lanes"][NEW].update({"gpu_pools": ["ADA_24"], "network_volumes": []})
    assert runpod_endpoints.endpoint_body(bare, NEW, "cu132-t2150-abc")["networkVolumes"] == []
    # No pool or no data center is not a choice, as no volume is.
    for key in ("gpu_pools", "data_center_ids"):
        empty = copy.deepcopy(PENDING)
        empty["lanes"][NEW][key] = []
        with pytest.raises(SystemExit, match=f"has no {key}: neither the lane"):
            runpod_endpoints.endpoint_body(empty, NEW, "cu132-t2150-abc")
    # The registry credential has no lane key.
    no_registry = copy.deepcopy(reg)
    del no_registry["endpoint_defaults"]["registry"]
    with pytest.raises(SystemExit, match="^endpoint_defaults is missing registry$"):
        runpod_endpoints.endpoint_body(no_registry, NEW, "cu132-t2150-abc")


def test_the_readme_lane_on_another_card_is_a_lane_the_registry_takes():
    """The example under "A lane on another card" is checked as a lane would
    be, so it cannot fall behind the rules."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("#### A lane on another card", 1)[1].split("```json", 1)[1]
    ((lane_id, lane),) = json.loads("{" + block.split("```", 1)[0] + "}").items()
    reg = copy.deepcopy(REGISTRY)
    reg["lanes"][lane_id] = lane
    assert lanes_mod.validate(reg, MANIFEST, PACKS) == []
    # Another pool, other data centers and no volume: each list is the lane's own.
    assert all(lane[key] != REGISTRY["endpoint_defaults"][key]
               for key in lanes_mod.PLACEMENT_KEYS)
    assert lanes_mod.without_volume(reg, lane) and lane["model_storage"] == "job"
    assert lanes_mod.is_long_cadence(lane) and lane["label"].startswith(lane["cadence"])
    assert lane["gpu"] in lane["label"]
    body = runpod_endpoints.endpoint_body(reg, lane_id, f"{lane['image']}-abc")
    assert body["name"] == lane["endpoint_name"]
    assert body["gpu"]["pools"] == lane["gpu_pools"]
    assert body["dataCenterIds"] == lane["data_center_ids"] and body["networkVolumes"] == []
    # It runs an image that exists and shares no endpoint: nothing is built or repointed.
    assert lanes_mod.image_matrix(reg) == lanes_mod.image_matrix(REGISTRY)
    assert lane["endpoint_name"] not in {runpod_endpoints.endpoint_name(other)
                                         for other in REGISTRY["lanes"].values()}


def test_status_holds_an_endpoint_to_the_pools_data_centers_and_volumes_of_its_lane(capsys):
    diff = runpod_endpoints.differences
    reg = _status_registry()
    # A lane that names none is held to the defaults. The order of a list does not count.
    assert diff(reg, "main", _live()) == []
    reg["endpoint_defaults"]["data_center_ids"] = ["EU-SE-1", "EU-RO-1"]
    assert diff(reg, "main", {**_live(), "dataCenterIds": ["EU-RO-1", "EU-SE-1"]}) == []
    assert diff(reg, "main", _live()) == [
        "dataCenterIds is ['EU-RO-1'] on RunPod, the manifest says ['EU-RO-1', 'EU-SE-1']"]
    # A lane that names its own is held to those.
    reg = _status_registry()
    reg["lanes"]["main"].update(OTHER_CARD)
    assert diff(reg, "main", _live()) == [
        "gpu.pools is ['ADA_24'] on RunPod, the manifest says ['BLACKWELL_96']",
        "dataCenterIds is ['EU-RO-1'] on RunPod, the manifest says ['US-CA-2', 'US-NC-1']",
        "networkVolumes is ['n0xnkwnogx'] on RunPod, the manifest says []"]
    other = {**_live(), "gpu": {"pools": ["BLACKWELL_96"], "count": 1, "minCudaVersion": "13.0"},
             "dataCenterIds": ["US-CA-2", "US-NC-1"], "networkVolumes": []}
    assert diff(reg, "main", other) == []
    # An endpoint without a volume: an empty list, or none at all.
    assert diff(reg, "main", {k: v for k, v in other.items() if k != "networkVolumes"}) == []
    assert diff(reg, "main", {**other, "networkVolumes": None, "gpu": None}) == [
        "gpu.pools is [] on RunPod, the manifest says ['BLACKWELL_96']"]
    # Something that is no list is printed as it is.
    assert diff(reg, "main", {**other, "dataCenterIds": "US-CA-2"}) == [
        "dataCenterIds is 'US-CA-2' on RunPod, the manifest says ['US-CA-2', 'US-NC-1']"]
    # A retired lane is held to 0 workers and nothing else, whatever it names.
    reg["lanes"]["old"].update(OTHER_CARD)
    assert diff(reg, "old", _live(workers_max=0)) == []
    # `status` prints them and counts them.
    client = FakeEndpoints({"ep-main": _live(), "ep-old": _live(workers_max=0)})
    assert runpod_endpoints.status(reg, client, None, strict=True) == 1
    out = capsys.readouterr().out
    assert ("main: differs: gpu.pools is ['ADA_24'] on RunPod, the manifest says "
            "['BLACKWELL_96']\n") in out
    assert "3 of the settings above differ from manifest/lanes.json" in out
    client = FakeEndpoints({"ep-main": other, "ep-old": _live(workers_max=0)})
    assert runpod_endpoints.status(reg, client, None, strict=True) == 0
    # A registry that names no such list holds the endpoint to none of them.
    del reg["endpoint_defaults"], reg["lanes"]["main"]["gpu_pools"]
    assert diff(reg, "main", _live()) == [
        "dataCenterIds is ['EU-RO-1'] on RunPod, the manifest says ['US-CA-2', 'US-NC-1']",
        "networkVolumes is ['n0xnkwnogx'] on RunPod, the manifest says []"]


def test_validate_checks_the_pools_data_centers_and_volumes_a_lane_names():
    ok = _registry(old=RETIRED_LANE, main={
        "role": "primary", "gpu_pools": ["ADA_48_PRO", "AMPERE_48"],
        "data_center_ids": ["US-IL-1"], "network_volumes": ["vol2"]})
    assert lanes_mod.validate(ok) == []
    assert lanes_mod.PLACEMENT_KEYS == ("gpu_pools", "data_center_ids", "network_volumes")

    def problems(key, value):
        bad = copy.deepcopy(ok)
        bad["lanes"]["main"][key] = value
        return lanes_mod.validate(bad)

    for key in ("gpu_pools", "data_center_ids"):
        assert problems(key, None) == []
        for value in ([], "ADA_24", ["ADA_24", 3], [""], [None], {"ADA_24": 1}, 24, True):
            assert problems(key, value) == [
                f"lane main: {key} must be a list of at least one id, not {value!r}"], (key, value)
    assert problems("network_volumes", None) == []
    for value in ("vol2", ["vol2", 3], [""], {"vol2": 1}, 0, False):
        assert problems("network_volumes", value) == [
            "lane main: network_volumes must be a list of volume ids, or [] for no volume, "
            f"not {value!r}"], value
    # The types are checked for a retired lane too.
    bad = copy.deepcopy(ok)
    bad["lanes"]["old"]["gpu_pools"] = "ADA_24"
    assert lanes_mod.validate(bad) == [
        "lane old: gpu_pools must be a list of at least one id, not 'ADA_24'"]


def test_validate_refuses_a_lane_without_a_volume_that_keeps_its_models_on_one():
    """The model sync fills the network volume. A lane without a volume reads
    none of it, so its models are fetched for each job."""
    ok = _registry(old=RETIRED_LANE, main={"role": "primary"},
                   big={"cadence": "weekly", **OTHER_CARD})
    assert lanes_mod.validate(ok) == []
    assert lanes_mod.without_volume(ok, ok["lanes"]["big"])
    assert not lanes_mod.without_volume(ok, ok["lanes"]["main"])
    assert not lanes_mod.without_volume(ok, None)
    said = "lane big: it has no network volume, so model_storage must be 'job', not {!r}"
    for value in (None, "volume"):
        bad = copy.deepcopy(ok)
        bad["lanes"]["big"]["model_storage"] = value
        assert lanes_mod.validate(bad) == [said.format(value)]
    bad = copy.deepcopy(ok)
    del bad["lanes"]["big"]["model_storage"]
    assert lanes_mod.validate(bad) == [said.format(None)]
    # With a volume, its own or the default one, both ways of storing are fine.
    for volumes in (["vol2"], None):
        for where in ("volume", "job", None):
            fine = copy.deepcopy(ok)
            fine["lanes"]["big"].update({"network_volumes": volumes, "model_storage": where})
            assert lanes_mod.validate(fine) == [], (volumes, where)
    # The list in effect counts: defaults without a volume leave every lane
    # without one, unless the lane names one.
    none = copy.deepcopy(ok)
    none["endpoint_defaults"] = {"network_volumes": []}
    assert lanes_mod.validate(none) == [
        "lane main: it has no network volume, so model_storage must be 'job', not None"]
    none["lanes"]["main"]["network_volumes"] = ["vol2"]
    assert lanes_mod.validate(none) == []
    assert not lanes_mod.without_volume(none, none["lanes"]["main"])
    # A retired lane runs nothing, so it reads no model.
    retired = copy.deepcopy(ok)
    retired["lanes"]["old"]["network_volumes"] = []
    assert lanes_mod.validate(retired) == []


def test_lanes_that_share_an_endpoint_name_the_same_pools_data_centers_and_volumes():
    """An endpoint has one list of each, whatever number of lanes use it."""
    assert lanes_mod.ENDPOINT_KEYS == ("image", "endpoint_name", "workers_max", "disk_gb",
                                       "idle_timeout_s", "env", *lanes_mod.PLACEMENT_KEYS)
    ok = _shared_registry()
    for key, value in (("gpu_pools", ["ADA_48_PRO"]), ("data_center_ids", ["US-IL-1"]),
                       ("network_volumes", ["vol2"])):
        bad = copy.deepcopy(ok)
        bad["lanes"]["week"][key] = value
        assert lanes_mod.validate(bad) == [
            f"lane week: {key} is {value!r}, but night, which shares its endpoint, says None"], key
        bad["lanes"]["night"][key] = value
        assert lanes_mod.validate(bad) == []
    # Without the volume, both of them fetch their models for each job.
    both = copy.deepcopy(ok)
    for lane_id in ("night", "week"):
        both["lanes"][lane_id].update({k: OTHER_CARD[k] for k in lanes_mod.PLACEMENT_KEYS})
    both["lanes"]["week"]["model_storage"] = "job"
    assert lanes_mod.validate(both) == [
        "lane night: it has no network volume, so model_storage must be 'job', not None"]
    both["lanes"]["night"]["model_storage"] = "job"
    assert lanes_mod.validate(both) == []
