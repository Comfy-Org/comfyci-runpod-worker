"""run_report: the job summary of a GPU regression run, rendered from an
in-memory results tree (no network), and the git-backed reader against a local
bare repo."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

import output_kinds
import run_report
from conftest import git, make_remote, workflow_steps

SHA = "c" * 40
PREV = "b" * 40
FIRST = "a" * 40
RESULTS = "d" * 40  # the results commit every link is pinned to
LANE = "py313-torch2.14.1-cu130"
OTHER = "py313-torch2.15.0-cu132"
LEGACY = "py312-torch2.11.0-cu128"
IDLE = "py314-torch2.16.0-cu132"
P = f"regression/lanes/{LANE}"
RAW = f"https://raw.githubusercontent.com/Comfy-Org/comfyci-runpod-worker/{RESULTS}"
COMFY = "https://github.com/Comfy-Org/ComfyUI"
RUN_URL = "https://github.com/Comfy-Org/comfyci-runpod-worker/actions/runs/4242"
FIXTURES = Path(__file__).parent / "fixtures" / "run_report"
ROOT = Path(__file__).resolve().parents[2]
ZW = run_report.ZW

# A registry of its own, so these tests do not follow the committed lane state.
LANES = {"primary": LANE, "lanes": {
    LANE: {"label": "py3.13 / torch 2.14.1 / cu130 / RTX 4090", "cadence": "per-commit",
           "blocking": True, "role": "primary", "layout": 2, "prefix": P, "endpoint_id": "ep-130"},
    OTHER: {"label": "py3.13 / torch 2.15.0 / cu132 / RTX 4090", "cadence": "per-commit",
            "blocking": False, "role": "lane", "layout": 2,
            "prefix": f"regression/lanes/{OTHER}", "endpoint_id": "ep-132"},
    LEGACY: {"label": "py3.12 / torch 2.11.0 / cu128 / RTX 4090", "cadence": "retired",
             "blocking": False, "role": "legacy", "layout": 1, "prefix": "regression",
             "endpoint_id": "ep-128"},
    IDLE: {"label": "py3.14 / torch 2.16.0 / cu132 / RTX 4090", "cadence": "per-commit",
           "blocking": False, "role": "lane", "layout": 2,
           "prefix": f"regression/lanes/{IDLE}", "endpoint_id": None}}}

IDENT = {"frames_compared": 1, "frame_count_ref": 1, "frame_count_cand": 1, "identical": True,
         "mean_mse": 0.0, "mean_psnr_db": None, "min_psnr_db": None, "max_abs_diff": 0,
         "mean_pct_pixels_changed": 0.0, "worst_frame_index": 0}
DIFF = {**IDENT, "identical": False, "mean_mse": 3.442175, "mean_psnr_db": 42.76,
        "min_psnr_db": 41.5, "max_abs_diff": 145, "mean_pct_pixels_changed": 69.986}
LIMITS = {"max_mean_mse": 2.0, "min_mean_psnr_db": 45.0, "max_pct_pixels_changed": 1.0}
ENV = {"image": "cu130-t2141", "image_sha": "4029bd69b0bd165f620cbdec5a3fc156a4574143",
       "python": "3.13.16 (main, Oct 14 2026) [Clang 23.1.3 ]", "torch": "2.14.1+cu130",
       "cuda": "13.0", "driver_version": "595.91.07",
       "runtime_backends": [
           "\x1b[32m[INFO]\x1b[0m Found comfy_kitchen backend cuda: {'available': True, "
           "'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln'",
           "\x1b[32m[INFO]\x1b[0m Found comfy_kitchen backend triton: {'available': True, "
           "'disabled': True, 'unavailable_reason': None",
           "\x1b[32m[INFO]\x1b[0m Found comfy_kitchen backend hip: {'available': False, "
           "'disabled': False, 'unavailable_reason': 'PyTorch ROCm/HIP runtime not available'"]}
# Everything a hostile commit, PR, log or worker could put in a free-text field.
EVIL = ("@octocat fixes #4242 and GH-7 | pipe `tick` <script>alert(1)</script> **bold** _em_ "
        "[x](https://evil.example/) https://evil.example/a www.evil.example $a+b$ "
        "\x1b[31mred\x1b[0m\x07\n::error::not a command\n# heading")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "RESULTS_REPO",
                "RESULTS_BRANCH", "REPORT_DASHBOARD_URL", "REPORT_WORKER_REPO"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(run_report.lanes_mod, "load_lanes", lambda path=None: LANES)


# -- builders -------------------------------------------------------------------
def _cell(wf, verdict="pass", **kw):
    """One summary cell, shaped as run_regression.process_workflow returns it."""
    ran = verdict in ("pass", "fail", "no_baseline")
    compared = verdict in ("pass", "fail")
    cell = {"workflow_id": wf, "worker_status": "ok" if ran else "timeout", "verdict": verdict,
            "detail": None, "vs_golden": IDENT if verdict == "pass" else DIFF if compared else None,
            "vs_previous": IDENT if verdict == "pass" else None,
            "thresholds_used": LIMITS if compared else None,
            "golden_tag": "v0.42.0" if compared else None,
            "previous_commit": PREV if ran else None, "previous_verdict": "pass" if ran else None,
            "figures": False, "output_sha256": "1" * 64 if ran else None,
            "thumbnail": f"{wf}_00001_.webp" if ran else None,
            "gpu_name": "cuda:0 NVIDIA GeForce RTX 4090" if ran else None,
            "timings": {"prompt_exec_s": 98.07, "server_start_s": 12.44} if ran else {},
            "vram_peak_mb": 21801.6 if ran else None, "rss_peak_mb": 34513.6 if ran else None,
            "comfy_version": "0.45.0" if ran else None,
            "torch_version": "2.14.1+cu130" if ran else None,
            "python_version": ENV["python"] if ran else None, "custom_nodes": None,
            "prior_verdict": "pass", "prior_commit": PREV}
    cell.update(kw)
    return cell


def _summary(cells, lane=LANE, commit=SHA, **kw):
    verdicts = [c["verdict"] for c in cells]
    summary = {"schema_version": 2, "branch": "master", "commit": commit, "run_ts": 1794702636,
               "overall": "fail" if {"fail", "execution_error"} & set(verdicts) else "pass",
               "has_infra_error": "infra_error" in verdicts, "lane": lane, "attempt": 1,
               "run_url": RUN_URL, "run_id": 4242, "env": ENV,
               "commit_meta": {"subject": "Add a node (#17754)", "author": "kijai",
                               "committed_ts": 1794702029, "parents": [PREV], "pr": 17754,
                               "pr_title": "Add a node", "pr_author": "kijai",
                               "pr_url": f"{COMFY}/pull/17754"},
               "tested_range": {"prev": PREV, "commits_between": 1,
                                "compare_url": f"{COMFY}/compare/{PREV}...{commit}"},
               "workflows": {c["workflow_id"]: c for c in cells}}
    summary.update(kw)
    return summary


def _run(wf, status="ok", **kw):
    run = {"status": status, "error": None,
           "validation": {"missing_nodes": [], "stripped_inputs": [], "filled_defaults": [],
                          "ok": True} if status == "ok" else None,
           "runpod_job_id": "job-1", "attempt": 1, "delay_ms": 14717, "delay_s": 14.717,
           "execution_ms": 143178,
           "outputs": [{"filename": f"{wf}_00001_.png", "bytes": 1650207, "sha256": "1" * 64}]
           if status == "ok" else []}
    run.update(kw)
    return run


def _entry(commit, t, cells, run=None, n=1, pr=17754, subject="Add a node"):
    """An index entry; `cells` maps workflow id -> index cell or bare verdict."""
    w = {wf: c if isinstance(c, dict) else
         {"v": c, "d": None, "sha": "1" * 64 if c in ("pass", "fail") else None, "exec": 93.34,
          "g": "v0.42.0" if c in ("pass", "fail") else None}
         for wf, c in cells.items()}
    return {"c": commit, "t": t, "run": run,
            "o": "fail" if any(c["v"] in ("fail", "execution_error") for c in w.values())
            else "pass",
            "m": {"s": f"{subject} (#{pr})", "a": "kijai", "pr": pr, "ct": t - 600},
            "prev": None, "range": {"n": n, "url": "u"}, "w": w}


def _head(entries, first_bad=None):
    return {"count": len(entries), "first_bad": first_bad or {}, "entries": entries}


def _tree(summary, runs=None, lane=LANE, head=None, registry=None, extra=None):
    """The results tree a run leaves behind: its summary and run records, the
    lane's index head and the lane registry."""
    run_dir = f"{LANES['lanes'][lane]['prefix']}/runs/master/{summary['commit']}"
    cells = summary["workflows"]
    files = {f"{run_dir}/summary.json": summary}
    for wf, cell in cells.items():
        files[f"{run_dir}/{wf}/run.json"] = (runs or {}).get(wf) or _run(wf, cell["worker_status"])
    if head is None:
        this = {wf: {"v": c["verdict"], "d": c.get("detail"), "sha": c.get("output_sha256"),
                     "exec": (c.get("timings") or {}).get("prompt_exec_s"),
                     "g": c.get("golden_tag")}
                for wf, c in cells.items()}
        head = _head([_entry(summary["commit"], 1794702636, this, run=4242),
                      _entry(PREV, 1794700000, dict.fromkeys(cells, "pass"), pr=17753)])
    files[f"regression/index/master/{lane}.json"] = head
    files["regression/index/lanes.json"] = registry or {"lanes": {lane: {"goldens": {}}}}
    files.update(extra or {})
    return files


def _report(files, lane=LANE, commit=SHA, sha=RESULTS, **kw):
    reader = run_report.DictReader(files, sha)
    return run_report.lane_report(reader, LANES, commit, "master", lane, **kw)


def _section(out, title):
    """The lines of one '### title' section."""
    lines = out.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"### {title}"))
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("### ") or lines[i] == "---"), len(lines))
    return lines[start:end]


def _outside_pre(out):
    """The Markdown outside <pre> blocks: where inline escaping applies."""
    return re.sub(r"<pre>\n.*?\n</pre>", "", out, flags=re.S)


# -- a green run ------------------------------------------------------------------
def test_green_run_reports_every_workflow_with_pinned_images():
    out = _report(_tree(_summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i")])))
    lines = out.split("\n")
    assert lines[0] == ("## ✅ GPU regression · py3.13 / torch 2.14.1 / cu130 / RTX 4090 · "
                        "`cccccccc`")
    assert "All 2 workflows match their reference." in lines
    assert "What needs attention" not in out

    assert (f"| Commit | [`{SHA[:12]}`]({COMFY}/commit/{SHA}) · Add a node (\\#{ZW}17754) · "
            f"by kijai · committed 2026-11-15 00:20 UTC · parent [`{PREV[:7]}`]({COMFY}/commit/"
            f"{PREV}) |") in lines
    assert f"| Pull request | [#17754]({COMFY}/pull/17754) · Add a node · by kijai |" in lines
    # No "attempt 1 of 2" on a run with a verdict: nothing tests it a second time.
    assert ("| Tested | 2026-11-15 00:30 UTC · 10 min after the commit · "
            f"[Actions run]({RUN_URL}) |") in lines
    assert f"| Covers | 1 commit since [`{PREV[:7]}`]({COMFY}/commit/{PREV}) |" in lines
    assert (f"| Lane | py3.13 / torch 2.14.1 / cu130 / RTX 4090 · `{LANE}` · branch `master` · "
            "role primary · cadence per-commit · blocking |") in lines
    assert ("| Environment | GPU cuda:0 NVIDIA GeForce RTX 4090 · driver 595.91.07 · "
            "Python 3.13.16 · torch 2.14.1+cu130 · CUDA 13.0 · ComfyUI 0.45.0 · "
            "worker image cu130-t2141 · image build [`4029bd6`](https://github.com/Comfy-Org/"
            f"comfyci-runpod-worker/commit/{ENV['image_sha']}) · "
            "backends: cuda, triton (disabled), hip (not available) |") in lines

    assert "### All workflows (2)" in lines
    assert ("| Workflow | Result | Output | vs reference | vs previous | Reference | Exec s (usual "
            "range) | Server start s (checkout s) | Peak VRAM GB | Peak RSS GB | Output "
            "sha256 |") in lines
    run_dir = f"{RAW}/{P}/runs/master/{SHA}"
    # The thumbnail sits right after the result, where a narrow page still shows it.
    assert (f'| `flux_dev_t2i` | ✅ pass | <a href="{run_dir}/flux_dev_t2i/outputs/'
            f'flux_dev_t2i_00001_.png"><img src="{run_dir}/flux_dev_t2i/outputs/thumbs/'
            'flux_dev_t2i_00001_.webp" width="96" alt="flux_dev_t2i output"></a> | bit-identical '
            f"| identical to [`{PREV[:7]}`]({COMFY}/commit/{PREV}) | `v0.42.0` | 98.1 "
            "| 12.4 | 21.3 | 33.7 | `111111111111` |") in lines

    # Every output file, behind a fold, and each workflow's run record.
    assert "<details><summary>Output files (2)</summary>" in lines
    assert (f"| `sdxl_t2i` | [sdxl\\_t2i\\_00001\\_.png]({run_dir}/sdxl_t2i/outputs/"
            f"sdxl_t2i_00001_.png) | 1.57 MB | `{'1' * 64}` |  |") in lines
    assert (f"Run records (run.json): [`flux_dev_t2i`]({run_dir}/flux_dev_t2i/run.json), "
            f"[`sdxl_t2i`]({run_dir}/sdxl_t2i/run.json).") in lines

    # Links are absolute, and every results link is pinned to the commit read.
    urls = re.findall(r'(?:src|href)="([^"]+)"', out) + re.findall(r"\]\(([^)\s]+)\)", out)
    assert urls and all(u.startswith("https://") for u in urls)
    results_links = [u for u in urls if "raw.githubusercontent.com" in u or "/tree/" in u]
    assert len(results_links) > 6 and all(f"/{RESULTS}/" in u for u in results_links)
    assert "data:" not in out and "/results/" not in out

    assert (f"Files: [summary.json]({run_dir}/summary.json), [every file of this run]("
            f"https://github.com/Comfy-Org/comfyci-runpod-worker/tree/{RESULTS}/{P}/runs/master/"
            f"{SHA}). The results links are pinned to results commit `dddddddd`.") in lines
    assert any(line.startswith("Result labels: ✅ pass, ") for line in lines)
    assert ("Longer history and charts: [ci.comfy.org](https://ci.comfy.org) · [this commit on "
            f"this lane](https://ci.comfy.org/regression/master/{SHA}?lane={LANE})") in lines
    assert "Shortened" not in out


def test_facts_name_the_baked_packs_the_checkout_time_and_only_real_backends():
    env = {**ENV, "custom_nodes_sha256": "0127" * 16, "runtime_backends": [
        *ENV["runtime_backends"], "\x1b[32m[INFO]\x1b[0m pytorch version: 2.14.1+cu130",
        "\x1b[32m[INFO]\x1b[0m Using pytorch attention",
        "\x1b[32m[INFO]\x1b[0m aimdo: control.c:283:INFO:comfy-aimdo inited for GPU: RTX 4090"]}
    cell = _cell("flux_dev_t2i", timings={"prompt_exec_s": 98.07, "server_start_s": 12.44,
                                          "checkout_s": 2})
    summary = _summary([cell], env=env)
    summary["commit_meta"]["parents"] = [PREV, FIRST, "not a sha"]
    out = _report(_tree(summary))
    # Startup log lines that are not backends stay out of the row.
    assert ("worker image cu130-t2141 · image build [`4029bd6`](https://github.com/Comfy-Org/"
            f"comfyci-runpod-worker/commit/{ENV['image_sha']}) · baked packs `012701270127` · "
            "backends: cuda, triton (disabled), hip (not available) |") in out
    assert "pytorch version" not in out and "aimdo" not in out
    assert (f"· parents [`{PREV[:7]}`]({COMFY}/commit/{PREV}), [`{FIRST[:7]}`]({COMFY}/commit/"
            f"{FIRST}) |") in out
    assert "| 98.1 | 12.4 (2.0) | 21.3 | 33.7 |" in out
    # A run from before the checkout was timed, and one that only got that far.
    assert run_report.start_cell({"server_start_s": 12.44}) == "12.4"
    assert run_report.start_cell({"checkout_s": 2}) == "(2.0)"
    assert run_report.start_cell({}) == ""


def test_a_time_is_shown_against_the_workflows_usual_range_not_the_previous_run():
    wf = "flux_dev_t2i"

    def head(times, this=98.07):
        cell = lambda secs, v="pass": {"v": v, "d": None, "sha": "1" * 64, "exec": secs,
                                       "g": "v0.42.0"}
        older = [_entry(f"{i:040x}", 1794700000 - i, {wf: c if isinstance(c, dict) else cell(c)})
                 for i, c in enumerate(times, 1)]
        return _head([_entry(SHA, 1794702636, {wf: cell(this)}, run=4242), *older])

    def row(times, secs=98.07):
        cell = _cell(wf, timings={"prompt_exec_s": secs, "server_start_s": 12.44})
        out = _report(_tree(_summary([cell]), head=head(times, secs)))
        return next(line for line in out.splitlines() if line.startswith(f"| `{wf}` |"))

    # Ten passing runs from 90 to 108 s: the range is their 10th and 90th percentile.
    ten = [90, 92, 94, 96, 98, 100, 102, 104, 106, 108]
    assert "| 98.1 (usual 92 to 106) |" in row(ten)
    # Inside or a little over the range says nothing more, well over it is marked.
    assert "| 120.0 (usual 92 to 106) |" in row(ten, 120.0)
    assert "| 159.1 (usual 92 to 106) ⚠️ slow |" in row(ten, 159.1)
    # This commit's own time, failed runs and runs without a time are not part of the range.
    noisy = [*ten, {"v": "fail", "d": None, "sha": "2" * 64, "exec": 900.0, "g": "v0.42.0"},
             {"v": "pass", "d": None, "sha": "1" * 64, "exec": None, "g": "v0.42.0"}]
    assert "| 98.1 (usual 92 to 106) |" in row(noisy)
    # Too few runs to call anything usual: the time stands alone.
    assert "| 98.1 | 12.4 |" in row(ten[:7])
    # Only the newest 30 passing runs count.
    assert "(usual 50 to 50)" in row([50] * 30 + [500] * 30, 50.0)
    # A workflow that takes under ten seconds keeps a decimal.
    assert "| 0.5 (usual 0.5 to 0.6) |" in row([0.5] * 8 + [0.6] * 2, 0.5)

    out = _report(_tree(_summary([_cell(wf)]), head=head(ten)))
    assert run_report.EXEC_NOTE in out.splitlines()
    assert "not part of the result" in run_report.EXEC_NOTE


def test_layout_1_lane_reads_the_bare_regression_prefix():
    summary = _summary([_cell("sdxl_t2i")], lane=LEGACY)
    out = _report(_tree(summary, lane=LEGACY), lane=LEGACY)
    assert f"{RAW}/regression/runs/master/{SHA}/sdxl_t2i/outputs/sdxl_t2i_00001_.png" in out
    assert "role legacy · retired: history only · informational" in out
    assert "### Recent commits on this lane" in out  # regression/index/master/<lane>.json


def test_a_run_without_thumbnails_shows_the_png_itself_unless_it_is_huge():
    cell = _cell("sdxl_t2i", thumbnail=None)
    png = f"{RAW}/{P}/runs/master/{SHA}/sdxl_t2i/outputs/sdxl_t2i_00001_.png"
    assert f'<a href="{png}"><img src="{png}" width="96"' in _report(_tree(_summary([cell])))
    big = {"sdxl_t2i": _run("sdxl_t2i", outputs=[
        {"filename": "sdxl_t2i_00001_.png", "bytes": 30 * 1024 * 1024, "sha256": "1" * 64}])}
    out = _report(_tree(_summary([cell]), runs=big))
    assert f"| [PNG]({png}) |" in out and "<img" not in out


def test_truncated_outputs_are_listed_as_not_published():
    runs = {"vhs": _run("vhs", outputs=[
        {"filename": "vhs_00001_.png", "bytes": 47555, "sha256": "1" * 64},
        {"filename": "vhs_video.gif", "bytes": 83174, "sha256": "2" * 64},
        {"filename": "vhs_huge.png", "bytes": 16 * 1024 * 1024, "sha256": "3" * 64,
         "truncated": True}])}
    out = _report(_tree(_summary([_cell("vhs")]), runs=runs))
    base = f"{RAW}/{P}/runs/master/{SHA}/vhs/outputs"
    assert f"| `vhs` | [vhs\\_video.gif]({base}/vhs_video.gif) | 81.2 KB | `{'2' * 64}` |  |" in out
    assert (f"| `vhs` | vhs\\_huge.png | 16.00 MB | `{'3' * 64}` | not published: over the "
            "worker's size limit |") in out
    assert "vhs_huge.png)" not in out  # no link to a file that is not there


# -- execution errors -----------------------------------------------------------------
def test_execution_error_shows_status_detail_and_log_tail():
    tail = ("\x1b[31mTraceback\x1b[0m line\n::error::from the log\n<b>bold</b> & more @octocat\n"
            "last line\n")
    cells = [_cell("flux_dev_t2i"),
             _cell("z_image_turbo_t2i", "execution_error", prior_verdict="execution_error")]
    runs = {"z_image_turbo_t2i": _run("z_image_turbo_t2i", "timeout", error={
        "timeout_s": 900, "prompt_id": "ec6a84ae", "log_tail": tail})}
    first_bad = {"z_image_turbo_t2i": {"commit": FIRST, "run_ts": 1794000000,
                                       "kind": "execution_error", "prev_good": "9" * 40,
                                       "pr": 17507, "author": "bigcat88", "golden": None,
                                       "runs": 3}}
    head = _head([_entry(SHA, 3, {"flux_dev_t2i": "pass", "z_image_turbo_t2i": "execution_error"}),
                  _entry(PREV, 2, {"flux_dev_t2i": "pass"})], first_bad)
    out = _report(_tree(_summary(cells), runs=runs, head=head))
    lines = out.split("\n")

    assert lines[0].startswith("## ❌ GPU regression · ")
    assert "1 of 2 workflows failed, inherited from an earlier commit." in lines
    assert "#### ❌ `z_image_turbo_t2i`: execution error" in lines
    assert ("The workflow did not finish on this commit. Worker status: <code>timeout</code>. "
            f"It already ended in an execution error at [`{PREV[:7]}`]({COMFY}/commit/{PREV}), "
            "so the error is inherited from an earlier commit.") in lines
    assert (f"Failing since [`{FIRST[:7]}`]({COMFY}/commit/{FIRST}) ([#17507]({COMFY}/pull/17507)"
            f", bigcat88), first seen 2026-11-06, 3 runs, last passed at [`9999999`]({COMFY}/"
            f"commit/{'9' * 40}).") in lines
    assert ("- **Timeout:** the worker gave up after 900 s (prompt <code>ec6a84ae</code>)"
            in lines)
    assert ("- RunPod job <code>job-1</code>, queued 14.7 s, ran 143.2 s, RunPod attempt 1"
            in lines)
    # The log tail: folded, in <pre>, HTML-escaped, colour codes gone, and no
    # line that the runner could read as a workflow command.
    assert "<details><summary>Server log tail (74 characters)</summary>" in lines
    assert ("<pre>\nTraceback line\n&#58;:error::from the log\n&lt;b&gt;bold&lt;/b&gt; &amp; more "
            "@octocat\nlast line\n</pre>") in out
    assert "\x1b" not in out
    assert not any(line.startswith("::") for line in lines)
    assert "| `z_image_turbo_t2i` | ❌ execution error |  |  |  |  |  |  |  |  |  |" in lines
    assert "gh workflow run" not in out  # nothing to approve


def test_new_execution_error_names_the_last_good_commit():
    cells = [_cell("wan", "execution_error", prior_verdict="pass", prior_commit=PREV)]
    out = _report(_tree(_summary(cells), runs={"wan": _run("wan", "execution_error")}))
    assert "1 of 1 workflow failed, new at this commit." in out
    assert (f"Before this commit it last ran to the end at [`{PREV[:7]}`]({COMFY}/commit/"
            f"{PREV}) (pass), so the error is new at this commit.") in out
    assert "- The worker recorded no detail for this failure." in out
    # No prior result at all: say that it is unknown, do not guess.
    cells = [_cell("wan", "execution_error", prior_verdict=None, prior_commit=None)]
    out = _report(_tree(_summary(cells), runs={"wan": _run("wan", "execution_error")},
                        head=_head([])))
    assert "1 of 1 workflow failed." in out
    assert "it is not known whether the error is new at this commit" in out


def test_a_run_of_several_commits_does_not_pin_a_new_failure_on_one():
    """HEAD polling can skip commits: a failure that is new in such a run
    started in one of the commits it covers, not certainly in the tested one."""
    rng = {"prev": PREV, "commits_between": 3}
    cells = [_cell("flux_dev_t2i", "fail", detail="new_drift", vs_previous=DIFF),
             _cell("wan", "execution_error", prior_verdict="pass", prior_commit=PREV)]
    first_bad = {wf: {"commit": SHA, "run_ts": 1794702636, "kind": kind, "prev_good": PREV,
                      "pr": 17754, "author": "kijai", "golden": tag, "runs": 1}
                 for wf, kind, tag in (("flux_dev_t2i", "fail", "v0.42.0"),
                                       ("wan", "execution_error", None))}
    head = _head([_entry(SHA, 1794702636, {"flux_dev_t2i": {"v": "fail", "d": "new_drift",
                                                            "sha": "1" * 64, "g": "v0.42.0"},
                                           "wan": "execution_error"}, run=4242, n=3),
                  _entry(PREV, 1794700000, {"flux_dev_t2i": "pass", "wan": "pass"})], first_bad)
    files = _tree(_summary(cells, tested_range=rng), head=head,
                  runs={"wan": _run("wan", "execution_error")})
    out = _report(files)
    lines = out.split("\n")
    cmp_ = f"[compare]({COMFY}/compare/{PREV}...{SHA})"
    assert "2 of 2 workflows failed, new in this run, which covers 3 commits." in lines
    assert "#### ❌ `flux_dev_t2i`: output changed (new in this run)" in lines
    assert (f"the change is new in this run. This run covers 3 commits, so the change is in one "
            f"of them ({cmp_}).") in out
    assert (f"(pass), so the error is new in this run. This run covers 3 commits, so the error "
            f"started in one of them ({cmp_}).") in out
    assert "| `flux_dev_t2i` | ❌ output changed (new in this run) |" in out
    # The chain is not pinned on the tested commit's pull request either.
    since = (f"ailing since [`{SHA[:7]}`]({COMFY}/commit/{SHA}) (a run that covered 3 commits), "
             "first seen 2026-11-15, 1 run, ")
    assert (f"F{since}reference `v0.42.0`, last passed at [`{PREV[:7]}`]({COMFY}/commit/{PREV})."
            in lines)
    assert (f"- `wan`: execution error, f{since}last passed at [`{PREV[:7]}`]({COMFY}/commit/"
            f"{PREV})") in lines
    row = next(line for line in _section(out, "Recent commits on this lane")
               if "**this report**" in line)
    assert row.endswith("| ❌ 2 of 2 failed | `flux_dev_t2i` output changed (new in this run); "
                        "`wan` execution error |")
    assert "new at this commit" not in out.replace("new at this commit (new in this run", "")
    # The same in the overview, for the commit's row and for the matrix.
    out = run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA, "master")
    assert out.count("`flux_dev_t2i` output changed (new in this run)") == 2
    # Mixed with an inherited failure, each part keeps its own wording.
    cells[1] = _cell("wan", "execution_error", prior_verdict="execution_error")
    out = _report(_tree(_summary(cells, tested_range=rng), head=head,
                        runs={"wan": _run("wan", "execution_error")}))
    assert "2 of 2 workflows failed: 1 new in this run, 1 inherited." in out


@pytest.mark.parametrize("status, error, expected", [
    ("execution_error",
     {"node_id": "12", "node_type": "KSampler", "message": "CUDA out of memory",
      "traceback": ["Traceback (most recent call last):\n", "  File \"x.py\", line 1\n",
                    "RuntimeError: <boom>\n"], "log_tail": "tail"},
     ["- **Message:** CUDA out of memory",
      "- **Node:** type <code>KSampler</code>, id <code>12</code>", "Traceback:",
      "<pre>\nTraceback (most recent call last):\n  File \"x.py\", line 1\n"
      "RuntimeError: &lt;boom&gt;\n</pre>",
      "<details><summary>Server log tail (4 characters)</summary>"]),
    ("prompt_rejected", {"detail": '{"error": {"type": "prompt_outputs_failed_validation"}}'},
     ["The server rejected the prompt:",
      '<pre>\n{"error": {"type": "prompt_outputs_failed_validation"}}\n</pre>']),
    ("validation_error",
     {"missing_nodes": ["FooNode"], "stripped_inputs": ["KSampler.x=1"], "filled_defaults": [],
      "ok": False, "error": "object_info fetch failed: timed out"},
     ["- **Missing node classes:** <code>FooNode</code>",
      "- **Inputs removed:** <code>KSampler.x=1</code>",
      "- **Validation:** object\\_info fetch failed: timed out"]),
    ("custom_node_error",
     {"message": "custom node pack(s) requested by the workflow did not load: "
                 "ComfyUI-KJNodes: IMPORT FAILED",
      "custom_nodes": {"ComfyUI-KJNodes": {"import": "failed", "seconds": 0.2}},
      "startup_failure": "the server exited", "returncode": 1},
     ["- **Message:** custom node pack(s) requested by the workflow did not load: "
      "ComfyUI-KJNodes: IMPORT FAILED",
      "- **Custom-node packs:** <code>ComfyUI-KJNodes</code> failed (0.2 s)",
      "- **Server exit code:** 1", "- **Server start:** the server exited"]),
    ("timeout", None, ["- The worker recorded no detail for this failure."]),
])
def test_failure_detail_by_worker_status(status, error, expected):
    cells = [_cell("wf", "execution_error", worker_status=status)]
    out = _report(_tree(_summary(cells), runs={"wf": _run("wf", status, error=error)}))
    assert f"Worker status: <code>{status.replace('_', chr(92) + '_')}</code>." in out
    for text in expected:
        assert text in out


def test_harness_error_is_reported_from_the_summary_cell():
    """A workflow whose orchestration raised has no run.json: the summary cell
    carries the error as a string."""
    cell = {"workflow_id": "wf", "worker_status": "harness_error", "verdict": "infra_error",
            "error": "RuntimeError('could not reach RunPod')", "prior_verdict": None,
            "prior_commit": None}
    files = _tree(_summary([cell]))
    del files[f"{P}/runs/master/{SHA}/wf/run.json"]
    out = _report(files)
    assert "Worker status: <code>harness\\_error</code>." in out
    assert "- **Message:** RuntimeError('could not reach RunPod')" in out


# -- output changes ------------------------------------------------------------------
def _drift_files(cell, first_bad=None, extra=None, current_tag="v0.42.0", registry=None):
    figures = {f"{P}/runs/master/{SHA}/flux_dev_t2i/figures/{name}": None
               for name in ("side_by_side_golden.png", "diff_heatmap_golden.png",
                            "side_by_side_prev.png", "diff_heatmap_prev.png")}
    head = _head([_entry(SHA, 3, {"flux_dev_t2i": {"v": "fail", "d": cell.get("detail"),
                                                   "sha": cell.get("output_sha256"),
                                                   "g": cell.get("golden_tag")},
                                  "sdxl_t2i": "pass"}),
                  _entry(PREV, 2, {"flux_dev_t2i": "pass", "sdxl_t2i": "pass"})], first_bad)
    return _tree(_summary([cell, _cell("sdxl_t2i")]), head=head, registry=registry, extra={
        f"{P}/golden/flux_dev_t2i/current.json": {
            "tag": current_tag, "blessed_by": "marawan206", "blessed_ts": 1792378306,
            "reason": "First baselines for the lane"},
        f"{P}/golden/flux_dev_t2i/v0.42.0/noise_floor.json": {**IDENT, "measured": True},
        **(figures if cell.get("figures") else {}), **(extra or {})})


def test_new_output_change_shows_numbers_limits_figures_and_how_to_accept():
    cell = _cell("flux_dev_t2i", "fail", detail="new_drift", figures=True, vs_previous=DIFF,
                 output_sha256="2" * 64)
    out = _report(_drift_files(cell))
    lines = out.split("\n")
    assert lines[0].startswith("## ❌ ")
    assert "1 of 2 workflows failed, new at this commit." in lines
    assert "#### ❌ `flux_dev_t2i`: output changed (new at this commit)" in lines
    prev = f"[`{PREV[:7]}`]({COMFY}/commit/{PREV})"
    assert ("The output differs from the reference beyond its limits, and it differs from the "
            f"previous tested commit {prev}: the change is new at this commit.") in lines
    # Each number sits next to its limit; a crossed limit is bold.
    assert ("| Reference `v0.42.0` | **3.4422** (limit ≤ 2) | **42.76**, min 41.50 (limit ≥ 45) "
            "| **69.986** (limit ≤ 1) | 145 | 1 |") in lines
    assert f"| Previous tested commit {prev} | 3.4422 | 42.76, min 41.50 | 69.986 | 145 | 1 |" \
        in lines
    assert "| Noise floor (the reference rendered twice) | bit-identical |  |  |  | 1 |" in lines
    assert ("Reference: `v0.42.0`, approved 2026-10-19 02:51 UTC by marawan206. Reason: First "
            "baselines for the lane") in lines
    figures = f"{RAW}/{P}/runs/master/{SHA}/flux_dev_t2i/figures"
    for name in ("side_by_side_golden.png", "diff_heatmap_golden.png", "side_by_side_prev.png",
                 "diff_heatmap_prev.png"):
        assert f'<a href="{figures}/{name}"><img src="{figures}/{name}" width="720" alt="' in out
    assert "<details><summary>Figures against the previous tested commit</summary>" in lines
    base = ("gh workflow run golden-baseline.yml --repo Comfy-Org/comfyci-runpod-worker "
            f"-f ref={SHA} -f workflows=flux_dev_t2i -f lane={LANE}")
    assert f"{base} -f bless=false" in lines and f"{base} -f bless_only=true" in lines
    row = next(line for line in lines
               if line.startswith("| `flux_dev_t2i` | ❌ output changed (new at this commit) | <a "))
    assert "</a> | MSE 3.4422 · PSNR 42.76 dB · 69.986 % px |" in row


def test_only_crossed_limits_are_bold_and_a_frame_count_mismatch_is_flagged():
    vs_golden = {**DIFF, "mean_mse": 1.5, "mean_psnr_db": 46.0, "min_psnr_db": 45.5,
                 "frame_count_cand": 2, "frames_compared": 1}
    cell = _cell("flux_dev_t2i", "fail", detail="new_drift", vs_golden=vs_golden)
    out = _report(_drift_files(cell))
    assert ("| Reference `v0.42.0` | 1.5000 (limit ≤ 2) | 46.00, min 45.50 (limit ≥ 45) | "
            "**69.986** (limit ≤ 1) | 145 | **2 here, 1 in the reference** |") in out
    # A batch of commits: the change is in one of them, not pinned on this one.
    summary = _summary([cell], tested_range={"prev": PREV, "commits_between": 3})
    out = _report(_tree(summary))
    cmp_ = f"[compare]({COMFY}/compare/{PREV}...{SHA})"
    assert (f"| Covers | 3 commits since [`{PREV[:7]}`]({COMFY}/commit/{PREV}) ({cmp_}): a "
            "change seen here is in one of them |") in out
    assert f"This run covers 3 commits, so the change is in one of them ({cmp_})." in out


def test_covers_row_says_when_commits_inside_the_range_were_tested():
    """The range starts at the previous-run pointer, which only moves on a run
    that finished every workflow. In a streak of execution errors it grows with
    every run, though each commit in it was tested."""
    last_full = "7" * 40
    cells = [_cell("flux_dev_t2i", previous_commit=last_full),
             _cell("wan", "execution_error", prior_verdict="execution_error")]
    rng = {"prev": last_full, "commits_between": 4}
    streak = {"flux_dev_t2i": "pass", "wan": "execution_error"}
    entries = [_entry(SHA, 1794702636, streak, n=4),
               _entry(PREV, 1794700000, streak, n=3, pr=17753),
               _entry("6" * 40, 1794690000, streak, n=1, pr=17752),
               _entry(last_full, 1794680000, {"flux_dev_t2i": "pass", "wan": "pass"}, pr=17751),
               _entry("5" * 40, 1794670000, {"flux_dev_t2i": "pass", "wan": "pass"}, pr=17750)]
    runs = {"wan": _run("wan", "execution_error")}
    out = _report(_tree(_summary(cells, tested_range=rng), head=_head(entries), runs=runs))
    cmp_ = f"[compare]({COMFY}/compare/{last_full}...{SHA})"
    assert (f"| Covers | 4 commits since [`{last_full[:7]}`]({COMFY}/commit/{last_full}) ({cmp_}), "
            "the commit of the previous run that finished every workflow. The lane tested 2 of "
            "them in between; outputs are still compared with that run, so a change seen here "
            "is in one of the 4 |") in out
    # Nothing in between has a result: the plain wording.
    out = _report(_tree(_summary(cells, tested_range=rng), runs=runs,
                        head=_head([entries[0], entries[3]])))
    assert f"({cmp_}): a change seen here is in one of them |" in out


def test_figures_are_shown_from_the_listing_when_the_summary_predates_the_flag():
    cell = _cell("flux_dev_t2i", "fail", detail="new_drift", vs_previous=DIFF)
    del cell["figures"]
    listing = {f"{P}/runs/master/{SHA}/flux_dev_t2i/figures/side_by_side_golden.png": None}
    out = _report(_drift_files(cell, extra=listing))
    assert "figures/side_by_side_golden.png\" width=\"720\"" in out
    assert "diff_heatmap_golden.png" not in out  # only what is published


def test_a_pass_that_is_not_bit_identical_shows_its_numbers_next_to_the_limits():
    """A workflow can pass while its output moved: how close it is to a limit,
    the noise floor and the figures the run rendered belong in the report."""
    near = {**DIFF, "mean_mse": 0.5, "mean_psnr_db": 51.2, "min_psnr_db": 50.9,
            "mean_pct_pixels_changed": 0.4, "max_abs_diff": 3}
    cell = _cell("flux_dev_t2i", vs_golden=near, vs_previous=near, figures=True)
    figures = f"{P}/runs/master/{SHA}/flux_dev_t2i/figures"
    out = _report(_tree(_summary([cell, _cell("sdxl_t2i")]), extra={
        f"{P}/golden/flux_dev_t2i/v0.42.0/noise_floor.json": {**IDENT, "measured": True},
        f"{figures}/side_by_side_golden.png": None, f"{figures}/side_by_side_prev.png": None}))
    lines = out.split("\n")
    assert lines[0].startswith("## ✅ ") and "All 2 workflows match their reference." in lines
    assert "### What needs attention" not in lines
    block = _section(out, "Passed without being bit-identical")
    assert block[2] == ("These outputs differ from their reference or from the previous tested "
                        "commit, within the limits.")
    assert "<details><summary><code>flux_dev_t2i</code>: numbers and limits</summary>" in block
    assert ("| Reference `v0.42.0` | 0.5000 (limit ≤ 2) | 51.20, min 50.90 (limit ≥ 45) | 0.400 "
            "(limit ≤ 1) | 3 | 1 |") in block
    assert "| Noise floor (the reference rendered twice) | bit-identical |  |  |  | 1 |" in block
    assert f'<img src="{RAW}/{figures}/side_by_side_golden.png" width="720"' in "\n".join(block)
    assert f'<img src="{RAW}/{figures}/side_by_side_prev.png" width="720"' in "\n".join(block)
    assert "sdxl_t2i" not in "\n".join(block)  # bit-identical: nothing to show
    assert "gh workflow run" not in out  # nothing to approve
    # Every workflow bit-identical: no section.
    assert "Passed without" not in _report(_tree(_summary([_cell("sdxl_t2i")])))


def test_a_reference_approved_since_the_run_is_named():
    cell = _cell("flux_dev_t2i", "fail", detail="new_drift")
    out = _report(_drift_files(cell, current_tag="v0.43.0"))
    assert "The lane's reference today is `v0.43.0`, approved 2026-10-19 02:51 UTC by " \
           "marawan206." in out
    assert "It is not the reference this run was compared with (`v0.42.0`)." in out


def test_inherited_change_links_to_the_first_failing_commits_figures():
    cell = _cell("flux_dev_t2i", "fail", detail="inherited", vs_previous=IDENT,
                 previous_verdict="fail", output_sha256="2" * 64)
    first_bad = {"flux_dev_t2i": {"commit": FIRST, "run_ts": 1794000000, "kind": "fail",
                                  "prev_good": "9" * 40, "pr": 16488, "author": "comfyanonymous",
                                  "golden": "v0.42.0", "runs": 12}}
    first = f"{P}/runs/master/{FIRST}/flux_dev_t2i/figures"
    files = _drift_files(cell, first_bad, extra={
        f"{first}/side_by_side_golden.png": None, f"{first}/diff_heatmap_golden.png": None})
    out = _report(files)
    lines = out.split("\n")
    assert "1 of 2 workflows failed, inherited from an earlier commit." in lines
    assert "#### ❌ `flux_dev_t2i`: output changed (inherited)" in lines
    assert (f"Failing since [`{FIRST[:7]}`]({COMFY}/commit/{FIRST}) ([#16488]({COMFY}/pull/16488)"
            ", comfyanonymous), first seen 2026-11-06, 12 runs, reference `v0.42.0`, last passed "
            f"at [`9999999`]({COMFY}/commit/{'9' * 40}).") in lines
    assert (f"| Previous tested commit [`{PREV[:7]}`]({COMFY}/commit/{PREV}) | bit-identical "
            "|  |  |  | 1 |") in lines
    assert (f"The figures of the first failing commit [`{FIRST[:7]}`]({COMFY}/commit/{FIRST}) "
            "show the same difference: [Reference, this commit and their difference]("
            f"{RAW}/{first}/side_by_side_golden.png), [Difference from the reference, worst "
            f"frame]({RAW}/{first}/diff_heatmap_golden.png).") in out
    assert not [img for img in re.findall(r"<img[^>]+>", out) if "/figures/" in img]
    assert "-f bless_only=true" in out  # an inherited change can still be accepted
    # A run from before inherited failures stopped rendering figures shows its own.
    own = f"{P}/runs/master/{SHA}/flux_dev_t2i/figures"
    out = _report({**files, f"{own}/side_by_side_golden.png": None})
    assert f'<img src="{RAW}/{own}/side_by_side_golden.png" width="720"' in out
    assert "no figures of its own" not in out and "-f bless_only=true" in out


def test_accepted_change_is_labelled_and_not_counted_as_open():
    """The lane's reference was approved to this very output after the run."""
    cell = _cell("flux_dev_t2i", "fail", detail="inherited", output_sha256="2" * 64)
    registry = {"lanes": {LANE: {"goldens": {"flux_dev_t2i": {"tag": "v0.43.0",
                                                               "sha": "2" * 64}}}}}
    # A head that still lists the chain: an accepted change must not repeat it.
    first_bad = {"flux_dev_t2i": {"commit": FIRST, "run_ts": 1794000000, "kind": "fail",
                                  "prev_good": "9" * 40, "pr": 16488, "author": "kijai",
                                  "golden": "v0.42.0", "runs": 12}}
    out = _report(_drift_files(cell, first_bad, current_tag="v0.43.0", registry=registry))
    lines = out.split("\n")
    assert lines[0].startswith("## ✅ ")
    assert ("No failures; 1 of 2 workflows match their reference. 1 changed output accepted as "
            "the reference after the run.") in lines
    # Nothing to act on: the block is not under "What needs attention".
    assert "### What needs attention" not in lines
    block = _section(out, "Accepted after the run")
    assert block[2] == "#### 🟡 `flux_dev_t2i`: output changed (accepted)"
    assert any("so nothing is open" in line for line in block)
    assert not any("ailing since" in line for line in block)
    assert any(line.startswith("| `flux_dev_t2i` | 🟡 output changed (accepted) |")
               for line in lines)
    # The history row of the commit agrees with the title.
    assert any("**this report**" in line and "| ✅ pass | `flux_dev_t2i` output changed "
               "(accepted) |" in line for line in lines)
    assert "gh workflow run" not in out
    # The same output on a lane whose reference is something else stays open.
    other = {"lanes": {LANE: {"goldens": {"flux_dev_t2i": {"tag": "v0.43.0", "sha": "3" * 64}}}}}
    assert "output changed (inherited)" in _report(_drift_files(cell, registry=other))
    # Alone, it is not "0 of 1 workflow match".
    out = _report(_tree(_summary([cell]), registry=registry))
    assert "No failures. 1 changed output accepted as the reference after the run." in out


def test_a_failure_against_the_reference_it_shares_a_first_frame_with_is_not_accepted():
    """The output sha is the first PNG alone. A workflow with several frames can
    fail on a later one while its first frame is the reference's: only a
    reference approved since the run (another tag) makes the change accepted."""
    vs_golden = {**DIFF, "frames_compared": 4, "frame_count_ref": 4, "frame_count_cand": 5}
    cell = _cell("vhs_video_roundtrip", "fail", detail="new_drift", vs_golden=vs_golden,
                 output_sha256="7" * 64)
    registry = {"lanes": {LANE: {"goldens": {"vhs_video_roundtrip": {"tag": "v0.42.0",
                                                                      "sha": "7" * 64}}}}}
    files = _tree(_summary([cell, _cell("sdxl_t2i")]), registry=registry)
    out = _report(files)
    lines = out.split("\n")
    assert lines[0].startswith("## ❌ ")
    assert "1 of 2 workflows failed, new at this commit." in lines
    assert "#### ❌ `vhs_video_roundtrip`: output changed (new at this commit)" in lines
    assert "**5 here, 4 in the reference**" in out and "gh workflow run" in out
    assert "output changed (accepted)" not in out and "nothing is open" not in out
    # History row and overview say the same.
    assert any("**this report**" in line and "| ❌ 1 of 2 failed |" in line for line in lines)
    overview = run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA,
                                          "master")
    assert "| ❌ 1 of 2 failed |" in overview and "(accepted)" not in overview
    # A summary or an index cell that does not say which reference it was
    # compared with is not taken for accepted either.
    assert run_report.cell_state("fail", None, "7" * 64, None,
                                 {"tag": "v0.43.0", "sha": "7" * 64}) == "changed"
    assert run_report.cell_state("fail", None, "7" * 64, "v0.42.0",
                                 {"tag": "v0.43.0", "sha": "7" * 64}) == "accepted"
    assert run_report.cell_state("fail", None, "7" * 64, "v0.42.0", {"sha": "7" * 64}) == "changed"


def test_accepted_changes_get_a_heading_of_their_own_after_what_is_open():
    cells = [_cell("flux_dev_t2i", "fail", detail="inherited", output_sha256="2" * 64),
             _cell("new_wf", "no_baseline")]
    registry = {"lanes": {LANE: {"goldens": {"flux_dev_t2i": {"tag": "v0.43.0",
                                                               "sha": "2" * 64}}}}}
    lines = _report(_tree(_summary(cells), registry=registry)).split("\n")
    headings = [line for line in lines if line.startswith(("### ", "#### "))]
    assert headings[:4] == ["### What needs attention", "#### ⚪ `new_wf`: no reference yet",
                            "### Accepted after the run",
                            "#### 🟡 `flux_dev_t2i`: output changed (accepted)"]
    # One blank line between the two parts, as between any two blocks.
    at = lines.index("### Accepted after the run")
    assert lines[at - 1] == "" and lines[at - 2] != ""


def test_retired_lane_offers_no_commands():
    cell = _cell("flux_dev_t2i", "fail", detail="new_drift")
    out = _report(_tree(_summary([cell], lane=LEGACY), lane=LEGACY), lane=LEGACY)
    assert "output changed (new at this commit)" in out and "gh workflow run" not in out


# -- infra errors and missing references -------------------------------------------------
@pytest.mark.parametrize("attempt, retry", [
    (1, "This was attempt 1 of 2: the cron tests the commit once more while it is still the "
        "newest commit on master."),
    (2, "This was attempt 2 of 2: the cron does not retry it. Dispatch **GPU regression** with "
        "this commit to test it again."),
    (3, "This was attempt 3: the cron does not retry it."),
])
def test_infra_error_is_not_a_verdict_and_says_whether_the_cron_retries(attempt, retry):
    cells = [_cell("flux_dev_t2i"),
             _cell("sdxl_t2i", "infra_error", worker_status="missing_model")]
    runs = {"sdxl_t2i": _run("sdxl_t2i", "missing_model", attempt=2, error={
        "message": "models absent from volume",
        "missing": ["checkpoints/sd_xl_base_1.0.safetensors"]})}
    out = _report(_tree(_summary(cells, attempt=attempt), runs=runs))
    lines = out.split("\n")
    assert lines[0].startswith("## ⚠️ GPU regression · ")  # not red
    assert ("No failures; 1 workflow hit infrastructure errors (not a verdict on the commit)."
            in lines)
    assert "#### ⚠️ `sdxl_t2i`: infra error" in lines
    assert ("An infrastructure problem stopped this workflow. Worker status: "
            f"<code>missing\\_model</code>. This is not a verdict on the commit. {retry}") in out
    assert "- **Message:** models absent from volume" in lines
    assert ("- **Missing models:** <code>checkpoints/sd\\_xl\\_base\\_1.0.safetensors</code>"
            in lines)
    assert "RunPod attempt 2" in out
    assert "| `sdxl_t2i` | ⚠️ infra error |" in out
    # Here the attempt tells something, so the Tested row names it.
    assert f"· {retry[9:].split(':')[0]} · [Actions run]({RUN_URL}) |" in out


def test_no_reference_says_how_to_make_one():
    cells = [_cell("flux_dev_t2i"), _cell("new_wf", "no_baseline")]
    out = _report(_tree(_summary(cells)))
    lines = out.split("\n")
    # The title carries the icon of the commit's own row in the history table.
    assert lines[0].startswith("## ⚪ ")
    assert any("**this report**" in line and "| ⚪ 1 of 2 no reference yet |" in line
               for line in lines)
    assert ("No failures; 1 of 2 workflows match their reference. 1 workflow with no reference "
            "yet.") in lines
    assert "#### ⚪ `new_wf`: no reference yet" in lines
    assert ("No reference is approved for this workflow on this lane yet, so the output was "
            "not compared with anything.") in lines
    assert ("gh workflow run golden-baseline.yml --repo Comfy-Org/comfyci-runpod-worker "
            f"-f ref=RELEASE_TAG -f workflows=new_wf -f lane={LANE} -f bless=false") in lines
    assert "| `new_wf` | ⚪ no reference yet |" in out


# -- skipped runs and missing results ------------------------------------------------------
def test_skipped_run_shows_the_published_result_and_who_tested_it():
    out = _report(_tree(_summary([_cell("flux_dev_t2i")])), skipped=True)
    lines = out.split("\n")
    assert lines[2:4] == ["> [!NOTE]", (
        "> This run did not test this commit again: the lane already has a result for it, "
        f"tested 2026-11-15 00:30 UTC in [an earlier run]({RUN_URL}). The report below shows "
        "that result.")]
    assert "The workflow matches its reference." in lines
    assert "### All workflows (1)" in lines
    # A summary from before the run link existed: the start marker has it.
    summary = _summary([_cell("flux_dev_t2i")], run_url=None, run_id=None)
    marker = {"commit": SHA, "branch": "master", "lane": LANE, "attempt": 1,
              "started_ts": 1794700000, "run_url": RUN_URL.replace("4242", "77")}
    files = _tree(summary, head=_head([]),
                  extra={f"{P}/runs/master/{SHA}/attempt.json": marker})
    assert f"in [an earlier run]({RUN_URL.replace('4242', '77')})" in _report(files, skipped=True)


def test_skipped_run_with_only_a_start_marker_stops_after_saying_so():
    marker = {"commit": SHA, "branch": "master", "lane": LANE, "attempt": 2,
              "started_ts": 1794700000, "run_url": RUN_URL}
    files = {f"{P}/runs/master/{SHA}/attempt.json": marker,
             f"regression/index/master/{LANE}.json": _head([_entry(PREV, 2, {"wf": "pass"})])}
    out = _report(files, skipped=True)
    lines = out.split("\n")
    assert lines[0] == ("## ➖ GPU regression · py3.13 / torch 2.14.1 / cu130 / RTX 4090 · "
                        "`cccccccc`")
    assert "This run did not test this commit, and the lane has no published result for it." \
        in lines
    assert ("The last start marker on the results branch: attempt 2 of 2, started 2026-11-14 "
            f"23:46 UTC, [the run that started it]({RUN_URL}).") in lines
    assert ("The cron stops after 2 attempts. Dispatch **GPU regression** with this commit to "
            "test it again.") in lines
    for absent in ("All workflows", "What needs attention", "Recent commits", "Result labels"):
        assert absent not in out


def test_no_summary_and_no_marker():
    out = _report({}, skipped=True)
    assert "There is no start marker for it either." in out
    out = _report({})
    assert ("No result is published for this commit on this lane: the run stopped before it "
            "wrote a summary.") in out


# -- old and broken data -------------------------------------------------------------------
def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_summary_from_before_schema_2_renders():
    """A real 2026-09 summary: no lane, env, commit_meta, detail or figures.
    The index classified it since, and knows the commit."""
    summary = _fixture("prev2_drift_summary.json")
    commit = summary["commit"]
    assert "schema_version" not in summary and "lane" not in summary
    run_dir = f"regression/runs/master/{commit}"
    entry = _entry(commit, summary["run_ts"],
                   {"flux_dev_t2i": {"v": "fail", "d": "new_drift", "sha": "3" * 64},
                    "sdxl_t2i": "pass"}, pr=16488, subject="Port some optimizations")
    files = {f"{run_dir}/summary.json": summary,
             f"{run_dir}/flux_dev_t2i/run.json": _fixture("prev2_drift_flux_run.json"),
             f"{run_dir}/flux_dev_t2i/figures/side_by_side_golden.png": None,
             f"regression/index/master/{LEGACY}.json": _head([entry])}
    out = _report(files, lane=LEGACY, commit=commit)
    lines = out.split("\n")
    assert "1 of 2 workflows failed, new at this commit." in lines
    assert "#### ❌ `flux_dev_t2i`: output changed (new at this commit)" in lines
    assert any(line.startswith(f"| Commit | [`{commit[:12]}`]({COMFY}/commit/{commit}) · "
                               f"Port some optimizations (\\#{ZW}16488) · by kijai")
               for line in lines)
    # No attempt and no run link in a summary that old.
    assert "| Tested | 2026-09-23 03:24 UTC · 10 min after the commit |" in lines
    assert "figures/side_by_side_golden.png\" width=\"720\"" in out
    # The output sha comes from run.json, the PNG stands in for the thumbnail.
    assert "`3fa6e2ad80aa`" in out
    assert "RunPod" not in out
    # Without the index: still a report, the change simply is not classified.
    del files[f"regression/index/master/{LEGACY}.json"]
    out = _report(files, lane=LEGACY, commit=commit)
    assert "#### ❌ `flux_dev_t2i`: output changed" in out and "1 of 2 workflows failed." in out
    assert "Recent commits" not in out


def test_old_infra_record_without_detail_renders():
    """A real record from before schema 2: error null, timings null, and the
    queue delay in milliseconds under delay_s."""
    summary = _fixture("prev2_infra_summary.json")
    commit = summary["commit"]
    run_dir = f"regression/runs/master/{commit}"
    out = _report({f"{run_dir}/summary.json": summary,
                   f"{run_dir}/sdxl_t2i/run.json": _fixture("prev2_infra_sdxl_run.json")},
                  lane=LEGACY, commit=commit)
    assert "#### ⚠️ `sdxl_t2i`: infra error" in out
    assert "Worker status: <code>checkout\\_error</code>." in out
    assert "The lane is retired, so nothing runs it again." in out
    assert "- The worker recorded no detail for this failure." in out
    assert "queued 0.1 s, ran 0.9 s, RunPod attempt 2" in out


@pytest.mark.parametrize("summary", [
    {"commit": SHA},
    {"commit": SHA, "workflows": []},
    {"commit": SHA, "workflows": {"wf": None, "other": "junk", "third": {}}},
    {"commit": SHA, "workflows": {"wf": {"verdict": "fail", "vs_golden": "x", "vs_previous": 7,
                                         "timings": 5, "custom_nodes": [1], "golden_tag": 9,
                                         "thresholds_used": "none", "thumbnail": 3,
                                         "output_sha256": 12, "previous_commit": ["x"]}}},
    {"commit": SHA, "commit_meta": "x", "tested_range": 3, "env": [], "run_ts": "soon",
     "attempt": "two", "run_url": 5, "run_id": "9",
     "workflows": {"wf": {"verdict": "new_kind", "vram_peak_mb": "big", "rss_peak_mb": None}}},
    {"commit": SHA, "env": {"runtime_backends": "cuda", "python": 3, "image_sha": 7},
     "tested_range": {"prev": 5, "commits_between": "many"},
     "commit_meta": {"subject": None, "pr": "17754", "committed_ts": "then"},
     "workflows": {"wf": {"verdict": "pass", "vs_golden": {"identical": False},
                          "vs_previous": {"error": 5}, "timings": {"prompt_exec_s": "fast"}}}},
])
def test_summaries_with_missing_or_odd_keys_never_crash(summary):
    summary = {"lane": LANE, **summary}
    junk_run = {"status": 5, "error": 7, "outputs": [1, {"filename": 5}, {"bytes": "x"}],
                "validation": 3, "attempt": "x", "delay_s": "soon"}
    head = {"count": "many", "first_bad": {"wf": 5, "other": {"commit": 9}},
            "entries": [1, {"c": 5}, {"c": SHA, "w": 3, "m": 7, "t": "x", "range": 2},
                        {"c": PREV, "w": {"wf": 1, "other": {"v": 3}}, "m": {"pr": "x"}}]}
    files = {f"{P}/runs/master/{SHA}/summary.json": summary,
             f"{P}/runs/master/{SHA}/attempt.json": {"attempt": "x", "run_url": 5},
             f"{P}/runs/master/{SHA}/wf/run.json": junk_run,
             f"regression/index/master/{LANE}.json": head,
             "regression/index/lanes.json": {"lanes": {LANE: {"goldens": {"wf": 5}}}},
             f"{P}/golden/wf/current.json": {"tag": 5, "blessed_ts": "x", "reason": 7}}
    for kw in ({}, {"skipped": True}, {"manifest": {"workflows": {"a": 5, "b": {"enabled": 1,
                                                                             "lanes": "x"}}},
                                       "packs": {"packs": 5}}):
        out = _report(files, **kw)
        assert out.startswith("## ") and "| Lane | " in out
        assert run_report.byte_size(out) < run_report.MAX_BYTES
    if "new_kind" in json.dumps(summary):
        assert "| `wf` | ❔ unknown |" in out


HUGE = 10 ** 400  # valid JSON, and too large for a float


@pytest.mark.parametrize("where, key, value", [
    # A list or an object where a word belongs: not usable as a dict key.
    ("index_cell", "v", []), ("index_cell", "d", {"a": 1}), ("index_cell", "sha", [1]),
    ("index_cell", "g", []), ("summary_cell", "verdict", []), ("summary_cell", "detail", {}),
    ("summary_cell", "golden_tag", {}), ("first_bad", "kind", []),
    # Integers no float can hold: every number is formatted or divided somewhere.
    ("first_bad", "runs", HUGE), ("first_bad", "run_ts", HUGE), ("head", "count", HUGE),
    ("summary_cell", "vram_peak_mb", HUGE), ("summary_cell", "rss_peak_mb", -HUGE),
    ("summary", "run_ts", HUGE), ("summary", "attempt", HUGE), ("error", "timeout_s", HUGE),
    ("error", "returncode", HUGE), ("run", "execution_ms", HUGE), ("run", "delay_ms", HUGE),
    ("run", "attempt", HUGE), ("marker", "attempt", HUGE), ("marker", "started_ts", HUGE),
    ("timings", "prompt_exec_s", HUGE), ("vs_golden", "mean_mse", HUGE),
    ("vs_golden", "frame_count_cand", HUGE), ("output", "bytes", HUGE),
    # A lone surrogate, as Python writes a file name with an undecodable byte:
    # valid JSON that UTF-8 cannot encode.
    ("error", "message", "no such file: x\udcff.png"), ("error", "log_tail", "a\ud800b\n"),
    ("error", "traceback", ["Traceback:\n", "  \udfff\n"]), ("commit_meta", "subject", "x\udc80"),
    ("output", "filename", "x\udcff.png"),
])
def test_one_wrongly_typed_or_oversized_value_never_replaces_the_report(where, key, value):
    """Any one of these used to turn the whole report into the 'could not be
    built' note, for as long as the value stayed among the newest entries."""
    cells = [_cell("flux_dev_t2i", "fail", detail="new_drift", vs_golden=dict(DIFF)),
             _cell("wan", "execution_error")]
    run = _run("wan", "execution_error", error={"message": "boom", "timeout_s": 900})
    flux_run = _run("flux_dev_t2i")
    marker = {"commit": SHA, "attempt": 1, "started_ts": 1794700000, "run_url": RUN_URL}
    first_bad = {"wan": {"commit": SHA, "run_ts": 1794702636, "kind": "execution_error",
                         "prev_good": PREV, "pr": 17754, "author": "kijai", "runs": 1}}
    summary = _summary(cells)
    files = _tree(summary, runs={"wan": run, "flux_dev_t2i": flux_run},
                  extra={f"{P}/runs/master/{SHA}/attempt.json": marker})
    head = files[f"regression/index/master/{LANE}.json"]
    head["first_bad"] = first_bad
    target = {"index_cell": head["entries"][1]["w"]["flux_dev_t2i"], "first_bad": first_bad["wan"],
              "summary_cell": cells[0], "head": head, "summary": summary, "error": run["error"],
              "run": run, "marker": marker, "timings": cells[0]["timings"],
              "vs_golden": cells[0]["vs_golden"], "output": flux_run["outputs"][0],
              "commit_meta": summary["commit_meta"]}[where]
    if where == "index_cell":
        target["v"] = "fail"  # the first three only matter on a failing cell
    target[key] = value
    files = json.loads(json.dumps(files))  # as it would be read from the branch
    for out in (_report(files), _report(files, skipped=True),
                run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA,
                                           "master")):
        assert out.startswith("## ") and "report not available" not in out
        assert "### Recent commits" in out
        out.encode("utf-8")  # what writing the summary does


def test_json_nested_too_deep_to_parse_reads_as_absent(tmp_path):
    deep = tmp_path / "summary.json"
    deep.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    assert run_report.read_json_file(deep) is None
    out = _report(_tree(_summary([_cell("flux_dev_t2i")])), summary_file=str(deep), skipped=True)
    assert "The workflow matches its reference." in out


# -- workflows that did not run -------------------------------------------------------------
MANIFEST = {"workflows": {
    "flux_dev_t2i": {"enabled": True, "lanes": None},
    "sdxl_t2i": {"enabled": True},
    "kjnodes_sdxl_t2i": {"enabled": True, "lanes": [OTHER]},
    "vhs_video_roundtrip": {"enabled": True, "lanes": [OTHER]},
    "staged_wf": {"enabled": False}}}


def test_workflows_not_run_by_lane_filter_and_by_partial_dispatch():
    summary = _summary([_cell("flux_dev_t2i")])
    # The manifest the run used was snapshotted next to its results.
    snapshot = {f"{P}/manifest-snapshot/{SHA}.json": MANIFEST}
    section = _section(_report(_tree(summary, extra=snapshot)), "Workflows not run on this lane")
    assert section[2:4] == [
        f"- ➖ not run, the manifest runs them only on `{OTHER}`: `kjnodes_sdxl_t2i`, "
        "`vhs_video_roundtrip`",
        "- ➖ not run, on this lane but not part of this run (a dispatch can name only some "
        "workflows): `sdxl_t2i`"]
    assert "staged_wf" not in "\n".join(section)
    # No snapshot: today's manifest, which may be newer than the run.
    section = _section(_report(_tree(summary), manifest=MANIFEST),
                       "Workflows not run on this lane")
    assert section[3] == (
        "- ➖ not run, on this lane in today's manifest but without a result in this run (a "
        "dispatch can name only some workflows, and a run can be older than a workflow): "
        "`sdxl_t2i`")
    # Everything the lane runs ran: no section.
    full = _summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i")])
    manifest = {"workflows": {k: v for k, v in MANIFEST["workflows"].items() if "lanes" not in v
                              or v["lanes"] is None}}
    assert "not run on this lane" not in _report(_tree(full), manifest=manifest)


# -- schema notes and custom-node packs ------------------------------------------------------
def test_schema_notes_list_what_the_worker_changed():
    runs = {"sdxl_t2i": _run("sdxl_t2i", validation={
        "missing_nodes": [], "stripped_inputs": ["KSampler.new_knob=1.0"],
        "filled_defaults": ["VAEDecode.tile=64", "SaveImage.x='a|b'"], "ok": True})}
    out = _report(_tree(_summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i")]), runs=runs))
    assert _section(out, "Schema notes")[4:] == [
        "- `sdxl_t2i`: inputs not in this commit's schema, removed: "
        "<code>KSampler.new\\_knob=1.0</code>",
        "- `sdxl_t2i`: missing inputs filled from the schema's defaults: "
        "<code>VAEDecode.tile=64</code>, <code>SaveImage.x='a&#124;b'</code>", ""]
    assert "Schema notes" not in _report(_tree(_summary([_cell("flux_dev_t2i")])))


def test_custom_node_packs_table():
    sha = "e27a505b3ba6ce42687fe00500deda103d9d6071"
    cell = _cell("kjnodes_sdxl_t2i", custom_nodes={
        "ComfyUI-KJNodes": {"import": "ok", "install_s": 0.03, "kind": "git", "pip_s": 0,
                            "seconds": 0.1, "sha": sha, "source": "cache"},
        "Unlisted-Pack": {"import": "failed", "sha": "f" * 40, "source": "fetched",
                          "seconds": None}})
    packs = {"packs": {"ComfyUI-KJNodes": {"repo": "https://github.com/kijai/ComfyUI-KJNodes",
                                           "sha": sha, "label": "KJNodes 1.4.7"}}}
    section = _section(_report(_tree(_summary([cell])), packs=packs), "Custom-node packs")
    assert section[4:6] == [
        "| `kjnodes_sdxl_t2i` | <code>ComfyUI-KJNodes</code> (KJNodes 1.4.7) | ok | "
        f"[`e27a505`](https://github.com/kijai/ComfyUI-KJNodes/tree/{sha}) | cache | 0.03 | 0 "
        "| 0.1 |",
        "| `kjnodes_sdxl_t2i` | <code>Unlisted-Pack</code> | **failed** | `fffffff` | fetched "
        "|  |  |  |"]
    # A repo that is not a plain GitHub repository is never linked: the host
    # counts, not only the owner/name shape.
    for repo in ("https://evil.example/kijai/ComfyUI-KJNodes",
                 "https://github.com.evil.example/kijai/ComfyUI-KJNodes",
                 "https://github.com/kijai/ComfyUI-KJNodes/../../evil"):
        packs["packs"]["ComfyUI-KJNodes"]["repo"] = repo
        out = _report(_tree(_summary([cell])), packs=packs)
        assert "evil" not in out and "| `e27a505` | cache |" in out
    assert "Custom-node packs" not in _report(_tree(_summary([_cell("flux_dev_t2i")])))


# -- recent history -------------------------------------------------------------------------
def test_history_lists_the_newest_commits_and_open_failures():
    entries = [_entry(f"{i:040x}", 1794000000 + i * 3600, {"flux_dev_t2i": "pass"},
                      pr=17000 + i) for i in range(1, 26)]
    entries[23] = _entry(entries[23]["c"], entries[23]["t"], {
        "flux_dev_t2i": {"v": "fail", "d": "inherited", "sha": "5" * 64, "g": "v0.42.0"},
        # The reference since: approved under another tag than the run was compared with.
        "sdxl_t2i": {"v": "fail", "d": "new_drift", "sha": "2" * 64, "g": "v0.42.0"},
        "wan": "execution_error", "ltx": "infra_error"}, run=991, n=3, pr=17024)
    # An old commit tested again just now: its run time is the newest of all,
    # its commit time is not, and commit time is what orders history.
    retested = entries[3]["c"]
    entries[3] = {**entries[3], "t": 1794000000 + 30 * 3600}
    this = _entry(SHA, 1794000000 + 26 * 3600, {"flux_dev_t2i": "pass"}, run=4242)
    first_bad = {"wan": {"commit": entries[23]["c"], "kind": "execution_error", "runs": 1,
                         "prev_good": entries[22]["c"], "pr": 17024, "author": "kijai"}}
    registry = {"lanes": {LANE: {"goldens": {"sdxl_t2i": {"tag": "v0.43.0", "sha": "2" * 64}}}}}
    # Stored in no particular order: the report sorts by commit time.
    head = {**_head(entries + [this], first_bad), "count": 136}
    out = _report(_tree(_summary([_cell("flux_dev_t2i")]), head=head, registry=registry))
    section = _section(out, "Recent commits on this lane")
    bad = entries[23]["c"]
    # The chain started in a run of three commits: no single pull request is named.
    assert section[2:5] == ["Open failures on this lane:", "", (
        f"- `wan`: execution error, failing since [`{bad[:7]}`]({COMFY}/commit/{bad}) "
        "(a run that covered 3 commits), 1 run, last passed at "
        f"[`{entries[22]['c'][:7]}`]({COMFY}/commit/{entries[22]['c']})")]
    rows = [line for line in section if line.startswith("| [`")]
    assert len(rows) == 20
    assert not any(f"/commit/{retested})" in row for row in rows)  # still the 23rd newest
    assert rows[0] == (f"| [`{SHA[:7]}`]({COMFY}/commit/{SHA}) **this report** | Add a node "
                       f"(\\#{ZW}17754) | [#17754]({COMFY}/pull/17754) | kijai | "
                       f"[2026-11-07 23:20]({RUN_URL}) | ✅ pass |  |")
    assert rows[2] == (
        f"| [`{bad[:7]}`]({COMFY}/commit/{bad}) +3 commits | Add a node (\\#{ZW}17024) | "
        f"[#17024]({COMFY}/pull/17024) | kijai | [2026-11-07 21:20](https://github.com/Comfy-Org/"
        "comfyci-runpod-worker/actions/runs/991) | ❌ 2 of 4 failed | `flux_dev_t2i` output "
        "changed (inherited); `sdxl_t2i` output changed (accepted); `wan` execution error; "
        "`ltx` infra error |")
    assert rows[-1].startswith(f"| [`{entries[6]['c'][:7]}`]")  # the 20th newest
    assert "The newest 20 of 136 runs on this lane." in section
    # A report about a commit that is not among them says so.
    assert "The commit of this report is not among them." in _report(
        _tree(_summary([_cell("flux_dev_t2i")]), head=_head(entries)))
    # No index head published for the lane: no section.
    files = _tree(_summary([_cell("flux_dev_t2i")]))
    del files[f"regression/index/master/{LANE}.json"]
    assert "Recent commits" not in _report(files)


def test_prior_result_comes_from_the_index_in_commit_order():
    """A summary without prior_verdict (an older one, or one rebuilt on the
    runner) takes it from the index: the newest red or green result before
    this commit. A later commit's result is not a prior one."""
    newer, older = "9" * 40, "8" * 40
    cell = _cell("wan", "execution_error", prior_verdict=None, prior_commit=None)
    entries = [_entry(newer, 1794800000, {"wan": "execution_error"}),
               _entry(SHA, 1794702636, {"wan": "execution_error"}),
               _entry(PREV, 1794700000, {"wan": "infra_error"}),  # looked through
               _entry(older, 1794600000, {"wan": "pass"})]
    out = _report(_tree(_summary([cell]), runs={"wan": _run("wan", "execution_error")},
                        head=_head(entries)))
    assert (f"Before this commit it last ran to the end at [`{older[:7]}`]({COMFY}/commit/{older}"
            ") (pass), so the error is new at this commit.") in out
    committed = 1794702029  # the summary's commit time
    assert run_report.prior_result(entries, SHA, "wan", {}, committed) == {
        "verdict": "pass", "commit": older}
    # The summary's own record wins over the index.
    assert run_report.prior_result(entries, SHA, "wan", {"prior_verdict": "fail",
                                                         "prior_commit": PREV}, committed) == {
        "verdict": "fail", "commit": PREV}


def test_an_older_commit_does_not_take_a_later_failing_chain_as_its_own():
    """head.first_bad is the chain that is open now. A report about an older
    red commit (a skipped tick, a report-only run, a re-test) is not part of a
    chain that started after it."""
    newer, between = "9" * 40, "8" * 40
    cells = [_cell("wan", "execution_error", prior_verdict=None, prior_commit=None),
             _cell("flux_dev_t2i", "fail", detail="inherited", previous_verdict="fail")]
    entries = [_entry(newer, 1794800000, {"wan": "execution_error", "flux_dev_t2i": "fail"},
                      pr=17900),
               _entry(between, 1794750000, {"wan": "pass", "flux_dev_t2i": "pass"}, pr=17800),
               _entry(SHA, 1794702636, {"wan": "execution_error", "flux_dev_t2i": "fail"})]
    first_bad = {wf: {"commit": newer, "run_ts": 1794800000, "kind": kind, "prev_good": between,
                      "pr": 17900, "author": "kijai", "runs": 1}
                 for wf, kind in (("wan", "execution_error"), ("flux_dev_t2i", "fail"))}
    later = f"{P}/runs/master/{newer}/flux_dev_t2i/figures/side_by_side_golden.png"
    files = _tree(_summary(cells), runs={"wan": _run("wan", "execution_error")},
                  head=_head(entries, first_bad), extra={later: None})
    out = _report(files)
    attention = "\n".join(_section(out, "What needs attention"))
    assert "ailing since" not in attention and newer[:7] not in attention
    # Without a chain or an earlier result, the error's origin is not guessed.
    assert "it is not known whether the error is new at this commit" in attention
    assert ("An inherited failure has no figures of its own. They are on the commit where the "
            "change started.") in attention
    # The lane's open failures are still listed, as what is open today.
    assert f"- `wan`: execution error, failing since [`{newer[:7]}`]" in out
    # Seen from the commit where the chain starts, or from a later one, it applies.
    for commit in (newer, "a" * 39 + "f"):
        summary = _summary(cells, commit=commit)
        summary["commit_meta"]["committed_ts"] = 1794800000 if commit == newer else 1794900000
        out = _report(_tree(summary, runs={"wan": _run("wan", "execution_error")},
                            head=_head(entries, first_bad), extra={later: None}), commit=commit)
        assert out.count(f"Failing since [`{newer[:7]}`]") == 2


# -- hostile data ----------------------------------------------------------------------------
CLIMB = "../../../../../../../../evil-org/evil-repo/main/x"


def _hostile_files():
    cells = [
        _cell("ok_wf", golden_tag=EVIL, gpu_name=EVIL, comfy_version=EVIL, thumbnail="../x.webp",
              custom_nodes={EVIL: {"import": EVIL, "sha": EVIL, "source": EVIL}}),
        _cell("broken_wf", "execution_error", worker_status=EVIL),
        _cell("drift_wf", "fail", detail="new_drift", vs_previous={"error": EVIL}),
        _cell(EVIL, "no_baseline", thumbnail=EVIL),
        # An id that climbs out of the run directory, on the path that shows figures.
        _cell(CLIMB, "fail", detail="new_drift", figures=True, thumbnail="x.webp")]
    summary = _summary(
        cells, env={**ENV, "driver_version": EVIL, "torch": EVIL, "image": EVIL,
                    "runtime_backends": [f"\x1b[32m[INFO]\x1b[0m {EVIL}"]},
        commit_meta={"subject": EVIL, "author": EVIL, "committed_ts": 1794702029, "pr": 17754,
                     "pr_title": EVIL, "pr_author": "@octocat",
                     "parents": [EVIL, "../../x", "javascript:alert(1)"]},
        tested_range={"prev": PREV, "commits_between": 3, "compare_url": "javascript:alert(1)"},
        # The right shape on the wrong host: only github.com is ever linked.
        run_url="https://evil.example/o/r/actions/runs/1", run_id=None)
    runs = {
        "ok_wf": _run("ok_wf", outputs=[
            {"filename": EVIL, "bytes": 5, "sha256": EVIL},
            {"filename": "../../../etc/passwd", "bytes": 5, "sha256": "4" * 64},
            {"filename": "ok_wf_00001_.png", "bytes": 100, "sha256": "1" * 64}],
            validation={"missing_nodes": [EVIL], "stripped_inputs": [EVIL],
                        "filled_defaults": [EVIL], "ok": True}),
        "broken_wf": _run("broken_wf", "execution_error", runpod_job_id=EVIL, error={
            "message": EVIL, "node_type": EVIL, "node_id": EVIL, "detail": EVIL,
            "missing": [EVIL], "custom_nodes": {EVIL: {"import": EVIL, "seconds": 1}},
            "startup_failure": EVIL, "traceback": [f"Traceback:\n  {EVIL}\n"],
            "log_tail": f"</pre></details>\n{EVIL}\n"})}
    head = _head(
        [{**_entry(SHA, 3, {"ok_wf": "pass", "broken_wf": "execution_error", EVIL: EVIL}),
          "m": {"s": EVIL, "a": EVIL, "pr": "17754) [x](https://evil.example", "ct": 2},
          "run": "1) [x](https://evil.example", "range": {"n": 3, "url": "javascript:alert(1)"}},
         {**_entry(PREV, 2, {"ok_wf": "pass"}, run=7), "run_repo": "evil.example/x/y) [z]("}],
        {"broken_wf": {"commit": SHA, "kind": EVIL, "prev_good": EVIL, "pr": EVIL,
                       "author": EVIL, "runs": 1, "golden": EVIL},
         EVIL: {"commit": EVIL, "kind": "fail", "author": EVIL}})
    return _tree(summary, runs=runs, head=head, extra={
        f"{P}/golden/drift_wf/current.json": {"tag": EVIL, "blessed_by": "@octocat",
                                              "blessed_ts": 1792378306, "reason": EVIL},
        f"{P}/runs/master/{SHA}/attempt.json": {
            "attempt": 1, "run_url": "https://github.com.evil.example/o/r/actions/runs/1"}})


def test_hostile_strings_never_reach_the_summary_unescaped():
    files = _hostile_files()
    manifest = {"workflows": {EVIL: {"enabled": True, "lanes": [EVIL]},
                              "<b>wf</b>": {"enabled": True}}}
    out = _report(files, manifest=manifest, packs={"packs": {EVIL: {"repo": EVIL, "sha": EVIL,
                                                                     "label": EVIL}}})
    md = _outside_pre(out)
    assert EVIL.split("\n")[0][:20] not in md  # the raw text is nowhere outside <pre>
    # HTML: no tag the data brought, inside or outside <pre>.
    assert "<script" not in out and "</pre></details>\n@" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in md
    tags = set(re.findall(r"</?([a-zA-Z0-9]+)", out))
    assert tags <= {"a", "img", "code", "pre", "details", "summary"}
    # Markdown: pipes, backticks, emphasis, links and headings are inert.
    assert "&#124; pipe" in md and " | pipe" not in md
    assert "\\`tick\\`" in md and "`tick`" not in md
    assert "\\*\\*bold\\*\\*" in md and "**bold**" not in md and " _em_ " not in md
    assert "\\[x\\](https:" in md and "[x](https://evil.example" not in md
    assert not re.search(r"(?m)^#+ heading", md)
    # Mentions, issue references and URLs do not link: a zero-width space breaks each.
    assert f"@{ZW}octocat" in md and "@octocat" not in md
    assert f"\\#{ZW}4242" in md and "#4242" not in md
    assert f"GH{ZW}-7" in md and "GH-7" not in md
    assert f"https:{ZW}//evil.example/a" in md and f"www{ZW}.evil.example" in md
    assert f"{ZW}${ZW}a+b{ZW}${ZW}" in md
    # Every link the report built goes to a known host; nothing from the data is a target.
    targets = (re.findall(r'(?:src|href)="([^"]*)"', out)
               + re.findall(r"(?<!\\)\]\(([^)]*)\)", md))  # a real link: "]" not escaped
    assert targets
    for url in targets:
        assert re.match(r"https://(github\.com|raw\.githubusercontent\.com|ci\.comfy\.org)(/|$)",
                        url), url
    assert "javascript:" not in out and "/etc/passwd)" not in out and "../" not in "".join(targets)
    # The id that climbs out of the run directory builds no link, image or command.
    climb = "\n".join(_section(out, "What needs attention")).split(f"{CLIMB}</code>: ", 1)[1]
    climb = climb.split("\n#### ", 1)[0]
    assert climb.startswith("output changed (new in this run)")
    assert "<img" not in climb and "githubusercontent" not in climb
    assert "gh workflow run" not in climb
    # Control characters and colour codes are gone; no line is a workflow command.
    assert "\x1b" not in out and "\x07" not in out
    assert not [line for line in out.split("\n") if line.startswith("::")]
    assert "&#58;:error::not a command" in out
    # Tables keep their shape: the hostile cell did not add columns.
    rows = _section(out, "All workflows")[4:9]
    assert len(rows) == 5 and all(row.count(" | ") == 10 for row in rows)


def test_hostile_strings_in_the_overview():
    out = run_report.overview_report(run_report.DictReader(_hostile_files(), RESULTS), LANES,
                                     SHA, "master")
    assert "<script" not in out and "@octocat" not in out and "#4242" not in out
    assert "javascript:" not in out and "](https://evil.example" not in out
    assert not [line for line in out.split("\n") if line.startswith("::")]
    assert set(re.findall(r"</?([a-zA-Z0-9]+)", out)) <= {"code"}


@pytest.mark.parametrize("text, expected", [
    ("plain words, 1.5 + 2", "plain words, 1.5 + 2"),
    ("a | b", "a &#124; b"),
    ("line one\nline two\ttabbed", "line one line two tabbed"),
    ("<img src=x onerror=1>", "&lt;img src=x onerror=1&gt;"),
    ("&lt;", "&amp;lt;"),
    ("`code` *em* _em_ ~~no~~ [l](u)", "\\`code\\` \\*em\\* \\_em\\_ \\~\\~no\\~\\~ \\[l\\](u)"),
    ("# title", f"\\#{ZW} title"),
    ("fix #123 for @user", f"fix \\#{ZW}123 for @{ZW}user"),
    ("see GH-12 and gh-3, not GH-x", f"see GH{ZW}-12 and gh{ZW}-3, not GH-x"),
    ("http://a.b/c www.x.y WWW.x", f"http:{ZW}//a.b/c www{ZW}.x.y WWW{ZW}.x"),
    ("cost $5 or $a$", f"cost {ZW}${ZW}5 or {ZW}${ZW}a{ZW}${ZW}"),
    ("\x1b[1;31mred\x1b[0m \x1b]8;;http://x\x07link\x1b]8;;\x07 \x00\x08bell", "red link bell"),
    # An OSC sequence that never closes keeps the text after it.
    ("prefix \x1b]8;;x and the real error", "prefix 8;;x and the real error"),
    # A lone surrogate cannot be written as UTF-8.
    ("no such file: x\udcff.png", "no such file: x.png"),
    # Zero-width space, right-to-left override, isolate: invisible, so by code point.
    (f"zero{chr(0x200B)}width {chr(0x202E)}flip{chr(0x2066)}", "zerowidth flip"),
    (None, ""), (12.5, "12.5"), (["a"], "\\['a'\\]"),
])
def test_md_escapes_inline_text(text, expected):
    assert run_report.md(text) == expected


def test_md_and_pre_helpers():
    assert run_report.md("x" * 400) == "x" * 299 + "…"
    assert run_report.md("ab", limit=None) == "ab"
    assert run_report.code("a`b") == "<code>a\\`b</code>" and run_report.code(None) == ""
    assert run_report.pre("a <b>\n::warning::x\n::") == (
        "<pre>\na &lt;b&gt;\n&#58;:warning::x\n&#58;:\n</pre>")
    assert run_report.clean("a\r\nb\rc\td", lines=True) == "a\nb\nc    d"
    # A closed OSC sequence goes with its text (BEL or ESC \ ends it). One that
    # is never closed loses its two opening bytes, not the log lines after it.
    assert run_report.clean("a\x1b]0;t\x1b\\b \x1b]0;t\x07c", lines=True) == "ab c"
    tail = "loading\n\x1b]0;title\nTraceback (most recent call last):\nRuntimeError: CUDA\n"
    assert run_report.clean(tail, lines=True) == (
        "loading\n0;title\nTraceback (most recent call last):\nRuntimeError: CUDA\n")


@pytest.mark.parametrize("parts, expected", [
    (("regression", "runs", "master", SHA), f"regression/runs/master/{SHA}"),
    (("regression/lanes/x", "golden"), "regression/lanes/x/golden"),
    (("a", ".."), None), (("a", "."), None), (("a", ""), None), (("a//b",), None),
    (("a", "b\\c"), None), (("a", "b\nc"), None), (("a", 5), None), (("a", "x" * 201), None),
    (("a", "sub/file name.png"), "a/sub/file name.png"),
])
def test_join_path_only_builds_plain_paths(parts, expected):
    assert run_report.join_path(*parts) == expected


def test_links_are_built_only_from_checked_parts(monkeypatch):
    assert run_report.sha_link("zzz") == "<code>zzz</code>"
    assert run_report.sha_link("abcdef1") == f"[`abcdef1`]({COMFY}/commit/abcdef1)"
    assert run_report.pr_link("12") == "" and run_report.pr_link(True) == ""
    assert run_report.pr_link(-1) == "" and run_report.pr_link(12) == f"[#12]({COMFY}/pull/12)"
    assert run_report.compare_link(PREV, "x") == ""
    assert run_report.run_url("7") is None and run_report.run_url(0) is None
    assert run_report.checked_run_url(RUN_URL) == RUN_URL
    assert run_report.checked_run_url(RUN_URL + "/../x") is None
    # The host is pinned, not only the shape of the path.
    for url in ("http://github.com/a/b/actions/runs/1", "https://evil.example/a/b/actions/runs/1",
                "https://github.com.evil.example/a/b/actions/runs/1",
                "https://github.com@evil.example/a/b/actions/runs/1"):
        assert run_report.checked_run_url(url) is None
        assert not run_report.ARTIFACT_URL_RE.fullmatch(url + "/artifacts/2")
        assert not run_report.PACK_REPO_RE.fullmatch(url.split("/actions")[0])
    assert run_report.ARTIFACT_URL_RE.fullmatch(RUN_URL + "/artifacts/2")
    assert run_report.PACK_REPO_RE.fullmatch("https://github.com/kijai/ComfyUI-KJNodes")
    reader = run_report.DictReader({}, RESULTS)
    assert run_report.raw_url(reader, "a/b c/d#e?.png") == f"{RAW}/a/b%20c/d%23e%3F.png"
    assert run_report.raw_url(run_report.Reader(), "a") is None  # nothing to pin to
    # Runs link to the repository the workflow runs in; results to RESULTS_REPO.
    monkeypatch.setenv("GITHUB_REPOSITORY", "fork/worker")
    monkeypatch.setenv("RESULTS_REPO", "fork/results")
    assert run_report.run_url(7) == "https://github.com/fork/worker/actions/runs/7"
    assert run_report.raw_url(reader, "a").startswith(
        "https://raw.githubusercontent.com/fork/results/")
    # An index entry names the repository its run was in: the action also runs
    # from other repositories, and a bare id would link into the wrong one.
    assert run_report.entry_run_url({"run": 7, "run_repo": "Comfy-Org/ComfyUI"}) == (
        f"{COMFY}/actions/runs/7")
    assert run_report.entry_run_url({"run": 7}) == "https://github.com/fork/worker/actions/runs/7"
    assert run_report.entry_run_url({"run": 7, "run_repo": "x/y) [z](https://evil.example"}) == (
        "https://github.com/fork/worker/actions/runs/7")
    assert run_report.entry_run_url({"run": "7", "run_repo": "Comfy-Org/ComfyUI"}) is None
    monkeypatch.setenv("GITHUB_REPOSITORY", "evil.example/x/y) [z](")
    monkeypatch.setenv("RESULTS_REPO", "not a slug")
    assert run_report.run_url(7) == RUN_URL.replace("4242", "7")  # back to the default


def test_history_rows_link_each_run_in_the_repository_that_ran_it():
    entries = [_entry(SHA, 1794702636, {"flux_dev_t2i": "pass"}, run=4242),
               {**_entry(PREV, 1794700000, {"flux_dev_t2i": "pass"}, run=555, pr=17753),
                "run_repo": "Comfy-Org/ComfyUI"}]
    files = _tree(_summary([_cell("flux_dev_t2i")]), head=_head(entries))
    for out in (_report(files), _overview(files)):
        assert f"]({COMFY}/actions/runs/555)" in out
        assert "comfyci-runpod-worker/actions/runs/555" not in out
        assert f"]({RUN_URL})" in out


def test_worker_repository_links_survive_a_run_in_another_repository(monkeypatch):
    """The composite action runs in its caller's repository. GITHUB_REPOSITORY
    is ComfyUI there, which has no golden-baseline.yml and no image builds."""
    cells = [_cell("flux_dev_t2i", "fail", detail="new_drift"), _cell("new_wf", "no_baseline")]
    files = _tree(_summary(cells))
    monkeypatch.setenv("GITHUB_REPOSITORY", "Comfy-Org/ComfyUI")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.setenv("REPORT_WORKER_REPO", "comfy-org/comfyci-runpod-worker")
    out = _report(files)
    assert out.count("gh workflow run golden-baseline.yml --repo comfy-org/comfyci-runpod-worker "
                     ) == 4
    assert ("image build [`4029bd6`](https://github.com/comfy-org/comfyci-runpod-worker/commit/"
            f"{ENV['image_sha']})") in out
    assert "--repo Comfy-Org/ComfyUI" not in out and f"ComfyUI/commit/{ENV['image_sha']}" not in out
    # Runs are the caller's own: this one is linked there.
    assert run_report.this_run_url() == f"{COMFY}/actions/runs/99"
    # Not a slug: back to the repository the workflow runs in.
    monkeypatch.setenv("REPORT_WORKER_REPO", "x/y) [z](https://evil.example")
    assert "--repo Comfy-Org/ComfyUI " in _report(files)
    # The action names the repository it checks out.
    action = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert "repository: comfy-org/comfyci-runpod-worker" in action
    assert "REPORT_WORKER_REPO: comfy-org/comfyci-runpod-worker" in action


def test_dashboard_line_follows_the_environment(monkeypatch):
    files = _tree(_summary([_cell("flux_dev_t2i")]))
    monkeypatch.setenv("REPORT_DASHBOARD_URL", "https://ci.example.org/")
    # The commit's own page on the dashboard, next to the front page.
    assert ("Longer history and charts: [ci.example.org](https://ci.example.org) · [this commit "
            f"on this lane](https://ci.example.org/regression/master/{SHA}?lane={LANE})\n"
            ) in _report(files)
    for dropped in ("", "javascript:alert(1)", "https://x.example/) [y](https://z"):
        monkeypatch.setenv("REPORT_DASHBOARD_URL", dropped)
        assert "Longer history" not in _report(files)


# -- size limit -----------------------------------------------------------------------------
def _big_files(n, tail_chars=6000, files_per_wf=0, trace_lines=40):
    cells = [_cell(f"wf_{i:03d}", "execution_error") for i in range(n)]
    trace = [f"  File \"/comfyui/x.py\", line {i}, in step {'y' * 150}\n"
             for i in range(trace_lines)]
    runs = {c["workflow_id"]: _run(c["workflow_id"], "execution_error", error={
        "message": "boom", "traceback": trace, "log_tail": "z" * tail_chars},
        outputs=[{"filename": f"{c['workflow_id']}_{j:05d}_.png", "bytes": 1, "sha256": "1" * 64}
                 for j in range(files_per_wf)]) for c in cells}
    return _tree(_summary(cells), runs=runs)


def test_report_is_shortened_to_stay_under_the_size_limit():
    """GitHub drops a step summary over 1 MiB whole: the cap is a hard rule."""
    assert run_report.MAX_BYTES <= 900_000 < 1024 * 1024
    out = _report(_big_files(200, tail_chars=200_000, trace_lines=3))
    assert run_report.byte_size(out) <= run_report.MAX_BYTES
    # One oversized log tail is cut to what the worker would send anyway.
    assert "Server log tail (last 1500 of 200000 characters)" in out
    assert "Traceback:" in out  # tracebacks were not touched yet
    note = next(line for line in out.split("\n") if line.startswith("Shortened to stay under"))
    assert note == ("Shortened to stay under GitHub's size limit for a job summary: log tails "
                    "cut to their last 1500 characters. The linked files hold everything.")


def test_shortening_order_is_logs_then_tracebacks_then_files_then_history():
    seen = []

    def render(limits):
        seen.append(limits)
        return "x" * (10 if limits.history_rows == 0 else 100)

    assert run_report.fit(render, max_bytes=50) == "x" * 10
    order = [(s.log_chars, s.traceback_lines, s.files_per_workflow, s.history_rows) for s in seen]
    assert order == [(6000, 40, 200, 20), (1500, 40, 200, 20), (0, 40, 200, 20), (0, 5, 200, 20),
                     (0, 0, 200, 20), (0, 0, 3, 20), (0, 0, 0, 20), (0, 0, 0, 10), (0, 0, 0, 5),
                     (0, 0, 0, 0)]
    assert seen[-1].matrix_rows == 0
    assert run_report.Limits().shortened() == []
    assert seen[-1].shortened() == ["log tails left out", "tracebacks left out",
                                    "output file lists left out", "recent commits left out"]
    # Nothing fits: no text at all, never an oversized one.
    assert run_report.fit(lambda limits: "x" * 100, max_bytes=50) is None
    # Fits at once: rendered once, nothing shortened.
    seen.clear()
    assert run_report.fit(render, max_bytes=1000) == "x" * 100 and len(seen) == 1


def test_every_shortening_step_is_said_in_the_report():
    out = _report(_big_files(600))
    assert run_report.byte_size(out) <= run_report.MAX_BYTES
    assert "A server log tail was recorded; it is left out here to keep the summary small." in out
    assert "A traceback was recorded; it is left out here to keep the summary small." in out
    note = next(line for line in out.split("\n") if line.startswith("Shortened to stay under"))
    assert "log tails left out; tracebacks left out" in note


def test_the_size_limit_counts_utf8_bytes_not_characters():
    """GitHub's limit is in bytes, and every report has multi-byte characters
    (the result icons, the middle dots, the zero-width spaces)."""
    assert run_report.byte_size("✅ · é") == 3 + 1 + 2 + 1 + 2
    assert run_report.fit(lambda limits: "✅" * 40, max_bytes=119) is None  # 40 characters
    assert run_report.fit(lambda limits: "✅" * 40, max_bytes=120) == "✅" * 40
    out = _report(_tree(_summary([_cell("flux_dev_t2i")])))
    assert run_report.byte_size(out) > len(out)


def test_output_file_lists_are_cut_before_history_is():
    """A run with thousands of output files keeps three per workflow and says
    how many more there are, instead of ending as the 'does not fit' note."""
    out = _report(_big_files(20, tail_chars=10, files_per_wf=250, trace_lines=1))
    assert run_report.byte_size(out) <= run_report.MAX_BYTES
    lines = out.split("\n")
    assert "<details><summary>Output files (5000)</summary>" in lines
    assert "| `wf_000` | +247 more files, listed in run.json |  |  |  |" in lines
    assert len([line for line in lines if line.startswith("| `wf_000` | [wf")]) == 3
    note = next(line for line in lines if line.startswith("Shortened to stay under"))
    assert note.endswith("tracebacks left out; output file lists cut to 3 files per workflow. "
                         "The linked files hold everything.")
    assert "recent commits" not in note and "### Recent commits on this lane" in lines
    assert any(line.startswith("Run records (run.json): ") for line in lines)
    # At the full limits a list stops at 200 files, with the rest counted.
    reader = run_report.DictReader(_big_files(1, files_per_wf=250), RESULTS)
    d = run_report.gather_lane(reader, LANES, SHA, "master", LANE)
    full = run_report.render_lane(d, run_report.Limits()).split("\n")
    assert len([line for line in full if line.startswith("| `wf_000` | [wf")]) == 200
    assert "| `wf_000` | +50 more files, listed in run.json |  |  |  |" in full
    # Left out altogether: no list, and no run records either.
    none = run_report.render_lane(d, run_report.Limits(files_per_workflow=0))
    assert "Output files" not in none and "Run records" not in none


def test_history_and_matrix_rows_follow_the_limits():
    entries = [_entry(f"{i:040x}", 1794000000 + i * 3600, {"flux_dev_t2i": "pass"})
               for i in range(1, 31)]
    reader = run_report.DictReader(_tree(_summary([_cell("flux_dev_t2i")]), head=_head(entries)),
                                   RESULTS)
    d = run_report.gather_lane(reader, LANES, SHA, "master", LANE)
    for rows in (20, 10, 5):
        text = run_report.render_lane(d, run_report.Limits(history_rows=rows))
        section = _section(text, "Recent commits on this lane")
        assert len([line for line in section if line.startswith("| [`")]) == rows
        assert f"The newest {rows} of 30 runs on this lane." in text
    text = run_report.render_lane(d, run_report.Limits(history_rows=0))
    assert "The newest" not in text and "Open failures on this lane: none." in text
    o = run_report.gather_overview(reader, LANES, SHA, "master")
    for rows in (25, 10, 5):
        text = run_report.render_overview(o, run_report.Limits(matrix_rows=rows))
        assert len([line for line in _section(text, "Recent commits")
                    if line.startswith("| [`")]) == rows
    assert "### Recent commits" not in run_report.render_overview(
        o, run_report.Limits(matrix_rows=0))


def test_a_traceback_is_capped_in_characters_as_well_as_lines():
    trace = [f"  File x.py, line {i}, in step {'y' * 400}\n" for i in range(60)]
    cells = [_cell("wan", "execution_error")]
    out = _report(_tree(_summary(cells), runs={"wan": _run("wan", "execution_error", error={
        "message": "boom", "traceback": trace})}))
    assert "Traceback (last 40 of 60 lines):" in out
    shown = re.search(r"<pre>\n(.*?)\n</pre>", out, re.S).group(1)
    assert len(shown) == run_report.TRACEBACK_CHARS
    assert shown.endswith("line 59, in step " + "y" * 400)


def test_a_report_that_cannot_fit_becomes_a_short_note(tmp_path, monkeypatch, capsys):
    step = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    reader = run_report.DictReader(_big_files(6000, tail_chars=10), RESULTS)
    assert run_report.lane_report(reader, LANES, SHA, "master", LANE) is None
    assert run_report.main(["lane", "--commit", SHA, "--branch", "master", "--lane", LANE],
                           reader=reader) == 0
    text = step.read_text(encoding="utf-8")
    assert "The report could not be built (it does not fit GitHub's size limit)." in text
    assert run_report.byte_size(text) < 2000


# -- every lane -----------------------------------------------------------------------------
def _overview_files(with_legacy=True, other_summary=True):
    this = _summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i"), _cell("only_here")])
    other = _summary([_cell("flux_dev_t2i", "fail", detail="new_drift", output_sha256="2" * 64),
                      _cell("sdxl_t2i"), _cell("wan", "infra_error")], lane=OTHER, run_id=4343,
                     run_url=RUN_URL.replace("4242", "4343"), run_ts=1794702700)
    old = "8" * 40
    heads = {
        LANE: _head([_entry(SHA, 30, {"flux_dev_t2i": "pass", "sdxl_t2i": "pass"}, run=4242),
                     _entry(PREV, 20, {"flux_dev_t2i": "pass"}, pr=17753),
                     _entry(old, 10, {"flux_dev_t2i": "execution_error"}, pr=17000)],
                    {"sdxl_t2i": {"commit": old, "kind": "fail", "runs": 4, "pr": 17000,
                                  "author": "kijai", "prev_good": FIRST}}),
        OTHER: _head([_entry(SHA, 31, {"flux_dev_t2i": {"v": "fail", "d": "new_drift",
                                                         "sha": "2" * 64}, "sdxl_t2i": "pass"},
                             run=4343),
                      _entry(old, 11, {"flux_dev_t2i": "execution_error"}, pr=17000)])}
    files = {f"{P}/runs/master/{SHA}/summary.json": this,
             f"regression/index/master/{LANE}.json": heads[LANE],
             f"regression/index/master/{OTHER}.json": heads[OTHER],
             f"regression/index/master/{LEGACY}.json": _head([_entry(SHA, 30, {"x": "fail"})]),
             "regression/index/lanes.json": {"lanes": {LANE: {"goldens": {}},
                                                       OTHER: {"goldens": {}}}}}
    if other_summary:
        files[f"regression/lanes/{OTHER}/runs/master/{SHA}/summary.json"] = other
    if with_legacy:
        files[f"regression/runs/master/{SHA}/summary.json"] = _summary(
            [_cell("sdxl_t2i")], lane=LEGACY, run_url=None, run_id=None)
    return files


def _overview(files):
    return run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA, "master")


def test_overview_puts_every_lane_side_by_side():
    out = _overview(_overview_files())
    lines = out.split("\n")
    main, other = LANES["lanes"][LANE]["label"], LANES["lanes"][OTHER]["label"]
    legacy = LANES["lanes"][LEGACY]["label"]
    assert lines[0] == "## GPU regression · all lanes · `cccccccc`"
    assert any(line.startswith(f"| Commit | [`{SHA[:12]}`]({COMFY}/commit/{SHA}) · Add a node")
               for line in lines)
    assert f"| Pull request | [#17754]({COMFY}/pull/17754) · Add a node · by kijai |" in lines

    assert _section(out, "This commit on every lane")[2:8] == [
        "| Lane | Role | Result | Tested | Not passing |", "|---|---|---|---|---|",
        f"| {main} | primary · blocking | ✅ pass | [2026-11-15 00:30 UTC]({RUN_URL}) |  |",
        f"| {other} | lane · informational | ❌ 1 of 3 failed | [2026-11-15 00:31 UTC]("
        f"{RUN_URL.replace('4242', '4343')}) | `flux_dev_t2i` output changed (new at this "
        "commit); `wan` infra error |",
        # A retired lane shows because it has a result for this commit.
        f"| {legacy} | legacy · retired | ✅ pass | 2026-11-15 00:30 UTC |  |", ""]
    # Only workflows two lanes ran can disagree: `only_here` and `wan` cannot.
    assert _section(out, "Lanes that disagree at this commit")[2:] == [
        f"- `flux_dev_t2i` passes on {main}; fails on {other}", ""]
    old = "8" * 40
    assert _section(out, "Open failures")[2:] == [f"{main}:", "", (
        f"- `sdxl_t2i`: output changed, failing since [`{old[:7]}`]({COMFY}/commit/{old}) "
        f"([#17000]({COMFY}/pull/17000), kijai, Add a node (\\#{ZW}17000)), 4 runs, last passed "
        f"at [`{FIRST[:7]}`]({COMFY}/commit/{FIRST})"), ""]

    matrix = _section(out, "Recent commits")
    assert matrix[2] == f"| Commit | Subject | PR | Author | {main} | {other} | Note |"
    run_other = RUN_URL.replace("4242", "4343")
    assert matrix[4:7] == [
        f"| [`{SHA[:7]}`]({COMFY}/commit/{SHA}) **this report** | Add a node (\\#{ZW}17754) | "
        f"[#17754]({COMFY}/pull/17754) | kijai | [✅ pass]({RUN_URL}) | [❌ 1 of 2 failed]("
        f"{run_other}): `flux_dev_t2i` output changed (new at this commit) | lanes differ |",
        # A commit one lane never ran.
        f"| [`{PREV[:7]}`]({COMFY}/commit/{PREV}) | Add a node (\\#{ZW}17753) | "
        f"[#17753]({COMFY}/pull/17753) | kijai | ✅ pass | ➖ not run |  |",
        # Red on both lanes: they agree.
        f"| [`{old[:7]}`]({COMFY}/commit/{old}) | Add a node (\\#{ZW}17000) | "
        f"[#17000]({COMFY}/pull/17000) | kijai | ❌ 1 of 1 failed: `flux_dev_t2i` execution "
        "error | ❌ 1 of 1 failed: `flux_dev_t2i` execution error |  |"]
    # The retired lane has no column, and its history is one pinned link away.
    assert legacy not in matrix[2]
    assert _section(out, "Other lanes")[2:4] == [
        f"- {LANES['lanes'][IDLE]['label']} is not tested yet: it has no endpoint or is not a "
        "per-commit lane.",
        f"- {legacy} is retired: nothing runs on it any more, and its history is kept in "
        f"[its run index]({RAW}/regression/index/master/{LEGACY}.json)."]
    assert any(line.startswith("Result labels: ") for line in lines)


def test_overview_lane_without_a_result_and_retired_lane_without_one():
    out = _overview(_overview_files(with_legacy=False, other_summary=False))
    rows = _section(out, "This commit on every lane")[4:]
    assert rows == [
        f"| {LANES['lanes'][LANE]['label']} | primary · blocking | ✅ pass | "
        f"[2026-11-15 00:30 UTC]({RUN_URL}) |  |",
        f"| {LANES['lanes'][OTHER]['label']} | lane · informational | ➖ no result |  |  |", ""]
    assert "Lanes that disagree at this commit" not in out
    assert "is retired" in out  # still named, with the link to its history


def test_overview_classifies_an_old_summary_from_the_index():
    """A summary from before `detail` and `output_sha256`: the index entry of
    the commit has both, so the overview can still tell new from accepted."""
    cell = _cell("flux_dev_t2i", "fail")
    del cell["detail"], cell["output_sha256"]
    files = _overview_files(with_legacy=False, other_summary=False)
    files[f"{P}/runs/master/{SHA}/summary.json"] = _summary([cell])
    files[f"regression/index/master/{LANE}.json"] = _head([_entry(SHA, 30, {
        "flux_dev_t2i": {"v": "fail", "d": "new_drift", "sha": "2" * 64}})])
    assert "| ❌ 1 of 1 failed | " in _overview(files)
    assert "`flux_dev_t2i` output changed (new at this commit)" in _overview(files)
    files["regression/index/lanes.json"] = {"lanes": {LANE: {"goldens": {
        "flux_dev_t2i": {"tag": "v0.43.0", "sha": "2" * 64}}}}}
    assert "| ✅ pass | " in _overview(files)
    assert "`flux_dev_t2i` output changed (accepted)" in _overview(files)


def test_overview_reads_a_retired_lanes_index_when_the_lane_has_the_commit():
    """Most summaries on the retired lane are older than `detail`,
    `output_sha256` and `commit_meta`. Its index entry has them, so the
    overview agrees with the lane report and can name the commit."""
    cell = _cell("flux_dev_t2i", "fail")
    del cell["detail"], cell["output_sha256"]
    summary = _summary([cell, _cell("sdxl_t2i")], lane=LEGACY, run_url=None, run_id=None)
    del summary["commit_meta"], summary["tested_range"]
    entry = _entry(SHA, 1794702636, {"sdxl_t2i": "pass", "flux_dev_t2i": {
        "v": "fail", "d": "new_drift", "sha": "2" * 64, "g": "v0.42.0"}}, run=991)
    older = _entry(PREV, 1794700000, {"flux_dev_t2i": "execution_error"}, pr=17753)
    head = _head([entry, older], {"flux_dev_t2i": {"commit": PREV, "kind": "execution_error"}})
    files = {f"regression/runs/master/{SHA}/summary.json": summary,
             f"regression/index/master/{LEGACY}.json": head,
             "regression/index/lanes.json": {"lanes": {LEGACY: {"goldens": {
                 "flux_dev_t2i": {"tag": "v0.43.0", "sha": "2" * 64}}}}}}
    out = _overview(files)
    legacy = LANES["lanes"][LEGACY]["label"]
    assert (f"| Commit | [`{SHA[:12]}`]({COMFY}/commit/{SHA}) · Add a node (\\#{ZW}17754) · by "
            "kijai · committed 2026-11-15 00:20 UTC |") in out
    assert (f"| {legacy} | legacy · retired | ✅ pass | 2026-11-15 00:30 UTC | `flux_dev_t2i` "
            "output changed (accepted) |") in out
    assert _report(files, lane=LEGACY).startswith("## ✅ ")  # as the lane report says
    # The retired lane still has no column, and its open failures are not listed.
    assert "### Recent commits" not in out and "None on the active lanes." in out
    # Without the commit on that lane, its index is not read at all.
    del files[f"regression/runs/master/{SHA}/summary.json"]

    class Counting(run_report.DictReader):
        read = []

        def read_json(self, path):
            self.read.append(path)
            return super().read_json(path)

    run_report.overview_report(Counting(files, RESULTS), LANES, SHA, "master")
    assert f"regression/index/master/{LEGACY}.json" not in Counting.read
    assert f"regression/index/master/{LANE}.json" in Counting.read


def test_overview_counts_an_accepted_change_as_green_when_lanes_are_compared():
    """Accepted is no longer open: next to a lane where the workflow fails, the
    lanes disagree, for the commit and in the matrix of recent commits."""
    accepted = _cell("flux_dev_t2i", "fail", detail="new_drift", output_sha256="2" * 64)
    red = _cell("flux_dev_t2i", "fail", detail="new_drift", output_sha256="3" * 64)
    index_cell = {"v": "fail", "d": "new_drift", "g": "v0.42.0"}
    files = {
        f"{P}/runs/master/{SHA}/summary.json": _summary([accepted]),
        f"regression/lanes/{OTHER}/runs/master/{SHA}/summary.json": _summary([red], lane=OTHER),
        f"regression/index/master/{LANE}.json": _head([_entry(SHA, 30, {
            "flux_dev_t2i": {**index_cell, "sha": "2" * 64}}, run=4242)]),
        f"regression/index/master/{OTHER}.json": _head([_entry(SHA, 31, {
            "flux_dev_t2i": {**index_cell, "sha": "3" * 64}}, run=4343)]),
        "regression/index/lanes.json": {"lanes": {
            LANE: {"goldens": {"flux_dev_t2i": {"tag": "v0.43.0", "sha": "2" * 64}}},
            OTHER: {"goldens": {"flux_dev_t2i": {"tag": "v0.42.0", "sha": "9" * 64}}}}}}
    out = _overview(files)
    main, other = LANES["lanes"][LANE]["label"], LANES["lanes"][OTHER]["label"]
    assert _section(out, "Lanes that disagree at this commit")[2:] == [
        f"- `flux_dev_t2i` passes on {main}; fails on {other}", ""]
    row = next(line for line in _section(out, "Recent commits") if "**this report**" in line)
    assert (f"| [✅ pass]({RUN_URL}): `flux_dev_t2i` output changed (accepted) | [❌ 1 of 1 "
            f"failed]({RUN_URL.replace('4242', '4343')}): `flux_dev_t2i` output changed (new at "
            "this commit) | lanes differ |") in row


def test_overview_with_nothing_published():
    out = _overview({})
    assert out.startswith("## GPU regression · all lanes · `cccccccc`")
    assert "➖ no result" in out and "None on the active lanes." in out
    assert "### Recent commits" not in out
    out = run_report.overview_report(run_report.Reader(), LANES, SHA, "master")
    assert "The results branch could not be read" in out


# -- files on the runner -------------------------------------------------------------------
def _write_local(tmp_path, summary, runs=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    workdir = tmp_path / "regression-work"
    for wf, run in (runs or {}).items():
        (workdir / wf).mkdir(parents=True)
        (workdir / wf / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return str(path), str(workdir)


def test_local_summary_and_run_records_win_over_the_published_copy(tmp_path):
    published = _summary([_cell("flux_dev_t2i")])
    local = _summary([_cell("flux_dev_t2i", "execution_error")], run_ts=1794709999, attempt=2)
    path, workdir = _write_local(tmp_path, local, {"flux_dev_t2i": _run(
        "flux_dev_t2i", "timeout", runpod_job_id="local-job", error={"timeout_s": 600})})
    out = _report(_tree(published), summary_file=path, workdir=workdir)
    assert "1 of 1 workflow failed" in out and "attempt 2 of 2" in out
    assert "RunPod job <code>local-job</code>" in out
    # The branch holds an older summary of the commit: say that this run's is missing.
    assert ("> The results branch does not hold this run's summary. The report is built from "
            "the files on the runner, so images and links to published files are left out."
            ) in out
    assert "raw.githubusercontent.com" not in out
    # Without the runner's records, the older run's published record is not taken for this
    # run's: the branch's run.json would name another RunPod job.
    out = _report(_tree(published), summary_file=path)
    assert "RunPod job" not in out and "- The worker recorded no detail for this failure." in out

    # Published and identical: no warning, and the files are linked.
    path, workdir = _write_local(tmp_path / "same", published)
    out = _report(_tree(published), summary_file=path, workdir=workdir)
    assert "[!WARNING]" not in out and "flux_dev_t2i_00001_.webp" in out


@pytest.mark.parametrize("change", [{"commit": PREV}, {"lane": OTHER}, {"lane": None}])
def test_local_summary_of_another_commit_or_lane_is_ignored(tmp_path, change):
    local = {**_summary([_cell("flux_dev_t2i", "execution_error")]), **change}
    path, workdir = _write_local(tmp_path, local)
    out = _report(_tree(_summary([_cell("flux_dev_t2i")])), summary_file=path, workdir=workdir)
    assert "The workflow matches its reference." in out
    assert _report(_tree(_summary([_cell("flux_dev_t2i")])),
                   summary_file=str(tmp_path / "missing.json")).startswith("## ✅ ")


def test_run_that_stopped_before_publishing_reports_from_the_workdir(tmp_path):
    """No summary anywhere (the publish failed): the per-workflow files the
    run left on the runner still say what happened."""
    workdir = tmp_path / "regression-work"
    for wf, run, comparison in (
            ("flux_dev_t2i", _run("flux_dev_t2i"), {
                "verdict": "fail", "detail": "new_drift", "vs_golden": DIFF, "vs_previous": DIFF,
                "thresholds_used": LIMITS, "golden_tag": "v0.42.0", "previous_commit": PREV,
                "previous_verdict": "pass", "figures": True}),
            ("z_image", _run("z_image", "timeout", error={"timeout_s": 900}), {
                "verdict": "execution_error", "detail": None}),
            ("half_done", _run("half_done"), None)):
        (workdir / wf / "outputs" / "thumbs").mkdir(parents=True)
        (workdir / wf / "run.json").write_text(json.dumps(run), encoding="utf-8")
        if comparison:
            (workdir / wf / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (workdir / "flux_dev_t2i" / "outputs" / "thumbs" / "flux_dev_t2i_00001_.webp").write_bytes(b"x")
    (workdir / "..hidden").mkdir()
    marker = {"commit": SHA, "attempt": 1, "started_ts": 1794700000, "run_url": RUN_URL}
    files = {f"{P}/runs/master/{SHA}/attempt.json": marker,
             f"regression/index/master/{LANE}.json": _head([
                 _entry(PREV, 2, {"flux_dev_t2i": "pass", "z_image": "pass"})])}
    out = _report(files, workdir=str(workdir), summary_file=str(tmp_path / "none.json"),
                  artifact_url=RUN_URL + "/artifacts/99")
    lines = out.split("\n")
    assert lines[0].startswith("## ❌ ")
    assert ("> This run stopped before it published its results. The report is rebuilt from the "
            "files on the runner, so images and links to published files are missing. The "
            "attached files hold the outputs.") in lines
    assert "2 of 3 workflows failed, new at this commit. 1 workflow without a result." in lines
    # Which attempt it was and which run: from the start marker, pushed before the GPU jobs.
    assert f"| Tested | attempt 1 of 2 · [Actions run]({RUN_URL}) |" in lines
    assert "| `flux_dev_t2i` | ❌ output changed (new at this commit) |" in out
    assert "| `z_image` | ❌ execution error |" in out
    assert "| `half_done` | ❔ unknown |" in out
    assert "**3.4422** (limit ≤ 2)" in out
    assert "- **Timeout:** the worker gave up after 900 s" in out
    assert "raw.githubusercontent.com" not in out and "<img" not in out
    assert (f"[the files attached to this run]({RUN_URL}/artifacts/99) (a zip, kept 90 days, "
            "needs a GitHub login)") in out
    assert "### Recent commits on this lane" in out  # history is still read
    # An artifact link that is not one on github.com is dropped.
    for url in ("https://evil.example/x", "https://evil.example/o/r/actions/runs/1/artifacts/2",
                RUN_URL + "/artifacts/2) [x](https://evil.example"):
        assert "attached to this run](" not in _report(files, workdir=str(workdir),
                                                        artifact_url=url)


RETRY_URL = RUN_URL.replace("4242", "5555")


def _retry_that_died(tmp_path):
    """Attempt 1 published an infra error. Attempt 2 (the cron's retry, or a
    dispatch) pushed its start marker, tested, and died before it published."""
    cells = [_cell("flux_dev_t2i"), _cell("sdxl_t2i", "infra_error",
                                          worker_status="checkout_error")]
    marker = {"commit": SHA, "branch": "master", "lane": LANE, "attempt": 2,
              "started_ts": 1794705000, "run_url": RETRY_URL}
    files = _tree(_summary(cells, attempt=1),
                  extra={f"{P}/runs/master/{SHA}/attempt.json": marker})
    workdir = tmp_path / "regression-work"
    for wf, run, comparison in (
            ("flux_dev_t2i", _run("flux_dev_t2i", "execution_error", runpod_job_id="retry-job",
                                  error={"message": "CUDA out of memory"}),
             {"verdict": "execution_error", "detail": None}),
            ("sdxl_t2i", _run("sdxl_t2i"), {
                "verdict": "fail", "detail": None, "vs_golden": DIFF, "thresholds_used": LIMITS,
                "golden_tag": "v0.42.0"})):
        (workdir / wf).mkdir(parents=True)
        (workdir / wf / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (workdir / wf / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    return files, str(workdir), str(tmp_path / "summary.json")


def test_a_run_that_did_not_publish_is_not_reported_as_the_earlier_attempt(tmp_path):
    """The runner's records come before a summary that an earlier attempt
    published: the job that died must not show the older result as its own."""
    files, workdir, missing = _retry_that_died(tmp_path)
    out = _report(files, summary_file=missing, workdir=workdir)
    lines = out.split("\n")
    assert lines[0].startswith("## ❌ ")
    assert lines[2:4] == ["> [!WARNING]", (
        "> This run stopped before it published its results. The report is rebuilt from the "
        "files on the runner, so images and links to published files are missing. The attached "
        "files hold the outputs. The results branch holds an earlier result for this commit, "
        f"tested 2026-11-15 00:30 UTC in [an earlier run]({RUN_URL}): the history table still "
        "shows that one.")]
    assert lines[5] == "2 of 2 workflows failed: 1 new at this commit, 1 not classified."
    # This attempt and this run, from the start marker: not the published attempt's.
    assert f"| Tested | attempt 2 of 2 · [Actions run]({RETRY_URL}) |" in lines
    assert "- **Message:** CUDA out of memory" in lines
    assert "RunPod job <code>retry-job</code>" in out
    # Nothing of the published attempt is mixed in: not its infra error, not the
    # output sha of a workflow that has no output now, not its files.
    assert "| `flux_dev_t2i` | ❌ execution error |  |  |  |  |  |  |  |  |  |" in lines
    assert "| `sdxl_t2i` | ❌ output changed |" in out
    assert "raw.githubusercontent.com" not in out
    assert "infra error" not in "\n".join(_section(out, "All workflows"))
    # A later tick that skips the commit shows what is published, and says so.
    # Records lying on the runner are another run's: they stay out of it.
    out = _report(files, summary_file=missing, workdir=workdir, skipped=True)
    assert out.split("\n")[0].startswith("## ⚠️ ") and "did not test this commit again" in out
    assert "stopped before it published" not in out
    assert (f"| `flux_dev_t2i` | [flux\\_dev\\_t2i\\_00001\\_.png]({RAW}/{P}/runs/master/{SHA}/"
            "flux_dev_t2i/outputs/flux_dev_t2i_00001_.png) |") in out
    assert "CUDA out of memory" not in out


def test_a_run_that_left_nothing_says_the_result_shown_is_an_earlier_one(tmp_path):
    """The job was to test the commit and has no record of it (it failed before
    the first workflow finished). What is published is shown, with a warning."""
    files = _tree(_summary([_cell("flux_dev_t2i")]))
    out = _report(files, summary_file=str(tmp_path / "summary.json"),
                  workdir=str(tmp_path / "regression-work"))
    assert out.split("\n")[2:4] == ["> [!WARNING]", (
        "> This run left no result on the runner. The report below shows the result that is "
        "already published for this commit, tested 2026-11-15 00:30 UTC in [an earlier run]("
        f"{RUN_URL}).")]
    assert "The workflow matches its reference." in out
    # Not when the report is only read (no runner files were named), or skipped.
    for kw in ({}, {"skipped": True, "summary_file": str(tmp_path / "summary.json")}):
        assert "left no result" not in _report(files, **kw)


def test_the_retry_sentence_counts_an_attempt_that_died_like_the_cron_does(tmp_path):
    """regression_skip counts the start marker of attempt 2: the commit is not
    tested again, and the report of the published attempt 1 must not say it is."""
    files, _, _ = _retry_that_died(tmp_path)
    summary = files[f"{P}/runs/master/{SHA}/summary.json"]
    marker = files[f"{P}/runs/master/{SHA}/attempt.json"]
    assert run_report.regression_skip.decide(SHA, summary, marker=marker)[0] is False
    out = _report(files, skipped=True)
    assert ("This was attempt 1 of 2. Attempt 2 started after it and published no result, so "
            "the cron does not retry it. Dispatch **GPU regression** with this commit to test it "
            "again.") in out
    assert "the cron tests the commit once more" not in out
    assert "· attempt 1 of 2 · [Actions run]" in out  # the Tested row names the shown attempt
    # The marker of the shown attempt itself changes nothing.
    files[f"{P}/runs/master/{SHA}/attempt.json"] = {**marker, "attempt": 1}
    assert "This was attempt 1 of 2: the cron tests the commit once more" in _report(
        files, skipped=True)


def test_unreadable_results_branch_still_reports_the_local_run(tmp_path):
    path, workdir = _write_local(tmp_path, _summary([_cell("flux_dev_t2i")]),
                                 {"flux_dev_t2i": _run("flux_dev_t2i")})
    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE,
                                 summary_file=path, workdir=workdir)
    assert ("> The results branch could not be read. History, reference details, images and "
            "links to published files are left out.") in out
    assert "| `flux_dev_t2i` | ✅ pass |  | bit-identical |" in out  # no image to link
    assert "raw.githubusercontent.com" not in out and "Recent commits" not in out
    # Nothing is known to be published: no link to the commit's dashboard page.
    assert out.endswith("Longer history and charts: [ci.comfy.org](https://ci.comfy.org)\n")
    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE, skipped=True)
    assert ("The results branch could not be read, so the published result of this commit on "
            "this lane cannot be shown.") in out


# -- command line ---------------------------------------------------------------------------
class Boom(run_report.Reader):
    """A reader that fails the way a broken checkout or odd data might."""
    sha = RESULTS

    def read_json(self, path):
        raise RuntimeError(f"cannot read {path}: {EVIL}")


def _args(*extra):
    return ["lane", "--commit", SHA, "--branch", "master", "--lane", LANE, *extra]


def test_main_writes_the_step_summary_and_prints_only_a_status_line(tmp_path, monkeypatch, capsys):
    step = tmp_path / "step.md"
    step.write_text("earlier step\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    reader = run_report.DictReader(_hostile_files(), RESULTS)
    assert run_report.main(_args("--out", "-"), reader=reader) == 0
    text = step.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n## ❌ GPU regression · ")  # appended, not replaced
    printed = capsys.readouterr()
    # Inside Actions nothing from the data is ever printed, not even with --out -.
    assert re.fullmatch(r"run report: wrote \d+ bytes to the job summary\n", printed.out)
    assert printed.err == ""


def test_main_never_fails_the_job(tmp_path, monkeypatch, capsys):
    step = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    monkeypatch.setenv("GITHUB_REPOSITORY", "Comfy-Org/comfyci-runpod-worker")
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    assert run_report.main(_args("--artifact-url", RUN_URL + "/artifacts/5"), reader=Boom()) == 0
    text = step.read_text(encoding="utf-8")
    assert text.split("\n")[:3] == [
        "## GPU regression · report not available", "",
        "The report could not be built (RuntimeError). The run's results are not affected."]
    # The exception's message came from data: only its class is shown.
    assert "octocat" not in text and "cannot read" not in text
    assert ("Where to look instead: [this run's files on the results branch](https://github.com/"
            f"Comfy-Org/comfyci-runpod-worker/tree/results/{P}/runs/master/{SHA}), "
            f"[the commit]({COMFY}/commit/{SHA}), [this Actions run]({RUN_URL}), "
            f"[the files attached to this run]({RUN_URL}/artifacts/5), "
            "[the dashboard](https://ci.comfy.org).") in text
    printed = capsys.readouterr()
    assert "octocat" not in printed.out + printed.err

    for argv, reason in (
            (["overview", "--commit", SHA, "--branch", "master"], "RuntimeError"),
            (_args()[:-1] + ["no-such-lane"], "the lane is not in manifest/lanes.json"),
            (["lane", "--commit", "HEAD; rm -rf /", "--branch", "master", "--lane", LANE],
             "--commit is not a hex sha"),
            (["lane", "--commit", SHA, "--branch", "../x", "--lane", LANE],
             "--branch is not a plain branch name"),
            (["lane", "--commit", SHA], "its arguments were not accepted"),
            (["bogus"], "its arguments were not accepted")):
        step.write_text("", encoding="utf-8")
        assert run_report.main(argv, reader=Boom()) == 0
        assert f"The report could not be built ({reason})." in step.read_text(encoding="utf-8")
    capsys.readouterr()
    # A summary file that cannot be written is not a failure either.
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "no" / "such" / "dir" / "s.md"))
    assert run_report.main(_args(), reader=run_report.DictReader({}, RESULTS)) == 0
    assert capsys.readouterr().out == (
        "run report: could not write the summary (FileNotFoundError)\n")


def test_main_outside_actions_writes_to_out_or_stdout(tmp_path, capsys):
    reader = run_report.DictReader(_tree(_summary([_cell("flux_dev_t2i")])), RESULTS)
    out = tmp_path / "report.md"
    assert run_report.main(_args("--out", str(out)), reader=reader) == 0
    assert out.read_text(encoding="utf-8").startswith("## ✅ GPU regression · ")
    assert re.fullmatch(r"run report: wrote \d+ bytes to the --out file\n", capsys.readouterr().out)
    assert run_report.main(_args("--out", "-"), reader=reader) == 0
    assert capsys.readouterr().out.startswith("## ✅ GPU regression · ")
    # No destination: say so, print nothing else.
    assert run_report.main(_args(), reader=reader) == 0
    assert capsys.readouterr().out == (
        "run report: nowhere to write (no GITHUB_STEP_SUMMARY, no --out)\n")
    assert run_report.main(["overview", "--commit", SHA, "--branch", "master", "--out", str(out)],
                           reader=reader) == 0
    assert out.read_text(encoding="utf-8").startswith("## GPU regression · all lanes · ")


def test_every_lane_flag_reaches_the_report(tmp_path):
    """The workflow's Report step passes these: each one changes the report."""
    published = _summary([_cell("flux_dev_t2i")])
    reader = run_report.DictReader(_tree(published), RESULTS)
    out = tmp_path / "report.md"

    def text(*flags):
        assert run_report.main(_args("--out", str(out), *flags), reader=reader) == 0
        return out.read_text(encoding="utf-8")

    # --skipped: the note that this run tested nothing.
    assert "did not test this commit again" not in text()
    assert "> This run did not test this commit again" in text("--skipped")
    # --summary: the summary this run wrote wins over the published one.
    local = _summary([_cell("flux_dev_t2i", "execution_error")], run_ts=1794709999, attempt=2)
    path, workdir = _write_local(tmp_path / "local", local, {"flux_dev_t2i": _run(
        "flux_dev_t2i", "timeout", runpod_job_id="local-job", error={"timeout_s": 600})})
    assert "1 of 1 workflow failed" in text("--summary", path)
    # --workdir: the run records next to it.
    assert "RunPod job <code>local-job</code>" not in text("--summary", path)
    assert "RunPod job <code>local-job</code>" in text("--summary", path, "--workdir", workdir)
    # --workdir alone: a run that wrote no summary reports from its records.
    (Path(workdir) / "flux_dev_t2i" / "comparison.json").write_text(
        json.dumps({"verdict": "execution_error"}), encoding="utf-8")
    assert "This run stopped before it published its results." in text("--workdir", workdir)
    # --artifact-url: the link to the attached files.
    url = RUN_URL + "/artifacts/77"
    assert "attached to this run" not in text()
    assert f"[the files attached to this run]({url})" in text("--artifact-url", url)


# -- the git-backed reader ------------------------------------------------------------------
SUMMARY_PATH = f"{P}/runs/master/{SHA}/summary.json"
RUN_PATH = f"{P}/runs/master/{SHA}/flux_dev_t2i/run.json"


def _remote(tmp_path):
    summary = _summary([_cell("flux_dev_t2i")])
    return make_remote(tmp_path, {
        **{path: obj for path, obj in _tree(summary).items()},
        f"{P}/runs/master/{SHA}/flux_dev_t2i/outputs/flux_dev_t2i_00001_.png": b"\x89PNG seed",
        f"{P}/runs/master/{SHA}/flux_dev_t2i/figures/side_by_side_golden.png": b"\x89PNG fig",
        "regression/broken.json": "{not json", "regression/list.json": "[1, 2]",
        "regression/deep.json": "[" * 100000 + "]" * 100000})


def _checkout(tmp_path, remote):
    """A checkout of another branch with the results repo as origin, like the
    workflow's own checkout of this repository."""
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", "-b", "main", str(checkout)], check=True)
    git(checkout, "remote", "add", "origin", str(remote))
    return checkout


def test_git_reader_reads_the_branch_at_one_pinned_commit(tmp_path):
    remote = _remote(tmp_path)
    checkout = _checkout(tmp_path, remote)
    reader = run_report.GitReader(checkout)
    head = git(remote, "rev-parse", "results").strip()
    assert reader.sha == head and len(head) == 40
    assert reader.read_json(SUMMARY_PATH)["commit"] == SHA
    assert reader.read_json(RUN_PATH)["status"] == "ok"
    assert reader.read_json(f"{P}/runs/master/{PREV}/summary.json") is None      # absent
    assert reader.read_json("regression/broken.json") is None                    # not JSON
    assert reader.read_json("regression/list.json") is None                      # not an object
    assert reader.read_json("regression/deep.json") is None                      # nested too deep
    assert reader.read_json(f"{P}/runs/master/{SHA}") is None                    # a directory
    assert reader.list_dir(f"{P}/runs/master/{SHA}") == ["flux_dev_t2i", "summary.json"]
    assert reader.list_dir(f"{P}/runs/master/{SHA}/flux_dev_t2i/figures/") == [
        "side_by_side_golden.png"]
    assert reader.list_dir("regression/nowhere") == []
    assert not (checkout / "regression").exists()  # nothing is checked out

    # The branch moves on; this reader keeps reading the commit it fetched.
    seed = tmp_path / "seed"
    (seed / SUMMARY_PATH).write_text(json.dumps({"commit": "moved"}), encoding="utf-8")
    git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "next")
    git(seed, "push", "-q", str(remote), "HEAD:refs/heads/results")
    assert reader.read_json(SUMMARY_PATH)["commit"] == SHA and reader.sha == head
    later = run_report.GitReader(checkout)
    assert later.sha != head and later.read_json(SUMMARY_PATH) == {"commit": "moved"}
    assert reader.read_json(SUMMARY_PATH)["commit"] == SHA  # FETCH_HEAD moved, the pin did not

    # The whole report through the reader, links pinned to that commit.
    out = run_report.lane_report(reader, LANES, SHA, "master", LANE)
    assert out.startswith("## ✅ GPU regression · ")
    assert (f"https://raw.githubusercontent.com/Comfy-Org/comfyci-runpod-worker/{head}/{P}/runs/"
            f"master/{SHA}/flux_dev_t2i/outputs/flux_dev_t2i_00001_.png") in out


def test_git_reader_prefetch_loads_blobs_in_one_fetch(tmp_path):
    remote = _remote(tmp_path)
    checkout = _checkout(tmp_path, remote)
    reader = run_report.GitReader(checkout)
    reader.prefetch([SUMMARY_PATH, RUN_PATH, f"{P}/runs/master/{SHA}/absent.json", ""])
    # Both blobs are local now: reading them needs no remote.
    git(checkout, "remote", "set-url", "origin", str(tmp_path / "gone"))
    assert reader.read_json(SUMMARY_PATH)["commit"] == SHA
    assert reader.read_json(RUN_PATH)["status"] == "ok"
    # One that was not prefetched cannot load any more: None, not an error.
    assert reader.read_json("regression/index/lanes.json") is None
    reader.prefetch(["regression/index/lanes.json"])  # the fetch fails quietly
    assert reader.list_dir(f"{P}/runs/master/{SHA}") == ["flux_dev_t2i", "summary.json"]


def test_git_reader_without_fetch_reads_head_of_a_results_checkout(tmp_path):
    remote = _remote(tmp_path)
    clone = tmp_path / "results-checkout"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(remote), str(clone)], check=True)
    reader = run_report.GitReader(clone, fetch=False)
    assert reader.sha == git(remote, "rev-parse", "results").strip()
    reader.prefetch([SUMMARY_PATH])
    assert reader.read_json(SUMMARY_PATH)["commit"] == SHA
    # A full clone stays one: prefetch does not turn it into a partial clone.
    promisor = subprocess.run(["git", "-C", str(clone), "config", "--get",
                               "remote.origin.promisor"], capture_output=True, text=True)
    assert promisor.stdout.strip() == ""


def test_cli_with_a_checkout_and_with_none(tmp_path, monkeypatch, capsys):
    checkout = _checkout(tmp_path, _remote(tmp_path))
    out = tmp_path / "report.md"
    assert run_report.main(_args("--results-dir", str(checkout), "--out", str(out))) == 0
    assert out.read_text(encoding="utf-8").startswith("## ✅ GPU regression · ")
    assert run_report.main(["overview", "--commit", SHA, "--branch", "master", "--results-dir",
                            str(checkout), "--out", str(out)]) == 0
    assert "| ✅ pass |" in out.read_text(encoding="utf-8")
    # No git repository there: the report says the branch could not be read.
    with pytest.raises(run_report.ReportError):
        run_report.GitReader(tmp_path / "empty")
    empty = tmp_path / "empty"
    empty.mkdir()
    for flags in ((), ("--no-fetch",)):
        assert run_report.main(_args("--results-dir", str(empty), "--out", str(out), *flags)) == 0
        assert "The results branch could not be read" in out.read_text(encoding="utf-8")
    assert "Traceback" not in capsys.readouterr().err


def test_cli_no_fetch_reads_a_results_checkout_that_has_no_remote(tmp_path):
    """The composite action reports from the checkout the suite pushed from."""
    clone = tmp_path / "results-checkout"
    subprocess.run(["git", "clone", "-q", "-b", "results", str(_remote(tmp_path)), str(clone)],
                   check=True)
    git(clone, "remote", "remove", "origin")
    out = tmp_path / "report.md"
    args = _args("--results-dir", str(clone), "--out", str(out))
    assert run_report.main([*args, "--no-fetch"]) == 0
    assert out.read_text(encoding="utf-8").startswith("## ✅ GPU regression · ")
    # Without the flag it fetches first, and there is nowhere to fetch from.
    assert run_report.main(args) == 0
    assert "The results branch could not be read" in out.read_text(encoding="utf-8")


# -- the workflows that call it ----------------------------------------------------------------
def test_the_regression_workflow_reports_every_lane_job_and_all_lanes():
    steps = workflow_steps(".github/workflows/regression.yml")
    lane, overview = (text for name, text in steps if name == "Report")
    # After a test, a failed test and a skipped one, and never as a failure of its own.
    assert "\n        if: always()\n        continue-on-error: true\n" in lane
    for line in ("COMMIT: ${{ needs.lanes.outputs.commit }}", "LANE: ${{ matrix.lane }}",
                 "SKIPPED: ${{ steps.target.outputs.skip }}",
                 "ARTIFACT_URL: ${{ steps.upload.outputs.artifact-url }}",
                 'args=(lane --commit "$COMMIT" --branch master --lane "$LANE"',
                 '--summary "$RUNNER_TEMP/summary.json"',
                 '--workdir "$RUNNER_TEMP/regression-work")',
                 'if [ "$SKIPPED" = "true" ]; then args+=(--skipped); fi',
                 'if [ -n "$ARTIFACT_URL" ]; then args+=(--artifact-url "$ARTIFACT_URL"); fi',
                 'python scripts/run_report.py "${args[@]}"'):
        assert line in lane, line
    # The suite writes the summary and the records where the report looks for them.
    suite = dict(steps)["Run regression suite"]
    assert '--workdir "$RUNNER_TEMP/regression-work"' in suite
    assert '--summary-out "$RUNNER_TEMP/summary.json"' in suite
    upload = dict(steps)["Attach the run's files"]
    assert "id: upload" in upload and "continue-on-error: true" in upload
    assert "${{ runner.temp }}/regression-work" in upload
    assert "${{ runner.temp }}/summary.json" in upload
    # All lanes side by side, once the lane jobs are done, whatever they ended as.
    assert "continue-on-error: true" in overview
    assert "COMMIT: ${{ needs.lanes.outputs.commit }}" in overview
    assert 'run: python scripts/run_report.py overview --commit "$COMMIT" --branch master' in (
        overview)
    text = (ROOT / ".github" / "workflows" / "regression.yml").read_text(encoding="utf-8")
    assert ("  overview:\n    needs: [lanes, regression]\n    if: always() && needs.lanes.result "
            "== 'success' && needs.lanes.outputs.matrix != '[]'\n") in text
    assert "    name: ${{ matrix.lane }} · ${{ needs.lanes.outputs.short }}\n" in text


def test_the_composite_action_reports_from_the_checkout_it_pushed_from():
    steps = dict(workflow_steps("action.yml"))
    assert '--summary-out "$RUNNER_TEMP/summary.json"' in steps["Run regression suite"]
    assert '--results-workdir "$RUNNER_TEMP/results-checkout"' in steps["Run regression suite"]
    report = steps["Report"]
    assert "\n      if: always() && inputs.lane != ''\n      continue-on-error: true\n" in report
    for part in ("python comfyci-runpod-worker/scripts/run_report.py lane",
                 '--commit "$REPORT_COMMIT" --branch "$REPORT_BRANCH" --lane "$REPORT_LANE"',
                 '--summary "$RUNNER_TEMP/summary.json" --workdir regression-work',
                 '--results-dir "$RUNNER_TEMP/results-checkout" --no-fetch',
                 "REPORT_WORKER_REPO: comfy-org/comfyci-runpod-worker"):
        assert part in report, part
    # The inputs reach the command through the environment, never by expansion into it.
    assert "${{" not in report.split("run: |", 1)[1]


# -- text workflows ----------------------------------------------------------------------------
WF = "llm_qwen_t2t"
TXT = f"{WF}_00001.txt"
T_REF = ("A regression test compares what a program produces today with what it produced "
         "before.\nWhen the two differ, either the program changed on purpose or it broke.\n"
         "The test cannot tell which: a person has to look.")
T_NEW = T_REF.replace("either the program changed on purpose or it broke",
                      "the program either broke or was changed on purpose")
T_TAG = "v0.45.4"
T_RUN = f"{P}/runs/master/{SHA}/{WF}/outputs/{TXT}"
T_GOLDEN = f"{P}/golden/{WF}/{T_TAG}/outputs/{TXT}"
T_PREV = f"[`{PREV[:7]}`]({COMFY}/commit/{PREV})"
DIFF_TITLE = ("**Difference from the reference**, with removed text as <code>[-…-]</code> and "
              "added text as <code>{+…+}</code>")
DIFF_FOLDED = DIFF_TITLE + ", long lines folded"
# What a model could write to break out of the report or to run a workflow command.
T_EVIL = ("@octocat see #9911 <script>alert(1)</script> [x](https://evil.example/) **bold**\n"
          "::error::not a command\n</pre>\n</details>\n| a | b |\n# heading \x1b[31mred\x1b[0m\x07 "
          "<img src=x onerror=alert(1)> $a+b$ `tick`")


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_vs(ref, cand):
    """vs_golden or vs_previous of a text workflow, as the comparison writes it."""
    one = output_kinds.compare_texts(output_kinds.normalise_text(ref),
                                     output_kinds.normalise_text(cand))
    diff = one.pop("first_diff", None)
    out = {"kind": "text", "files_compared": 1, "file_count_ref": 1, "file_count_cand": 1, **one}
    return {**out, "first_diff": {"file": TXT, **diff}} if diff else out


def _text_cell(verdict="pass", text=T_REF, ref=T_REF, prev=T_REF, **kw):
    """The summary cell of a text workflow that generated `text`."""
    compared = verdict in ("pass", "fail")
    return _cell(WF, verdict, **{
        "vs_golden": _text_vs(ref, text) if compared else None,
        "vs_previous": _text_vs(prev, text) if prev is not None else None,
        "thresholds_used": None, "golden_tag": T_TAG if compared else None, "thumbnail": None,
        "output_sha256": _sha256(text), "output_kind": "text", **kw})


def _text_run(text, **kw):
    return _run(WF, outputs=[{"filename": TXT, "bytes": len(text.encode("utf-8")),
                              "sha256": _sha256(text), **kw}])


def _text_files(cell, text=T_REF, ref=T_REF, others=(), first_bad=None, registry=None,
                extra=None):
    """What a run with a text workflow leaves on the results branch: the
    generated text under the run, the reference text under the golden."""
    cells = [*others, cell]
    this = {c["workflow_id"]: {"v": c["verdict"], "d": c.get("detail"),
                               "sha": c.get("output_sha256"), "g": c.get("golden_tag")}
            for c in cells}
    head = _head([_entry(SHA, 3, this), _entry(PREV, 2, dict.fromkeys(this, "pass"))], first_bad)
    files = {T_RUN: text,
             f"{P}/golden/{WF}/current.json": {"tag": T_TAG, "blessed_by": "dev",
                                               "blessed_ts": 1792378306},
             f"{P}/golden/{WF}/{T_TAG}/noise_floor.json": {**_text_vs(T_REF, T_REF),
                                                          "measured": True}}
    if ref is not None:
        files[T_GOLDEN] = ref
    return _tree(_summary(cells), runs={WF: _text_run(text)}, head=head, registry=registry,
                 extra={**files, **(extra or {})})


def _pre_blocks(out):
    return re.findall(r"<pre>\n(.*?)\n</pre>", out, flags=re.S)


def test_a_passing_text_workflow_shows_its_text_next_to_the_image_workflows():
    out = _report(_text_files(_text_cell(), others=[_cell("flux_dev_t2i")]))
    lines = out.split("\n")
    assert lines[0].startswith("## ✅ ") and "All 2 workflows match their reference." in lines
    assert "What needs attention" not in out
    # No thumbnail: a link to the text. "identical text" where an image says "bit-identical".
    assert (f"| `{WF}` | ✅ pass | [text]({RAW}/{T_RUN}) | identical text | identical to {T_PREV} "
            f"| `{T_TAG}` | 98.1 | 12.4 | 21.3 | 33.7 | `{_sha256(T_REF)[:12]}` |") in lines
    assert any(line.startswith('| `flux_dev_t2i` | ✅ pass | <a href="') and "| bit-identical |"
               in line for line in lines)
    assert f"{WF}/outputs/thumbs" not in out and f"{WF}/figures" not in out
    # The text itself, folded, after the tables; the file is linked.
    section = _section(out, "Generated text")
    assert section[:5] == ["### Generated text", "", f"`{WF}`, identical to its reference:", "",
                           "<details><summary>Generated text (208 characters)</summary>"]
    assert "\n".join(section).count("<pre>") == 1 and _pre_blocks(out) == [T_REF]
    assert f"[The whole file]({RAW}/{T_RUN})" in section
    assert lines.index("### Generated text") > lines.index("### All workflows (2)")
    assert lines.index("### Generated text") < lines.index("### Recent commits on this lane")
    assert f"| `{WF}` | [llm\\_qwen\\_t2t\\_00001.txt]({RAW}/{T_RUN}) | 208 B |" in out
    # A reference that is identical is the same text: it is not read or shown again.
    assert "Reference text" not in out and "Difference from the reference" not in out
    # A run of image workflows alone has no such section.
    assert "Generated text" not in _report(_tree(_summary([_cell("flux_dev_t2i")])))


def test_a_changed_text_shows_where_it_differs_both_texts_and_how_to_accept():
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    out = _report(_text_files(cell, T_NEW, others=[_cell("flux_dev_t2i")]))
    lines = out.split("\n")
    assert lines[0].startswith("## ❌ ")
    assert "1 of 2 workflows failed, new at this commit." in lines
    assert f"#### ❌ `{WF}`: output changed (new at this commit)" in lines
    # A text has no limits to cross: it matches or it differs.
    assert ("The generated text differs from the reference, and it differs from the previous "
            f"tested commit {T_PREV}: the change is new at this commit.") in lines
    assert ("| Compared with | Result | Similarity | First difference | Characters | Lines |"
            in lines)
    assert (f"| Reference `{T_TAG}` | **differs** | 86.5 % | line 2, column 22 | 209 here, 208 in "
            "the reference | 3 |") in lines
    # Bold marks what fails: the run is judged against the reference only.
    assert (f"| Previous tested commit {T_PREV} | differs | 86.5 % | line 2, column 22 | "
            "209 here, 208 in the previous run | 3 |") in lines
    assert "| Noise floor (the reference generated twice) | identical |  |  | 208 | 3 |" in lines
    assert f"Reference: `{T_TAG}`, approved 2026-10-19 02:51 UTC by dev" in lines

    # The difference, readable, then the two texts behind folds, each with its file.
    # The marks make its second line longer than 100 columns, and the title says so:
    # "line 2, column 22" counts in the file.
    at = lines.index(DIFF_FOLDED + ":")
    assert lines[at + 1:at + 3] == ["", "<pre>"]
    diff, generated, reference = _pre_blocks(out)
    assert diff.split("\n")[1:3] == [
        "When the two differ, [-either -]the program{+ either broke or was+} changed on [-purpose "
        "or it", "broke.-]{+purpose.+}"]
    assert (generated, reference) == (T_NEW, T_REF)
    assert "<details><summary>Generated text (209 characters)</summary>" in lines
    assert "<details><summary>Reference text (208 characters)</summary>" in lines
    assert lines.index(f"[The whole file]({RAW}/{T_RUN})") < lines.index(
        f"[The whole file]({RAW}/{T_GOLDEN})")
    base = ("gh workflow run golden-baseline.yml --repo Comfy-Org/comfyci-runpod-worker "
            f"-f ref={SHA} -f workflows={WF} -f lane={LANE}")
    assert f"{base} -f bless=false" in lines and f"{base} -f bless_only=true" in lines
    assert lines.index(f"{base} -f bless=false") > lines.index("</details>")

    assert (f"| `{WF}` | ❌ output changed (new at this commit) | [text]({RAW}/{T_RUN}) | 86.5 % "
            f"similar · differs from line 2 | changed from {T_PREV} | `{T_TAG}` | 98.1 | 12.4 | "
            f"21.3 | 33.7 | `{_sha256(T_NEW)[:12]}` |") in lines
    # Nothing an image failure shows: no figures, no "limits", and no section for a pass.
    assert "<img" not in "\n".join(_section(out, "What needs attention"))
    assert "beyond its limits" not in out and "figures" not in out.lower()
    assert "### Generated text" not in out
    # The model's words stay inside <pre>, the lines of the first difference included.
    assert "program" not in _outside_pre(out)


def test_an_inherited_text_failure_shows_its_own_texts():
    cell = _text_cell("fail", T_NEW, prev=T_NEW, detail="inherited", previous_verdict="fail")
    chain = {WF: {"commit": FIRST, "run_ts": 1794600000, "kind": "fail", "prev_good": None,
                  "pr": 17700, "author": "dev", "golden": T_TAG, "runs": 3}}
    out = _report(_text_files(cell, T_NEW, first_bad=chain))
    lines = out.split("\n")
    assert f"#### ❌ `{WF}`: output changed (inherited)" in lines
    assert ("The generated text differs from the reference. It is identical to the previous "
            f"tested commit {T_PREV}, which already failed: the change is inherited from an "
            "earlier commit.") in lines
    assert any(line.startswith("Failing since ") and "3 runs" in line for line in lines)
    assert f"| Previous tested commit {T_PREV} | identical |  |  | 209 | 3 |" in lines
    # Texts cost nothing to show again: no pointer to the commit where the change started.
    assert DIFF_FOLDED + ":" in lines and len(_pre_blocks(out)) == 3
    assert "has no figures of its own" not in out
    assert (f"| `{WF}` | ❌ output changed (inherited) | [text]({RAW}/{T_RUN}) | 86.5 % similar · "
            f"differs from line 2 | identical to {T_PREV} |") in out


def test_a_text_workflow_without_a_reference_shows_what_it_generated():
    cell = _text_cell("no_baseline", prev=None)
    out = _report(_text_files(cell, ref=None))
    lines = out.split("\n")
    assert f"#### ⚪ `{WF}`: no reference yet" in lines
    assert ("To make one, generate it from a release tag, look at the output, then approve it "
            "with the second command:") in lines
    base = ("gh workflow run golden-baseline.yml --repo Comfy-Org/comfyci-runpod-worker "
            f"-f ref=RELEASE_TAG -f workflows={WF} -f lane={LANE}")
    assert f"{base} -f bless=false" in lines and f"{base} -f bless_only=true" in lines
    # The text to look at before approving, after the commands.
    at = lines.index("<details><summary>Generated text (208 characters)</summary>")
    assert at > lines.index(f"{base} -f bless_only=true") and _pre_blocks(out) == [T_REF]
    assert f"| `{WF}` | ⚪ no reference yet | [text]({RAW}/{T_RUN}) |  |  |  | 98.1 |" in out
    assert "Reference text" not in out and "Difference" not in out
    # An image workflow keeps its own wording.
    image = _report(_tree(_summary([_cell("flux_dev_t2i", "no_baseline")])))
    assert "To make one, render it from a release tag, look at the output" in image


def test_a_reference_text_that_cannot_be_read_leaves_the_numbers_and_the_generated_text():
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    # The golden's file is not on the branch: the comparison's own numbers still say where.
    out = _report(_text_files(cell, T_NEW, ref=None))
    lines = out.split("\n")
    assert (f"| Reference `{T_TAG}` | **differs** | 86.5 % | line 2, column 22 | 209 here, 208 in "
            "the reference | 3 |") in lines
    assert "The reference text could not be read, so the difference is not shown." in lines
    assert "Difference from the reference" not in out and "Reference text" not in out
    assert _pre_blocks(out) == [T_NEW]
    assert "<details><summary>Generated text (209 characters)</summary>" in lines


def test_hostile_generated_text_stays_inside_pre_blocks():
    """Generated text is model output. It must not mention anyone, link
    anywhere, close the block it is in or start a workflow command."""
    cell = _text_cell("fail", T_EVIL, ref=T_EVIL.upper(), prev=T_EVIL.lower(), detail="new_drift")
    out = _report(_text_files(cell, T_EVIL, ref=T_EVIL.upper()))
    diff, generated, reference = _pre_blocks(out)
    for block in (diff, generated, reference):
        assert "<" not in block and ">" not in block  # every tag is text
        assert "\x1b" not in block and "\x07" not in block
    assert generated.split("\n") == [
        "@octocat see #9911 &lt;script&gt;alert(1)&lt;/script&gt; [x](https://evil.example/) "
        "**bold**", "&#58;:error::not a command", "&lt;/pre&gt;", "&lt;/details&gt;", "| a | b |",
        "# heading red &lt;img src=x onerror=alert(1)&gt; $a+b$ `tick`"]
    assert "&#58;:ERROR::NOT A COMMAND" in reference
    outside = _outside_pre(out).lower()
    for word in ("octocat", "9911", "script", "evil.example", "bold", "heading", "onerror",
                 "not a command", "a+b", "tick"):
        assert word not in outside, word
    # No workflow command, and the blocks are closed only by the report.
    assert not [line for line in out.split("\n") if line.strip().startswith("::")]
    assert out.count("<pre>") == out.count("</pre>") == 3
    assert out.count("<details>") == out.count("</details>")
    # The lines at the first difference are model output too: the table names the place only.
    row = next(line for line in out.split("\n") if line.startswith(f"| Reference `{T_TAG}`"))
    assert row == f"| Reference `{T_TAG}` | **differs** | 20.8 % | line 1, column 2 | 197 | 6 |"
    # A file name that is no plain name is not read, linked or shown.
    files = _text_files(cell, T_EVIL, ref=T_EVIL.upper())
    files[f"{P}/runs/master/{SHA}/{WF}/run.json"]["outputs"][0]["filename"] = "../../x.txt"
    out = _report(files)
    assert _pre_blocks(out) == [] and "[text](" not in out and "x.txt](" not in out


def test_no_line_of_a_pre_block_can_be_read_as_a_workflow_command():
    """The runner trims a line before it looks for "::", so blanks before it
    make it no safer. A summary is not read for commands; a step that prints
    the report (--out - outside Actions) is."""
    text = ("::stop-commands::x\n  ::add-mask::y\n\t::error::z\n\u00a0::warning::w\n"
            "\u3000::notice::v\nfine ::here\n\n ::set-output name=a::b")
    block = run_report.pre(text).split("\n")
    assert not [line for line in block if line.strip().startswith("::")]
    assert block[1:-1] == ["&#58;:stop-commands::x", "  &#58;:add-mask::y", "\t&#58;:error::z",
                           "\u00a0&#58;:warning::w", "\u3000&#58;:notice::v", "fine ::here", "",
                           " &#58;:set-output name=a::b"]
    # Through a report: generated text with an indented command, printed to stdout.
    cell = _text_cell("no_baseline", "ok\n  ::add-mask::secret", prev=None)
    out = _report(_text_files(cell, "ok\n  ::add-mask::secret", ref=None))
    assert "  &#58;:add-mask::secret" in out.split("\n")
    assert not [line for line in out.split("\n") if line.strip().startswith("::")]


def test_a_long_text_is_cut_where_it_is_shown_and_nothing_else_is_dropped():
    long = "\n".join(f"line {i:04d} of a text that goes on and on" for i in range(400))
    cell = _text_cell("fail", long, ref=long.upper(), detail="new_drift")
    out = _report(_text_files(cell, long, ref=long.upper()))
    lines = out.split("\n")
    # Each text stops at Limits.text_chars characters, and its heading says so.
    assert (f"<details><summary>Generated text ({len(long)} characters, the first 4000 shown)"
            "</summary>") in lines
    assert (f"<details><summary>Reference text ({len(long)} characters, the first 4000 shown)"
            "</summary>") in lines
    assert DIFF_TITLE + ", cut to its first 8000 characters:" in lines
    diff, generated, reference = _pre_blocks(out)
    assert generated.endswith("\n…") and len(generated) <= 4002 and len(diff) <= 8002
    assert generated.startswith("line 0000 of a text") and "line 0399" not in generated
    assert f"[The whole file]({RAW}/{T_RUN})" in lines
    # That is not the size limit at work: nothing else was shortened.
    assert "Shortened to stay under" not in out and "### Recent commits on this lane" in lines
    assert "<details><summary>Output files (1)</summary>" in lines


def test_a_text_too_large_to_read_is_linked_not_shown():
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    files = _text_files(cell, T_NEW)
    size = run_report.TEXT_READ_MAX + 1
    files[f"{P}/runs/master/{SHA}/{WF}/run.json"]["outputs"][0]["bytes"] = size
    out = _report(files)
    assert ("The generated text is 256.0 KB, too large to show here. [The whole file]"
            f"({RAW}/{T_RUN}).") in out.split("\n")
    assert _pre_blocks(out) == [] and "Reference text" not in out
    assert f"[text]({RAW}/{T_RUN})" in out  # the table still links it


def test_generated_texts_are_the_last_thing_the_size_limit_takes():
    """The steps that existed keep their order, so a report without text is
    shortened exactly as before; for a text workflow the text is the result."""
    steps = run_report.SHRINK_STEPS
    assert steps[-2:] == ({"text_chars": 1000}, {"text_chars": 0})
    assert not any("text_chars" in step for step in steps[:-2]) and len(steps) == 11
    assert run_report.Limits().text_chars == 4000
    assert run_report.Limits(text_chars=1000).shortened() == [
        "generated texts cut to their first 1000 characters"]
    assert run_report.Limits(history_rows=0, matrix_rows=0, text_chars=0).shortened() == [
        "recent commits left out", "generated texts left out"]

    long = "\n".join(f"line {i:04d} of a text that goes on and on" for i in range(400))
    cell = _text_cell("fail", long, ref=long.upper(), detail="new_drift")
    reader = run_report.DictReader(_text_files(cell, long, ref=long.upper()),
                                   RESULTS)
    d = run_report.gather_lane(reader, LANES, SHA, "master", LANE)
    full = run_report.render_lane(d, run_report.Limits())
    # Cut: each text to 1000 characters, the difference to twice that.
    cut = run_report.render_lane(d, run_report.Limits(text_chars=1000))
    assert f"Generated text ({len(long)} characters, the first 1000 shown)" in cut
    assert DIFF_TITLE + ", cut to its first 2000 characters:" in cut
    assert [len(block) <= 2002 for block in _pre_blocks(cut)] == [True] * 3
    # Left out: one sentence with the link, and the numbers stay.
    none = run_report.render_lane(d, run_report.Limits(text_chars=0))
    assert ("The generated text is left out here to keep the summary small. [The whole file]"
            f"({RAW}/{T_RUN}).") in none.split("\n")
    assert _pre_blocks(none) == [] and "| **differs** |" in none
    assert len(none) < len(cut) < len(full)

    # Through fit(): the text goes only after logs, tracebacks, file lists and history.
    seen = []

    def render(limits):
        seen.append(limits)
        return run_report.render_lane(d, limits)

    budget = run_report.byte_size(none) + 200
    out = run_report.fit(render, max_bytes=budget)
    assert out is not None and run_report.byte_size(out) <= budget
    assert [s.text_chars for s in seen] == [4000] * 10 + [1000, 0]
    assert seen[-1].history_rows == 0 and seen[-1].log_chars == 0
    note = next(line for line in out.split("\n") if line.startswith("Shortened to stay under"))
    assert note.endswith("recent commits left out; generated texts left out. The linked files "
                         "hold everything.")


def test_the_difference_is_marked_once_however_often_the_report_is_rendered(monkeypatch):
    """A report over the size limit is rendered again at every step down, and
    marking the difference of two long texts is the slowest thing in it."""
    calls = []
    real = output_kinds.word_diff
    monkeypatch.setattr(output_kinds, "word_diff", lambda a, b: calls.append(1) or real(a, b))
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    reader = run_report.DictReader(_text_files(cell, T_NEW), RESULTS)
    d = run_report.gather_lane(reader, LANES, SHA, "master", LANE)
    blocks = [_pre_blocks(run_report.render_lane(d, run_report.Limits(**step)))
              for step in ({}, *run_report.SHRINK_STEPS[:-1])]
    assert len(calls) == 1 and all(len(shown) == 3 for shown in blocks)


def test_a_text_that_passes_within_a_similarity_limit_shows_its_difference():
    cell = _text_cell("pass", T_NEW, thresholds_used={"min_text_similarity": 0.8})
    out = _report(_text_files(cell, T_NEW))
    lines = out.split("\n")
    assert lines[0].startswith("## ✅ ")
    section = _section(out, "Passed without being bit-identical")
    assert f"<details><summary><code>{WF}</code>: numbers and limits</summary>" in section
    assert (f"| Reference `{T_TAG}` | differs | 86.5 % (limit ≥ 80 %) | line 2, column 22 | "
            "209 here, 208 in the reference | 3 |") in section
    assert DIFF_FOLDED + ":" in section and len(_pre_blocks(out)) == 3
    assert "### Generated text" not in out  # shown once, where its difference is
    # Below the limit it fails, the number is bold and the sentence names the limit.
    low = _text_cell("fail", T_NEW, detail="new_drift",
                     thresholds_used={"min_text_similarity": 0.98})
    out = _report(_text_files(low, T_NEW))
    assert "| **differs** | **86.5 %** (limit ≥ 98 %) | line 2, column 22 |" in out
    assert "The generated text differs from the reference beyond its similarity limit, and" in out


def test_a_text_that_matches_its_reference_again_is_not_listed_with_numbers_and_limits():
    """The commit after a failing one restores the text: it differs from the
    previous run and equals its reference. A text is not compared bit by bit
    and that pass crossed no limit, so it is shown as a text that matches."""
    out = _report(_text_files(_text_cell("pass", T_REF, prev=T_NEW)))
    lines = out.split("\n")
    assert lines[0].startswith("## ✅ ")
    assert "Passed without being bit-identical" not in out and "numbers and limits" not in out
    assert _section(out, "Generated text")[:3] == ["### Generated text", "",
                                                   f"`{WF}`, identical to its reference:"]
    assert _pre_blocks(out) == [T_REF] and "**differs**" not in out
    # The table still says that it changed from the run before.
    assert f"| identical text | changed from {T_PREV} |" in out
    # An image workflow in that state keeps its place: it has limits.
    image = _cell("flux_dev_t2i", vs_previous={"identical": False, "mean_mse": 0.5})
    assert "### Passed without being bit-identical" in _report(_tree(_summary([image])))


def test_a_difference_that_cannot_be_shown_and_a_file_under_another_name_are_said():
    # Zero-width and control characters are not shown, so the two texts read the same.
    hidden = T_REF.replace("regression", "regres\u200bsion").replace("person", "per\x0cson")
    cell = _text_cell("fail", hidden, detail="new_drift")
    out = _report(_text_files(cell, hidden))
    lines = out.split("\n")
    assert (f"| Reference `{T_TAG}` | **differs** | 93.3 % | line 1, column 9 | 210 here, 208 in "
            "the reference | 3 |") in lines
    assert ("The two texts differ only in characters that are not shown here (control or "
            "zero-width characters, or a tab where the other text has blanks). Compare the two "
            "files.") in lines
    assert "Difference from the reference" not in out and _pre_blocks(out) == [T_REF, T_REF]
    # The same file in both, and a second one that the run did not write: nothing to mark,
    # and the table names what is missing.
    fewer = {**_text_vs(T_REF, T_REF), "identical": False, "similarity": 0.999999,
             "file_count_ref": 2}
    out = _report(_text_files(_text_cell("fail", vs_golden=fewer, detail="new_drift")))
    assert "**differs** (**1 file here, 2 in the reference**)" in out
    assert "differ only in characters" not in out and "Difference from the reference" not in out

    # ComfyUI names the file another way: the reference is there, under its old name.
    moved = {**_text_vs(T_REF, T_REF), "identical": False, "similarity": 0.0, "files_compared": 0}
    cell = _text_cell("fail", vs_golden=moved, detail="new_drift")
    out = _report(_text_files(cell, ref=None, extra={
        f"{P}/golden/{WF}/{T_TAG}/outputs/{WF}_00001_.txt": T_REF}))
    lines = out.split("\n")
    assert (f"| Reference `{T_TAG}` | **differs** (**no text file with the same name as in the "
            "reference**) | 0.0 % |  | 208 | 3 |") in lines
    assert ("The reference has no text file named <code>llm\\_qwen\\_t2t\\_00001.txt</code>, so "
            "the difference is not shown.") in lines
    assert "could not be read" not in out and _pre_blocks(out) == [T_REF]


def test_a_text_with_long_lines_says_that_they_are_folded():
    """An answer is often one long paragraph. The table counts lines and
    columns in the file; what is shown has more and shorter lines."""
    para = " ".join(f"word{i}" for i in range(150))
    cell = _text_cell("fail", para.replace("word75", "other"), ref=para, detail="new_drift")
    out = _report(_text_files(cell, para.replace("word75", "other"), ref=para))
    lines = out.split("\n")
    assert (f"| Reference `{T_TAG}` | **differs** | 99.3 % | line 1, column 516 | "
            f"{len(para) - 1} here, {len(para)} in the reference | 1 |") in lines
    assert DIFF_FOLDED + ":" in lines
    assert (f"<details><summary>Generated text ({len(para) - 1} characters, long lines folded)"
            "</summary>") in lines
    assert (f"<details><summary>Reference text ({len(para)} characters, long lines folded)"
            "</summary>") in lines
    assert all(len(line) <= 100 for block in _pre_blocks(out) for line in block.split("\n"))
    # Cut and folded: both are said.
    out = run_report.render_lane(
        run_report.gather_lane(run_report.DictReader(_text_files(cell, para, ref=para), RESULTS),
                               LANES, SHA, "master", LANE), run_report.Limits(text_chars=200))
    assert f"Generated text ({len(para)} characters, the first 200 shown, long lines folded)" in out
    # Lines that fit are shown as they are, and no heading says otherwise.
    assert "folded" not in _report(_text_files(_text_cell()))


def test_a_text_that_was_not_sent_or_not_written_is_a_failed_comparison():
    lost = {"kind": "text", "error": "output not transferred, over the worker's inline size "
                                     f"limit: {TXT}"}
    cell = _text_cell("fail", vs_golden=lost, vs_previous=lost, output_sha256="9" * 64)
    files = _text_files(cell)
    files[f"{P}/runs/master/{SHA}/{WF}/run.json"] = _text_run(T_REF, truncated=True)
    del files[T_RUN]
    out = _report(files)
    lines = out.split("\n")
    assert ("The generated text could not be compared with the reference. There is no previous "
            "run to tell whether the change is new at this commit.") in lines
    assert (f"| Reference `{T_TAG}` | comparison failed: output not transferred, over the "
            "worker's inline size limit: llm\\_qwen\\_t2t\\_00001.txt |  |  |  |  |") in lines
    assert _pre_blocks(out) == [] and "[text](" not in out
    assert f"| `{WF}` | ❌ output changed |  | comparison failed | comparison failed |" in out
    assert "not published: over the worker's size limit" in out


def test_an_accepted_text_change_is_closed_by_the_text_sha():
    """The lane's reference names the text file's sha, like the run does."""
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    registry = {"lanes": {LANE: {"goldens": {WF: {"tag": "cccccccccccc",
                                                 "sha": _sha256(T_NEW)}}}}}
    out = _report(_text_files(cell, T_NEW, registry=registry, extra={
        f"{P}/golden/{WF}/current.json": {"tag": "cccccccccccc", "blessed_by": "dev",
                                          "blessed_ts": 1794790000, "reason": "accept #17754"}}))
    lines = out.split("\n")
    assert lines[0].startswith("## 🟡 ") or lines[0].startswith("## ✅ ")
    assert f"#### 🟡 `{WF}`: output changed (accepted)" in lines
    assert "1 changed output accepted as the reference after the run." in out
    assert f"It is not the reference this run was compared with (`{T_TAG}`)." in lines
    assert f"| `{WF}` | 🟡 output changed (accepted) | [text]({RAW}/{T_RUN}) |" in out


def test_a_report_from_the_runners_files_shows_the_generated_text(tmp_path):
    """The run could not publish and the results branch cannot be read: the
    text the runner holds is shown, with the numbers of its comparison."""
    workdir = tmp_path / "regression-work"
    (workdir / WF / "outputs").mkdir(parents=True)
    (workdir / WF / "outputs" / TXT).write_bytes(T_NEW.encode("utf-8"))
    cell = _text_cell("fail", T_NEW, detail="new_drift")
    comparison = {k: cell[k] for k in ("verdict", "detail", "vs_golden", "vs_previous",
                                       "thresholds_used", "golden_tag", "previous_commit",
                                       "previous_verdict", "figures")}
    comparison["output_kind"] = "text"
    (workdir / WF / "run.json").write_text(json.dumps(_text_run(T_NEW)), encoding="utf-8")
    (workdir / WF / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")

    rebuilt = run_report.summary_from_workdir(str(workdir), SHA, "master", LANE)
    assert rebuilt["workflows"][WF]["output_kind"] == "text"
    assert rebuilt["workflows"][WF]["output_sha256"] == _sha256(T_NEW)
    assert rebuilt["workflows"][WF]["thumbnail"] is None

    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE,
                                 workdir=str(workdir))
    lines = out.split("\n")
    assert "> [!WARNING]" in lines
    assert f"#### ❌ `{WF}`: output changed (new at this commit)" in lines
    assert "| **differs** | 86.5 % | line 2, column 22 |" in out
    assert _pre_blocks(out) == [T_NEW]
    assert "The reference text could not be read, so the difference is not shown." in lines
    assert "raw.githubusercontent.com" not in out and "[text](" not in out
    # The same run with its summary published: the branch gives the reference and the links.
    files = _text_files(cell, "the published copy is not the one read", ref=T_REF)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(files[f"{P}/runs/master/{SHA}/summary.json"]), encoding="utf-8")
    out = _report(files, summary_file=str(path), workdir=str(workdir))
    assert _pre_blocks(out)[1:] == [T_NEW, T_REF] and DIFF_FOLDED + ":" in out
    assert f"[text]({RAW}/{T_RUN})" in out
    # A file larger than its record says is read up to the report's limit, no further.
    (workdir / WF / "outputs" / TXT).write_bytes(b"x" * (run_report.TEXT_READ_MAX + 5000))
    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE,
                                 workdir=str(workdir))
    assert f"Generated text ({run_report.TEXT_READ_MAX} characters, the first 4000 shown)" in out


def test_an_image_summary_rebuilt_from_the_runner_names_no_kind(tmp_path):
    workdir = tmp_path / "regression-work"
    (workdir / "flux_dev_t2i").mkdir(parents=True)
    (workdir / "flux_dev_t2i" / "run.json").write_text(json.dumps(_run("flux_dev_t2i")))
    (workdir / "flux_dev_t2i" / "comparison.json").write_text(json.dumps(
        {"verdict": "pass", "vs_golden": IDENT, "vs_previous": IDENT}))
    cell = run_report.summary_from_workdir(str(workdir), SHA, "master", LANE)["workflows"][
        "flux_dev_t2i"]
    assert "output_kind" not in cell and cell["output_sha256"] == "1" * 64
    assert list(cell)[-1] == "custom_nodes"


def test_the_kind_of_a_cell_is_the_one_it_names_or_the_one_its_comparison_carries():
    assert run_report.cell_kind({}) == run_report.cell_kind(_cell("flux_dev_t2i")) == "image"
    assert run_report.cell_kind({"output_kind": "text"}) == "text"
    # A summary cell written without the key still says so in its metrics.
    assert run_report.cell_kind({"vs_golden": {"kind": "text", "identical": True}}) == "text"
    assert run_report.cell_kind({"vs_golden": None, "vs_previous": {"kind": "text"}}) == "text"
    for odd in ("audio", ["text"], 3, None):
        assert run_report.cell_kind({"output_kind": odd, "vs_golden": IDENT}) == "image"
    run = {"outputs": [{"filename": "a_00001_.png", "sha256": "p"},
                       {"filename": "a_00001.txt", "sha256": "t"}]}
    assert run_report.output_sha(run, "image") == "p" and run_report.output_sha(run, "text") == "t"


def test_text_metric_cells_by_shape():
    cells = run_report.text_metric_cells
    assert cells(None) == ["not compared", "", "", "", ""]
    assert cells({"kind": "text", "error": "no text files to compare (ref=1, cand=0)"})[0] == (
        "comparison failed: no text files to compare (ref=1, cand=0)")
    assert cells(_text_vs(T_REF, T_REF)) == ["identical", "", "", "208", "3"]
    # A blank-only change is never shown as fully alike.
    assert cells(_text_vs("a b", "a  b"))[:3] == ["**differs**", "over 99.9 %", "line 1, column 3"]
    # A missing file is said where the result is, and never passes for a small difference.
    fewer = {**_text_vs(T_REF, T_REF), "identical": False, "similarity": 0.999999,
             "file_count_ref": 2, "file_count_cand": 1}
    assert cells(fewer, {"min_text_similarity": 0.5})[0] == (
        "**differs** (**1 file here, 2 in the reference**)")
    assert cells({**fewer, "file_count_cand": 3})[0] == (
        "**differs** (**3 files here, 2 in the reference**)")
    # So is a file that has no pair of the same name, with as many files on both sides.
    renamed = {**fewer, "file_count_cand": 2, "files_compared": 1}
    assert cells(renamed, {"min_text_similarity": 0.5})[0] == (
        "**differs** (**only 1 of 2 text files with the same name as in the reference**)")
    assert cells({**renamed, "files_compared": 0, "similarity": 0.0})[:2] == [
        "**differs** (**no text file with the same name as in the reference**)", "0.0 %"]
    # Bold marks what fails. The previous run and the noise floor are not what a run is judged by.
    assert cells(_text_vs(T_REF, T_NEW), None, "the previous run", judged=False)[0] == "differs"
    assert cells(fewer, None, "the previous run", judged=False)[0] == (
        "differs (1 file here, 2 in the previous run)")
    # With several files the place names its file.
    two = {**_text_vs(T_REF, T_NEW), "file_count_ref": 2, "file_count_cand": 2}
    assert cells(two)[2] == "line 2, column 22 of <code>llm\\_qwen\\_t2t\\_00001.txt</code>"
    # Odd values never raise and never print.
    assert cells({"kind": "text", "identical": False, "similarity": "x", "first_diff": "y",
                  "chars_cand": [1], "lines_ref": None}) == ["**differs**", "", "", "", ""]


def test_main_never_prints_generated_text(tmp_path, monkeypatch, capsys):
    step = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    cell = _text_cell("fail", T_EVIL, detail="new_drift")
    reader = run_report.DictReader(_text_files(cell, T_EVIL), RESULTS)
    assert run_report.main(_args("--out", "-"), reader=reader) == 0
    assert "&#58;:error::not a command" in step.read_text(encoding="utf-8")
    printed = capsys.readouterr()
    assert re.fullmatch(r"run report: wrote \d+ bytes to the job summary\n", printed.out)
    assert printed.err == ""


def test_readers_return_the_start_of_a_text_file(tmp_path):
    files = {"a.txt": "héllo wörld", "b.json": {"x": 1}}
    reader = run_report.DictReader(files, RESULTS)
    assert reader.read_text("a.txt") == "héllo wörld" and reader.read_text("a.txt", 3) == "hé"
    assert reader.read_text("b.json") is None and reader.read_text("nope.txt") is None
    assert run_report.Reader().read_text("a.txt") is None

    raw = b"caf\xc3\xa9 \xff\xfe end\r\nsecond line\n"
    huge = run_report.TEXT_READ_MAX + 300_000
    remote = make_remote(tmp_path, {**_tree(_summary([_cell("flux_dev_t2i")])),
                                    f"{P}/golden/{WF}/{T_TAG}/outputs/{TXT}": raw,
                                    f"{P}/golden/{WF}/{T_TAG}/outputs/big.txt": b"x" * 5000,
                                    f"{P}/golden/{WF}/{T_TAG}/outputs/huge.txt": b"y" * huge,
                                    f"{P}/golden/{WF}/{T_TAG}/outputs/empty.txt": b""})
    reader = run_report.GitReader(_checkout(tmp_path, remote))
    path = f"{P}/golden/{WF}/{T_TAG}/outputs/{TXT}"
    # The blob as the branch holds it; bytes that are not UTF-8 are replaced, never raised.
    assert output_kinds.normalise_text(reader.read_text(path)) == "café �� end\nsecond line"
    assert reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs/big.txt", 100) == "x" * 100
    assert len(reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs/big.txt")) == 5000
    # A blob of any size is read up to the limit and no further: git is stopped there.
    assert reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs/huge.txt") == (
        "y" * run_report.TEXT_READ_MAX)
    assert reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs/empty.txt") == ""
    assert reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs/none.txt") is None  # absent
    assert reader.read_text(f"{P}/golden/{WF}/{T_TAG}/outputs") is None           # a directory
    assert reader.read_text(f"{P}/golden/{WF}") is None
    assert not (tmp_path / "checkout" / "regression").exists()  # nothing is checked out
