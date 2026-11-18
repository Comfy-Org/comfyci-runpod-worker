"""run_report: the job summary of a GPU regression run, rendered from an
in-memory results tree (no network), and the git-backed reader against a local
bare repo."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
ONCE_MORE = ("This was attempt 1 of 2: the cron tests the commit once more while it is still the "
             "newest commit on master.")


@pytest.mark.parametrize("attempt, retry", [
    # Models that are not on the volume: another run right away would not find
    # them either, so the cron leaves the commit alone (regression_skip.NO_RETRY).
    (1, "Its models are not on the volume yet. Another run right away would end the same way, "
        "so the cron does not test the commit again. Dispatch **GPU regression** with this "
        "commit to test it again."),
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
    summary = _summary(cells, attempt=attempt)
    out = _report(_tree(summary, runs=runs))
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
    # The Tested row names the attempt where it tells something: a retry
    # happened. "attempt 1 of 2" would promise a second run of a commit whose
    # only untested workflow the cron leaves alone.
    if attempt == 1:
        assert "attempt 1" not in out and f"commit · [Actions run]({RUN_URL}) |" in out
    else:
        named = f"attempt {attempt} of 2" if attempt == 2 else f"attempt {attempt}"
        assert f"· {named} · [Actions run]({RUN_URL}) |" in out
    # The report says what the cron does: neither tests the commit again.
    assert not run_report.regression_skip.decide(SHA, summary)[0]
    assert ONCE_MORE not in out


def test_the_cron_retries_a_commit_for_a_workflow_that_another_run_can_test():
    """The sentence a per-commit report always had, for a status that is not
    in regression_skip.NO_RETRY. One such workflow is enough: the cron tests
    the whole commit again, the workflow without its models included."""
    for status in ("worker_error", "server_start_timeout", "checkout_error"):
        cells = [_cell("flux_dev_t2i"), _cell("sdxl_t2i", "infra_error", worker_status=status)]
        summary = _summary(cells)
        out = _report(_tree(summary))
        assert (f"Worker status: <code>{status.replace('_', chr(92) + '_')}</code>. This is not "
                f"a verdict on the commit. {ONCE_MORE}") in out, status
        assert run_report.regression_skip.decide(SHA, summary)[0]
        # A second attempt can follow, so the Tested row names this one.
        assert f"· attempt 1 of 2 · [Actions run]({RUN_URL}) |" in out
    mixed = [_cell("flux_dev_t2i", "infra_error", worker_status="worker_error"),
             _cell("sdxl_t2i", "infra_error", worker_status="missing_model")]
    summary = _summary(mixed)
    out = _report(_tree(summary))
    assert out.count(ONCE_MORE) == 2 and "not on the volume" not in out
    assert run_report.regression_skip.decide(SHA, summary)[0]
    # A cell from before the status was recorded is taken as one a run can test.
    old = _cell("sdxl_t2i", "infra_error", worker_status=None)
    assert ONCE_MORE in _report(_tree(_summary([old])))


NO_ATTEMPT = "This run is no attempt of the lane, so it changes nothing for the cron."


def test_a_run_that_is_no_attempt_of_the_lane_says_nothing_about_the_crons_retry(tmp_path):
    """A run that names its workflows, or that publishes nothing, is not
    counted: its summary records the attempts of the runs before it. The page
    read that number as its own. A first look with `publish` off at a commit
    with the lane's clean result said "attempt 1 of 2: the cron tests the
    commit once more", and the cron skips that commit."""
    untested = _cell("sdxl_t2i", "infra_error", worker_status="worker_error")
    first = ("An infrastructure problem stopped this workflow. Worker status: "
             "<code>worker\\_error</code>. This is not a verdict on the commit. ")

    def tested(out):
        return next(line for line in out.split("\n") if line.startswith("| Tested |"))

    # Publish off: the summary is on the runner alone, with the attempts so far.
    published = _summary([_cell("flux_dev_t2i")])
    files = _tree(published, extra={f"{P}/runs/master/{SHA}/attempt.json": {"attempt": 1}})
    look = _summary([untested], run_ts=published["run_ts"] + 900, scope=["sdxl_t2i"])
    for local in (look, {k: v for k, v in look.items() if k != "scope"}):
        path, workdir = _write_local(tmp_path / str("scope" in local), local, {
            "sdxl_t2i": _run("sdxl_t2i", "worker_error", outputs=[])})
        out = _report(files, summary_file=path, workdir=workdir)
        assert first + NO_ATTEMPT in out
        assert "attempt" not in tested(out) and "attempt 1 of 2" not in out
        assert "the cron tests the commit once more" not in out
        assert "the cron does not retry it" not in out
    # What the cron does at that commit: nothing.
    assert not run_report.regression_skip.decide(SHA, published, None, False, {"attempt": 1})[0]
    # A run of named workflows that published (`scope`): no attempt either,
    # whatever number its summary carries.
    for attempt in (1, 2, 3):
        out = _report(_tree(_summary([untested], scope=["sdxl_t2i"], attempt=attempt)))
        assert first + NO_ATTEMPT in out and "attempt" not in tested(out), attempt
    held = _cell("sdxl_t2i", "infra_error", worker_status="missing_model")
    out = _report(_tree(_summary([held], scope=["sdxl_t2i"])))
    assert NO_ATTEMPT in out and "so the cron does not test the commit again" not in out
    # The lane's own run keeps its words, and so does a run whose summary may
    # well be published: the results branch could not be read.
    assert ONCE_MORE in _report(_tree(_summary([untested])))
    path, workdir = _write_local(tmp_path / "unread", _summary([untested]))
    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE,
                                 summary_file=path, workdir=workdir)
    assert ONCE_MORE in out and NO_ATTEMPT not in out and "· attempt 1 of 2" in tested(out)
    # A nightly or weekly lane has no cron that retries: its words do not change.
    night = _smoke_cell(MUSIC, verdict="infra_error", worker_status="worker_error")
    out = _night_report([night], scope=[MUSIC])
    assert NEXT_RUN in out and NO_ATTEMPT not in out and "attempt" not in tested(out)


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
    assert "1 of 1 workflow failed" in out
    # A run that published nothing is no attempt of the lane: the page names none.
    assert "attempt 2" not in out
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
    # A report that hangs must not hold the group every later commit waits in.
    assert "\n        timeout-minutes: 10\n" in lane
    assert "\n        timeout-minutes: 10\n" in overview
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
LINE_TITLE = ("**Difference from the reference**, line by line, with <code>-</code> before a "
              "removed line and <code>+</code> before an added one")
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
    # 400 such lines are too many words to mark one by one: the difference is a
    # line diff, and its title explains the signs of a line diff.
    assert LINE_TITLE + ", cut to its first 8000 characters:" in lines
    assert "[-…-]" not in out
    diff, generated, reference = _pre_blocks(out)
    assert diff.split("\n")[:3] == ["--- reference", "+++ candidate", "@@ -1,400 +1,400 @@"]
    assert diff.split("\n")[3] == "-LINE 0000 OF A TEXT THAT GOES ON AND ON"
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
    assert LINE_TITLE + ", cut to its first 2000 characters:" in cut
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
    # No pair of files was compared, so no similarity is shown: not as "0.0 %" either.
    assert (f"| Reference `{T_TAG}` | **differs** (**no text file with the same name as in the "
            "reference**) |  |  | 208 | 3 |") in lines
    assert (f"| `{WF}` | ❌ output changed (new at this commit) | [text]({RAW}/{T_RUN}) | text "
            f"file under another name | identical to {T_PREV} |") in out
    assert "0.0 %" not in out and "% similar" not in out
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
    for odd in ("video", ["text"], 3, None):
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
        "**differs** (**no text file with the same name as in the reference**)", ""]
    # A similarity that is not a share of words says what it is a share of. The basis
    # is read from results: one that is not known adds nothing.
    by_lines = {**_text_vs(T_REF, T_NEW), "similarity_basis": "lines"}
    assert cells(by_lines)[1] == "86.5 % by lines"
    assert cells(by_lines, {"min_text_similarity": 0.98})[1] == (
        "**86.5 % by lines** (limit ≥ 98 %)")
    assert cells({**by_lines, "similarity_basis": "prefix"})[1] == "86.5 % by the shared start"
    for odd in ("words", "<b>lines</b>", ["lines"], None):
        assert cells({**by_lines, "similarity_basis": odd})[1] == "86.5 %"
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


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups: not on Windows")
def test_a_text_read_ends_when_the_fetch_that_git_started_stalls(tmp_path, monkeypatch):
    """A blob that is not local is fetched by a child of `git cat-file`, and
    that child holds the pipe the text is read from. Stopping git alone leaves
    the read waiting for as long as the fetch stalls, so the whole group is
    stopped. The stand-in for git starts such a child and waits for it."""
    started = tmp_path / "started"
    fetch = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text('x'); time.sleep(60)"
    stand_in = [sys.executable, "-c", "import subprocess, sys; subprocess.Popen("
                "[sys.executable, '-c', sys.argv[1], sys.argv[2]]).wait()", fetch, str(started)]
    real = subprocess.Popen
    monkeypatch.setattr(run_report.subprocess, "Popen", lambda args, **kw: real(stand_in, **kw))
    monkeypatch.setattr(run_report, "GIT_TIMEOUT_S", 2)
    reader = run_report.GitReader.__new__(run_report.GitReader)  # no checkout: git is replaced
    reader.workdir, reader.sha = str(tmp_path), "0" * 40
    begun = time.monotonic()
    assert reader.read_text(T_GOLDEN) is None
    assert time.monotonic() - begun < 30 and started.exists()


def test_a_similarity_that_is_not_a_share_of_words_says_what_it_is():
    """Above 5,000 words the similarity is a share of lines: the report names
    that next to the number, where it would read as a share of words."""
    words = " ".join(f"w{i}" for i in range(output_kinds.WORDS_MAX + 1))
    ref, new = words + "\nlast", words + "\nother"
    cell = _text_cell("fail", new, ref=ref, detail="new_drift",
                      thresholds_used={"min_text_similarity": 0.98})
    assert (cell["vs_golden"]["similarity"], cell["vs_golden"]["similarity_basis"]) == (
        0.5, "lines")
    out = _report(_text_files(cell, new, ref=ref))
    assert (f"| Reference `{T_TAG}` | **differs** | **50.0 % by lines** (limit ≥ 98 %) | line 2, "
            "column 1 |") in out
    assert "| 50.0 % similar by lines · differs from line 2 |" in out
    # Too many words to mark one by one as well: the title is the one of a line diff.
    assert LINE_TITLE + ", long lines folded, cut to its first 8000 characters:" in out


# -- smoke level -----------------------------------------------------------------------------
NIGHTLY = "py313-torch2.14.1-cu130-nightly"
NP = f"regression/lanes/{NIGHTLY}"
# A registry with a lane that runs once a night, like the nightly lane of the registry.
NIGHT_LANES = {"primary": LANE, "lanes": {**LANES["lanes"], NIGHTLY: {
    "label": "nightly · py3.13 / torch 2.14.1 / cu130 / RTX 4090", "cadence": "nightly",
    "blocking": False, "role": "lane", "layout": 2, "prefix": NP, "endpoint_id": "ep-night"}}}
LAST, HEAD = "e9afbcc" + "0" * 33, "05e878f" + "3" * 33
ARTIFACT = RUN_URL + "/artifacts/99"
SILENT = "silent or constant: 138 bytes of audio a second, under 4000"
MUSIC, MESH, CLIP = "ace_step_15_t2a", "hunyuan3d_2_1_t23d", "wan_clip"
FLAC = f"{MUSIC}_00001.flac"
# The two names as inline Markdown shows them: every underscore escaped.
FLAC_MD, MESH_MD = FLAC.replace("_", "\\_"), MESH.replace("_", "\\_")
SMOKE_HEADER = ("| Workflow | Result | Output | Checked | Exec s (usual range) | Server start s "
                "(checkout s) | Peak VRAM GB | Peak RSS GB | Output sha256 |")
MIXED_HEADER = SMOKE_HEADER.replace("| Checked |", "| vs reference | vs previous | Reference | "
                                                   "Checked |")
FILES_HEADER = ["| File | Type | Size | Measured |", "|---|---|---|---|"]


def _png(name, **kw):
    return {"name": name, "type": "image", "bytes": 156156, "sha256": "9f" * 32, "width": 1024,
            "height": 1024, "std": 69.17, "flat_share": 0.6409, **kw}


def _glb(name, **kw):
    return {"name": name, "type": "mesh", "bytes": 1292352, "sha256": "c4" * 32,
            "vertices": 35781, "faces": 71836, "extent": [1.3743, 1.5627, 1.2226], **kw}


def _flac(name=FLAC, **kw):
    return {"name": name, "type": "audio", "bytes": 3528044, "sha256": "10" * 32,
            "sample_rate": 48000, "channels": 2, "bits": 16, "seconds": 30.0,
            "bytes_per_second": 117330, "pcm_md5": "0005ea55925db0b0cece53ae24268b4f", **kw}


def _smoke_cell(wf, files=(), problems=(), verdict=None, notes=(), **kw):
    """The summary cell of a smoke-level workflow, shaped as run_regression
    builds it: nothing is compared, and `level` and `smoke` say what was checked."""
    verdict = verdict or ("fail" if problems else "pass")
    names = sorted(f["name"] for f in files)
    main = next((f for f in files if f["name"].endswith((".png", ".flac", ".txt"))), {})
    thumb = next((n[:-4] + ".webp" for n in names if n.endswith(".png")), None)
    return _cell(wf, verdict, **{
        "vs_golden": None, "vs_previous": None, "thresholds_used": None, "golden_tag": None,
        "previous_commit": None, "previous_verdict": None, "output_sha256": main.get("sha256"),
        "thumbnail": thumb, "level": "smoke",
        "smoke": {"files": list(files), "problems": list(problems), "notes": list(notes)}, **kw})


def _silent(**kw):
    """An audio workflow that wrote 30 s of silence."""
    quiet = _flac(bytes=11040, bytes_per_second=138)
    return _smoke_cell(MUSIC, [quiet], [{"file": FLAC, "check": "silent", "text": SILENT}],
                       output_kind="audio", **kw)


def _smoke_run(cell):
    """The run record behind a smoke cell: the output files the worker listed."""
    files = [f for f in (cell.get("smoke") or {}).get("files") or []
             if isinstance(f, dict) and isinstance(f.get("name"), str)]
    return _run(cell["workflow_id"], cell["worker_status"], outputs=[
        {"filename": f["name"], "bytes": f.get("bytes"), "sha256": f.get("sha256"),
         **({"truncated": True} if f.get("sent") is False else {})} for f in files])


def _index_cell(cell):
    out = {"v": cell["verdict"], "d": cell.get("detail"), "sha": cell.get("output_sha256"),
           "exec": (cell.get("timings") or {}).get("prompt_exec_s"), "g": cell.get("golden_tag")}
    return {**out, "l": "smoke"} if cell.get("level") == "smoke" else out


def _smoke_files(cells, lane=LANE, commit=SHA, before=None, first_bad=None, extra=None, **kw):
    """The results tree of a run with smoke-level workflows. `before` holds
    the lane's earlier index entries (default: one, where everything passed)."""
    summary = _summary(cells, lane=lane, commit=commit, **kw)
    run_dir = f"{NIGHT_LANES['lanes'][lane]['prefix']}/runs/master/{commit}"
    files = {f"{run_dir}/summary.json": summary}
    for cell in cells:
        files[f"{run_dir}/{cell['workflow_id']}/run.json"] = _smoke_run(cell)
    rng = summary.get("tested_range") or {}
    this = {**_entry(commit, 1794702636, {c["workflow_id"]: _index_cell(c) for c in cells},
                     run=4242, n=rng.get("commits_between") or 1), "prev": rng.get("prev")}
    if before is None:
        before = [_entry(rng.get("prev") or PREV, 1794700000,
                         {c["workflow_id"]: {**_index_cell(c), "v": "pass"} for c in cells},
                         pr=17753)]
    files[f"regression/index/master/{lane}.json"] = _head([this, *before], first_bad)
    files["regression/index/lanes.json"] = {"lanes": {lane: {"goldens": {}}}}
    files.update(extra or {})
    return files


def _smoke_report(files, lane=LANE, commit=SHA, **kw):
    reader = run_report.DictReader(files, RESULTS)
    return run_report.lane_report(reader, NIGHT_LANES, commit, "master", lane, **kw)


def _nightly(cells, **kw):
    """A report of a nightly run that covers 7 commits since LAST."""
    rng = {"prev": LAST, "commits_between": 7, "compare_url": f"{COMFY}/compare/{LAST}...{HEAD}"}
    files = _smoke_files(cells, lane=NIGHTLY, commit=HEAD, tested_range=rng, **kw)
    return _smoke_report(files, lane=NIGHTLY, commit=HEAD)


def _block(out, wf):
    """The lines of one workflow's block under "What needs attention"."""
    lines = _section(out, "What needs attention")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"#### ❌ `{wf}`"))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#### ")),
               len(lines))
    return [line for line in lines[start:end] if line]


def test_a_lane_of_smoke_passes_reads_smoke_pass_and_has_nothing_not_passing():
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png"), _png(f"{MESH}_00002_.png"),
                              _glb(f"{MESH}_mesh_00001_.glb")])
    music = _smoke_cell(MUSIC, [_flac()], output_kind="audio")
    out = _smoke_report(_smoke_files([mesh, music]))
    lines = out.split("\n")
    assert lines[0] == ("## ✅ GPU regression · py3.13 / torch 2.14.1 / cu130 / RTX 4090 · "
                        "`cccccccc`")
    # The sentence that says what a smoke pass is worth stands right under the headline.
    assert lines[2:5] == ["All 2 workflows passed their smoke checks.", "",
                          run_report.SMOKE_NOTE]
    assert "What needs attention" not in out
    assert lines[lines.index("### All workflows (2)") + 2] == SMOKE_HEADER
    run_dir = f"{RAW}/{P}/runs/master/{SHA}"
    assert (f'| `{MESH}` | ✅ smoke pass | <a href="{run_dir}/{MESH}/outputs/{MESH}_00001_.png">'
            f'<img src="{run_dir}/{MESH}/outputs/thumbs/{MESH}_00001_.webp" width="96" '
            f'alt="{MESH} output"></a> | 2 images 1024x1024 · mesh 35,781 vertices | 98.1 | 12.4 '
            "| 21.3 | 33.7 | `9f9f9f9f9f9f` |") in lines
    # No PNG: a link named after what the output is.
    assert (f"| `{MUSIC}` | ✅ smoke pass | [audio]({run_dir}/{MUSIC}/outputs/{FLAC}) | 30.0 s · "
            "48000 Hz · stereo | 98.1 | 12.4 | 21.3 | 33.7 | `101010101010` |") in lines
    # The file list says what was measured on each file.
    assert (f"| `{MUSIC}` | [{FLAC_MD}]({run_dir}/{MUSIC}/outputs/{FLAC}) | 3.36 MB | "
            f"`{'10' * 32}` | 30.0 s · 48000 Hz · 2 channels · 117,330 bytes/s |") in lines
    assert (f"| `{MESH}` | [{MESH_MD}\\_mesh\\_00001\\_.glb]({run_dir}/{MESH}/outputs/"
            f"{MESH}_mesh_00001_.glb) | 1.23 MB | `{'c4' * 32}` | 35,781 vertices · 71,836 faces "
            "· box 1.37 x 1.56 x 1.22 |") in lines
    history = _section(out, "Recent commits on this lane")
    assert next(row for row in history if "**this report**" in row).endswith(
        "| ✅ smoke pass |  |")
    assert f"{run_report.LEGEND} {run_report.SMOKE_LEGEND}" in lines
    assert "reference level (the main tests)" in run_report.SMOKE_LEGEND
    # Nothing offers to approve an output: a smoke-level workflow has no reference.
    assert "golden-baseline.yml" not in out
    assert "The workflow passed its smoke checks." in _smoke_report(_smoke_files([music]))


def test_a_page_without_a_smoke_result_is_the_page_it_was():
    out = _report(_tree(_summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i")])))
    lines = out.split("\n")
    assert run_report.LEGEND in lines and run_report.legend() == run_report.LEGEND
    assert "smoke" not in out and "Checked" not in out and "first run" not in out
    assert lines[2:4] == ["All 2 workflows match their reference.", ""]
    assert lines[4].startswith("|  |  |")
    overview = _overview(_overview_files())
    assert run_report.LEGEND in overview.split("\n") and "smoke" not in overview


def test_the_headline_counts_smoke_passes_apart_and_failures_by_where_they_started():
    image = [_cell("flux_dev_t2i"), _cell("sdxl_t2i")]
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    out = _smoke_report(_smoke_files([*image, mesh]))
    assert "No failures: 2 match their reference, 1 passed its smoke checks." in out.split("\n")
    # The roll-up of a mixed lane says how many of each level it stands for.
    history = _section(out, "Recent commits on this lane")
    assert all(row.endswith("| ✅ pass (2 reference, 1 smoke) |  |")
               for row in history if row.startswith("| [`"))
    assert out.startswith("## ✅ ")
    # One reference match alone, next to a smoke failure that is new here.
    out = _smoke_report(_smoke_files([image[0], mesh, _silent()]))
    assert ("1 of 3 workflows failed: 1 new at this commit. 1 matches its reference, 1 passed "
            "its smoke checks.") in out.split("\n")
    # A changed output is compared with the previous run, so it is new at this commit
    # whenever the run covers one commit: where its own last result stands says nothing.
    drift = _cell("sdxl_t2i", "fail", detail="new_drift", vs_previous=DIFF, prior_commit=FIRST)
    out = _smoke_report(_smoke_files([drift, mesh]))
    assert "1 of 2 workflows failed: 1 new at this commit. 1 passed its smoke checks." in (
        out.split("\n"))
    # A nightly run: its failures by where they started, then what was not tested.
    first = _smoke_cell("new_wf", problems=[{"file": None, "check": "missing", "text": "x"}],
                        prior_verdict=None, prior_commit=None)
    standing = _smoke_cell("old_wf", [_png("old_wf_00001_.png")], [
        {"file": "old_wf_00001_.png", "check": "blank", "text": "blank"}],
        prior_verdict="fail", prior_commit=LAST)
    untested = _smoke_cell("slow_wf", verdict="infra_error", prior_commit=LAST)
    before = [_entry(LAST, 1794616236, {MESH: "pass", MUSIC: "pass", "old_wf": "fail",
                                        "slow_wf": "pass"})]
    out = _nightly([mesh, _silent(prior_commit=LAST), standing, first, untested], before=before)
    assert ("3 of 5 workflows failed: 1 new in this run, 1 inherited, 1 on its first run. "
            "1 was not tested. 1 passed its smoke checks.") in out.split("\n")
    assert out.startswith("## ❌ ")
    out = _nightly([mesh, untested, {**untested, "workflow_id": "slower_wf"}])
    assert "No failures: 1 passed its smoke checks. 2 were not tested." in out.split("\n")
    assert out.startswith(f"## {run_report.WARN} ")


def test_a_smoke_failure_shows_each_problem_the_files_and_where_it_started():
    cell = _silent(prior_commit=LAST)
    ranges = {MUSIC: {"prev": LAST, "commits_between": 7, "compare_url": "u"}}
    out = _nightly([cell, _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])], ranges=ranges)
    run_dir = f"{RAW}/{NP}/runs/master/{HEAD}"
    last = f"[`e9afbcc`]({COMFY}/commit/{LAST})"
    assert _block(out, MUSIC) == [
        f"#### ❌ `{MUSIC}`: smoke check failed (new since `e9afbcc`, 7 commits)",
        "The workflow finished, and its output did not pass the smoke checks.",
        f"- <code>{FLAC_MD}</code>: {SILENT}",
        *FILES_HEADER,
        f"| [{FLAC_MD}]({run_dir}/{MUSIC}/outputs/{FLAC}) | audio | 10.8 KB "
        "| 30.0 s · 48000 Hz · 2 channels · 138 bytes/s |",
        f"It last passed at {last}. 7 commits since then: [compare]({COMFY}/compare/{LAST}..."
        f"{HEAD}). To test one commit of that range (needs write access to the worker "
        "repository):",
        "```",
        "gh workflow run nightly-weekly.yml --repo Comfy-Org/comfyci-runpod-worker "
        f"-f cadence=nightly -f commit=<sha> -f workflows={MUSIC}",
        "```"]
    link = f"[audio]({run_dir}/{MUSIC}/outputs/{FLAC})"
    assert f"| `{MUSIC}` | ❌ smoke check failed | {link} | 30.0 s · 48000 Hz · stereo |" in out
    assert "golden-baseline.yml" not in out  # nothing to approve at smoke level
    # Without a range of its own the run's range is the workflow's, when both start
    # at the commit of its last result.
    assert f"#### ❌ `{MUSIC}`: smoke check failed (new since `e9afbcc`, 7 commits)" in _nightly(
        [cell])
    # Its last result is older than the run's range: how many commits is not known.
    older = _nightly([_silent(prior_commit=FIRST)])
    assert f"#### ❌ `{MUSIC}`: smoke check failed (new since `aaaaaaa`)" in older
    assert (f"It last passed at [`aaaaaaa`]({COMFY}/commit/{FIRST}). The commits since then: "
            f"[compare]({COMFY}/compare/{FIRST}...{HEAD}). To test one commit") in older

    # On the per-commit lane a run of one commit names that commit, and no command.
    out = _smoke_report(_smoke_files([_silent()]))
    assert _block(out, MUSIC)[0] == f"#### ❌ `{MUSIC}`: smoke check failed (new at this commit)"
    assert _block(out, MUSIC)[-1] == f"It last passed at [`{PREV[:7]}`]({COMFY}/commit/{PREV})."
    assert "gh workflow run" not in out
    assert "1 of 1 workflow failed: 1 new at this commit." in out.split("\n")
    # It did not finish before, and finishes now with a bad output: new, said as it is.
    out = _smoke_report(_smoke_files([_silent(prior_verdict="execution_error")]))
    assert _block(out, MUSIC)[-1] == (
        f"Its last result before this one, at [`{PREV[:7]}`]({COMFY}/commit/{PREV}), was an "
        "execution error: it finishes now, and its output does not pass.")


def test_a_smoke_failure_on_the_per_commit_lane_is_placed_by_its_own_last_result():
    """Nothing is compared with the previous run, so the run's range is the
    failure's only when the workflow's last result stands at its start."""
    rng = {"prev": PREV, "commits_between": 3}
    out = _smoke_report(_smoke_files([_silent()], tested_range=rng))
    block = _block(out, MUSIC)
    assert block[0] == f"#### ❌ `{MUSIC}`: smoke check failed (new since `bbbbbbb`, 3 commits)"
    assert block[-1] == (f"It last passed at [`bbbbbbb`]({COMFY}/commit/{PREV}). 3 commits since "
                         f"then: [compare]({COMFY}/compare/{PREV}...{SHA}).")
    assert "1 of 1 workflow failed: 1 new in this run." in out.split("\n")
    # The command that tests one commit of a range is for a nightly or weekly lane.
    assert "gh workflow run" not in out
    # Its last result is older than the run's one commit: the failure may be older too.
    out = _smoke_report(_smoke_files([_silent(prior_commit=FIRST)]))
    assert _block(out, MUSIC)[0].endswith("smoke check failed (new since `aaaaaaa`)")
    assert "1 of 1 workflow failed: 1 new in this run." in out.split("\n")
    # No commit is recorded for its last result: the words of the run stand in.
    out = _smoke_report(_smoke_files([_silent(prior_commit=None)]))
    assert _block(out, MUSIC)[0].endswith("smoke check failed (new at this commit)")
    assert _block(out, MUSIC)[-1].startswith("| ")  # no sentence about a commit it cannot name
    assert "1 of 1 workflow failed: 1 new at this commit." in out.split("\n")
    out = _smoke_report(_smoke_files([_silent(prior_commit=None)], tested_range=rng))
    assert _block(out, MUSIC)[0].endswith("smoke check failed (new in this run)")
    # A range the summary gives for the workflow itself wins over the run's.
    own = {MUSIC: {"prev": FIRST, "commits_between": 5}, "other": {"commits_between": 9}}
    out = _smoke_report(_smoke_files([_silent()], tested_range=rng, ranges=own))
    assert _block(out, MUSIC)[0].endswith("smoke check failed (new since `aaaaaaa`, 5 commits)")
    assert run_report.own_range({"ranges": {MUSIC: {"commits_between": 1}}}, MUSIC,
                                {"commit": PREV}) == {"prev": PREV, "n": 1}
    assert run_report.own_range(None, MUSIC, {}) == {"prev": None, "n": None}


def test_a_smoke_failure_reads_no_reference_and_a_page_knows_of_older_smoke_rows():
    class Counting(run_report.DictReader):
        seen = []

        def read_json(self, path):
            self.seen.append(path)
            return super().read_json(path)

        def prefetch(self, paths):
            self.seen.extend(paths)

    files = _smoke_files([_silent(golden_tag="v0.42.0"), _cell("flux_dev_t2i")], extra={
        f"{P}/golden/{MUSIC}/current.json": {"tag": "v0.42.0"}})
    out = run_report.lane_report(Counting(files, RESULTS), NIGHT_LANES, SHA, "master", LANE)
    assert f"#### ❌ `{MUSIC}`: smoke check failed" in out and Counting.seen
    assert not [path for path in Counting.seen if "/golden/" in path]
    assert "Reference:" not in out and "v0.42.0` |  | 30.0 s" not in out
    # The level of an old cell that does not name it is the one its index cell names.
    cell = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    files = _smoke_files([cell])
    del files[f"{P}/runs/master/{SHA}/summary.json"]["workflows"][MESH]["level"]
    out = _smoke_report(files)
    assert f"| `{MESH}` | ✅ smoke pass |" in out and "The workflow passed its smoke checks." in out
    column = run_report.LaneColumn(
        lane_id=LANE, lane=LANES["lanes"][LANE], references={},
        summary=files[f"{P}/runs/master/{SHA}/summary.json"],
        head={}, entries=run_report.entries_of(files[f"regression/index/master/{LANE}.json"]))
    assert column.states(SHA) == {MESH: "smoke_pass"}
    # A run without a smoke row on a lane whose history has one: the page still shows
    # the smoke words, so it explains them.
    was_smoke = [_entry(PREV, 1794700000, {"flux_dev_t2i": {"v": "fail", "l": "smoke"}})]
    out = _smoke_report(_smoke_files([_cell("flux_dev_t2i")], before=was_smoke))
    lines = out.split("\n")
    assert lines[2:5] == ["The workflow matches its reference.", "", run_report.SMOKE_NOTE]
    assert f"{run_report.LEGEND} {run_report.SMOKE_LEGEND}" in lines
    assert "| ❌ 1 of 1 failed | `flux_dev_t2i` smoke check failed |" in out
    assert "Checked" not in out  # the run itself has no smoke row
    # A run with a smoke row and no history to read explains them just the same.
    files = _smoke_files([_smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])])
    del files[f"regression/index/master/{LANE}.json"]
    out = _smoke_report(files)
    assert run_report.SMOKE_NOTE in out.split("\n") and run_report.SMOKE_LEGEND in out
    # So does a lane whose only smoke words are in an open failure.
    chain = {"gone": {"commit": PREV, "kind": "execution_error", "level": "smoke", "runs": 3}}
    out = _smoke_report(_smoke_files([_cell("flux_dev_t2i")], before=[], first_bad=chain))
    assert run_report.SMOKE_NOTE in out.split("\n")
    # A chain that a smoke-level execution error started keeps the words of its kind.
    assert "- `gone`: execution error, failing since " in out
    assert run_report.SMOKE_NOTE not in _smoke_report(
        _smoke_files([_cell("flux_dev_t2i")], before=[])).split("\n")
    # A reference row never borrows a cell from the checks, and a smoke row none from
    # a comparison.
    view = run_report.View(wf="wf", name="`wf`", safe=True, cell={"vs_golden": IDENT}, run={},
                           state="smoke_pass", origin=None, prior={}, sha=None, level="smoke")
    assert run_report.reference_cell(view) == ""
    view.level = None
    assert run_report.reference_cell(view) == "bit-identical"


def test_the_files_of_a_smoke_failure_are_cut_with_the_other_file_lists():
    names = [f"{MESH}_{i:05d}_.png" for i in range(1, 6)]
    blank = [{"file": names[0], "check": "blank", "text": "blank"}]
    data = run_report.gather_lane(
        run_report.DictReader(_smoke_files([_smoke_cell(MESH, [_png(n) for n in names], blank)]),
                              RESULTS), NIGHT_LANES, SHA, "master", LANE)
    view = data.views[0]
    full = run_report.smoke_block(data, view)
    assert len([line for line in full if line.startswith(f"| [{MESH_MD}\\_0000")]) == 5
    cut = run_report.smoke_block(data, view, run_report.Limits(files_per_workflow=3))
    assert len([line for line in cut if line.startswith("| [")]) == 3
    # Only a shortened report cuts this list, and it cuts the list under "Output
    # files" to the same three: the record is where the other two are.
    assert "2 more files are listed in run.json." in cut
    assert not [line for line in cut if "Output files" in line]
    gone = run_report.smoke_block(data, view, run_report.Limits(files_per_workflow=0))
    assert not [line for line in gone if line.startswith("|")]
    assert gone[2] == "- <code>hunyuan3d\\_2\\_1\\_t23d\\_00001\\_.png</code>: blank"
    # A run that recorded a failure and no problem says so, and at most 20 are listed.
    view.cell["smoke"]["problems"] = []
    assert "- The run recorded no problem for this failure." in run_report.smoke_block(data, view)
    view.cell["smoke"]["problems"] = [{"file": None, "text": f"p{i}"} for i in range(30)]
    assert len([line for line in run_report.smoke_block(data, view)
                if line.startswith("- p")]) == 20


def test_a_standing_smoke_failure_is_inherited_and_names_its_chain():
    chain = {MUSIC: {"commit": LAST, "run_ts": 1794616236, "kind": "fail", "prev_good": FIRST,
                     "pr": 17700, "author": "kijai", "golden": None, "runs": 2,
                     "level": "smoke", "range": {"n": 6, "url": "u"}}}
    before = [_entry(LAST, 1794616236, {MUSIC: {"v": "fail", "l": "smoke"}}, n=6, pr=17700)]
    out = _nightly([_silent(prior_verdict="fail", prior_commit=LAST)], before=before,
                   first_bad=chain)
    last = f"[`e9afbcc`]({COMFY}/commit/{LAST})"
    # The chain's own range is named, and no pull request: 6 commits cannot be one.
    since = (f"ailing since {last} (a run that covered 6 commits, [compare]({COMFY}/compare/"
             f"{FIRST}...{LAST})), first seen 2026-11-14, 2 runs, last passed at [`aaaaaaa`]("
             f"{COMFY}/commit/{FIRST})")
    assert _block(out, MUSIC)[0] == f"#### ❌ `{MUSIC}`: smoke check failed (inherited)"
    assert _block(out, MUSIC)[-2:] == [
        f"It already failed at {last}: the failure is inherited from an earlier run.",
        f"F{since}."]
    assert "1 of 1 workflow failed: 1 inherited." in out.split("\n")
    assert f"- `{MUSIC}`: smoke check failed, f{since}" in out.split("\n")
    assert "17700" not in "\n".join(_block(out, MUSIC)) and "nightly-weekly.yml" not in out
    # A chain whose own range is one commit names the pull request, as a chain always did.
    one = {MUSIC: {**chain[MUSIC], "range": {"n": 1, "url": "u"}}}
    assert (f"failing since {last} ([#17700]({COMFY}/pull/17700), kijai, Add a node "
            f"(\\#{ZW}17700))") in run_report.failing_since(one[MUSIC], before)
    # A range of unknown size blames no pull request either, and states no number:
    # GitHub did not answer, or a later run passed inside the range and the
    # count of the chain's first run is no longer the chain's.
    unknown = run_report.failing_since({**chain[MUSIC], "range": {"n": None}}, before)
    assert (f"failing since {last} (one of the commits since its last pass, [compare]({COMFY}/"
            f"compare/{FIRST}...{LAST})), first seen ") in unknown
    assert "covered" not in unknown and "17700" not in unknown
    # Without a pass before it there is nothing to compare with.
    never_passed = {**chain[MUSIC], "range": {"n": None}, "prev_good": None, "level": None}
    assert "(a run that covered an unknown number of commits), first seen" in (
        run_report.failing_since(never_passed, before))
    # A chain without a range of its own is read from the run's range, as before.
    plain = {k: v for k, v in chain[MUSIC].items() if k != "range"}
    assert "(a run that covered 6 commits), first seen" in run_report.failing_since(plain, before)
    # A workflow that never passed: only a smoke-level chain is worded as a first run.
    # A reference-level chain keeps the words it had.
    never = {"commit": LAST, "kind": "fail", "pr": 17700, "author": "kijai", "runs": 2}
    assert run_report.failing_since(never) == (
        f"failing since {last} ([#17700]({COMFY}/pull/17700), kijai), 2 runs")
    assert run_report.failing_since({**never, "level": "smoke"}) == (
        f"failing since {last} (the workflow's first run on this lane), 2 runs")


def test_first_run_has_a_wording_of_its_own():
    """A workflow that fails its first run is a problem of the workflow. It
    must not read like a change in the commits the run covers."""
    missing = [{"file": None, "check": "missing",
                "text": "SaveVideo wrote no .mp4 file with the prefix wan_clip_video"}]
    cell = _smoke_cell(CLIP, problems=missing, prior_verdict=None, prior_commit=None)
    chain = {CLIP: {"commit": HEAD, "run_ts": 1794702636, "kind": "fail", "prev_good": None,
                    "pr": 17754, "author": "kijai", "golden": None, "runs": 1,
                    "level": "smoke"}}
    out = _nightly([cell], before=[], first_bad=chain)
    assert _block(out, CLIP) == [
        f"#### ❌ `{CLIP}`: smoke check failed (first run)",
        "The workflow finished, and its output did not pass the smoke checks.",
        "- SaveVideo wrote no .mp4 file with the prefix wan\\_clip\\_video",
        run_report.FIRST_RUN]
    assert "1 of 1 workflow failed: 1 on its first run." in out.split("\n")
    assert f"| `{CLIP}` | ❌ smoke check failed |  | no file |" in out
    # The lane's open failures name no range and no pull request for it.
    assert (f"- `{CLIP}`: smoke check failed, failing since [`05e878f`]({COMFY}/commit/{HEAD}) "
            "(the workflow's first run on this lane), first seen 2026-11-15, 1 run") in out
    assert "new at this commit" not in out.replace("new at this commit (new in this run", "")
    assert "new in this run" not in out.replace("(new in this run, when", "")
    # On a nightly lane an execution error with no earlier result is a first run too.
    broken = _smoke_cell(CLIP, verdict="execution_error", prior_verdict=None, prior_commit=None)
    out = _nightly([broken], before=[])
    assert ("The workflow did not finish on this commit. Worker status: <code>timeout</code>. "
            + run_report.FIRST_RUN) in out.split("\n")
    assert "1 of 1 workflow failed: 1 on its first run." in out.split("\n")
    # On the per-commit lane that case keeps the words it had.
    out = _smoke_report(_smoke_files([broken], before=[]))
    assert "it is not known whether the error is new at this commit" in out

    origin = run_report.origin_of
    assert origin("smoke_fail", {}, {}, SHA) == "first"
    assert origin("smoke_fail", {"verdict": "pass"}, {}, SHA) == "new"
    assert origin("smoke_fail", {"verdict": "execution_error"}, {}, SHA) == "new"
    assert origin("smoke_fail", {"verdict": "fail"}, {}, SHA) == "inherited"
    # A chain that started at another commit says the failure is older than this run.
    assert origin("smoke_fail", {}, {"commit": PREV}, SHA) == "inherited"
    assert origin("smoke_fail", {}, {"commit": SHA}, SHA) == "first"
    assert origin("exec", {}, {}, SHA, long_cadence=True) == "first"
    assert origin("exec", {}, {"commit": PREV}, SHA, long_cadence=True) == "inherited"
    assert origin("exec", {"verdict": "pass"}, {}, SHA, long_cadence=True) == "new"
    # The per-commit lane's execution errors are classified as they always were.
    assert origin("exec", {}, {}, SHA) is None
    assert origin("exec", {}, {"commit": SHA}, SHA) == "new"
    assert origin("smoke_pass", {"verdict": "fail"}, {}, SHA) is None


def test_the_checked_column_stands_where_nothing_is_compared():
    image = _cell("flux_dev_t2i")
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png"), _glb(f"{MESH}_mesh_00001_.glb")])
    # A lane of smoke rows alone: Checked in place of the three reference columns.
    out = _smoke_report(_smoke_files([mesh, _silent()]))
    table = _section(out, "All workflows")
    assert table[2] == SMOKE_HEADER and "vs reference" not in out
    assert all(row.count(" | ") == 8 for row in table[4:6])
    # A mixed lane keeps the three and adds Checked after them. A smoke row fills
    # only Checked, a reference row leaves it empty.
    out = _smoke_report(_smoke_files([image, mesh, _silent()]))
    table = _section(out, "All workflows")
    assert table[2] == MIXED_HEADER
    cells = {row.split(" | ")[0]: row.split(" | ") for row in table[4:7]}
    assert cells["| `flux_dev_t2i`"][3:7] == [
        "bit-identical", f"identical to [`{PREV[:7]}`]({COMFY}/commit/{PREV})", "`v0.42.0`", ""]
    assert cells[f"| `{MESH}`"][3:7] == ["", "", "", "1 image 1024x1024 · mesh 35,781 vertices"]
    assert cells[f"| `{MUSIC}`"][3:7] == ["", "", "", "30.0 s · 48000 Hz · stereo"]
    # A lane without a smoke row has no such column: its table is the one it had.
    out = _smoke_report(_smoke_files([image]))
    assert _section(out, "All workflows")[2] == SMOKE_HEADER.replace(
        "| Checked |", "| vs reference | vs previous | Reference |")

    def checked(*files, **kw):
        return run_report.checked_cell(run_report.View(
            wf="wf", name="`wf`", safe=True, run={}, state=kw.pop("state", "smoke_pass"),
            origin=None, prior={}, sha=None, level=kw.pop("level", "smoke"),
            cell={"smoke": {"files": list(files)}}))

    video = {"name": "v.mp4", "type": "video", "bytes": 310000, "width": 640, "height": 352,
             "frames": 25, "seconds": 1.042, "bytes_per_frame_mp": 14280}
    text = {"name": "t.txt", "type": "text", "bytes": 1300, "chars": 1204, "words": 231}
    assert checked(video) == "video 640x352 · 25 frames"
    # The count is of the characters that are not blank, and says so: the heading
    # of a shown text counts every character and has another number.
    assert checked(text) == "text 1,204 non-blank characters"
    assert checked(_flac("a.flac", channels=1)) == "30.0 s · 48000 Hz · mono"
    assert checked(_flac("a.flac", channels=6)) == "30.0 s · 48000 Hz · 6 channels"
    # Next to another type the audio file is named, and equal files are said once.
    assert checked(_png("a.png"), _flac("a.flac")) == (
        "1 image 1024x1024 · audio 30.0 s · 48000 Hz · stereo")
    assert checked(_png("a.png"), _png("b.png", width=512)) == "2 images"
    assert checked(_glb("a.glb"), _glb("b.glb"), _glb("c.glb", vertices=1)) == "3 meshes"
    assert checked(_flac("a.flac"), _flac("b.flac")) == "2 audio files 30.0 s · 48000 Hz · stereo"
    # A file of a type that has no check, and one the worker listed and did not send.
    assert checked(_png("a.png"), {"name": "a.gif", "type": None, "bytes": 5}) == (
        "1 image 1024x1024 · 1 other file, not checked")
    assert checked(_png("a.png"), {**video, "sent": False, "published": False}) == (
        "1 image 1024x1024 · 1 file not sent, checked by size only")
    assert checked({**video, "sent": False}) == "1 file not sent, checked by size only"
    assert checked(_flac("a.flac"), {**video, "sent": False}) == (
        "audio 30.0 s · 48000 Hz · stereo · 1 file not sent, checked by size only")
    # A run that wrote nothing, and one that did not get far enough to be checked.
    assert checked(state="smoke_fail") == "no file" and checked(state="exec") == ""
    assert checked(_png("a.png"), level=None) == ""
    # Numbers of the wrong type are left out, never printed and never raised on.
    assert checked({"name": "a.png", "type": "image", "width": "<b>", "height": [1]},
                   {"name": "a.glb", "type": "mesh", "vertices": True},
                   {"name": 7, "type": "image"}, "not a file") == "1 image · mesh"


def test_smoke_facts_by_type():
    facts = run_report.smoke_facts
    assert facts(_png("a.png")) == ("1024x1024 · brightness spread 69.17 · 64.1 % of the pixels "
                                    "have one brightness")
    assert facts(_glb("a.glb")) == "35,781 vertices · 71,836 faces · box 1.37 x 1.56 x 1.22"
    assert facts(_glb("a.glb", vertices=1, faces=1, extent=None)) == "1 vertex · 1 face"
    assert facts(_flac()) == "30.0 s · 48000 Hz · 2 channels · 117,330 bytes/s"
    assert facts({"type": "text", "chars": 1204, "words": 1}) == (
        "1,204 non-blank characters · 1 word")
    assert facts({"type": "video", "width": 640, "height": 352, "frames": 25, "seconds": 1.042,
                  "bytes_per_frame_mp": 14280}) == (
        "640x352 · 25 frames · 1.0 s · 14,280 bytes a frame for each megapixel after the "
        "first frame")
    # A file that could not be read has nothing measured. One that was not sent says so.
    assert facts({"type": "mesh"}) == ""
    assert facts(_png("a.png", sent=False)) == "not sent, checked by size only"
    assert facts({"type": None}) == facts({"type": "<b>"}) == (
        "not checked: no check for this file type")
    assert facts({"type": None, "sent": False}) == (
        "not sent, and not checked: no check for this file type")
    odd = {"type": "audio", "seconds": HUGE, "sample_rate": "x", "channels": {"a": 1},
           "bytes_per_second": [1]}
    assert facts(odd) == ""


def test_a_smoke_text_workflow_shows_its_text_under_generated_text():
    wf, name = "llm_smoke", "llm_smoke_00001.txt"
    entry = {"name": name, "type": "text", "bytes": len(T_REF), "sha256": _sha256(T_REF),
             "chars": 173, "words": 36}
    path = f"{P}/runs/master/{SHA}/{wf}/outputs/{name}"
    cell = _smoke_cell(wf, [entry], output_kind="text")
    out = _smoke_report(_smoke_files([cell, _cell("flux_dev_t2i")], extra={path: T_REF}))
    lines = out.split("\n")
    section = _section(out, "Generated text")
    assert section[:5] == ["### Generated text", "", f"`{wf}`, smoke level, not compared:", "",
                           "<details><summary>Generated text (208 characters)</summary>"]
    assert _pre_blocks(out) == [T_REF] and "identical to its reference" not in out
    assert f"[The whole file]({RAW}/{path})" in section
    # One page, one word for each count: the 173 are the characters that are not blank.
    assert any(row.startswith(f"| `{wf}` | ✅ smoke pass | [text]({RAW}/{path}) |  |  |  | text "
                              "173 non-blank characters |") for row in lines)
    assert any("173 non-blank characters · 36 words" in row for row in lines)
    # One that failed its checks shows its text in its block, where the failure is read.
    again = [{"file": name, "check": "repeat", "text": "one word is 30 of the 36 words"}]
    out = _smoke_report(_smoke_files([_smoke_cell(wf, [entry], again, output_kind="text")],
                                     extra={path: T_EVIL}))
    assert "### Generated text" not in out
    block = "\n".join(_section(out, "What needs attention"))
    assert "<details><summary>Generated text (" in block and block.count("<pre>") == 1
    assert "- <code>llm\\_smoke\\_00001.txt</code>: one word is 30 of the 36 words" in block
    # Model output stays inside the <pre> block, whatever the level.
    assert "<script>" not in out and "@octocat" not in _outside_pre(out)
    assert not [line for line in out.split("\n") if line.startswith("::")]
    # A text over 1 MiB fails its check and is kept off the results branch: no link
    # says it is there.
    big = {**entry, "bytes": 2 * 1024 * 1024, "published": False}
    large = [{"file": name, "check": "size", "text": "2097152 bytes: a text output may have at "
              "most 1048576"}]
    out = _smoke_report(_smoke_files([_smoke_cell(wf, [big], large, output_kind="text")]),
                        artifact_url=ARTIFACT)
    assert f"| `{wf}` | ❌ smoke check failed | text, in the [attached files]({ARTIFACT}) |" in out
    assert "The generated text is 2.00 MB, too large to show here." in out.split("\n")
    assert f"{name})" not in out and "The whole file" not in out


def test_a_file_that_is_not_published_is_not_linked_as_if_it_were():
    """A smoke run keeps a file over 1 MiB off the results branch until the
    run before wrote the same file. The report must not link what is not there."""
    big = _png(f"{MESH}_00001_.png", bytes=1278115, published=False)
    mesh = _smoke_cell(MESH, [big, _png(f"{MESH}_00002_.png"),
                              _glb(f"{MESH}_mesh_00001_.glb", published=False)])
    music = _smoke_cell(MUSIC, [_flac(published=False)], output_kind="audio")
    clip = _smoke_cell(CLIP, [
        _png(f"{CLIP}_00001_.png"),
        {"name": f"{CLIP}_video_00001_.mp4", "type": "video", "bytes": 9800000,
         "sha256": "ab" * 32, "sent": False, "published": False}],
        notes=[f"{CLIP}_video_00001_.mp4 was not sent (over the worker's size limit): checked "
               "by size only"])
    files = _smoke_files([mesh, music, clip])
    out = _smoke_report(files, artifact_url=ARTIFACT)
    run_dir = f"{RAW}/{P}/runs/master/{SHA}"
    # The thumbnail is always published. The full picture is in the attached files.
    assert (f'| `{MESH}` | ✅ smoke pass | <a href="{ARTIFACT}"><img src="{run_dir}/{MESH}/outputs/'
            f'thumbs/{MESH}_00001_.webp" width="96" alt="{MESH} output"></a> |') in out
    assert f"| `{MUSIC}` | ✅ smoke pass | audio, in the [attached files]({ARTIFACT}) |" in out
    held = ("not published: over 1 MiB, and the run before did not write the same file (in the "
            f"[attached files]({ARTIFACT}))")
    assert (f"| `{MESH}` | {MESH_MD}\\_00001\\_.png | 1.22 MB | `{'9f' * 32}` | {held}; 1024x1024 "
            "· brightness spread 69.17 · 64.1 % of the pixels have one brightness |") in out
    assert (f"| `{MESH}` | [{MESH_MD}\\_00002\\_.png]({run_dir}/{MESH}/outputs/{MESH}_00002_.png) "
            "| 152.5 KB |") in out
    for name in (f"{MESH}_00001_.png", f"{MESH}_mesh_00001_.glb", FLAC,
                 f"{CLIP}_video_00001_.mp4"):
        assert f"{name})" not in out and f'{name}"' not in out  # no link to it anywhere
    # A file the worker listed and did not send: said in the row and in the file list.
    assert "| 1 image 1024x1024 · 1 file not sent, checked by size only |" in out
    assert (f"| `{CLIP}` | wan\\_clip\\_video\\_00001\\_.mp4 | 9.35 MB | `{'ab' * 32}` | not "
            "published: over the worker's size limit, checked by size only |") in out
    # A report made later knows no artifact: it points to the run that tested the commit.
    out = _smoke_report(files)
    assert f'<a href="{RUN_URL}"><img src="{run_dir}/{MESH}/outputs/thumbs/' in out
    assert f"| audio, attached to [the run that tested it]({RUN_URL}) |" in out
    # In a failure's block the file is named, and the rule is said once under the table.
    quiet = _silent()
    quiet["smoke"]["files"][0]["published"] = False
    block = _block(_smoke_report(_smoke_files([quiet]), artifact_url=ARTIFACT), MUSIC)
    assert block[3:7] == [
        *FILES_HEADER,
        f"| {FLAC_MD}, not published (in the [attached files]({ARTIFACT})) | audio | 10.8 KB "
        "| 30.0 s · 48000 Hz · 2 channels · 138 bytes/s |",
        "A file over 1 MiB is published only when the run before wrote the same file. Until "
        "then it is in the files attached to the run, which GitHub keeps 90 days."]
    # The link of an audio workflow goes to its main file: the first FLAC by name
    # that starts with the workflow id, as the runner's sha rule has it.
    two = _smoke_cell(MUSIC, [_flac("aaa_intro.flac"), _flac(f"{MUSIC}_00002.flac"), _flac()],
                      output_kind="audio")
    out = _smoke_report(_smoke_files([two]))
    assert f"| [audio]({run_dir}/{MUSIC}/outputs/{FLAC}) | 3 audio files 30.0 s · 48000 Hz" in out
    other = _smoke_cell(MUSIC, [_flac("zzz.flac"), _flac("aaa.flac")], output_kind="audio")
    out = _smoke_report(_smoke_files([other]))
    assert f"| [audio]({run_dir}/{MUSIC}/outputs/aaa.flac) | 2 audio files 30.0 s" in out
    # Either mark keeps a file from being linked: not published, or not sent at all.
    marks = run_report.View(wf="wf", name="`wf`", safe=True, run={}, state="smoke_pass",
                            origin=None, prior={}, sha=None, level="smoke", cell={"smoke": {
                                "files": [{"name": "a.mp4", "sent": False}, {"name": "c.png"},
                                          {"name": "b.png", "published": False}]}})
    assert run_report.held_back(marks) == {"a.mp4", "b.png"}
    # Past the 50 files a run lists, only a small file is known to be published.
    listed = {"outputs": [{"filename": "a.png", "bytes": 5, "sha256": "1" * 64},
                          {"filename": "b.glb", "bytes": 5 * 1024 * 1024, "sha256": "2" * 64}]}
    lane = run_report.gather_lane(run_report.DictReader({}, RESULTS), NIGHT_LANES, SHA, "master",
                                  LANE)
    note = run_report.smoke_file_note
    assert note(lane, listed["outputs"][0], {}) == (True, "")
    assert note(lane, listed["outputs"][1], {}) == (
        False, "not linked: past the 50 files the run says it published")


def test_hostile_names_and_problem_texts_of_a_smoke_run_stay_inert():
    """File names, types, numbers and problem sentences of `smoke` come from
    the worker's files. None of them may add markup, a link, a mention or a
    workflow command."""
    name = "evil | `x` <img src=x onerror=alert(1)> [a](https://evil.example) @octocat #1.png"
    files = [
        _png(name), _png("../../../../etc/passwd.png"), _png("ok dir/fine name.png"),
        {"name": "::error::boom.glb", "type": "<b>mesh</b>", "bytes": "many", "sha256": EVIL,
         "vertices": "<script>alert(1)</script>", "faces": EVIL, "extent": [EVIL, 1, 2]},
        {"name": "ok.flac", "type": "audio", "bytes": 5, "sha256": "1" * 64, "seconds": HUGE,
         "sample_rate": EVIL, "channels": {"x": 1}, "bytes_per_second": [1]},
        {"name": ["not", "a", "name"], "type": "image"}, "not a file", None]
    problems = [
        {"file": name, "check": "decode", "text": EVIL},
        {"file": "::error::boom.glb", "check": "decode",
         "text": "::error::could not be read: JSONDecodeError: <script>alert(1)</script>"},
        {"file": None, "check": EVIL, "text": "::warning::@octocat | a | b |"},
        "not an object", {"file": ["a"], "text": None}, {"file": EVIL, "text": "x" * 5000}]
    cell = _smoke_cell("hostile_wf", [f for f in files if isinstance(f, dict)
                                      and isinstance(f.get("name"), str)], problems,
                       notes=[EVIL, "::notice::x", 7, None], prior_commit=EVIL)
    cell["smoke"]["files"] = files
    out = _smoke_report(_smoke_files([cell]), artifact_url=ARTIFACT)
    md = _outside_pre(out)
    assert "<script" not in out and "<img src=x" not in out and "<b>" not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in md
    assert set(re.findall(r"</?([a-zA-Z0-9]+)", out)) <= {"a", "img", "code", "details", "summary"}
    assert "&#124; pipe" in md and " | pipe" not in md and "evil &#124; " in md
    assert f"@{ZW}octocat" in md and "@octocat" not in md and "#4242" not in md
    assert "**bold**" not in md and "`tick`" not in md
    # No line is a workflow command: each such text stands behind a list mark or in a cell.
    assert not [line for line in out.split("\n") if line.lstrip().startswith("::")]
    assert "- <code>::error::boom.glb</code>: ::error::could not be read: JSONDecodeError:" in md
    assert f"- ::warning::@{ZW}octocat &#124; a &#124; b &#124;" in md
    assert "- Note: ::notice::x" in md and "- no detail recorded" in md
    # Links go to known hosts, and no name climbs out of the run's folder.
    targets = (re.findall(r'(?:src|href)="([^"]*)"', out)
               + re.findall(r"(?<!\\)\]\(([^)]*)\)", md))
    assert targets
    for url in targets:
        assert re.match(r"https://(github\.com|raw\.githubusercontent\.com|ci\.comfy\.org)(/|$)",
                        url), url
    assert "../" not in "".join(targets) and "evil.example)" not in "".join(targets)
    run_dir = f"{RAW}/{P}/runs/master/{SHA}/hostile_wf/outputs"
    assert f"[ok dir/fine name.png]({run_dir}/ok%20dir/fine%20name.png)" in md
    assert f"[::error::boom.glb]({run_dir}/%3A%3Aerror%3A%3Aboom.glb) | other |  | not " in md
    # A type that is no known type is "other", a size that is no number is left out,
    # and a long sentence is cut. The tables keep their shape.
    block = _block(out, "hostile_wf")
    rows = block[block.index(FILES_HEADER[1]) + 1:][:5]
    assert len(rows) == 5 and all(row.count(" | ") == 3 for row in rows)
    assert "x" * 400 not in out
    # A last result that names no commit: the heading keeps to the run, and no
    # sentence is built around what is not a sha.
    assert block[0] == "#### ❌ `hostile_wf`: smoke check failed (new at this commit)"
    assert "It last passed" not in out and "Its last result" not in out
    table = _section(out, "All workflows")
    assert table[4].count(" | ") == 8
    assert "| 3 images 1024x1024 · audio · 1 other file, not checked |" in table[4]


def test_the_title_of_a_nightly_run_names_the_commits_it_covers():
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")], prior_commit=LAST)
    out = _nightly([mesh], published_new_bytes=412345)
    lines = out.split("\n")
    assert lines[0] == ("## ✅ GPU regression · nightly · py3.13 / torch 2.14.1 / cu130 / RTX 4090 "
                        "· `e9afbcc`..`05e878f` (7 commits)")
    # The row says what the number is: the output files of smoke-level workflows,
    # as the lane's own run of the commit published them. It does not say "this
    # run": a skipped night and a re-test render the same summary.
    row = "| New smoke-level output the lane's run published |"
    assert f"{row} 402.7 KB |" in lines and "this run |" not in out
    assert f"{row} 0 B |" in _nightly([mesh], published_new_bytes=0).split("\n")
    # A nightly run of one commit, and a run that has no range, name the tested commit.
    one = _smoke_files([mesh], lane=NIGHTLY, commit=HEAD)
    assert _smoke_report(one, lane=NIGHTLY, commit=HEAD).split("\n")[0].endswith("· `05e878f3`")
    # A per-commit run keeps its title and its facts, however many commits it covers.
    rng = {"prev": PREV, "commits_between": 3}
    out = _smoke_report(_smoke_files([_cell("flux_dev_t2i")], tested_range=rng))
    assert out.split("\n")[0].endswith("· `cccccccc`") and "the lane's run published" not in out
    for odd in (True, "412345", -1, None, [1]):
        assert "the lane's run published" not in _nightly([mesh], published_new_bytes=odd)


def test_cell_state_and_rollup_know_the_smoke_level():
    state = run_report.cell_state
    assert state("pass", level="smoke") == "smoke_pass"
    assert state("fail", level="smoke") == "smoke_fail"
    assert state("execution_error", level="smoke") == "exec"
    assert state("infra_error", level="smoke") == "infra"
    # No reference can accept a smoke output, and a missing reference is no smoke result.
    ref = {"sha": "2" * 64, "tag": "v0.43.0"}
    assert state("fail", "new_drift", "2" * 64, "v0.42.0", ref, level="smoke") == "smoke_fail"
    assert state("fail", "new_drift", "2" * 64, "v0.42.0", ref) == "accepted"
    assert state("no_baseline", level="smoke") == state(["pass"], level="smoke") == "unknown"
    for level in (None, "reference", ["smoke"], 1):
        assert state("pass", level=level) == "pass"

    rollup = run_report.rollup
    assert rollup({"a": "smoke_pass", "b": "smoke_pass"}) == "✅ smoke pass"
    assert rollup({"a": "pass", "b": "accepted", "c": "smoke_pass"}) == (
        "✅ pass (2 reference, 1 smoke)")
    assert rollup({"a": "pass"}) == "✅ pass"
    # Red and not-tested roll-ups keep their shape.
    assert rollup({"a": "smoke_fail", "b": "smoke_pass", "c": "pass"}) == "❌ 1 of 3 failed"
    assert rollup({"a": "infra", "b": "smoke_pass"}) == f"{run_report.WARN} 1 of 2 infra error"
    assert run_report.not_passing({"a": "smoke_pass", "b": "pass"}) == ""
    assert run_report.not_passing({"a": "smoke_fail", "b": "accepted"}) == (
        "`a` smoke check failed; `b` output changed (accepted)")
    # Every result that needs a block has a place in the order of the blocks.
    needs_block = set(run_report.LABELS) - set(run_report.PASSED) - {"not_run"}
    assert needs_block == set(run_report.ATTENTION_ORDER)
    assert set(run_report.RED) | set(run_report.GREEN) <= set(run_report.LABELS)


def test_a_summary_rebuilt_from_the_runner_keeps_the_level_and_the_checks(tmp_path):
    """A run that could not publish reports from its own files. The cell is
    built from comparison.json, which names the level and holds the checks."""
    workdir = tmp_path / "regression-work"
    cell = _silent()
    comparison = {**{k: cell[k] for k in ("verdict", "detail", "vs_golden", "vs_previous",
                                          "thresholds_used", "golden_tag", "previous_commit",
                                          "previous_verdict", "figures")},
                  "output_kind": "audio", "level": "smoke", "smoke": cell["smoke"]}
    record = _smoke_run(cell)
    # A second sound file, first by name: it is not the one that identifies the run.
    record["outputs"].insert(0, {"filename": "aaa_intro.flac", "bytes": 5, "sha256": "f" * 64})
    for wf, files in ((MUSIC, {"run.json": record, "comparison.json": comparison}),
                      ("flux_dev_t2i", {"run.json": _run("flux_dev_t2i"), "comparison.json": {
                          "verdict": "pass", "vs_golden": IDENT, "vs_previous": IDENT}})):
        (workdir / wf).mkdir(parents=True)
        for name, data in files.items():
            (workdir / wf / name).write_text(json.dumps(data), encoding="utf-8")
    rebuilt = run_report.summary_from_workdir(str(workdir), SHA, "master", LANE)["workflows"]
    assert list(rebuilt[MUSIC])[-4:] == ["custom_nodes", "output_kind", "level", "smoke"]
    assert rebuilt[MUSIC]["smoke"] == cell["smoke"] and rebuilt[MUSIC]["level"] == "smoke"
    # The audio file identifies the run, by the rule the runner uses.
    assert rebuilt[MUSIC]["output_sha256"] == "10" * 32
    assert "level" not in rebuilt["flux_dev_t2i"] and "smoke" not in rebuilt["flux_dev_t2i"]
    assert list(rebuilt["flux_dev_t2i"])[-1] == "custom_nodes"
    out = run_report.lane_report(run_report.Reader(), LANES, SHA, "master", LANE,
                                 workdir=str(workdir))
    assert f"#### ❌ `{MUSIC}`: smoke check failed (first run)" in out
    assert f"- <code>ace\\_step\\_15\\_t2a\\_00001.flac</code>: {SILENT}" in out
    assert MIXED_HEADER in out and "raw.githubusercontent.com" not in out

    run = {"outputs": [{"filename": "other.flac", "sha256": "o"},
                       {"filename": f"{MUSIC}_00002.flac", "sha256": "b"},
                       {"filename": FLAC, "sha256": "a"}, {"filename": "x.png", "sha256": "p"}]}
    assert run_report.output_sha(run, "audio", MUSIC) == "a"
    assert run_report.output_sha(run, "image", MUSIC) == "p"
    assert run_report.output_sha({}, "audio", MUSIC) is None


def _smoke_overview_files():
    mixed = [_cell("flux_dev_t2i"), _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")]), _silent()]
    files = _smoke_files(mixed, first_bad={
        MUSIC: {"commit": SHA, "run_ts": 1794702636, "kind": "fail", "prev_good": PREV,
                "pr": 17754, "author": "kijai", "golden": None, "runs": 1, "level": "smoke"},
        "gone_wf": {"commit": PREV, "run_ts": 1794700000, "kind": "execution_error",
                    "prev_good": FIRST, "pr": 17753, "author": "kijai", "runs": 2}})
    return files


def test_overview_words_smoke_results_and_lists_chains_of_workflows_the_lane_runs():
    files = _smoke_overview_files()
    reader = run_report.DictReader(files, RESULTS)
    out = run_report.overview_report(reader, LANES, SHA, "master")
    lines = out.split("\n")
    main = LANES["lanes"][LANE]["label"]
    # The sentence stands under the title, and the legend names the smoke labels.
    assert lines[:3] == ["## GPU regression · all lanes · `cccccccc`", "", run_report.SMOKE_NOTE]
    assert f"{run_report.LEGEND} {run_report.SMOKE_LEGEND}" in lines
    assert (f"| {main} | primary · blocking | ❌ 1 of 3 failed | [2026-11-15 00:30 UTC]({RUN_URL}) "
            f"| `{MUSIC}` smoke check failed |") in lines
    matrix = _section(out, "Recent commits")
    assert f"[❌ 1 of 3 failed]({RUN_URL}): `{MUSIC}` smoke check failed | ➖ not run |" in matrix[4]
    assert "| ✅ pass (1 reference, 2 smoke) | ➖ not run |" in matrix[5]
    since = f"failing since [`{SHA[:7]}`]({COMFY}/commit/{SHA})"
    failures = _section(out, "Open failures")
    assert failures[2:5] == [f"{main}:", "", f"- `{MUSIC}`: smoke check failed, {since} "
                             f"([#17754]({COMFY}/pull/17754), kijai, Add a node (\\#{ZW}17754)), "
                             "first seen 2026-11-15, 1 run, last passed at "
                             f"[`{PREV[:7]}`]({COMFY}/commit/{PREV})"]
    # Without the manifest every chain of the index is listed, as before.
    assert failures[5].startswith("- `gone_wf`: execution error, failing since ")
    # With it, only the chains of workflows the lane runs today. `gone_wf` was removed
    # while it was red, `off_wf` is switched off, `other_wf` runs on another lane only.
    manifest = {"workflows": {MUSIC: {"enabled": True}, MESH: {"enabled": True, "lanes": None},
                              "flux_dev_t2i": {"enabled": True, "lanes": [LANE]},
                              "off_wf": {"enabled": False}, "other_wf": {"enabled": True,
                                                                        "lanes": [OTHER]}}}
    out = run_report.overview_report(reader, LANES, SHA, "master", manifest=manifest)
    assert "gone_wf" not in out and f"- `{MUSIC}`: smoke check failed, " in out
    assert run_report.lane_workflows(manifest, LANE) == {MUSIC, MESH, "flux_dev_t2i"}
    assert run_report.lane_workflows(manifest, OTHER) == {MUSIC, MESH, "other_wf"}
    for nothing in (None, {}, {"workflows": []}, {"workflows": {}}):
        assert run_report.lane_workflows(nothing, LANE) is None
    # A chain of a smoke workflow the lane no longer runs brings no smoke words either.
    del manifest["workflows"][MUSIC]
    files[f"{P}/runs/master/{SHA}/summary.json"] = _summary([_cell("flux_dev_t2i")])
    files[f"regression/index/master/{LANE}.json"]["entries"] = [
        _entry(SHA, 1794702636, {"flux_dev_t2i": "pass"}, run=4242)]
    out = run_report.overview_report(reader, LANES, SHA, "master", manifest=manifest)
    assert "smoke" not in out and "None on the active lanes." in out
    assert "smoke check failed" in run_report.overview_report(reader, LANES, SHA, "master")
    # The sentence and the legend follow what the page shows. Only the commit's own
    # summary names a smoke level (an index from before the level was indexed) ...
    files = _smoke_overview_files()
    head = files[f"regression/index/master/{LANE}.json"]
    head["first_bad"] = {}
    for entry in head["entries"]:
        for index_cell in entry["w"].values():
            index_cell.pop("l", None)
    out = run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA, "master")
    assert out.split("\n")[2] == run_report.SMOKE_NOTE and run_report.SMOKE_LEGEND in out
    assert f"| `{MUSIC}` smoke check failed |" in out and "✅ pass | ➖ not run |" in out
    # ... or only an older commit of the table has a smoke result.
    files = _smoke_overview_files()
    head = files[f"regression/index/master/{LANE}.json"]
    head["first_bad"] = {}
    head["entries"][0] = _entry(SHA, 1794702636, {"flux_dev_t2i": "pass"}, run=4242)
    files[f"{P}/runs/master/{SHA}/summary.json"] = _summary([_cell("flux_dev_t2i")])
    out = run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA, "master")
    assert out.split("\n")[2] == run_report.SMOKE_NOTE and run_report.SMOKE_LEGEND in out
    assert "| ✅ pass (1 reference, 2 smoke) | ➖ not run |" in out


def test_the_overview_job_reads_the_manifest_of_its_checkout(tmp_path, monkeypatch):
    """`overview` from the command line lists a chain only for a workflow that
    manifest/workflows.json still runs on the lane."""
    manifest = tmp_path / "workflows.json"
    manifest.write_text(json.dumps({"workflows": {MUSIC: {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setattr(run_report.lanes_mod, "MANIFEST_FILE", manifest)
    out = tmp_path / "overview.md"
    argv = ["overview", "--commit", SHA, "--branch", "master", "--out", str(out)]
    reader = run_report.DictReader(_smoke_overview_files(), RESULTS)
    assert run_report.main(argv, reader=reader) == 0
    text = out.read_text(encoding="utf-8")
    assert f"- `{MUSIC}`: smoke check failed, failing since " in text and "gone_wf" not in text
    # No manifest to read: nothing is left out.
    monkeypatch.setattr(run_report.lanes_mod, "MANIFEST_FILE", tmp_path / "missing.json")
    assert run_report.main(argv, reader=reader) == 0
    assert "- `gone_wf`: execution error, failing since " in out.read_text(encoding="utf-8")


def _night_column(entries, cadence="nightly"):
    lane = {**NIGHT_LANES["lanes"][NIGHTLY], "cadence": cadence}
    return run_report.LaneColumn(lane_id=NIGHTLY, lane=lane, summary=None, head={},
                                 entries=run_report.entries_of(_head(entries)), references={})


def test_the_nightly_and_weekly_table_shows_each_lanes_last_full_run():
    """One row per lane, from its newest full run. A dispatch of one workflow
    is not the lane's result, and a lane whose cron stopped is marked."""
    runs = {NIGHTLY: {MUSIC, MESH, CLIP}}
    night = 1794702636
    full = {**_entry(LAST, night, {MUSIC: {"v": "fail", "l": "smoke"},
                                   MESH: {"v": "infra_error", "l": "smoke"},
                                   CLIP: {"v": "infra_error", "l": "smoke"}}, run=4100, n=7),
            "prev": FIRST}
    # Newer than the full run: one workflow, tested again by a dispatch.
    retest = {**_entry(HEAD, night + 7200, {MUSIC: {"v": "pass", "l": "smoke"}}, run=4200),
              "prev": LAST, "named": True}
    label = NIGHT_LANES["lanes"][NIGHTLY]["label"]
    block = run_report.long_lanes_block([_night_column([retest, full])], runs, now=night + 3600)
    assert block[:4] == ["### Nightly and weekly lanes", "",
                         "| Lane | Last full run | Covers | Result | Not passing | Not tested |",
                         "|---|---|---|---|---|---|"]
    run_4100 = RUN_URL.replace("4242", "4100")
    covers = f"[`aaaaaaa`..`e9afbcc`, 7 commits]({COMFY}/compare/{FIRST}...{LAST})"
    # "Not tested" counts the workflows that ended in an infra error.
    assert block[4:] == [f"| {label} | [2026-11-15 00:30 UTC]({run_4100}) | {covers} | ❌ 1 of 3 "
                         f"failed | `{MUSIC}` smoke check failed; `{MESH}` infra error; `{CLIP}` "
                         "infra error | 2 |"]
    # Stale: a nightly lane after 36 hours, a weekly lane after 8 days.
    stale = run_report.long_lanes_block([_night_column([full])], runs, now=night + 37 * 3600)
    last = f"[2026-11-15 00:30 UTC]({run_4100})"
    assert f"| {last} {run_report.WARN} no run for 37 hours |" in stale[4]
    fresh = run_report.long_lanes_block([_night_column([full])], runs, now=night + 35 * 3600)
    assert "no run for" not in fresh[4]
    week = _night_column([full], cadence="weekly")
    assert "no run for" not in run_report.long_lanes_block([week], runs, now=night + 7 * 86400)[4]
    assert f"{run_report.WARN} no run for 10 days |" in run_report.long_lanes_block(
        [week], runs, now=night + 10 * 86400)[4]
    # A workflow was just added, so no entry holds them all: the last full run is still
    # the lane's, and its row does not read as a run of some workflows. The same row when
    # the lane's workflows are not known.
    more = {NIGHTLY: {MUSIC, MESH, CLIP, "new_wf"}}
    for wanted in (more, None):
        assert run_report.long_lanes_block([_night_column([retest, full])], wanted,
                                           now=night + 3600) == block
    # A lane that has no full run yet shows its newest run, and says what that run is.
    block = run_report.long_lanes_block([_night_column([retest])], more, now=night + 7300)
    assert (f"| [`05e878f`]({COMFY}/commit/{HEAD}), 1 commit | ✅ smoke pass · 1 of 4 workflows |  "
            "| 0 |") in block[4]
    block = run_report.long_lanes_block([_night_column([retest])], None, now=night + 7300)
    assert "| ✅ smoke pass · named workflows only |  | 0 |" in block[4]
    # A lane that has not run yet, and no lane at all.
    assert run_report.long_lanes_block([_night_column([])], runs)[4] == (
        f"| {label} | no run yet |  | ➖ no result |  |  |")
    assert run_report.long_lanes_block([], runs) == []
    # Odd index data never raises: a time that is no time, a range that is no range.
    odd = {**full, "t": "soon", "prev": EVIL, "range": "x", "run": None}
    row = run_report.long_lanes_block([_night_column([odd])], runs, now=night)[4]
    assert f"| time not recorded | [`e9afbcc`]({COMFY}/commit/{LAST}) | ❌ 1 of 3 failed |" in row


# -- cadences: the nightly and weekly lanes in the reports -----------------------------------
WEEKLY = "py313-torch2.14.1-cu130-weekly"
# A nightly lane that runs, and a weekly lane that is declared before its endpoint exists.
LONG_LANES = {"primary": LANE, "lanes": {**NIGHT_LANES["lanes"], WEEKLY: {
    "label": "weekly · py3.13 / torch 2.14.1 / cu130 / RTX 4090", "cadence": "weekly",
    "blocking": False, "role": "lane", "layout": 2, "prefix": f"regression/lanes/{WEEKLY}",
    "endpoint_id": None}}}
CADENCES = {"workflows": {
    "flux_dev_t2i": {"enabled": True},
    "sdxl_t2i": {"enabled": True, "lanes": [LANE, OTHER]},
    MUSIC: {"enabled": True, "level": "smoke", "cadence": "nightly"},
    MESH: {"enabled": True, "level": "smoke", "cadence": "nightly", "lanes": [NIGHTLY]},
    CLIP: {"enabled": True, "level": "smoke", "cadence": "weekly"},
    "off_wf": {"enabled": False, "cadence": "nightly"}}}
NIGHT_TS = 1794702636  # 2026-11-15 00:30 UTC


def test_a_lane_column_knows_a_nightly_or_weekly_lane_that_can_run():
    def column(lane):
        return run_report.LaneColumn(lane_id="x", lane=lane, summary=None, head={}, entries=[],
                                     references={})

    for cadence in ("nightly", "weekly"):
        lane = column({"cadence": cadence, "endpoint_id": "ep"})
        assert lane.long_cadence and not lane.active and not lane.retired
    # Without an endpoint nothing runs on it yet. The other cadences are not long ones.
    for lane in ({"cadence": "nightly", "endpoint_id": None}, {"cadence": "weekly"},
                 {"cadence": "per-commit", "endpoint_id": "ep"},
                 {"cadence": "retired", "endpoint_id": "ep"}, {}):
        assert not column(lane).long_cadence, lane
    assert column({"cadence": "per-commit", "endpoint_id": "ep"}).active


def test_the_workflows_a_lane_runs_today_follow_its_cadence():
    lanes = LONG_LANES["lanes"]
    today = run_report.lane_workflows
    assert today(CADENCES, LANE, lanes[LANE]) == {"flux_dev_t2i", "sdxl_t2i"}
    assert today(CADENCES, OTHER, lanes[OTHER]) == {"flux_dev_t2i", "sdxl_t2i"}
    assert today(CADENCES, NIGHTLY, lanes[NIGHTLY]) == {MUSIC, MESH}
    assert today(CADENCES, WEEKLY, lanes[WEEKLY]) == {CLIP}
    # A retired lane, and a call without the lane's entry: the `lanes` list alone.
    assert today(CADENCES, LEGACY, lanes[LEGACY]) == {"flux_dev_t2i", MUSIC, CLIP}
    assert today(CADENCES, LANE) == {"flux_dev_t2i", "sdxl_t2i", MUSIC, CLIP}
    # A report is a courtesy: a cadence nobody knows raises nothing and leaves nothing out.
    odd = {"workflows": {"a": {"enabled": True, "cadence": "daily"},
                         "b": {"enabled": True, "cadence": ["nightly"]}, "c": "junk"}}
    assert today(odd, LANE, lanes[LANE]) == {"a", "b"}
    assert today(odd, NIGHTLY, lanes[NIGHTLY]) == {"a", "b"}
    for unknown in (None, "junk", {}, {"cadence": "hourly"}, {"cadence": ["nightly"]},
                    {"cadence": "retired"}):
        assert not run_report.other_cadence({"cadence": "nightly"}, unknown), unknown
        assert today(CADENCES, LANE, unknown) == today(CADENCES, LANE)
    assert run_report.other_cadence({"cadence": "nightly"}, lanes[LANE])
    assert run_report.other_cadence({}, lanes[NIGHTLY])


def test_a_workflow_of_another_cadence_is_not_listed_as_not_run():
    """The nightly and weekly workflows are not missing from a per-commit
    lane: they belong to other lanes. The same the other way round."""
    not_run = "Workflows not run on this lane"
    summary = _summary([_cell("flux_dev_t2i")])
    for extra, kw in (({}, {"manifest": CADENCES}),
                      ({f"{P}/manifest-snapshot/{SHA}.json": CADENCES}, {})):
        out = _report(_tree(summary, extra=extra), **kw)
        listed = [line for line in _section(out, not_run)[2:] if line]
        assert len(listed) == 1 and listed[0].endswith(": `sdxl_t2i`")
        for wf in (MUSIC, MESH, CLIP, "off_wf"):
            assert wf not in out, wf
    # Everything of the lane's cadence ran: no section, whatever the other lanes run.
    full = _summary([_cell("flux_dev_t2i"), _cell("sdxl_t2i")])
    assert not_run not in _report(_tree(full), manifest=CADENCES)
    # The nightly lane: a nightly workflow without a result is missing, a per-commit
    # or weekly one is not.
    files = _smoke_files([_smoke_cell(MUSIC, [_flac()], output_kind="audio")], lane=NIGHTLY,
                         commit=HEAD)
    out = _smoke_report(files, lane=NIGHTLY, commit=HEAD, manifest=CADENCES)
    listed = [line for line in _section(out, not_run)[2:] if line]
    assert len(listed) == 1 and listed[0].endswith(f": `{MESH}`")
    for wf in ("flux_dev_t2i", "sdxl_t2i", CLIP):
        assert wf not in out, wf
    # A retired lane ran what the manifest's list allowed, whatever the cadence.
    out = _report(_tree(_summary([_cell("flux_dev_t2i")], lane=LEGACY), lane=LEGACY),
                  lane=LEGACY, manifest=CADENCES)
    listed = [line for line in _section(out, not_run)[2:] if line]
    assert [line.rsplit(": ", 1)[1] for line in listed] == [
        "`sdxl_t2i`", f"`{MESH}`", f"`{MUSIC}`, `{CLIP}`"]


def _long_overview_files():
    """The per-commit lanes of `_overview_files`, and a nightly lane whose
    last run covered the 7 commits up to HEAD and found the audio silent."""
    files = _overview_files()
    run = {**_entry(HEAD, NIGHT_TS, {MUSIC: {"v": "fail", "l": "smoke"},
                                     MESH: {"v": "pass", "l": "smoke"}}, run=4100, n=7),
           "prev": LAST}
    files[f"regression/index/master/{NIGHTLY}.json"] = _head([run], {
        MUSIC: {"commit": HEAD, "run_ts": NIGHT_TS, "kind": "fail", "prev_good": LAST,
                "runs": 1, "level": "smoke", "range": {"n": 7, "url": "u"}},
        # Left from the time the lane ran it: it is a weekly workflow now.
        CLIP: {"commit": HEAD, "run_ts": NIGHT_TS, "kind": "fail", "prev_good": LAST,
               "runs": 1, "level": "smoke"}})
    return files


def test_the_overview_shows_where_the_nightly_and_weekly_lanes_stand():
    """Every per-commit run page says where the nightly and weekly runs stand
    and links to them. Those lanes get a table of their own, not a column: a
    column would read "not run" on nearly every row."""
    reader = run_report.DictReader(_long_overview_files(), RESULTS)
    out = run_report.overview_report(reader, LONG_LANES, SHA, "master", manifest=CADENCES,
                                     now=NIGHT_TS + 3600)
    lines = out.split("\n")
    assert [line for line in lines if line.startswith("### ")] == [
        "### This commit on every lane", "### Lanes that disagree at this commit",
        "### Nightly and weekly lanes", "### Open failures", "### Recent commits",
        "### Other lanes"]
    main, other = LANES["lanes"][LANE]["label"], LANES["lanes"][OTHER]["label"]
    nightly, weekly = (LONG_LANES["lanes"][lane]["label"] for lane in (NIGHTLY, WEEKLY))
    run_4100 = RUN_URL.replace("4242", "4100")
    assert _section(out, "Nightly and weekly lanes")[2:] == [
        "| Lane | Last full run | Covers | Result | Not passing | Not tested |",
        "|---|---|---|---|---|---|",
        f"| {nightly} | [2026-11-15 00:30 UTC]({run_4100}) | [`e9afbcc`..`05e878f`, 7 commits]("
        f"{COMFY}/compare/{LAST}...{HEAD}) | ❌ 1 of 2 failed | `{MUSIC}` smoke check failed | 0 |",
        ""]
    # The page shows a smoke result, so it says what smoke level means.
    assert lines[2] == run_report.SMOKE_NOTE
    assert f"{run_report.LEGEND} {run_report.SMOKE_LEGEND}" in lines
    # Its open failures are listed with those of the per-commit lanes, worded from
    # the workflow's own range, and only for the workflows the lane runs today.
    failures = _section(out, "Open failures")
    assert failures[2] == f"{main}:" and failures[4].startswith("- `sdxl_t2i`: output changed, ")
    assert failures[5:] == ["", f"{nightly}:", "", (
        f"- `{MUSIC}`: smoke check failed, failing since [`05e878f`]({COMFY}/commit/{HEAD}) "
        f"(a run that covered 7 commits, [compare]({COMFY}/compare/{LAST}...{HEAD})), first seen "
        f"2026-11-15, 1 run, last passed at [`e9afbcc`]({COMFY}/commit/{LAST})"), ""]
    assert CLIP not in out
    # No column, no row of its own commits, and no note: it is tested.
    matrix = _section(out, "Recent commits")
    assert matrix[2] == f"| Commit | Subject | PR | Author | {main} | {other} | Note |"
    assert "05e878f" not in "\n".join(matrix)
    assert nightly not in "\n".join(_section(out, "This commit on every lane"))
    legacy = LANES["lanes"][LEGACY]["label"]
    assert [line[:len(line) - len(line.split(" is ", 1)[1])] for line in
            _section(out, "Other lanes")[2:5]] == [
        f"- {LANES['lanes'][IDLE]['label']} is ", f"- {weekly} is ", f"- {legacy} is "]
    assert f"- {weekly} is not tested yet: it has no endpoint." in lines
    # A cron that GitHub dropped leaves no run and no mail: the mark is its only sign.
    stale = run_report.overview_report(reader, LONG_LANES, SHA, "master", manifest=CADENCES,
                                       now=NIGHT_TS + 37 * 3600)
    assert f"| [2026-11-15 00:30 UTC]({run_4100}) {run_report.WARN} no run for 37 hours |" in stale
    assert stale.replace(f" {run_report.WARN} no run for 37 hours", "") == out
    # Without the manifest nothing is left out, as on the per-commit lanes.
    bare = run_report.overview_report(reader, LONG_LANES, SHA, "master", now=NIGHT_TS + 3600)
    assert f"- `{CLIP}`: smoke check failed, failing since " in bare
    # The clock is read when no time is given.
    assert "### Nightly and weekly lanes" in run_report.overview_report(
        reader, LONG_LANES, SHA, "master", manifest=CADENCES)


def test_the_overview_reads_a_nightly_lane_from_its_last_full_run():
    """A dispatch that tested one workflow again is newer than the lane's run
    of the night. The row, and with it the smoke words of the page, come from
    the newest run that did not name its workflows."""
    files = _overview_files()
    again = {**_entry(SHA, NIGHT_TS + 7200, {"wan_ref": "pass"}, run=4200), "prev": HEAD,
             "named": True}
    night = {**_entry(HEAD, NIGHT_TS, {MUSIC: {"v": "pass", "l": "smoke"},
                                       MESH: {"v": "pass", "l": "smoke"}, "wan_ref": "pass"},
                      run=4100, n=7), "prev": LAST}
    files[f"regression/index/master/{NIGHTLY}.json"] = _head([again, night])
    manifest = {"workflows": {**CADENCES["workflows"],
                              "wan_ref": {"enabled": True, "cadence": "nightly"}}}
    out = run_report.overview_report(run_report.DictReader(files, RESULTS), LONG_LANES, SHA,
                                     "master", manifest=manifest, now=NIGHT_TS + 9000)
    row = _section(out, "Nightly and weekly lanes")[4]
    assert f"| [2026-11-15 00:30 UTC]({RUN_URL.replace('4242', '4100')}) | " in row
    assert row.endswith("| ✅ pass (1 reference, 2 smoke) |  | 0 |")
    # Nothing else on the page is at smoke level: the row alone brings the words.
    assert out.split("\n")[2] == run_report.SMOKE_NOTE and run_report.SMOKE_LEGEND in out
    assert "smoke check failed" not in out.replace(run_report.SMOKE_LEGEND, "")
    # Without the manifest the lane's workflows are not known. The mark of the dispatch
    # is in the index, so the row is the same one.
    bare = run_report.overview_report(run_report.DictReader(files, RESULTS), LONG_LANES, SHA,
                                      "master", now=NIGHT_TS + 9000)
    assert _section(bare, "Nightly and weekly lanes")[4] == row


def test_a_nightly_lane_with_a_reference_level_failure_brings_no_smoke_words():
    files = _long_overview_files()
    files[f"regression/index/master/{NIGHTLY}.json"] = _head([
        {**_entry(HEAD, NIGHT_TS, {MUSIC: "fail", MESH: "pass"}, run=4100, n=7), "prev": LAST}])
    out = run_report.overview_report(run_report.DictReader(files, RESULTS), LONG_LANES, SHA,
                                     "master", manifest=CADENCES, now=NIGHT_TS)
    assert f"| ❌ 1 of 2 failed | `{MUSIC}` output changed | 0 |" in out
    assert "smoke" not in out and run_report.LEGEND in out


def test_lanes_declared_before_their_endpoint_add_a_note_each_and_nothing_else():
    """The change that declares the nightly and the weekly lane must not move
    a byte of the per-commit pages but for the two notes that name them."""
    files = _overview_files()
    before = _overview(files)
    declared = {"primary": LANE, "lanes": {
        **LANES["lanes"],
        NIGHTLY: {**NIGHT_LANES["lanes"][NIGHTLY], "endpoint_id": None},
        WEEKLY: LONG_LANES["lanes"][WEEKLY]}}
    notes = [f"- {declared['lanes'][lane]['label']} is not tested yet: it has no endpoint."
             for lane in (NIGHTLY, WEEKLY)]
    for manifest in (None, MANIFEST, CADENCES):
        after = run_report.overview_report(run_report.DictReader(files, RESULTS), declared, SHA,
                                           "master", manifest=manifest).split("\n")
        base = run_report.overview_report(run_report.DictReader(files, RESULTS), LANES, SHA,
                                          "master", manifest=manifest).split("\n")
        at = after.index(notes[0])
        assert after[at:at + 2] == notes and after[:at] + after[at + 2:] == base
        assert "### Nightly and weekly lanes" not in after
    assert before == run_report.overview_report(run_report.DictReader(files, RESULTS), LANES,
                                                SHA, "master", manifest=MANIFEST)
    # An index of such a lane, left by a dispatch or a later change, is not shown
    # as long as the lane has no endpoint.
    files[f"regression/index/master/{NIGHTLY}.json"] = _long_overview_files()[
        f"regression/index/master/{NIGHTLY}.json"]
    after = run_report.overview_report(run_report.DictReader(files, RESULTS), declared, SHA,
                                       "master").split("\n")
    assert after[:at] + after[at + 2:] == before.split("\n")


# -- a nightly or weekly run: what it did not finish, reach or test ---------------------------
NIGHT_RUN = f"{NP}/runs/master/{HEAD}"
DISPATCH = ("gh workflow run nightly-weekly.yml --repo Comfy-Org/comfyci-runpod-worker "
            "-f cadence=nightly")
NEXT_RUN = "Nothing tests it again before the lane's next run."


def _late(wf):
    """The cell of a workflow that the run's deadline did not reach."""
    return {"workflow_id": wf, "worker_status": "not_reached", "verdict": "infra_error",
            "error": "not started: its job can take 45 min, and 12 min were left before the "
                     "run's deadline", "level": "smoke", "prior_verdict": "pass",
            "prior_commit": LAST}


def _night_files(cells, runs=None, **kw):
    """The results tree of a nightly run of HEAD. `runs` holds run records
    that differ from what the cells say. A workflow that was not reached has
    none: no job was submitted for it."""
    files = _smoke_files(cells, lane=NIGHTLY, commit=HEAD, **kw)
    files.update({f"{NIGHT_RUN}/{wf}/run.json": run for wf, run in (runs or {}).items()})
    for cell in cells:
        if cell["worker_status"] == "not_reached":
            del files[f"{NIGHT_RUN}/{cell['workflow_id']}/run.json"]
    return files


def _night_report(cells, runs=None, **kw):
    return _smoke_report(_night_files(cells, runs, **kw), lane=NIGHTLY, commit=HEAD)


def test_a_summary_of_a_run_that_did_not_finish_says_so():
    """A nightly or weekly run pushes what it has every 10 workflows. When it
    dies, that summary is what the results branch holds, and a report of it
    must not read like the lane's result."""
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    music = _smoke_cell(MUSIC, [_flac()], output_kind="audio")
    files = _night_files([mesh, music], partial={"done": 2, "of": 55})
    # The index is written by the final publish alone: this commit is not in it.
    head = f"regression/index/master/{NIGHTLY}.json"
    files[head] = _head([_entry(LAST, 1794616236, {MESH: "pass", MUSIC: "pass"})])
    manifest = {"workflows": {wf: {"enabled": True, "level": "smoke", "cadence": "nightly"}
                              for wf in (MESH, MUSIC, CLIP)}}
    manifest["workflows"]["flux_dev_t2i"] = {"enabled": True}
    out = _smoke_report(files, lane=NIGHTLY, commit=HEAD, manifest=manifest)
    lines = out.split("\n")
    # Green so far is not green.
    assert lines[0] == ("## ⚠️ GPU regression · nightly · py3.13 / torch 2.14.1 / cu130 / "
                        "RTX 4090 · `05e878f3`")
    assert lines[1:5] == ["", "> [!WARNING]", (
        "> This run did not finish: 2 of 55 workflows have a result. The report shows what the "
        "run pushed before it stopped, and the other workflows are not in it. A run that is "
        "still going reads the same until it publishes its last result."), ""]
    assert "| Progress | 2 of 55 workflows have a result. The run did not finish |" in lines
    assert "All 2 workflows passed their smoke checks." in lines
    assert "### All workflows (2)" in lines
    # The workflows of the lane without a result, and none of another cadence.
    assert (f"- ➖ not run, on this lane, and the run had no result for them when it pushed this "
            f"summary: `{CLIP}`") in lines
    assert "The commit of this report is not among them." in out
    # The page of a report-only run, or of a run that skipped the commit, is
    # another run's than the one that did not finish, and says so.
    out = _smoke_report(files, lane=NIGHTLY, commit=HEAD, skipped=True)
    assert out.split("\n")[2:7] == ["> [!WARNING]", (
        "> The run that tested this commit did not finish: 2 of 55 workflows have a result. "
        "The report shows what the run pushed before it stopped, and the other workflows are "
        "not in it. A run that is still going reads the same until it publishes its last "
        "result."), "", "> [!NOTE]", (
        "> This run did not test this commit again: the lane already has part of a result for "
        f"it, tested 2026-11-15 00:30 UTC in [an earlier run]({RUN_URL}). The report below "
        "shows that result.")]
    # A failure it already had keeps the title red.
    red = _night_files([mesh, _silent(prior_commit=LAST)], partial={"done": 2, "of": 55})
    assert _smoke_report(red, lane=NIGHTLY, commit=HEAD).startswith("## ❌ GPU regression · ")
    # Numbers that cannot be read are not shown as numbers.
    for odd in ({}, {"done": "2", "of": 55}, {"done": 9, "of": 3}, {"done": True, "of": 5}):
        out = _night_report([mesh], partial=odd)
        assert "> This run did not finish: not every workflow has a result. The report" in out
    # The run's final summary has no such key, and its report no such words.
    out = _night_report([mesh, music])
    assert out.startswith("## ✅ ") and "This run did not finish" not in out
    assert "Progress" not in out and "had no result for them" not in out
    text = run_report.progress_text
    assert text({"partial": {"done": 40, "of": 55}}) == "40 of 55 workflows have a result"
    assert text({"partial": {"done": 0, "of": 1}}) == "0 of 1 workflow has a result"
    assert text({}) == text(None) == text({"partial": None}) == text({"partial": [1]}) == ""


def test_a_run_that_died_says_what_its_checkpoints_left_on_the_results_branch(tmp_path):
    """The job that was stopped reports from its own files, which hold more
    than its last checkpoint. That checkpoint is no "earlier result"."""
    published = _summary([_cell("flux_dev_t2i")], partial={"done": 1, "of": 3})
    workdir = tmp_path / "regression-work"
    for wf in ("flux_dev_t2i", "sdxl_t2i"):
        (workdir / wf).mkdir(parents=True)
        (workdir / wf / "run.json").write_text(json.dumps(_run(wf)), encoding="utf-8")
        (workdir / wf / "comparison.json").write_text(json.dumps(
            {"verdict": "pass", "vs_golden": IDENT, "golden_tag": "v0.42.0"}), encoding="utf-8")
    out = _report(_tree(published), workdir=str(workdir), summary_file=str(tmp_path / "no.json"))
    assert ("> This run stopped before it published its results. The report is rebuilt from the "
            "files on the runner, so images and links to published files are missing. The "
            "attached files hold the outputs. The results branch holds part of a result for "
            "this commit: 1 of 3 workflows have a result, pushed by a run before it had them "
            "all.") in out.split("\n")
    assert "an earlier result for this commit" not in out and "### All workflows (2)" in out
    # The rebuilt summary is not partial itself: it has every file of the runner.
    assert "This run did not finish" not in out and "Progress" not in out


def test_a_nightly_run_that_was_stopped_is_not_green(tmp_path):
    """A nightly or weekly run tests a few workflows at a time. One that is
    killed during its 25th job leaves 24 green folders on the runner. Its page
    had a green check and "All 24 workflows passed", and listed the workflows
    it never started as left out by a dispatch."""
    workdir = tmp_path / "regression-work"

    def left(verdicts):
        for wf, verdict in verdicts.items():
            (workdir / wf).mkdir(parents=True, exist_ok=True)
            (workdir / wf / "run.json").write_text(json.dumps(_run(wf)), encoding="utf-8")
            (workdir / wf / "comparison.json").write_text(json.dumps({
                "verdict": verdict, "level": "smoke",
                "smoke": {"files": [], "notes": [], "problems": [
                    {"file": None, "check": "missing", "text": "wrote no file"}] * (
                        verdict == "fail")}}), encoding="utf-8")

    def page(lane=NIGHTLY, commit=HEAD):
        files = {f"regression/index/master/{lane}.json": _head([_entry(LAST, 1794616236, {
            MESH: "pass", MUSIC: "pass"})]), "regression/index/lanes.json": {"lanes": {}}}
        manifest = {"workflows": {wf: {"enabled": True, "level": "smoke", "cadence": (
            "nightly" if lane == NIGHTLY else "per-commit")} for wf in (MESH, MUSIC, CLIP)}}
        return _smoke_report(files, lane=lane, commit=commit, workdir=str(workdir),
                             summary_file=str(tmp_path / "no.json"), manifest=manifest)

    left({MESH: "pass", MUSIC: "pass"})
    lines = page().split("\n")
    # Every workflow it has is green, and the run is not.
    assert lines[0] == ("## ⚠️ GPU regression · nightly · py3.13 / torch 2.14.1 / cu130 / "
                        "RTX 4090 · `05e878f3`")
    assert lines[2:4] == ["> [!WARNING]", (
        "> This run stopped before it published its results. The report is rebuilt from the "
        "files on the runner, so images and links to published files are missing. The attached "
        "files hold the outputs. It shows the workflows that had a result when the run stopped, "
        "and no other.")]
    assert "All 2 workflows passed their smoke checks." in lines
    assert (f"- ➖ not run, on this lane, and the run left no result for them: it stopped before "
            f"it published (a dispatch can also name only some workflows): `{CLIP}`") in lines
    assert not [line for line in lines if "not part of this run" in line]
    # A failure among them keeps the title red.
    left({MUSIC: "fail"})
    assert page().startswith("## ❌ GPU regression · nightly")
    # A per-commit run submits every job at once, and its page is the one it was.
    left({MUSIC: "pass"})
    out = page(LANE, SHA)
    assert out.startswith("## ✅ GPU regression · py3.13")
    assert "It shows the workflows that had a result" not in out
    assert "it stopped before it published" not in out and f"`{CLIP}`" in out
    assert not run_report.stopped_early(run_report.LaneData(
        commit=SHA, branch="master", lane_id=NIGHTLY, lane=NIGHT_LANES["lanes"][NIGHTLY],
        prefix=NP, run_dir=None, reader=run_report.Reader(), source="published"))


def test_an_execution_error_of_a_smoke_workflow_names_its_last_result_in_smoke_words():
    """On the per-commit lane too. Nothing was compared before the error, so
    the last result is no "pass" of the legend and no "output changed"."""
    prev = f"[`{PREV[:7]}`]({COMFY}/commit/{PREV})"
    for prior, said in (("pass", "smoke pass"), ("fail", "smoke check failed")):
        broken = _smoke_cell(CLIP, verdict="execution_error", prior_verdict=prior,
                             prior_commit=PREV)
        out = _smoke_report(_smoke_files([broken]))
        assert (f"Before this commit it last ran to the end at {prev} ({said}), so the error is "
                "new at this commit.") in out, prior
    # A reference-level workflow keeps the words of its level.
    for prior, said in (("pass", "pass"), ("fail", "output changed")):
        broken = _cell("flux_dev_t2i", "execution_error", prior_verdict=prior, prior_commit=PREV)
        out = _report(_tree(_summary([broken])))
        assert (f"Before this commit it last ran to the end at {prev} ({said}), so the error is "
                "new at this commit.") in out, prior


def test_a_changed_output_after_a_run_that_left_none_says_what_is_missing():
    """A full run of a nightly or weekly lane always moves the pointer. So the
    run before can be one in which this workflow ended in an error and wrote
    nothing. "There is no previous run" stood under a row that named it."""
    last = f"[`e9afbcc`]({COMFY}/commit/{LAST})"
    cell = _cell("flux_dev_t2i", "fail", previous_commit=LAST, previous_verdict=None,
                 prior_verdict="execution_error", prior_commit=LAST)
    out = _nightly([cell, _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])])
    assert ("The output differs from the reference beyond its limits. The run before, at "
            f"{last}, left no output of this workflow, so nothing tells whether the change is "
            "new in this run.") in out.split("\n")
    assert "There is no previous run" not in out
    assert f"| Covers | 7 commits since {last}" in out
    # No pointer at all: there is no run before.
    first = _cell("flux_dev_t2i", "fail", previous_commit=None, previous_verdict=None)
    out = _smoke_report(_smoke_files([first], lane=NIGHTLY, commit=HEAD, tested_range=None),
                        lane=NIGHTLY, commit=HEAD)
    assert ("The output differs from the reference beyond its limits. There is no previous run "
            "to tell whether the change is new at this commit.") in out.split("\n")
    # The per-commit lane keeps its sentence: its pointer moves only on runs with outputs.
    out = _report(_tree(_summary([_cell("flux_dev_t2i", "fail")])))
    assert "There is no previous run to tell whether the change is new at this commit." in out
    assert "left no output of this workflow" not in out


def test_an_output_the_worker_did_not_send_is_not_said_to_be_attached():
    """The artifact takes the runner's folder, and a file over the worker's
    size limit never reached the runner. The row pointed to the attached files
    while the Checked cell beside it said "not sent"."""
    big = _flac(bytes=20 * 1024 * 1024, sent=False, published=False)
    note = f"{FLAC} was not sent (over the worker's size limit): checked by size only"
    cell = _smoke_cell(MUSIC, [big], output_kind="audio", notes=[note])
    out = _smoke_report(_smoke_files([cell]), artifact_url=ARTIFACT)
    assert (f"| `{MUSIC}` | ✅ smoke pass | audio, not sent | 1 file not sent, checked by size "
            "only |") in out
    assert "attached files](" not in out.split("### All workflows")[1].split("<details>")[0]
    # Without an artifact too: nothing points to the run for a file it never had.
    assert f"| `{MUSIC}` | ✅ smoke pass | audio, not sent |" in _smoke_report(_smoke_files([cell]))
    # A file that was sent and held back is in the attached files, as before.
    held = _smoke_cell(MUSIC, [_flac(published=False)], output_kind="audio")
    out = _smoke_report(_smoke_files([held]), artifact_url=ARTIFACT)
    assert f"| `{MUSIC}` | ✅ smoke pass | audio, in the [attached files]({ARTIFACT}) |" in out
    # A picture the worker did not send has no thumbnail either: the same words.
    huge = _png(f"{MESH}_00001_.png", bytes=20 * 1024 * 1024, sent=False, published=False)
    cell = {**_smoke_cell(MESH, [huge]), "thumbnail": None}
    assert f"| `{MESH}` | ✅ smoke pass | PNG, not sent |" in _smoke_report(
        _smoke_files([cell]), artifact_url=ARTIFACT)


def test_a_smoke_failure_after_a_reference_failure_is_not_inherited_from_it():
    """A workflow that moves to smoke level while its output differs from its
    reference. Its first smoke check fails: that is a new failure, placed by
    the workflow's last pass, and not the old one going on. A summary without
    the workflow's last result is read from the index by the same rule."""
    blank = [{"file": f"{MESH}_00001_.png", "check": "blank", "text": "blank"}]
    cell = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")], blank, prior_verdict=None,
                       prior_commit=None)
    before = [_entry(PREV, 1794700000, {MESH: "fail"}), _entry(FIRST, 1794690000, {MESH: "pass"})]
    out = _smoke_report(_smoke_files([cell], before=before))
    assert f"#### ❌ `{MESH}`: smoke check failed (new since `aaaaaaa`)" in out.split("\n")
    assert "inherited" not in "\n".join(_block(out, MESH))
    # After a failed smoke check it is the same failure.
    smoke_before = [_entry(PREV, 1794700000, {MESH: {"v": "fail", "l": "smoke"}})]
    out = _smoke_report(_smoke_files([cell], before=smoke_before))
    assert f"#### ❌ `{MESH}`: smoke check failed (inherited)" in out.split("\n")
    # A reference-level cell reads every earlier result, as it did.
    prior = run_report.prior_result
    entries = run_report.entries_of(_head(before))
    assert prior(entries, SHA, MESH, {"level": "smoke"}) == {"verdict": "pass", "commit": FIRST}
    assert prior(entries, SHA, MESH, {}) == {"verdict": "fail", "commit": PREV}
    assert prior(entries, SHA, MESH, {"level": "smoke", "prior_verdict": "fail",
                                      "prior_commit": PREV}) == {"verdict": "fail", "commit": PREV}


def test_only_the_row_of_a_run_that_named_its_workflows_says_how_many_it_has():
    """The history row of a step of a search says "1 of N workflows", so a
    green step does not read as a green night. The row took that from a count
    against the lane's workflows of today: once a workflow had joined the
    lane, every older full run read "1 of 2 workflows" too, the words of a
    run that named one workflow."""
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    music = _smoke_cell(MUSIC, [_flac()], output_kind="audio")
    manifest = {"workflows": {wf: {"enabled": True, "level": "smoke", "cadence": "nightly"}
                              for wf in (MESH, MUSIC)}}
    # A full run from when the lane ran one workflow, and a step after it.
    full = _entry(FIRST, 1794530000, {MESH: {"v": "pass", "l": "smoke"}}, pr=17699)
    step = {**_entry(LAST, 1794616236, {MESH: {"v": "pass", "l": "smoke"}}, pr=17700),
            "named": True}

    def results(before, **kw):
        files = _night_files([mesh, music], before=before)
        out = _smoke_report(files, lane=NIGHTLY, commit=HEAD, **kw)
        rows = [line for line in _section(out, "Recent commits on this lane")
                if line.startswith("| [`")]
        return [row.split(" | ")[-2] for row in rows]

    assert results([step, full], manifest=manifest) == [
        "✅ smoke pass", "✅ smoke pass · 1 of 2 workflows", "✅ smoke pass"]
    # The mark alone says so: not a count, and nothing but `true`.
    for odd in (None, False, "yes", 1):
        assert results([{**step, "named": odd}, full], manifest=manifest)[1] == "✅ smoke pass"
    whole = {**step, "w": {MESH: {"v": "pass", "l": "smoke"}, MUSIC: {"v": "pass", "l": "smoke"}}}
    assert results([whole, full], manifest=manifest)[1] == "✅ smoke pass · 2 of 2 workflows"
    # The lane's workflows are not known: no row counts against them.
    assert results([step, full]) == ["✅ smoke pass"] * 3
    # A per-commit lane keeps its rows, whatever an entry says.
    cells = [_cell("flux_dev_t2i"), _cell("sdxl_t2i")]
    head = _head([_entry(SHA, 1794702636, {"flux_dev_t2i": "pass", "sdxl_t2i": "pass"}, run=4242),
                  {**_entry(PREV, 1794700000, {"flux_dev_t2i": "pass"}, pr=17753), "named": True}])
    out = _report(_tree(_summary(cells), head=head), manifest=MANIFEST)
    assert "workflows |" not in "\n".join(_section(out, "Recent commits on this lane"))


def test_a_failure_whose_chain_has_a_later_pass_does_not_name_the_older_one_as_its_last():
    """A step of a search for the commit: a run that names the workflow at a
    commit inside the range, and it passes. The open failure then names that
    pass as the last. The page of the failing run still said "It last passed
    at e9afbcc. 7 commits since then" in the failure's own block: one page,
    two last passes, and a count that reached back past the newer one."""
    mid = "7c1d2e3" + "5" * 33
    last, step = (f"[`{sha[:7]}`]({COMFY}/commit/{sha})" for sha in (LAST, mid))
    chain = {"commit": HEAD, "run_ts": 1794702636, "kind": "fail", "prev_good": mid, "pr": 17754,
             "author": "kijai", "golden": None, "runs": 1, "level": "smoke",
             "range": {"n": None, "url": f"{COMFY}/compare/{mid}...{HEAD}"}}
    before = [{**_entry(mid, 1794650000, {MUSIC: {"v": "pass", "l": "smoke"}}, pr=17720),
               "named": True},
              _entry(LAST, 1794616236, {MUSIC: {"v": "pass", "l": "smoke"}}, pr=17700)]
    ranges = {MUSIC: {"prev": LAST, "commits_between": 7, "compare_url": "u"}}
    out = _nightly([_silent(prior_commit=LAST)], before=before, first_bad={MUSIC: chain},
                   ranges=ranges)
    block = _block(out, MUSIC)
    assert block[0] == f"#### ❌ `{MUSIC}`: smoke check failed (new since `7c1d2e3`)"
    assert block[-4:] == [
        f"When this run was made it had last passed at {last}. A later run of this workflow "
        f"passed at {step}, which lies between that commit and this one. The commits since that "
        f"pass: [compare]({COMFY}/compare/{mid}...{HEAD}). To test one commit of that range "
        "(needs write access to the worker repository):",
        "```", f"{DISPATCH} -f commit=<sha> -f workflows={MUSIC}", "```"]
    assert "7 commits" not in "\n".join(block) and "It last passed at" not in out
    # The line of the open failure names the same last pass and the same commits.
    assert (f"- `{MUSIC}`: smoke check failed, failing since [`05e878f`]({COMFY}/commit/{HEAD}) "
            f"(one of the commits since its last pass, [compare]({COMFY}/compare/{mid}...{HEAD})"
            f"), first seen 2026-11-15, 1 run, last passed at {step}") in out.split("\n")

    def own_block(first_bad, cells=None, entries=before):
        cells = cells or [_silent(prior_commit=LAST)]
        return _block(_nightly(cells, before=entries, first_bad=first_bad, ranges=ranges), MUSIC)

    # A failing step between the two starts the chain: its commits are the ones to look at.
    failed = "8d2e3f4" + "6" * 33
    block = own_block({MUSIC: {**chain, "commit": failed}})
    assert f"The commits since that pass: [compare]({COMFY}/compare/{mid}...{failed})." in block[-4]
    # Nothing passed since: the block says what it said, with the run's count.
    was = (f"It last passed at {last}. 7 commits since then: [compare]({COMFY}/compare/{LAST}..."
           f"{HEAD}). To test one commit")
    for same in ({MUSIC: {**chain, "prev_good": LAST}}, {}):
        block = own_block(same)
        assert block[0].endswith("(new since `e9afbcc`, 7 commits)") and was in block[-4], same
    # A last pass that is not a later one (it stands before the run's, or
    # nowhere in the history) is not taken for one.
    for odd in (FIRST, "9" * 40, None, 7):
        entries = [*before, _entry(FIRST, 1794530000, {MUSIC: {"v": "pass", "l": "smoke"}})]
        assert was in own_block({MUSIC: {**chain, "prev_good": odd}}, entries=entries)[-4], odd
    # Its last result was no pass: that sentence stays as it is.
    block = own_block({MUSIC: chain}, [_silent(prior_commit=LAST, prior_verdict="execution_error")])
    assert block[-4].startswith(f"Its last result before this one, at {last}, was an execution")
    assert "7 commits since then" in block[-4]

    # An execution error on such a lane is placed the same way.
    broken = _smoke_cell(MUSIC, verdict="execution_error", prior_commit=LAST)
    block = own_block({MUSIC: {**chain, "kind": "execution_error"}}, [broken])
    assert block[0] == f"#### ❌ `{MUSIC}`: execution error (new since `7c1d2e3`)"
    assert block[1] == (
        "The workflow did not finish on this commit. Worker status: <code>timeout</code>. When "
        f"this run was made it had last passed at {last}. A later run of this workflow passed at "
        f"{step}, which lies between that commit and this one. The error started after that "
        f"pass. The commits since that pass: [compare]({COMFY}/compare/{mid}...{HEAD}). To test "
        "one commit of that range (needs write access to the worker repository):")
    # The per-commit lane reads the same, without the command of the other lanes.
    files = _smoke_files([_silent()], before=[{**before[0], "t": 1794701000}, _entry(
        PREV, 1794600000, {MUSIC: {"v": "pass", "l": "smoke"}}, pr=17753)],
        first_bad={MUSIC: {**chain, "commit": SHA}},
        tested_range={"prev": PREV, "commits_between": 3})
    block = _block(_smoke_report(files), MUSIC)
    assert block[0].endswith("(new since `7c1d2e3`)") and block[-1] == (
        f"When this run was made it had last passed at [`bbbbbbb`]({COMFY}/commit/{PREV}). A "
        f"later run of this workflow passed at {step}, which lies between that commit and this "
        f"one. The commits since that pass: [compare]({COMFY}/compare/{mid}...{SHA}).")


def test_a_weekly_page_names_no_attempt_when_the_cron_leaves_the_commit_alone():
    """The Tested row said "attempt 1 of 2" on a page whose only untested
    workflows are ones the cron does not test again."""
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    out = _night_report([mesh, _late("wan_a"), _late("wan_b")])
    assert not run_report.regression_skip.decide(HEAD, _summary(
        [mesh, _late("wan_a")], lane=NIGHTLY, commit=HEAD))[0]
    tested = next(line for line in out.split("\n") if line.startswith("| Tested |"))
    assert "attempt" not in tested and f"[Actions run]({RUN_URL}) |" in tested
    # One workflow that another run can test: a second attempt can follow.
    flaky = _smoke_cell(MUSIC, verdict="infra_error", worker_status="worker_error")
    out = _night_report([mesh, _late("wan_a"), flaky])
    tested = next(line for line in out.split("\n") if line.startswith("| Tested |"))
    assert "· attempt 1 of 2 ·" in tested


def test_a_workflow_that_a_later_run_tested_again_is_marked():
    """A dispatch that names workflows at a commit the lane has a result for
    replaces their cells and records itself under `retests`. The facts of the
    summary are still those of the lane's own run, so the rows say which
    results are another run's."""
    later, last = RUN_URL.replace("4242", "5555"), RUN_URL.replace("4242", "6666")
    retests = [{"run_ts": 1794706236, "run_url": later, "run_id": 5555, "env": ENV,
                "workflows": ["sdxl_t2i"]}]
    cells = [_cell("flux_dev_t2i"), _cell("sdxl_t2i")]
    out = _report(_tree(_summary(cells, retests=retests)))
    lines = out.split("\n")
    assert ("| Tested | 2026-11-15 00:30 UTC · 10 min after the commit · "
            f"[Actions run]({RUN_URL}) |") in lines
    assert (f"| Tested again | `sdxl_t2i` · 2026-11-15 01:30 UTC · [Actions run]({later}) |"
            in lines)
    assert [line[:90] for line in lines if line.startswith("| `sdxl_t2i` | ✅ pass")] == [
        f"| `sdxl_t2i` | ✅ pass · tested again in [a later run]({later}) | <a href="[:90]]
    assert any(line.startswith("| `flux_dev_t2i` | ✅ pass | <a href=") for line in lines)
    # Tested again twice: the table lists both runs, and the row links the last.
    retests.append({"run_ts": 1794709836, "run_url": last, "run_id": 6666, "env": ENV,
                    "workflows": ["flux_dev_t2i", "sdxl_t2i"]})
    out = _report(_tree(_summary(cells, retests=retests)))
    assert (f"| Tested again | `sdxl_t2i` · 2026-11-15 01:30 UTC · [Actions run]({later}); "
            f"`flux_dev_t2i`, `sdxl_t2i` · 2026-11-15 02:30 UTC · [Actions run]({last}) |") in out
    assert out.count(f"tested again in [a later run]({last})") == 2 and later + ") | <a" not in out
    # A re-test that names no run, and records that are not what they should be.
    for odd, said in (([{"workflows": ["sdxl_t2i"]}], "| Tested again | `sdxl_t2i` |"),
                      ([{"run_ts": "x", "run_url": "https://evil.example/", "run_id": "1",
                         "workflows": ["sdxl_t2i", 7, EVIL]}], "| Tested again | `sdxl_t2i`, "),
                      ("sdxl_t2i", None), ([None, 3, {"workflows": "sdxl_t2i"}, {}], None)):
        out = _report(_tree(_summary(cells, retests=odd)))
        assert out and "evil.example/)" not in out
        assert ("Tested again" in out) == bool(said) and (not said or said in out), odd
    assert "tested again in a later run |" in _report(_tree(_summary(
        cells, retests=[{"workflows": ["sdxl_t2i"]}])))
    # A summary without re-tests reads as it did.
    assert "ested again" not in _report(_tree(_summary(cells)))
    assert run_report.retests_of(None) == {} == run_report.retests_of({"retests": {}})


def test_the_workflows_a_deadline_did_not_reach_are_one_list():
    """A weekly run that runs out of time can leave dozens of workflows
    without a job. They share one story, so they share one block."""
    mesh = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    out = _night_report([mesh, _late("wan_a"), _late("wan_b"), _late("wan_c")])
    lines = out.split("\n")
    assert out.startswith("## ⚠️ ")
    assert "No failures: 1 passed its smoke checks. 3 were not tested." in lines
    at = lines.index("### What needs attention")
    assert lines[at:at + 13] == [
        "### What needs attention", "",
        "#### ⚠️ Not reached before the run's deadline (3)", "",
        "The run stops submitting jobs when one could still be running at its deadline. No job "
        "was submitted for these workflows, so they have no result. This is not a verdict on "
        "the commit.", "",
        "`wan_a`, `wan_b`, `wan_c`", "",
        "Nothing tests them again before the lane's next run. To test them now (needs write "
        "access to the worker repository):", "",
        "```", f"{DISPATCH} -f commit={HEAD} -f workflows=wan_a,wan_b,wan_c", "```"]
    assert not [line for line in lines if line.startswith("#### ") and "`wan_" in line]
    assert "An infrastructure problem stopped this workflow" not in out
    assert "| `wan_a` | ⚠️ infra error |" in out
    # Next to a failure and to a workflow that was not tested for another reason:
    # each of those has its block, and the list comes last.
    flaky = _smoke_cell(MUSIC, verdict="infra_error", worker_status="worker_error",
                        prior_commit=LAST)
    out = _night_report([_silent(prior_commit=LAST, workflow_id="bad_wf"), flaky, _late("wan_a")])
    heads = [line for line in out.split("\n") if line.startswith("#### ")]
    assert [head.split(":")[0] for head in heads] == [
        "#### ❌ `bad_wf`", f"#### ⚠️ `{MUSIC}`",
        "#### ⚠️ Not reached before the run's deadline (1)"]
    # A name that is no workflow id is listed, and no command is built from it.
    out = _night_report([mesh, _late("wan_a"), _late("x; rm -rf ~")])
    assert "Nothing tests them again before the lane's next run." in out.split("\n")
    assert "-f workflows=wan_a" not in out and "rm -rf" in out
    # A per-commit lane that was given a deadline: the list, and nothing about a next run.
    out = _smoke_report(_smoke_files([mesh, _late("wan_a")]))
    assert "#### ⚠️ Not reached before the run's deadline (1)" in out
    assert "lane's next run" not in out and "nightly-weekly.yml" not in out


def test_an_untested_workflow_on_a_nightly_lane_says_what_the_run_did_and_what_is_left():
    """No cron tests the commit again the next minute on such a lane. The run
    itself tries once more when that can help and its deadline allows, the
    page says whether it did, and after the run the reader has the command."""
    first = ("An infrastructure problem stopped this workflow. Worker status: <code>{}</code>. "
             "This is not a verdict on the commit. ")
    command = ["", "```", f"{DISPATCH} -f commit={HEAD} -f workflows={MUSIC}", "```"]
    no_second = ("The run did not try it again at its end: its deadline did not allow a second "
                 "job, or the run stopped before it came to that.")

    def page(status, run=None, **cell):
        cell = _smoke_cell(MUSIC, verdict="infra_error", worker_status=status, prior_commit=LAST,
                           **cell)
        runs = {MUSIC: _run(MUSIC, status, outputs=[], **(run or {}))}
        return _night_report([cell, _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])],
                             runs).split("\n")

    def block(status, run=None, **cell):
        lines = page(status, run, **cell)
        at = lines.index(f"#### ⚠️ `{MUSIC}`: infra error")
        return lines[at + 2:at + 7]

    # The result of the second job the run gave it (`tries` in its cell). The
    # record of that job is a first attempt's: the page does not call it one.
    assert block("worker_error", tries=2) == [
        first.format("worker\\_error") + "The run tried it once more at its end, and this is the "
        f"result of that second job. {NEXT_RUN} To test it now (needs write access to the worker "
        "repository):", *command]
    lines = page("worker_error", tries=2)
    assert [line for line in lines if "the run's second job for this workflow" in line]
    assert not [line for line in lines if "RunPod attempt" in line]
    # Whatever the second job ended with: it was tried.
    for status, run in (("infra_error", {"reason": "timed_out"}), ("missing_model", None)):
        assert "The run tried it once more at its end" in block(status, run, tries=2)[0]
    # No second job: the cell of the first stays, and the page says why there was none.
    assert block("worker_error") == [
        first.format("worker\\_error") + f"{no_second} {NEXT_RUN} To test it now (needs write "
        "access to the worker repository):", *command]
    assert [line for line in page("worker_error") if "RunPod attempt 1" in line]
    assert block("bad_request")[0] == (
        first.format("bad\\_request") + "The worker refused the request as it was sent. Another "
        f"attempt would have ended the same way, so the run did not try it again. {NEXT_RUN} To "
        "test it now (needs write access to the worker repository):")
    for odd in (1, "2", True, [2]):
        assert no_second in block("worker_error", tries=odd)[0], odd
    # Its job used up its whole time: RunPod ended it, or the runner gave up on it.
    assert block("infra_error", {"reason": "timed_out"})[0] == (
        first.format("infra\\_error") + "Its job used up its whole time, so the run did not try "
        f"it again: a second job can cost as much. {NEXT_RUN} To test it now (needs write access "
        "to the worker repository):")
    for status, why in (("missing_model", "Its models are not on the volume yet"),
                        ("model_access_error", "The model host refuses a file it needs")):
        assert block(status)[0] == (
            first.format(status.replace("_", "\\_")) + f"{why}. Another attempt would have ended "
            f"the same way, so the run did not try it again. {NEXT_RUN} To test it now (needs "
            "write access to the worker repository):"), status
    # Nothing here speaks of a cron that tests the commit once more.
    out = _night_report([_smoke_cell(MUSIC, verdict="infra_error", worker_status="worker_error")])
    assert "the cron" not in out and "1 was not tested." in out
    assert out.startswith(f"## {run_report.WARN} ")
    # A name that is no workflow id gets the words, and no command built from it.
    odd = "x; rm -rf ~"
    out = _night_report([_smoke_cell(odd, verdict="infra_error", worker_status="worker_error"),
                         _smoke_cell("y && z", verdict="execution_error", prior_commit=FIRST)],
                        ranges={"y && z": {"prev": FIRST, "commits_between": 12}})
    assert [line for line in out.split("\n")
            if line.endswith(f"or the run stopped before it came to that. {NEXT_RUN}")]
    assert "12 commits since then: " in out and "new since `aaaaaaa`, 12 commits" in out
    assert "gh workflow run" not in out and "```" not in out
    # The per-commit lane keeps its words for the same cell.
    out = _smoke_report(_smoke_files([_smoke_cell(MUSIC, verdict="infra_error",
                                                  worker_status="worker_error")]))
    assert ONCE_MORE in out and "next run" not in out


def test_an_execution_error_on_a_nightly_lane_is_placed_by_the_workflows_own_last_result():
    """The run covers 7 commits, and this workflow last ran to the end 12
    commits ago: the run before did not test it."""
    broken = _smoke_cell(CLIP, verdict="execution_error", prior_commit=FIRST)
    ranges = {CLIP: {"prev": FIRST, "commits_between": 12, "compare_url": "u"}}
    out = _nightly([broken, _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])], ranges=ranges)
    block = _block(out, CLIP)
    assert block[0] == f"#### ❌ `{CLIP}`: execution error (new since `aaaaaaa`, 12 commits)"
    assert block[1:5] == [
        "The workflow did not finish on this commit. Worker status: <code>timeout</code>. "
        f"Before this commit it last ran to the end at [`aaaaaaa`]({COMFY}/commit/{FIRST}) "
        "(smoke pass): the error started after that commit. 12 commits since then: [compare]("
        f"{COMFY}/compare/{FIRST}...{HEAD}). To test one commit of that range (needs write "
        "access to the worker repository):",
        "```", f"{DISPATCH} -f commit=<sha> -f workflows={CLIP}", "```"]
    assert "This run covers 7 commits, so the error started in one of them" not in out
    assert "1 of 2 workflows failed: 1 new in this run. 1 passed its smoke checks." in (
        out.split("\n"))
    # The facts table still says what the run covers, and that it places no failure.
    assert (f"| Covers | 7 commits since [`e9afbcc`]({COMFY}/commit/{LAST}) ([compare]({COMFY}"
            f"/compare/{LAST}...{HEAD})), the commit of the lane's previous full run. Each "
            "failure says where it started: a workflow's own last result can be older |"
            ) in out.split("\n")
    # One that was already failing is inherited, in its heading too.
    out = _nightly([_smoke_cell(CLIP, verdict="execution_error", prior_verdict="execution_error",
                                prior_commit=LAST)])
    assert _block(out, CLIP)[:2] == [
        f"#### ❌ `{CLIP}`: execution error (inherited)",
        "The workflow did not finish on this commit. Worker status: <code>timeout</code>. It "
        f"already ended in an execution error at [`e9afbcc`]({COMFY}/commit/{LAST}), so the "
        "error is inherited from an earlier commit."]
    # A reference-level workflow on such a lane reads the same way.
    out = _nightly([_cell("flux_dev_t2i", "execution_error", prior_commit=FIRST)], ranges={
        "flux_dev_t2i": {"prev": FIRST, "commits_between": 12}})
    assert "#### ❌ `flux_dev_t2i`: execution error (new since `aaaaaaa`, 12 commits)" in out
    assert "1 of 1 workflow failed: 1 new in this run." in out.split("\n")
    # Its last result is called what it was: a match with a reference, not a smoke pass.
    assert ") (pass): the error started after that commit. 12 commits since then: " in out
    # The per-commit lane words an execution error as it always did, from the run.
    rng = {"prev": PREV, "commits_between": 3}
    out = _report(_tree(_summary([_cell("flux_dev_t2i", "execution_error")], tested_range=rng,
                                 ranges={"flux_dev_t2i": {"prev": FIRST, "commits_between": 9}})))
    lines = out.split("\n")
    assert "#### ❌ `flux_dev_t2i`: execution error" in lines
    assert ("1 of 1 workflow failed, new in this run, which covers 3 commits." in lines)
    assert "so the error is new in this run. This run covers 3 commits, so the error started in " \
        "one of them" in out
    assert "| Covers | 3 commits since " in out and "previous full run" not in out


# -- a job that fetched its models --------------------------------------------------------
FETCH_HEADER = SMOKE_HEADER.replace("| Peak VRAM GB |",
                                    "| Models GB | Fetch s (MB/s) | Peak VRAM GB |")


def _fetched(cell, fetch_s, **fetch):
    """(a cell whose job fetched its models, its run record). `fetch_s` is the
    whole of the job's fetch step, `fetch` what the worker reported of it."""
    report = {"files": 2, "bytes": 0, "seconds": 0.0, "mb_per_s": None, "streams": 4,
              "token": True, "reused": 0, "reused_bytes": 0, "evicted": 0, **fetch}
    cell = {**cell, "timings": {**cell["timings"], "model_fetch_s": fetch_s}}
    return cell, {**_smoke_run(cell), "model_fetch": report, "timings": cell["timings"]}


def test_a_job_that_fetched_its_models_says_how_much_and_how_fast():
    """The two columns are the standing measurement of download speed: what a
    weekly run really downloads, and how long that takes."""
    mesh, mesh_run = _fetched(_smoke_cell(MESH, [_png(f"{MESH}_00001_.png")]), 180.4,
                              bytes=17_300_000_000, seconds=175.2, mb_per_s=98.7)
    # Every file of this one was still on the worker: nothing to download, no speed.
    music, music_run = _fetched(_smoke_cell(MUSIC, [_flac()], output_kind="audio"), 0.3,
                                files=0, reused=3, reused_bytes=7_500_000_000)
    out = _night_report([mesh, music], runs={MESH: mesh_run, MUSIC: music_run})
    table = _section(out, "All workflows")
    assert table[2] == FETCH_HEADER
    rows = {line.split(" | ")[0]: line for line in table[4:6]}
    assert " | 12.4 | 17.3 | 180.4 (99) | 21.3 | " in rows[f"| `{MESH}`"]
    assert " | 12.4 | 0.00 (+7.50 found on the worker) | 0.3 | 21.3 | " in rows[f"| `{MUSIC}`"]
    assert run_report.FETCH_NOTE in table and run_report.EXEC_NOTE in table
    assert ("| Model downloads | 17.3 GB downloaded by 2 jobs in 175 s of downloading, 99 MB/s "
            "on average. 7.50 GB more were still on a worker from an earlier job |"
            ) in out.split("\n")
    assert "Hugging Face token" not in out
    # A workflow of the run that fetched nothing has two empty cells.
    plain = _smoke_cell(CLIP, [_png(f"{CLIP}_00001_.png")])
    out = _night_report([mesh, plain], runs={MESH: mesh_run})
    row = next(line for line in out.split("\n") if line.startswith(f"| `{CLIP}` "))
    assert " | 12.4 |  |  | 21.3 | " in row
    assert "| Model downloads | 17.3 GB downloaded by 1 job in 175 s of downloading, " in out


def test_a_page_without_a_fetch_has_neither_column_nor_the_row():
    out = _night_report([_smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])])
    assert _section(out, "All workflows")[2] == SMOKE_HEADER
    assert "Model downloads" not in out and run_report.FETCH_NOTE not in out
    assert "Models GB" not in out and "Fetch s" not in out
    # A record that carries something else under the key is no fetch.
    cell = _smoke_cell(MESH, [_png(f"{MESH}_00001_.png")])
    for junk in (None, [], "yes", 3, {}):
        out = _night_report([cell], runs={MESH: {**_smoke_run(cell), "model_fetch": junk}})
        assert _section(out, "All workflows")[2] == SMOKE_HEADER, junk


def test_a_fetch_without_a_token_is_said_in_the_facts():
    """Without a token public files still download, on a limit that every
    user of the worker's address shares. Nothing else would tell anyone that
    the endpoint's secret is missing."""
    mesh, mesh_run = _fetched(_smoke_cell(MESH, [_png(f"{MESH}_00001_.png")]), 180.4,
                              bytes=17_300_000_000, seconds=175.2, mb_per_s=98.7, token=False)
    music, music_run = _fetched(_smoke_cell(MUSIC, [_flac()], output_kind="audio"), 60.0,
                                bytes=2_000_000_000, seconds=24.8, mb_per_s=80.6)
    out = _night_report([mesh, music], runs={MESH: mesh_run, MUSIC: music_run})
    assert ("| Model downloads | 19.3 GB downloaded by 2 jobs in 200 s of downloading, 96 MB/s "
            f"on average. ⚠️ No Hugging Face token was there for `{MESH}`: public files still "
            "download, on a limit shared with every user of the worker's address, and a gated "
            "file is refused. Check `HF_TOKEN` on the lane's endpoint |") in out.split("\n")
    # Numbers that are no numbers are left out, and the page is still there.
    odd, odd_run = _fetched(_smoke_cell(MESH, [_png(f"{MESH}_00001_.png")]), "fast",
                            bytes="many", seconds=None, mb_per_s="quick", reused_bytes=[1])
    out = _night_report([odd], runs={MESH: odd_run})
    row = next(line for line in out.split("\n") if line.startswith(f"| `{MESH}` "))
    assert " | 12.4 |  |  | 21.3 | " in row
    assert "| Model downloads | 0.00 GB downloaded by 1 job |" in out.split("\n")


def _unfetched(status, error, **fetch):
    """The page of a nightly run with one workflow whose models did not arrive."""
    cell = {**_smoke_cell(MESH, verdict="infra_error", worker_status=status),
            "timings": {"checkout_s": 1.9, "pip_s": 0.0, "model_fetch_s": 3.1}}
    run = {**_run(MESH, status, error=error), "timings": cell["timings"],
           "model_fetch": {"files": 0, "bytes": 0, "seconds": 0.0, "mb_per_s": None, "streams": 4,
                           "token": True, "reused": 0, "reused_bytes": 0, "evicted": 0, **fetch}}
    return _block(_night_report([cell], runs={MESH: run}).replace("#### ⚠️", "#### ❌"), MESH)


def test_a_refused_model_says_who_has_to_do_what():
    refused = {"model": "checkpoints/ltx.safetensors", "error": "HTTP 403 Forbidden",
               "http_status": 403, "host": "huggingface.co", "repo": "Lightricks/LTX-2.5"}
    block = _unfetched("model_access_error", {
        "message": "the model host answered 403 for Lightricks/LTX-2.5", "host": "huggingface.co",
        "repo": "Lightricks/LTX-2.5", "http_status": 403, "failed": [refused]})
    assert block[1].startswith(
        "An infrastructure problem stopped this workflow. Worker status: "
        "<code>model\\_access\\_error</code>. This is not a verdict on the commit. The model "
        "host refuses a file it needs. Another attempt would have ended the same way, so the "
        "run did not try it again.")
    assert "- **Message:** the model host answered 403 for Lightricks/LTX-2.5" in block
    assert ("- **What to do:** accept the licence of <code>Lightricks/LTX-2.5</code> on "
            "<code>huggingface.co</code> with the account behind `HF_TOKEN`, or correct the URL "
            "in the manifest.") in block
    assert ("- **Not downloaded:** <code>checkpoints/ltx.safetensors</code> (HTTP 403 Forbidden)"
            ) in block
    # The other two answers of a host that says no.
    for status, said in ((401, "<code>huggingface.co</code> wants a token it accepts for "
                               "<code>org/gated</code>: check `HF_TOKEN` on the lane's endpoint, "
                               "and that its account may read the repository."),
                         (404, "the file is not at <code>org/gated</code> on "
                               "<code>huggingface.co</code>: correct the URL in the manifest.")):
        block = _unfetched("model_access_error", {
            "message": f"the model host answered {status} for org/gated",
            "host": "huggingface.co", "repo": "org/gated", "http_status": status})
        assert f"- **What to do:** {said}" in block, status
    # What the worker sends is not trusted: it is escaped, or left out.
    block = _unfetched("model_access_error", {
        "message": "no", "host": "<b>evil</b>", "repo": None, "http_status": 403,
        "failed": ["junk", {"model": "a|b", "error": None}]})
    assert ("- **What to do:** accept the licence of the file's repository on "
            "<code>&lt;b&gt;evil&lt;/b&gt;</code> with the account behind `HF_TOKEN`, or correct "
            "the URL in the manifest.") in block
    assert "- **Not downloaded:** <code>a&#124;b</code> (no reason recorded)" in block
    for junk in ("403", 403.0, True, None, 500, [403]):
        block = _unfetched("model_access_error", {"message": "no", "http_status": junk})
        assert not [line for line in block if "What to do" in line], junk


def test_a_download_that_failed_names_the_files():
    lost = [{"model": "vae/ae.safetensors", "error": "sha256 mismatch: got abc"}]
    block = _unfetched("model_fetch_error", {
        "message": "model download failed: vae/ae.safetensors: sha256 mismatch: got abc",
        "failed": lost})
    assert "<code>model\\_fetch\\_error</code>" in block[1]
    assert ("- **Message:** model download failed: vae/ae.safetensors: sha256 mismatch: got abc"
            ) in block
    assert ("- **Not downloaded:** <code>vae/ae.safetensors</code> (sha256 mismatch: got abc)"
            ) in block
    assert not [line for line in block if "What to do" in line]
    # A full disk names no file: the message says it all.
    block = _unfetched("model_fetch_error", {
        "message": "model download failed: disk full: 12.3 GB free, 65.0 GB needed",
        "failed": [{"error": "disk full: 12.3 GB free, 65.0 GB needed"}]})
    assert not [line for line in block if "Not downloaded" in line]
    many = [{"model": f"vae/m{i}", "error": "HTTP 503"} for i in range(12)]
    block = _unfetched("model_fetch_error", {"message": "down", "failed": many})
    assert [line for line in block if line.endswith("<code>vae/m9</code> (HTTP 503), +2 more")]
