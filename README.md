# comfyci-runpod-worker

GPU regression testing for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on
[RunPod serverless](https://docs.runpod.io/serverless/overview), feeding
[ci.comfy.org](https://ci.comfy.org/).

For every new commit on ComfyUI master, the [GPU regression](.github/workflows/regression.yml)
workflow runs the curated workflows in
[`manifest/workflows.json`](manifest/workflows.json) on a RunPod serverless
endpoint **at that exact commit** (the worker checks the commit out at request
time), then compares the fixed-seed outputs against two baselines:

- **golden** — blessed reference outputs generated from a known-good release tag
- **previous** — the last completed run on the same branch

Metrics (per frame): MSE, PSNR mean/min, max abs channel diff, % pixels
changed, plus diff heatmaps and side-by-side strips. Everything is published
under `regression/` where the dashboard reads it.

## Results storage

Two interchangeable backends behind one path layout (`scripts/storage.py`,
pick with `--storage` / `REGRESSION_STORAGE`):

- **github** (current default): results live on this repo's orphan `results`
  branch; the dashboard reads them from
  `https://raw.githubusercontent.com/Comfy-Org/comfyci-runpod-worker/results/regression/`.
  Zero extra secrets — the scheduled workflow pushes with its own
  `GITHUB_TOKEN`. Fine at current volume; migrate before the run rate or the
  workflow set grows much (git history only accumulates).
- **gcs**: `gs://comfy-ci-results/regression/` — the long-term home once a
  `GCS_SERVICE_ACCOUNT_JSON` with write access exists. Switching is an env
  change here plus `NEXT_PUBLIC_REGRESSION_BASE` on the dashboard.

## Layout

| Path | What |
|---|---|
| `Dockerfile`, `docker/` | Worker image: CUDA + torch + pre-cloned ComfyUI. Commit under test is checked out per request. |
| `src/` | RunPod handler (`handler.py`), commit checkout, workflow runner, volume model sync |
| `manifest/workflows.json` | Single source of truth: workflows, models, seeds, thresholds (staged workflow JSONs not yet in the manifest live in `manifest/workflows/`) |
| `scripts/` | Orchestration run on the (CPU) CI runner: submit/poll, compare, publish, golden generation |
| `.github/workflows/regression.yml` | Scheduled poller that tests each new master commit (interim, until the core CI job below) |
| `action.yml` | Composite action for ComfyUI's `test-ci.yml` (GCS phase) |

## Adding a workflow

One PR to this repo:

1. Export the workflow from ComfyUI via **Workflow → Export (API)** and save it
   as `manifest/workflows/<id>_api.json`. Make sure it ends in a `SaveImage`
   node (PNG output — lossless is what gets compared). Fix all seeds in the
   JSON.
2. Add an entry under `workflows` in `manifest/workflows.json`:
   - `workflow_path`, `seed_overrides` (node id → input patch, belt-and-braces
     for the seed), `models` (name + download url + ComfyUI models
     subdirectory), and optionally `timeout_s`, `gpu_type`, `thresholds`.
   - Start with `"enabled": false`.
3. Merge. `sync-models.yml` downloads the new models onto the network volume
   automatically.
4. Generate + bless a golden for it: run the **Golden baselines** workflow with
   the current release tag and `workflows: <id>`, inspect, bless.
5. Flip `"enabled": true` in a follow-up PR.

## Verdicts

| Verdict | Meaning | CI effect |
|---|---|---|
| `pass` | metrics within thresholds vs golden | green |
| `fail` | metrics exceed thresholds vs golden | **red** |
| `execution_error` | workflow errored on the commit under test | **red** |
| `no_baseline` | no blessed golden yet | warning, green |
| `infra_error` | RunPod/queue/checkout/model-volume problem (after 1 retry) | warning, green |

Thresholds come from the manifest entry if set, otherwise from the manifest
defaults loosened to 3× the golden's own re-run noise floor
(`noise_floor.json`), so nondeterministic workflows don't false-alarm.

## One-time infrastructure setup

1. **Docker image**: `build-image.yml` pushes
   `ghcr.io/comfy-org/comfyci-runpod-worker` using the built-in
   `GITHUB_TOKEN` (no registry secrets). After the first build, make the
   package public (repo → Packages → package settings → Change visibility)
   so RunPod can pull it without credentials.
2. **RunPod**: create a Network Volume (≥250 GB, datacenter with 4090
   availability), then a serverless endpoint from the image with the volume
   attached, GPU type `RTX 4090`, max workers 3, idle timeout ~60s. Optionally
   set `HF_TOKEN` as an endpoint env var for gated models.
3. **Repo secrets** (this repo): `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, and
   optionally `HF_TOKEN` (gated models). For the GCS phase, additionally
   `GCS_SERVICE_ACCOUNT_JSON` (write access to the CI bucket) and `GCS_BUCKET`
   (`comfy-ci-results`).
4. **Seed the volume**: run `sync-models.yml` (workflow_dispatch).
5. **First goldens**: run **Golden baselines** with the latest release tag and
   `bless: true`.
6. Done — `regression.yml` now tests each new master commit on its own.
7. **(GCS phase) Wire into core CI**: add the job below to ComfyUI's
   `.github/workflows/test-ci.yml` plus the `RUNPOD_API_KEY` /
   `RUNPOD_ENDPOINT_ID` secrets there, and retire the scheduled poller:

```yaml
  gpu-regression:
    runs-on: ubuntu-latest
    steps:
      - name: RunPod regression suite
        uses: comfy-org/comfyci-runpod-worker@main
        with:
          comfy-commit: ${{ github.sha }}
          branch: ${{ github.ref_name }}
          gcs-bucket: comfy-ci-results
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          RUNPOD_ENDPOINT_ID: ${{ secrets.RUNPOD_ENDPOINT_ID }}
          GCS_SERVICE_ACCOUNT_JSON: ${{ secrets.GCS_SERVICE_ACCOUNT_JSON }}
```

## Results layout

Identical on both backends (branch root or bucket root):

```
regression/
  runs/<branch>/<commit>/<workflow_id>/{outputs/, run.json, comparison.json, figures/}
  runs/<branch>/<commit>/summary.json      # per-commit rollup (dashboard entrypoint)
  latest/<branch>.json                     # previous-run pointer, written last
  manifest-snapshot/<commit>.json
  golden/<workflow_id>/<tag>/{outputs/, run_r1.json, run_r2.json, noise_floor.json, blessed.json}
  golden/<workflow_id>/current.json        # active blessed tag
```

## Notes

- Goldens are valid **per GPU type and per worker image torch/CUDA**: after
  changing either, regenerate and re-bless (`blessed.json` records both).
- Local dry run without publishing anything:
  `python scripts/run_regression.py --commit <sha> --branch test --skip-publish`
