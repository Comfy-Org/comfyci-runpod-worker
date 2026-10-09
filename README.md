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
| `manifest/lanes.json` | Environment lanes (python / torch / cuda / GPU, cadence, storage prefix) published to the dashboard through the run index |
| `.github/workflows/build-index.yml` | Backfills or repairs the run index for a branch/lane (workflow_dispatch) |
| `scripts/` | Orchestration run on the (CPU) CI runner: submit/poll, compare, publish, golden generation |
| `.github/workflows/regression.yml` | Scheduled poller that tests each new master commit (interim, until the core CI job below) |
| `action.yml` | Composite action for ComfyUI's `test-ci.yml` (GCS phase) |

## Adding a workflow

One PR to this repo:

1. Export the workflow from ComfyUI via **Workflow → Export (API)** and save it
   as `manifest/workflows/<id>_api.json`. Make sure it ends in a `SaveImage`
   node (PNG output — lossless is what gets compared) with
   `filename_prefix: <id>`. Fix all seeds in the JSON. For template
   workflows, resolve switch nodes to their default branch and drop the
   unused branch and preview-only nodes, so only the models that actually run
   need syncing.
2. Add an entry under `workflows` in `manifest/workflows.json`:
   - `workflow_path`, `seed_overrides` (node id → input patch, belt-and-braces
     for the seed), `models` (name + download url + ComfyUI models
     subdirectory, plus `sha256` when the host publishes one), and optionally
     `timeout_s`, `gpu_type`, `thresholds`, `comfy_flags`.
   - Input images for `LoadImage` nodes are listed under `models` too, with
     `"directory": "input"` (pin the url to a commit, not a branch), and the
     entry sets `"comfy_flags": ["--input-directory", "/runpod-volume/models/input"]`
     so ComfyUI reads them from the volume.
   - Start with `"enabled": false`.
   - `python -m pytest scripts/tests -q` lints the manifest against every
     workflow it references (loaded files and `models` match both ways, every
     node feeds the `SaveImage`, seed override targets, model folders,
     `SaveImage` prefix = workflow id).
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

A failure vs golden is also classified against the previous tested run
(`detail` in `summary.json`): `new_drift` when this commit's outputs differ
from the previous run (the change happened here), `inherited` when they are
bit-identical to a previous run that already failed against the same golden,
and null when there is nothing to compare against or the outputs match a
previous run that did not fail (the golden changed, not the code). Inherited
failures publish no figures (`figures: false`): the figures of the first
failing commit already show the diff. `summary.json` carries
`schema_version: 2` and `has_infra_error` (infra errors never turn `overall`
red). Each workflow also records `prior_verdict` / `prior_commit`: its most
recent red or green verdict before this commit, read from the run index, which
(unlike the previous-run pointer) includes runs that had execution errors.

## Accepting an intentional change (re-bless)

When a commit changes numerics on purpose (a kernel optimisation that moves
every pixel by a rounding error, say), the golden is stale, not the code.
Re-bless with a reason so the history explains itself:

- From a release tag on master, with a fresh two-run noise floor: run
  **Golden baselines** with `ref=<tag>`, `workflows=<id>`, `bless=false`,
  inspect `golden/<id>/<tag>/outputs/` and `noise_floor.json`, then re-run with
  `bless_only=true` and a `reason`.
- Without GPU time, from a run that already exists on the results branch:
  `from_run=master/<full sha>`, `workflows=<id>`, `reason=...`, `bless=true`.
  The golden is named after the commit (12-char sha; pass `ref=<tag>` to name
  it after a tag that resolves to that same commit), has no `run_r2.json`, and
  its noise floor is inherited from the previous golden and marked
  `measured: false`.

An existing `golden/<id>/<tag>/` is never overwritten unless `force` is set.
`golden/<id>/current.json` records `reason`, `supersedes` and the golden's
`output_sha256`; `golden/<id>/history.json` keeps every bless; the run index
is refreshed in the same publish so accepted drift stops being reported as an
open regression.

## Run index

Every publish maintains a compact index under `regression/index/`: a per-lane
head file with the newest 500 runs and the first bad commit of each open
failing chain, complete monthly shards, and `lanes.json` with the lane
registry from `manifest/lanes.json`, per-branch latest pointers and the
current golden tag and output sha per workflow. Entries carry the commit
subject, author and PR (looked up with the Actions token; the PR title, author
and avatar too when the commit landed through a PR), the tested range
to the previous run (HEAD polling can skip commits), per-workflow verdicts
with the drift class, output sha, metrics and timings. Entries are ordered by
commit time, so re-testing an older commit never reorders history. The
dashboard's history and matrix views read only these files.

**Build run index** (workflow_dispatch) backfills or repairs a branch/lane
from the committed summaries; dispatch it once after the first deploy and
whenever an index file needs regenerating.

## PR comments (PULSE)

After each run, `scripts/notify_pr.py` can leave one comment on the ComfyUI
pull request that landed the tested commit. Its first line starts with
**(PULSE)**; below it, a row per workflow (verdict, new or inherited, PSNR,
mean MSE, % pixels changed, exec time, peak VRAM), the lane and the run's page
on ci.comfy.org.

- **When**: a workflow's outputs changed at this commit (`detail: new_drift`),
  or it hit an execution error while its prior run (`prior_verdict`) produced
  outputs. Never for pass, inherited drift, infra errors, missing goldens, a
  workflow with no prior run, a re-test of a commit older than the previous
  run, or a run that covered several new commits (the change cannot be pinned
  on one PR).
- **One comment per PR**: found again by the hidden marker
  `<!-- pulse-regression:v1 -->` and the token owner's login, and edited in
  place, so reruns never add a second one. A later clean run of the same
  commit turns it into a short resolved note.
- **Whose PRs**: only PRs authored by a ComfyUI code owner, i.e. the users on
  the `*` rule of ComfyUI's `CODEOWNERS` (team entries are ignored). The repo
  variable `PULSE_ALLOWED_AUTHORS` (comma-separated logins) replaces that list.
  If `CODEOWNERS` cannot be read and no override is set, nothing is posted.
  Only the primary lane on master comments.
- **Modes** (repo variable `PULSE_PR_COMMENTS`): `dry-run` (default) renders
  the comment into the job summary and posts nothing; `on` posts; `off` skips
  the step's work entirely. `on` without the token falls back to dry-run.
- **Token** (repo secret `PULSE_GH_TOKEN`): a personal access token of the
  account the comments should come from; they appear as that user. Prefer a
  fine-grained token with resource owner Comfy-Org, repository
  `Comfy-Org/ComfyUI` only, and Issues and Pull requests set to read and
  write. Fine-grained tokens for an organisation's repositories may need
  approval by an org owner, depending on the org's token policy. A classic
  token with the `public_repo` scope also works, but it can push to every
  public repository the account can write to, and anyone who can edit this
  repo's workflows can use it. Reads (PR lookup, `CODEOWNERS`) use the Actions
  token.
- **Non-blocking**: the step runs even when the suite fails, has
  `continue-on-error`, and turns every API or network problem into a warning.

To turn it on: add `PULSE_GH_TOKEN` (Settings → Secrets and variables →
Actions), check the dry-run output in a few job summaries, then set the
variable `PULSE_PR_COMMENTS` to `on`. Setting it to `off` stops it at once.

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
   optionally `HF_TOKEN` (gated models) and `PULSE_GH_TOKEN` (PR comments,
   see above). For the GCS phase, additionally
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
                                           # figures/ only when the frames differ and the
                                           # failure is not inherited
  runs/<branch>/<commit>/summary.json      # per-commit rollup (dashboard entrypoint)
  latest/<branch>.json                     # previous-run pointer, written last
  manifest-snapshot/<commit>.json
  golden/<workflow_id>/<tag>/{outputs/, run_r1.json, run_r2.json, noise_floor.json, blessed.json}
  golden/<workflow_id>/current.json        # active blessed tag (+ reason, supersedes, output_sha256)
  golden/<workflow_id>/history.json        # every bless
  index/lanes.json                         # lane registry + per-branch latest + golden shas
  index/<branch>/<lane>.json               # newest 500 runs + first bad commit per workflow
  index/<branch>/<lane>/<YYYY-MM>.json     # complete monthly shards
```

Run outputs also carry 256-px WebP previews under `<workflow_id>/outputs/thumbs/`
for the dashboard's list views.

## Notes

- Goldens are valid **per GPU type and per worker image torch/CUDA**: after
  changing either, regenerate and re-bless (`blessed.json` records both).
- Local dry run without publishing anything:
  `python scripts/run_regression.py --commit <sha> --branch test --skip-publish`
- Script tests (run on every PR by the *Script tests* workflow):
  `pip install -r scripts/requirements.txt pytest && python -m pytest scripts/tests`
- The results checkout is a blobless sparse clone: only the aggregate
  directories are materialised, run directories on first use.
- `run.json` `delay_s` is in seconds from summary `schema_version: 2` on;
  earlier records hold RunPod's raw millisecond value under the same key.
