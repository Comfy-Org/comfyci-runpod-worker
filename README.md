# comfyci-runpod-worker

GPU regression testing for [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on
[RunPod serverless](https://docs.runpod.io/serverless/overview). Every result
is shown in the Actions run that produced it (see
[What a run shows](#what-a-run-shows)); the same data feeds
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
under `regression/`. The run's own job summaries are rendered from there, and
the dashboard reads the same files.

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
| `manifest/custom_nodes.json` | Custom-node packs (repo, pinned commit) the pack workflows install on the worker, baked into the images CI builds (see [Custom-node packs](#custom-node-packs)) |
| `.github/workflows/build-index.yml` | Backfills or repairs the run index for a branch/lane (workflow_dispatch) |
| `scripts/` | Orchestration run on the (CPU) CI runner: submit/poll, compare, publish, golden generation |
| `scripts/template_api_export.py`, `scripts/import_template.py` | Turn a workflow template into a corpus entry (see [Adding a workflow](#adding-a-workflow)) |
| `.github/workflows/regression.yml` | Scheduled poller that tests each new master commit on every per-commit lane and reports the results in the run: one job summary per lane, one for all lanes, and the run's files as an artifact (interim, until the core CI job below) |
| `.github/workflows/nightly-weekly.yml` | Tests master HEAD once a night and once a week on the lanes of those cadences (see [Nightly and weekly lanes](#nightly-and-weekly-lanes)) |
| `scripts/run_report.py` | Renders those job summaries from the results branch (see [What a run shows](#what-a-run-shows)) |
| `action.yml` | Composite action for ComfyUI's `test-ci.yml` (GCS phase) |

## Adding a workflow

Templates are the corpus's main source: a template that stops producing its
image is a release blocker, and the comparison here is what notices. One PR to
this repo per template.

1. Dump `/object_info` from the ComfyUI version the corpus runs against, with
   the template's models present so their combo values resolve, and import the
   template from a [workflow_templates](https://github.com/Comfy-Org/workflow_templates)
   checkout:

   ```
   curl -s http://127.0.0.1:8188/object_info > object_info.json
   python scripts/import_template.py image_pixeldit_t2i \
       --templates-dir ../workflow_templates/templates \
       --object-info object_info.json --id pixeldit_t2i
   ```

   That writes `manifest/workflows/<id>_api.json` and prints the manifest entry.
   It converts the template the way the frontend's Export (API) does, picks the
   branch a constant switch selects (so the other branch's models never have to
   be synced), drops preview nodes, normalises the output to a `SaveImage` PNG
   prefixed with the workflow id, and fills `models` from the download links in
   the template's own model-links note, with the sha256 ComfyUI publishes.

   It refuses rather than guess. A template whose widget layout lives on the
   node instance, or one that needs custom nodes, has to be exported by hand
   from ComfyUI (**Workflow → Export (API)**) instead; `scripts/tests/test_template_api_export.py`
   documents which conversion rules are covered. A loader whose models/
   subdirectory it does not know is also a refusal, not a guess: add the
   `(class_type, input name)` pair to `DIRECTORY_BY_LOADER_INPUT`. A template
   whose result is text is refused too: see "A text workflow" below.

2. Paste the entry under `workflows` in `manifest/workflows.json` and choose a
   `timeout_s` the model can finish in. Leave `"enabled": false`.

   `output_kind` says what the workflow's main output is, and so how it is
   compared. An entry without the key is an image workflow: its `SaveImage`
   PNGs are compared pixel by pixel. `"output_kind": "text"` makes it a text
   workflow: its `.txt` file is compared with the reference text.
   `"output_kind": "audio"` is a sound workflow: a FLAC file written by
   `SaveAudioAdvanced` with `format` `"flac"`. Nothing compares audio yet, so
   such an entry must be at smoke level (next paragraph). An unknown value
   stops a run before any GPU job starts, and so does an audio entry that is
   not at smoke level. A 3D or a video workflow is an image workflow: it
   saves a picture under the workflow id, and its mesh (`SaveGLB`) or clip
   (`SaveVideo`) is a second file, prefixed `<id>_`.

   `level` says how the workflow is checked. An entry without the key is at
   **reference level (the main tests)**: its output is compared with an
   approved reference, the golden. `"level": "smoke"` puts it at **smoke
   level**: it only has to finish and to write usable files (see
   [Smoke level](#smoke-level)). A smoke-level entry has no golden, so its
   `thresholds` and `golden_tag` are null and step 5 falls away. Use it where
   an output cannot be the same on every run, or where nothing can compare
   it yet. It does not catch a wrong but plausible output.

   A smoke-level entry can say what its author knows about the output in
   `expect`. Every key is optional: `width` and `height` (the PNG files),
   `frames` (the video), `seconds` (the audio, within 5 % and never tighter
   than 0.5 s) and `min_chars` (the text). For example
   `"expect": {"width": 1024, "height": 1024, "seconds": 30}`. An unknown
   key, a value that is not a number above 0, or `expect` on a
   reference-level entry stops a run before any GPU job starts.

   `cadence` says how often the workflow runs: on every commit (no key, or
   `"per-commit"`), `"nightly"` or `"weekly"`. It runs on the lanes of that
   cadence and on no other (see [Lanes](#lanes)).

   For a hand-exported workflow, the entry needs the same shape: `workflow_path`,
   `seed_overrides` (node id → input patch, belt-and-braces for the seed),
   `models` (name + download url + ComfyUI models subdirectory, plus `sha256`
   when the host publishes one), and optionally `gpu_type`, `thresholds`,
   `comfy_flags`. Input images for `LoadImage` nodes (and clips for
   `VHS_LoadVideo`) are listed under `models` too, with `"directory": "input"`
   (pin the url to a commit, not a branch), and the entry sets
   `"comfy_flags": ["--input-directory", "/runpod-volume/models/input"]` so
   ComfyUI reads them from the volume. A model that lives in a subfolder of a
   models directory keeps the subfolder in its `name`, the way the loader's
   combo lists it (`"name": "bbox/face_yolov8m.pt", "directory": "ultralytics"`).

   A workflow that uses custom nodes lists its packs under `custom_nodes`
   (names from `manifest/custom_nodes.json`) and restricts itself with `lanes`
   to the lanes whose image bakes them; see [Custom-node packs](#custom-node-packs).

   Every `models` entry also carries `bytes`, the size of the download. The
   importer writes it next to the sha256. For an entry written by hand,
   `python scripts/import_template.py --fill-bytes` asks each URL for its size
   and adds the missing ones to the manifest, changing nothing else. The lint
   adds the sizes up, each file once, and fails when they pass 90 % of the
   network volume (`endpoint_defaults.volume_gb` in `manifest/lanes.json`).
   Nothing deletes from the volume, so a full one fails every sync after it.
   The lint counts, and the model sync downloads, the models of the same
   workflows: those that run on a lane which reads its models from the volume
   (`lanes.volume_workflows`). A workflow that runs only on lanes with
   `"model_storage": "job"`, or on lanes without a volume, is in neither.
   Nothing fetches models for a job yet, so the lint also fails for an
   enabled workflow of that kind that loads a file which is not among the
   volume's: "its models are on no volume". One whose files another workflow
   puts on the volume passes. The rule goes when jobs can fetch models.

   **A text workflow** (a model that answers in text, like the
   `llm_qwen38_27b_t2t` entry) is written by hand: the importer only writes
   image workflows and stops on a text template with what to do. Convert the
   template with `scripts/template_api_export.py`, then make sure of this:

   - The entry says `"output_kind": "text"`.
   - The graph ends in one `SaveText` node with `filename_prefix` equal to the
     workflow id and `format` `"txt"`. ComfyUI then writes `<id>_00001.txt`,
     the file that is compared. A template usually ends in `PreviewAny`, which
     writes no file: replace it. Any second output node is prefixed `<id>_`,
     and a second `SaveText` is compared too, by file name.
   - Text generation is greedy: every `TextGenerate` node has
     `sampling_mode` `"off"` and an `mtp` that is written out and is not
     `"auto"` (use `"off"`). The text has to be the same on every run, because
     it passes only on an exact match. `mtp` `"auto"` follows measured
     timings, and the worker removes the `sampling_mode.<name>` values that
     sampling needs.
   - `seed_overrides` is `{}` (greedy decoding has no seed) and `thresholds`
     is `null`. Only when an exact match is known not to hold, set
     `"thresholds": {"min_text_similarity": 0.98}`, a number above 0 and at
     most 1 (see [Verdicts](#verdicts)).
   - Input images are listed under `models` with `"directory": "input"`, as
     for any workflow.

3. Check it. `python -m pytest scripts/tests -q` lints the manifest against
   every workflow it references (loaded files and `models` match both ways,
   every node feeds an output node, seed override targets, model folders,
   `SaveImage` prefix = workflow id; for a text workflow the `SaveText` rules
   and greedy generation above). Then confirm a real server accepts the
   prompt, which is what catches a conversion that produced a valid-looking
   file with the wrong input names:

   ```
   curl -sS -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' \
       -d "{\"prompt\": $(cat manifest/workflows/pixeldit_t2i_api.json)}"
   ```

   A 200 means every node, input and link survived; the run can then fail on
   missing weights without telling you anything new.

4. Merge. `sync-models.yml` downloads the new models onto the network volume
   automatically, through a one-shot job on the primary lane's endpoint (every
   lane shares the volume, so one sync serves them all).
5. Generate + bless a golden for it: run the **Golden baselines** workflow with
   the current release tag and `workflows: <id>`, inspect, bless. For a text
   workflow, read `golden/<id>/<tag>/outputs/<id>_00001.txt` before you bless
   it: a useless answer would pass from then on. An empty one is refused, and
   so is a text golden whose two runs differ (see [Verdicts](#verdicts)).
6. Flip `"enabled": true` in a follow-up PR.

## Verdicts

| Verdict | Meaning | CI effect |
|---|---|---|
| `pass` | metrics within thresholds vs golden | green |
| `fail` | metrics exceed thresholds vs golden | **red** |
| `execution_error` | workflow errored on the commit under test, or a custom-node pack it requested did not import (worker status `custom_node_error`) | **red** |
| `no_baseline` | no blessed golden yet | warning, green |
| `infra_error` | RunPod/queue/checkout/model-volume/pack-install problem (after 1 retry) | warning, green |

That is what the verdicts mean at reference level (the main tests), where an
output is compared with its golden. A workflow at smoke level uses the same
five names, and its `pass` and `fail` mean something else: see
[Smoke level](#smoke-level).

On a lane with the `nightly` or `weekly` cadence an `infra_error` that is
still there at the end of the run is **red**: nothing tests the commit again
the next minute there (see
[How a nightly or weekly run goes](#how-a-nightly-or-weekly-run-goes)).

Thresholds come from the manifest entry if set, otherwise from the manifest
defaults loosened to 3× the golden's own re-run noise floor
(`noise_floor.json`), so nondeterministic workflows don't false-alarm.

That is the rule for an image workflow. A text workflow (`"output_kind":
"text"` in its entry) has another one:

- **Pass rule.** The generated text must equal the golden's text. Line ends,
  blanks at the end of a line and empty lines at the end of the file do not
  count. Everything else does: a changed word, a second blank inside a line, a
  letter's case. The image thresholds never apply, and `thresholds_used` is
  null.
- **A similarity limit, only on request.** An entry can set
  `"thresholds": {"min_text_similarity": 0.98}`. A text that differs then
  still passes when its similarity to the golden is at least that. Similarity
  is the share of words the two texts have in common, in order (Python's
  `difflib` ratio; over lines above 5,000 words, and above 5,000 lines the
  share of the longer text that both start with). The same steps down are
  taken when a match would take too long, which happens when two long texts
  share little more than one frequent word: the work is counted in steps, not
  seconds, so the result is the same on every machine, and
  `similarity_basis` names the basis that was used. A similarity by lines or
  by the shared start is never above the share of words the two texts have in
  common in any order, so a step down cannot pass a limit that the words
  fail. Wherever such a number is shown, "by lines" or "by the shared start"
  follows it. The noise floor never loosens the rule by itself: one changed
  word moves the rest of a generated text, so "3× the noise" means nothing
  for text.
- **What cannot be compared fails.** No `.txt` file, a file over 1 MiB, a file
  the worker listed and did not send, another number of text files than the
  golden has, or a text file under another name: each is a `fail`, never a
  pass, with or without a similarity limit.
- **Bless gate.** Golden generation runs a text workflow twice, like every
  workflow. A golden whose two runs differ is refused: `bless` and
  `bless_only` print `::error::[<id>] golden <tag> is not blessed: ...`, leave
  `current.json` alone and the job fails. The golden's files are uploaded all
  the same: the first run's text, and in `noise_floor.json` the line and
  column where the second run parts from it. The one exception is a golden
  generated or promoted again under the tag that is current (`force`): a
  refused one leaves the files of the current reference as they are, with or
  without `bless`. An exact-match golden that does not match its own second
  run would fail at random, and with PULSE on it would comment on pull
  requests. Make the workflow deterministic, or set `min_text_similarity` in
  its entry: the gate then opens only when the second run passes against the
  first under that limit. A golden with an empty text file, one whose two
  runs could not be compared, and one whose noise floor is an image
  workflow's are refused too. A golden promoted from a run (`from_run`) is
  held to the same: an empty text is not blessed, and a run without a text
  file, or with one that the worker listed and did not send, is not promoted.

A text change that is new at a commit adds one line under the table of a
PULSE comment: the similarity and the line and column of the first
difference, or that the text file has another name than the golden's. The
text itself never enters a comment.

Each commit is tested once per lane. The cron runs every 5 minutes, the shortest
interval GitHub schedules (GitHub can start a run late, or drop one, when it is
busy). A test run takes about five minutes, so a result should follow a commit
by about ten when GitHub keeps to the schedule.
It skips master
HEAD on a lane as soon as that lane has published a `summary.json` for it,
whatever the verdict: a red commit stays red until master moves, it is not
re-run every tick. Before it submits any GPU job, every run pushes a start
marker, `runs/master/<sha>/attempt.json` (attempt number, Actions run URL), so
a run that dies before publishing its summary (a failed results push, a killed
job) is still counted. A summary with infra errors, or a marker with no
summary, gets one more attempt: two in all per commit per lane, counted in the
summary's and the marker's `attempt`. Two cases follow from that rule. A
workflow that another run right away would not test either is no reason for
the retry: its models are not on the volume yet (`missing_model`), the model
host refuses a file (`model_access_error`), or the run's deadline came before
it could start (`not_reached`). A summary whose untested workflows are all of
that kind is left alone, and its report says so. And a summary with `partial`
(a nightly or weekly run pushes one every 10 workflows) belongs to a run that
did not finish: it counts as a marker with no summary. A run that names its
workflows at such a commit does not merge into it: it writes a summary of its
own with `scope`, as at a commit without a result, and the cron's retry still
follows. A run that cannot push its marker stops before any GPU work.
To test a commit again on purpose, dispatch **GPU
regression** with `commit=<sha>` (and `lane=` for one lane), when no GPU
regression run is in progress, or check afterwards that the dispatch ran:
all runs share one concurrency group, and GitHub cancels a pending run as
soon as a newer one (the next cron tick) queues behind the running one. With a
tick every 5 minutes, a dispatch made while a test run is in progress will
usually be replaced that way, so dispatch between test runs. The
decision lives in `scripts/regression_skip.py`. The cron used to key on the
previous-run pointer, which does not advance on execution errors, so a red
HEAD was re-run on both lanes every half hour (53 runs for 11 commits during
the 2026-11-03/04 z_image timeouts).

**A run that names its workflows.** A dispatch of **GPU regression** with
`workflows=<ids>` tests only those workflows. It is a re-test or a first look,
not the lane's run of the commit, so it pushes no start marker and never moves
the previous-run pointer. What it publishes depends on what the lane has for
the commit:

- No result yet: the summary holds the named workflows and lists their ids
  under `scope`. The cron does not count such a summary as tested: its next
  tick tests the commit with every workflow, and that run replaces the
  summary. A second named run at the same commit adds its cells, and its ids
  to `scope`. What a nightly or weekly run that did not finish pushed on its
  way (`partial`) is no result either: the named run replaces it. In the run
  index the entry of such a summary carries `"named": true`. It has its row
  in the history, and it is not the lane's latest run: `latest`, in the
  index head and in `index/lanes.json`, stays at the newest entry without
  the mark, also when the named run is at a newer commit.
- A result: only the cells of the named workflows are replaced. Every other
  cell stays as it is, byte for byte, and so do `run_ts`, `run_url`, `run_id`,
  `attempt`, `env`, `commit_meta` and `tested_range`, which describe the
  lane's run. `overall` and `has_infra_error` are worked out again over every
  cell, and the re-test records itself under `retests` (`run_ts`, `run_url`,
  `run_id`, `env`, `workflows`). The history row keeps its link to the lane's
  run, and the PR comment step decides over every cell of the commit. The job
  itself is red or green by the workflows it tested.

A lane tests a commit once, so a dispatch for a commit that already has the
lane's result must name it (`commit=<sha>`). To try a workflow without
writing anything, turn `publish` off: the commit is then tested whatever the
lane has for it, nothing reaches the results branch, no pull request gets a
comment, and the result is on the run's page and in its attached files.

**Waiting for a worker.** A run submits all its jobs at once, and an endpoint
has 3 workers. A job may wait 600 seconds in RunPod's queue, and 60 seconds
more for each job submitted before it in the run (the order of the manifest).
A job that waits longer is cancelled and submitted once more. The extra wait
is added to the time the runner gives the job as a whole, its `timeout_s`
plus 900 seconds. That is the per-commit lane. A nightly or weekly lane has
only a few jobs on its endpoint at a time (see
[How a nightly or weekly run goes](#how-a-nightly-or-weekly-run-goes)).

A failure vs golden is also classified against the previous tested run
(`detail` in `summary.json`): `new_drift` when this commit's outputs differ
from the previous run (the change happened here), `inherited` when they are
bit-identical to a previous run that already failed against the same golden,
and null when there is nothing to compare against or the outputs match a
previous run that did not fail (the golden changed, not the code). Inherited
failures publish no figures (`figures: false`): the figures of the first
failing commit already show the diff. A text workflow never has figures: its
difference is shown from the two published texts. `summary.json` carries
`schema_version: 2` and `has_infra_error` (infra errors never turn `overall`
red), and `run_url` / `run_id`: the Actions run that tested the commit (null
outside Actions). Each workflow also records `prior_verdict` / `prior_commit`:
its most recent red or green verdict before this commit, read from the run
index, which (unlike the previous-run pointer) includes runs that had
execution errors.

### Smoke level

A workflow whose entry says `"level": "smoke"` is not compared with anything.
It passes when it finished, every save node wrote its file, and every file
passes the checks of its type. It has no golden and no bless step: **Golden
baselines** leaves it out of `workflows=all` and refuses it when it is named.

| Verdict | Meaning at smoke level | CI effect |
|---|---|---|
| `pass` | it finished, every expected file is there and passes its checks | green |
| `fail` | it finished, and a file is missing or does not pass. Also when it wrote no file at all (worker status `no_outputs`) | **red** |
| `execution_error` | the workflow did not finish, as at reference level | **red** |
| `infra_error` | not tested, as at reference level | warning, green (red on a nightly or weekly lane) |
| `no_baseline` | never: there is no reference to miss | |

The checks are in `scripts/smoke.py`. They run on the CI runner, on the files
the worker sent back. A file's type comes from its extension, and a file of
another extension is listed and not checked.

| Type | File | It passes when |
|---|---|---|
| image | `.png` | it decodes fully, each side has at least 64 px, it is not blank (the grey image has a standard deviation of at least 1.0) and not one flat colour with a speck (at most 99.5 % of the pixels share one grey level) |
| text | `.txt` | it is UTF-8 and at most 1 MiB, has at least 8 characters that are not blank and 4 different ones, and with 20 words or more no word is more than half of them (a word has a letter or a digit: the bars of a table do not count) |
| audio | `.flac` | it is a FLAC file that ends with a whole frame, 8,000 to 192,000 Hz, 1 to 8 channels, at least 1.0 s long, and not silent: at least 4,000 bytes of audio frames a second (FLAC stores silence in a few bytes) |
| mesh | `.glb` | it is a GLB whose first primitive has positions, with at least 100 vertices and 100 faces, every position finite, and a bounding box larger than 0.0001 in at least two directions |
| video | `.mp4` | it is an MP4 with a video track, at least 2 frames, each side at least 64 px, and not blank or still: a frame after the first has at least 100 bytes plus 500 for each megapixel, on average (the first frame is stored whole, and alone it carries a small black clip over that) |

The video limit is not settled. It rests on 267 made-up clips: flat colours,
one picture repeated, moving stripes, a blurred picture that pans or gets
brighter, with and without grain (`MEASURED_CLIPS` in
`scripts/tests/test_smoke.py` has every number). Each blank or still one has
at most 0.62 of what a frame needs, and each moving one at least 1.15. No
clip of a model is among them, and no manifest entry has a video at smoke
level today. Before the first video workflow gets the smoke level, measure
real clips of its model (`bytes_per_frame_mp` in `smoke.files`, times the
megapixels) and check the limit against them.

- **A file that cannot be read fails.** A cut or broken file is a problem of
  that file (check `decode`), never a crash of the run.
- **`expect`** in the entry adds what its author knows: `width` and `height`
  (PNG), `frames` (video), `seconds` (audio), `min_chars` (text).
- **Which files must be there** comes from the workflow's own save nodes:
  `SaveImage` a `.png`, `SaveText` a `.txt`, `SaveAudioAdvanced` a `.flac`,
  `SaveGLB` a 3D file and `SaveVideo` an `.mp4`, each named with the node's
  `filename_prefix` and the counter ComfyUI adds (`<prefix>_00001_.png`). A
  file of another node under a longer prefix (`<prefix>_gif_00001.png`) is
  not that node's file. A node that wrote nothing is a `missing` problem. The
  lint holds `SaveVideo` to `format` `mp4` or `auto` with the codec `h264` or
  `auto`: another container has another extension, and the limit above was
  measured on H.264.
- **A file the worker did not send** (it is over the worker's size limit)
  cannot be opened. It passes on its size alone from 1,024 bytes, and the
  report says "not sent, checked by size only". A text is held to its 1 MiB
  all the same: that needs no look at the file.
- **A file whose name leaves the outputs folder** (`../x.png`, an absolute
  path) fails, whatever its type: "could not be read: its name leaves the
  outputs folder" (check `decode`). The runner writes such a file nowhere,
  and it never passes on its size. Only a broken worker names one.
- **New, inherited or first run.** Nothing is compared, so a failed check is
  placed by the workflow's own last verdict (`prior_verdict`): new when it
  passed before, inherited when it already failed, and "first run" when it
  has no earlier result on the lane. A workflow that fails its first run is a
  problem of the workflow, and is not worded like a change in a commit.
- **PULSE** comments on a smoke failure only when the workflow passed its
  checks the time before. The row reads `smoke check failed`.

Smoke level does not catch a wrong but plausible output: half a picture, a
quiet hiss where music should be, a mesh of the wrong object. That is what
reference level is for. To move a workflow there, remove `"level"` from its
entry and approve a published run as its reference (`from_run`, see
[Accepting an intentional change](#accepting-an-intentional-change-re-bless)).
Until then its verdict is `no_baseline`. A run that holds an output file
which was not published (see [Results layout](#results-layout)) is refused
as a reference. Approve the second of two runs with the same output: all of
its files are published.

## What a run shows

Everything a run found is on the run's own page. Nobody needs ci.comfy.org to
read a result.

**Where to look.** Open the repo's **Actions** tab, pick **GPU regression**,
then the run. Each lane job is named after its lane and the commit it tested
(`py313-torch2.14.1-cu130 · ba546e92`), and one more job is called
`All lanes · <sha>`. Every one of them writes a job summary, shown on the run
page below the job graph. `scripts/run_report.py` renders the summaries from
the results branch, read at one pinned commit, and from the files the run left
on the runner. A report step gets ten minutes: one that hangs is stopped, so
it cannot hold up the commits that wait for the run to end.

**A lane's summary** covers one commit on one lane:

- a headline: whether an output broke, and whether that is new at this commit
  or inherited from an earlier one. A run that covers several commits says
  "new in this run": the change is in one of them;
- the commit, its parent and pull request, when it was tested, how many
  commits the run covers, the lane, the branch and the environment (GPU,
  driver, Python, torch, CUDA, ComfyUI version, worker image, baked packs).
  On a lane with the `nightly` or `weekly` cadence a run covers many commits:
  its title names the range (`e9afbcc..05e878f (7 commits)`), and the table
  says how much new output of smoke-level workflows the lane's run published;
- a warning under the title when the summary is one that a nightly or weekly
  run pushed on its way and no final one replaced: "The run that tested this
  commit did not finish: 40 of 55 workflows have a result". The table then
  has a **Progress** row. A nightly or weekly run that was stopped reports
  from the files on its runner: its title has the warning mark also when
  every workflow it has is green, and the workflows it left without a result
  are listed as such.
  A **Tested again** row lists the workflows that a later run tested again
  at the commit, and their rows in the table of workflows link that run;
- "What needs attention", one block per workflow that did not pass. For a
  changed output: each metric next to its limit, the comparison with the
  previous tested commit, where the reference (the golden) came from, the diff
  figures and the two commands that approve the new output. For an execution
  or infra error: the worker status, the error message, the node, the
  traceback and the tail of the server log. On a nightly or weekly lane a
  failure is placed by the workflow's own last result ("new since `e9afbcc`,
  7 commits", with the command that tests one commit of that range), a
  workflow that was not tested says whether the run tried it again (its cell
  then has `"tries": 2`) and how to test it now, and the workflows the run's
  deadline did not reach stand in one list. A changed output that was approved
  as the reference after the run is labelled "accepted", is no longer counted
  as open and sits under a heading of its own. The label needs both: the
  output is the lane's reference today, and that reference is another one than
  the run was compared with;
- a table of every workflow: result, an output thumbnail that links to the
  full image, metrics, timings, peak memory and the output sha. Each run time
  stands next to the workflow's usual range on the lane (what 8 in 10 of its
  last 30 passing runs took) and is marked slow above 1.5 times the top of it.
  Times differ between workers by up to a third and are never part of the
  result;
- for a workflow that passed without being bit-identical: its numbers next to
  the limits, the noise floor and the diff figures, folded;
- for a text workflow, text where an image workflow has images. Its row in
  the table links the `.txt` file and says "identical text", or the
  similarity and the line where the texts part. A text that changed has a
  block under "What needs attention": whether it is identical, the
  similarity, the line and column of the first difference, and the character
  and line counts, each against the reference, the previous tested commit and
  the noise floor. Below that come the difference (removed text as `[-…-]`,
  added text as `{+…+}`), then the generated text and the reference text,
  folded, each with a link to its file, and the two commands that approve the
  new text. Texts too long to mark word by word get a line diff, with `-`
  before a removed line and `+` before an added one, and the heading says
  which of the two it is. When the two texts differ only in characters that
  cannot be shown, or the reference has its text under another file name, a
  sentence says so where the difference would be. Under another file name
  nothing was compared, so no similarity is shown. A text that matches its
  reference is shown, folded, under "Generated text", also when it differs
  from the run before. A text is cut to its first 4,000 characters and the
  difference to 8,000, and lines over 100 columns are folded; the heading
  says both, because the line and column in the table count in the file. A
  file over 256 KB is linked, not shown. Generated text is model output: it
  appears only inside `<pre>` blocks, escaped, and is never printed to the
  job log;
- for a workflow at smoke level, what was checked where a reference-level
  workflow has what was compared. Its result reads "smoke pass" or "smoke
  check failed", and one sentence under the headline says what that is
  worth: such a workflow is only checked to finish and to write a usable
  file. Its row has a **Checked** cell with its files by type ("2 images
  1024x1024 · mesh 35,781 vertices", "30.0 s · 48000 Hz · stereo") and
  nothing in the three reference columns. A lane of smoke-level workflows
  alone has no reference columns, and a lane without one has no Checked
  column. The Output cell is a thumbnail when the run wrote a PNG, else a
  link named after the kind. The file list says what was measured on each
  file. A failed check has a block under "What needs attention": each
  problem with its file, the files with what was measured, and where the
  failure started: "new at this commit", "new since `<sha>`, N commits",
  "inherited" or "first run". No command approves anything there. A text
  that passed its checks is shown under "Generated text", marked "smoke
  level, not compared". One that failed is shown in its block. The headline
  counts smoke passes apart from matches with a reference, and the roll-up
  of a lane reads "smoke pass", or "pass (13 reference, 2 smoke)" when it
  holds both. An output file that a smoke run kept off the results branch
  (see [Results layout](#results-layout)) is not linked: the report points
  to the files attached to the run. The legend explains the two smoke
  labels. A page without a smoke-level result reads as it always did;
- every output file, a link to each `run.json`, schema notes, the status of
  custom-node packs, and the workflows that did not run on the lane;
- the lane's newest 20 commits with their results, each linked to the run that
  tested it, and the lane's open failures.

**The "All lanes" summary** shows the commit on every lane, the workflows the
lanes disagree on, each lane's open failures, and the newest 25 commits as a
table with one column per lane. An open failure is listed only for a workflow
the lane still runs: one that was renamed, moved or switched off while it was
red is left out. A nightly or weekly lane that has an endpoint gets no column:
it tests one commit in many. It has a row in the table "Nightly and weekly
lanes" instead, built from its last full run (the newest that did not name
its workflows): when it ran, with a link to the run, the
commits it covered, its result, what did not pass and how many workflows were
not tested. A last run older than 36 hours on a nightly lane, or 8 days on a
weekly one, is marked "no run for ...". GitHub can drop a scheduled run
without a word, and that mark is then the only sign. The open failures of
those lanes are listed with the others.

**A commit that was already tested.** A cron tick that finds the commit
tested on a lane tests nothing there. Its summary shows the published result
and links to the run that tested it.

**Any commit, again.** Dispatch **GPU regression** with `report_only: true`
(plus `commit=<sha>`, and `lane=` for one lane) to get the same summaries for
a commit without testing it. No GPU job starts and nothing is written. Such a
run neither waits for a test run nor cancels one. It shows every lane in
`manifest/lanes.json`, a retired lane and one without an endpoint too; a test
run only has jobs for the provisioned per-commit lanes. An abbreviated
`commit` (7 characters or more) is expanded to the full sha first, in a test
run as well.

**Attached files.** Each lane job that tested something attaches an artifact
named `gpu-regression-<lane>-<sha8>`: the run records, comparisons, outputs,
figures and `summary.json`. It is also what is left when the publish to the
results branch failed; the summary then says so, and reports from those files,
also when an earlier attempt already published a result for the commit.

**How long it lasts, and who can read it.** GitHub keeps a run, its summaries
and its artifacts for 90 days on a public repository, then deletes them. The
results branch keeps everything, and a report-only run brings the summary of
any commit back. Reading a run's summary or downloading its artifact needs a
GitHub login (any account). The images in a summary load from the results
branch, pinned to the commit the report read. A summary stays under GitHub's
1 MiB limit: when a run is too large, log tails, tracebacks, file lists and
history rows are shortened in that order, and the summary says which.
Generated texts go last, first cut to 1,000 characters, then left out with a
link to the file.

**What stays on ci.comfy.org.** The site reads the same files. A summary is a
page about one commit, so the views over many runs are only there: a lane's
pass rate and how fresh its last run is, the ComfyUI version of each run, each
workflow's last change and its MSE over time, the history of custom-node pack
imports, and anything older than the newest 20 runs. Each lane summary links
to its commit's page on the site.

To render a report on your own machine: `python scripts/run_report.py lane
--commit <sha> --branch master --lane <lane id> --results-dir <clone> --out
report.md` (`overview` for all lanes). `<clone>` is a clone of this repository
kept for this: the script fetches the results branch into it without file
contents, which turns it into a blobless partial clone.

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
`force` adds and replaces files and removes none. A text golden is therefore
not written over one that holds a text file the new outputs do not have: that
file would stay in the reference and every run would fail on the file count.
The job names the file; remove it from the results branch, or use another ref.
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
to the previous run (HEAD polling can skip commits), the id of the Actions run
that tested the commit and the repository it ran in (`run` and `run_repo`, so
a history row can link to it; absent on entries indexed before they were
recorded), per-workflow verdicts with the drift class, output sha, metrics and
timings. Entries are ordered by commit time, so re-testing an older commit
never reorders history. The history tables in a run's job summaries and the
dashboard's history and matrix views read only these files.

**Build run index** (workflow_dispatch) backfills or repairs a branch/lane
from the committed summaries; dispatch it once after the first deploy and
whenever an index file needs regenerating.

## PR comments (PULSE)

After each run, `scripts/notify_pr.py` can leave one comment on the ComfyUI
pull request that landed the tested commit. Its first line starts with
**(PULSE)**; below it, a row per workflow (verdict, new or inherited, PSNR,
mean MSE, % pixels changed, exec time, peak VRAM), the lane, and two links:
"Run details" to the Actions run that tested the commit, whose job summary
holds the full report (`run_url` in `summary.json`), and "results page" to the
commit's page on ci.comfy.org. The comment stays on the pull request and
GitHub deletes the run after 90 days, so the second link is the one that
lasts. A summary without a run has only the page, as "Run details".

- **When**: a workflow's outputs changed at this commit (`detail: new_drift`),
  or it hit an execution error while its prior run (`prior_verdict`) produced
  outputs, or a smoke-level workflow failed its checks while its prior run
  passed them. That last case has the headline "GPU workflow failed" and the
  verdict `smoke check failed`, with no numbers: nothing was compared. Never
  for pass, inherited drift, a smoke failure that was already there, infra
  errors, missing goldens, a workflow with no prior run, a re-test of a
  commit older than the previous run, or a run that covered several new
  commits (the change cannot be pinned on one PR).
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

## Lanes

A lane is one environment the corpus runs in: a worker image (Python, torch
build, CUDA channel), the RunPod endpoint running that image, a cadence and a
storage prefix. `manifest/lanes.json` declares the images once under `images`
and the lanes under `lanes`; `scripts/lanes.py` derives the build matrix, the
per-commit matrix and the matrix of a nightly or weekly run from it, and
`scripts/lanes.py validate` (run by the tests) keeps the file consistent.
Goldens are per lane: cu130+ images enable comfy-kitchen kernels that cu128
cannot, so a new lane always starts with its own baselines.

Today's lanes:

| Lane | Image | Role | Cadence |
|---|---|---|---|
| `py313-torch2.14.1-cu130` | `cu130-t2141` | primary, blocking | per-commit |
| `py313-torch2.15.0-cu132` | `cu132-t2150` | lane, informational | per-commit (from its first endpoint) |
| `py313-torch2.14.1-cu130-nightly` | `cu130-t2141` | lane, informational | nightly (from its first endpoint) |
| `py313-torch2.14.1-cu130-weekly` | `cu130-t2141` | lane, informational | weekly (from its first endpoint, and for workflows with models once those are fetched for each job) |
| `py312-torch2.11.0-cu128` | `cu128-t2110` (never rebuilt) | legacy | retired: history only |

- **Role**: exactly one lane is `primary`, the one `primary` names. It is the
  lane PULSE comments for, the one the dashboard opens on, and the endpoint
  the model sync runs on, so `validate` requires it to be per-commit. The
  others are `lane`, or `legacy` for a lane whose endpoint is never repointed
  automatically.
- **Cadence**: `per-commit` lanes with an `endpoint_id` run on every new master
  commit (**GPU regression**). `nightly` and `weekly` lanes with an
  `endpoint_id` run once a night and once a week (**GPU regression (nightly
  and weekly)**, see [below](#nightly-and-weekly-lanes)).
  `retired` means history only: the lane stays out of every matrix
  (even with `lane=`), of image builds and repoints, and `run_regression.py`,
  golden generation and `runpod_endpoints.py create/set-image` refuse it.
  Blessing or promoting its goldens still works (storage only).
- **`blocking`** is informational: it is shown on the dashboard, and nothing
  gates ComfyUI merges on it.

**Which lanes run a workflow** is decided by two keys of its entry in
`manifest/workflows.json`:

- `cadence`: `per-commit` (the default, also when the key is absent),
  `nightly` or `weekly`. A workflow runs on the lanes of its own cadence and
  on no other. Moving a workflow to another cadence is that one word, and a
  change of the primary lane edits no workflow entry.
- `lanes`: a list of lane ids that narrows it further (null, the default,
  means every lane of the cadence). Workflows that need custom-node packs
  must use it (see below).

`run_regression.py`, `golden_baseline.py`, the reports and the plans of both
workflow files share the one rule (`lanes.why_not_on_lane`). It is a hard
filter: a workflow that a dispatch names on a lane that does not run it is
skipped, with a log line that says why ("cadence nightly, the lane is
per-commit", or "manifest lanes: ..."), and the dashboard shows the cell as
not run. A report does not list a workflow of another cadence under "not run
on this lane". On a retired lane only the `lanes` list counts, so its goldens
of any cadence can still be blessed.

The two keys together can leave a workflow on no lane, and no run would turn
red for a workflow that never runs. So `validate` fails for an enabled
workflow that no lane runs (retired lanes do not count): "cadence 'nightly'
and lanes [...] leave it on no lane". A pack workflow names its lanes, so
moving one to `nightly` also means naming a nightly lane in its list.

A golden lives under a lane's prefix. A reference-level workflow that changes
cadence therefore has no golden on its new lane: its first run there ends in
`no_baseline`, and that run's output is approved without GPU time (**Golden
baselines** with `from_run`, `lane=<the new lane>` and `bless=true`, see
[Accepting an intentional change](#accepting-an-intentional-change-re-bless)).
Its history is split between the two lanes. A smoke-level workflow moves with
the edit alone.

**Always pass `lane=`** to **Golden baselines** and **Build run index**: left
empty, they resolve to the lane that owns the bare `regression` prefix
(layout 1), which is the retired cu128 lane.

**Check the endpoints against the manifest.** `RUNPOD_API_KEY=... python
scripts/runpod_endpoints.py status` prints the endpoint of every provisioned
lane, and each setting that differs from `lanes.json`: the largest number of
workers (the lane's `workers_max`, else `endpoint_defaults.workers.max`), the
idle timeout (`idle_timeout_s`, else `endpoint_defaults.workers.idleTimeout`),
the disk (`disk_gb`, else `endpoint_defaults.disk_gb`), and the GPU pools, the
data centers and the network volumes (the lane's `gpu_pools`,
`data_center_ids` and `network_volumes`, else the lists of
`endpoint_defaults`; the order of a list does not count). A retired lane's
endpoint is held to 0 workers and nothing else. `--strict` exits 1 when
something differs. Without another flag the command only reads.

**RunPod lowers the workers of an endpoint nobody uses.** RunPod's
documentation says an endpoint without a request for 3 days is lowered to 2
workers, and after 7 days to 0, where it stays until someone raises it. Jobs
sent to it then only wait in the queue. `status --restore-workers` raises the
largest number of workers to the manifest's when it is lower, with one PATCH.
It never lowers one and never touches a retired lane's endpoint. If the
account's worker quota refuses the raise, the command ends with RunPod's
message. The nightly and weekly workflow runs it before each test run. On a
per-commit lane that stood idle (a new lane before its first run), run it by
hand with `--lane <lane id>`. An id that is no lane ends the command with
"unknown lane", as it does for `create` and `set-image`.

### Nightly and weekly lanes

Workflows that are too slow or too many for every commit run once a night or
once a week. **GPU regression (nightly and weekly)**
(`.github/workflows/nightly-weekly.yml`) tests ComfyUI master HEAD:

| Cadence | Cron (UTC) | Lanes |
|---|---|---|
| nightly | `17 1 * * *`, every day at 01:17 | every `nightly` lane that has an `endpoint_id` and an enabled nightly workflow |
| weekly | `17 3 * * 0`, Sunday at 03:17 | the same for `weekly` |

`python scripts/lanes.py cadence nightly` prints the plan. With no such lane
the plan is `[]`, it says why each lane of the cadence is left out, and the
run ends after its plan job. So a lane can be declared before its endpoint
exists, which is the state of both lanes today: no workflow has either
cadence yet, and neither lane has an endpoint.

- **One lane per cadence.** Everything the suite stores about a commit on a
  lane assumes that one run wrote it: the summary, the start marker, the
  previous-run pointer, the goldens. So a cadence is a lane with its own
  prefix, never a second run inside a per-commit lane. The label names the
  cadence, because the dashboard prints only a lane's label.
- **Its own workflow file and group.** The runs have their own name in the
  Actions list ("GPU regression · nightly"), and wait in the concurrency
  group `gpu-nightly-weekly`. A per-commit tick is in another group and never
  replaces one. One group for both cadences, so the two never run at once.
  `queue: max` lets several runs wait in line. If GitHub ever rejects the
  file for that line, delete the line and do not dispatch while a run waits.
- **Its own endpoint, shared by the two lanes.** Both lanes name
  `"endpoint_name": "comfyci-cu130-nightly-weekly"`, so a nightly or weekly
  run never takes a worker from a per-commit run. Lanes that share an
  endpoint (the same `endpoint_name` or the same `endpoint_id`) must ask for
  the same `image`, `endpoint_name`, `workers_max`, `disk_gb`,
  `idle_timeout_s`, `env`, `gpu_pools`, `data_center_ids` and
  `network_volumes`, and one name cannot stand for two ids: `validate` checks
  it. It compares what each lane asks for with `endpoint_defaults` filled in,
  so a lane that writes a default out agrees with one that leaves it out.
  `runpod_endpoints.py set-image` sends such an endpoint one PATCH, and
  `create` refuses to make it twice.
- **Each lane job** runs the steps of the per-commit lane job (a test holds
  the two files to the same text), with one more before the suite: `status
  --lane <lane> --restore-workers`. There is no "All lanes" job and no PR
  comment. The "All lanes" summary of every per-commit run shows where these
  lanes stand (see [What a run shows](#what-a-run-shows)).
- **A commit is tested once per lane**, by the rule of the per-commit lanes
  (see [Verdicts](#verdicts)). A night without a new commit tests nothing and
  shows the published result.
- **Who is told.** GitHub mails a failed scheduled run to one person: whoever
  last changed the `cron` lines of the workflow file.

To start a run by hand:

```
gh workflow run nightly-weekly.yml --repo Comfy-Org/comfyci-runpod-worker \
    -f cadence=nightly [-f commit=<sha>] [-f workflows=<ids>] [-f lane=<lane id>] \
    [-f publish=false]
```

`workflows` and `publish` work as on **GPU regression**: a run that names its
workflows is a re-test or a first look, and with `publish` off nothing is
written. The plan knows the workflows a dispatch names: a lane whose only
workflow is still switched off (`"enabled": false`) is tested when the
dispatch names that workflow. To show a published result of one of these
lanes again without testing, dispatch **GPU regression** with
`report_only: true` and `lane=<the lane>`: it lists every lane of the
registry.

#### How a nightly or weekly run goes

A per-commit run is over in five minutes, and the cron tests its commit again
when something went wrong. A nightly or weekly run can take hours and cost
dollars, and the lane's next run is a day or a week away. So
`scripts/run_regression.py` runs a lane of these two cadences in its own way:

| | Per-commit lane | Nightly or weekly lane |
|---|---|---|
| Jobs on the endpoint at once | all | the lane's `max_in_flight`, or `--max-in-flight` |
| Attempts for a job | 2, the second right away | 1, and one pass at the end of the run |
| Wait in RunPod's queue | 600 s, plus 60 s for each job before it | 1,800 s |
| A look at a job | every 10 s | every 3 s |
| The run's own deadline | none | the lane's `timeout_minutes` minus 25, or `--deadline-minutes` |
| A workflow that was not tested | a warning. The cron tests the commit once more | **red** (exit code 1) |
| The previous-run pointer | moves when every workflow produced outputs | moves on every full run |

- **A few jobs at a time.** Fifty jobs submitted at once to two workers would
  mostly be cancelled for waiting in the queue. A job is submitted only when
  one of the lane's slots is free, so its queue clock starts when a worker can
  take it. The runner then hands the next job to a worker itself, and the
  worker waits, billed, until it does. So the runner looks at a job every 3
  seconds, and the endpoint keeps an idle worker for 60.
- **The deadline.** GitHub ends the lane's job after `timeout_minutes`. Before
  the run submits a job, it asks whether the job could still be running 25
  minutes before that. A job can take its `timeout_s` plus 900 seconds, plus
  the 1,200 seconds more it may wait in the queue. When that does not fit, no
  job is submitted: the workflow's cell says `"worker_status": "not_reached"`
  (verdict `infra_error`), and the run goes on with the next workflow, which
  may be smaller. So no job is in flight when the deadline comes, and the last
  25 minutes belong to the publish and the report. `lanes.py validate`
  refuses a lane whose `timeout_minutes` is 25 or less, and an enabled
  workflow whose job can never fit before the deadline of a lane it runs on:
  either would be `not_reached` in every run.
- **A job the runner cannot see.** When the status of a job cannot be read
  for 5 minutes (the runner's network, RunPod), the runner gives the job up
  and cancels it, so it does not run on beside the second job the workflow
  gets at the end of the run. The per-commit lanes have the same 5 minutes.
- **Checkpoints.** After every 10 workflows the run writes a summary of what
  it has, with `"partial": {"done": 10, "of": 55}`, and pushes it with the
  workflow folders written so far. `done` counts the workflows that have a
  result, not those the deadline did not reach. A run that dies, is cancelled
  or loses its network leaves that behind. **GPU regression** with
  `report_only: true` and the lane shows it ("The run that tested this commit
  did not finish: 40 of 55 workflows have a result"), and the next run of the
  commit is its second attempt. The run index is written by the final publish
  alone, so a run that died is in no history table. A run that names its
  workflows pushes nothing early. Nor does a second full run of a commit that
  has the lane's full result (the cron's retry, a dispatch): that result stays
  on the branch until the run is done, and a run that dies takes nothing away.
- **The pass at the end.** When every workflow has had its one attempt, those
  that were not tested for a reason that can pass (the queue, the network, a
  worker error) are tried once more, while the deadline allows. Not tried
  again: a job that used up its whole time, because a second one can cost
  another hour (RunPod ended it, or the runner gave up on it, and its
  `run.json` says `"reason": "timed_out"`). And a workflow that another job
  would not change: `missing_model`, `model_access_error`, `not_reached`,
  `bad_request`. The cell of a workflow that got its second job says
  `"tries": 2`, whatever that job ended with.
- **A red run.** The exit code is 1 when a workflow failed, as on every lane,
  and also when one stayed untested. GitHub mails a red scheduled run, and
  nothing else would tell anyone.
- **A stopped run.** When a run is cancelled or reaches its time limit, GitHub
  sends the step an interrupt, 7.5 seconds later a terminate, and 2.5 seconds
  after that it kills the step. On the first signal the script cancels its
  RunPod jobs in flight, and once more when RunPod has answered every
  submission that was on its way: a job is known only then. That second pass
  comes after 1 second at the earliest and after 4.5 at the latest. After 6
  seconds the script ends, whatever RunPod has answered, so its last line is
  out before the terminate. The line names the jobs it cancelled, and apart
  from them any job whose cancel RunPod did not take: that one may run on,
  and is to be cancelled by hand. It also counts the submissions RunPod had
  not answered: each may have started a job that has no id yet, to be looked
  for on the endpoint. There is no time to push anything then, which is what
  the checkpoints are for. The step starts the script with `exec`, so the
  script itself is the process that gets the signals.
- **Who broke it.** A nightly run covers about 6 commits and a weekly run
  about 44, and a workflow's last result is not always in the run before. So
  the summary says, for each red workflow, how many commits lie between the
  workflow's own last red or green result and this commit: `ranges`, with
  `{"prev", "commits_between", "compare_url"}` per workflow id. The report
  words the failure from it ("new since `e9afbcc`, 7 commits"), and a
  workflow without an earlier result reads "first run". The open failing
  chain carries the range too, so nobody is pointed at the pull request of
  the newest of 44 commits. To find the commit, test single commits of the
  range. The report prints the command:

  ```
  gh workflow run nightly-weekly.yml --repo Comfy-Org/comfyci-runpod-worker \
      -f cadence=weekly -f commit=<sha> -f workflows=<id>
  ```

  Each such step publishes under its own commit with `scope`, moves no
  pointer, and shortens the chain: six steps halve 44 commits down to one.
  A step that fails becomes the chain's start, with the commits it counted
  since the last pass. A step that passes becomes the chain's last pass. The
  commits the chain's start counted then reach back past that pass, so the
  open failure names "one of the commits since its last pass" with their
  compare link and no number: only GitHub can count them, and a passing step
  does not ask. So after a search that ends on a pass the reader gets a
  compare link without a count, also when one commit is left. The page of the
  failing run follows the chain from then on. Its block says "When this run
  was made it had last passed at `e9afbcc`", names the commit of the later
  pass, and gives the compare link from that pass without the count the run
  recorded. In the lane's history each step's row says "1 of N workflows", so
  a green step does not read as a green week. The row of a full run never
  says so, also not one from before a workflow joined the lane.

The keys a lane may add. Every one is optional, and `validate` checks its
type:

| Key | Read by | Meaning |
|---|---|---|
| `endpoint_name` | `runpod_endpoints.py create`, `validate` | the endpoint's name on RunPod (default `comfyci-<image>`). Lanes with the same name share one endpoint |
| `workers_max` | `runpod_endpoints.py create`, `status` | the endpoint's largest number of workers (default `endpoint_defaults.workers.max`, 3) |
| `disk_gb` | `runpod_endpoints.py create`, `status` | container disk of the endpoint's workers (default `endpoint_defaults.disk_gb`, 40) |
| `idle_timeout_s` | `runpod_endpoints.py create`, `status` | how long a worker stays up after a job (default `endpoint_defaults.workers.idleTimeout`, 20) |
| `env` | `runpod_endpoints.py create` | environment variables of the endpoint, on top of `endpoint_defaults.env` |
| `gpu_pools` | `runpod_endpoints.py create`, `status` | the RunPod GPU pools the endpoint's workers run on, at least one (default `endpoint_defaults.gpu_pools`: `ADA_24`, the RTX 4090). The lane's `gpu` stays the label people read and selects nothing |
| `data_center_ids` | `runpod_endpoints.py create`, `status` | the data centers the workers may run in, at least one (default `endpoint_defaults.data_center_ids`: `EU-RO-1`) |
| `network_volumes` | `runpod_endpoints.py create`, `status`, `validate`, the manifest lint | the network volumes the workers mount (default `endpoint_defaults.network_volumes`: the model volume). `[]` means no volume: the lane must then say `"model_storage": "job"`, and a workflow that runs only on such lanes does not count against the volume's size |
| `timeout_minutes` | `lanes.py cadence`, `nightly-weekly.yml`, `run_regression.py`, `validate` | the time limit of the lane's job (default 360, GitHub's own). On a nightly or weekly lane the run's own deadline is this minus 25, so `validate` asks for more than 25 there, and for every enabled workflow of the lane that its `timeout_s` plus 2,100 seconds is under that deadline |
| `max_in_flight` | `run_regression.py` | how many jobs a run has on the endpoint at once (default: all of them, as on the per-commit lanes). `--max-in-flight` overrides it |
| `model_storage` | the manifest lint, `sync_models.py` | `"job"`: the lane's models are not on the network volume. The model sync leaves out a workflow that runs only on such lanes, and the lint does not count it against the volume's size. Nothing fetches models for a job yet, so such a workflow ends as `missing_model` unless another lane's workflow puts the same files on the volume. The lint fails for an enabled one that does not find every file there. Default `"volume"` |

None of them is published: the run index copies a fixed list of lane keys.

### Adding a lane

The cu132 lane is the worked example.

1. Add the image under `images` (python, cuda, `torch_index`, `torch_spec`
   with exact torch and torchvision pins; no torchaudio, ComfyUI dropped it)
   and the lane under `lanes` with `"endpoint_id": null`, `layout: 2` and
   `prefix: regression/lanes/<lane id>`:

   ```json
   "cu132-t2150": {"python": "3.13", "cuda": "13.2",
                   "torch_index": "https://download.pytorch.org/whl/cu132",
                   "torch_spec": "torch==2.15.0 torchvision==0.29.2"}

   "py313-torch2.15.0-cu132": {"label": "py3.13 / torch 2.15.0 / cu132 / RTX 4090",
     "python": "3.13", "torch": "2.15.0", "cuda": "13.2", "gpu": "RTX 4090",
     "cadence": "per-commit", "blocking": false, "role": "lane", "layout": 2,
     "prefix": "regression/lanes/py313-torch2.15.0-cu132", "image": "cu132-t2150",
     "endpoint_id": null, "min_cuda_driver": "13.2"}
   ```

   Check the pins on the index first: torchvision does not always have a
   release per torch release (torch 2.15.0 pairs with torchvision 0.29.2, a
   single `py3-none` wheel for torch 2.14 and later; there is no 0.30). Add
   the lane to the `lanes` filter of the pack workflows if they should run
   there. `min_cuda_driver` is the endpoint's host driver floor; RunPod's
   serverless 4090 hosts report CUDA 13.2 (driver 595.x). Merge, or build and
   provision from the branch first (see the cu132 order below).
2. `build-image.yml` builds the images a push to main changed (or dispatch
   it from the branch with the image id) and tags each `:<image>` and
   `:<image>-<sha>`. A change to the Dockerfile, `src/`, `docker/` or
   `manifest/custom_nodes.json` builds every buildable image. A change to
   `lanes.json` alone builds only an image whose entry is new or differs
   (`lanes.py images --since <commit>`), so adding a lane to an existing
   image or writing an endpoint id builds and repoints nothing. A lane that
   has its endpoint and moves to another image, leaves the `legacy` role or
   is no longer `retired` does build that image: the build is what points the
   endpoint at it. The commit to compare with is the head of the last push
   run of the workflow that succeeded, so a build that failed is made up for
   by the next push that starts the workflow, whatever that push changes.
   When GitHub names no such run, it is the commit before the push. Every
   image CI builds bakes every pack in
   `manifest/custom_nodes.json`, so a pack requirement that does not install
   on the lane's Python breaks the build. **Python 3.14**: the registry's
   `opencv-python(-headless)==4.12.0.88` pin needs `numpy<2.3`, and no numpy
   below 2.3.2 has cp314 wheels, so a 3.14 image needs the opencv pin bumped
   (4.13.0.92 or later, never 4.13.0.90). That pin is global, so the bump
   also moves the cu130 image (numpy and opencv) and needs its pack goldens
   checked; that is why the cu132 lane stays on 3.13.
3. Create the serverless endpoint from the `:<image>-<sha>` tag, once, by hand
   (it is the one billable step; idle endpoints cost nothing at 0 minimum
   workers):

   ```
   RUNPOD_API_KEY=... python scripts/runpod_endpoints.py create \
       --lane <lane id> --tag <image>-<sha> --write
   ```

   The request comes from `endpoint_defaults` in `lanes.json` (pool, workers,
   volume, data center, registry credential, timeouts) plus the lane's CUDA
   floor and the endpoint keys the lane names (`endpoint_name`,
   `workers_max`, `disk_gb`, `idle_timeout_s`, `env`, `gpu_pools`,
   `data_center_ids`, `network_volumes`, see
   [Nightly and weekly lanes](#nightly-and-weekly-lanes) and
   [A lane on another card](#a-lane-on-another-card)); `--dry-run` prints
   it, `--write` records the new id as the lane's
   `endpoint_id`. Commit that. The account has a 10-worker quota across all
   endpoints; if the create is refused, give the lane `"workers_max": 2`.
   From then on `build-image.yml` repoints the lane at each new sha tag
   (`runpod_endpoints.py set-image`); legacy and retired lanes are never
   moved. A repoint is a rolling release: workers already running the
   old image keep serving jobs until RunPod replaces them, which can take a
   long time when the new image is large and the new workers sit throttled
   behind them, so a run right after a repoint may silently use the old image
   (its `env.image_sha` tells). Check `isStale` in the endpoint's worker list
   before generating goldens; setting the endpoint's max workers to 0 and back
   to its normal value flushes the stale workers within seconds.
4. Generate and bless goldens for the lane: **Golden baselines** with the
   current release tag, `workflows: all`, `lane: <lane id>`, inspect, then
   `bless_only`.
5. The per-commit cron picks the lane up on its next tick (a lane with a null
   `endpoint_id` is skipped), and ci.comfy.org lists it in the lane selector
   after the first publish.

**A nightly or weekly lane** takes the same steps, with these differences:

1. It runs an image that exists, so there is nothing to build: add only the
   lane, with `"cadence": "nightly"` or `"weekly"`, the cadence as the first
   word of its `label`, and its own `prefix`. The merge builds and repoints
   nothing.
2. Give it an endpoint of its own, not a per-commit lane's: name it with
   `endpoint_name`, and put the endpoint's settings on the lane
   (`workers_max`, `disk_gb`, `idle_timeout_s`, `env`) and the job's
   `timeout_minutes`. Two lanes that share the endpoint carry the same
   `endpoint_name` and the same settings.
3. Create the endpoint once, for one of the lanes, from the tag the primary
   lane's endpoint runs (`status` prints it). Look at the request first:

   ```
   python scripts/runpod_endpoints.py create --lane <lane id> \
       --tag <image>-<sha> --dry-run
   RUNPOD_API_KEY=... python scripts/runpod_endpoints.py create \
       --lane <lane id> --tag <image>-<sha> --write
   ```

   A lane that shares the endpoint gets the same `endpoint_id` by hand.
   Every variable in `env` that reads a RunPod secret needs that secret to
   exist first. Count the workers before: the account's quota of 10 is over
   all its endpoints, each of today's endpoints takes 3, and a retired
   endpoint counts until it is at 0. If the create is refused, give the
   lanes of the endpoint `"workers_max": 2`. Create it on the day the first
   workflow of the cadence merges: RunPod lowers the workers of an endpoint
   nobody uses, and until a first run nobody restores them.
4. A reference-level workflow needs its golden on the lane (see
   [Lanes](#lanes)). A smoke-level one needs nothing.
5. Its cron picks the lane up from the first night or Sunday on which it has
   an endpoint and an enabled workflow of its cadence. Until then
   `python scripts/lanes.py cadence <cadence>` prints `[]` and says why.

#### A lane on another card

Every lane gets the GPU pool, the data center and the network volume of
`endpoint_defaults`: the RTX 4090 pool in EU-RO-1, with the model volume. A
lane can name its own with `gpu_pools`, `data_center_ids` and
`network_volumes`. Each replaces the default list whole, and a key the lane
leaves out keeps the default. A weekly lane for models that need more than
24 GB, on the 96 GB pool, in three data centers, without the model volume:

```json
"py313-torch2.14.1-cu130-weekly-96gb": {
  "label": "weekly · py3.13 / torch 2.14.1 / cu130 / RTX PRO 6000",
  "python": "3.13", "torch": "2.14.1", "cuda": "13.0", "gpu": "RTX PRO 6000",
  "cadence": "weekly", "blocking": false, "role": "lane", "layout": 2,
  "prefix": "regression/lanes/py313-torch2.14.1-cu130-weekly-96gb",
  "image": "cu130-t2141", "endpoint_id": null, "min_cuda_driver": "13.0",
  "endpoint_name": "comfyci-cu130-weekly-96gb",
  "gpu_pools": ["BLACKWELL_96"],
  "data_center_ids": ["US-CA-2", "US-NC-1", "EUR-IS-2"],
  "network_volumes": [], "model_storage": "job",
  "workers_max": 1, "disk_gb": 120, "idle_timeout_s": 60,
  "env": {"HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}"},
  "max_in_flight": 1, "timeout_minutes": 350}
```

- **The pool.** `GET https://api.runpod.io/v2/catalog/gpus` (with the API
  key) lists every card with its `pool`: `ADA_24` is the RTX 4090,
  `ADA_48_PRO` the 48 GB cards (L40S, RTX 6000 Ada), `BLACKWELL_96` the RTX
  PRO 6000 with 96 GB. With `?include=AVAILABILITY&product=SERVERLESS` it
  also names the data centers that have each card now, and the CUDA versions
  of their hosts (`min_cuda_driver` is the floor).
- **`gpu` is the label.** It is published with the lane, and people read it
  on the dashboard and in a PR comment. It selects nothing, so write the card
  the pool gives.
- **No volume.** A network volume lives in one data center, and only workers
  there mount it. The model volume is in EU-RO-1, so a lane in other data
  centers runs without it: `"network_volumes": []`. `validate` then asks for
  `"model_storage": "job"`, the lint does not count a workflow that runs only
  on such lanes against the volume, and the model sync does not download its
  models. Nothing fetches models for a job yet, so such a lane cannot run
  until that exists, and the lint fails for an enabled workflow with models
  of its own on it.
- **Another volume.** A lane may name a volume in its own data center
  instead. The model sync fills only the volume of the primary lane's
  endpoint, so the models have to be copied there by hand, and the lint
  counts the lane's model files against `endpoint_defaults.volume_gb` like
  those of any other lane.
- **Its own endpoint.** Give the lane an `endpoint_name`. Lanes that share an
  endpoint must name the same three lists.
- **Its workflows.** A weekly workflow without a `lanes` list runs on every
  weekly lane. So name this lane in the `lanes` list of each workflow that
  needs the card, and the other weekly lane in the lists of the rest.
- **Workers.** Count them before the create: the account's quota of 10 is
  over all its endpoints.

The steps are those of any lane: create the endpoint (step 3 above,
`--dry-run` shows the pool, the data centers and the volumes it will ask
for), then the references. `status` holds the endpoint to the lane's lists.

### Changing the primary lane

1. In `lanes.json`, set `primary` to the new lane, give it `role: primary`
   (and `blocking: true`), and move the old primary to `role: lane` or
   `legacy`. `validate` refuses a primary that is not per-commit or not in the
   primary role, and a second lane in that role. The new primary needs its
   goldens blessed and an `endpoint_id`.
2. What follows the flip on merge: PULSE comments for the new primary only,
   the dashboard opens on it, and the
   model sync runs on its endpoint (`scripts/sync_models.py`; the
   `RUNPOD_ENDPOINT_ID` secret is only a fallback when the primary has no
   endpoint).
3. The run index (`index/lanes.json`) picks the new primary up at the next
   publish. To publish it at once, dispatch **Build run index** with
   `branch=master` and `lane=<new primary>` (CPU only).
4. The nightly and weekly lanes follow the primary lane's image, and a lane
   is tied to its image by its id and its prefix. So they are replaced, not
   edited: add a `<new lane>-nightly` and a `<new lane>-weekly` lane on the
   new image, with the endpoint keys and the `endpoint_id` of the old pair,
   and retire the old pair (below) with `"endpoint_id": null`: the endpoint
   is the new pair's now, and `status` holds the endpoint of a retired lane
   to 0 workers. Point the endpoint at the new image (`runpod_endpoints.py
   set-image --lane <new nightly lane> --tag <image>-<sha>`, one PATCH for
   both lanes) and raise its CUDA floor in the RunPod console to the new
   lanes' `min_cuda_driver`. No workflow
   entry changes, except a `lanes` list that names the old nightly or weekly
   lane. Smoke-level workflows need nothing else. Reference-level ones need
   their goldens approved on the new lanes (see [Lanes](#lanes)).

### Retiring a lane

Keep the lane in `lanes.json`: its history stays readable only while the
registry still describes it (a removed lane's index entry freezes with its
last role, and every default-prefix path breaks if the layout-1 lane goes).

1. Set `cadence: retired` and `role: legacy` (and `blocking: false`). Keep
   `layout`, `prefix`, its image entry and its `endpoint_id`. If it was the
   primary, change the primary first (above).
2. Merge. The cron stops testing it at the next tick; image builds and
   repoints leave it alone.
3. Scale its endpoint to max 0 workers in the RunPod console, once no GPU
   regression run that started before the merge is still using it. Do not
   delete it yet: an idle endpoint costs nothing and max 0 frees its share of
   the worker quota. `runpod_endpoints.py status` names a retired endpoint
   that still has workers. ci.comfy.org needs to know the `retired` cadence,
   or it shows the lane as stale a day after its last run.
4. The model sync already follows the primary, so nothing depends on the
   retired endpoint. From now on always pass `lane=` to **Golden baselines**
   and **Build run index**.

### Moving from cu128 to cu132 (operational order)

The PR that makes cu130 the primary, retires cu128 and adds the cu132 lane,
in order:

1. Deploy the dashboard's support for the `retired` cadence (comfyci-dashboard,
   "Handle retired lanes and share lane ordering") first, before the bless in
   step 4. That bless is the first publish of the branch's registry to
   `index/lanes.json`: cu130 as the primary and cu128 as `retired`, `legacy`
   and informational, while main still tests cu128 as its blocking primary.
   Without that support the dashboard only knows per-commit, nightly and
   weekly: its lane selector captions cu128 "retired · informational", its
   lane page turns amber after 24 hours without a run ("no run in the last
   24 h on this retired lane") and adds "the worker may be stalled" after 48,
   and the frontend lane caption calls it "every core commit". Main's next
   publish restores cu128's old entry; the first publish after the merge
   (step 5) retires it for good.
2. Build the cu132 image from the branch (CPU, GitHub-hosted):
   `gh workflow run build-image.yml --ref lanes-cu132 -f images=cu132-t2150 -f repoint=false`.
   It is tagged `cu132-t2150-<branch head sha>`.
3. Create the endpoint and record it on the branch:

   ```
   python scripts/runpod_endpoints.py create \
       --lane py313-torch2.15.0-cu132 --tag cu132-t2150-<sha> --dry-run
   RUNPOD_API_KEY=... python scripts/runpod_endpoints.py create \
       --lane py313-torch2.15.0-cu132 --tag cu132-t2150-<sha> --write
   ```

   Commit and push the `lanes.json` change to the branch. 3 + 3 + 3 workers
   plus comfy-pulse's 1 fills the 10-worker quota exactly; if the create is
   refused, add `"workers_max": 2` to the lane for the create rather than
   scaling cu128 down while it is still the primary on main, and take it out
   again before you commit the `endpoint_id` (step 7 raises the endpoint to
   3). `create` and `status` read `workers_max`: dropping it changes nothing
   on the endpoint, a re-create gets 3, and `status` then expects 3.
4. Generate goldens from the branch (a fresh endpoint has no stale workers):
   `gh workflow run golden-baseline.yml --ref lanes-cu132 -f lane=py313-torch2.15.0-cu132 -f ref=v0.42.0 -f workflows=all -f bless=false`
   (v0.42.0 is the tag the cu130 goldens use). Inspect
   `regression/lanes/py313-torch2.15.0-cu132/golden/<workflow>/v0.42.0/` on the
   results branch, then bless:
   `gh workflow run golden-baseline.yml --ref lanes-cu132 -f lane=py313-torch2.15.0-cu132 -f ref=v0.42.0 -f workflows=all -f bless_only=true -f reason="cu132 lane baseline"`.
   A bless from the branch publishes the branch's registry to
   `index/lanes.json` (cu130 primary, cu128 retired; hence step 1) until
   main's next publish rewrites it; step 9 makes it stick.
5. Merge. That starts **Build worker images** and **Sync models** (on the
   cu130 endpoint), and the cron stops testing cu128 from its next tick. Wait
   for both runs to finish. The build makes the images the merge changed and
   repoints their lanes (see [Adding a lane](#adding-a-lane), step 2): both
   `cu130-t2141` and `cu132-t2150` when the merge touches the Dockerfile,
   `src/`, `docker/` or the pack registry, only an image whose entry is new
   or changed on main otherwise, and none when the merge only adds a lane or
   an endpoint id. A lane whose image was not built keeps the tag its
   endpoint has.
6. Expect the cron to test master HEAD on cu132 within minutes of the
   merge, before the repoint or before the stale workers are flushed, so on
   the branch's image or a mix of old and new workers. A commit is tested
   once per lane, so the cron does not test it again on the rebuilt image;
   step 8 does.
7. When no **GPU regression**, **Golden baselines** or **Sync models** run is
   in progress (max workers 0 empties an endpoint's worker list within
   seconds, cutting off any job a worker is running):
   - scale the cu128 endpoint `xwdhxbhd9qixjj` to max 0 workers. Keep it and
     its `endpoint_id`; nothing runs on it any more, and this frees its 3
     workers of the quota;
   - flush stale workers on the cu130 and cu132 endpoints: max workers 0,
     then back to the previous max. If the cu132 endpoint was created with
     max 2 (step 3), raise it to 3 now that cu128 is at 0;
   - check that `isStale` is gone from both worker lists.
8. Test master HEAD on the rebuilt images, both lanes in one run (no `lane=`):
   `gh workflow run regression.yml -f commit=<master HEAD sha>`. Check that
   the run is not cancelled (a cron tick that queues behind a running run
   cancels a pending dispatch; dispatch it again if so). In each lane's
   `runs/master/<sha>/summary.json`, every workflow's `env.image_sha` must
   equal the merge commit, the sha in the new `-<sha>` image tags (for an
   image the merge did not build: the sha in the tag its endpoint has); one
   that does not ran on a stale worker: flush again and re-dispatch. Re-bless
   drifted workflows with `from_run`.
9. Publish the primary flip: `gh workflow run build-index.yml -f branch=master -f lane=py313-torch2.14.1-cu130`.
10. The `RUNPOD_ENDPOINT_ID` secret (set 2026-08-18, so the cu128 endpoint) is
    now only the sync's fallback: point it at the cu130 endpoint or delete it.

## Custom-node packs

Four workflows exercise custom-node packs on real GPUs, so a core commit or a
torch lane that breaks a pack is caught within the hour and bit-compared
against goldens like every other workflow: `kjnodes_sdxl_t2i` (KJNodes image
nodes on an SDXL render), `vhs_video_roundtrip` (VideoHelperSuite decode of a
clip, PNG frames, gif encode), `impact_sdxl_detailer` (Impact-Pack face
detection and detailing with the Impact-Subpack ultralytics detector) and
`manager_import_smoke` (ComfyUI-Manager import, no pack nodes).

`manifest/custom_nodes.json` is the registry: packs keyed by their
`custom_nodes` directory name, with `repo`, `sha` (a full commit), `label`,
`kind` (`git`, the default, or `pip` for a Python package such as Manager 4.x,
which ComfyUI loads with `--enable-manager` and must not be cloned into
`custom_nodes`), `requirements` (install the pack's requirements.txt), and
optionally `pip_args`, `comfy_flags` the pack needs, `files` written into the
checkout before the server starts (Manager's config, which keeps its startup
offline), and for a pip pack the log line that proves it loaded (`log_ok`) or
that ComfyUI prints when it is absent (`log_missing`). `constraints` are extra
version pins for pack installs on top of the lane's torch pins (both opencv
variants in lockstep, so Manager's startup fixer finds nothing to reinstall).
Pins follow the frontend's custom-node gate
(`browser_tests/fixtures/data/customNodeManifest.core.json` in
ComfyUI_frontend), so both CI systems test the same pack commits; bump them
by PR.

A workflow entry names its packs under `custom_nodes` and restricts itself
with `lanes` to lanes whose image bakes them. Pack requirements are baked into
the images CI builds (`build` not false in `lanes.json`); the legacy cu128
image is never rebuilt, so pack workflows never run on its lane and their
cells show as not run there. `scripts/lanes.py validate` and the manifest lint
enforce both rules.

How a job runs a pack (`src/custom_nodes.py`):

- The image build fetches every pack at its sha (`git init`, `fetch --depth 1
  <sha>`, `checkout FETCH_HEAD`: the gate's own recipe) into
  `/comfy-cache/custom_nodes/<name>/<sha>`, installs its requirements under
  the torch pins plus the registry's, and records under
  `/comfy-config/custom_nodes/` the requirements hash (or the pip pin) it
  baked. The image's `env` block carries `custom_nodes_sha256`, the registry
  it was built from. A pack's `install.py` is never run: dependencies come
  from its requirements.txt, models from the workflow manifest.
- Per job, after the commit checkout, the worker removes the packs an earlier
  job left behind, copies each requested pack from the cache into
  `ComfyUI/custom_nodes/<name>` (fetching it first when that pin is not
  cached) and pip-installs its requirements only when their hash differs from
  the baked one, so a pin bump works before the rebuild lands. A job that
  requests no packs runs the bare `custom_nodes/` the image shipped. ComfyUI
  ignores `custom_nodes/` and `user/` in git, so the checkout's `git clean`
  never touches them.
- Once the server is up, ComfyUI's own import summary (`Import times for
  custom nodes:`, read with the console formatter's colour codes and `[INFO]`
  tags stripped) is read from the server log. A requested pack that failed to
  import, or has no line there, fails the run with worker status
  `custom_node_error` (verdict `execution_error`) before the workflow is
  validated, whether or not the workflow uses its nodes. A server that dies
  or never answers is reported the same way only when the log proves a pack
  is at fault (the summary was printed and the pack is not ok in it, or
  ComfyUI said a pip pack is absent); a server that fell over before loading
  packs (CUDA init, OOM, a core import error) keeps `server_died` /
  `server_start_timeout` (infra, retried once) with the packs' status as
  detail. A pack that could not be fetched or pip-installed is
  `custom_node_install_error` (infra, retried once).
- `run.json` carries `custom_nodes` (per pack: sha, where it came from, pip
  time, import status and time), `summary.json` copies it per workflow, the
  run index keeps the import status per pack (`cn`), and `blessed.json`
  records the packs a golden was generated with.

Adding a pack: declare it in the registry, add the workflow with
`custom_nodes` and `lanes`, merge (the baking images rebuild and their lanes
are repointed), let `sync-models.yml` fetch any models it needs, generate and
bless the golden, then enable the workflow. Bumping a pin: edit the sha,
merge, re-bless the workflows that use the pack.

## Frontend CI index

`frontend-index.yml` (cron `13,43 * * * *`, offset from the regression cron)
runs `scripts/frontend_index.py`, which turns three public data sources of
`Comfy-Org/ComfyUI_frontend` into compact static JSON under `frontend/` on
the `results` branch, next to `regression/`. ci.comfy.org's Frontend tab
reads only these files (`NEXT_PUBLIC_FRONTEND_BASE`, default
`https://raw.githubusercontent.com/Comfy-Org/comfyci-runpod-worker/results/frontend`).

```
frontend/
  perf/head.json             metric registry, rule, scenarios, newest 100 entries, latest comparison
  perf/<YYYY-MM>.json        complete monthly shards (newest first)
  ci/main.json               column registry, newest 300 main commits, first_bad, coverage
  ci/main/<YYYY-MM>.json     complete monthly shards
  ci/custom-nodes.json       newest 90 custom-nodes gate runs with shard rows, core pin, streak
```

All times are UTC epoch seconds, every file has `schema_version: 1` and every
head carries `updated_ts`. A tick is a fetch phase, budgeted by the API's
`X-RateLimit-Remaining` with a 150-request reserve, then a write phase: the
three writers are derived writers (re-run on the freshest tree when a push
retries behind a concurrent publisher), merge by key and only write a file
whose content changed, so a tick with nothing new pushes nothing.

**Perf** (`perf-data` branch, `baselines/perf-<ts>-<sha8>.json`, of which the
frontend keeps only the newest 20): one entry per baseline, kept forever.

| key | meaning |
|---|---|
| `id`, `c`, `t` | `<ts>-<sha8>` (the file is `baselines/perf-<id>.json`), commit, baseline time |
| `n`, `nx` | samples per scenario; `nx` lists the scenarios with a different count |
| `s` | per scenario, the mean of each sample metric in `metrics` order (8 significant digits, bytes as integers, null when absent) |
| `cls` | per scenario, one letter per metric: `r` regression, `i` improvement, `n` neutral, `x` noisy (CV > 50 %), `-` fewer than two history values |
| head `metrics`, `targets`, `window`, `rule`, `stats` | the frontend's 12 reported metrics with label, unit and `min_abs_delta`; `p5_fps: 52`; the 20-run window; the z > 2 / CV > 50 % rule; current = median, history = mean, sample stddev, delta against the history mean |
| head `latest.cmp` | per scenario, `[current median, history mean, history stddev, history n, z, cls]` per metric for the newest baseline |

`cls` is computed once, at ingest, with the frontend's own rule
(`scripts/perf-stats.ts`, `scripts/perf-report.ts`): the baseline's medians
against the means of the up to 20 entries preceding it by time, so every
historical point carries the badge the frontend would have shown.

**CI on main** (push runs of every `CI:` workflow): one entry per commit.

| key | meaning |
|---|---|
| `c`, `ct`, `o` | commit, commit time, rollup of the core columns (`failure`, `pending`, `cancelled`, `success`; skipped never blocks) |
| `m`, `mi` | subject, author, PR, PR title/author/avatar; `mi: true` while `m` is provisional (from the run's `head_commit`, upgraded with the commits API at 25 commits per tick, newest first, monthly shards included) |
| `w` | per column id: `{id: run id, c: conclusion, s: status while not completed, t: start, d: duration s, a: attempt when > 1}` |
| head `columns` | ordered registry `{id, label, paths, name, group}`, groups core / website / billing / validation; lint matches both lint files; any other `CI:` workflow seen on a main push is appended as group `other` |
| head `first_bad` | per column, the start of the open failing chain (red opens, success closes, cancelled/pending/missing neither) |
| head `coverage` | `from_ts`, `to_ts`: the index is complete for runs created between them (`to_ts` is the tick time of the last complete listing, as of the last write: a tick that learns nothing else is not written); `partial`: the last backfill stopped short (budget) |

Incremental ticks list runs created since the oldest unsettled commit (a
cell still running, or a core column missing on a commit younger than 3 h),
at least 6 h back and never later than `coverage.to_ts`, so a stalled cron
or a broken tick leaves no hole. When that listing overflows its five pages
(days away), the gap above `to_ts` is closed in day windows while the
budget lasts and the next tick continues from where it stopped. Reruns of
commits older than 6 h are only caught by a backfill (the API cannot sort
by `updated_at`). Dispatch the workflow with `backfill_days` to walk day
windows back from `coverage.from_ts` (a 90-day backfill is about 500
requests plus metadata upgrades spread over the following ticks); `only`
(csv of `perf`, `ci`, `gate`) limits a run.

**Custom-nodes gate** (`ci-tests-custom-nodes.yaml`, scheduled daily against
a pinned ComfyUI core, plus manual dispatches; PR runs are excluded):

| key | meaning |
|---|---|
| `id`, `t`, `ev`, `c`, `d`, `s`, `a` | run id, start, event, conclusion, duration, status while running, attempt when > 1 |
| `sha`, `pin`, `ref`, `rs` | frontend commit, `COMFYUI_PIN` read from the workflow file at that commit (null until read; a read skipped by the budget or failed is repaired on a later tick, one request per sha), the core ref the run used (`pin` for scheduled runs; unknown for dispatches until phase 2), `pin` or `dispatch` |
| `rows`, `fs`, `final`, `jobs` | `rows[proof_row][shard]` job conclusions (`"0"` are the clean shards, other rows are detection proofs), failed step per non-green job, the verdict job's conclusion, `[matrix jobs, red jobs]` |
| head `pin`, `streak` | the newest pin; `{state, since, days, runs, capped}` over the clean rows (`capped` when the streak reaches the oldest stored run) |

Steady state costs 5-9 requests per tick (3 when nothing is new) against the
Actions token's 1,000 per hour. `FRONTEND_GH_TOKEN` (a fine-grained PAT
with Actions read on the frontend repo) is optional and reserved for phase 2
(dispatching the gate, reading run logs and artifacts); without it the
`GITHUB_TOKEN` is used. Local dry run, nothing pushed:
`python scripts/frontend_index.py --skip-publish --results-workdir /tmp/results`.

## One-time infrastructure setup

1. **Docker images**: `build-image.yml` pushes one image per entry in
   `manifest/lanes.json` `images` to `ghcr.io/comfy-org/comfyci-runpod-worker`
   using the built-in `GITHUB_TOKEN` (no registry secrets). After the first
   build, make the package public (repo → Packages → package settings → Change
   visibility) so RunPod can pull it without credentials, or give the endpoint
   a registry credential.
2. **RunPod**: create a Network Volume (≥200 GB, datacenter with 4090
   availability), then one serverless endpoint per lane from its image with
   the volume attached (see *Lanes*). Optionally set `HF_TOKEN` as an endpoint
   env var for gated models. The volume's size is `endpoint_defaults.volume_gb`
   in `manifest/lanes.json`: change it there when the volume grows, because
   the lint holds the manifest's model files to 90 % of it.
3. **Repo secrets** (this repo): `RUNPOD_API_KEY`, and optionally `HF_TOKEN`
   (gated models; sent with the sync job) and `PULSE_GH_TOKEN` (PR comments,
   see above). Every lane names its endpoint in `manifest/lanes.json`;
   `RUNPOD_ENDPOINT_ID` is only a fallback for the model sync when the primary
   lane has no `endpoint_id`. For the GCS phase, additionally
   `GCS_SERVICE_ACCOUNT_JSON` (write access to the CI bucket) and `GCS_BUCKET`
   (`comfy-ci-results`).
4. **Seed the volume**: run `sync-models.yml` (workflow_dispatch). The sync
   runs on the primary lane's endpoint (`scripts/sync_models.py`). Only one
   endpoint may write the network volume at a time: the workflow is a single
   concurrency group, so a sync queued while another runs waits for it, and
   never start a sync by other means (another endpoint, a pod) while one is
   running. GitHub keeps only one waiting run per group: a newer queued sync
   cancels an older waiting one. Each sync fetches every model of the
   manifest it checked out, so a later sync from main covers an earlier one,
   but a sync dispatched from a branch (`--ref <branch>`, to stage models
   before a merge) does not: run the sync from main again after it.
   Regression and golden jobs only read the volume and can run alongside a
   sync.
5. **First goldens**: run **Golden baselines** with the latest release tag,
   `lane: <lane id>` and `bless: true`.
6. Done — `regression.yml` now tests each new master commit on its own.
7. **(GCS phase) Wire into core CI**: add the job below to ComfyUI's
   `.github/workflows/test-ci.yml` plus the `RUNPOD_API_KEY` secret there, and
   retire the scheduled poller. Pass the lane (the action's default resolves
   to the retired layout-1 lane and is refused):

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
          lane: py313-torch2.14.1-cu130   # `primary` in manifest/lanes.json
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
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
  runs/<branch>/<commit>/attempt.json      # start marker, pushed before the GPU jobs
  latest/<branch>.json                     # previous-run pointer, written last
  manifest-snapshot/<commit>.json
  golden/<workflow_id>/<tag>/{outputs/, run_r1.json, run_r2.json, noise_floor.json, blessed.json}
  golden/<workflow_id>/current.json        # active blessed tag (+ reason, supersedes, output_sha256)
  golden/<workflow_id>/history.json        # every bless
  index/lanes.json                         # lane registry + per-branch latest + golden shas
  index/<branch>/<lane>.json               # newest 500 runs + first bad commit per workflow
  index/<branch>/<lane>/<YYYY-MM>.json     # complete monthly shards
frontend/                                  # ComfyUI_frontend CI index (see above)
  perf/head.json                           # perf metric registry, newest 100 baselines, latest comparison
  perf/<YYYY-MM>.json                      # complete monthly shards
  ci/main.json                             # column registry, newest 300 main commits, first_bad, coverage
  ci/main/<YYYY-MM>.json                   # complete monthly shards
  ci/custom-nodes.json                     # newest 90 custom-nodes gate runs, core pin, streak
```

Run outputs also carry 256-px WebP previews under `<workflow_id>/outputs/thumbs/`
for the run's job summary and the dashboard's list views.

A text workflow (`"output_kind": "text"`) publishes its text where an image
workflow publishes its PNGs:

```
runs/<branch>/<commit>/<workflow_id>/outputs/<workflow_id>_00001.txt   # the generated text
golden/<workflow_id>/<tag>/outputs/<workflow_id>_00001.txt             # the reference text
```

It has no `figures/` and no `outputs/thumbs/`. Its files differ from an image
workflow's in these keys, and an image workflow's files do not change at all:

- `comparison.json` and `blessed.json` end with `"output_kind": "text"`, and
  the workflow's cell in `summary.json` carries it too. An image workflow's
  files never name a kind.
- `vs_golden`, `vs_previous` and `noise_floor.json` carry `"kind": "text"`,
  `identical`, `similarity` (0 to 1) with its `similarity_basis`, the
  character, line and file counts of both sides (`files_empty_ref` too, only
  when a text file of the reference is empty), and `first_diff` when the
  texts differ: `file`, `line` and `col` (from 1), `offset` (how many
  characters the two share from the start), and `ref` and `cand`, the two
  lines there (model output: escape them). The noise floor of a golden
  promoted from a run has the counts of the promoted text itself; its
  `identical` and `similarity` are the earlier golden's, with
  `"measured": false`.
- `thresholds_used` is null, or `{"min_text_similarity": x}` when the entry
  sets one. `figures` is false and `thumbnail` is null.
- `output_sha256` is the sha256 of the text file as the worker wrote it (with
  several text files, the sha256 of their `<sha256>  <name>` lines). The cell,
  `blessed.json`, `current.json` and the golden in `index/lanes.json` all use
  that one rule, so approving a changed text closes its failing chain.
- A cell in the run index has `"k": "text"`; its `mse`, `psnr`, `pct` and
  `th` are null.

A smoke-level workflow (`"level": "smoke"`, see [Smoke level](#smoke-level))
publishes the same files, with new keys after the ones a reference-level
workflow has. A reference-level image workflow's files do not change at all:

- Its cell in `summary.json` and its `comparison.json` end with
  `"level": "smoke"` and `smoke` (after `output_kind`, when the kind is not
  image). `detail`, `vs_golden`, `vs_previous`, `thresholds_used`,
  `golden_tag` and the `previous_*` keys are null, and `figures` is false.
  A cell of a workflow that did not finish has the three lists of `smoke`
  empty.
- `smoke.files` lists the output files, at most 50: `name`, `type` (`image`,
  `text`, `audio`, `mesh`, `video`, or null for an extension that is not
  checked), `bytes` and `sha256` as the worker reported them, `sent: false`
  for a file the worker listed and did not send, `published: false` for a
  file that is not on the results branch, and what was measured. Image:
  `width`, `height`, `std`, `flat_share`. Text: `chars`, `words`. Audio:
  `sample_rate`, `channels`, `bits`, `seconds`, `bytes_per_second`,
  `pcm_md5`. Mesh: `vertices`, `faces`, `extent`. Video: `width`, `height`,
  `frames`, `seconds`, `bytes_per_frame_mp`.
- `smoke.problems` lists why the run does not pass, at most 20:
  `{"file": name or null, "check": id, "text": sentence}`. The check ids are
  `missing`, `decode`, `size`, `expect`, `blank`, `flat`, `empty`, `repeat`,
  `duration`, `silent`, `nonfinite`, `geometry` and `frames`. The verdict is
  `pass` when the workflow finished and this list is empty. `smoke.notes`
  holds sentences that are not failures. File names come from the worker:
  escape them, and the sentences that quote them.
- `output_sha256` is the sha256 of the main output: the first PNG by name
  for an image workflow, the text rule above for a text workflow, and for an
  audio workflow the first `.flac` by name that starts with the workflow id.
  `thumbnail` is set whenever the run wrote a PNG that can be read: one PNG
  that cannot does not cost the others their preview.
- **Which output files are published.** A smoke-level output need not repeat
  from run to run, and the results branch only stays small because equal
  files are stored once. So an output file of a smoke run is published when
  it is at most 1 MiB, or when it has the sha256 the same file name had in
  the workflow's previous result on the lane. A file that repeats is stored
  from its second run on, and one that never repeats is never stored. Any
  other file stays in the run's artifact, which GitHub keeps 90 days, and
  its entry says `"published": false`. `run.json`, `comparison.json`, the
  thumbnails and the measured facts are always published.
- A cell in the run index has `"l": "smoke"`, after `k`. Its `mse`, `psnr`,
  `pct`, `pv` and `g` are null, and `d` is `"inherited"` for a failure of a
  workflow whose last verdict was already `fail`.
- An open failing chain (`first_bad` in the index head) that a smoke cell
  started carries `"level": "smoke"`. A workflow that changes its level while
  it is red starts a new chain there, and the `prior_verdict` of a smoke cell
  looks through an earlier failure at reference level: a failed smoke check
  and an output that differs from a reference are not one failure. The
  lane's `goldens` in `index/lanes.json` hold no entry for a smoke workflow,
  and a chain is listed only for a workflow that is enabled on the lane.
- A summary of a lane with the `nightly` or `weekly` cadence can carry
  `published_new_bytes`: the bytes of the output files of its smoke-level
  workflows that the run published and the previous result had not (what a
  reference-level workflow published is not in it). Its report shows the
  number. It carries `ranges` when a workflow is red: per workflow id
  `{"prev", "commits_between", "compare_url"}`, the commits between the
  workflow's own last red or green result and this commit. A workflow on its
  first run has no entry. An index entry carries them as `wr`, per workflow
  `{"n", "url", "prev"}`, and a chain that started in such an entry carries
  `{"n", "url"}` as `range`. When a later run passed inside that range (the
  chain's `prev_good` is no longer the range's `prev`), the chain's `range`
  has `n` null and the compare link from `prev_good` to the chain's commit.
- A summary with `"partial": {"done": n, "of": m}` is one that a nightly or
  weekly run pushed before it had every result. The run's final summary
  replaces it and has no such key. Read one that has it as "this run did not
  finish": `n` workflows have a result (a workflow that the deadline did not
  reach has a cell and is not counted), and the run index has no entry for
  it.
- A cell with `"worker_status": "not_reached"` (verdict `infra_error`) is a
  workflow that a run did not start, because its job could still have been
  running at the run's deadline. It has `error`, a sentence, and no
  `run.json`. The `run.json` of a job that used up its whole time on a
  nightly or weekly lane has `"reason": "timed_out"`. A cell with
  `"tries": 2` is the result of the second job such a run gave the workflow
  at its end.
- `attempt` counts the lane's runs of the commit. A run that names its
  workflows, or that publishes nothing, is none: its summary records the
  attempts counted so far (at least 1). The report of such a run names no
  attempt, and says of an untested workflow on a per-commit lane that the run
  changes nothing for the cron.
- An index entry with `"named": true` (after `s`, before `wr`) is that of a
  summary with `scope`: a run that named its workflows at a commit without a
  full result. No other entry has the key, and the lane's full run of the
  commit takes it away. `latest`, in the index head and in the lane's
  `branches` in `index/lanes.json`, is the newest entry without it, and null
  when the lane has no such entry.

## Notes

- Goldens are valid **per lane** (GPU type and worker image torch/CUDA): after
  changing either, regenerate and re-bless (`blessed.json` records both, and
  every run record carries an `env` block with the image id, build sha,
  python, torch, CUDA, host driver and the kernel backends ComfyUI logged).
- Custom-node pins are part of the image too: a bump in
  `manifest/custom_nodes.json` rebuilds the baking images and moves the pack
  workflows' outputs, so re-bless them (`blessed.json` records the pins).
- Local dry run without publishing anything:
  `python scripts/run_regression.py --commit <sha> --branch test --skip-publish`
- Script tests (run on every PR by the *Script tests* workflow):
  `pip install -r scripts/requirements.txt pytest && python -m pytest scripts/tests`
- The results checkout is a blobless sparse clone: only the aggregate
  directories are materialised, run directories on first use.
- `run.json` `delay_s` is in seconds from summary `schema_version: 2` on;
  earlier records hold RunPod's raw millisecond value under the same key.
