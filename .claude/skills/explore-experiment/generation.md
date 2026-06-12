# Generation: Creating Visualizations

This module collects reference *recipes* for generating figures — not a required sequence. The inspect-viz pre-built views below are the quickest path when one fits; for bespoke figures, reach for matplotlib, seaborn, or plotly directly (the ROC / calibration overlays further down are already matplotlib). Draw on what fits the experiment.

## Setup

```python
import os
from inspect_viz import Data
from inspect_viz.plot import write_html
from inspect_viz.view import (
    scores_by_task,
    scores_heatmap,
    scores_radar_by_task,
    scores_radar_by_task_df,
    scores_by_model,
    scores_by_factor,
)
from cruijff_kit.tools.inspect.viz_helpers import sanitize_columns_for_viz

# Create output directory
output_dir = os.path.join(experiment_dir, "exploration")
os.makedirs(output_dir, exist_ok=True)

# IMPORTANT: sanitize column names before passing to inspect-viz.
# Metric names with "/" (e.g. risk_scorer_cruijff_kit/auc_score) cause
# DuckDB parse errors in inspect-viz. Use the sanitized df for all
# Data.from_dataframe() calls; use the original df for report generation.
viz_df = sanitize_columns_for_viz(logs_df)
data = Data.from_dataframe(viz_df)
```

## View Function Signatures

**Before using these functions**, run `help()` to verify the API hasn't changed:

```python
from inspect_viz.view import scores_by_task
help(scores_by_task)
```

The signatures below are reference examples — the actual function may have additional or renamed parameters.

### scores_by_task

Compare scores across multiple tasks or conditions.

```python
plot = scores_by_task(
    data,                           # Data object
    task_name='task_name',          # Column with task names
    score_value="score_match_accuracy",  # Score column
    score_stderr="score_match_stderr",   # Standard error column (optional)
    score_label="Match Accuracy",   # Label for y-axis
    ci=0.95                         # Confidence interval
)
```

### scores_heatmap

Model × task matrix visualization.

**When to skip:** If each model appears in only one task (1:1 mapping between
model and vis_label), the heatmap is just a diagonal — skip it. Check before
generating:

```python
# Only generate heatmap when multiple models share tasks
models_per_task = logs_df.groupby('task_name')['model'].nunique()
if models_per_task.max() > 1:
    plot = scores_heatmap(
        data,
        task_name='task_name',
        model_name="model_display_name",
        model_label="Model",
        score_value="score_match_accuracy",
        tip=True,
        title="",
        orientation="vertical"
    )
else:
    print("Skipping heatmap: each model maps to a single task (diagonal matrix)")
```

**Compound vis_labels (cross-evaluation experiments):**

When vis_labels encode both model and condition (e.g., `"original (acs_income_shuf_s1)"`),
`task_name` will be unique per row and the heatmap check above will see a diagonal.
To produce a useful heatmap, split the compound label into separate columns:

```python
# Split "original (acs_income_shuf_s1)" into training="original", eval_condition="acs_income_shuf_s1"
hm_df = df.copy()
hm_df['training'] = hm_df['task_name'].apply(lambda x: x.split(' (')[0] if ' (' in x else x)
hm_df['eval_condition'] = hm_df['task_name'].apply(
    lambda x: x.split('(')[1].rstrip(')') if '(' in x else x
)

# Now check for heatmap eligibility using the split columns
models_per_condition = hm_df.groupby('eval_condition')['training'].nunique()
if models_per_condition.max() > 1:
    viz_hm = sanitize_columns_for_viz(hm_df)
    data_hm = Data.from_dataframe(viz_hm)
    plot = scores_heatmap(
        data_hm,
        task_name='eval_condition',
        model_name='training',
        model_label='Training',
        score_value='score_match_accuracy',
        tip=True,
        title='',
        orientation='vertical'
    )
```

### scores_radar_by_task

Radar plot for multiple metrics comparison.

```python
# First prepare the data
radar_df = scores_radar_by_task_df(
    logs_df,                        # Original dataframe (not Data object)
    invert=["score_match_accuracy", "score_includes_accuracy"],  # Scores to include
    normalization="min_max",        # "min_max", "percentile", or "absolute"
    domain=(0, 1)                   # Domain for normalization
)

# Then create the plot
plot = scores_radar_by_task(Data.from_dataframe(radar_df))
```

### scores_by_factor

Compare binary factor effects.

```python
plot = scores_by_factor(
    data,
    factor="prompt_type",           # Boolean column
    factor_labels=("No Prompt", "Prompt"),  # Labels for False, True
    score_value="score_match_accuracy",
    score_stderr="score_match_stderr",
    score_label="Match Accuracy",
    model="model",                  # Column for grouping by model
    model_label="Model",
    ci=0.95
)
```

### scores_by_model

Compare models directly. **Requires single-task experiments only.**

```python
plot = scores_by_model(
    data,
    score_value="score_match_accuracy",
    score_stderr="score_match_stderr",
    score_label="Match Accuracy",
    ci=0.95,
    height=200                      # Plot height in pixels
)
```

**Note:** This view will fail if the data contains multiple tasks. For multi-task experiments, use `scores_heatmap` or `scores_by_task` instead.

## Output File Naming

Name files for what they show — `{content}_{metric}.png` / `.html` (e.g. `auc_by_condition.png`, `scores_by_task_match.html`) — so the reader knows the figure from its filename.

## Dynamic Metric Detection

Instead of hardcoding metrics, detect them from the dataframe:

```python
from cruijff_kit.tools.inspect.viz_helpers import detect_metrics, display_name

# Automatically detect available metrics
detected = detect_metrics(logs_df)
# detected.accuracy -> e.g., ['match', 'includes']
# detected.supplementary -> e.g., ['risk_scorer_cruijff_kit/ece', ...]

# Generate plots for accuracy metrics
for metric in detected.accuracy:
    score_col = f"score_{metric}_accuracy"
    # ... create accuracy plot

# Generate plots for supplementary metrics (calibration, risk)
# NOTE: use sanitized column names (/ -> __) for inspect-viz score_value,
# but the original names for display_name() lookups.
for metric in detected.supplementary:
    score_col = f"score_{metric}".replace("/", "__")  # sanitized for viz
    label = display_name(metric)  # "ECE", "Brier Score", "AUC", etc.
    # These work with any view that accepts score_value parameter
    plot = scores_by_task(data, task_name='task_name', score_value=score_col, score_label=label)
```

## PNG Export

In addition to HTML, export PNG snapshots for reports and presentations:

```python
from inspect_viz.plot import write_html, write_png

# Always save HTML
write_html(html_path, plot)
print(f"  HTML saved: {html_path}")

# Auto-detect playwright and export PNG if available
try:
    from playwright.sync_api import sync_playwright
    write_png(png_path, plot)
    print(f"  PNG saved: {png_path}")
except ImportError:
    print("  PNG skipped: playwright not installed (pip install playwright && playwright install chromium)")
```

**Note:** PNG export requires playwright with chromium. If not installed, PNG generation is skipped with a warning.

## Error Handling

**If view function fails:**
- Log error with traceback
- Continue with other views
- Report failed views in summary

**If output directory is not writable:**
- Log error
- Suggest alternative location
- Do not proceed with generation

**If data is missing required columns:**
- Log which columns are missing
- Skip views that require those columns
- Report in summary

## Per-Sample Risk Plots (ROC & Calibration)

When the experiment used `risk_scorer`, generate overlay plots from per-sample data. These are **matplotlib PNGs** (not inspect-viz HTML), gated on `detected.has_risk_scorer`:

```python
from cruijff_kit.tools.inspect.viz_helpers import (
    extract_per_sample_risk_data, generate_roc_overlay,
    generate_calibration_overlay, generate_prediction_histogram,
)

if detected.has_risk_scorer:
    risk_data = extract_per_sample_risk_data(kept)  # kept = deduplicated .eval paths
    if risk_data:
        roc_path = generate_roc_overlay(risk_data, f"{output_dir}/roc_curves.png")
        cal_path = generate_calibration_overlay(risk_data, f"{output_dir}/calibration_curves.png")
        hist_path = generate_prediction_histogram(risk_data, f"{output_dir}/prediction_histogram.png")
        if roc_path:
            generated_pngs.append(str(roc_path))
        if cal_path:
            generated_pngs.append(str(cal_path))
        if hist_path:
            generated_pngs.append(str(hist_path))
```

**Performance note:** `extract_per_sample_risk_data()` reads full eval logs (all samples), so it's slower than the aggregate `evals_df_prep()` path. For a 5000-sample eval this typically takes a few seconds per file.

**Skipped models:** Models with <2 valid samples or only one class (e.g., zeroshot models that always predict the same thing) are silently skipped. The function logs which models were skipped at INFO level.

## Figure craft

A few habits that make a figure earn its place — guidance, not mandates:

- **Show the mechanism, not just the metric.** ROC and calibration curves re-express AUC — they show a metric a second way. Often more illuminating is the raw quantity the metrics are *computed from*: a per-cell distribution of the model's predicted score (e.g. a small-multiples grid of P(1) histograms across an ordinal/condition axis). One such view can carry two metrics at once — where the mass sits is the *bias* (mean prediction), how wide it is is the *discrimination* (a collapsed, narrow distribution has no spread left to rank with, so AUC falls toward chance). When two metrics share a cause, prefer the figure that shows the cause over a second figure that restates one of them.
- **Use color to carry a dimension, not just to decorate.** A two-color palette leaves an encoding channel on the table. When a figure spans an ordinal axis (ladder rungs, k, epochs), map that axis to a sequential colormap (e.g. `viridis`) so the reader sees the ordering in the color itself; reserve categorical palettes for unordered groups (models). A richer, deliberate palette reads as more legible, not busier — lean into color rather than defaulting to the sparest set that technically distinguishes the lines.

## Interpretation (Required)

**Required.** The deterministic backbone is summarize's job; interpretation is yours. Do a **hypothesis-first re-read** of the data — not a restatement of the tables. Open with a jargon-free **Bottom line**, then write five rigorous sections, each **mandatory or an explicit "n/a — reason"** (no silent skips). The bar: a canned summary of `length_sweep_nth14` once just restated the saturation table and lifted next-steps from the design doc; a hypothesis-first re-read of the *same numbers* surfaced a monotonicity violation, a relative-target-position confound, and a prose-vs-table self-contradiction it had missed. Clear that bar — and clear it *readably* (see Audience, below).

### Describe, don't just adjudicate

Your job is to **characterize the shape of the result** the way a sharp, creative data scientist would — lead with the phenomenon and its mechanism, and cite numbers as the evidence *for* a described pattern, not as entries in a scorecard. The pre-registered hypotheses are the expectations you measure reality against; they are an *input* to the description, not its skeleton. Two consequences:

- **The most interesting pattern is often the one nobody pre-registered — chase it.** A rank inversion, a non-monotone valley, a single cell that behaves unlike its neighbors: these rarely map onto a tidy H1/H2, and they are usually where the real finding lives. Give them room even though no hypothesis named them.
- **Keep the discipline; demote its bookkeeping.** Still run the adversarial pre-pass (below) and still reach a verdict on every pre-registered claim — that honesty check is non-negotiable (Principle #1, and it is what stops description from sliding into pleasant story-telling about noise). But the verdict is the *first word of a described paragraph*, not the whole entry, and the scorecard view belongs inline or in the audit log, never as the report's organizing spine. You may *optionally* draft a short free-form "pattern description" passage high in the report, unconstrained by the section template, to set the scene before the structured sections.

### Audience: lead with the conclusion, make the rigor optional

Write so a curious non-specialist — a student new to the project, a collaborator from another field — can read the **top of the report and come away with the real finding**, not a teaser for it. The rigorous sections stay (rigor is cheap to keep and it's what makes explore trustworthy), but *reading and understanding all of them must not be the price of admission*. Structure for progressive disclosure:

- **Open with a "Bottom line"** — 3–5 jargon-free sentences: what was tested, what you actually found, and the surprises. Someone who stops reading here should still have the correct takeaway. It goes *above* the data tables, not after the adjudication.
- **Define each metric the first time it appears**, in one clause. "AUC — how well the model *ranks* high earners above low earners; 0.5 is a coin flip" costs a line and unlocks the whole table. Don't assume argmax / Brier / ECE / AUC are known.
- **Point at the figure that carries the story.** A reader with the headline plot and the bottom line has the result; the prose is the escort, not the gate.
- **Tables are reference, not the reading path.** A table is welcome *alongside* the prose — keep it small and scannable, for at-a-glance lookup. It must never be where the finding lives: every number and verdict in it is also stated in the narrative, so a reader who skips the table loses nothing but convenience. No dense "here's the whole grid, parse it yourself" dump as the focal artifact.
- **Demote bookkeeping.** The self-consistency audit (section 4) is for auditors, not readers — put its output in an appendix or the audit log (`explore-experiment.log`), not the main reading path. Same for compute utilization.
- **Voice.** Write as a smart, creative data scientist describing patterns to peers: phenomenon-first, connected narrative prose, not a bulleted list of findings. Be vivid and precise; name the strangest cell and chase it. Favor short declarative sentences and commit to each call rather than hedging. Concision serves warmth: trim every sentence standing between the reader and the finding.

The five sections still happen in full — they're the *depth* layer a skeptical reader descends into to verify the bottom line, not the layer a newcomer must climb to reach it.

### The five sections

1. **What the data does, vs. what we expected.** Decompose `experiment.hypothesis` into individual falsifiable claims and **walk through them one at a time, each as its own short paragraph** — describe the phenomenon and its mechanism first, then say where it met or broke the expectation. Anchor each paragraph with a one-word verdict (**Confirmed / Violated / Inconclusive**) and cite cell-level evidence, but let the description carry the weight (a verdict table may accompany the prose but must not replace it — see Audience). If the hypothesis is absent, infer predictions from `experiment.question` + `variables` and flag that you're doing so.
2. **Cross-cell pattern audit.** Scan the results grid — but split the work by what a script can honestly do:
   - **Saturation cells** (accuracy ≥0.95 default) are pure counting: structure-free, true for every experiment. This is summarize's to compute; `summary.md` carries the `✓` flags, so read them from there rather than recomputing.
   - **Base-rate floor** is *not* structure-free. The arithmetic is — the eval-set class balance, which summarize reports as provenance — but whether that split is the *meaningful* baseline depends on intent (a balanced split's 50/50 says nothing; the truer floor in a fine-tuning experiment is usually the base-model eval). Read the class balance from `summary.md`; deciding what the floor actually is stays your judgment, made here.
   - **Monotonicity** (does a metric move one direction as an *ordinal* variable climbs?) and **equivalent-cell agreement** (do cells the design intends to be the same actually agree?) depend on knowing your experiment's shape — which variables are ordered, which cells are meant to match. A generic script would have to guess that, so they stay judgment calls: make them here. (Monotonicity could go deterministic only if the design declares its ordinal variables — a separate schema project, not assumed here.)
3. **Mechanistic interpretation.** What does each variable's variation actually test? Give a parsimonious explanation for every surprise, and **name confounds explicitly** (e.g. "k was varied, but that also moved the target's *relative* position — those are entangled"). Where a figure makes a mechanism visible, **tie the visual feature to the scalar it explains** ("the mass collapsing to a spike *is* the AUC drop") rather than describing the figure and the number in separate breaths.
4. **Self-consistency audit.** Check every numerical claim in your prose against the source table, literally: `claim → table value`. Catch prose that says "saturates at N=500" while its own table shows the ≥0.95 cutoff at N=100. **Do the check, but put its output in an appendix or the audit log** — it's verification bookkeeping, not part of the reader's path to the finding (see Audience).
5. **Calibrated next steps.** Each item names what to vary / what mechanism it tests / what outcome is informative. "The design didn't actually test X — fix the design first" is a valid and valuable item.

### Adversarial pre-pass

Before you read the results, state — from the hypothesis alone — what cell-level evidence *would* falsify each claim. Then check whether that evidence appears. Doing this first, not after, is what keeps the adjudication honest instead of post-hoc.

### Worked shape (hypothesis-first)

```markdown
## Bottom line
We checked whether the model stays accurate as the input gets longer (more items, "k").
It mostly does — except in a surprising dip around the middle lengths, where it does *worse*
than at either extreme. That dip turned out to be an artifact of *where the answer sits* in
the sequence, not the length itself. Plain-language takeaway: "longer = harder" is wrong here;
"answer-in-the-middle = harder" is the real pattern. See `accuracy_by_k.png`.

### What the data does, vs. what we expected
- **Violated.** Accuracy is *not* monotonic in k. At N=25 and N=100 the hardest cell is
  k=30, not k=100 — the curve sags in the middle and recovers, the opposite of the
  "more items, harder" story we expected. (Claim under test: "accuracy is monotonic in k.")
- **Inconclusive / mis-stated.** The "k=15 saturates by N=500" claim can't be adjudicated as
  written — see self-consistency; its own table puts saturation at N=100.

### Cross-cell pattern audit
- Monotonicity (k, per N): violated at every sub-saturation N (valley at k=30).
- Saturation (≥0.95): k=15 hits 0.996 at N=100 — saturation is N=100, not N=500.

### Mechanistic interpretation
- x=14 lands ~47% through the sequence at k=30 — nearest the middle. Difficulty tracks
  *relative* target position, not absolute length. **Confound:** varying k implicitly
  varied relative target position; they are entangled in this design.

### Calibrated next steps
1. Hold relative target position fixed while varying k — separates length from position.
   Informative if the k=30 valley disappears.

---
#### Appendix: self-consistency audit
- Prose "k=15 saturates at N=500" contradicts its own table (k=15 = 0.996 at N=100). Fix.
```

Only omit interpretation if the user explicitly asks for plots-only (e.g., "just generate the plots, no analysis").

## Report Generation

**You author `exploration/report.md` directly** — there is no report-assembling
function. Write the markdown yourself, the same way you'd write `summary.md`:
prose, tables, and figures you compose, not a skeleton a generator fills in. This
is what keeps the report reading like prose a person wrote, and keeps the
deterministic backbone in one place (`summary.md`).

### Don't re-render the deterministic tables — cite them

`summary.md` (from summarize-experiment) is the **single source of truth** for the
canonical metric tables (run status, accuracy/AUC/Brier/ECE, training loss). Do
**not** rebuild those tables in `report.md`. Reference them — `see ../summary.md`
— and pull only the specific numbers your argument turns on, quoting them inline
("3B p5_noise reaches 0.731 AUC vs the 0.537 baseline"). One source of truth, no
drift between the two files.

### Shape (lead with the conclusion — see Audience, above)

Compose `report.md` as:

1. `# Claude's Exploration` + a one-line metadata header (experiment, date, models
   evaluated, **Generated by:** *<your actual model name>*).
2. **Bottom line** — 3–5 jargon-free sentences (above any data).
3. The five interpretation sections (defining each metric on first use).
4. **Figures embedded inline** at the point you discuss them — `![descriptive
   caption](figure.png)` with a real caption, *not* a bare filename and *not* a
   dumped "Visualizations" appendix at the end. Use the bare filename (relative
   path) so the PDF converter resolves it.
5. **Demote bookkeeping** — the self-consistency audit and compute utilization go
   in an appendix or the audit log, not the main reading path.
6. **Provenance footer** — attribution plus a *summarized* pointer to the source
   logs (a count + common directory or glob, e.g. "10 .eval logs under
   `{1B,3B}-Instruct_base/eval/acs_income_p*/logs/`"), kept in a collapsible
   `<details>` block. Do **not** enumerate every absolute path — that path-dump
   verbosity overflows the PDF with long lines and buries the provenance that
   matters.

### Track which figures you embed

Keep a list of the PNGs you generated *this run* and embed only those — never
stale outputs from a previous session. (You're embedding them inline by hand, so
this is just discipline about which filenames you reference.)

### If a metric you want isn't in `summary.md`

Compute it from the eval data you already loaded (`evals_df_prep` /
`extract_per_sample_risk_data`, the viz helpers) and state how you derived it.
Don't hand-transcribe numbers you haven't verified.

## Compute Utilization Analysis

After generating visualizations and before the report, add compute metrics. This requires run logs from `run-experiment` (or from `src/tools/run/submit_*.py` directly).

**Workflow:**

1. Call `harvest_jids_from_run_logs(experiment_dir)` from `src/tools/slurm/compute_gpu_metrics.py`. It returns `(jids_dict, warnings)` where `jids_dict` is `{"finetune": [(name, jid), ...], "eval": [(name, jid), ...]}`. The helper already prints `WARNING:` lines to stderr for any missing or malformed log file, mirroring the canonical regexes in `run-experiment/logging.md`. **If `warnings` is non-empty, do not silently skip the section** — append a "**Compute Utilization unavailable:** ..." note to `report.md` listing each warning so the absence is visible to the operator.
2. Check if jobstats is available with `check_jobstats_available()`
3. For each job:
   a. Run `seff {job_id}` and parse with `parse_seff_output()`. If `time_limit` is None (some clusters omit it), run `sacct -j {job_id} --format=Timelimit -P -n` and parse with `parse_sacct_time_limit()`.
   b. Read `gpu_metrics.csv` with `summarize_gpu_metrics()`. **Paths differ by job type:**
      - Fine-tuning: `{output_dir}/{run}/artifacts/gpu_metrics.csv`
      - Evaluation of a fine-tuned checkpoint: `{output_dir}/{run}/artifacts/epoch_{N}/gpu_metrics.csv`
      - Base/control eval (no epoch): `{output_dir}/{run}/artifacts/gpu_metrics.csv`
   c. If jobstats available: run `run_jobstats(job_id)` for CPU metrics (JSON) and `run_jobstats(job_id, json_mode=False)` for notes. Parse with `parse_jobstats_json()` and `extract_jobstats_notes()`.
4. Build job dicts combining all sources:
   - **CPU**: from jobstats (`cpu_efficiency_pct`, `cpu_mem_used_gb`, `cpu_mem_allocated_gb`), or seff `cpu_efficiency` as fallback
   - **GPU utilization**: dual-source — set `gpu_util_jobstats_pct` from `parse_jobstats_json()["gpu_util_pct"]` (Prometheus average), and `gpu_util_min`/`gpu_util_max` from `summarize_gpu_metrics()` (nvidia-smi range). `format_compute_table` renders this as `avg% (min–max%)` when both are present.
   - **GPU memory / power**: from nvidia-smi CSV (`gpu_mem_used_mean_gb`, `gpu_mem_total_gb`, `power_mean_w`)
   - **Throughput**: call `enrich_job_with_throughput(job, slurm_out_path)` from `src/tools/slurm/throughput_parsers.py`. For fine-tunes, the slurm-out lives at `{output_dir}/{run}/artifacts/slurm-{job_id}.out`; for evals, at `{output_dir}/{run}/artifacts/epoch_{N}/slurm-{job_id}.out`. The helper adds `tps_gpu_train_mean` (finetune) or `tps_gpu_eval_e2e` + `total_tokens` (eval) to the job dict. On parse failure it warns to stderr and leaves the dict unchanged — downstream `estimate_compute` will raise a clear error if it later tries to scale from a job that lacks tps fields.
5. Format with `format_compute_table(jobs, recommendations=recs)` → markdown table with optional recommendations
6. Build and save compute_metrics.json using `compute_summary.py`:
   ```python
   from cruijff_kit.tools.slurm.compute_summary import build_summary, save_summary

   summary = build_summary(
       jobs=jobs,  # list of job metric dicts from steps 3-4
       experiment_summary_path=os.path.join(experiment_dir, "experiment_summary.yaml"),
   )
   save_summary(summary, os.path.join(experiment_dir, "exploration", "compute_metrics.json"))
   ```
   `build_summary()` reads `experiment_summary.yaml` to extract metadata (model, dataset_size, epochs, batch_size, date) and wraps the job list in the summary format. `save_summary()` writes the JSON file.
7. Write the `format_compute_table` output into `report.md` yourself — as an
   appendix or demoted section, not the main reading path (see Audience). If
   `harvest_jids_from_run_logs` returned warnings, surface a "**Compute
   Utilization unavailable:** ..." note instead of silently skipping.

**Key functions** from `tools.slurm.compute_gpu_metrics`:
- `check_jobstats_available() -> bool` — auto-detect jobstats on PATH
- `run_jobstats(job_id, json_mode=True) -> dict | str | None` — run jobstats (JSON or text mode), 30s timeout
- `parse_jobstats_json(js_data: dict) -> dict` — CPU metrics (cores, efficiency, memory) and GPU metrics (`gpu_util_pct`, `gpu_mem_used_gb`, `gpu_mem_total_gb`) from jobstats JSON
- `extract_jobstats_notes(text: str) -> list[str]` — actionable recommendations from jobstats text output
- `parse_seff_output(stdout: str) -> dict` — wall time, time limit, memory, CPU efficiency (fallback)
- `parse_sacct_time_limit(stdout: str) -> str | None` — fallback for time limit when seff omits it
- `summarize_gpu_metrics(csv_path: Path) -> dict` — GPU util (mean, min, max), memory, power from nvidia-smi CSV
- `generate_gpu_recommendations(jobs) -> dict[str, list[str]]` — GPU memory optimization suggestions for finetune jobs
- `format_compute_table(jobs, recommendations=None) -> str` — markdown table with CPU + GPU columns. GPU Util shows `avg% (min–max%)` when `gpu_util_jobstats_pct` and `gpu_util_min`/`gpu_util_max` are both present in the job dict.

**Data sources:** nvidia-smi CSV for GPU memory/power and utilization range (min/max), jobstats for CPU metrics and GPU utilization average (Prometheus-sampled, more reliable than nvidia-smi's 30s polling), seff for job metadata + CPU fallback.

**Error handling:** Missing seff → skip seff columns; missing gpu_metrics.csv → show `-` for GPU columns; jobstats unavailable or fails → fall back to seff for CPU data; missing or malformed run logs → emit a loud `WARNING:` to stderr (handled inside `harvest_jids_from_run_logs()`) and surface a "*Compute Utilization unavailable: ...*" note in `report.md` (do not silently skip — see issue #451); partial data → generate table with whatever is available.

## Logging

Log generation actions using `GENERATE_PLOT`, `GENERATE_REPORT`, and `COMPUTE_METRICS` action types. See `logging.md` for format specification.
