---
name: explore-experiment
description: Open-ended, Claude-driven exploration of a completed experiment — you decide per-experiment which figures matter and what the results mean. Optional and runnable any time after run-experiment (summarize-experiment is the required post-run step). Produces "Claude's Exploration": a non-deterministic report of what you examined and concluded, with an audit log.
---

# Explore Experiment

You are Claude, exploring a completed experiment. **You** decide — for this specific experiment — which figures are worth making and what the results mean. There is no fixed plot menu: the value of this step is your per-experiment judgment, not a deterministic pipeline.

The artifact you produce is **"Claude's Exploration"** — a deliberately non-deterministic record of what you looked at and concluded *this time*. The human usually does not intervene while you work, so the name is honest about what this is: your exploration, not a reproducible deterministic report. To keep it auditable anyway (Principle #1: Scientific), you log what you did.

## The summarize / explore boundary

`summarize-experiment` is the required, script-driven, deterministic backbone — the always-correct facts that hold for *every* experiment (run status, loss, accuracy with CIs, provenance including eval-set class balance, and the structure-free cross-cell fact of saturation). **Read its `summary.md` first.**

> **If something is meaningful for *every* experiment, it belongs to `summarize`. Everything else is yours.**

You *interpret* the facts summarize establishes — which figures matter here, what the results mean, what to do next. You do not recompute the backbone.

## Your Task

1. Read `summary.md` (from summarize-experiment) and `experiment_summary.yaml` — understand the question / hypothesis and the deterministic facts summarize already established.
2. Decide what, if anything, is worth visualizing for *this* experiment (see "Choose what to show" — it is a toolkit, not a checklist).
3. Interpret the results to the bar in "Interpret — clear the bar."
4. Write "Claude's Exploration" (`exploration/report.md`) and an audit log (`logs/explore-experiment.log`).

## Prerequisites

- experiment_summary.yaml exists (from design-experiment)
- Evaluations complete (from run-experiment)
- Plotting toolkit installed — inspect-viz, matplotlib, seaborn, plotly are all bundled by `pip install -e .`
- Conda environment activated (from claude.local.md)
- **Optional:** playwright for PNG export (`pip install playwright && playwright install chromium`)

## Dependency version check

Run before proceeding to catch stale envs (user pulled new pins but didn't re-run `pip install -e .`):

```bash
python scripts/check_env.py
```

- **Exit 0**: proceed.
- **Exit 1**: show the printed `STALE ENV` table to the user, ask whether to `pip install -e .` first or continue anyway.

## Workflow

### 1. Locate Experiment → `parsing.md`

Find experiment directory and parse experiment_summary.yaml for:
- Experimental variables (model, task, factors)
- Run structure and naming conventions
- Evaluation matrix

**See `parsing.md` for:**
- How to find the experiment directory
- YAML parsing logic
- Extracting variable types for inference

### 2. Load Evaluation Logs → `data_loading.md`

Use helper functions from `src/tools/inspect/viz_helpers.py`:
- `deduplicate_eval_files()` - Remove duplicate evals (keeps most recent per model+epoch)
- `evals_df_prep()` - Prepare eval-level dataframes
- `parse_eval_metadata()` - Extract epoch/is_finetuned/source_model from JSON metadata
- `detect_metrics()` - Dynamically detect available score columns

**See `data_loading.md` for:**
- Automatic deduplication of re-run evaluations
- Generic metadata extraction using vis_label and JSON metadata
- How to construct subdirs list from experiment_summary.yaml
- Preparing data for each view type

### 3. Choose what to show — a toolkit, not a menu

There is no fixed plot menu. Decide what, if anything, is worth showing for *this* experiment. Generating **zero** plots is a valid outcome if `summary.md` already tells the story; so is a bespoke figure no pre-built view covers. But when a result has any shape worth seeing, lean toward showing it — a figure the reader can scan beats a sentence describing one, and a near-empty report under-serves them. Err toward the figure when in doubt. Two craft habits worth applying as you choose (detail in `generation.md` → "Figure craft"): prefer a figure that shows the *mechanism* — the raw quantity the metrics are computed from — over one that re-expresses a metric you already have; and use color to carry a dimension (map an ordinal axis to a sequential colormap) rather than defaulting to the sparest palette.

The environment ships four plotting tools — peers chosen by fit, not a ranking:

| Tool | Reach for it when… | Output |
|------|---------------------|--------|
| **inspect-viz** | a pre-built eval view fits as-is (scores by task / model / factor, heatmap, radar) | interactive HTML |
| **matplotlib** | you want full control of a static figure (already how the ROC / calibration / histogram overlays are built) | PNG |
| **seaborn** | statistical cuts with good defaults — distributions, regressions, annotated heatmaps | PNG (matplotlib-backed) |
| **plotly** | you want custom interactivity a pre-built inspect-viz view can't give | interactive HTML |

`pandas` (`df.plot`, `df.style`), `scipy`, and `scikit-learn` are also on hand for quick cuts and the stats beneath the plots. Nothing here is mandatory or exhaustive — if something else in the environment fits the question better, use it.

### 4. Generate the figures → `generation.md`

Build whatever you chose. `generation.md` holds reference *recipes* — not a required sequence. A bespoke matplotlib / seaborn / plotly figure written from scratch is as valid as a pre-built view.

**See `generation.md` for:**
- inspect-viz pre-built view signatures and parameters
- matplotlib overlay helpers (ROC, calibration, prediction histogram) for `risk_scorer` evals
- Dynamic metric detection, PNG export, and output-file naming

### 5. Interpret — clear the bar → `generation.md`

**Required.** Interpretation is the heart of explore: a hypothesis-first re-read of the data, not a restatement of the tables. Write five sections, each **mandatory or an explicit "n/a — reason"** (no silent skips):

1. **What the data does, vs. what we expected** — decompose `experiment.hypothesis` into individual falsifiable claims and walk through them one at a time, each as its own short paragraph: describe the phenomenon and mechanism first, anchor with a one-word verdict (Confirmed / Violated / Inconclusive) plus cell-level evidence. A compact table is fine as at-a-glance reference, but the prose must stand alone (repeat every number/verdict in the narrative) — tables are reference, never the focal artifact. Lead with description; chase the pattern nobody pre-registered. If there is no hypothesis, infer predictions from `experiment.question` + `variables` and flag the gap. (See `generation.md` → "Describe, don't just adjudicate".)
2. **Cross-cell pattern audit** — scan the results grid, split by what a script can honestly do:
   - **Saturation** (accuracy ≥0.95) is pure counting — structure-free, true for every experiment. This is summarize's to compute; `summary.md` carries it, so read it from there instead of recomputing.
   - **Base-rate floor** is *not* structure-free. The arithmetic is — the eval-set class balance, which summarize reports as provenance — but whether that split is the *meaningful* baseline depends on intent. A deliberately balanced split's 50/50 says nothing; the truer floor in a fine-tuning experiment is usually the base-model eval, not label prevalence. Reading the class balance is free; deciding what the floor actually is stays your judgment, made here.
   - **Monotonicity** (does a metric move one direction as an *ordinal* variable climbs?) and **equivalent-cell agreement** (do cells the design means to be the same actually agree?) depend on knowing your experiment's shape — which variables are ordered, which cells are meant to match. A generic script would have to guess that, so they stay your judgment: make the calls here.
3. **Mechanistic interpretation** — what each variable's variation actually tests; a parsimonious explanation for each surprise; name confounds explicitly.
4. **Self-consistency audit** — check every numerical claim in your prose against the source table (`claim → table value`), literally, not interpretively.
5. **Calibrated next steps** — each specifies what to vary / what mechanism it tests / what outcome is informative. "The design didn't actually test X — fix the design first" is a valid item.

**Adversarial pre-pass:** before reading the data, state from the hypothesis alone what cell-level evidence *would* falsify each claim — then check whether it appears. This guards against post-hoc rationalization.

Record what you concluded — and why each figure was worth making — in the audit log (`logs/explore-experiment.log`). Only omit interpretation if the user explicitly asks for plots-only.

### 6. Write Report → `generation.md`

**Author `exploration/report.md` yourself** — there is no report-assembling
function. Compose the markdown directly (prose, inline figures, provenance), the
way you wrote `summary.md`:
1. Open with a jargon-free **Bottom line**, then the five interpretation sections.
2. **Cite `../summary.md`** for the canonical metric tables — don't re-render
   them; quote only the specific numbers your argument turns on.
3. Embed figures **inline with descriptive captions** at the point you discuss
   them (no dumped "Visualizations" appendix).
4. Close with a **summarized** provenance footer (count + glob, not every
   absolute path) in a collapsible `<details>` block.

See `generation.md` → "Report Generation" for the full shape and rationale.

### 6b. Compute Utilization Report → `generation.md`

Add a compute-utilization appendix. Call `harvest_jids_from_run_logs(experiment_directory)` (from `src/tools/slurm/compute_gpu_metrics.py`) — it returns `(jids_dict, warnings)`:

- **If `warnings` is non-empty**, surface a visible "**Compute Utilization unavailable:** ..." note in `report.md` (one line per warning) — never silently skip the section (issue #451). The warning text tells the operator how to recover (re-run `run-experiment`, or rebuild the log via `src/tools/run/submit_*.py`).
- **If JIDs are present**, gather per-job metrics (`seff`, `summarize_gpu_metrics()`), format with `format_compute_table()`, save `exploration/compute_metrics.json`, and write the table into `report.md` yourself as a demoted appendix.

Full recipe (paths, helpers, dual-source GPU util) in `generation.md` → "Compute Utilization Analysis".

### 7. Logging → `logging.md`

Document the process in `{experiment_directory}/logs/explore-experiment.log`. This log is what keeps a deliberately non-deterministic step auditable (Principle #1): record which data you read, which figures you chose to make and **why**, and what you concluded.

**See `logging.md` for:**
- Plain text format specification
- Action types (`LOAD_EVALS`, `CHOOSE_FIGURE`, `GENERATE_PLOT`, `RECORD_INTERPRETATION`, …)
- Example log entries

---

## inspect-viz pre-built views (one tool's catalog)

These are the ready-made views **inspect-viz** offers — convenient when one fits as-is, but only one of the four tools in "Choose what to show." For anything they don't cover, drop to matplotlib / seaborn / plotly.

| View | Use Case | Required Data |
|------|----------|---------------|
| `scores_by_task` | Multiple tasks/conditions | task_name column |
| `scores_heatmap` | Model × task matrix | model + task columns |
| `scores_radar_by_task` | Multiple metrics | Multiple score columns |
| `scores_by_factor` | Binary factors | Boolean factor column |
| `scores_by_model` | Cross-model comparison (single task only) | model column, single task |

**Note:** `scores_by_model` requires a single-task experiment. For multi-task experiments, use `scores_by_task`, `scores_heatmap`, or `scores_radar_by_task` instead.

## User Questions

### Existing Exploration Outputs

If `exploration/` directory exists with files, **ask the user**:

```
Found existing exploration outputs in exploration/. What would you like to do?

1. Keep existing files, add new outputs (Recommended)
2. Clean exploration directory first
```

If user chooses option 2, delete contents of `exploration/` before generating new outputs.

### Defaults — apply these, don't ask

- **Figures:** you choose them (no fixed menu); embed only the ones you generated *this run*, never stale outputs from a prior session.
- **Deduplication:** automatic (keep most recent per model+epoch).
- **Epochs:** latest only, unless the user asks for all.
- **Metrics:** plot all detected (don't hardcode).
- **PNG:** auto-detect playwright; export if present, skip with a warning if not.

## Output Structure

After running, the experiment directory will contain:

```
{experiment_directory}/
├── exploration/
│   ├── report.md             # Claude's Exploration (you author this)
│   ├── <your figures>.png    # whichever figures you chose to make
│   └── compute_metrics.json  # if compute logs were available
├── logs/
│   └── explore-experiment.log
└── experiment_summary.yaml
```

The figure set is **not** fixed — `report.md` and the audit log are the only guaranteed outputs; everything else depends on what you chose to show.

## Error Handling

**If experiment_summary.yaml not found:**
- Report error to user
- Suggest running design-experiment skill first
- Do not proceed

**If no evaluation logs found:**
- Report error to user
- Suggest running run-experiment skill first
- Do not proceed

**If a figure fails to generate:**
- Log error details in explore-experiment.log
- Continue with the remaining figures
- Report which ones failed in your summary

## Validation Before Completion

Before reporting success, confirm `report.md` and `logs/explore-experiment.log` exist, and that every figure you reference inline was actually generated this run.

## Output Summary

When done, briefly tell the user what you examined and concluded, which figures you made (by their *actual* filenames), and where the audit log is. **End with the full absolute path to `report.md` on its own line** so they can command-click it:

```
Full report: {experiment_directory}/exploration/report.md
```

## Relationship to Other Skills

- **After:** run-experiment — optional, and can be run any time afterward. (`summarize-experiment` is the required post-run step; explore-experiment is an optional deeper pass.)
- **Reads:** experiment_summary.yaml, .eval files, ../summary.md
- **Creates:** `exploration/report.md`, the figures you chose, and the audit log

**Workflow position:**
```
design-experiment → scaffold-experiment → run-experiment → summarize-experiment → (optional) explore-experiment
```

## Module Organization

| Module | Purpose |
|--------|---------|
| parsing.md | Experiment location and YAML parsing |
| data_loading.md | Loading logs via `viz_helpers.py` |
| generation.md | Figure recipes, interpretation, report + compute |
| logging.md | Audit-trail format |
