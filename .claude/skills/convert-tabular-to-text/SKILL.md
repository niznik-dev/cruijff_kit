---
name: convert-tabular-to-text
description: Convert tabular data (CSV, Stata, Parquet, etc.) to textual representations for LLM fine-tuning and evaluation. Supports dictionary, narrative, and LLM-generated formats with configurable perturbations.
---

# Convert Tabular to Text

Convert tabular data into textual representations for use in LLM fine-tuning and evaluation experiments. This skill runs **after** design-experiment and **before** scaffold-experiment.

```
create-tabular-schema → design-experiment → convert-tabular-to-text → scaffold-experiment
```

## Prerequisites

- **Schema file** must already exist in `ck-data/schemas/`. If it doesn't, run `create-tabular-schema` first.
- **experiment_summary.yaml** should exist (from design-experiment) for guided mode. Standalone usage is also supported.

## Your Task

Walk the user through generating text datasets from tabular source data. Each invocation of the underlying `convert.py` tool produces one output file for one (condition x split) pair. This skill orchestrates multiple calls to produce all the files an experiment needs.

**IMPORTANT:** Throughout this workflow, create a detailed log at `{experiment_dir}/logs/convert-tabular-to-text.log` (or `{scratch_dir}/ck-data/generated/logs/convert-tabular-to-text.log` for standalone usage). Write entries incrementally as actions complete. See [logging.md](logging.md) for the complete format specification and action types.

## Step 1: Check for Existing Datasets

Ask: **"Do you have existing generated datasets, or do we need to create new ones?"**

Check `ck-data/generated/` (under the user's scratch directory from `claude.local.md`) for any matching files:

```bash
ls {scratch_dir}/ck-data/generated/*.json 2>/dev/null
```

If the user has existing datasets, collect their paths and skip to Step 4.

## Step 2: Load Schema

Load the schema from `ck-data/schemas/`:

```bash
ls {scratch_dir}/ck-data/schemas/ 2>/dev/null
```

**If a matching schema exists:** Load it and confirm with the user.

**If no schema exists:** Stop and advise the user to run `create-tabular-schema` first. The schema defines columns, types, and value maps that are needed for generation.

## Step 3: Determine Template Style

**If the user has an `experiment_summary.yaml` with a `data_generation` section:**

The template style is already specified per condition. Read the `template`, `template_file`, and `style_guidance` fields from each condition. Confirm with the user: "The experiment design specifies [template types] — proceeding with those."

For each condition with `template: narrative`:
- **If `template_file` is set:** Use the specified template file. Verify it exists:
  ```bash
  ls {template_file_path}
  ```
- **If `template_file` is not set and `style_guidance` is set:** A custom template needs to be generated. Spawn the `generate-jinja-template` subagent (see below).
- **If neither `template_file` nor `style_guidance` is set:** Use the built-in generic narrative template.

For each condition with `template: llm_narrative`:
- Confirm `ANTHROPIC_API_KEY` is set and `anthropic` is installed.
- Read `style_guidance` from the condition (if present) — it will be passed as `--style-guidance` at generation time.
- A `--cache-path` should be used in Step 6 for consistency across re-runs.

**If no experiment_summary.yaml exists (standalone usage):**

Present the available template styles (see `design-experiment/references/data_generation.md` for the full reference):

> **Which template style would you like?**
>
> 1. **Dictionary** — Bulleted key-value pairs. Simple, explicit, no prose.
> 2. **Narrative** — Natural prose from a Jinja2 template.
>    - Do you have an existing Jinja2 template file, or would you like to generate one?
> 3. **LLM narrative** (`llm_narrative`) — API-generated unique text per row. Non-deterministic. Requires `ANTHROPIC_API_KEY`.

If the user picks **narrative without an existing template** or **llm_narrative**, gather style preferences:

1. **Intended use** — Fine-tuning, evaluation, or both? (Affects token density.)
2. **Style preference** — Concise/dense or natural/verbose? Any specific tone?
3. **Any other guidance** — e.g., avoid certain phrasings, emphasize particular features.

### Generating Custom Narrative Templates

If a custom narrative template is needed (no `template_file` but `style_guidance` is present, or the user requests one in standalone mode), spawn the `generate-jinja-template` subagent:

```
Agent: generate-jinja-template
Prompt: Generate a Jinja2 narrative template for:
  - Schema: {schema_path}
  - Source data: {source_path}
  - Output: {scratch_dir}/ck-data/templates/{dataset_name}.j2
  - Features: {feature_list} (if known from experiment_summary.yaml)
  - Intended use: {from style_guidance or user input}
  - Style preference: {from style_guidance or user input}
  - Additional instructions: {from style_guidance or user input}
```

When the agent returns:
1. Show the user the generated template content and sample renderings
2. Ask if they want any changes (if so, describe changes and re-run the agent, or edit the `.j2` file directly)
3. Once approved, note the template path — it will be used as `--template-file` in Step 6

## Step 4: Determine Conditions

**If the user has an `experiment_summary.yaml` with a `data_generation` section:**
- Read conditions from `data_generation.conditions`
- Read shared settings (target, context, question, seed, split_ratio) from `data_generation`

**If no experiment_summary.yaml exists (standalone usage):**
Walk the user through defining conditions. For each condition:

1. **Condition name** — identifier (e.g., `dict_full`, `dict_synonym`, `narrative_reduced`)
2. **Features** — which columns to include (can differ per condition)
3. **Template** — `dictionary`, `narrative`, or `llm_narrative`
4. **Perturbations** — list from: `synonym`, `shorthand`, `reorder`, `clause_addition` (can be empty)

Also gather shared settings:
- **Target column** and threshold or mapping
- **Context text** — the preamble/system prompt text
- **Context placement** — `preamble` or `system_prompt`
- **Question text** — the question appended to each entry
- **Seed** — random seed (default: 42)
- **Split ratio** — train fraction (default: 0.8)
- **Subsampling ratio** — fraction of source data to use, e.g., 0.33 for 33% (optional, omit to use all rows)
- **Missing value handling** — `skip` (default) or `include`. When `skip`, NaN/missing feature values are omitted from the text. When `include`, they are represented with the missing value text.
- **Missing value text** — text to use for missing values when handling is `include` (default: `missing`). Examples: `"missing"`, `"not reported"`, `"unknown"`.

## Step 5: Determine Which Files to Generate

**If an `experiment_summary.yaml` exists:** Scan the runs list to determine which (condition x split) pairs are needed:

- Training runs need `{train_condition}_train_s{seed}.json`
- Evaluation runs need `{eval_condition}_test_s{seed}.json`
- Deduplicate — multiple runs may share the same train or test file

**If standalone:** Ask the user which splits to generate for each condition.

Report the plan:
```
Files to generate:
  1. dict_full_train_s42.json      (train split, 80% of source)
  2. dict_full_test_s42.json       (test split, 20% of source)
  3. dict_synonym_test_s42.json    (test split, 20% of source)
  4. narrative_reduced_test_s42.json (test split, 20% of source)
```

Get user confirmation before proceeding.

## Step 6: Generate Datasets

For each (condition x split) pair, call `convert.py`:

```bash
cd {cruijff_kit_path} && python -m text_gen.convert \
  --source {source_path} \
  --schema {schema_path} \
  --condition-name {condition_name} \
  --features {comma_separated_features} \
  --template {template_type} \
  --perturbations {comma_separated_perturbations} \
  --target-column {target_column} \
  --target-threshold {threshold} \
  --context "{context_text}" \
  --context-placement {placement} \
  --question "{question_text}" \
  --split {split} \
  --split-ratio {split_ratio} \
  --seed {seed} \
  --subsampling-ratio {subsampling_ratio} \
  --missing-value-handling {missing_value_handling} \
  --missing-value-text "{missing_value_text}" \
  --output {scratch_dir}/ck-data/generated/{condition_name}_{split}_s{seed}.json
```

For one-to-many expansion (e.g., multiple reorderings per row), add `--one-to-many-copies {N} --one-to-many-perturbation {perturbation}`. The one_to_many perturbation must not also appear in `--perturbations`.

For narrative templates with a custom template file, add `--template-file {path}`.

For categorical target mappings, use `--target-mapping '{json_string}'` instead of `--target-threshold`.

For LLM narrative mode, add `--cache-path {scratch_dir}/ck-data/generated/.llm_cache/{condition_name}.json` and `--style-guidance "{user's style instructions}"` (if provided in Step 2b).

For experiments that want a parquet sidecar of the underlying source rows (e.g., to train a competing baseline model on the same train/test splits), add `--emit-source-parquet {scratch_dir}/ck-data/generated/{condition_name}_{split}_s{seed}.parquet`. This writes the post-subsample, post-split DataFrame — all original source columns (including the target) — with rows in 1:1 correspondence with the JSON entries. Trigger via `data_generation.emit_source_parquet: true` in experiment_summary.yaml.

- **Only emit from one condition per split.** All conditions in an experiment share the same underlying `split_df` (same seed, subsample, split_ratio), so emitting the parquet from every condition would write redundant copies. Pick a canonical condition (typically the first, or a dictionary-format one) and pass `--emit-source-parquet` only on its train and test invocations. Leave the flag off for the other conditions.

**Important:** Use the same `--seed` for all conditions in an experiment so they operate on the same underlying data rows.

## Step 7: Validate

After generation, validate all output files using the checklist in [validation.md](validation.md). Run all checks, collect issues, and present the summary table and sample entries to the user.

If any blocking issues are found, fix them before proceeding. If only warnings, report them and let the user decide.

## Step 8: Report Final Paths

Present the complete list of generated files with their paths:

```
Generated datasets:
  - {scratch_dir}/ck-data/generated/dict_full_train_s42.json (800 rows, 420 KB)
  - {scratch_dir}/ck-data/generated/dict_full_test_s42.json (200 rows, 105 KB)
  - {scratch_dir}/ck-data/generated/dict_synonym_test_s42.json (200 rows, 108 KB)
  - {scratch_dir}/ck-data/generated/narrative_reduced_test_s42.json (200 rows, 95 KB)

These paths should be used in experiment_summary.yaml run parameters:
  - dataset_path: for training datasets
  - eval_dataset_path: for evaluation datasets
```

If an `experiment_summary.yaml` exists, remind the user that it is read-only — the paths are already documented in the `data_generation` section and referenced in the runs list.

## Important Notes

- **Same seed = same rows.** All conditions for the same experiment must use the same seed so they operate on the same underlying data rows. Only the text representation differs.
- **LLM narrative is non-deterministic.** The seed parameter does not apply to `llm_narrative` mode. The cache provides run-to-run consistency but initial generation varies.
- **experiment_summary.yaml is read-only.** Runtime metadata (row counts, file sizes) lives in `.meta.json` sidecar files, not in the experiment design document.

## Module Organization

| Module | Purpose |
|--------|---------|
| SKILL.md | Main workflow (this file) |
| validation.md | Per-file and cross-file validation checklist |
| logging.md | Plain text audit trail specification |
