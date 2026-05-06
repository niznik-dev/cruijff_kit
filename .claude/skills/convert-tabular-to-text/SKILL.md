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

Present the available template styles (see `design-experiment/references/tabular_to_text_gen.md` for the full reference):

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

**If an `experiment_summary.yaml` exists:** Scan the runs list and evaluation matrix to determine which files are needed:

- **Training conditions** need `{train_condition}_train_{hash8}.json`. The file contains both `"train"` and `"validation"` top-level keys — `convert.py` generates the bundle in a single invocation when passed `--split train` alongside `--validation-ratio`.
- **Evaluation conditions** need a separate test file: `{eval_condition}_test_{hash8}.json`. This file contains only the `"test"` key.
- **Deduplicate:** multiple runs may share the same train or test file.

**Filename hash.** Filenames include `{hash8}`, a content hash over the canonicalized generation config. If experiment_summary.yaml was produced by `design-experiment`, the hashed paths are already in each run's `dataset_path` / `eval_dataset_path` — read them directly. Otherwise, compute them with the shared helper as documented in [`design-experiment/references/tabular_to_text_gen.md`](../design-experiment/references/tabular_to_text_gen.md) (section "Computing the filename hash"). Full docs: [`src/tabular_to_text_gen/TABULAR_DATASET_NAMING.md`](../../../src/tabular_to_text_gen/TABULAR_DATASET_NAMING.md).

**If standalone:** Ask the user which conditions are used for training vs evaluation, and confirm `split_ratio` and `validation_ratio`.

Report the plan:
```
Files to generate (70% train / 10% validation / 20% test):
  Training conditions (bundled train + validation):
    1. dict_full_train_a1b2c3d4.json       {"train": ~7k rows, "validation": ~1k rows}
  Evaluation files (test only):
    2. dict_full_test_a1b2c3d4.json        {"test": ~2k rows}
    3. dict_synonym_test_e5f6a7b8.json     {"test": ~2k rows}
    4. narrative_reduced_test_9c8d7e6f.json {"test": ~2k rows}
```

Note any files that already exist in `{scratch_dir}/ck-data/generated/` — these will be reused, not regenerated.

Get user confirmation before proceeding.

## Step 6: Generate Datasets

`convert.py` takes `--output-dir` (not a full output path) and derives `{condition}_{split}_{hash8}.json` itself. If the file already exists, generation is skipped with a `REUSE:` log line; pass `--force` to regenerate.

### Recommended: YAML-driven invocation (when `experiment_summary.yaml` exists)

This is the preferred path. `convert.py` reads all generation parameters (source, schema, target, context, question, splits, seed, subsampling, missing-value handling, features, template, perturbations, style guidance, one-to-many, emit-source-parquet) directly from `data_generation` in the YAML. The only CLI args are which condition + split to produce and where to write.

```bash
cd {cruijff_kit_path} && python -m cruijff_kit.tabular_to_text_gen.convert \
  --experiment-summary {path_to_experiment_summary.yaml} \
  --condition-name {condition_name} \
  --split {train|test} \
  --output-dir {scratch_dir}/ck-data/generated
```

Loop over all conditions × splits needed by the experiment:

```bash
for COND in dict_subset dict_full; do
  for SPLIT in train test; do
    python -m cruijff_kit.tabular_to_text_gen.convert \
      --experiment-summary {path_to_experiment_summary.yaml} \
      --condition-name "$COND" --split "$SPLIT" \
      --output-dir {scratch_dir}/ck-data/generated
  done
done
```

Parquet sidecars (`data_generation.emit_source_parquet: true` plus `emit_source_parquet_condition: <name>`) are honored automatically — the sidecar is emitted only when the running `--condition-name` matches.

If you pass any generation-parameter CLI flag alongside `--experiment-summary` (e.g., `--source`, `--question`), `convert.py` ignores it and logs a warning. The YAML always wins.

### Fallback: direct CLI invocation (only when no experiment_summary.yaml exists)

**Do not use this path if an `experiment_summary.yaml` is available** — use the YAML-driven invocation above. The direct CLI path exists only for the genuinely standalone case.

**Before falling back to direct CLI, first ask the user:**

> "I don't see an `experiment_summary.yaml` for this work. Would you like to design an experiment first (via the `design-experiment` skill)? That's the recommended path — the YAML becomes the single source of truth for all downstream steps and avoids shell-quoting pitfalls. If you just want a one-off ad-hoc generation, say so and I'll proceed with direct CLI args."

Only proceed with the direct CLI invocation below if the user explicitly says they want ad-hoc generation (no experiment design). **Quote carefully** — wrap `--context` and `--question` in single quotes, and never pass them through `eval` or unquoted `$VAR` expansion.

```bash
cd {cruijff_kit_path} && python -m cruijff_kit.tabular_to_text_gen.convert \
  --source {source_path} \
  --schema {schema_path} \
  --condition-name {condition_name} \
  --features {comma_separated_features} \
  --template {template_type} \
  --perturbations {comma_separated_perturbations} \
  --target-column {target_column} \
  --target-threshold {threshold} \
  --context '{context_text}' \
  --context-placement {placement} \
  --question '{question_text}' \
  --split {train|test} \
  --split-ratio {split_ratio} \
  --validation-ratio {validation_ratio} \
  --seed {seed} \
  --subsampling-ratio {subsampling_ratio} \
  --missing-value-handling {missing_value_handling} \
  --output-dir {scratch_dir}/ck-data/generated
```

Additional CLI-only extras (also available via YAML fields of the same name):
- **one-to-many:** `--one-to-many-copies N --one-to-many-perturbation <name>`
- **custom narrative template:** `--template-file {path}`
- **categorical target:** `--target-mapping '{json_string}'` instead of `--target-threshold`
- **LLM narrative:** `--cache-path {scratch_dir}/ck-data/generated/.llm_cache/{condition_name}.json` and `--style-guidance '{instructions}'`
- **parquet sidecar:** `--emit-source-parquet` (flag). Only pass on one canonical condition per experiment to avoid redundant copies.

### Train/test pairs

For a condition used for **both** training and evaluation, invoke twice (once with `--split train`, once with `--split test`). Both share the same `{hash8}` (split is not part of the hash). In YAML-driven mode the split is the only thing that differs between the two calls.

**Important:** Use the same `--seed` for all conditions in an experiment so they operate on the same underlying data rows. In YAML-driven mode this is automatic — the seed comes from `data_generation.seed`.

## Step 7: Validate

After generation, validate all output files using the checklist in [validation.md](validation.md). Run all checks, collect issues, and present the summary table and sample entries to the user.

If any blocking issues are found, fix them before proceeding. If only warnings, report them and let the user decide.

## Step 8: Report Final Paths

Present the complete list of generated files with their paths:

```
Generated datasets:
  - {scratch_dir}/ck-data/generated/dict_full_train_a1b2c3d4.json (800 rows, 420 KB)
  - {scratch_dir}/ck-data/generated/dict_full_test_a1b2c3d4.json (200 rows, 105 KB)
  - {scratch_dir}/ck-data/generated/dict_synonym_test_e5f6a7b8.json (200 rows, 108 KB)
  - {scratch_dir}/ck-data/generated/narrative_reduced_test_9c8d7e6f.json (200 rows, 95 KB)

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
