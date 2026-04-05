# text_gen Architecture

This document describes each file in the text_gen library, its inputs and outputs, and how the pieces connect in the conversion pipeline.

## Pipeline Overview

```
Source CSV/TSV/etc + Schema YAML + Condition config
        │
        ▼
    readers.py          → pandas DataFrame
        │
        ▼
    convert.py          → deterministic train/test split (by seed)
        │
        ▼
    features.py         → list[(ColumnSchema, raw_value)] per row
        │
        ▼
    templates/*.py      → list[Segment] per row
        │
        ▼
    perturbations/*.py  → list[Segment] (transformed) per row
        │
        ▼
    segments.py         → body text string per row
        │
        ▼
    output.py           → {"input": ..., "output": ...} per row
        │
        ▼
    .json + .meta.json  (written to disk)
```

## File-by-File Reference

### convert.py — CLI Entrypoint

**Role:** Orchestrates the full pipeline for one (condition × split) pair.

| | Description |
|---|---|
| **Inputs** | CLI arguments: `--source`, `--schema`, `--condition-name`, `--features`, `--template`, `--perturbations`, `--target-column`, `--target-threshold`, `--context`, `--context-placement`, `--question`, `--split`, `--split-ratio`, `--seed`, `--output`. Alternatively, `--conditions-file` replaces `--features`/`--template`/`--perturbations`. |
| **Outputs** | One `.json` data file and one `.meta.json` sidecar metadata file. |
| **Calls** | `readers.read_tabular`, `Schema.from_yaml`, `features.validate_features`, `features.select_features`, `templates.get_template`, `perturbations.build_perturbation_chain`, `segments.render_segments`, `output.build_output_entry`, `output.write_output`, `output.write_metadata` |
| **Key logic** | `split_dataframe()` — deterministic train/test (or train/validation/test) split using a seeded `random.Random` shuffle of row indices. All conditions sharing the same seed operate on the same underlying rows. `load_condition_from_file()` — reads a conditions YAML as an alternative to CLI feature/template/perturbation args. |

---

### lib/readers.py — Tabular File Readers

**Role:** Auto-detect file format from extension and load into a pandas DataFrame.

| | Description |
|---|---|
| **Inputs** | A file path string (`.csv`, `.tsv`, `.dta`, `.parquet`, `.xlsx`, `.xls`, `.sas7bdat`) |
| **Outputs** | `pandas.DataFrame` |
| **Called by** | `convert.py`, `schema.py` (for skeleton generation) |

Raises `ValueError` for unsupported extensions.

---

### lib/schema.py — Schema Loading and Validation

**Role:** Load a YAML schema file into structured Python objects. Provides lookup methods and a helper for generating skeleton schemas from source data.

| | Description |
|---|---|
| **Inputs** | Schema YAML file path (for `from_yaml`); source file path + column list (for `generate_skeleton`) |
| **Outputs** | `Schema` object containing `dict[str, ColumnSchema]` keyed by source column name |
| **Called by** | `convert.py` |

**Key classes:**
- `ColumnSchema` — dataclass holding one column's metadata: `key`, `display_name`, `type` (numeric/categorical), `unit`, `synonyms`, `shorthand_map`, `restatements`
- `Schema` — dataclass holding `name`, `description`, `columns`. Methods: `from_yaml()`, `get_column()`, `generate_skeleton()`

`generate_skeleton()` reads a source file, infers numeric vs categorical types from dtypes, and returns a dict suitable for YAML serialization. This is used by the skill agent to propose a starter schema for user review.

---

### lib/features.py — Feature Selection and Filtering

**Role:** Given a schema and a condition's feature list, validate the features and extract them from each data row.

| | Description |
|---|---|
| **Inputs** | `validate_features`: list of feature keys, Schema, optional target column name. `select_features`: a row dict, feature keys, Schema. |
| **Outputs** | `validate_features`: list of warning strings (empty if clean). `select_features`: `list[tuple[ColumnSchema, str]]` — ordered (schema, raw_value) pairs. |
| **Called by** | `convert.py` |

Logs a warning (does not error) if the target column appears in the feature list.

---

### lib/segments.py — Segment Dataclass and Rendering

**Role:** Define the intermediate representation that flows between templates and perturbations, and render it into final text.

| | Description |
|---|---|
| **Inputs** | `render_segments`: list of Segments, template_type string, optional custom separator |
| **Outputs** | A single text string (the "body" of the prompt) |
| **Called by** | `convert.py` |

**Segment dataclass fields:**
- `field` — source column key (e.g., `"AGEP"`)
- `display_name` — human-readable name, possibly perturbed (e.g., `"age"` → `"years of age"`)
- `value` — formatted value string, possibly perturbed (e.g., `"New York"` → `"NY"`)
- `text` — fully rendered text for this segment (e.g., `"The age is: 51 years old."`)
- `metadata` — dict carrying schema info (synonyms, shorthand_map, restatements, type, unit) so perturbations can look up what they need
- `is_added` — `True` if inserted by clause_addition perturbation

**Rendering rules:**
- `dictionary`: newline-joined with `"- "` prefix per segment
- `narrative` / `llm_narrative`: space-joined
- Custom separator overrides both

---

### lib/templates/base.py — Abstract Template Interface

**Role:** Define the contract all templates must follow.

| | Description |
|---|---|
| **Inputs** | N/A (abstract class) |
| **Outputs** | N/A |
| **Implemented by** | `DictionaryTemplate`, `NarrativeTemplate`, `LLMNarrativeTemplate` |

**Interface:**
- `render_row(features, schema) → list[Segment]`
- `template_type → str` (property)

---

### lib/templates/dictionary.py — Key-Value Dictionary Template

**Role:** Produce folktexts-style bulleted key-value segments.

| | Description |
|---|---|
| **Inputs** | `list[tuple[ColumnSchema, str]]` (feature pairs from `features.py`) and `Schema` |
| **Outputs** | `list[Segment]` — one Segment per feature |
| **Called by** | `convert.py` via `get_template("dictionary")` |

**Text format per segment:** `"The {display_name} is: {value}."` where value includes the unit for numeric columns (e.g., `"51 years old"`).

---

### lib/templates/narrative.py — Jinja2 Narrative Template

**Role:** Produce deterministic prose using Jinja2 template files.

| | Description |
|---|---|
| **Inputs** | Feature pairs, Schema, and a `.j2` template file (defaults to `builtin_templates/default_narrative.j2`) |
| **Outputs** | `list[Segment]` — one Segment per sentence |
| **Called by** | `convert.py` via `get_template("narrative", template_file=...)` |

Template resolution order: absolute path → builtin_templates/ directory → relative path. Sentences are split on `"."` to produce one Segment per sentence.

---

### lib/templates/llm_narrative.py — LLM-Generated Narrative (Optional)

**Role:** Generate natural-language descriptions via the Anthropic API.

| | Description |
|---|---|
| **Inputs** | Feature pairs, Schema, optional cache file path |
| **Outputs** | `list[Segment]` — one Segment per sentence |
| **Called by** | `convert.py` via `get_template("llm_narrative", cache_path=...)` |
| **Dependencies** | `anthropic` package + `ANTHROPIC_API_KEY` env var (checked at init time) |

Maintains a JSON cache keyed on `sha256(features)`. Cached entries skip the API call on re-runs. LLM outputs are non-deterministic; the seed parameter does not apply.

---

### lib/templates/builtin_templates/default_narrative.j2

**Role:** Default Jinja2 template for narrative mode.

| | Description |
|---|---|
| **Inputs** | A `features` list (each item has `display_name`, `value`, `type`, `unit`, `field`) |
| **Outputs** | Rendered text string |
| **Used by** | `NarrativeTemplate` when no custom template is specified |

Produces text like: `"The respondent's age is 51 years old. Their state is New York."`

---

### lib/perturbations/engine.py — Perturbation Compose Engine

**Role:** Build and apply a chain of perturbation functions.

| | Description |
|---|---|
| **Inputs** | List of perturbation type names (e.g., `["synonym", "reorder"]`), optional seed |
| **Outputs** | A callable chain: `(list[Segment], row_index) → list[Segment]` |
| **Called by** | `convert.py` |
| **Calls** | Individual perturbation functions from the registry |

**Seeding:** Each perturbation × row combination gets an independent `random.Random` instance seeded from `f"{seed}:{perturbation_name}:{row_index}"`. This ensures:
- Same seed + same config = same output
- Different perturbation types don't interfere with each other's randomness
- Different rows are independent

---

### lib/perturbations/synonym.py

**Role:** Swap display names with random synonyms from the schema.

| | Description |
|---|---|
| **Inputs** | `list[Segment]`, seeded `random.Random` |
| **Outputs** | `list[Segment]` with updated `display_name` and `text` fields |

Example: `"The occupation is: Teacher."` → `"The profession is: Teacher."`

---

### lib/perturbations/shorthand.py

**Role:** Replace full-form values with abbreviated forms (or vice versa).

| | Description |
|---|---|
| **Inputs** | `list[Segment]`, seeded `random.Random`, `direction` parameter |
| **Outputs** | `list[Segment]` with updated `value` and `text` fields |

Example: `"The state is: New York."` → `"The state is: NY."`

---

### lib/perturbations/reorder.py

**Role:** Randomly shuffle segment order.

| | Description |
|---|---|
| **Inputs** | `list[Segment]`, seeded `random.Random` |
| **Outputs** | `list[Segment]` in shuffled order (individual segments unchanged) |

---

### lib/perturbations/clause_addition.py

**Role:** Insert redundant restatement clauses at random positions.

| | Description |
|---|---|
| **Inputs** | `list[Segment]`, seeded `random.Random`, `n_clauses` parameter |
| **Outputs** | `list[Segment]` with additional segments inserted (marked `is_added=True`) |

Uses restatement templates from the schema (e.g., `"This person is {value} years old"`). Supports `{value}` and `{decade}` placeholders. Only features with non-empty `restatements` lists are eligible.

---

### lib/output.py — Output Assembly and Writing

**Role:** Compute target labels, assemble JSON entries, write output files and metadata sidecars.

| | Description |
|---|---|
| **Inputs** | Body text, context, question, target value, threshold/mapping config, output path |
| **Outputs** | `.json` file (entries wrapped under split key) and `.meta.json` sidecar |
| **Called by** | `convert.py` |

**Key functions:**

`compute_label(target_value, threshold, mapping)`:
- Threshold mode: `"1"` if value > threshold, else `"0"`
- Mapping mode: looks up value in the mapping dict
- Neither: returns raw value as string

`build_output_entry(body_text, context, context_placement, question, ...)`:
- `preamble` mode: `{"input": "{context}\n\n{body}\n\n{question}", "output": label}`
- `system_prompt` mode: `{"input": "{body}\n\n{question}", "output": label}` — context is ignored (system prompt is an experiment-level setting in experiment_summary.yaml)

`write_output(entries, path, split)`:
- Writes `{split: [entries]}` — e.g., `{"train": [{...}, ...]}`

`write_metadata(...)`:
- Writes `.meta.json` sidecar with full provenance: condition name, split, seed, row count, file size, source path, schema path, features, template, perturbations, target config, context, question, timestamp, version

---

## Output File Format

Each generated file contains one split, wrapped under the split key:

```json
{
  "train": [
    {"input": "Context.\n\n- The age is: 51 years old.\n...\n\nQuestion?", "output": "1"},
    {"input": "...", "output": "0"}
  ]
}
```

This is compatible with the existing cruijff_kit torchtune and inspect-ai data loaders, which select a split by key name.

When `context_placement: system_prompt`, the context text is **not** written into dataset entries. The system prompt is an experiment-level setting managed in `experiment_summary.yaml` and applied at training/evaluation time by torchtune and inspect-ai. Entries contain only `{"input", "output"}` regardless of context placement.

---

## Naming Convention

Generated files: `{condition_name}_{split}_s{seed}.json`

Examples:
- `dict_full_train_s42.json` + `dict_full_train_s42.meta.json`
- `dict_synonym_test_s42.json` + `dict_synonym_test_s42.meta.json`

---

## Test Suite

Tests live in `text_gen/tests/` and run with pytest:

```bash
cd text_gen/tests && python -m pytest -v
```

| Test file | Module under test | Test count |
|---|---|---|
| `test_segments.py` | `lib/segments.py` | 8 |
| `test_readers.py` | `lib/readers.py` | 4 |
| `test_schema.py` | `lib/schema.py` | 8 |
| `test_features.py` | `lib/features.py` | 6 |
| `test_templates.py` | `lib/templates/` | 11 |
| `test_perturbations.py` | `lib/perturbations/` | 15 |
| `test_output.py` | `lib/output.py` | 11 |
| `test_convert.py` | `convert.py` | 17 |

Shared fixtures in `conftest.py` provide reusable `Schema`, `ColumnSchema`, `Segment`, CSV, YAML, and conditions file objects.

---

## Workflow Context

text_gen is a standalone step in the cruijff_kit experiment pipeline:

```
design-experiment → text_gen (this tool) → scaffold-experiment → run-experiment → analyze-experiment
```

The user runs `convert.py` once per (condition × split) pair. The generated `.json` files are referenced in `experiment_summary.yaml` run parameters as `dataset_path` and `eval_dataset_path`. The `.meta.json` sidecars provide row counts and provenance without needing to parse the full data files.
