# Tabular Dataset Naming & Content Hashing

Generated datasets are named by a content hash of the generation config, so
identical configs always resolve to the same filename and can be reused across
experiments without regeneration. This document describes the scheme, the hash
inputs, the sidecar metadata, and the helper API shared by `convert.py` and the
scaffold skills.

## Filename scheme

```
{output_dir}/{condition}_{split}_{hash8}.json
{output_dir}/{condition}_{split}_{hash8}.meta.json
{output_dir}/{condition}_{split}_{hash8}.parquet   # when emit_source_parquet: true
```

- `{condition}` — human-readable label (`dict_full`, `narr_reorder_x3`, …)
- `{split}` — `train` (may bundle train+validation), `validation`, or `test`
- `{hash8}` — first 8 hex chars of SHA-256 over the canonicalized config

`split` is **not** in the hash: train/validation/test artifacts of the same
config share the same hash and differ only by the split suffix.

## Hash inputs

The canonical config dict is filled with defaults, serialized with
`json.dumps(..., sort_keys=True, separators=(",", ":"))`, and SHA-256'd. Fields:

| Field | Form |
|---|---|
| `source_sha256` | SHA-256 of source file content |
| `schema_sha256` | SHA-256 of schema YAML content |
| `features` | list, **order preserved** (ordering affects output) |
| `template` | type string (`dictionary` / `narrative` / `llm_narrative`) |
| `template_file_sha256` | SHA-256 of custom Jinja2 template content, else `null` |
| `perturbations` | list, order as authored |
| `target` | `{column}` plus `threshold` (float) and/or `mapping` when present |
| `context` | string |
| `context_placement` | `preamble` / `system_prompt` |
| `question` | string |
| `split_ratio`, `validation_ratio` | floats (ratio is a hash input even though which rows land in which split isn't — `seed` covers that) |
| `seed` | int |
| `subsampling_ratio` | float or `null` |
| `missing_value_handling` | `skip` / `include` |
| `missing_value_text` | only retained when `handling == "include"` |
| `one_to_many` | `{copies, perturbation}` or `null` |
| `style_guidance` | string when `template` is `llm_narrative` or `narrative`; otherwise forced to `null` |

### Canonicalization rules

Implemented in `canonicalize()` in `lib/config_hash.py`. Before hashing:

- Absent fields fill to defaults so trivial YAML edits don't fork the hash:
  `perturbations=[]`, `subsampling_ratio=1.0`, `missing_value_handling="skip"`,
  `context=""`, `context_placement="preamble"`, `question=""`,
  `one_to_many=None`, `template_file_sha256=None`, `style_guidance=None`.
- `missing_value_text` is dropped unless `missing_value_handling == "include"`.
- `style_guidance` is dropped (set to `None`) for templates where it doesn't
  affect output (i.e., for `dictionary`).
- `target.threshold` is normalized to `float` so `50000` and `50000.0` hash the
  same.
- `features` ordering is preserved — different orderings hash differently
  because they render differently.

### Excluded from the hash

- `split` — train/validation/test share a hash by design.
- `emit_source_parquet` / `emit_source_parquet_condition` — these control
  whether a parquet sidecar is written; the JSON content is identical either
  way.

### `llm_narrative` non-determinism

API output is not reproducible. Two generations with identical configs may
produce different text under the same filename. The sidecar records
`non_deterministic: true` so drift is detectable; reuse logic still treats
same-hash files as equivalent (no regeneration unless `--force`).

## Sidecar (`.meta.json`)

One sidecar per output file. Written by `lib/output.py::write_metadata`.
`config_hash` is identical across splits of the same config; per-split fields
(`split`, `row_count`, `size_bytes`, `extra_splits`) differ.

```json
{
  "condition_name": "dict_full",
  "split": "train",
  "seed": 42,
  "split_ratio": 0.8,
  "row_count": 7234,
  "size_bytes": 2843012,
  "source": {"path": "/abs/path/data.csv", "sha256": "…"},
  "source_rows_total": 10000,
  "schema": {"path": "/abs/path/schema.yaml", "sha256": "…"},
  "features": ["AGEP", "COW", "SCHL"],
  "template": "dictionary",
  "perturbations": [],
  "target": {"column": "PINCP", "threshold": 50000.0},
  "context": "…",
  "context_placement": "preamble",
  "question": "…",
  "generated_at": "2026-04-20T14:22:10+00:00",
  "non_deterministic": false,
  "config_hash": "a1b2c3d4e5f6…",
  "config_hash_short": "a1b2c3d4",
  "config": { /* the canonicalized dict that was hashed */ },
  "extra_splits": {"validation": 724}
}
```

Optional fields appear only when relevant: `template_file` (with sha256),
`one_to_many`, `extra_splits` (for bundled train+validation files).

## Helper API

Location: `tabular_to_text_gen/lib/config_hash.py`. This is the single shared
place where raw YAML is mapped to the canonical hash input, so `convert.py` and
the scaffold skills always resolve the same filename for the same config.

```python
def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's content as lowercase hex."""

def canonicalize(config: dict) -> dict:
    """Fill defaults, drop irrelevant fields, normalize types. JSON-serializable."""

def hash_config(config: dict) -> str:
    """Full hex SHA-256 over canonicalize(config) with sorted keys."""

def hash_config_short(config: dict, n: int = 8) -> str:
    """First n chars of hash_config(config). Default 8."""

def build_generation_config(data_generation: dict, condition_name: str) -> dict:
    """Build the canonical config dict for one condition by merging the shared
    data.data_generation block with per-condition settings. Hashes the source
    and schema files as a side effect."""

def resolve_dataset_path(data_generation, condition_name, split, output_dir) -> str:
    """Return the absolute path convert.py would write for (condition, split).
    Calls build_generation_config + hash_config_short. Used by scaffold skills
    to plumb dataset paths into torchtune/inspect configs."""
```

### Callers

- `tabular_to_text_gen/convert.py` — computes the hash, writes output + sidecar,
  checks for existing file before generating (reuse).
- `design-experiment` skill — writes `dataset_path` / `eval_dataset_path` into
  `experiment_summary.yaml` using `resolve_dataset_path` so the final path is
  fixed at plan time.
- `scaffold-torchtune`, `scaffold-inspect` agents — read the paths from
  `experiment_summary.yaml` (set by `design-experiment`) and wire them into the
  generated configs.
- `convert-tabular-to-text` skill — iterates conditions, invokes `convert.py`;
  if a file already exists, `convert.py` logs `REUSE:` and skips.

### Shell invocation pattern (for skills)

```bash
python -c "
import json, sys
from cruijff_kit.tabular_to_text_gen.lib.config_hash import hash_config_short
print(hash_config_short(json.load(sys.stdin)))
" < condition_config.json
```

## Reuse detection

Before generating, `convert.py` checks whether
`{condition}_{split}_{hash8}.json` exists in `--output-dir`. If yes:

- Default: skip generation, log `REUSE: {path} already exists; skipping (use --force to override)`.
- With `--force`: regenerate and overwrite both the JSON and its `.meta.json`.

The parquet sidecar is only written when `--emit-source-parquet` is set (from
the CLI) or when `data_generation.emit_source_parquet` is true AND the current
condition matches `emit_source_parquet_condition` (from
`experiment_summary.yaml`). In YAML-driven mode this ensures the parquet is
written exactly once, from the designated canonical condition.

## CLI

`convert.py` takes `--output-dir` (not `--output`); the filename is always
derived. Two invocation modes:

1. **Direct CLI**: pass `--source`, `--schema`, `--features`, `--template`,
   `--target-column`, etc. directly.
2. **YAML-driven**: pass `--experiment-summary path/to/experiment_summary.yaml`
   plus `--condition-name`, `--split`, `--output-dir`. All generation
   parameters are read from `data.data_generation`. Any generation-parameter
   CLI arg passed alongside `--experiment-summary` is ignored with a warning,
   so the on-disk YAML and the generated files cannot disagree.

Only `--condition-name`, `--split`, `--output-dir`, `--force`, and
`--cache-path` remain user-controlled under YAML-driven mode.
