# Model-Organism Datasets

Reference for authoring a `data_generation.model_organism` block in
`experiment_summary.yaml`. Used when the experiment's training/eval data
comes from the model-organisms framework at
`sanity_checks/model_organisms/` (inputs × rules × formats × designs).

When this block is present, `scaffold-experiment` runs
`tools/experiment/prepare_data.py` before launching subagents, which
materializes the declared datasets on disk.

## When to use this

- The user wants a **sanity check** or **ablation** using deterministic
  sequence data (bits / digits / letters + an output rule like parity,
  majority, count, first, last, etc.).
- Generation is **cheap, deterministic Python** — fast, no external deps.
- For expensive or flaky generators (LLM rewrites, web scrapers), use a
  dedicated skill like `convert-tabular-to-text` instead. Do **not** put
  those in this block.

## Parameter reference

Required per entry:

| Field | Description | Notes |
|---|---|---|
| `name` | Dataset identifier | Used in logs; not a filename |
| `input_type` | `bits` \| `digits` \| `letters` | Alphabet |
| `rule` | Output rule | See `sanity_checks/model_organisms/rules.py` registry. Universal rules: `first`, `last`, `nth`, `length`, `constant`, `coin`. Bits-only: `parity`, `majority`. Digits/letters-only: `min`, `max`. |
| `k` | Sequence length | Positive int |
| `N` | Number of samples | Must not exceed alphabet^k |
| `seed` | Random seed | Any int |
| `design` | `memorization` \| `in_distribution` \| `ood` | Controls train/val construction |
| `output_path` | Where to write the JSON | Relative to experiment_dir, or absolute |

Optional:

| Field | Default | Description |
|---|---|---|
| `fmt` | `spaced` | Separator: `spaced` \| `dense` \| `comma` \| `tab` \| `pipe` |
| `rule_kwargs` | `{}` | Extra params for rules that take them (e.g. `{p: 0.7}` for `coin`, `{x: 3}` for `nth`) |
| `split` | `0.8` | Train fraction for `in_distribution` and `ood` |
| `ood_tests` | — | **Required** when `design: ood`; list of per-split overrides |

## Example: memorization

Train == validation (identical rows). Used to check whether a model can
fit a specific label mapping without generalizing.

```yaml
data_generation:
  model_organism:
    - name: bits_parity_k8_memo
      input_type: bits
      rule: parity
      k: 8
      N: 200
      seed: 1729
      design: memorization
      output_path: data/bits_parity_k8_memo.json
```

## Example: in-distribution generalization

N unique sequences drawn from one distribution, split into train/val.

```yaml
data_generation:
  model_organism:
    - name: digits_majority_k10
      input_type: digits
      rule: majority
      k: 10
      N: 500
      seed: 42
      design: in_distribution
      split: 0.8
      output_path: data/digits_majority_k10.json
```

## Example: out-of-distribution

Primary train/val as in_distribution; each `ood_tests` entry becomes a
`validation_ood_i` split. Entries may override `input_type`, `k`,
`format`, and `rule_kwargs`; `N` is required.

```yaml
data_generation:
  model_organism:
    - name: bits_parity_indist8_ood
      input_type: bits
      rule: parity
      k: 8
      N: 400
      seed: 1729
      design: ood
      split: 0.8
      ood_tests:
        - {k: 12, N: 100}
        - {k: 16, N: 100}
        - {k: 8, N: 100, fmt: dense}
      output_path: data/bits_parity_ood.json
```

## Also required: back-fill `data.training`

The generator writes one JSON with `{train, validation, metadata[, validation_ood_*]}`.
The existing `data.training` block in the YAML is **still required** —
point it at the same file so scaffold-torchtune can read it.

```yaml
data:
  training:
    path: /full/path/to/experiment_dir/data/bits_parity_k8_memo.json
    label: bits_parity_k8_memo
    format: json
    size_kb: 25              # approximate
    splits:
      train: 200             # equals N for memorization
      validation: 200        # equals train for memorization
      test: 0
```

For `in_distribution` with `split: 0.8`:
- `splits.train` = round(N * 0.8)
- `splits.validation` = N - train

For `ood`: same as in_distribution, plus one unreported `validation_ood_i`
split per `ood_tests` entry (those don't appear in `splits`).

## Conversation flow — what to ask the user

When the user says "I want a sanity check on parity" (or similar), gather:

1. **Input type and rule** — "bits parity? or digits/letters with a different rule?"
2. **Sequence length k** — single int (or a small list if varying k across runs)
3. **Sample count N** — default 100–500 for sanity checks
4. **Design** — memorization, in_distribution, or ood?
   - If ood: ask for the test-variant specs (k values, N per variant, any format overrides)
5. **Seed** — default 42 unless they care
6. **Format** — default `spaced`; ask only if they bring it up
7. **Rule kwargs** — only if the rule needs them (e.g., `coin` needs `p`; `nth` needs `x`)

Then author both the `data_generation.model_organism` entry and the
`data.training` block. Log the schema version and dataset parameters in
`design-experiment.log` per `logging.md`.
