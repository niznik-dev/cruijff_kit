# Model-Organism Datasets

Reference for authoring a `data.data_generation` block with
`tool: model_organism` in `experiment_summary.yaml`. Used when the
experiment's training/eval data comes from the model-organisms framework at
`src/tools/model_organisms/` (inputs × rules × formats × designs).

When this block is present, `scaffold-experiment` runs
`src/tools/experiment/prepare_data.py` before launching subagents, which
dispatches on the `tool:` field and materializes the declared dataset on
disk.

One generator, one dataset per experiment. Multi-dataset support is a
deferred follow-up — see the tracking issue in the repo.

## When to use this

- The user wants to run a **model-organism experiment** or an
  **ablation** using deterministic sequence data (bits / digits /
  letters + an output rule like parity, majority, count, first, last,
  etc.).
- Generation is **cheap, deterministic Python** — fast, no external deps.
- For expensive or flaky generators (LLM rewrites, web scrapers), use a
  dedicated skill like `convert-tabular-to-text` instead. Do **not** put
  those in this block.

## Parameter reference

Required:

| Field | Description | Notes |
|---|---|---|
| `tool` | Generator identifier | Must be `model_organism` for this schema |
| `name` | Dataset identifier | Used in logs; not a filename |
| `input_type` | `bits` \| `digits` \| `letters` | Alphabet |
| `rule` | Output rule | See `src/tools/model_organisms/rules.py` registry. Universal rules: `first`, `last`, `nth`, `length`, `constant`, `coin`. Bits-only: `parity`, `majority`. Digits/letters-only: `min`, `max`. Bits/digits-only (linear DGP): `weighted_sum`, `weighted_sum_binary`. |
| `k` | Sequence length | Positive int |
| `N` | Number of samples | Must not exceed alphabet^k |
| `seed` | Random seed | Any int |
| `design` | `memorization` \| `in_distribution` \| `ood` | Controls train/val construction |
| `output_path` | Where to write the JSON | Relative to experiment_directory, or absolute |

Optional:

| Field | Default | Description |
|---|---|---|
| `fmt` | `spaced` | Separator: `spaced` \| `dense` \| `comma` \| `tab` \| `pipe` |
| `rule_kwargs` | `{}` | Extra params for rules that take them (e.g. `{p: 0.7}` for `coin`, `{x: 3}` for `nth`) |
| `split_ratio` | `0.8` | Train fraction for `in_distribution` and `ood` |
| `ood_tests` | — | **Required** when `design: ood`; list of per-split overrides |

### `rule_kwargs` for `weighted_sum` / `weighted_sum_binary`

Both rules implement a linear DGP `output = w·x + intercept`; the binary
variant additionally thresholds (or sigmoid-samples) the result. They
share the following `rule_kwargs`:

| Field | Type | Default | Description |
|---|---|---|---|
| `weights` | `list[int] \| None` | `None` | Explicit weight vector of length `k`. If given, distribution params are ignored. |
| `weight_max` | `int` | `3` | Magnitudes drawn uniformly from `{-W, …, -1, 1, …, W}` (excluding zero). |
| `sparsity` | `float` | `0.0` | Probability that a drawn weight is masked to zero (applied after the magnitude draw). |
| `weight_seed` | `int \| None` | `None` (uses dataset `seed`) | Seed for the **DGP**: weight draws, sparsity mask, and (binary rule) per-sequence label-noise stream. See "Seed semantics" below. |
| `intercept` | `int \| "balanced"` | `0` | DGP intercept. `"balanced"` resolves to `round(-sum(w_i) · E[x_i])` (`E[x_i]` = 0.5 for bits, 4.5 for digits). |
| `noise_scale` | `float` | `0.0` | **Binary only.** `0` → deterministic `z > 0` threshold; `> 0` → Bernoulli sample with `p = σ(z / noise_scale)`. Stochastic samples are deterministic per `(weight_seed, sequence)`. |

**Seed semantics.** These rules expose two seeds with separate roles:
- `seed` (positional arg to `generate()`) controls *which sequences you observe* (primary + OOD sampling, `coin` labels, MC samplers).
- `weight_seed` (in `rule_kwargs`, defaults to `seed`) controls *what the DGP is and how its noise behaves* (weights, sparsity mask, binary-stochastic label stream).

Fixing `weight_seed` and varying `seed` gives you fresh sequence samples of the *same* DGP — including the same per-sequence label-noise pattern. Fixing `seed` and varying `weight_seed` gives you the same sequence sample under different DGPs. Note this differs from `coin`, whose stochastic labels key on the dataset `seed`; `weighted_sum_binary` keys on `weight_seed` deliberately, so that the "same DGP, fresh sample" pattern is a clean comparison.

**Output format** (non-binary): spaced + signed + zero-padded. Width is
computed once per dataset from the resolved weights and intercept:
`max(1, ceil(log10(sum(|w_i|) · alphabet_max + |intercept| + 1)))`. Example:
weights `[1, 2, -1, 3, 0, 1, -2, 1]` on digits → max `|output| = 99` →
2-digit width → output `42` renders as `+ 4 2`.

**Top-level dataset metadata** for these rules also records:
`resolved_weights`, `intercept`, `format_width`, `weight_seed`,
`noise_scale` (binary only), and `bayes_accuracy` (binary stochastic
only — the optimal-classifier upper bound, useful for interpreting FT
gains).

**OOD note:** the resolved DGP (weights, intercept, format width) is
fixed at the dataset level; OOD splits inherit it. Use OOD specs to
vary the input distribution (e.g., `format`, `N`), not the DGP.

`weighted_sum_binary` generalizes `majority` (with weights `[1,…,1]` and
`intercept="balanced"`) and `constant` (degenerate all-zero weights).

## Example: memorization

Train == validation (identical rows). Used to check whether a model can
fit a specific label mapping without generalizing.

```yaml
data:
  training:
    path: /full/path/to/experiment_directory/data/bits_parity_k8_memo.json
    dataset_label: bits_parity_k8_memo
    format: json
    size_kb: 25
    splits:
      train: 200
      validation: 200
      test: 0
  data_generation:
    tool: model_organism
    name: bits_parity_k8_memo
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
data:
  training:
    path: /full/path/to/experiment_directory/data/digits_majority_k10.json
    dataset_label: digits_majority_k10
    format: json
    size_kb: 60
    splits:
      train: 400
      validation: 100
      test: 0
  data_generation:
    tool: model_organism
    name: digits_majority_k10
    input_type: digits
    rule: majority
    k: 10
    N: 500
    seed: 42
    design: in_distribution
    split_ratio: 0.8
    output_path: data/digits_majority_k10.json
```

## Example: out-of-distribution

Primary train/val as in_distribution; each `ood_tests` entry becomes a
`validation_ood_i` split. Entries may override `input_type`, `k`,
`format`, and `rule_kwargs`; `N` is required.

```yaml
data:
  training:
    path: /full/path/to/experiment_directory/data/bits_parity_ood.json
    dataset_label: bits_parity_indist8_ood
    format: json
    size_kb: 80
    splits:
      train: 320
      validation: 80
      test: 0
  data_generation:
    tool: model_organism
    name: bits_parity_indist8_ood
    input_type: bits
    rule: parity
    k: 8
    N: 400
    seed: 1729
    design: ood
    split_ratio: 0.8
    ood_tests:
      - {k: 12, N: 100}
      - {k: 16, N: 100}
      - {k: 8, N: 100, fmt: dense}
    output_path: data/bits_parity_ood.json
```

## Split accounting for `data.training.splits`

The generator writes one JSON with `{train, validation, metadata[, validation_ood_*]}`.
The `data.training` block is **still required** — point it at the same file.

- **memorization**: `splits.train` = `N`, `splits.validation` = `N` (identical rows).
- **in_distribution** with `split_ratio: s`: `splits.train` = round(N × s), `splits.validation` = N − train.
- **ood**: same as in_distribution, plus one unreported `validation_ood_i`
  split per `ood_tests` entry (those don't appear in `splits`).

## Conversation flow — what to ask the user

When the user says "I want to test parity" or "I want a model-organism experiment with parity" (or similar), gather:

1. **Input type and rule** — "bits parity? or digits/letters with a different rule?"
2. **Sequence length k** — single int (or a small list if varying k across runs)
3. **Sample count N** — default 100–500 for quick experiments
4. **Design** — memorization, in_distribution, or ood?
   - If ood: ask for the test-variant specs (k values, N per variant, any format overrides)
5. **Seed** — default 42 unless they care
6. **Format** — default `spaced`; ask only if they bring it up
7. **Rule kwargs** — only if the rule needs them (e.g., `coin` needs `p`; `nth` needs `x`; `weighted_sum*` accept the linear-DGP fields documented above)

Then author both the `data.data_generation` entry (with `tool: model_organism`)
and the `data.training` block. Log the schema version and dataset parameters in
`design-experiment.log` per `logging.md`.
