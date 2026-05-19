# Continuous Variable Prediction

Extends the folktexts experiments to predict continuous values (age, income,
hours worked, commute time, etc.) instead of binary classification (0/1).

> **Note:** This document is intentionally kept separate from `README.md` while
> continuous-target support is still new. Once the feature has seen broader
> use and testing, this content will be folded into the main `README.md`.

## Data preparation

Continuous-target datasets are produced by the project's tabular-to-text
pipeline (`src/tabular_to_text_gen/convert.py`, or the `convert-tabular-to-text`
skill). The pipeline reads raw ACS
PUMS data and renders each row as natural-language
description text, using the specified schema.

To produce a **continuous** target instead of a binary one, omit
`--target-threshold` (or set `target.threshold: null` in the YAML). The raw
numeric value flows through to the `output` field.

Example: produce a 10k test set with PINCP as a continuous target.

```bash
python -m tabular_to_text_gen.convert \
    --schema src/tabular_to_text_gen/schemas/acs_example.yaml \
    --source /path/to/psam_p_all.parquet \
    --target-column PINCP \
    --question "What is this person's yearly income in dollars? Respond with only a number." \
    --context "You are analyzing data from the American Community Survey, conducted among US residents in 2018. You will be given a description of a survey respondent. Based on the information provided, predict this person's yearly income in dollars. Respond with only a number." \
    --context-placement system_prompt \
    --test-size 10000 \
    --seed 123 \
    --output {ck_data_dir}/generated/pincp_continuous_test_10000_s123.json
```

(omit `--target-threshold` to keep the value continuous.)

Available continuous targets in the ACS schema:

| Target | Column | Range (US PUMS 2018) |
|--------|--------|------------------|
| **Age** | `AGEP` | 0–95 years |
| **Income (personal yearly)** | `PINCP` | ≈ −$15k – $1.5M (with negative values for losses) |
| **Hours worked / week** | `WKHP` | 1–99 hours |
| **Commute time** | `JWMNP` | 1–200 min |

## Evaluation task

Continuous prediction uses a separate inspect-ai script with regression scoring:

**Script:** `blueprints/folktexts/inspect_task_acs_continuous.py`

**Available aliases:**
- `@acs_age` — Age prediction (continuous, years)
- `@acs_income_continuous` — Income prediction (continuous, dollars)
- `@acs_hours` — Hours worked prediction (continuous, hours/week)
- `@acs_commute` — Commute time prediction (continuous, minutes)
- `@acs_continuous` — Generic alias (when you want a custom task name)

### Usage

Evaluate a fine-tuned model (or zero-shot a base model — set
`use_chat_template=true`):

```bash
inspect eval blueprints/folktexts/inspect_task_acs_continuous.py@acs_income_continuous \
    --model hf/local \
    -M model_path=/path/to/checkpoint/epoch_0 \
    -T data_path={ck_data_dir}/generated/pincp_continuous_test_10000_s123.json \
    -T config_path=/path/to/eval_config.yaml
```

### Task parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | required | Path to JSON dataset |
| `config_path` | `""` | Path to YAML config (reads `prompt`, `system_prompt`, `scorer`) |
| `split` | `test` | Data split: `train`, `validation`, or `test` |
| `temperature` | `1e-7` | Generation temperature |
| `max_tokens` | `10` | Max tokens to generate (higher than binary — numbers can be multi-digit) |
| `use_chat_template` | `True` | Whether to use chat-format wrapping (False for base models) |
| `top_logprobs` | `20` | Number of top tokens to return logprobs for |

### Scoring

Regression metrics via `continuous_scorer` (registered in
`src/tools/inspect/scorers/`):

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `rmse` | Root Mean Squared Error |
| `r_squared` | Coefficient of determination (R²=0 means predicting the target mean) |
| `parse_rate` | Fraction of model outputs successfully parsed as numbers |

To use the continuous scorer, add to the eval config YAML:

```yaml
scorer:
  - name: continuous_scorer
```

## Files in this branch

| File | Type | Description |
|------|------|-------------|
| `blueprints/folktexts/inspect_task_acs_continuous.py` | New | Inspect-ai eval task (mirrors `inspect_task.py`, supports `assistant_prefix` and `top_logprobs`) |
| `src/tools/inspect/scorers/continuous_scorer.py` | New | Regression scorer: MAE, RMSE, R², parse_rate |
| `src/tools/inspect/scorers/__init__.py` | Edited | Registered `continuous_scorer` in `SCORER_FACTORIES` and `SCORER_REGISTRY` |
| `tests/unit/scorers/test_continuous_scorer.py` | New | Unit tests for the continuous scorer (parse, score, metrics, registry wiring) |
