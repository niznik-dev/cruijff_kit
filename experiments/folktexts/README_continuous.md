# Continuous Variable Prediction

Extends the folktexts experiments to predict continuous values (age, income, hours worked, commute time) instead of binary classification (0/1).

## Available Continuous Targets

The HuggingFace dataset (`acruz/folktexts`) contains raw continuous values alongside binary labels. These targets are available without modifying the folktexts library:

| Target | Column | Question | Source Task | Range |
|--------|--------|----------|-------------|-------|
| **Age** | `AGEP` | How old is this person? | ACSIncome | 17–95 years |
| **Income** | `PINCP` | What is this person's yearly income in dollars? | ACSMobility | $0–$1.5M+ |
| **Hours worked** | `WKHP` | How many hours per week does this person work? | ACSIncome, ACSMobility | 1–99 hours |
| **Commute time** | `JWMNP` | What is this person's commute time in minutes? | ACSMobility | 1–200 min |

**Note on source tasks:** Different ACS tasks include different columns. `PINCP` (raw income) is not available in ACSIncome — only in ACSMobility and ACSPublicCoverage. `AGEP` is available in all 5 tasks but ACSMobility restricts ages to 19–34, so ACSIncome (ages 17–95) is the recommended source for age prediction.

## Dataset

- **Format**: Verbose natural language descriptions with continuous numeric output
- **Location**: `data/green/acs/acs_age_continuous_6000_80P.json` (and similar for other targets)
- **Splits**: 80% train / 10% validation / 10% test
- **Source**: Same HuggingFace dataset as binary tasks (`acruz/folktexts`)
- **Pinned revision**: `ad89c177` — same as `extract_acs_verbose.py`

### Description text stripping

When predicting a continuous variable, the target's line is removed from the input description to prevent data leakage. For example, when predicting age, the line `"- The age is: 42 years old."` is stripped via regex so the model must infer age from other features (education, occupation, marital status, etc.).

## Evaluation Task

Continuous prediction uses a separate inspect-ai script with regression scoring:

**Script:** `experiments/folktexts/inspect_task_acs_continuous.py`

**Available aliases:**
- `@acs_age` — Age prediction (continuous, years)
- `@acs_income_continuous` — Income prediction (continuous, dollars)
- `@acs_hours` — Hours worked prediction (continuous, hours/week)
- `@acs_commute` — Commute time prediction (continuous, minutes)

### Usage

Evaluate a fine-tuned model:
```bash
inspect eval experiments/folktexts/inspect_task_acs_continuous.py@acs_age \
    --model hf/model_name \
    -M model_path=/path/to/checkpoint/epoch_0 \
    -T data_path=/path/to/acs_age_continuous_6000_80P.json \
    -T config_path=/path/to/setup_finetune.yaml
```

### Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | required | Path to JSON dataset |
| `config_path` | `""` | Path to setup_finetune.yaml (reads prompt/system_prompt/scorer) |
| `split` | `test` | Data split: train, validation, or test |
| `temperature` | `1e-7` | Generation temperature |
| `max_tokens` | `10` | Max tokens to generate (higher than binary — numbers can be multi-digit) |

### Scoring

Regression metrics via `continuous_scorer` (registered in `tools/inspect/scorers/`):

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error — average distance between prediction and target |
| `rmse` | Root Mean Squared Error — like MAE but penalizes large errors more |
| `r_squared` | Coefficient of determination — 0 means no better than predicting the mean, 1 is perfect |
| `parse_rate` | Fraction of model outputs successfully parsed as numbers |

To use the continuous scorer, add to `setup_finetune.yaml`:
```yaml
scorer:
  - name: continuous_scorer
```

## Data Preparation Script

### extract_acs_continuous.py

Extracts continuous-target datasets from HuggingFace. Supports all 4 continuous variables.

```bash
# Extract age prediction data from ACSIncome (full age range 17-95)
python experiments/folktexts/extract_acs_continuous.py \
    --target AGEP --source-task ACSIncome \
    --train-size 40000 --val-size 5000 --test-size 5000

# Extract income prediction data from ACSMobility
python experiments/folktexts/extract_acs_continuous.py \
    --target PINCP --source-task ACSMobility \
    --train-size 40000 --val-size 5000 --test-size 5000

# Extract hours worked data
python experiments/folktexts/extract_acs_continuous.py \
    --target WKHP --source-task ACSIncome \
    --train-size 40000 --val-size 5000 --test-size 5000
```

Supported targets: `AGEP`, `PINCP`, `WKHP`, `JWMNP`

## First Results (Age Prediction)

1B model, 6K samples (4800 train / 600 val / 600 test), standard hyperparameters:

| Epoch | MAE (years) | RMSE | R² | Parse Rate |
|-------|------------|------|-----|-----------|
| 0 | 10.5 | 13.9 | 0.162 | 1.0 |
| 1 | 11.0 | 15.0 | 0.027 | 1.0 |
| 2 | 10.9 | 14.9 | 0.044 | 1.0 |

- Parse rate 1.0 — model outputs valid numbers for every sample
- R² > 0 at epoch 0 — model learned patterns beyond the mean baseline
- Performance degrades after epoch 0 — overfitting on small dataset (6K). Scaling to 50K should help.

## Files Changed

3 new files, 2 lines added to 1 existing file. No existing code modified.

| File | Type | Description |
|------|------|-------------|
| `experiments/folktexts/extract_acs_continuous.py` | New | Data extraction with continuous targets |
| `experiments/folktexts/inspect_task_acs_continuous.py` | New | Inspect-ai eval task (mirrors `inspect_task_acs.py`) |
| `tools/inspect/scorers/continuous_scorer.py` | New | Regression scorer: MAE, RMSE, R², parse_rate |
| `tools/inspect/scorers/__init__.py` | Edited | Added import + registry entry for continuous_scorer |

## References

- [FolkTexts paper (NeurIPS 2024)](https://github.com/socialfoundations/folktexts)
- [HuggingFace dataset](https://huggingface.co/datasets/acruz/folktexts)
- [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)
