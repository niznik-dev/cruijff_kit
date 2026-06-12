# Folktexts Experiments

Experiments using US Census (ACS) data with tabular-to-semantic mappings from the [folktexts](https://github.com/socialfoundations/folktexts) package (Cruz et al., NeurIPS 2024) for LLM calibration and prediction research.

**New to cruijff_kit?** Start with the [ACS Example](../../docs/ACS_EXAMPLE.md), which walks through a complete ACS income experiment from scratch.

## Available Tasks

All tasks use the same 10 demographic features but predict different outcomes:

| Task | Question | Threshold | Classes |
|------|----------|-----------|---------|
| **ACSIncome** | What is this person's estimated yearly income? | >$50,000 | Below/Above $50k |
| **ACSEmployment** | What is this person's employment status? | ESR==1 | Employed civilian / Other |
| **ACSMobility** | Has this person moved in the last year? | MIG!=1 | Stayed / Moved |
| **ACSPublicCoverage** | Does this person have public health insurance? | PUBCOV==1 | Yes / No |
| **ACSTravelTime** | What is this person's commute time? | >20 min | ≤20 min / >20 min |

## Dataset

- **Format**: Condensed key:value pairs (most efficient for fine-tuning)
- **Location**: `{ck_data_directory}/folktexts/acs_income_condensed_50000_80P.json` (and similar for other tasks)
- **Splits**: 80% train / 10% validation / 10% test
- **Source**: American Community Survey (ACS) PUMS 2018 via [folktexts](https://huggingface.co/datasets/acruz/folktexts)
- **Pinned revision**: `ad89c177` (2024-11-28) — see `FOLKTEXTS_REVISION` in `generate_data.py`

## Binary Evaluation Task

All five binary ACS tasks use a single unified inspect-ai script with task-specific aliases. Continuous-target prediction uses a separate script — see [Continuous Variable Prediction](#continuous-variable-prediction).

**Script:** `blueprints/folktexts/inspect_task.py`

**Available aliases:**
- `@acs_income` - Income prediction (>$50k)
- `@acs_employment` - Employment prediction
- `@acs_mobility` - Mobility prediction (moved in last year)
- `@acs_publiccoverage` - Public health coverage prediction
- `@acs_traveltime` - Travel time prediction (>20 min commute)

### Usage

Evaluate a fine-tuned model:
```bash
inspect eval blueprints/folktexts/inspect_task.py@acs_income \
    --model hf/model_name \
    -M model_path=/path/to/checkpoint/epoch_0 \
    -T data_path=/path/to/acs_income_condensed_50000_80P.json \
    -T config_path=/path/to/setup_finetune.yaml
```

Evaluate a base model (no fine-tuning):
```bash
inspect eval blueprints/folktexts/inspect_task.py@acs_employment \
    --model hf/Llama-3.2-1B-Instruct \
    -M model_path=/path/to/pretrained/Llama-3.2-1B-Instruct \
    -T data_path=/path/to/acs_employment_condensed_50000_80P.json \
    -T config_path=/path/to/eval_config.yaml
```

**Note:** The `config_path` provides `system_prompt` and `prompt` settings. For base models without a setup_finetune.yaml, create a simple YAML with these fields.

### Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | required | Path to JSON dataset |
| `config_path` | `""` | Path to setup_finetune.yaml (reads prompt/system_prompt) |
| `split` | `test` | Data split: train, validation, or test |
| `temperature` | `1e-7` | Generation temperature |
| `max_tokens` | `5` | Max tokens to generate |
| `use_chat_template` | `True` | Wrap prompt with the model's chat template. Set `False` for base (non-instruct) models. |
| `assistant_prefix` | `""` | If set, prefill an assistant turn with this string. See "Evaluating base models" below. |
| `top_logprobs` | `20` | Number of top tokens to return logprobs for. |

### Evaluating base (non-instruct) models

Base models often won't emit a clean `"1"` / `"0"` when handed a chat-template prompt — they continue the document instead of answering. Two knobs help:

- `use_chat_template=False` skips the chat wrapper so the prompt is fed as raw text.
- `assistant_prefix="Answer: "` (or whatever leads naturally into the label in your prompt) prefills an assistant turn, so the next token the model generates lands in the answer position. The scorer then matches `"1"` / `"0"` as usual.

Example:

```bash
inspect eval inspect_task.py@acs_income --model hf/local \
    -M model_path=meta-llama/Meta-Llama-3.1-8B \
    -T config_path=/path/to/setup_finetune.yaml \
    -T data_path=/path/to/acs_income_condensed_50000_80P.json \
    -T use_chat_template=False \
    -T assistant_prefix="Answer: "
```

### Scoring

Exact match on binary output: `"1"` (positive class) or `"0"` (negative class).

## Continuous Variable Prediction

The tasks above predict **binary** outcomes. The same ACS data also supports **continuous** (regression) targets — predicting a raw numeric value (age, income, hours worked, commute time) instead of a class.

### Data preparation

Continuous datasets come from the tabular-to-text pipeline (`src/tabular_to_text_gen/convert.py`, or the `convert-tabular-to-text` skill), which renders each ACS PUMS row as description text. To produce a continuous target, **omit `--target-threshold`** (or set `target.threshold: null` in the YAML) so the raw numeric value flows through to the `output` field.

Available continuous targets in the ACS schema (`src/tabular_to_text_gen/schemas/acs_example.yaml`):

| Target | Column | Range (US PUMS 2018) |
|--------|--------|----------------------|
| Age | `AGEP` | 0–95 years |
| Income (personal yearly) | `PINCP` | ≈ −$15k – $1.5M (negative values = losses) |
| Hours worked / week | `WKHP` | 1–99 hours |
| Commute time | `JWMNP` | 1–200 min |

### Evaluation task

Continuous prediction uses a separate inspect-ai script, `blueprints/folktexts/inspect_task_acs_continuous.py`, with regression scoring.

**Aliases:** `@acs_age`, `@acs_income_continuous`, `@acs_hours`, `@acs_commute`, plus `@acs_continuous` (generic, for custom task names).

```bash
inspect eval blueprints/folktexts/inspect_task_acs_continuous.py@acs_income_continuous \
    --model hf/local \
    -M model_path=/path/to/checkpoint/epoch_0 \
    -T data_path=/path/to/pincp_continuous_test.json \
    -T config_path=/path/to/eval_config.yaml
```

Task parameters mirror `inspect_task.py` (see the table above), except `max_tokens` defaults to `10` rather than `5` — continuous answers can be multi-digit.

### Scoring (continuous)

Regression metrics via `continuous_scorer` (registered in `src/tools/inspect/scorers/`):

| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `rmse` | Root Mean Squared Error |
| `r_squared` | Coefficient of determination (R²=0 means predicting the target mean) |
| `parse_rate` | Fraction of model outputs successfully parsed as numbers |

Enable it in the eval config YAML:

```yaml
scorers:
  - name: continuous_scorer
```

## Research Questions

Potential experiments using this data:

1. **Cross-task generalization**: How does accuracy vary across the 5 prediction tasks? Are some tasks harder than others?

2. **Data efficiency**: What's the minimum training data needed for good accuracy? (50K samples appears sufficient for ~80% on Income)

3. **Model size vs task difficulty**: Do larger models help more on harder tasks?

4. **Calibration impact of fine-tuning**: Does fine-tuning improve or harm model calibration?

## Training Parameters (80GB GPU)

Recommended batch sizes and sequence lengths for fine-tuning on della (A100 80GB):

### Condensed Format (recommended for fine-tuning)

| Model Size | batch_size | max_seq_len | GPU Memory | Time (50K samples) |
|------------|------------|-------------|------------|-------------------|
| 1B | 192 | 512 | ~72 GB | ~7 min |
| 3B | 96 | 512 | ~60 GB | ~15 min |
| 8B | 32 | 512 | ~75 GB | ~35 min |

### Terse Format

| Model Size | batch_size | max_seq_len | GPU Memory |
|------------|------------|-------------|------------|
| 1B | 256 | 384 | ~70 GB |
| 3B | 128 | 384 | ~65 GB |
| 8B | 48 | 384 | ~75 GB |

### Verbose Format

| Model Size | batch_size | max_seq_len | GPU Memory |
|------------|------------|-------------|------------|
| 1B | 48 | 1024 | ~75 GB |
| 3B | 24 | 1024 | ~70 GB |
| 8B | 8 | 1024 | ~78 GB |

**Notes**:
- Condensed format offers the best balance of context and efficiency
- Terse format is fastest but may lose semantic clarity
- Verbose format preserves full context but requires more memory
- Times are approximate for 1 epoch on 50K training samples
- Requires 80GB GPU (e.g., A100-80GB with `--constraint=gpu80`)

## Data Preparation Scripts

Scripts for generating ACS datasets from HuggingFace:

### generate_data.py

Extracts verbose-format datasets from HuggingFace for all 5 ACS tasks.

```bash
# Extract 50K samples for employment task
python blueprints/folktexts/generate_data.py \
    --task ACSEmployment \
    --output acs_employment_verbose_50000_80P.json \
    --train-size 40000 --val-size 5000 --test-size 5000
```

Supported tasks: `ACSIncome`, `ACSEmployment`, `ACSMobility`, `ACSPublicCoverage`, `ACSTravelTime`

### modifiers/convert_formats.py

Converts verbose datasets to condensed and terse formats. Auto-detects task from filename.

```bash
# Convert to condensed + terse formats
python blueprints/folktexts/modifiers/convert_formats.py \
    --input acs_employment_verbose_50000_80P.json \
    --output-dir {ck_data_directory}/folktexts/
```

Outputs: `acs_{task}_condensed_*.json` and `acs_{task}_terse_*.json`

### Task-Specific Fields

Fields vary by task. Common fields: age, sex, race, marital status, education, relationship.

| Task | Additional Fields |
|------|-------------------|
| Income | worker class, occupation, birthplace, hours/week |
| Employment | disability, citizenship, military, ancestry, nativity, hearing/vision/cognition |
| Mobility | employment status, commute time, yearly income |
| PublicCoverage | employment status, yearly income, resident state |
| TravelTime | occupation, transportation mode, PUMA codes, poverty ratio |

## References

- [FolkTexts paper (NeurIPS 2024)](https://github.com/socialfoundations/folktexts) - "Instruction-Tuning Harms Calibration"
- [HuggingFace dataset](https://huggingface.co/datasets/acruz/folktexts)
- [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)
