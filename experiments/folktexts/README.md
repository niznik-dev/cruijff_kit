# Folktexts Experiments

Experiments using US Census (ACS) data via the folktexts framework for LLM calibration and prediction research.

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
- **Location**: `data/green/acs/acs_income_condensed_50000_80P.json` (and similar for other tasks)
- **Splits**: 80% train / 10% validation / 10% test
- **Source**: American Community Survey (ACS) PUMS 2018 via [folktexts](https://huggingface.co/datasets/acruz/folktexts)

See `data/green/acs/README.md` for full dataset documentation.

## Evaluation Tasks

Each task has a corresponding inspect-ai evaluation script:

| Task | Script |
|------|--------|
| ACSIncome | `experiments/folktexts/inspect_task_acs_income.py` |
| ACSEmployment | `experiments/folktexts/inspect_task_acs_employment.py` |
| ACSMobility | `experiments/folktexts/inspect_task_acs_mobility.py` |
| ACSPublicCoverage | `experiments/folktexts/inspect_task_acs_publiccoverage.py` |
| ACSTravelTime | `experiments/folktexts/inspect_task_acs_traveltime.py` |

### Usage

Evaluate a fine-tuned model:
```bash
inspect eval experiments/folktexts/inspect_task_acs_income.py@acs_income \
    --model hf/model_name \
    -M model_path=/path/to/checkpoint/epoch_0 \
    -T data_path=/path/to/acs_income_condensed_50000_80P.json \
    -T config_path=/path/to/setup_finetune.yaml
```

### Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | required | Path to JSON dataset |
| `config_path` | `""` | Path to setup_finetune.yaml (reads prompt/system_prompt) |
| `split` | `test` | Data split: train, validation, or test |
| `temperature` | `1e-7` | Generation temperature |
| `max_tokens` | `5` | Max tokens to generate |

### Scoring

Exact match on binary output: `"1"` (positive class) or `"0"` (negative class).

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

### extract_acs_verbose.py

Extracts verbose-format datasets from HuggingFace for all 5 ACS tasks.

```bash
# Extract 50K samples for employment task
python experiments/folktexts/extract_acs_verbose.py \
    --task ACSEmployment \
    --output acs_employment_verbose_50000_80P.json \
    --train-size 40000 --val-size 5000 --test-size 5000
```

Supported tasks: `ACSIncome`, `ACSEmployment`, `ACSMobility`, `ACSPublicCoverage`, `ACSTravelTime`

### convert_acs_formats.py

Converts verbose datasets to condensed and terse formats. Auto-detects task from filename.

```bash
# Convert to condensed + terse formats
python experiments/folktexts/convert_acs_formats.py \
    --input acs_employment_verbose_50000_80P.json \
    --output-dir data/green/acs/
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
