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

## Efficient Training Settings

Based on optimization experiments (see `3B_condensed_efficient/NOTES.md`):

| Model | Batch Size | Seq Len | Memory | Time (50K samples) |
|-------|------------|---------|--------|-------------------|
| 1B | 192 | 512 | ~72 GB | ~7 min |
| 3B | 96 | 512 | ~60 GB | ~15 min |

Requires 80GB GPU (e.g., A100-80GB with `--constraint=gpu80`).

## References

- [FolkTexts paper (NeurIPS 2024)](https://github.com/socialfoundations/folktexts) - "Instruction-Tuning Harms Calibration"
- [HuggingFace dataset](https://huggingface.co/datasets/acruz/folktexts)
- [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)
