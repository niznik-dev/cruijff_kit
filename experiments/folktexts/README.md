# Folktexts Experiments

Experiments using US Census (ACS) data via the folktexts framework for LLM calibration and prediction research.

## Dataset

**ACS Income**: Binary classification predicting whether income exceeds $50,000.

- **Location**: `data/green/acs/acs_income.json`
- **Splits**: 5,000 train / 500 validation / 500 test
- **Source**: American Community Survey (ACS) PUMS 2018 via [folktexts](https://huggingface.co/datasets/acruz/folktexts)

See `data/green/acs/README.md` for full dataset documentation.

## Evaluation Task

**Task**: `acs_income`
**Script**: `experiments/folktexts/inspect_task_acs_income.py`

### Usage

Standalone evaluation (e.g., base model):
```bash
inspect eval experiments/folktexts/inspect_task_acs_income.py \
    --model hf/local \
    -M model_path=/path/to/model \
    -T dataset_path=data/green/acs/acs_income.json \
    -T split=test
```

With fine-tuning config (after scaffold-experiment):
```bash
inspect eval experiments/folktexts/inspect_task_acs_income.py \
    --model hf/local \
    -M model_path=/path/to/checkpoint \
    -T config_dir=/path/to/epoch_0
```

### Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | - | Path to JSON dataset (standalone mode) |
| `config_dir` | - | Path to epoch directory (fine-tuning mode) |
| `split` | `test` | Data split: train, validation, or test |
| `system_prompt` | `""` | System message for the model |
| `temperature` | `1e-7` | Generation temperature |
| `max_tokens` | `10` | Max tokens to generate |

### Scoring

Exact match on binary output: `"1"` (income > $50k) or `"0"` (income <= $50k).

## Research Questions

Potential experiments using this data:

1. **Calibration impact of fine-tuning**: Does fine-tuning on ACS data improve or harm model calibration? (Hypothesis: light fine-tuning may help, heavy fine-tuning may cause overconfidence)

2. **LoRA rank comparison**: How do different LoRA ranks affect prediction accuracy and calibration?

3. **Base model benchmarking**: How well do base models perform on income prediction out-of-the-box?

## Example experiment_summary.md Entry

When using with `design-experiment`:

```markdown
### Evaluation Tasks

| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| acs_income | `experiments/folktexts/inspect_task_acs_income.py` | `data/green/acs/acs_income.json` | Income prediction (>$50k) |
```

## References

- [FolkTexts paper (NeurIPS 2024)](https://github.com/socialfoundations/folktexts) - "Instruction-Tuning Harms Calibration"
- [HuggingFace dataset](https://huggingface.co/datasets/acruz/folktexts)
- [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html)
