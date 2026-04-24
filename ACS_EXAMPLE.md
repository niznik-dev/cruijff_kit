# Onboarding to cruijff_kit with the American Community Survey (ACS)

This guide walks you through a complete experiment using cruijff_kit: fine-tuning an LLM on ACS (American Community Survey) income prediction data, then evaluating the result. By the end, you'll have trained a model to predict whether a person's income exceeds $50,000 based on features in the ACS.

This example builds on the tabular to semantic mapping of ACS data found in [folktexts](https://github.com/socialfoundations/folktexts) (Cruz et al., NeurIPS 2024). cruijff_kit extends this work by making it easy to fine-tune models on ACS prediction tasks and measure how fine-tuning affects accuracy and calibration.

## Prerequisites

Before starting, make sure you have:

1. **cruijff_kit installed** - Follow the [Quick Start](README.md#quick-start) in the README
2. **`claude.local.md` configured** - Copy the template and fill in your cluster details:
   ```bash
   cp claude.local.md.template claude.local.md
   # Edit claude.local.md with your HPC username, scratch directory, SLURM account, etc.
   ```
3. **A model downloaded** - A list of models supported by cruijff_kit is in [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md). This guide uses Llama-3.2-1B-Instruct as its example, which you can download from HuggingFace:
   ```bash
   tune download meta-llama/Llama-3.2-1B-Instruct \
       --output-dir <your_models_dir>/Llama-3.2-1B-Instruct \
       --hf-token <your_hf_token>
   ```
4. **Access to an HPC cluster with GPUs** - cruijff_kit submits jobs via SLURM; by default, it requests one GPU for fine-tuning and evaluation tasks, but this scales with the size of the model and can be adjusted manually in the SLURM scripts generated during scaffolding. 

### The Dataset

To get familiar with the workflow, we recommend generating a small dataset with 1,000 samples (800 training, 100 validation, and 100 test) on the ACS Income task, but the scripts below can be used to generate datasets of any size from HuggingFace.

#### Download and Convert

```bash
# Step 1: Extract verbose-format data from HuggingFace
python projects/folktexts/extract_acs_verbose.py \
    --task ACSIncome \
    --output acs_income_verbose_1000_80P.json \
    --train-size 800 --val-size 100 --test-size 100

# Step 2: Convert to condensed format (recommended for fine-tuning)
python projects/folktexts/convert_acs_formats.py \
    --input acs_income_verbose_1000_80P.json \
    --output-dir data/green/acs/
```

This will produce `acs_income_condensed_1000_80P.json` (condensed) and `acs_income_terse_1000_80P.json` (ultra-compact). This guide uses the condensed version. This is an example entry in the outputted json file:

```
AGE: 37 | WORKER_CLASS: Working for a for-profit private company or organization |
EDUCATION: Bachelor's degree | MARITAL: Married | OCCUPATION: Computer and information
systems managers | BIRTHPLACE: Indiana | RELATIONSHIP: The reference person itself |
HOURS_WEEK: 45 | SEX: Male | RACE: White

Income >$50k?
```

The outcome classes are `1` (above $50k) or `0` (at or below $50k).

#### Balanced Sampling

By default, the extraction script samples randomly, which may produce imbalanced classes. For balanced datasets:

```bash
python projects/folktexts/extract_acs_verbose.py \
    --task ACSIncome \
    --output acs_income_verbose_balanced.json \
    --balanced \
    --train-size 800 --val-size 100 --test-size 100
```

## The Experiment

This onboarding experiment has two runs:

| Run | Description | Purpose |
|-----|-------------|---------|
| **base** | Evaluate model without fine-tuning | Baseline: how well does the model predict income out of the box? |
| **ft-lora** | Fine-tune with LoRA, then evaluate | Treatment: does fine-tuning on ACS data improve predictions? |

Both runs are evaluated using the `acs_income` inspect-ai task on the same 100 test samples.

---

## Running an Experiment with Claude

cruijff_kit uses a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills framework. This will walk you through the process of designing an experiment, running it on HPC orchestrated by Claude, and analyzing results in a final .md report. 

### Step 1: Design the Experiment

Run the design skill in a `claude` session with:

```
/design-experiment
```

Claude will guide you through a series of questions to design your first experiment. This allows you to specify what variables(s) you want to vary (LoRA rank, model sizes, learning rates, etc), as well as the default hyperparameters + number of epochs of training. The agent should ask you whether you want to run evaluation with Inspect AI on both the fine-tuned and not fine-tuned (base) model. It is also at this stage that you can determine the prompting (appending anything after the input from the data or adding a system prompt) and the relevant evaluation metrics (see scoring section below).

Example: 
- **Scientific question**: "Does fine-tuning Llama-3.2-1B-Instruct on ACS income data improve income prediction accuracy compared to the base instruct model?"
- **Runs**: Two runs - one base model evaluation, one fine-tuned with LoRA rank 8
- **Model**: Llama-3.2-1B-Instruct
- **Dataset**: `data/green/acs/acs_income_condensed_1000_80P.json` (or whatever dataset size you choose to generate above)
- **Evaluation task**: `acs_income` (from `projects/folktexts/inspect_task.py`)
- **Epochs**: 1
- **Scorers**: `match` (output from model will be correct if it exactly matches `0` or `1`)

Claude will verify that the model and data exist, then create an `experiment_summary.yaml` file in your experiment directory. **This is the canonical file that is read by all downstream tasks, so make sure to review it before proceeding!** The agent should prompt whether you want to scaffold next.

### Step 2: Scaffold the Experiment

```
/scaffold-experiment
```

This will spin up two subagents that generate all configuration files and SLURM scripts for the runs:
- `setup_finetune.yaml`, `finetune.yaml`, `finetune.slurm` for fine-tuning runs
- `eval_config.yaml` and `acs_income_epoch0.slurm` for evaluations
- For any base model comparisons, the generated directory will only contain relevant evaluation scripts. 

### Step 3: Run the Experiment

```
/run-experiment
```

Claude submits the SLURM jobs and monitors them:
1. First, submits the fine-tuning job(s) (the base model run has no training step)
2. Waits for fine-tuning to complete
3. Submits evaluation jobs for both runs
4. Waits for evaluations to complete

Total time: approximately 10-15 minutes for the 1B model with 1,000 samples.

### Step 4: Analyze Results

```
/analyze-experiment
```

This skill reads the evaluation logs from both runs and generates an `analysis/` directory containing:

- **`report.md`** - A markdown report with accuracy comparisons, confidence intervals, and if `risk_scorer` is selected for binary tasks, calibration metrics like ECE, Brier score, and AUC
- **HTML plots** - Side-by-side comparisons of base vs. fine-tuned performance, viewable in any browser
- **Static PNG exports** - If `playwright` is installed, PNG versions of all plots are generated automatically
- **Calibration and ROC curves** - If `risk_scorer` is selected, additional diagnostic plots show how well the model's confidence aligns with actual outcomes

You can also run `/summarize-experiment` for a lightweight text summary of key metrics.

---

## How Scoring Works

When you design an experiment, you choose one or more **scorers** that determine how model outputs are evaluated. Scorers are specified during `/design-experiment` and flow through to evaluation automatically. When using cruijff_kit, you have the flexibility to use built-in scorers from Inspect AI as well as custom evaluation tasks that you design yourself (like the `risk_scorer` below). 

### Built-in Scorers from Inspect AI

**`match`** - Exact string match between model output and target. For ACS tasks, the model outputs `"0"` or `"1"` and match checks if it got the right answer.
**`includes`** - Checks if the target appears anywhere in the model's response. Useful when models wrap their answer in extra text (e.g., "The answer is 1").

### Custom Scorer Example

**`risk_scorer`** - Extracts probabilities from the model's logprobs (raw next-token probabilities) rather than just checking the final answer. For binary tasks like ACS income prediction, it computes `P("1")` — the model's estimated probability that income exceeds $50k. This measure of confidence allows us to compute calibration metrics, such as:

- **Expected Calibration Error (ECE)**: How well do the model's probabilities match reality? If the model says "80% chance of high income" for a group of people, do ~80% of them actually have high income?
- **Brier Score**: A combination metric that assesses the model's calibration and discriminiation capabilities
- **AUC**: How well does the model rank people by income likelihood, regardless of the probability threshold?

```yaml
scorers:
  - match
  - risk_scorer
```

Using both `match` and `risk_scorer` together (as in this example) gives you accuracy from `match` and calibration/discrimination metrics from `risk_scorer`.

---

### Other ACS Tasks

Five prediction tasks based on the `folktexts` framework are available, all using the same ACS feature set:

| Task flag | Question |
|-----------|----------|
| `--task ACSIncome` | Income above $50,000? |
| `--task ACSEmployment` | Employed as a civilian? |
| `--task ACSMobility` | Moved in the last year? |
| `--task ACSPublicCoverage` | Has public health insurance? |
| `--task ACSTravelTime` | Commute longer than 20 minutes? |

Each task has a corresponding inspect-ai alias (e.g., `@acs_employment`). See `projects/folktexts/README.md` for more information. 

## Next Steps

After completing this onboarding experiment:

- **Try other ACS tasks** - Run a similar experiment with `ACSEmployment` or `ACSMobility` to see how task difficulty varies
- **Experiment with hyperparameters and other models** - Try different LoRA ranks, learning rates, or larger models like Llama-3.2-3B-Instruct
- **Scale up** - Generate a 50,000-sample dataset for more realistic results
- **Add calibration scoring** - If you ran with `match` only, re-run with `risk_scorer` added to see calibration metrics and ROC/calibration curves
- **Create custom evaluation tasks** - The ACS tasks in `projects/folktexts/inspect_task.py` are one example of an inspect-ai evaluation task, but you can create your own for any dataset. Use `/create-inspect-task` to build evaluations for your own data — the skill walks you through defining the dataset format, scorer configuration, and prompt template. Once created, your custom task plugs into the same `/design-experiment` → `/run-experiment` → `/analyze-experiment` pipeline

For detailed reference on all ACS tasks, data formats, and training parameters, see `projects/folktexts/README.md`.

## References

- **folktexts**: Cruz, A. et al. "Do LLMs Know What They're Predicting? Instruction-Tuning Harms Calibration." NeurIPS 2024. [GitHub](https://github.com/socialfoundations/folktexts) | [HuggingFace dataset](https://huggingface.co/datasets/acruz/folktexts)
- **ACS PUMS**: [American Community Survey Public Use Microdata](https://www.census.gov/programs-surveys/acs/microdata.html), US Census Bureau
