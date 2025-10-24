# cruijff_kit

cruijff_kit is a toolkit for doing research with social data and LLMs. We are building workflows for:

- fine-tuning LLMs
- evaluating LLMs

We also have helper utilities for things like automated SLURM script generation and preprocessing pipelines.

cruijff_kit is named after Dutch footballer and philosopher [Johan Cruijff](https://en.wikipedia.org/wiki/Johan_Cruyff). Many of these ideas were developed while we were doing research in Amsterdam, the city of his birth.

# ⚠️ Pre-Alpha Warning ⚠️

This project is in early development and things may break without notice; you may encounter bugs or changes between updates. If you would like to use this toolkit, please let us know and reach out for assistance - we'd love to collaborate! Your feedback and bug reports are valuable for development.

# Prerequisites

- Cloning and Pulling from GitHub (properly!)
  - [Version Control with Git](https://swcarpentry.github.io/git-novice/) from Software Carpentry
  - Specifically: Episode 7
- Basic Linux and Shell Commands
  - [The Unix Shell](https://swcarpentry.github.io/shell-novice/) from Software Carpentry
  - Specifically: Episodes 1, 2, 3, and optionally 6
- Working on a Remote Machine
  - A [Digital Ocean Tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-visual-studio-code-for-remote-development-via-the-remote-ssh-plugin) for SSH using VSCode
- Understanding Python package management with conda and pip
  - [Introduction to Conda for (Data) Scientists](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/) from The Carpentries Incubator
  - Specifically: Parts 1 and 2
- [Getting a HuggingFace Account and Token](https://huggingface.co/docs/hub/en/security-tokens)
- Basic Slurm Knowledge
  - We recommend the [Princeton Research Computing](https://researchcomputing.princeton.edu/support/knowledge-base/slurm) primer
- Optional: Experience with Python coding for reading the codebase and/or adding functionality
  - [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) from Software Carpentries

# Installation

## Quick Start (Recommended)

For most users, this single code block will set up everything you need:

```bash
# Create and activate environment
conda create -n cruijff python=3.13 -y
conda activate cruijff

# Install PyTorch with CUDA support
pip3 install torch --index-url https://download.pytorch.org/whl/cu126

# Install core packages
pip3 install torchao wandb h5py inspect-ai datasets peft transformers scikit-learn matplotlib

# Install torchtune nightly (recommended for validation loss support)
pip3 install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install cruijff_kit (from repository root)
pip install -e .
```

**Installation time**: Approximately 5-10 minutes depending on network speed.

## Step-by-Step Instructions

### 1. HPC-Specific Setup (Optional)

If you're on Princeton's della cluster:
```bash
ssh user@della-gpu.princeton.edu
module load anaconda3/2025.6
```

### 2. Create Conda Environment

You can name your environment anything, but we recommend `cruijff`:

```bash
conda create -n cruijff python=3.13 -y
conda activate cruijff
```

### 3. Install PyTorch

Install PyTorch with CUDA 12.6 support:

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install Core Dependencies

These packages are required for fine-tuning and evaluation:

```bash
# Core fine-tuning packages
pip3 install torchao wandb h5py inspect-ai datasets peft

# Evaluation packages
pip3 install transformers scikit-learn matplotlib
```

### 5. Install Torchtune

**Choose one option:**

**Option A: Nightly build (recommended)**
Includes validation loss tracking during fine-tuning:
```bash
pip3 install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

**Option B: Stable release (v0.6.1)**
For production environments:
```bash
pip3 install torchtune
```

**Why nightly?** The nightly build includes `val_loss` (validation loss) tracking, which is essential for monitoring overfitting during fine-tuning. This feature is not yet available in stable releases.

### 6. Install cruijff_kit

Navigate to the repository root and install as an editable package:

```bash
cd /path/to/cruijff_kit
pip install -e .
```

This allows you to use cruijff_kit utilities and import custom recipes in your Python code.

### 7. Install GitHub CLI (For Contributors Only)

If you plan to contribute to cruijff_kit, install the GitHub CLI for managing issues and pull requests:

```bash
conda install -c conda-forge gh -y
```

## Verify Installation

Check that everything installed correctly:

```bash
# Verify Python version
python --version  # Should show Python 3.13.x

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify torchtune version
pip show torchtune | grep Version

# Verify cruijff_kit
python -c "import cruijff_kit; print('cruijff_kit installed successfully')"
```

## Troubleshooting

**Issue**: Import errors for cruijff_kit
**Solution**: Ensure you ran `pip install -e .` from the repository root directory

# Downloading a model

Next you'll need a model to finetune and evaluate with. Here's how to get one *the torchtune way*:

## Step 1 - Request Access on HuggingFace Website (if necessary)

For Meta models in particular, you'll need to navigate to the model on the HuggingFace website, log in, and agree to their Community License Agreement. (If you don't already have an account, we've found filling out your profile after you create a HuggingFace account to avoid appearing like a bot could be important!) Once you have an email confirming that you have been granted access, you can continue to the next step.

For Meta, you can typically follow a URL like this: https://huggingface.co/meta-llama/<model_name> (see options below)

## Step 2 - Run the Command

If you don't already have access to the model via a group/shared directory (e.g. MSALGANIK for the research group on della), you can download the model using torchtune as follows:

```
tune download meta-llama/<model_name> --output-dir <model_dir> --hf-token <hf-token>
```
**model_name**: We've specifically worked with:
* Llama-2-7b-hf
* Llama-3.1-8B-Instruct
* Llama-3.2-1B-Instruct (most common)
* Llama-3.3-70B-Instruct

**model_dir**: A suggestion is `/scratch/gpfs/<your_sponsor>/$USER/torchtune_models/<model_name>` - you'll need this in finetune.yaml later

**hf-token**: You can get this from your HuggingFace account; **NEVER** commit this to a repo!

# Formatting Input Files to JSON

## With capitalization dataset (tasks/input_training)

For full instructions, see the README.md in tasks/capitalization

## With twin dataset

First, obtain the csv files for this project (named in the format twindat_sim_?k_NN.csv, where ? = thousands of rows and NN is either 24 or 99 (variables)) and place them in a nice folder - I suggest `zyg_raw`.

Then run the following (can be done on a login node - only takes a minute or so):

```
python preproc.py /path/to/csv/files
```

/path/to/csv/files: A suggestion is `/scratch/gpfs/$USER/zyg_raw`

When complete, multiple JSON files will be created in the same input folder you specify above; these can then be moved to a folder such as `/scratch/gpfs/$USER/zyg_in` - you'll need this in finetune.yaml as well

# Running Experiments

cruijff_kit supports two workflows for running experiments:
1. **Skills-based workflow** (recommended when using Claude Code)
2. **Manual workflow** (for users without Claude Code access)

## Skills-Based Workflow (with Claude Code)

If you have access to Claude Code, you can use skills to automate the entire experiment workflow:

### Step 1: Design the Experiment
Use the `design-experiment` skill to plan your experiment. This creates an `experiment_summary.md` file documenting all runs, parameters, and resources.

### Step 2: Scaffold the Experiment
Use the `scaffold-experiment` skill to automatically:
- Create directory structures for each run
- Generate `setup_finetune.yaml` configs
- Generate `finetune.yaml` and `finetune.slurm` files

### Step 3: Run the Experiment
Use the `run-experiment` skill to:
- Submit all jobs to SLURM
- Monitor their progress
- Update the experiment status automatically

### Step 4: Evaluate Results
(Planned) Use the `evaluate-experiment` skill to generate and run evaluations.

See `.claude/skills/*/SKILL.md` files for detailed documentation.

## Manual Workflow

### Single Run Example (Capitalization Task)

For a complete example with detailed instructions, see `tasks/capitalization/README.md`

### Single Run Example (Twin Dataset)

1. **Prepare your configuration file**

Create a `setup_finetune.yaml` file in your task directory. You can copy from a template:

```bash
cd tasks/your_task/
cp templates/finetuning/setup_finetune_json.yaml setup_finetune.yaml
```

2. **Edit the configuration**

Key settings to verify:
* `input_dir_base` - Path to your input data directory
* `input_formatting` - Subfolder name (usually empty string `''`)
* `dataset_label` - Dataset filename without extension
* `conda_env` - Your conda environment name (e.g., `cruijff`)
* `model_checkpoint` - Path to pretrained model
* `output_dir_base` - Where to save model checkpoints
* `lora_rank` - LoRA adapter rank (e.g., 8, 16, 32, 64)
* `learning_rate` - Learning rate (e.g., 1e-5, 5e-5)

3. **Generate SLURM scripts**

```bash
python ../../tools/torchtune/setup_finetune.py
```

This creates:
- `finetune.yaml` (torchtune configuration)
- `finetune.slurm` (SLURM batch script)

4. **Submit the job**

```bash
sbatch finetune.slurm
```

5. **Monitor the job**

```bash
squeue -u $USER
tail -f slurm-*.out
```

### Multi-Run Experiments (Manual)

For experiments with multiple runs (e.g., parameter sweeps):

1. Create experiment directory structure manually
2. For each run, create a subdirectory with a descriptive name
3. Copy and customize `setup_finetune.yaml` for each run
4. Generate configs for all runs:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && python ../../tools/torchtune/setup_finetune.py)
   done
   ```
5. Submit all jobs with a stagger delay to prevent cache collisions:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && sbatch finetune.slurm)
     sleep 5  # Prevent HuggingFace cache race conditions
   done
   ```


