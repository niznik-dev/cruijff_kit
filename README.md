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

## Current recommended instructions

Specific to della:
```
ssh user@della-gpu.princeton.edu
module load anaconda3/2025.6 
```

All machines with conda and GPU visibility (including della).  

You'll need to pick a name for your environment.  We recommend `cruijff`, but you can adjust as you wish.

```
conda create -n cruijff python=3.13
conda activate cruijff
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
pip3 install torchao wandb h5py inspect-ai datasets peft
pip3 install transformers scikit-learn matplotlib # These are only used for eval.py
```

Now you need to decide if you want the last stable release of torchtune or the torchtune nightly build. We recommend the nightly build so that you can evaluate while fine-tuning. [Working with val_loss (validation loss) which is not yet in a regular release.]

If you want the torchtune nightly build (recommended)
```
pip3 install --pre torchtune --extra-index-url https://download.pytorch.org/whl/
```

If you want the last stable torchtune (v0.6.1)
```
pip3 install torchtune
```

Finally, install cruijff_kit as a package so you can use cruijff_kit's utilities and workflows directly in your Python environment; assuming you've cloned cruijff_kit, navigate inside that folder and run:

```
pip install -e .
```

If you plan to contribute to cruijff_kit, you'll also need the GitHub CLI for managing issues and pull requests:

```
conda install -c conda-forge gh
```

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


