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

You'll need to pick a name for your environment.  We recommend `ttenv`, but you can adjust as you wish.

```
conda create -n ttenv python=3.13
conda activate ttenv
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

## With capitalization dataset (tests/input_training)

For full instructions, see the README.md in tests/capitalization

## With twin dataset

First, obtain the csv files for this project (named in the format twindat_sim_?k_NN.csv, where ? = thousands of rows and NN is either 24 or 99 (variables)) and place them in a nice folder - I suggest `zyg_raw`.

Then run the following (can be done on a login node - only takes a minute or so):

```
python preproc.py /path/to/csv/files
```

/path/to/csv/files: A suggestion is `/scratch/gpfs/$USER/zyg_raw`

When complete, multiple JSON files will be created in the same input folder you specify above; these can then be moved to a folder such as `/scratch/gpfs/$USER/zyg_in` - you'll need this in finetune.yaml as well

# A Test Run

## With capitalization dataset

For full instructions, see the README.md in tests/capitalization

## With twin dataset

Now that we have a yaml/slurm generator, we can leverage that to make the files for our run. Before running, please check:

* Make sure `input_dir_base` is set correctly to your choice in the previous section
* If you have subfolders in `input_dir_base`, make sure to change `input_formatting` to the name of the subfolder you want instead of an empty string (uncommon)
* Make sure `conda_env` is the same one you created previously

```
python setup_finetune.py --my_wandb_project my_first_tests --my_wandb_run_name my_first_test --input_dir_base /scratch/gpfs/$USER/zyg_in/ --input_formatting '' --conda_env ttenv
```

Then run `sbatch finetune.slurm` and watch the magic happen!


