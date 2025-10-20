# Setup Experiment Directories

You are helping the user create a well-organized directory structure for their fine-tuning experiments.

## Your Task

Based on the experiment plan, create a directory structure with clear naming conventions.

## Steps

### 1. Develop a Naming Convention

Come up with a helpful naming convention based on what the user wants to vary:
- If varying model sizes: include model size in the name (e.g., `1B`, `3B`, `8B`)
- If varying LoRA rank: include rank in the name (e.g., `rank8`, `rank16`)
- If varying datasets: include dataset identifier in the name
- For combinations: use clear separators (e.g., `3B_rank16_dataset1`)

**Suggest:**
- An overall folder name that describes the experiment set
- A naming convention for each experiment subfolder

**Get user feedback and approval before proceeding.**

### 2. Create Directory Structure

Once approved, create the directories in `/scratch/gpfs/MSALGANIK/mjs3/`:

```
/scratch/gpfs/MSALGANIK/mjs3/[experiment-set-name]/
├── [experiment-1]/
├── [experiment-2]/
├── [experiment-3]/
└── README.md
```

### 3. Write README.md

Create a `README.md` in the main experiment folder that explains:
- The purpose of this set of experiments
- What parameters are being varied
- The naming convention used for subfolders
- Date created and who created it
- Any other relevant context

## Directory Structure Best Practices

### Evaluation Storage (IMPORTANT)

**Store evaluations in each run's subdirectory**, not in a centralized folder.

**Recommended structure:**
```
[experiment-1]/
├── setup_finetune.yaml
├── finetune.yaml
├── finetune.slurm
├── epoch_0/
├── epoch_1/
└── evaluations/
    ├── task_1/
    ├── task_2/
    └── task_3/
```

**Why this approach?**
- Self-contained runs: Everything about a run lives in one place
- Easy archiving: Tar one directory to backup/share a complete run
- Simple cleanup: Delete a run directory, everything goes with it
- Matches tooling: `setup_inspect.py` naturally works in run directories
- inspect view can still compare across runs even when stored separately

**Note in runs_plan.md:** Include the full expected directory structure with the `evaluations/` subdirectory to set clear expectations.

## Example

For an experiment varying model size and LoRA rank:

```
/scratch/gpfs/MSALGANIK/mjs3/llama-lora-comparison_2025-10-18/
├── llama3_1B_rank8/
│   └── evaluations/  (evaluations will go here)
├── llama3_1B_rank16/
│   └── evaluations/
├── llama3_3B_rank8/
│   └── evaluations/
├── llama3_3B_rank16/
│   └── evaluations/
└── README.md
```

## Next Steps

After creating directories and writing the README, hand off to the next skill:

**Suggest to the user:** "I've created the directory structure. Would you like me to generate the torchtune configuration files for each run using the `create-torchtune-config` skill?"

The `create-torchtune-config` skill will:
- Read `runs_plan.md` to understand all run configurations
- Generate `setup_finetune.yaml` for each run directory
- Use `tools/torchtune/setup_finetune.py` to generate `finetune.yaml` and `finetune.slurm` for each run
- Create configs that match the exact specifications in the run plan (model paths, datasets, LoRA ranks, batch sizes, etc.)
