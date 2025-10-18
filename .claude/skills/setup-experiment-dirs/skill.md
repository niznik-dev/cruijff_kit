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

## Example

For an experiment varying model size and LoRA rank:

```
/scratch/gpfs/MSALGANIK/mjs3/llama-lora-comparison_2025-10-18/
├── llama3_1B_rank8/
├── llama3_1B_rank16/
├── llama3_3B_rank8/
├── llama3_3B_rank16/
└── README.md
```

## Next Steps

After creating directories, suggest using the `create-torchtune-config` skill to generate config files for each experiment.
