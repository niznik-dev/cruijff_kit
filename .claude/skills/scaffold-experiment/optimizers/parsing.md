# Parsing Experiment Configuration

This module handles parsing experiment_summary.md and claude.local.md to extract configuration needed for torchtune scaffolding.

## Parsing experiment_summary.md

Extract the following information:

### Required Information

1. **Experiment name** - From the title (line 1)
2. **Experiment directory** - From Quick Reference → Paths → Experiment
3. **All runs table** - Extract run names and their parameters
4. **Model path** - From Resources → Models
5. **Dataset path** - From Resources → Dataset
6. **Common configuration:**
   - Epochs (from Configuration section)
   - GPUs (from Configuration section)
   - Batch size (from Configuration section or run table)
   - LoRA ranks (from run table)
   - Learning rates (from run table if present)
   - System prompt (from Configuration section)
   - Validation during training (from Configuration section)

### Parsing the "All Runs" Table

The table format looks like:
```markdown
| Run Name | Model | LoRA Rank | Learning Rate | Batch Size | Type | Est. Time |
|----------|-------|-----------|---------------|------------|------|-----------|
| Llama-3.2-1B-Instruct_rank8_lr1e-5 | Llama-3.2-1B-Instruct | 8 | 1e-5 | 4 | Fine-tuned | ~10s |
| Llama-3.2-1B-Instruct_base | Llama-3.2-1B-Instruct | - | - | - | Control | N/A |
```

**Important:**
- Only process runs where `Type` = "Fine-tuned" (skip control runs)
- Extract parameters from table columns
- Parameters with `-` are not applicable (like control runs)

### Dataset Information

Extract from Resources → Dataset section:
- **Dataset path** (full path including filename)
- **Dataset label** - filename without extension (e.g., `words_4L_80P_300.json` → `words_4L_80P_300`)
- **Dataset extension** (`.json` or `.parquet`)
- **Parent directory** - will be used as `input_dir_base`

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Which conda environment to use
- `output_dir_base` - Where to write model checkpoints
- `my_wandb_project` - WandB project name
- `scratch_dir` - User's scratch directory
- `account` - SLURM account (under "SLURM Defaults") - **OPTIONAL**: only if user has multiple accounts and cluster requires explicit specification. Skip if not found.

## Error Handling

**If experiment_summary.md not found:**
- Report error to user
- Verify file exists before proceeding

**If required information missing from experiment_summary.md:**
- Report specific missing fields
- Ask user to provide missing information
- Do not proceed with incomplete data

**If claude.local.md not found or missing fields:**
- Report which fields are missing
- Ask user to provide values or update claude.local.md
