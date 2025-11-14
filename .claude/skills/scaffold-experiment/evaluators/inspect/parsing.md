# Parsing Evaluation Configuration

This module handles parsing experiment_summary.md and claude.local.md to extract evaluation configuration.

## Parsing experiment_summary.md

Extract the following information:

### Required Information

1. **Experiment name** - From the title (line 1)
2. **Experiment directory** - From Quick Reference → Paths → Experiment
3. **All runs table** - Extract run names and their configurations
4. **Model paths** - From Resources → Models
5. **Evaluation tasks** - From Resources → Evaluation Tasks table
6. **Evaluation plan** - From Evaluation Plan section:
   - Which epochs to evaluate
   - Which runs get which evaluations
   - Evaluation datasets (if different from training)
7. **System prompt** - From Configuration section (must match training)
8. **Output directory base** - Where fine-tuned models will be saved

## Parsing the "Evaluation Tasks" Table

Example format:
```markdown
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| capitalization | `path/to/cap_task.py` | `path/to/test.json` | Tests capitalization |
```

Extract:
- **Task name** - For naming evaluation outputs
- **Script path** - Inspect-ai task file path
- **Dataset path** - If specified and different from training
- **Description** - For documentation

## Parsing the "Evaluation Plan" Section

Determine:
- **Epochs to evaluate**: "last", "all", or specific list (e.g., "0,2")
- **Evaluation matrix**: Which runs evaluate on which tasks
- **Base model evaluations**: Control runs that need evaluation

**Example plan formats:**
- "Evaluate all fine-tuned models on epoch 0" → Generate `{task}_epoch0.slurm` for each run
- "Evaluate base model and last epoch" → Generate base model eval + final epoch eval
- "Evaluate epochs 0 and 2" → Generate `{task}_epoch0.slurm` and `{task}_epoch2.slurm`

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Which conda environment to use
- `scratch_dir` - User's scratch directory
- `account` - SLURM account (OPTIONAL, under "SLURM Defaults")

## Error Handling

**If evaluation task information missing:**
- Report what's missing (task script path, dataset, etc.)
- Ask user to update experiment_summary.md
- Don't proceed without complete information

**If evaluation plan is unclear:**
- Ask user for clarification on which epochs to evaluate
- Confirm which runs get which evaluations
