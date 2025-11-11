# Resources Validation

Validate that all required resources have been verified and logged.

## What to Verify

### Models
- ✓ All model directories exist
- ✓ Paths logged in design-experiment.log
- ✓ Approximate sizes noted

### Training Dataset
- ✓ Dataset file exists
- ✓ Size checked and logged
- ✓ Required splits present (train, validation if needed, test if needed)
- ✓ Format appropriate for torchtune

### Evaluation Scripts
- ✓ All inspect-ai task scripts exist OR
- ✓ Missing scripts noted as prerequisites
- ✓ If missing, user knows to run `create-inspect-task` first

### Disk Space
- ✓ Available space checked
- ✓ Sufficient for estimated checkpoint sizes
- ✓ Warning issued if tight

## Complete Validation Checklist

Before presenting the plan to user (step 9 of workflow):
- ✓ All models verified
- ✓ Dataset verified with correct splits
- ✓ Evaluation scripts verified (or noted as prerequisites)
- ✓ Time estimates calculated or clearly marked as preliminary
- ✓ Disk space checked
- ✓ All run names follow convention
- ✓ Evaluation matrix is consistent

## If Resources Are Missing

**Don't block the plan** - note prerequisites clearly:

```markdown
## Prerequisites

Before running `scaffold-experiment`, you must:

1. **Create evaluation task:** Run `create-inspect-task` to create the capitalization task script
   - Task name: capitalization
   - Expected location: `{repo_dir}/experiments/capitalization/cap_task.py`

2. **Download model:** (if model missing)
   - Use appropriate download tool for Llama-3.2-1B-Instruct

Once prerequisites are complete, you can proceed with scaffolding.
```
