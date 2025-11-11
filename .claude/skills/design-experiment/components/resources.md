# Resource Verification

Verify that all required resources exist before committing to the experiment plan.

## What to Verify

Use paths from `claude.local.md` for default locations.

### Models
**Command:** `ls {models_dir}/{model_name}`
- Verify each model directory exists
- Note approximate size

### Training Dataset
**Command:** `ls -lh {dataset_path}`
- Check file exists and note size
- Verify required splits (train, validation if needed, test if needed)

### Evaluation Task Scripts
**Command:** `ls {eval_script_path}`
- Verify each inspect-ai task script exists
- If missing, note as prerequisite (may need `create-inspect-task` skill first)

### Disk Space
**Command:** `df -h {scratch_dir}`
- Ensure sufficient space for checkpoints

## If Resources Missing

**Model:** Suggest downloading with appropriate tool

**Dataset:** Offer to help create it (if known task like capitalization)

**Eval script:** Note as prerequisite, proceed with plan anyway

**Disk space:** Warn user, suggest cleanup or alternative location

## Document in experiment_summary.md

```markdown
## Resources

### Models
- **Llama-3.2-1B-Instruct**: `{models_dir}/Llama-3.2-1B-Instruct`
  - Verified: ✓ (2025-10-22)
  - Size: ~2.5 GB

### Dataset
- **Path**: `{repo_dir}/data/green/capitalization/words_8L_80P_10000.json`
- **Format**: JSON
- **Size**: 655KB
- **Splits**: train (8000 samples), validation (1000 samples), test (1000 samples)
- **Verified**: ✓ (2025-10-22)

### Evaluation Tasks
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| capitalization | `{repo_dir}/experiments/capitalization/cap_task.py` | Same as training | Tests word capitalization accuracy |

**Note**: All paths verified during design phase. Evaluation task scripts must exist before scaffolding.
```

## Log All Verification Steps

All resource verification commands should be logged in `design-experiment.log`:

```
[2025-10-22 14:23:15] VERIFY_MODEL: Checking Llama-3.2-1B-Instruct
Command: ls {models_dir}/Llama-3.2-1B-Instruct
Result: Directory exists with 15 files (config.json, model.safetensors, etc.)
Explanation: Verifying base model exists before creating experiment plan

[2025-10-22 14:23:42] VERIFY_DATASET: Checking capitalization dataset
Command: ls -lh {repo_dir}/data/green/capitalization/words_8L_80P_10000.json
Result: File exists, 655KB
Explanation: Verifying training dataset exists and checking size
```
