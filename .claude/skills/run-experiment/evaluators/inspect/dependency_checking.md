# Dependency Checking - Prerequisites Verification

**CRITICAL:** Verify fine-tuning complete before submitting evaluations.

Evaluations CANNOT run without fine-tuned model checkpoints. This stage ensures prerequisites are met.

## Check 1: Fine-Tuning Jobs Status

Query if fine-tuning jobs are still running:

```bash
squeue -u $USER | grep finetune
```

**If fine-tuning jobs are running:**
- Report to user: "Fine-tuning jobs are still in progress"
- Ask: "Fine-tuning jobs are still running. Wait for completion or proceed anyway?"
- **If wait:** Poll every minute until all finetune jobs complete, then proceed
- **If proceed anyway:** Continue (evaluation jobs will likely fail when models don't exist yet)

**If no fine-tuning jobs running:**
- Fine-tuning complete (or was run separately)
- Proceed to checkpoint verification

## Check 2: Model Checkpoints Exist

For each evaluation in the list, verify required model checkpoint exists:

```bash
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/
```

**Expected files in checkpoint directory:**
- `adapter_model.bin` or weight files
- `adapter_config.json`
- Other model artifacts

**For each evaluation:**
- ✓ **Checkpoint exists:** Evaluation can proceed
- ✗ **Checkpoint missing:** Mark evaluation to skip, log warning

## Handling Missing Checkpoints

**If checkpoint missing for a run:**
- Log warning: "Model checkpoint missing for {run_name}/epoch_{N}"
- Add to skip list for evaluation_selection.md
- Continue checking other runs
- Report all missing checkpoints in summary

**Example log:**
```
[YYYY-MM-DD HH:MM:SS] CHECKPOINT_MISSING: r32_lr5e-5
Details: Expected path: /scratch/.../ck-out-r32_lr5e-5/epoch_0/
Result: Checkpoint not found - skipping evaluations for this run
Recommendation: Check if fine-tuning job failed for this run
```

## Output

**Runs ready for evaluation:** List of runs with verified checkpoints

**Runs to skip:** List of runs missing checkpoints with reasons
- Checkpoint missing (fine-tuning may have failed)
- Fine-tuning still in progress

## Error Conditions

**If ALL checkpoints missing:**
- Report error to user
- Suggest checking if fine-tuning ran successfully
- Ask: "No model checkpoints found. Run fine-tuning first?"
- Do not proceed to evaluation submission

**If SOME checkpoints missing:**
- Proceed with evaluations that have checkpoints
- Skip evaluations for missing checkpoints
- Report partial execution in summary

## Next Stage

Pass "runs ready for evaluation" to cache_prebuilding.md for HuggingFace datasets cache pre-building.
