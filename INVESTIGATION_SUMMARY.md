# Investigation Summary: Experimental Design vs Data Mismatch

**Date:** 2025-10-27
**Issue:** Validation detected mismatch between experiment_summary.md and actual .eval data
**Context:** Working on run-inspect-viz skill (Issue #170, Chunk 6)

## What We Investigated

The validation check in `run_inspect_viz.py` flagged that:
- experiment_summary.md specifies 2 experimental factors (Word Length: 5L/8L, Model Type: base/fine-tuned)
- But actual .eval files only contained 1 unique model with no visible variation

## Findings

### 1. `tasks/capitalization/` is a Development Directory

**Location:** `/home/mjs3/cruijff_kit/tasks/capitalization/`

**Contents:**
- 15 .eval files in `logs/`, all from the same model (`hf/local`)
- All evaluations run on 2025-10-16
- Task args show: `config_dir: /scratch/gpfs/MSALGANIK/mjs3/ck-out-oct16-finetune-5L-80P-1000/epoch_0/`
- No variation in model names, task names, or metadata
- Has experiment_summary.md describing a 2×2 factorial design (Word Length × Model Type)

**Conclusion:** This is a **test/development directory**, not a complete experiment. The experiment_summary.md describes a planned design that was never fully executed here.

### 2. Actual Experiments Are on Scratch

**Location:** `/scratch/gpfs/MSALGANIK/mjs3/`

**Found multiple experiment directories:**

#### cap_7L_llama32_lora_comparison_2025-10-18
**Structure:**
- `1B_baseline/` (has .eval)
- `1B_rank4/` (no .eval yet)
- `1B_rank16/` (no .eval yet)
- `1B_rank32/` (no .eval yet)
- `1B_rank64/`
- `3B_baseline/` (has .eval)
- `3B_rank4/`
- `3B_rank16/`
- `3B_rank32/`
- `3B_rank64/`

**Eval files:** 2 total
- `1B_baseline/logs/2025-10-18T10-03-46-04-00_cap-task_eA9yReQHMn5A32c84mWQcG.eval`
- `3B_baseline/logs/2025-10-18T10-03-44-04-00_cap-task_UU7fjDDss4deY8vxygqJNK.eval`

**Status:** Partially complete - only baseline models evaluated

#### cap_8L_llama32_lora_comparison_2025-10-18
**Structure:** Similar to 7L experiment

**Eval files:** 2 total
- `3B_rank4/logs/2025-10-18T14-27-33-04-00_cap-eval-3B-rank4_RR6aYeUhdXiZntv2u66vuB.eval`
- `3B_rank64/logs/2025-10-18T14-27-33-04-00_cap-eval-3B-rank64_kyVzgmACi2C3P35bs96Wn5.eval`

**Status:** Partially complete - only two fine-tuned models evaluated

#### Other experiments found
- `cap_10L_crosslen_eval_2025-10-21`
- `cap_10L_lora_rank_comparison_2025-10-21`
- `cap_10L_prompt_comparison_2025-10-22`
- `cap_cross_eval_5_9_13L_2025-10-19`
- `cap_cross_eval_5_9_13L_2025-10-20`
- `capitalization_model_size_comparison`
- Many `ck-out-oct16-finetune-words_*L_*` directories (4L through 16L)

## Conclusions

1. **The validation is working correctly** - it successfully detected that experiment_summary.md doesn't match the data

2. **tasks/capitalization/ is not a complete experiment** - it's a development/testing directory

3. **Real experiments are incomplete:**
   - cap_7L experiment: Only baselines evaluated (2 out of ~10 conditions)
   - cap_8L experiment: Only 2 fine-tuned models evaluated (2 out of ~10 conditions)

4. **Experimental conditions ARE captured** - but in the directory structure, not in .eval metadata
   - Model size: 1B vs 3B (directory names)
   - LoRA rank: baseline, rank4, rank16, rank32, rank64 (directory names)
   - Word length: 7L vs 8L (experiment directory names)

## Recommendations for run-inspect-viz

1. **For development/testing:** Add `--skip-validation` flag to allow visualizing incomplete data

2. **For proper experiments:** The validation should check:
   - Directory structure (subdirectories as conditions)
   - Not just model names in .eval files

3. **For metadata:** Consider enhancing inspect evaluations to include:
   - Experimental condition in metadata field
   - Factor levels (word_length, lora_rank, model_size)

4. **For this specific case:**
   - The validation correctly stopped execution
   - User can either wait for experiment to complete or run with `--skip-validation`

## Next Steps

When continuing with Issue #170:
1. Add `--skip-validation` or `--continue-on-validation-failure` flag
2. Consider supporting experiments where conditions are in directory structure
3. Test with a complete experiment when one is available
4. Document expected metadata structure for proper validation

## Files Modified

- `tools/inspect/run_inspect_viz.py` - Added `validate_design_vs_data()` method
- Successfully tested validation with tasks/capitalization/

## Git Status

All changes committed and pushed to branch `170-run-inspect-viz-skill`
- Latest commit: "Add validation to check experimental design matches data" (4a8a1da)
