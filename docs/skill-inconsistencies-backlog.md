# Skill Documentation Inconsistencies - Backlog

Remaining issues from audit (2025-01-05). Critical issues have been fixed.

## High Severity

### 1. `inspect.slurm` vs `{task}_epoch{N}.slurm` naming
**Location:** `.claude/agents/scaffold-inspect.md` line 3
**Issue:** Title says "generates inspect.slurm scripts" but actual naming is `{task_name}_epoch{N}.slurm`
**Fix:** Update line 3 description to reflect actual naming convention

### 2. `run-experiment` workflow documentation incomplete
**Location:** `.claude/skills/design-experiment/SKILL.md` line 58
**Issue:** References `run-experiment` but workflow docs reference old "All Runs" tables
**Fix:** Update run-experiment workflow docs to parse YAML structure instead of markdown tables

### 3. `config_dir` vs `dataset_path` inconsistent
**Locations:**
- `.claude/agents/scaffold-inspect.md` (lines 362-407)
- `.claude/skills/create-inspect-task/SKILL.md` (lines 403, 791-859)
**Issue:** Unclear when to use `config_dir` (fine-tuned) vs direct params (base models)
**Fix:** Add clear documentation distinguishing the two patterns

## Medium Severity

### 4. `learning_rate` vs `lr` parameter naming
**Locations:**
- `.claude/agents/scaffold-torchtune.md` line 200
- `.claude/skills/design-experiment/param_selection.md` line 129
**Issue:** experiment_summary.yaml uses `learning_rate`, setup_finetune.yaml uses `lr`
**Note:** scaffold-torchtune.md has a comment explaining this, but could be clearer

### 5. `eval_config.yaml` not documented in design-experiment
**Location:** `.claude/agents/scaffold-inspect.md` lines 191, 413, 430
**Issue:** scaffold-inspect creates `eval_config.yaml` for base models but design-experiment doesn't mention this
**Fix:** Add note to design-experiment or scaffold-inspect clarifying this is auto-generated

### 6. Log file format inconsistency (`.log` vs `.jsonl`)
**Locations:** Various skill logging.md files
**Issue:** Some skills use `.log` (text), others use `.jsonl` (JSON Lines)
**Note:** May be intentional - needs documentation explaining why

### 7. Base model directory creation unclear
**Location:** `.claude/agents/scaffold-inspect.md` lines 169-233
**Issue:** Doesn't clearly state directory naming for base model evals
**Fix:** Add explicit directory creation examples for base/control runs

## Skipped (WIP)

### analyze-experiment still references `.md`
**Location:** `.claude/skills/analyze-experiment/SKILL.md` (lines 12, 23, 81, 283)
**Issue:** References `experiment_summary.md` instead of `.yaml`
**Note:** Skipped - someone else will fix this (WIP skill)

---

*Created during PR #230 review*
