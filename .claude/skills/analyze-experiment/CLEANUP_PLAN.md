# Cleanup Plan: analyze-experiment Skill

**Date:** 2026-01-30
**Context:** Review of skill after testing on ACS income balanced experiment

---

## Remaining Capitalization References (need cleanup)

| File | Line(s) | Issue |
|------|---------|-------|
| SKILL.md | 8, 229, 252 | "capitalization experiments" mentions |
| SKILL.md | 51 | References legacy `load_experiment_logs()` |
| parsing.md | 37-47 | Example uses `word_length: 5` |
| parsing.md | 61-64 | Example uses `word_length: [5, 6, 7]` |
| parsing.md | 85-91 | Example uses "capitalization" task |
| parsing.md | 163-178 | Entire "Capitalization Experiment Patterns" section |
| parsing.md | 204 | Variables list has "word_length" |
| generation.md | 127-132 | File naming examples use "wordlen", "prompt", "crossorg" |
| viz_helpers.py | 15, 222-228 | Docstring examples reference capitalization |

---

## Structural Issues

### 1. Documentation/Reality Mismatch
The skill documents a complex inference workflow (parsing experiment_summary.yaml for variables), but in practice we didn't use any of that - everything came from `vis_label` and eval metadata.

### 2. Unused Code in Docs
- parsing.md has ~50 lines of Python (`extract_variable_info`, `_make_var_info`) that Claude reads but never actually runs
- generation.md has a 100-line `generate_all_plots` function that's reference-only

### 3. Redundant Modules
- inference.md (192 lines) could fold into SKILL.md since it's just reference
- parsing.md's core value is just "get run names from config" - the rest is unused

### 4. Legacy Function
`load_experiment_logs()` in viz_helpers.py is the old regex-based approach. We now use `evals_df_prep()` + `parse_eval_metadata()`. Consider removing.

### 5. No Logging Implemented
Despite logging.md being 195 lines, no actual logging happened during the test run.

---

## What's Actually Used vs Documented

| Documented | Actually Used |
|------------|---------------|
| Parse experiment_summary.yaml variables | Only used to get run names (subdirs) |
| Infer views from variable types | Asked user directly |
| Complex metadata extractors | `parse_eval_metadata()` handles it |
| `load_experiment_logs()` | Not used - used lower-level functions |
| JSONL logging | Not implemented |

---

## Recommended Approach: Option 1 (Aggressive ~40% reduction)

- Merge parsing.md + data_loading.md into single "data.md"
- Fold inference.md into SKILL.md
- Remove all unused Python code examples
- Delete `load_experiment_logs()` from viz_helpers.py
- Simplify logging.md to just the format spec (no examples)

---

## Files Changed This Session

Already updated (capitalization references removed):
- data_loading.md ✓
- inference.md ✓
- logging.md ✓

Still need updates:
- SKILL.md
- parsing.md
- generation.md
- viz_helpers.py

---

## Related Issue

Created #280: Add claude.local.md validation to design-experiment skill
