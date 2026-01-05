# Merging pre-yaml-refactor into main

This document details the merge conflicts and resolution strategy.

## Background

- **Main branch**: Sarah's PR #232 refactored to `experiment_summary.yaml` format and subagent architecture
- **pre-yaml-refactor branch**: PR #241/#242 made improvements (template paths, `torchtune_model_name`, MIG support, model-aware resources)
- **Goal**: Merge main into pre-yaml-refactor, then PR back to main

## Current Status

After `git reset --hard origin/pre-yaml-refactor && git merge main`:

### Resolved (10 modify/delete conflicts - accept main's deletions)

All deleted via `git rm`:
- `.claude/skills/scaffold-experiment/evaluators/README.md`
- `.claude/skills/scaffold-experiment/evaluators/inspect/scenario_selection.md`
- `.claude/skills/scaffold-experiment/evaluators/inspect/slurm_generation.md`
- `.claude/skills/scaffold-experiment/evaluators/inspect/validation.md`
- `.claude/skills/scaffold-experiment/examples/complete_example.md`
- `.claude/skills/scaffold-experiment/optimizers/torchtune/yaml_generation.md`
- `.claude/skills/scaffold-experiment/templates/setup_finetune_template.yaml`
- `.claude/skills/scaffold-experiment/templates/slurm_template.sh`
- `.claude/skills/scaffold-experiment/workflows/inspect.md`
- `.claude/skills/scaffold-experiment/workflows/torchtune.md`

### Remaining (2 content conflicts)

#### 1. `.claude/agents/scaffold-torchtune.md` - 3 conflicts

**Conflict A: Template paths (lines 168-176)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| `experiments/capitalization/setup_finetune.yaml` | `experiments/capitalization/templates/finetuning/setup_finetune_json.yaml` |
| `experiments/capitalization/yaml_examples/setup_finetune_parquet.yaml` | `experiments/capitalization/templates/finetuning/setup_finetune_parquet.yaml` |
| Checks "dataset path extension in experiment_summary.md" | Checks "`data.training.format` in experiment_summary.yaml" |

**Decision**: Accept main's YAML field names, but need to verify which template paths actually exist.

**Conflict B: Model parameter (lines 201-206)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| `torchtune_model_name: {from experiment_summary.md...}` | `model_checkpoint: {from models.base[0].path}` |

**Decision**: Accept main's version (`model_checkpoint` with YAML path).

**Conflict C: Output directory notes (lines 242-247)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| Experiment name grouping + dataset type note | Parse `output.base_directory` from YAML |

**Decision**: Accept main's version (YAML-based parsing).

---

#### 2. `.claude/agents/scaffold-inspect.md` - 3 conflicts

**Conflict A: Base model eval config approach (lines 186-226)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| Values "baked directly into SLURM script" | Creates `eval_config.yaml` intermediate file |

**Decision**: Accept main's version (eval_config.yaml approach for consistency).

**Conflict B: Fine-tuned model path format (lines 325-331)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| `MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"` | Uses `OUTPUT_BASE` variable + `CONFIG_PATH` |

**Decision**: Accept main's version (includes CONFIG_PATH for config_path parameter).

**Conflict C: Fine-tuned eval scenario description (lines 392-400)**
| HEAD (pre-yaml-refactor) | main |
|--------------------------|------|
| "reads setup_finetune.yaml... extracts values to pass directly" | "use setup_finetune.yaml from base output directory" + bash example |

**Decision**: Accept main's version.

---

## What's in the 10 Deleted Files

The deleted files contained supporting documentation that's now consolidated into the agent files. Here's what was unique:

### From `yaml_generation.md` and `setup_finetune_template.yaml`:
- **Template paths**: `setup_finetune.yaml` (not `templates/finetuning/...`)
- **`torchtune_model_name` parameter** instead of `model_checkpoint`
- **`prompt` field documentation** for train/eval parity
- **`dataset_type: chat_completion`** recommendation

### From `slurm_generation.md`:
- Epoch naming convention (0-indexed) documentation
- Directory structure diagrams
- Model-aware resource table (BUT this is already in scaffold-inspect.md!)

### From `workflows/*.md`:
- Step-by-step workflow guides (now in agent files)

## What's Already Safe (merged without conflict)

These are in the working tree from pre-yaml-refactor's non-conflicting sections:

1. **MIG support section** in scaffold-torchtune.md (lines 38-57) ✅
2. **Model-aware SLURM resources table** in scaffold-torchtune.md (lines 38-48) ✅
3. **Model-aware resource allocation** in scaffold-inspect.md (line 277) ✅

## What We Lose by Accepting Main's Conflict Resolutions

If we accept main's versions for all 6 conflicts:

1. **Template paths** - Main uses `templates/finetuning/setup_finetune_json.yaml` (may not exist)
2. **`model_checkpoint` instead of `torchtune_model_name`** - Different parameter name
3. **`eval_config.yaml` intermediate files** - Main creates config files; pre-yaml-refactor baked values into SLURM

---

## Resolution Commands

```bash
# Already done: git rm for 10 modify/delete conflicts

# For scaffold-torchtune.md - accept main's versions:
# Edit file to remove conflict markers, keeping main's side for all 3 conflicts

# For scaffold-inspect.md - accept main's versions:
# Edit file to remove conflict markers, keeping main's side for all 3 conflicts

# Then:
git add .claude/agents/scaffold-torchtune.md
git add .claude/agents/scaffold-inspect.md
git commit -m "Merge main into pre-yaml-refactor"
git push
```

---

## Post-Merge TODO

**NEXT SESSION — Move template files:**
```bash
mkdir -p experiments/capitalization/templates/finetuning
mv experiments/capitalization/yaml_examples/setup_finetune_json.yaml \
   experiments/capitalization/templates/finetuning/setup_finetune_json.yaml
mv experiments/capitalization/yaml_examples/setup_finetune_parquet.yaml \
   experiments/capitalization/templates/finetuning/setup_finetune_parquet.yaml
```

After that, verify:
1. [ ] Template paths in scaffold-torchtune.md point to files that exist
2. [ ] eval_config.yaml approach works for base model evaluations
3. [ ] MIG support section is present
4. [ ] Model-aware resources tables are present in both agent files
