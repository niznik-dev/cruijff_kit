# Skills Architecture Summary

## Overview

The cruijff_kit skills have been reorganized into a modular, orchestrated workflow architecture. Each skill now has a single, clear responsibility, and orchestrator skills coordinate the overall workflow.

## Architecture

```
Workflow Stages:
┌─────────────┐
│   DESIGN    │  design-experiment (orchestrator/planner)
└──────┬──────┘
       │
┌──────▼──────┐
│  SCAFFOLD   │  scaffold-experiment (modular skill)
│             │  ├─ optimizers/ (torchtune logic)
│             │  └─ evaluators/ (inspect-ai logic)
└──────┬──────┘
       │
┌──────▼──────┐
│     RUN     │  run-experiment (orchestrator)
│             │  ├─> run-torchtune (worker, sequential)
│             │  └─> run-inspect (worker, sequential)
└──────┬──────┘
       │
┌──────▼──────┐
│   ANALYZE   │  analyze-experiment (planned)
└─────────────┘
```

## Skill Hierarchy

### Orchestrators
Skills that coordinate other skills and track high-level workflow:

1. **design-experiment**
   - Plans experiments (variables, resources, estimates)
   - Documents full pipeline (torchtune + inspect-ai)
   - Creates experiment_summary.md
   - Status: ✅ Updated

2. **scaffold-experiment**
   - Directly implements torchtune and inspect-ai scaffolding logic
   - Modularized into optimizers/ and evaluators/ subdirectories
   - Creates scaffold.log (unified scaffolding tracking)
   - Status: ✅ Modularized (Issue #196)

3. **run-experiment**
   - Calls run-torchtune (waits for completion)
   - Calls run-inspect (sequential, after fine-tuning)
   - Creates run-experiment.log (orchestration tracking)
   - Status: ✅ Updated

4. **analyze-experiment**
   - Future: Analyzes results, generates reports
   - Status: 📋 Planned (placeholder created)

### Modular Skills
Skills with internal modular architecture:

1. **scaffold-experiment** (Issue #196 - Modularized)
   - Core SKILL.md (224 lines) provides high-level workflow
   - Tool-specific logic organized in subdirectories:
     - `optimizers/torchtune.md` (57 lines) + 5 submodules (48-91 lines each)
     - `evaluators/inspect.md` (60 lines) + 4 submodules (66-136 lines each)
   - Support files: templates/, examples/, workflows/
   - Benefits: Easier navigation, cheaper context, less hallucination risk
   - Obsoletes: scaffold-torchtune, scaffold-inspect (deleted)
   - Status: ✅ Segmented into focused modules

### Workers (Fine-tuning)
Skills that handle torchtune-specific operations:

1. **run-torchtune**
   - Submits finetune.slurm jobs to SLURM
   - Monitors jobs until completion (1-min polling)
   - Updates experiment_summary.md status table
   - Creates run-torchtune.log
   - Status: ✅ Created (extracted from run-experiment)

### Workers (Evaluation)
Skills that handle inspect-ai-specific operations:

1. **run-inspect**
   - Verifies fine-tuning complete and checkpoints exist
   - Submits evaluation SLURM jobs
   - Monitors jobs until completion (1-min polling)
   - Updates experiment_summary.md evaluation status table
   - Creates run-inspect.log
   - Status: ✅ Created (new)

### Supporting Skills
Skills that create reusable evaluation tasks:

1. **create-inspect-task**
   - Creates custom inspect-ai evaluation tasks
   - Can read from experiment_summary.md (experiment-guided mode)
   - Can run standalone (for general task creation)
   - Status: ⚠️ Existing (not modified in this refactor)

## Key Design Principles

### 1. Single Responsibility & Modularity
Each skill or module does one thing well:
- scaffold-experiment: Config generation (modularized internally)
  - optimizers/torchtune: Fine-tuning config generation
  - evaluators/inspect: Evaluation config generation
- run-torchtune: Only fine-tuning execution
- run-inspect: Only evaluation execution

### 2. Modular Documentation Pattern
Complex skills use modular documentation structure:
- Core SKILL.md provides high-level workflow (~150-200 lines)
- Detailed logic segmented into focused submodules (~50-100 lines each)
- Support files (templates, examples, workflows) provide reference
- Benefits: Easier to navigate, cheaper context, reduces hallucination
- Example: scaffold-experiment (224 line core + 11 submodules)

### 3. Sequential Dependencies
run-experiment enforces proper execution order:
- run-torchtune MUST complete before run-inspect starts
- Evaluation requires fine-tuned model checkpoints
- Clear dependency management

### 4. Independent Usability
Skills can be run standalone for targeted operations:
- scaffold-experiment (generate all configs)
- run-torchtune (just execute fine-tuning)
- run-inspect (just execute evaluation)

### 5. Comprehensive Logging
Three levels of logs:
- **Orchestration logs**: run-experiment.log
- **Skill logs**: scaffold.log, run-torchtune.log, run-inspect.log
- **Tool logs**: SLURM logs, inspect-ai logs, WandB logs

### 6. Tool Documentation
design-experiment documents which tools are used:
- **torchtune**: Fine-tuning (used by scaffold-experiment/optimizers, run-torchtune)
- **inspect-ai**: Evaluation (used by scaffold-experiment/evaluators, run-inspect)
- **analyze (future)**: Analysis (used by analyze-experiment)

## File Structure After Scaffolding

```
experiment_name/
├── experiment_summary.md         # Design (from design-experiment)
├── design-experiment.log         # Planning log
├── scaffold.log                  # Scaffolding log (from scaffold-experiment)
├── run-experiment.log            # Execution orchestration log
├── run-torchtune.log             # Fine-tuning execution details
├── run-inspect.log               # Evaluation execution details
├── rank8_lr1e-5/                 # Run directory
│   ├── setup_finetune.yaml       # From scaffold-torchtune
│   ├── finetune.yaml             # From scaffold-torchtune
│   ├── finetune.slurm            # From scaffold-torchtune
│   ├── slurm-12345.out           # From run-torchtune
│   └── eval/                     # From scaffold-inspect
│       ├── capitalization_epoch0.slurm  # From scaffold-inspect
│       ├── slurm-12346.out       # From run-inspect
│       └── logs/                 # From run-inspect
│           └── result.eval       # inspect-ai output
├── rank8_lr5e-5/
│   └── ...
└── analyze-experiment.log        # Future: Analysis details
```

## Typical Workflow

### User Perspective (Using Orchestrators)
```bash
# 1. Design the experiment
# Run design-experiment skill

# 2. Generate all configs
# Run scaffold-experiment skill

# 3. Execute everything
# Run run-experiment skill

# 4. Analyze results (future)
# Run analyze-experiment skill
```

### What Actually Happens (Worker Skills)
```bash
# design-experiment
#   → Creates experiment_summary.md with full pipeline docs

# scaffold-experiment
#   → Executes torchtune scaffolding (optimizers/torchtune.md)
#     → Creates directories, setup_finetune.yaml, finetune.yaml, finetune.slurm
#   → Executes inspect-ai scaffolding (evaluators/inspect.md)
#     → Creates eval/ dirs, inspect.slurm scripts

# run-experiment
#   → Calls run-torchtune
#     → Submits finetune.slurm jobs, monitors until complete
#     → ⏸ WAITS for ALL fine-tuning to finish
#   → Calls run-inspect (SEQUENTIAL)
#     → Verifies checkpoints exist
#     → Submits inspect.slurm jobs, monitors until complete

# analyze-experiment (future)
#   → Reads SLURM logs, inspect-ai logs
#   → Generates comparison tables, plots, reports
```

## Benefits of This Architecture

### 1. Modularity
- Easy to add new tools (e.g., dspy, new eval frameworks)
- Can swap implementations without affecting other parts
- Workers are independent, reusable components

### 2. Flexibility
- Users can run full workflow (orchestrators)
- Or run individual steps (scaffold-experiment, run-torchtune, run-inspect)
- Or skip steps (e.g., fine-tuning only, no evaluation)

### 3. Agent-Ready & Context-Efficient
- Skills can be called by automated agents
- Clear interfaces between skills
- Modular documentation reduces context usage (load only needed modules)
- Smaller files reduce hallucination risk

### 4. Debuggability
- Separate logs for each level (orchestration, skills, tools)
- Easy to identify where problems occur
- Can re-run individual steps after fixing issues
- Modular documentation makes finding relevant info faster

### 5. Extensibility
- New workflow stages can be added (e.g., hyperparameter search)
- New tools can be integrated by adding modules to optimizers/ or evaluators/
- Modules can be enhanced without changing core skill logic
- Pattern can be applied to other complex skills (e.g., create-inspect-task)

## Terminology Consistency

### Action Verbs
- **design**: Plan what to do (design-experiment)
- **scaffold**: Generate configs/setup (scaffold-experiment, scaffold-torchtune, scaffold-inspect)
- **run**: Execute jobs (run-experiment, run-torchtune, run-inspect)
- **analyze**: Interpret results (analyze-experiment)

### Naming Pattern
- **Orchestrators**: {verb}-experiment
- **Workers**: {verb}-{tool}
- Clear distinction between coordination and implementation

## Migration Notes

### Changes from Previous Version

**scaffold-experiment (Issue #196 - Modularized):**
- Was: 520-line monolithic SKILL.md + separate worker skills (scaffold-torchtune, scaffold-inspect)
- Now: 224-line core SKILL.md + modular documentation (optimizers/, evaluators/)
- Benefits:
  - 57% reduction in core skill size (520→224 lines)
  - Easier navigation (find relevant info quickly)
  - Cheaper context (load only needed modules)
  - Less hallucination risk (smaller, focused files)
  - Deleted 2 obsolete worker skills (reduced from 9→7 skills)

**run-experiment:**
- Was: Only handled fine-tuning execution
- Now: Orchestrates both fine-tuning and evaluation execution sequentially
- Benefits: Complete workflow automation, proper dependency management

**New skills:**
- run-inspect: Evaluation execution (previously not automated)

### Backward Compatibility

**User impact:**
- Orchestrator skill names unchanged (scaffold-experiment, run-experiment)
- Output structure unchanged (same directories, same files)
- experiment_summary.md format enhanced but compatible

**Developer impact:**
- Must understand modular documentation pattern
- New logging structure (scaffold.log instead of scaffold-torchtune.log + scaffold-inspect.log)
- Can reference specific modules for targeted information

## Future Enhancements

### Short-term
- Apply modular pattern to create-inspect-task (982 lines → ~150 line core + modules)
- Apply modular pattern to run-experiment, run-torchtune, run-inspect if needed
- Implement analyze-experiment skill
- Add dry-run modes
- Improve error recovery and resumability

### Medium-term
- Add dspy integration (scaffold-dspy, run-dspy workers)
- Create visualization dashboards for real-time monitoring
- Add email/Slack notifications on completion

### Long-term
- Automated hyperparameter tuning
- Multi-cluster support
- Results database for cross-experiment comparisons
- LLM-driven experiment design suggestions

## Testing Recommendations

Before deploying to production:
1. Test worker skills independently with small datasets
2. Test orchestrators with small experiments (1-2 runs)
3. Verify log files are created correctly at all levels
4. Check that sequential execution works (run-torchtune → run-inspect)
5. Test error handling (what happens if fine-tuning fails?)
6. Verify experiment_summary.md updates correctly

## Terminology Clarification: Experiments vs Tasks

**Important:** This PR also includes a directory rename from `tasks/` → `experiments/` to eliminate terminology confusion.

### The Problem
The word "task" was overloaded with three different meanings:
1. `tasks/` folder = Research domains (capitalization, twins, etc.)
2. Inspect-ai tasks = Evaluation scripts (`.py` files)
3. Skills refer to "tasks" in experiment_summary.md

### The Solution
Renamed `tasks/` → `experiments/` to clarify:
- **Experiment** = Research domain/type (e.g., capitalization experiment)
- **Task** = Inspect-ai evaluation script (e.g., `cap_task.py`)

### New Structure
```
experiments/                             # Research experiment types
├── capitalization/
│   ├── cap_task.py                     # Inspect-ai evaluation task
│   ├── input/                          # Dataset generation
│   └── templates/                      # Fine-tuning configs
└── synthetic_twins/
    ├── inspect_task_twins.py           # Inspect-ai evaluation task
    └── ...
```

### Benefits
- ✅ No more "task" overload - clear distinction between experiment types and evaluation tasks
- ✅ Matches skill terminology (design-experiment, scaffold-experiment, run-experiment)
- ✅ More accurate: These ARE experimental setups, not just "tasks"
- ✅ Natural hierarchy: Experiments contain tasks, datasets, and configs

## Summary

This architecture creates a clean, modular system that:
- ✅ Separates concerns (torchtune vs inspect-ai)
- ✅ Enables independent operation of components
- ✅ Supports full workflow automation
- ✅ Maintains clear orchestration hierarchy
- ✅ Uses modular documentation for complex skills (Issue #196)
- ✅ Reduces file sizes for better navigation and context efficiency
- ✅ Provides reusable pattern for other skills
- ✅ Uses consistent, intuitive terminology
- ✅ Eliminates "task" overload via experiments/ rename
- ✅ Provides comprehensive logging at all levels

**Recent improvements (Issue #196):**
- scaffold-experiment modularized: 520→224 line core + focused submodules
- Pattern ready for application to other skills (create-inspect-task, run-* skills)
- Reduced skill count from 9→7 (deleted obsolete workers)

All skills are documented and ready for use!
