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
│  SCAFFOLD   │  scaffold-experiment (orchestrator)
│             │  ├─> scaffold-torchtune (worker)
│             │  └─> scaffold-inspect (worker)
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
   - Calls scaffold-torchtune to generate fine-tuning configs
   - Calls scaffold-inspect to generate evaluation configs
   - Creates scaffold-experiment.log (orchestration tracking)
   - Status: ✅ Updated

3. **run-experiment**
   - Calls run-torchtune (waits for completion)
   - Calls run-inspect (sequential, after fine-tuning)
   - Creates run-experiment.log (orchestration tracking)
   - Status: ✅ Updated

4. **analyze-experiment**
   - Future: Analyzes results, generates reports
   - Status: 📋 Planned (placeholder created)

### Workers (Fine-tuning)
Skills that handle torchtune-specific operations:

1. **scaffold-torchtune**
   - Creates run directories
   - Generates setup_finetune.yaml files
   - Executes setup_finetune.py to create finetune.yaml + finetune.slurm
   - Creates scaffold-torchtune.log
   - Status: ✅ Created (extracted from scaffold-experiment)

2. **run-torchtune**
   - Submits finetune.slurm jobs to SLURM
   - Monitors jobs until completion (1-min polling)
   - Updates experiment_summary.md status table
   - Creates run-torchtune.log
   - Status: ✅ Created (extracted from run-experiment)

### Workers (Evaluation)
Skills that handle inspect-ai-specific operations:

1. **scaffold-inspect**
   - Creates eval/ subdirectories
   - Generates inspect.slurm scripts for each evaluation
   - Verifies inspect-ai task scripts exist
   - Creates scaffold-inspect.log
   - Status: ✅ Created (new)

2. **run-inspect**
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

### 1. Single Responsibility
Each worker skill does one thing well:
- scaffold-torchtune: Only fine-tuning config generation
- scaffold-inspect: Only evaluation config generation
- run-torchtune: Only fine-tuning execution
- run-inspect: Only evaluation execution

### 2. Orchestration Pattern
Orchestrators coordinate without implementing:
- Call worker skills in proper order
- Track high-level flow and timing
- Aggregate results
- Create orchestration logs (separate from worker logs)

### 3. Sequential Dependencies
run-experiment enforces proper execution order:
- run-torchtune MUST complete before run-inspect starts
- Evaluation requires fine-tuned model checkpoints
- Clear dependency management

### 4. Independent Usability
All worker skills can be run standalone:
- scaffold-torchtune (just generate fine-tuning configs)
- scaffold-inspect (just generate evaluation configs)
- run-torchtune (just execute fine-tuning)
- run-inspect (just execute evaluation)

### 5. Comprehensive Logging
Three levels of logs:
- **Orchestration logs**: scaffold-experiment.log, run-experiment.log
- **Worker logs**: scaffold-torchtune.log, run-torchtune.log, scaffold-inspect.log, run-inspect.log
- **Tool logs**: SLURM logs, inspect-ai logs, WandB logs

### 6. Tool Documentation
design-experiment now documents which tools are used:
- **torchtune**: Fine-tuning (used by scaffold-torchtune, run-torchtune)
- **inspect-ai**: Evaluation (used by scaffold-inspect, run-inspect)
- **analyze (future)**: Analysis (used by analyze-experiment)

## File Structure After Scaffolding

```
experiment_name/
├── experiment_summary.md         # Design (from design-experiment)
├── design-experiment.log         # Planning log
├── scaffold-experiment.log       # Scaffolding orchestration log
├── scaffold-torchtune.log        # Fine-tuning scaffolding details
├── scaffold-inspect.log          # Evaluation scaffolding details
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
#   → Calls scaffold-torchtune
#     → Creates directories, setup_finetune.yaml, finetune.yaml, finetune.slurm
#   → Calls scaffold-inspect
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
- Or run individual steps (workers)
- Or skip steps (e.g., fine-tuning only, no evaluation)

### 3. Agent-Ready
- Workers can be called by automated agents
- Clear interfaces between skills
- Each skill has single, well-defined output

### 4. Debuggability
- Separate logs for each level (orchestration, workers, tools)
- Easy to identify where problems occur
- Can re-run individual steps after fixing issues

### 5. Extensibility
- New workflow stages can be added (e.g., hyperparameter search)
- New tools can be integrated (e.g., different eval frameworks)
- Workers can be enhanced without changing orchestrators

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

**scaffold-experiment:**
- Was: Generated both fine-tuning and evaluation configs directly
- Now: Orchestrates two separate worker skills
- Benefits: Cleaner separation, easier to debug, can run workers independently

**run-experiment:**
- Was: Only handled fine-tuning execution
- Now: Orchestrates both fine-tuning and evaluation execution sequentially
- Benefits: Complete workflow automation, proper dependency management

**New skills:**
- scaffold-inspect: Evaluation config generation (previously inline in scaffold-experiment)
- run-inspect: Evaluation execution (previously not automated)

### Backward Compatibility

**User impact:**
- Orchestrator skill names unchanged (scaffold-experiment, run-experiment)
- Output structure unchanged (same directories, same files)
- experiment_summary.md format enhanced but compatible

**Developer impact:**
- Can now call worker skills directly for targeted operations
- Must understand orchestrator vs. worker distinction
- New logging structure (more files, but better organized)

## Future Enhancements

### Short-term
- Implement analyze-experiment skill
- Add dry-run modes to orchestrators
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

## Summary

This refactoring creates a clean, modular architecture that:
- ✅ Separates concerns (torchtune vs inspect-ai)
- ✅ Enables independent operation of components
- ✅ Supports full workflow automation
- ✅ Maintains clear orchestration hierarchy
- ✅ Prepares for future analyze-experiment implementation
- ✅ Uses consistent, intuitive terminology
- ✅ Provides comprehensive logging at all levels

All skills are documented and ready for use!
