# design-experiment

Interactive skill for planning LLM fine-tuning and evaluation experiments.

## Purpose

Create tool-agnostic experiment plans that specify:
- Which models to fine-tune and evaluate
- Which datasets and evaluation tasks to use
- Complete configuration for downstream skills
- Verified resource availability

**Outputs:**
- `experiment_summary.yaml` - Structured configuration consumed by scaffold-experiment and run-experiment
- `design-experiment.log` - Human-readable audit log for reproducibility

## Meta-Pattern: param_selection → validation → generation

This skill uses **action verbs** consistent with scaffold/run, but simplified because it handles one workflow (not multiplexed by tools).

### Why This Pattern?

**design-experiment PURPOSE:** Create tool-agnostic plans that OTHER skills execute
- Doesn't execute with torchtune or inspect-ai directly
- One linear conversation flow (not branching by tool type)

**Action verbs match scaffold/run:**
- **param_selection** (like scaffold's config_selection)
- **validation** (same as scaffold/run)
- **generation** (like scaffold's config_generation)

**Simpler than scaffold/run because:**
- scaffold/run: 20 files (2 tools × ~7 modules each + shared)
- design: 7 files (1 workflow, not tool-multiplexed)

### Pattern Structure

```
design-experiment/
├── SKILL.md                        (lean orchestrator)
├── param_selection.md              (interactive workflow)
├── validation.md                   (completeness checklist)
├── experiment_generation.md        (create outputs)
├── logging.md                      (cross-cutting)
├── templates/
│   └── experiment_summary.yaml    (YAML structure)
└── README.md                       (this file)
```

## Workflow Stages

### Stage 1: PARAM_SELECTION (param_selection.md)

Interactive conversation to gather all parameters:
1. **Determine location** - Auto-detect sanity_check vs experiment
2. **Understand experiment** - Scientific question, variables
3. **Confirm tools** - torchtune (preparation), inspect-ai (evaluation)
4. **Design training** - Models, datasets, hyperparameters
5. **Design evaluation** - Tasks, epochs, evaluation matrix
6. **Establish naming** - Experiment and run names
7. **Verify resources** - Check models, datasets, scripts exist
8. **Get approval** - Present plan to user for questions and improvements 
9. **Create files** - Proceed to generation

### Stage 2: VALIDATION (validation.md)

Before presenting to user (step 8), verify plan completeness:
- Preparation: Run names, parameters, configurations
- Evaluation: Epochs (0-indexed!), system prompt consistency (critical!)
- Resources: All verified and accessible

### Stage 3: GENERATION (experiment_generation.md)

After approval, create outputs:
- `experiment_summary.yaml` (structured configuration)
- `design-experiment.log` (human-readable audit trail)

After outputs are created, suggest the next step in the pipeline: scaffold-experiment.

## Cross-Cutting Concerns

### Logging (logging.md)

Creates human-readable audit log in `.log` format.

Throughout workflow, log structured events:
- Resource verification (models, datasets, eval tasks)
- Prior run searches and data extraction
- Decisions (naming, recipe, configuration)
- File creation and validation

**Format:** Plain text with timestamped action entries - each entry has a `[TIMESTAMP] ACTION_TYPE` header, key/value detail lines, and a result.

**Not a template** - it's guidance on HOW to log. Lives at top level because it's used during multiple stages.

### Templates (templates/)

Output structure reference:
- `experiment_summary.yaml` - Structured schema with required/optional fields and examples

## Key Principles

1. **Tool-agnostic planning** - Plan WHAT to do, not HOW to execute
2. **Action verb pattern** - Matches scaffold/run for consistency
3. **Structured output** - YAML for machine parsing, plain text logs for audit trail
4. **Resource verification** - Verify models, datasets, and eval tasks exist
5. **Validation before presentation** - Ensure plan is complete

## Integration

**Upstream:** User conversation

**Downstream:**
- `scaffold-experiment` reads experiment_summary.yaml to generate configs
- `run-experiment` reads experiment_summary.yaml to track progress
- `analyze-experiment` (planned) reads experiment_summary.yaml to interpret results

## Module Guidelines

### param_selection.md
- Complete interactive conversation flow
- Ask questions, don't assume
- Document what goes in experiment_summary.yaml
- Include conversation patterns
- Reference logging.md for what to log

### validation.md
- Checklist for each aspect (preparation, evaluation, resources)
- Identify common issues (system prompt mismatch, 1-indexed epochs)
- Don't block on missing resources - document as prerequisites

### experiment_generation.md
- File creation instructions
- Next steps conversation
- Prerequisites handling

### logging.md
- Cross-cutting: used during selection AND generation
- Complete format spec with examples
- When to log, what to log, what NOT to log

### templates/
- Output structures only (not guidance)
- Complete examples for complex sections

## Notes

- This skill is tool-agnostic TODAY (only torchtune/inspect-ai exist) but structured to support multiple tools in the future
- Logging is critical - design-experiment.log enables debugging, reproducibility, and improvement
- System prompt consistency between training and evaluation is critical for inspect-ai
- Validation before presenting ensures we don't waste user's time with incomplete plans
- Pattern consistency with scaffold/run (action verbs) while being appropriately simpler (fewer files)
