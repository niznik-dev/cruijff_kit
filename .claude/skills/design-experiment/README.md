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
- `design-experiment.jsonl` - Machine-readable audit log for reproducibility

## Meta-Pattern: param_selection → validation → generation

This skill uses **action verbs** consistent with scaffold/run, but simplified because it handles one workflow (not multiplexed by tools).

### Why This Pattern?

**design-experiment PURPOSE:** Create tool-agnostic plans that OTHER skills execute
- Doesn't execute with torchtune or inspect-ai directly
- Creates experiment_summary.md that scaffold/run skills read
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
├── SKILL.md                        (119 lines - lean orchestrator)
├── param_selection.md              (~350 lines - interactive workflow)
├── validation.md                   (~120 lines - completeness checklist)
├── experiment_generation.md        (~80 lines - create outputs)
├── logging.md                      (~100 lines - cross-cutting)
├── templates/
│   └── experiment_summary.yaml    (~200 lines - YAML structure)
└── README.md                       (this file)
```

**Total: 7 files**

## Workflow Stages

### Stage 1: PARAM_SELECTION (param_selection.md)

Interactive conversation to gather all parameters:
1. **Determine type/location** - Auto-detect sanity_check vs experiment
2. **Understand experiment** - Scientific question, variables
3. **Confirm tools** - torchtune (preparation), inspect-ai (evaluation)
4. **Design training** - Models, datasets, hyperparameters
5. **Design evaluation** - Tasks, epochs, evaluation matrix
6. **Establish naming** - Experiment and run names
7. **Verify resources** - Check models, datasets, scripts exist
8. **Get approval** - Present plan (after validation)
9. **Create files** - Proceed to generation

### Stage 2: VALIDATION (validation.md)

Before presenting to user (step 8), verify plan completeness:
- Preparation: Run names, parameters, configurations
- Evaluation: Epochs (0-indexed!), system prompt consistency (critical!)
- Resources: All verified and accessible

### Stage 3: GENERATION (experiment_generation.md)

After approval, create outputs:
- `experiment_summary.yaml` (structured configuration)
- `design-experiment.jsonl` (machine-readable audit trail)

Then suggest next steps (scaffold-experiment).

## Cross-Cutting Concerns

### Logging (logging.md)

Creates machine-readable audit log in `.jsonl` format.

Throughout workflow, log structured events:
- Resource verification (models, datasets, eval tasks)
- Prior run searches and data extraction
- Decisions (naming, recipe, configuration)
- File creation and validation

**Format:** JSON Lines - each line is a complete JSON object with timestamp, action, result, and action-specific fields.

**Not a template** - it's guidance on HOW to log. Lives at top level because it's used during multiple stages.

### Templates (templates/)

Output structure reference:
- `experiment_summary.yaml` - Structured schema with required/optional fields and examples

## File Organization

| Category | Files | Purpose | Lines |
|----------|-------|---------|-------|
| Orchestrator | SKILL.md | Coordinate workflow | 119 |
| Selection | param_selection.md | Gather parameters | ~350 |
| Validation | validation.md | Verify completeness | ~120 |
| Generation | experiment_generation.md | Create outputs | ~80 |
| Cross-cutting | logging.md | JSONL log format spec | ~100 |
| Templates | templates/experiment_summary.yaml | YAML schema & examples | ~200 |
| Documentation | README.md | Pattern explanation | - |

**Total:** 7 files, ~970 lines (down from 553 monolithic lines)

**Why more lines total?** Because we extracted embedded templates and added comprehensive guidance. The SKILL.md orchestrator is leaner (119 vs 553), and modules are focused and maintainable.

## Key Principles

1. **Tool-agnostic planning** - Plan WHAT to do, not HOW to execute
2. **Action verb pattern** - Matches scaffold/run for consistency
3. **Simpler structure** - 7 files vs 20 (because 1 workflow, not 2 tools)
4. **Structured output** - YAML for machine parsing, JSONL for audit logs
5. **Resource verification** - Verify models, datasets, and eval tasks exist
6. **Validation before presentation** - Ensure plan is complete

## Integration

**Upstream:** User conversation

**Downstream:**
- `scaffold-experiment` reads experiment_summary.yaml to generate configs
- `run-experiment` reads experiment_summary.yaml to track progress
- `analyze-experiment` (planned) reads experiment_summary.yaml to interpret results

## Comparison to Other Patterns

| Skill | Pattern | Files | Structure |
|-------|---------|-------|-----------|
| scaffold-experiment | optimizers/evaluators | 20 | Tool-specific (torchtune, inspect-ai) |
| run-experiment | optimizers/evaluators | 20 | Tool-specific (torchtune, inspect-ai) |
| design-experiment | selection/validation/generation | 7 | Tool-agnostic workflow |

**Key insight:**
- scaffold/run use **optimizers/evaluators** because they handle 2 tools
- design uses **selection/validation/generation** because it's 1 workflow
- All use **action verbs** for consistency
- Structure reflects PURPOSE, not blind pattern reuse

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

## Success Metrics

**Before refactoring:**
- SKILL.md: 553 lines (monolithic)
- Mixed concerns
- Hard to navigate

**After refactoring:**
- SKILL.md: 119 lines (78% reduction) ✓
- Action verb pattern matching scaffold/run ✓
- Appropriately simpler (7 vs 20 files) ✓
- Clear separation of concerns ✓
- Easy to maintain and extend ✓

## Notes

- This skill is tool-agnostic TODAY (only torchtune/inspect-ai exist) but structured to support multiple tools in the future
- Logging is critical - design-experiment.jsonl enables debugging, reproducibility, and improvement
- System prompt consistency between training and evaluation is critical for inspect-ai
- Validation before presenting ensures we don't waste user's time with incomplete plans
- Pattern consistency with scaffold/run (action verbs) while being appropriately simpler (fewer files)
