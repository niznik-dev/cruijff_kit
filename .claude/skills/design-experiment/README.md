# design-experiment

Interactive skill for planning LLM fine-tuning and evaluation experiments.

## Purpose

Create tool-agnostic experiment plans that specify:
- Which models to fine-tune and evaluate
- Which datasets and evaluation tasks to use
- Resource estimates (time, disk space, GPU hours)
- Complete configuration for downstream skills

**Output:** `experiment_summary.md` that scaffold-experiment and run-experiment skills consume

## Meta-Pattern: Components + Validation

This skill uses a **components + validation** pattern because it performs **tool-agnostic planning**, not tool-specific implementation.

### Why This Pattern?

**design-experiment PURPOSE:** Create plans that OTHER skills execute
- Doesn't execute with torchtune or inspect-ai directly
- Creates experiment_summary.md that scaffold/run skills read
- Organized by PLAN SECTIONS, not EXECUTION TOOLS

**Contrast with scaffold-experiment and run-experiment:**
- Those skills use **optimizers/evaluators** pattern
- They perform tool-specific implementation (torchtune, inspect-ai)
- They execute code and generate configs/jobs

### Pattern Structure

```
design-experiment/
├── SKILL.md (147 lines - lean orchestrator)
├── components/           ← Plan sections (NOT tools!)
│   ├── experiment_metadata.md
│   ├── tool_selection.md
│   ├── model_preparation.md
│   ├── evaluation_plan.md
│   ├── resources.md
│   └── estimation.md
├── validation/           ← Completeness checks
│   ├── preparation_validation.md
│   ├── evaluation_validation.md
│   └── resources_validation.md
├── workflows/            ← Conversation patterns
│   └── interactive_planning.md
└── templates/            ← Output structure
    └── experiment_summary_template.md
```

## Workflow Stages

### Stage 1: GATHERING (components/)
Collect user requirements interactively:
1. **experiment_metadata.md** - Determine type, location, naming
2. **tool_selection.md** - Confirm which tools (torchtune, inspect-ai)
3. **model_preparation.md** - Design training runs
4. **evaluation_plan.md** - Design evaluation runs
5. **resources.md** - Verify everything exists
6. **estimation.md** - Calculate time, disk, GPU hours

### Stage 2: VALIDATION (validation/)
Verify plan is feasible:
- **preparation_validation.md** - Runs table is complete
- **evaluation_validation.md** - Eval plan is consistent
- **resources_validation.md** - All resources verified

### Stage 3: ESTIMATION (components/estimation.md)
Calculate resource requirements from prior runs when possible

### Stage 4: DOCUMENTATION
Generate experiment_summary.md using template

## File Organization

| Category | Files | Purpose | Avg Lines |
|----------|-------|---------|-----------|
| Orchestrator | SKILL.md | Coordinate workflow | 147 |
| Planning | components/*.md | Gather requirements | 69 |
| Validation | validation/*.md | Verify completeness | 44 |
| Workflow | workflows/*.md | Conversation patterns | 128 |
| Templates | templates/*.md | Output structure | 206 |

**Total:** 12 files, 1029 lines (down from 553 monolithic lines)

## Key Principles

1. **Tool-agnostic planning** - Plan WHAT to do, not HOW to execute
2. **Comprehensive logging** - Log all verifications and calculations in design-experiment.log
3. **Resource verification** - Check everything exists before committing
4. **Conservative estimation** - Use prior runs when available, be cautious otherwise
5. **Validation before presentation** - Ensure plan is complete before showing user

## Integration

**Upstream:** User conversation

**Downstream:**
- `scaffold-experiment` reads experiment_summary.md to generate configs
- `run-experiment` reads experiment_summary.md to track progress
- `analyze-experiment` (planned) reads experiment_summary.md to interpret results

## Comparison to Other Patterns

| Skill | Pattern | Structure | Purpose |
|-------|---------|-----------|---------|
| design-experiment | components/validation | Plan sections | Tool-agnostic planning |
| scaffold-experiment | optimizers/evaluators | Tool directories | Tool-specific config generation |
| run-experiment | optimizers/evaluators | Tool directories | Tool-specific job execution |

**Key insight:** Structure reflects PURPOSE, not blind pattern reuse.

## Module Guidelines

### Planning Components
- Focus on ONE aspect of the plan
- Ask questions, don't make assumptions
- Document what should go in experiment_summary.md
- Keep under 150 lines

### Validation Modules
- Provide checklists for specific aspects
- Identify what makes a plan "complete"
- Flag common mistakes (e.g., 1-indexed epochs)
- Keep focused and concise

### Workflow Narrative
- Describe conversation flow
- Provide example dialogue
- Link to relevant components
- Show both gathering and validation stages

### Templates
- Provide complete structure reference
- Show examples for complex sections
- Document required vs optional sections
- Explain critical requirements (e.g., system prompt consistency)

## Success Metrics

**Before modularization:**
- SKILL.md: 553 lines (monolithic)
- Hard to navigate
- Embedded templates
- Mixed concerns

**After modularization:**
- SKILL.md: 147 lines (73% reduction) ✓
- All modules < 150 lines ✓
- Clear separation of concerns ✓
- Purpose-driven organization ✓
- Easy to maintain and extend ✓

## Notes

- This skill is tool-agnostic TODAY (only torchtune/inspect-ai exist) but structured to support multiple tools in the future
- Logging is critical - design-experiment.log enables debugging, reproducibility, and improvement
- System prompt consistency between training and evaluation is critical for inspect-ai
- Validation before presenting ensures we don't waste user's time with incomplete plans
