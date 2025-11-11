# Interactive Planning Workflow

This document describes the step-by-step conversation flow for guiding the user through experiment design.

## 10-Step Workflow

### 1. Determine experiment type and location
- Auto-detect sanity_check vs experiment based on working directory
- Set base directory appropriately
- Confirm with user
- See: `components/experiment_metadata.md`

### 2. Understand the experiment
- What variables are being tested?
- What's the scientific question?
- Should we include base model controls?

### 3. Confirm tool choices
- Ask which preparation tool (currently: torchtune only)
- Ask which evaluation tool (currently: inspect-ai only)
- Document for future multi-tool support
- See: `components/tool_selection.md`

### 4. Design training runs
- Which models? Which datasets? What hyperparameters?
- What LoRA ranks? How many epochs?
- Create comprehensive runs table
- See: `components/model_preparation.md`

### 5. Design evaluation runs
- Which trained models on which tasks?
- Which epochs to evaluate?
- What metrics and scorers?
- Create evaluation matrix if selective
- See: `components/evaluation_plan.md`

### 6. Establish naming
- Choose descriptive experiment name
- Define run naming pattern
- Follow conventions (task_factor_date format)
- See: `components/experiment_metadata.md`

### 7. Verify resources
- Check that models exist
- Check that datasets exist
- Check that eval scripts exist (or note as prerequisites)
- Log all verification steps
- See: `components/resources.md`

### 8. Estimate resources
- Calculate time for BOTH training and evaluation
- Calculate disk space for checkpoints
- Calculate total GPU hours
- Use prior runs when available
- Log all calculations
- See: `components/estimation.md`

### 9. Get approval
- Present the complete plan to user
- Show all runs, evaluations, estimates
- Adjust if needed based on feedback
- Validate completeness before presenting
- See: `validation/` modules

### 10. Create files
- After approval, write `experiment_summary.md`
- Write `design-experiment.log` with all verification steps
- Ask about next steps (scaffold-experiment?)
- See: `templates/experiment_summary_template.md`

## Conversation Patterns

### Opening
```
I'll help you design this experiment. Let me start by understanding what you want to test.

I see you're working in [directory]. This looks like a [sanity check / research experiment].
I'll create the experiment in: {base_dir}{experiment_name}/

What scientific question are you trying to answer?
```

### During Design
```
Great! So you're testing [variable] across [levels].

Let me verify the models exist...
[checks and logs]

Now let's estimate how long this will take. I'll look for similar prior runs...
[searches, extracts, calculates, logs]
```

### Before Approval
```
Here's the complete experiment plan:

**Overview:**
- X fine-tuning runs (varying [factors])
- Y evaluation tasks
- Z total evaluations

**Estimated resources:**
- Training time: ~X hours
- Eval time: ~Y minutes
- Disk space: ~Z GiB
- Total GPU hours: ~W

Does this look correct? Any adjustments needed?
```

### After Approval
```
Perfect! I've created:
- experiment_summary.md with the complete plan
- design-experiment.log with all verification steps and calculations

Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all the configs and SLURM scripts.
```

## Key Principles

1. **Ask, don't assume** - Even when you can auto-detect, confirm with user
2. **Log everything** - All verifications and calculations go in design-experiment.log
3. **Validate before presenting** - Use validation/ modules to ensure plan is complete
4. **Be conservative** - When estimating without prior data, give ranges and mark as preliminary
5. **Handle missing resources gracefully** - Note as prerequisites, don't block the plan
6. **System prompt consistency** - Critical for inspect-ai, verify it matches between training and eval
