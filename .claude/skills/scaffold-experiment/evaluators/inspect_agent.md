# scaffold-inspect Subagent - Evaluation

This module contains the specification for launching the scaffold-inspect subagent, which generates evaluation configurations for all runs.

---

## When to Use

Launch this subagent when:
- `tools.evaluation` in experiment_summary.yaml is set to `"inspect-ai"`
- You need to generate evaluation configurations for experimental runs

---

## Subagent Invocation

**Subagent type:** `scaffold-inspect`

**Launch via:** Task tool (use in parallel with preparation subagent)

---

## Prompt Template

```
Set up inspect-ai evaluation configurations for all runs in the experiment located at {experiment_dir}.

Your tasks:
1. Read experiment_summary.yaml to extract evaluation configurations
2. Read claude.local.md for environment-specific settings
3. Verify that inspect-ai task scripts exist at the specified paths
4. For each run and evaluation combination:
   - Create eval/ subdirectory in the run directory
   - Generate inspect.slurm script with correct model paths and task parameters
   - Configure output locations
5. Create a detailed log at {experiment_dir}/scaffold-inspect.log

Report back:
- Summary of all created evaluation scripts (paths)
- Any errors or warnings encountered
- Verification results for task script existence
- Path to the log file for detailed information
```

---

## What scaffold-inspect Does

The subagent performs these operations autonomously:

1. **Parses evaluation plan** from experiment_summary.yaml (which runs, which tasks, which epochs)
2. **Verifies task scripts exist** at paths specified in experiment_summary.yaml
3. **Creates `eval/` subdirectories** in each run directory
4. **Generates inspect.slurm scripts** for each evaluation with:
   - Correct model paths (base model or fine-tuned checkpoint)
   - Task script references
   - System prompt and evaluation parameters
   - Output directory configuration
5. **Creates detailed log** at `scaffold-inspect.log` with complete process information

---

## Expected Output Structure

After scaffold-inspect completes, the experiment directory will contain:

```
{experiment_dir}/
├── Llama-3.2-1B-Instruct_base/    # Control run
│   └── eval/
│       ├── capitalization_base.slurm
│       └── logs/
├── Llama-3.2-1B-Instruct_rank4/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
├── Llama-3.2-1B-Instruct_rank8/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
├── scaffold-torchtune.log
└── scaffold-inspect.log
```

---

## Subagent Report Format

When scaffold-inspect completes, it will report back with:

**On success:**
- Number of evaluation scripts created
- List of created eval directories and scripts
- Task script verification results
- Path to scaffold-inspect.log

**On failure:**
- Which evaluations failed and why
- Missing task scripts
- Path to scaffold-inspect.log for detailed diagnostics

---

## Error Handling

**If scaffold-inspect fails:**
- The subagent will report errors in its response
- Log the failure in orchestration log (see [logging.md](../logging.md))
- Note which evaluations couldn't be scaffolded
- Fine-tuning can still proceed (evaluation optional)
- Report the failure in final summary
- Direct user to scaffold-inspect.log for details

**Common failure scenarios:**
- Missing inspect-ai task scripts at specified paths
- Invalid evaluation matrix in experiment_summary.yaml
- Permission issues creating eval/ directories
- Missing environment variables in claude.local.md

---

## Integration with Orchestrator

1. **Launched by:** scaffold-experiment orchestrator (SKILL.md)
2. **Runs in parallel with:** scaffold-torchtune subagent
3. **Logs to:** scaffold-inspect.log (detailed), scaffold-experiment.log (high-level status)
4. **Reports back to:** scaffold-experiment orchestrator upon completion