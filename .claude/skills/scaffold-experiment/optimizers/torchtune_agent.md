# scaffold-torchtune Subagent - Preparation

This module contains the specification for launching the scaffold-torchtune subagent, which generates fine-tuning configurations for all runs.

---

## When to Use

Launch this subagent when:
- `tools.preparation` in experiment_summary.yaml is set to `"torchtune"`
- You need to generate fine-tuning configurations for experimental runs

---

## Subagent Invocation

**Subagent type:** `scaffold-torchtune`

**Launch via:** Task tool (use in parallel with evaluation subagent)

---

## Prompt Template

```
Set up torchtune fine-tuning configurations for all FINE-TUNED runs in the experiment located at {experiment_dir}.

Your tasks:
1. Read experiment_summary.yaml to extract run configurations
2. Read claude.local.md for environment-specific settings
3. Identify which runs are fine-tuned (type: "fine-tuned") vs control (type: "control")
4. For ONLY the fine-tuned runs (skip control/base model runs):
   - Create run directory based on run name in experiment_summary.yaml
   - Generate setup_finetune.yaml from appropriate template
   - Execute setup_finetune.py to generate finetune.yaml and finetune.slurm
   - Verify outputs were created successfully
5. For control/base model runs: Create ONLY the run directory (no training configs needed)
6. Create a detailed log at {experiment_dir}/logs/scaffold-torchtune.log
7. Verify that parameters in generated finetune.yaml files match directory names

Report back:
- Summary of all created runs (directory names and what was generated)
- Any errors or warnings encountered
- Verification results showing parameter correctness
- Path to the log file for detailed information
```

---

## What scaffold-torchtune Does

The subagent performs these operations autonomously:

1. **Creates run directories** for all runs using full run names from experiment_summary.yaml
2. **For fine-tuned runs:**
   - Generates `setup_finetune.yaml` configuration file
   - Executes `setup_finetune.py` to create `finetune.yaml` and `finetune.slurm`
   - Validates generated configurations
3. **For control/base runs:**
   - Creates directory only (no training configs needed)
4. **Creates detailed log** at `scaffold-torchtune.log` with complete process information
5. **Verifies parameter correctness** in generated files match expected values

---

## Expected Output Structure

After scaffold-torchtune completes, the experiment directory will contain:

```
{experiment_dir}/
├── Llama-3.2-1B-Instruct_base/    # Control run (directory only)
│   └── (no training configs)
├── Llama-3.2-1B-Instruct_rank4/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
├── Llama-3.2-1B-Instruct_rank8/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
└── logs/scaffold-torchtune.log
```

---

## Subagent Report Format

When scaffold-torchtune completes, it will report back with:

**On success:**
- Number of fine-tuned runs configured
- Number of control runs (directories created)
- List of created run directories
- Verification results (parameters match expected values)
- Path to scaffold-torchtune.log

**On failure:**
- Which runs failed and why
- Error messages from setup scripts
- Path to scaffold-torchtune.log for detailed diagnostics

---

## Error Handling

**If scaffold-torchtune fails:**
- The subagent will report errors in its response
- Log the failure in orchestration log (see [logging.md](../logging.md))
- Ask user if they want to continue with evaluation scaffolding anyway
- Report the failure in final summary
- Direct user to scaffold-torchtune.log for details

**Common failure scenarios:**
- Missing model files at specified paths
- Invalid parameters in experiment_summary.yaml
- setup_finetune.py execution errors
- Missing conda environment specified in claude.local.md
- Permission issues creating directories

---

## Integration with Orchestrator

1. **Launched by:** scaffold-experiment orchestrator (SKILL.md)
2. **Runs in parallel with:** scaffold-inspect subagent
3. **Logs to:** scaffold-torchtune.log (detailed), scaffold-experiment.log (high-level status)
4. **Reports back to:** scaffold-experiment orchestrator upon completion
