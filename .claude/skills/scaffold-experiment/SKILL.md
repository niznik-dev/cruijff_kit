# Scaffold Experiment

Automatically generate all experimental infrastructure - fine-tuning and evaluation configurations for all runs.

## Your Task

Read experiment_summary.md to identify tools being used, then:
1. Generate model optimizer configs (e.g., torchtune fine-tuning)
2. Generate model evaluator configs (e.g., inspect-ai evaluation)

## Prerequisites

- experiment_summary.md exists (from design-experiment skill)
- claude.local.md exists with environment settings

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains experiment_summary.md
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Workflow

### High-Level Steps

1. **Verify prerequisites** - Ensure experiment_summary.md and claude.local.md exist
2. **Read tool specifications** - Parse "Tools" section to identify optimizer (torchtune) and evaluator (inspect-ai)
3. **Scaffold optimizer** - Generate fine-tuning configs for all runs
4. **Scaffold evaluator** - Generate evaluation configs for all runs
5. **Create log** - Document process in scaffold.log
6. **Report summary** - Show user complete status

### Detailed Orchestration

For step-by-step execution:
- **Torchtune scaffolding**: See [workflows/torchtune.md](workflows/torchtune.md)
- **Inspect-ai scaffolding**: See [workflows/inspect.md](workflows/inspect.md)

For technical details:
- **Optimizer modules**: See [optimizers/](optimizers/) (torchtune logic)
- **Evaluator modules**: See [evaluators/](evaluators/) (inspect-ai logic)

For code templates:
- **SLURM script**: [templates/slurm_template.sh](templates/slurm_template.sh)
- **setup_finetune.yaml**: [templates/setup_finetune_template.yaml](templates/setup_finetune_template.yaml)

For concrete example:
- **Full walkthrough**: [examples/complete_example.md](examples/complete_example.md)

## Expected Outputs

After successful scaffolding:

```
{experiment_dir}/
├── scaffold.log
├── r8_lr1e-5/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
└── r16_lr5e-5/
    └── ...
```

## Validation Checklist

Before reporting success, verify:
- ✓ All run directories created with correct names
- ✓ Each run has setup_finetune.yaml, finetune.yaml, finetune.slurm
- ✓ **Parameters in finetune.yaml match directory names** (critical!)
- ✓ Evaluation directories exist with expected structure
- ✓ Each evaluation has corresponding SLURM script
- ✓ scaffold.log created with complete process details

## Error Handling

**If experiment_summary.md not found:**
- Suggest running design-experiment skill first
- Do not proceed

**If prerequisites missing:**
- Report which files are missing
- Ask user to provide or create them

**Partial success is acceptable:**
- Some runs fail but others succeed → Report both
- Fine-tuning configs generated but eval fails → Still useful
