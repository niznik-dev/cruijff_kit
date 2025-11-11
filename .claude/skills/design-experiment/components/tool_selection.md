# Tool Selection

Confirm which tools will be used for model preparation and evaluation.

## Available Tools

This skill documents experimental workflows that use specific tools at each stage:

### 1. Model Preparation: torchtune (current)
- Used by: `scaffold-torchtune` and `run-torchtune` skills
- Generates: `finetune.yaml`, `finetune.slurm`
- Produces: Model checkpoints in `output_dir_base`

### 2. Evaluation: inspect-ai (current)
- Used by: `scaffold-inspect` and `run-inspect` skills
- Generates: `inspect.slurm` and/or inspect task scripts
- Produces: Evaluation logs (`.eval` files)

### 3. Analysis: (future)
- Used by: `analyze-experiment` skill (planned)
- Produces: Comparison tables, plots, reports

## Questions to Ask

**Which tools will you use for this experiment?**

**Model preparation:**
- torchtune (currently the only option)
- *Future:* Other fine-tuning frameworks may be supported

**Evaluation:**
- inspect-ai (currently the only option)
- *Future:* Other evaluation frameworks may be supported

**Note:** While these are currently the only options, explicitly confirming and documenting tool choices now will make it easier to support multiple tools in future iterations.

## Document in experiment_summary.md

Include a **Tools** section that documents which tools are used:

```markdown
## Tools

- **Model Preparation:** torchtune
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-torchtune` and `run-torchtune` skills

- **Evaluation:** inspect-ai
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-inspect` and `run-inspect` skills
```
