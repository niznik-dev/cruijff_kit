# Evaluation Planning

Design the evaluation runs by determining tasks, epochs, and which runs get evaluated.

## Questions to Ask

### Which evaluation tasks?
- Which inspect-ai task(s) to run?
- For each task: name, script path, dataset path (if different from training), description
- Does the task exist or need to be created? (use `create-inspect-task` skill if needed)

### Which epochs to evaluate?
- **NOTE:** Epochs are 0-indexed. Training for 1 epoch produces `epoch_0`, training for 2 epochs produces `epoch_0` and `epoch_1`, etc.
- Last epoch only (default, most efficient)
  - After 1 epoch of training, this is `epoch_0`
  - After 2 epochs of training, this is `epoch_1`
- All epochs (compare training progression)
- Specific epochs (e.g., epoch 0 and final)
- Best by validation loss (requires validation during training)

### Which runs get which evaluations?
- All runs on all tasks (typical)
- Selective (e.g., only large models on expensive evals)
- If selective, create evaluation matrix

### Evaluation datasets:
- Same as training dataset (typical for overfitting checks)
- Different test set (typical for generalization evaluation)
- Multiple evaluation datasets (comprehensive assessment)

### Evaluation configuration:
- System prompt must match training for consistency
- Temperature typically 0.0 for deterministic evaluation
- Scorer selection (exact match, includes, model-graded, etc.)

**Important:** Base models evaluate once per task (no epoch suffix), fine-tuned models evaluate per epoch.

## Document in experiment_summary.md

### Evaluation Matrix Example (when runs have different evaluation plans):

```markdown
## Evaluation Plan

### Evaluation Matrix
| Run Name | capitalization_task | reasoning_task | Notes |
|----------|---------------------|----------------|-------|
| Llama-3.2-1B_rank4 | ✓ epoch 0,1 | ✓ epoch 0,1 | All evals |
| Llama-3.2-3B_rank4 | ✓ epoch 0,1 | - | Cap only |
| Llama-3.2-1B_base | ✓ | ✓ | Base control |
```
