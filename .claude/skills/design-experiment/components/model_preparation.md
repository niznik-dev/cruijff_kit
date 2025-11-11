# Model Preparation Planning

Design the training runs by determining models, datasets, and hyperparameters.

## Questions to Ask

### Which models?
- Which model(s) to fine-tune? (e.g., 1B, 3B, 8B)
- Check `{models_dir}` from `claude.local.md`

### Which dataset?
- Training dataset location and format
- Required splits: train, validation (optional), test (optional)

### What variables are you testing?
- Different model sizes?
- Different LoRA ranks?
- Different datasets or data sizes?
- Different hyperparameters?
- Combinations of the above?

### Should we include base model controls?
- Controls evaluate base models without fine-tuning to measure the effect of fine-tuning

## Training Configuration

### Basic settings:
- How many epochs? (default: 1-2)
- How many GPUs per job? (default: 1)
- Should validation run during training? (default: yes)
- System prompt for training and evaluation? (default: "")

### Advanced settings (calculate from prior runs if available):
- Batch sizes - estimate from GPU memory usage in prior runs
- Dataset packing - enabled by default, affects batch size
- For help estimating: check `{scratch_dir}/*/slurm-*.out` for similar runs

## Document in All Runs Table

Create a table documenting all fine-tuned and control runs:

```markdown
## All Runs

| Run Name | Model | LoRA Rank | Learning Rate | Batch Size | Type | Est. Time |
|----------|-------|-----------|---------------|------------|------|-----------|
| Llama-3.2-1B_rank8_lr1e-5 | Llama-3.2-1B-Instruct | 8 | 1e-5 | 4 | Fine-tuned | ~10min |
| Llama-3.2-1B_rank8_lr5e-5 | Llama-3.2-1B-Instruct | 8 | 5e-5 | 4 | Fine-tuned | ~10min |
| Llama-3.2-1B_base | Llama-3.2-1B-Instruct | - | - | - | Control | N/A |

**Notes**:
- **Type**: "Fine-tuned" for runs requiring training, "Control" for base model evaluation only
- **Run Name**: Should match directory structure (varying parameters only)
- Include all parameters that vary across runs as separate columns
- Use `-` for non-applicable parameters (like LoRA rank for control runs)
```
