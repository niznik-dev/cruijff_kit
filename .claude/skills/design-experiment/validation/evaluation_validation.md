# Evaluation Validation

Validate that the evaluation plan is complete and consistent.

## What to Check

### Task coverage
- All requested evaluation tasks are listed
- Task scripts exist (or noted as prerequisites)
- Evaluation datasets are defined

### Epoch consistency
- Epochs are 0-indexed (epoch_0, epoch_1, etc.)
- Epoch selection is appropriate:
  - Last epoch for efficiency
  - All epochs for progression analysis
  - Specific epochs as requested
- Base models don't have epoch suffixes
- Fine-tuned models specify which epochs to evaluate

### Evaluation matrix completeness
- All runs have evaluation assignments
- Selective evaluations are documented clearly
- Matrix shows which runs × which tasks × which epochs

### Configuration consistency
- **Critical:** System prompt matches between training and evaluation
- Temperature is specified (typically 0.0 for deterministic eval)
- Scorer is appropriate for the task
- Evaluation datasets are appropriate (same as training, or different test set)

## Validation Checklist

Before presenting the plan:
- ✓ Evaluation matrix is consistent
- ✓ System prompt matches training (critical for inspect-ai)
- ✓ Epoch numbering is 0-indexed
- ✓ Evaluation scripts verified (or noted as prerequisites)
