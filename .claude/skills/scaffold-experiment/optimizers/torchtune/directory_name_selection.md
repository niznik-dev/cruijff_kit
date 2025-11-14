# Directory Naming Algorithm

This module defines how to generate run directory names based on varying parameters.

## Goal

Directory names should only include parameters that vary across runs.

## Algorithm

1. For each parameter in the runs table (LoRA Rank, Learning Rate, Batch Size, etc.), check if it has different values across runs
2. Include only varying parameters in directory names
3. Use a consistent naming pattern: `{param1}{value1}_{param2}{value2}`

## Parameter Name Abbreviations

- LoRA Rank → `r`
- Learning Rate → `lr`
- Batch Size → `b`
- Model → use short model name (e.g., `Llama-3.2-1B`)

## Examples

**Experiment varying LoRA rank and learning rate:**
- `r8_lr1e-5/`
- `r16_lr5e-5/`
- `r32_lr1e-5/`

**Experiment varying only LoRA rank:**
- `r8/`
- `r16/`
- `r32/`

**Experiment varying model and LoRA rank:**
- `Llama-3.2-1B_r8/`
- `Llama-3.2-3B_r16/`

**Experiment varying batch size and learning rate:**
- `b4_lr1e-5/`
- `b8_lr5e-5/`

## Implementation Notes

- Directory names should be concise and descriptive
- Underscores (`_`) separate different parameters
- No spaces in directory names
- Use the same parameter order consistently across all runs in an experiment
- Value formatting should match how it appears in the runs table (e.g., keep scientific notation `1e-5`)
