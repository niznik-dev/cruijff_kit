# Plan Experiment

You are helping the user plan a fine-tuning experiment workflow for LLMs.

## Your Task

Guide the user through defining the structure of their experiments by asking questions and clarifying their goals.

## Questions to Ask

1. **What do you want to vary in this experiment?**
   - Different model sizes? (e.g., 1B-Instruct, 3B-Instruct, 8B-Instruct, 70B-Instruct)
   - Different LoRA ranks?
   - Different datasets?
   - Different hyperparameters?
   - Any combination of the above?

2. **Which torchtune recipe should be used?**
   - Default: `lora_finetune_single_device.py`
   - See all recipes: https://github.com/meta-pytorch/torchtune/tree/main/recipes
   - Reference: https://meta-pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html

3. **Should we include a control condition?**
   - Evaluate the base model without fine-tuning for comparison?

## Required Ingredients

For each experiment, identify the locations of:
- 1 open weight LLM (from `/scratch/gpfs/MSALGANIK/pretrained-llms`)
- 1 training set of input-output pairs
- 1 evaluation set of input-output pairs
- 1 file specifying hyperparameters for fine-tuning
- 1 file specifying hyperparameters for evaluation
- (Optional) 1 validation set of input-output pairs

## Important Notes

- Generally, training, validation, and evaluation files should be random splits of one data file
- Only use different data sources if the user explicitly requests it
- All base LLMs follow HuggingFace naming conventions
- Output location: `/scratch/gpfs/MSALGANIK/mjs3`

## Next Steps

Once you understand the experiment structure:
1. Summarize back to the user what you understand the set of experiments to be
2. Get user confirmation before proceeding
3. Suggest using the `setup-experiment-dirs` skill to create the directory structure
