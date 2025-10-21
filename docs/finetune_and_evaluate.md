This is a set of steps to follow for the process: [fine tune and evaluate]. Sometimes we want to do this once, but when we do this for research we often want to doing it many times where we vary parts of the recipe. Then we can compare the results as the parameters change individually or collectively.

For example, if we were trying to learn about chocolate chip cookies, we might make a batch of dough with low, medium, and high amounts of sugar and low, medium, and high amounts of butter. We would keep all other ingredients the same. Then we would bake the 9 sets of cookies all at the same temperature for the same amount of time in the same oven. Then we can try the cookies and see how they turned out. 

Now we want your help doing a similar, but the ingredients are different, the steps putting them together are different, and the way we evaluate them will be different.

For each experiment, you will need to know the locations of the following ingredients:
- 1 open weight LLM
- 1 training set of input-output pairs
- 1 evaluation sets of input-output pairs
- 1 file specifying all the hyper-parameters for the fine-tuning
- 1 file specifying all the hyper-parameters for the evaluation

You may also need to know the location of:
- 1 validation set of input-output pairs

Notes about the ingredients:
- generally the training, validation, and evaluation files will be random splits of one data file. But, sometimes we will explcitly train and test on different sources of data. Unless, we ask for that explicitly, please make sure the training, validation, and evaluation files come from the same source.

# Step: Figure out the structure of the experiments 

Ask the user to describe the experiment in words.  For example:
- they might want to try fine-tuning a series of models of increasing size (1B-Instruct, 3B-Instruct, 8B-Instruct, 70B-Instruct)
- they might want to try fine tuning over different lora ranks
- they might want to vary the dataset
- they might want to vary any combination of these three

The training the user will want to do should map to an existing torchtune recipe: https://meta-pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html.  Here's the list of existing receipes: https://github.com/meta-pytorch/torchtune/tree/main/recipes

If the user does not specify, you should assume the recipe is lora_finetune_single_device.py

If appropriate add a control condition where you evaluate the model without finetuning.

Describe back to the user what you understand the set of experiments to be.  If the user agrees, go to the next step.

# Step: Develop a naming convention

Now that you understand the set of experiments, set up a directory structure.

First, come up with a helpful naming convention for this set of experiments based on what the user has requested. Please suggest the name for the overall folder that will have all the experiments from this set and the naming convention for each subfolder.

Get feedback from the user about the names and naming convention.

Once the user has approved create a folder in the [place for output (defined below)] with the appropriate names.

Finally, write a file called README.md that explains the set of experiments and the naming convention you used.

# Step: write a fine-tuning config file in .yaml format

If the set of experiments all uses one torchtune recipe, the thing that will change from experiment to experiment is the config file.  

Based on the overall design of the experiment, pick the default config file that matches model in the receipe. Here's a list: https://github.com/meta-pytorch/torchtune/tree/main/recipes/configs

Make the neccesary changes to the config file to match the experiment.

Check that your .yaml file follows the recommended best practices for https://meta-pytorch.org/torchtune/main/deep_dives/configs.html#best-practices-for-writing-configs

Validate the .yaml file using "tune validate" (described more here: https://meta-pytorch.org/torchtune/main/tune_cli.html#validate-a-config)

Here are some things to check:
- double check all the lines that have a value that includes the name of the model. The mapping between the model name and the appropraite value is not always consistent. Here's an example of what can go wrong: when using some Llama models the value for checkpointer.model_type can be LLAMA3 even if the model is model is 3.X. You can check here for appropriate config values: https://github.com/meta-pytorch/torchtune/tree/main/recipes/configs. 

Once you have a .yaml config file in each subdirectory go to the next step.

# Step: Generate a slurm script for each experiment

Start with the default slurm script

# Step: Run all slurm scripts


# Step: Evaluate fine-tuned models

For evaluation, create a standalone evaluation script (e.g., `eval_cap.py`) in each experiment directory that:
- Directly specifies the dataset path and system prompt (don't rely on setup_finetune.yaml)
- Uses the inspect_ai framework to load the test dataset
- Defines the evaluation task with appropriate scorers




# Learning your way around my set-up

All base llms are located here: /scratch/gpfs/MSALGANIK/pretrained-llms. 

All base llms follow the hugging face naming conventions.

If you don't know where to put something: just ask.



# Background notes

Our computation will be done on an HPC cluster at Princeton University. That cluster is named "della". You can read more about it here: https://researchcomputing.princeton.edu/systems/della

The fine tuning is done with Torchtune. Here's the Torchtune documentation: github.com/pytorch/torchtune. Here's the Torchtune code: github.com/pytorch/torchtune.

Place for the output is: /scratch/gpfs/MSALGANIK/mjs3