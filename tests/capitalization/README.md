# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 0 - Consider Changing...

- input_dir_base: make sure this points to your copy of the repo
- account: can be omitted unless you know you have multiple accounts (make sure to remove the final \ after the conda_env line then!)

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!)

Place words_alpha.txt inside the input folder. Next, run `python sample_words.py --word-len 5 --num-words 10000` (you can choose your own params for this part) - this will generate a file like `finetune_words_5L_80P_10000.json` which we will use in finetuning.

### Part 2 - Finetuning

Use `generate_slurm_script.py` with the following arguments:

```bash
python generate_slurm_script.py \
  --my_wandb_project capitalization \
  --my_wandb_run_name oct1-prompt-1 \
  --input_dir_base /home/niznik/scratch/GitHub/cruijff-kit/tests/capitalization/input/ \
  --input_formatting '' \
  --dataset_filename finetune_words_5L_80P_10000.json \
  --system_prompt 'Capitalize the given word' \
  --batch_size 1 \
  --epochs 1 \
  --log_every_n_steps 1 \
  --run_val_every_n_steps 0 \
  --conda_env ttenv-nightly \
  --custom_recipe lora_finetune_single_device_val.py
```

Then, run

```
sbatch finetune_filled.slurm
```

(Currently, we need to extract all of the validation related things from the yaml file - will be addressed in [#49](https://github.com/niznik-dev/predicting-zygosity/issues/49))

### Part 3 - Upload to Weights & Biases

Run the following to upload your run:

```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```

### Part 4 - Test the model

Now navigate inside the tests/capitalization folder. Edit eval_inspect.slurm as appropriate. Then run

```
sbatch eval_inspect.slurm
```

and examine the slurm log file for the output; you can also run

```
inspect view
```

which will supply a link to a website to view the results of the evaluation with inspect-ai.

Did your finetuning and/or choice of prompt help?