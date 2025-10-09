# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!)

Place words_alpha.txt inside the input folder. Next, run `python sample_words.py --word-len 5 --num-words 1000` (you can choose your own params for this part) - this will generate a file like `words_5L_80P_1000.json` which we will use in finetuning.

### Part 2 - Finetuning

First, copy `total_config.yaml` from the capitalization test folder to the base directory of the repo. Next, open the new copy and consider the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Make sure dataset_filename matches what you generated in Part 1
- Change system_prompt if necessary

Then run

```
python generate_slurm_script.py --config total_config.yaml
```

Finally, run

```
sbatch finetune_filled.slurm
```

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