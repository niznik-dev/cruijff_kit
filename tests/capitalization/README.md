# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!). You'll probably want to put it in `tests/capitalization/input/`.  You can download this file from the command line as follows:

```bash
wget https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
```

Place words_alpha.txt inside the input folder. Next, run `python sample_words.py --word-len 5 --num-words 1000` (you can choose your own params for this part) - this will generate a file like `words_5L_80P_1000.json` which we will use in finetuning.

### Part 2 - Finetuning

You can run finetuning using either JSON format or parquet format. Choose one of the options below:

#### Option 1: Using JSON Format (Current Method)

First, copy `setup_finetune_json.yaml` from the capitalization test templates/finetuning folder to the base directory of the repo and rename it to `setup_finetune.yaml`. Next, open the file and consider the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Make sure dataset_filename matches what you generated in Part 1
- Change system_prompt if necessary

Then run

```
python setup_finetune.py
```

Finally, run

```
sbatch finetune.slurm
```

#### Option 2: Using Parquet Format

First, convert the JSON file to Parquet format. From the base directory of the repo, run:

```
python convert_json_to_parquet.py \
  --input_json tests/capitalization/input/words_5L_80P_1000.json \
  --output_dir tests/capitalization/input/words_5L_80P_1000_parquet
```

This will create Parquet files (train.parquet, validation.parquet, test.parquet) in the output directory.

Next, copy `setup_finetune_parquet.yaml` from the capitalization test templates/finetuning folder to the base directory of the repo and rename it to `setup_finetune.yaml`. Open the file and consider the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Make sure dataset_filename matches the dataset folder you just created
- Change system_prompt if necessary

Then run

```
python setup_finetune.py
```

Finally, run

```
sbatch finetune.slurm
```

### (Optional) Part 3 - Upload to Weights & Biases

(If you do not already have an account with Weights & Biases, go to [their website](https://wandb.ai) and create one. In the splash page that follows, an API key will be presented to you which you can copy and supply when prompted later)

Run the following to upload your run:

```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```

### Part 4 - Test the model

Now navigate inside the tests/capitalization folder. Edit inspect.slurm as appropriate (typically just adding your email). Then run

```
sbatch inspect.slurm
```

and examine the slurm log file for the output; you can also run

```
inspect view
```

which will supply a link to a website to view the results of the evaluation with inspect-ai.

Did your finetuning and/or choice of prompt help?