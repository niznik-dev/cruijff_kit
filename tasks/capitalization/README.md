# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!). You'll probably want to put it in `tasks/capitalization/input/`.  You can download this file from the command line as follows:

```bash
wget https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
```

Place words_alpha.txt inside the input folder. Next, run `python sample_words.py --word-len 5 --num-words 1000` (you can choose your own params for this part) - this will generate a file like `words_5L_80P_1000.json` in the instruct_dataset format with top level splits (train/validation/test) which we will use in finetuning.

(If you'd rather generate a chat_dataset, run `python sample_words.py --word-len 5 --num-words 1000 --use-chat-template` instead - this will create a folder with an "_c" appended and separate files inside per split (train/validation/test.json))

### Part 2 - Finetuning

You can run finetuning using either JSON format or parquet format. Choose one of the options below:

#### Option 1: Using JSON Format (Current Method)

First, copy `setup_finetune_json.yaml` from the capitalization task templates/finetuning folder one level up to the capitalization task folder - we'll be running things from here going forward. Next, rename it to `setup_finetune.yaml`. Then open the file and consider the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Match your data format
  - For this example, make sure dataset_label (the filename without ".json") and dataset_ext (".json") match what you generated in Part 1 
  - If you'd rather use the chat template, revisit sample_words.py to create that folder and then once again make sure dataset_label matches the directory (usually ending in "_c"); dataset_ext is still ".json"
- Change system_prompt if necessary

Then run

```
python ../../tools/torchtune/setup_finetune.py
```

Finally, run

```
sbatch finetune.slurm
```

#### Option 2: Using Parquet Format

First, convert the JSON file to Parquet format:

```
python ../../utils/convert_json_to_parquet.py \
  --input_json input/words_5L_80P_1000.json \
  --output_dir input/words_5L_80P_1000_parquet
```

This will create Parquet files (train.parquet, validation.parquet, test.parquet) in the output directory.

Next, copy `setup_finetune_parquet.yaml` from the capitalization task templates/finetuning folder one level up to the capitalization task folder - we'll be running things from here going forward. Next, rename it to `setup_finetune.yaml`. Then open the file and consider the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Make sure dataset_filename matches the dataset folder you just created
  - The dataset_label should be the parquet folder and dataset_ext should be ".parquet"
- Change system_prompt if necessary

Then run

```
python ../../tools/torchtune/setup_finetune.py
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

Generate inspect.slurm by running the following command:

```
python ../../tools/inspect/setup_inspect.py --finetune_epoch_dir /path/to/finetuned/model/epoch_0/
```

Then as the output says you can run:

```
sbatch inspect.slurm
```

and examine the slurm log file for the output; you can also run

```
inspect view
```

(on della, we recommend adding `--port=$(get_free_port)` after view)

which will supply a link to a website to view the results of the evaluation with inspect-ai.

Did your finetuning and/or choice of prompt help?

### Part 5 - Test the base model

If you've already generated one finetuned version of the base model, you can evaluate on the base model by running `setup_inspect.py` in a slightly different way:

```
Generate inspect.slurm by running the following command:

```
python ../../tools/inspect/setup_inspect.py --base_model_dir /path/to/base/model/ --finetune_epoch_dir /path/to/finetuned/model/epoch_0/
```

In this case, the base model will get used but `inspect.slurm` will reference the finetuned model's slurm scripts to get the proper parameters for GPUs, memory, etc.