# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!). You'll probably want to put it in `experiments/capitalization/input/`.  You can download this file from the command line as follows:

```bash
cd experiments/capitalization/input
wget https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
cd ../../..
```

Next, run `python experiments/capitalization/preprocess_input_data.py --word-len 5 --num-words 1000` (you can choose your own params for this part) - this will generate a file like `words_5L_80P_1000.json` in `data/green/capitalization/` with top level splits (train/validation/test) which we will use in finetuning.

### Part 2 - Finetuning

#### Dataset Types

By default, fine-tuning uses `chat_completion` which applies HuggingFace chat templates to ensure train/eval parity with inspect-ai evaluations.

#### Setup

Copy the appropriate config from `templates/finetuning/` and edit it:

```
cp templates/finetuning/setup_finetune_json.yaml setup_finetune.yaml
```

Then edit `setup_finetune.yaml` with the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Match your data format
  - For this example, make sure dataset_label (the filename without ".json") and dataset_ext (".json") match what you generated in Part 1
- Edit the `prompt` template if needed (e.g., `"Capitalize the given word: {input}\n"`)
  - Note: Use a colon separator rather than newline between instruction and input for best results

Then run

```
python ../../tools/torchtune/setup_finetune.py
```

Finally, run

```
sbatch finetune.slurm
```

#### Alternative: Using Parquet Format

**Note:** Parquet format requires `dataset_type: instruct_dataset` (legacy mode) because `chat_completion` only supports JSON files.

If you prefer Parquet format instead of JSON, first convert the JSON file:

```
python utils/convert_json_to_parquet.py \
  --input_json data/green/capitalization/words_5L_80P_1000.json \
  --output_dir data/green/capitalization/words_5L_80P_1000_parquet
```

This will create Parquet files (train.parquet, validation.parquet, test.parquet) in the output directory.

Then overwrite the default config with the Parquet example:

```
cp templates/finetuning/setup_finetune_parquet.yaml setup_finetune.yaml
```

Edit `setup_finetune.yaml` with the following changes:

- Change the run name to something unique (datestamp? number your system prompts?)
- Change input_dir_base to match where you cloned the repo
- Make sure dataset_filename matches the dataset folder you just created
  - The dataset_label should be the parquet folder and dataset_ext should be ".parquet"
- Change system_prompt if necessary

Then run `python ../../tools/torchtune/setup_finetune.py` and `sbatch finetune.slurm` as above.

### (Optional) Part 3 - Upload to Weights & Biases

(If you do not already have an account with Weights & Biases, go to [their website](https://wandb.ai) and create one. In the splash page that follows, an API key will be presented to you which you can copy and supply when prompted later)

Run the following to upload your run:

```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```

### Part 4 - Test the model

Create an `eval_config.yaml` in your eval directory with the experiment-specific config (see `inspect_agent.md` for the full schema), then render the SLURM script:

```bash
cd /path/to/experiment/run/eval
python ../../tools/inspect/setup_inspect.py \
  --config eval_config.yaml \
  --model_name Llama-3.2-1B-Instruct
```

Then submit the job:

```
sbatch capitalization_epoch0.slurm
```

and examine the slurm log file for the output; you can also run

```
inspect view
```

(on della, we recommend adding `--port=$(get_free_port)` after view)

which will supply a link to a website to view the results of the evaluation with inspect-ai.

Did your finetuning and/or choice of prompt help?

### Part 5 - Test the base model

To evaluate the base model, create an `eval_config.yaml` pointing to the base model path (with `finetuned: false` in metadata) and run [setup_inspect.py](../../tools/inspect/setup_inspect.py) the same way:

```bash
cd /path/to/experiment/base_run/eval
python ../../tools/inspect/setup_inspect.py \
  --config eval_config.yaml \
  --model_name Llama-3.2-1B-Instruct
```