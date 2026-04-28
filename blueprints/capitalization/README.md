# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 1 - Setup

(If not done already, clone this repo onto the machine you're using!)

First, obtain words_alpha.txt from the following repo: https://github.com/dwyl/english-words (and star it!). You'll probably want to put it in `blueprints/capitalization/input/`.  You can download this file from the command line as follows:

```bash
cd blueprints/capitalization/input
wget https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
cd ../../..
```

Next, run `python blueprints/capitalization/generate_data.py --word-len 5 --num-words 1000 --output_dir {ck_data_dir}/capitalization/` (you can choose your own params for this part) - this will generate a file like `words_5L_80P_1000.json` in `{ck_data_dir}/capitalization/` with top level splits (train/validation/test) which we will use in finetuning.

### Part 2 - Finetuning

#### Dataset Types

By default, fine-tuning uses `chat_completion` which applies HuggingFace chat templates to ensure train/eval parity with inspect-ai evaluations.

#### Setup

The recommended path is the `design-experiment` skill, which generates `setup_finetune.yaml` and `finetune.slurm` for you. See [ACS_EXAMPLE.md](../../ACS_EXAMPLE.md) for an end-to-end walkthrough of the design → scaffold → run workflow on a similar task.

If you'd rather write the config by hand, create `setup_finetune.yaml` in your run directory with at least:

```yaml
my_wandb_project: capitalization
my_wandb_run_name: <unique-run-name>
input_dir_base: {ck_data_dir}/capitalization/
dataset_label: words_5L_80P_1000
dataset_ext: '.json'
torchtune_model_name: Llama-3.2-1B-Instruct
prompt: "Capitalize the given word: {input}\n"
batch_size: 1
epochs: 1
custom_recipe: cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_stable
```

Use a colon separator (not a newline) between instruction and input for best results. Then run:

```
python ../../src/tools/torchtune/setup_finetune.py
sbatch finetune.slurm
```

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
python ../../src/tools/inspect/setup_inspect.py \
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

To evaluate the base model, create an `eval_config.yaml` pointing to the base model path (with `finetuned: false` in metadata) and run [setup_inspect.py](../../src/tools/inspect/setup_inspect.py) the same way:

```bash
cd /path/to/experiment/base_run/eval
python ../../src/tools/inspect/setup_inspect.py \
  --config eval_config.yaml \
  --model_name Llama-3.2-1B-Instruct
```