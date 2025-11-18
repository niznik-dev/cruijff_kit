# Cruijff

A Python script/command-line tool to run fine-tuning and prompt optimization experiments.

## Purpose

Cruijff is a toolkit that aims to streamline LLM evaluation pipelines. A typical research question is, “How well are LLMs able to perform Task X.” Investigating these sorts of questions involves constructing datasets, obtaining models to test, formalizing an evaluation strategy, and determining a fine-tuning process. There are excellent tools to help with each part of this process, but at the end of the day might we need to run all these processes *many* times to investigate the impact of numerous potential changes to our “recipe”. Cruijff’s purpose is to accelerate research by putting these pieces together into replicable “trials” and  “experiments”.

## Installation

### Using PyPI

```shell
uv pip install cruijff
```

### Using GitHub

```shell
git clone https://.../cruijff.git
cd cruijff
uv sync
```

## Usage

### Creating Tasks

Create a task module, e.g.:

```shell
tasks
|- capitalization
   |- data.py # defines how to build datasets for this task
   |- eval.py # defines how to evaluate this task
   |- eval.yaml # defines default evaluation config fo this task
```

Tasks need to implement two modules: `data.py` and `eval.py`. The first module *must* define a single function:

```python
# data.py
def build(dataset: str, data_dir: str | os.PathLike) -> None:
    """Build a dataset."""
    # Logic to build a dataset based on its name.
    # Save the dataset to the data directory.
    # Don't return anything
    raise NotImplementedError
```

The second module, `eval.py`, should define a *single* Inspect AI task:

```python
@task
def do_a_thing(data_path: str, data_ext: str, ...) -> Task:
    """Evaluate the performance of a model doing the thing."""
    # You're free to add any additional arguments to this function
    # All arguments will be passed via eval.yaml!
    # Return an inspect.Task at the end:
    return Task(...)
```

### Define Configuration

Create experiment configuration, e.g.,:

```yaml
# config.yaml
model_dir: /home/billdcat/cruijff/models
data_dir: /home/billdcat/cruijff/data
tasks_dir: /home/billdcat/cruijff/tasks

use_fine_tune: True
use_prompt_opt: False
run_control_trials: True

# torchtune_config: /home/billdcat/cruijff/torchtune.yaml
# inspect_config: /home/billdcat/cruijff/eval.yaml

# Default resource config
resources:
  default:
    nodes: 1
    ntasks_per_node: 1
    time: "00:30:00"
    account: cses
  meta-llama__Llama-3.2-1B-Instruct:
    mem: 8
    gres: 1
  meta-llama__Llama-3.2-70B-Instruct:
    mem: 8
    gres: 4
```

### Building Experiments

Build experiments via the `cruijff build` command:

```shell
cd /home/billdcat/cruijff/experiments
cruijff build \
  --config-path  '/home/billdcat/cruijff/experiments/config.yaml' \
  --task         'capitalization' \
  --dataset      'words_5L_80_1000.json \
  --model        'meta-llama/Llama-3.2-1B-Instruct' \
  --control      'torchtune.model.lora_alpha=32' \
  --treatment    'torchtune.model.lora_rank=4' \
  --treatment    'torchtune.model.lora_rank=64'
```

Treatments and controls can also be specified with JSON-formatted strings and may include `model`:

```shell
cd /home/billdcat/cruijff/experiments
cruijff build \
  --config-path  '/home/billdcat/cruijff/experiments/config.yaml' \
  --task         'capitalization' \
  --dataset      'words_5L_80_1000.json' \
  --controls     '{"torchtune.model.lora_alpha": 32}' \
  --treatments   '{"model": ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-70B-Instruct"], "torchtune.model.lora_rank": 4}'
```

Alternatively, controls and treatments can be saved in a JSON file and you can simply point to the file paths. This example demonstrates that you can also vary `dataset` in treatments:

```shell
cd /home/billdcat/cruijff/experiments
echo '{"torchtune.model.lora_alpha": 32}' |> controls.json
echo '{"dataset": ["words_5L_80_1000.json", "words_10L_80_1000.json"], "torchtune.model.lora_rank": 4}' |> treatments.json
cruijff build \
  --config-path     '/home/billdcat/cruijff/experiments/config.yaml' \
  --task            'capitalization' \
  --model           'meta-llama/Llama-3.2-1B-Instruct' \
  --control-path    '/home/billdcat/cruijff/experiments/controls.json' \
  --treatment-path  '/home/billdcat/cruijff/experiments/treatments.json'
```

Building an experiment does the following:

1. Converts configuration, controls, and treatments into a set of Trials, each of which defines a pipeline with a specific set of values for each stage.
2. Performs all necessary setup. Each trial will validate its configuration, construct a directory to store trial-specifc files, build and/or download any missing data or models needed, and create stage-specific Slurm scripts.

After building an experiment, you should have a directory structure that looks something like this:

```shell
experiment_root
|- config.yaml
|- my-awesome-experiment
   |- config.yaml
   |- trial-a
      |- config.yaml
      |- finetune.yaml
      |- finetune.slurm
      |- eval.yaml
      |- eval.slurm
      |- checkpoints
```

### Advanced Configuration

#### Stages

Each stage requires configuration. Some configuration is provided by the user, while other parts are “filled in” by `cruijff`. The typical process for buidling the configuration for a particular stage is:

1. Determine the default configuration for the stage. The default can either be user-provided in the experiment `config.yaml` or discovered based on other experiment details, e.g., use the `torchtune` configuration `llama3_2/1B_lora` based on the selected model, recipe, and resource configuration.
2. Next, modify the default to account for environment specific details such as the location of model and data cache directories.
3. Finally, apply any related control and treatment settings specified by the `run` command.

In order to specify custom default stage configuration, simply create a config file add its path to the experiment’s `config.yaml`:

```yaml
finetune_config: /Users/colinswaney/GitHub/cruijff/torchtune_custom.yaml
eval_config: /Users/colinswaney/GitHub/cruijff/inspect_custom.yaml
```

#### Resources

While it is possible to make fairly accurate guesses as to computational resources required to run each stage of a pipeline, it is practically difficult to map these onto HPC resources due to idiosyncrasies of cluster setup. Furthermore, there are often Slurm job settings that need to be set for all jobs, e.g., `account`. Thus, `cruijff` relies on user-provided resource configuration to determine Slurm resources requests.

Resource configuration should be added to the the experiment’s `config.yaml` file, like so:

```shell
resources:
  default:
    nodes: 1
    ntasks_per_node: 1
    time: "00:30:00"
    account: cses
  meta-llama__Llama-3.2-1B-Instruct:
    mem: 8
    gres: 1
  meta-llama__Llama-3.2-70B-Instruct:
    mem: 8
    gres: 4
```

Note that the `default` values apply to *all* models.

Future work can explore the addition of tools to automatically generate model- and stage-specific resource selection based on prior experiment runs.

### Running Experiments

To run your experiment, use `cruijff run`;

```shell
cruijff run /home/billdcat/cruijff/experiments/my-awesome-experiment
```

This will attempt to run the entire experiment pipeline and store all results

### Analyzing Results

Did the pipeline run smoothly? The `cruijff summary` command will print a summary of the success or failure of each trial and stage of the experiment:

```shell
cruijff summary /home/billdcat/cruijff/experiments/my-awesome-experiment
```

How well did the model(s) perform? Use `cruijff analyze` to generate a report of the experiments results:

```shell
cruijff analyze /home/billdcat/cruijff/experiments/my-awesome-experiment
```

## Details

### What are Stages?

#### Running Stages

Stages can be run individually using the `--stage` argument, e.g.,

```shell
# First, build the experiment
cruijff run \
  --config-path  '/home/billdcat/cruijff/experiments/config.yaml' \
  --task         'capitalization' \
  --dataset      'words_5L_80_1000.json \
  --model        'meta-llama/Llama-3.2-1B-Instruct' \
  --control      'torchtune.model.lora_alpha=32' \
  --treatment    'torchtune.model.lora_rank=4' \
  --treatment    'torchtune.model.lora_rank=64'

# Now, run the fine-tuning stage *only*
cruijff run --stage finetune /home/billdcat/cruijff/experiments/my-awesome-experiment

# And now run the evaluation stage
cruijff run --stage eval /home/billdcat/cruijff/experiments/my-awesome-experiment
```

#### Stage Structure

Every stage in the the "pipeline" consists of:

1. Code that defines what that stage does, and
2. Configuration that defines how that stage should be run

Combining stages together is a matter of linking stages together. Typically, stages will be run as Slurm jobs. To facilitate "orchestration" of the jobs, we use an **awaitable** Slurm job abstraction. This allows us to do things like:

```python
finetunes = [SlurmJob(x) for x in finetune_args]
await(finetunes)

evals = [SlurmJob(x) for x in eval_args]
await(evals)
```

Or, if we can run some stages concurrently:

```python
finetunes = [SlurmJob(x) for x in finetune_args]
await(finetunes)

optimization = [SlurmJob(x) for x in optimize_args]
await(optimization)
```

Our aim is **not** to support general pipelines. Instead, we hope to facilitate:

1. Customization of stages within a pre-defined set of possible stages (e.g., fine-tune, optimize, eval) via configuration, and
2. The abilitiy for developers to relatively painlessly add new possible stages.

#### Stage Dependencies

The pipeline can easily be modified to use different fine-tuning and evaluation packages. The key requirements are that:

1. We can run stages as a Slurm script, and
2. We can define **how** a stage works via a config file

Even if a package does not provide CLI or config file support out-of-the-box, we should be able to satisfy these needs for any package.

## FAQs

### I’m lost—how do I get started?

Clone the repo and ask your favorite code agent for help. It will be able to analyze our (well-documented) code and understand how to use the CLI. There are quite a few configuration details available for some stages. Ultimately, you should refer to the documentation for each stage’s underlying tool to figure what options are available—or ask your agent to investigate this for you.

### What happens if a trial stage fails?

First, that trial will not proceed to the next stage. Second, the error will captured in Slurm log files. You can easily view the trials and stages that failed/succeeded with the `progress` command:

```shell
cruijff progress /home/billdcat/cruijff/experiments/my-awesome-experiment
```

### Can I re-run a failed trial?

Yes. Normally, `cruijff` will look at the experiment directory, determine which stages have run successfully for each trial, and skip all completed stages. If you try to re-run a completed experiment, it will indicate that the experiment has already been run and do nothing. However, you can re-run **any** trial with the `--force` option. In this case, `cruijff` will delete all prior experiment results and re-run the experiment.

You can use the `--trials` argument to indicate the specific trials to re-run if you only want to (e.g.) re-run a failed trial:

```shell
cruijff run --force \
  --trials a,b,c \
  /home/billdcat/cruijff/experiments/my-awesome-experiment
```

You can even re-run only a specific failed stage of a trial:

```shell
cruijff run --force \
  --trials a \
  --stage eval \
  /home/billdcat/cruijff/experiments/my-awesome-experiment
```
