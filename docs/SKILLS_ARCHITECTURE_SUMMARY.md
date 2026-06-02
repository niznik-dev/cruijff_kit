# Skills Architecture Summary

## Overview

cruijff_kit's workflow is built from a small set of skills, each with a focused responsibility, that together take an experiment from design through analysis. Two of the skills (`scaffold-experiment`, `run-experiment`) are heavily modularized internally — their `SKILL.md` orchestrates submodules in `optimizers/` and `evaluators/` subdirectories, plus subagents in `.claude/agents/`.

## Terminology

Three distinct concepts, three distinct names:

- **Project** — task family / research domain blueprint (`blueprints/capitalization/`, `blueprints/folktexts/`, `blueprints/model_organisms/`)
- **Experiment** — a designed set of runs under a project, defined by an `experiment_summary.yaml`
- **Inspect-ai task** — evaluation script at `blueprints/{project}/inspect_task.py` — the `@task` function inspect-ai invokes

Scratch-side, every experiment lives at `ck-projects/{project}/{experiment_name}/` with all run subdirs, configs, checkpoints, logs, and eval outputs nested inside.

## Workflow Stages

```
DESIGN     →  design-experiment
              (optional: create-tabular-schema → convert-tabular-to-text)

SCAFFOLD   →  scaffold-experiment
                ├─ scaffold-torchtune  (subagent, parallel)
                └─ scaffold-inspect    (subagent, parallel)

RUN        →  run-experiment
                ├─ optimizers/torchtune/    (fine-tuning pipeline)
                └─ evaluators/inspect/      (evaluation pipeline; sequential, after fine-tuning)

SUMMARIZE  →  summarize-experiment   (required post-run step)

ANALYZE    →  analyze-experiment      (optional; any time after run)
              (optional: analyze-to-pdf)

CLEANUP    →  archive-experiment
```

## Skill Catalog

### Workflow skills

| Skill | Purpose |
|---|---|
| `design-experiment` | Plan runs, variables, and resources; produce `experiment_summary.yaml` |
| `scaffold-experiment` | Launch `scaffold-torchtune` + `scaffold-inspect` subagents to generate setup configs and SLURM scripts for all runs |
| `run-experiment` | Submit fine-tuning jobs, wait for completion, then submit evaluation jobs; monitor SLURM and update status tables in `experiment_summary.yaml` |
| `summarize-experiment` | Generate `summary.md` with key metrics (loss, accuracy) after experiment completion |
| `analyze-experiment` | Generate interactive HTML plots and `report.md` from inspect-ai logs (uses inspect-viz) |
| `archive-experiment` | Preserve experiment files, delete checkpoint directories |

### Data preparation

| Skill | Purpose |
|---|---|
| `create-tabular-schema` | Create schema YAML for tabular source data (columns, types, value maps). Run before `design-experiment` when starting from a CSV/Stata/Parquet source |
| `convert-tabular-to-text` | Convert tabular data to text JSON for LLM fine-tuning/eval. Uses content-hashed filenames; see `src/tabular_to_text_gen/TABULAR_DATASET_NAMING.md` |

### Utilities

| Skill | Purpose |
|---|---|
| `analyze-to-pdf` | Convert a markdown report → PDF via pandoc |
| `create-inspect-task` | Guided creation of custom inspect-ai eval tasks (experiment-guided or standalone) |
| `check-release` | Weekly release check; review changes since last tag |
| `create-meeting-agenda` | Auto-generate weekly software meeting agenda in the wiki |

### Subagents (`.claude/agents/`)

| Agent | Purpose |
|---|---|
| `scaffold-torchtune` | Generate fine-tuning configs and SLURM scripts. Invoked by `scaffold-experiment`, run in parallel with `scaffold-inspect` |
| `scaffold-inspect` | Generate evaluation configs and SLURM scripts. Invoked by `scaffold-experiment` |
| `generate-jinja-template` | Generate a Jinja2 narrative template for tabular-to-text conversion. Invoked by `convert-tabular-to-text` when a custom narrative template is needed |

## Internal Module Structure

Two skills use modular documentation to keep the entry-point `SKILL.md` short and load tool-specific logic on demand.

### scaffold-experiment

```
scaffold-experiment/
├── SKILL.md                       # Orchestration + subagent invocation
├── logging.md
├── parsing.md
├── optimizers/torchtune_agent.md  # Subagent prompt + handoff spec
└── evaluators/inspect_agent.md    # Subagent prompt + handoff spec
```

### run-experiment

```
run-experiment/
├── SKILL.md                       # High-level orchestration
├── logging.md
├── optimizers/torchtune/          # Fine-tuning pipeline
│   ├── main.md
│   ├── run_selection.md
│   ├── job_submission.md
│   ├── monitoring.md
│   ├── parsing.md
│   └── validation.md
├── evaluators/inspect/            # Evaluation pipeline (post fine-tuning)
│   ├── main.md
│   ├── evaluation_selection.md
│   ├── dependency_checking.md
│   ├── job_submission.md
│   ├── monitoring.md
│   ├── parsing.md
│   ├── validation.md
│   └── cache_prebuilding.md
└── shared/compute_monitoring.md
```

## File Structure After Scaffolding

```
ck-projects/{project}/{experiment_name}/
├── experiment_summary.yaml        # From design-experiment
├── summary.md                     # From summarize-experiment
├── logs/
│   ├── design-experiment.log
│   ├── scaffold-experiment.log
│   ├── run-experiment.log
│   ├── analyze-experiment.log
│   └── summarize-experiment.log
├── {run_name}/                    # One per run (e.g., rank8_lr1e-5)
│   ├── setup_finetune.yaml        # From scaffold-torchtune
│   ├── finetune.yaml
│   ├── finetune.slurm
│   ├── slurm-{jobid}.out          # From run-experiment (torchtune phase)
│   └── eval/
│       └── {task}_epoch{N}/         # One cell per (task, epoch) — issue #498
│           ├── eval_config.yaml     # From scaffold-inspect
│           ├── cell.slurm           # From scaffold-inspect
│           └── logs/
│               └── *.eval           # inspect-ai outputs
└── analysis/                      # From analyze-experiment
    ├── report.md
    └── *.html
```

## Key Design Principles

**1. Single responsibility.** Each skill does one thing. `scaffold-experiment` only generates configs; `run-experiment` only submits and monitors jobs; `analyze-experiment` only renders results.

**2. Sequential dependencies.** `run-experiment` enforces fine-tuning before evaluation — eval jobs require completed checkpoints. The order is encoded in the workflow modules, not in user discipline.

**3. Modular documentation.** For complex skills, `SKILL.md` is the entry point and submodules in `optimizers/` and `evaluators/` are loaded on demand. Smaller files reduce context cost and hallucination risk.

**4. Subagents for tool-specific scaffolding.** `scaffold-experiment` delegates torchtune and inspect-ai config generation to dedicated subagents in `.claude/agents/`, run in parallel since their outputs are independent.

**5. Comprehensive logging.** Each skill writes its own log under `logs/`; SLURM logs and inspect-ai logs sit alongside the run/eval directories. `experiment_summary.yaml` is updated in place with status tables for each phase.

**6. Standalone usability.** Skills can be run individually — e.g., re-run just `scaffold-experiment` after editing `experiment_summary.yaml`, or run `run-experiment` against an already-scaffolded experiment. State flows through `experiment_summary.yaml` and the on-disk directory structure, not in-memory.
