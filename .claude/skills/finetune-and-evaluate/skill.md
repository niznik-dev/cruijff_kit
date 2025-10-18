# Fine-tune and Evaluate

You are helping the user run a complete fine-tuning and evaluation workflow for LLM experiments.

## Overview

This is an end-to-end workflow for systematically fine-tuning LLMs and evaluating them. It's designed for research where you want to vary parameters (model size, LoRA rank, datasets, hyperparameters) and compare results.

Think of it like a baking experiment: you vary the ingredients (sugar, butter) while keeping everything else constant, then evaluate the results to see what works best.

## Complete Workflow

This skill orchestrates all the individual skills in sequence:

### 1. Plan Experiment (`plan-experiment`)
- Understand what the user wants to vary
- Identify all required ingredients (model, datasets, configs)
- Determine which torchtune recipe to use
- Decide if control conditions are needed

### 2. Setup Directory Structure (`setup-experiment-dirs`)
- Create a naming convention
- Set up organized directories
- Write a README explaining the experiment

### 3. Create Torchtune Configs (`create-torchtune-config`)
- Generate .yaml config files for each experiment
- Validate all configs
- Ensure model names and paths are correct

### 4. Generate SLURM Scripts (`create-slurm-script`)
- Create job scripts with appropriate resources
- Set time limits, GPU requirements, memory
- Configure for the della cluster

### 5. Run Experiments (`run-experiments`)
- Submit all SLURM jobs
- Monitor progress
- Track job IDs

### 6. Create Evaluation Scripts (`create-evaluation`)
- Write standalone evaluation scripts
- Use inspect_ai framework
- Define appropriate scorers

### 7. Evaluate Results
- Run evaluations on fine-tuned models
- Compare results across experiments
- Analyze which variations performed best

## When to Use This Skill

Use this complete workflow when:
- Starting a new fine-tuning research project
- Running systematic experiments with parameter variations
- You need end-to-end guidance through the entire process

## When to Use Individual Skills

Use individual skills when:
- You've already completed some steps
- You only need help with a specific part
- You're iterating on one component

## Important Context

- **Cluster:** della at Princeton University
- **Base LLMs:** `/scratch/gpfs/MSALGANIK/pretrained-llms`
- **Output location:** `/scratch/gpfs/MSALGANIK/mjs3`
- **Fine-tuning tool:** Torchtune (github.com/pytorch/torchtune)
- **Evaluation tool:** inspect_ai

## Required Ingredients

For each experiment, you'll need:
1. Open weight LLM
2. Training dataset (input-output pairs)
3. Evaluation dataset (input-output pairs)
4. Fine-tuning hyperparameters config
5. Evaluation hyperparameters config
6. (Optional) Validation dataset

**Note:** Training, validation, and evaluation should typically be random splits of the same dataset unless explicitly requested otherwise.

## Getting Started

When this skill is invoked, start by using the `plan-experiment` skill to understand what the user wants to do, then proceed through each step systematically.

Ask clarifying questions at each stage and get user approval before moving to the next step.
