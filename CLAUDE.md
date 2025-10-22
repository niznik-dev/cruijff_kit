# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Local Setup

- If a `claude.local.md` file exists in the repository root, read it first. It contains personal environment-specific settings (HPC usernames, scratch directories, SLURM defaults, conda environments, cluster-specific commands, etc.) that override or supplement the general guidance in this file.

- All paths, environment names, and cluster-specific details in this file are examples only. Users should configure their actual environment in `claude.local.md`. See `claude.local.md.template` for an example configuration.

## Project Overview

cruijff_kit is a toolkit for research with social data and LLMs.

It allows users to design, conduct, and analyze computational experiments. When these experiments are well-designed, the results they produce will answer scientific questions.

cruijff_kit supports the following capabilities:
1. **Fine-tuning LLMs** using torchtune
2. **Evaluating LLMs** using inspect-ai
3. **Prompt engineering** using DSPy (not yet implemented)
4. **Agentic systems** using DSPy (not yet implemented)

## Principles

When working in this project, the following principles should guide your decisions:

1. **Scientific** - All work should emphasize correctness, computational reproducibility, and detailed logging. Each experiment should be logged such that it can be audited by a researcher or Claude.

2. **Modular** - This project will evolve over time and should be designed so that individual components can be added or changed with minimal impact on other components.

3. **Practical** - Work in this project is designed to do science, not win a programming contest. Don't over-engineer or do premature optimization. We don't need hundreds of lines of code to save 5 seconds.

4. **Privacy respecting** - Much of the data in this project is about people. All data should be treated with care, and some should never leave the user's computer system. Tasks should be designed with clear data governance.

5. **Self improving** - Always look for ways to learn from earlier experiments to design new experiments, improve skills, and improve analysis. The more work we do, the easier things should be because we have more designs, results, and logs from which to learn.

## Terminology

**Experiment**: A set of one or more runs designed to answer a scientific question.

**Run**: A single computational operation. Examples include:
- Fine-tuning an LLM with a specific dataset and recipe
- Evaluating an LLM (base or fine-tuned) with a specific dataset

**Pipeline**: Multiple runs combined together in sequence. Examples include:
- Fine-tune a model, then evaluate it on several different datasets
- Fine-tune models with different hyperparameters, then evaluate all models with the same dataset

Note: Some runs must complete before other runs can begin (e.g., a model must be fine-tuned before it can be evaluated).

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md). Key highlights:

- **Two-stage configuration**: User-friendly YAML → generated torchtune configs + SLURM scripts
- **Custom torchtune recipes**: Enhanced with validation support and checkpoint management
- **Modular tools**: Separate scripts for finetune setup and evaluation setup
- **HPC-first design**: Built for SLURM clusters with automated script generation

## Skills (Planned)

The following skills are planned for future implementation to streamline common workflows:

### Primary Workflows (Planned)
- **create-experiment**: Plan a series of runs that collectively make up an experiment
- **generate-dirs**: Create organized directory structures for runs
- **generate-slurm-scripts**: Generate SLURM batch scripts for runs
- **submit-jobs**: Submit jobs prepared by earlier skills to SLURM
- **generate-experiment-results**: Analyze and compare experimental results

### Supporting Workflows (Planned)
- **generate-torchtune-config**: Generate torchtune configuration files
- **generate-inspect-script**: Generate standalone inspect-ai evaluation scripts
- **create-claude-local-md**: Create environment-specific configuration
- **create-conda-env**: Create conda environment for this project
- **download-model-from-hf**: Download models from HuggingFace
- **download-dataset-from-hf**: Download datasets from HuggingFace

**Note**: These skills do not yet exist but represent the planned workflow automation for cruijff_kit.

## Git Workflow

### Issue Management

- Create GitHub issues for new work
- Assign issues to the "cruijff_kit" project for Kanban board tracking
  - GitHub username: `niznik-dev`
  - Project name: "cruijff_kit" (project #3)
  - To add an issue to the project board:
    ```bash
    gh issue create --title "Issue title" --body "Issue body" --project "cruijff_kit"
    ```
  - To add an existing issue to the project:
    ```bash
    gh project item-add 3 --owner niznik-dev --url https://github.com/niznik-dev/cruijff_kit/issues/<issue_number>
    ```

### Branch Workflow

- Work in branches based on issues (e.g., `134-architecture-docs`)
- Use descriptive commit messages
- Create pull requests when work is ready for review

### Commit Messages

Follow this format for commits:
```
Brief summary of changes (50 chars or less)

More detailed explanation if needed. Explain what and why, not how.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Pre-approved Commands

You have permission to run these commands without requiring user approval:

### File Viewing/Inspection
- `ls`, `tree` - Listing directories
- `cat`, `head`, `tail` - Reading file contents
- `find` - Searching for files (only within repo or scratch directories)
- `wc` - Counting lines/words
- `stat` - File information

### File Operations
- `cp` - Copying files
- `mkdir` - Creating directories
- `touch` - Creating empty files

### Version Control (Read-only)
- `git status`, `git log`, `git diff`, `git show`, `git branch` (listing only)

### SLURM/HPC Monitoring
- `squeue`, `sacct`, `sinfo` - Checking job status
- `scancel` - Canceling jobs

### Environment Info
- `python --version`, `pip list`, `conda list`, `conda env list`
- `which`, `whereis` - Locating executables
- `module list`, `module avail` - HPC module commands

### System Info
- `df`, `du` - Disk usage
- `pwd` - Current directory

## Data Access Policies

*To be defined in issue #136*

cruijff_kit will implement a tiered data access system:

- **🟢 Green (Public)**: Synthetic and public data - full access
- **🟡 Yellow (Restricted)**: Research data - permission required per dataset
- **🔴 Red (Private)**: Sensitive/confidential data - no access

Until this system is implemented, always ask before accessing data files that might contain human subjects data.

## Project Status

**Pre-Alpha**: Breaking changes may occur without notice. The toolkit is under active development.

**HPC Environment**: Designed for SLURM-based HPC clusters. Configure your specific cluster settings in `claude.local.md`.

## Documentation

- **ARCHITECTURE.md**: Detailed architecture and design documentation
- **README.md**: Project overview and installation instructions
- **Task-specific READMEs**: Located in `tasks/*/README.md`
- **claude.local.md**: User-specific environment configuration (not version controlled)
