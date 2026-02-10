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

## Skills

cruijff_kit includes Claude Code skills to streamline common workflows. These skills are optional - all workflows can also be performed manually.

### Primary Workflows
- **design-experiment** ✅ - Plan a series of runs that collectively make up an experiment
- **scaffold-experiment** ✅ - Create organized directory structures, configs, and SLURM scripts for all runs
- **run-experiment** ✅ - Submit jobs to SLURM and monitor their progress until completion
- **summarize-experiment** ✅ - Generate summary.md with key metrics (loss, accuracy) after experiment completion
- **create-inspect-task** ✅ - Create custom inspect-ai evaluation tasks with guided workflow (supports experiment-guided and standalone modes)
- **analyze-experiment** ✅ - Generate interactive HTML visualizations from evaluation logs using inspect-viz (currently supports capitalization experiments)
- **analyze-to-pdf** ✅ - Convert analysis reports (markdown) to PDF using pandoc

**Note**: All skills are optional convenience tools. Users can perform the same operations manually by running the underlying Python scripts and shell commands directly.

**Architecture**: The primary workflow skills (scaffold-experiment, run-experiment) use modular documentation with `optimizers/` and `evaluators/` subdirectories for tool-specific logic. See [SKILLS_ARCHITECTURE_SUMMARY.md](SKILLS_ARCHITECTURE_SUMMARY.md) for the complete skills architecture.

## Workflow Testing

To validate that the complete workflow (design, scaffold, run) is functioning correctly after changes to skills or documentation, use the integration test specifications.

**Invocation:** When user says "test the workflow" or similar:
1. **Always use AskUserQuestion** to prompt which test variant to run
2. Read the appropriate specification file
3. Execute: `design-experiment` → `scaffold-experiment` → `run-experiment`
4. Verify expected outputs match validation checks

### Test Variants

| Option | Spec File | Runs | Duration | Tests |
|--------|-----------|------|----------|-------|
| **LoRA Comparison** | `.claude/workflow_test.yaml` | 2 fine-tuned (rank4, rank8) | ~12 min | Parameter variations |
| **Base vs Fine-tuned** | `.claude/workflow_test_base.yaml` | 1 base + 1 fine-tuned (rank4) | ~12 min | Base model evaluation |
| **Recipe Defaults** | `.claude/workflow_test_recipe.yaml` | 2 fine-tuned (rank4, rank16) | ~12 min | base_recipe inheritance |

All use Llama-3.2-1B-Instruct with words_5L_80P_1000.json in `ck-sanity-checks/`

**Purpose:** Catch regressions in skills, ensure documentation changes don't break workflows, validate end-to-end integration.

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

**Important**: Only add issues to the project board, not pull requests. PRs are automatically linked to the project via "Closes #N" in the PR description.

### Branch Workflow

- Work in branches based on issues (e.g., `134-architecture-docs`)
- Use descriptive commit messages
- Create pull requests when work is ready for review

### Pull Request Format

Follow the PR template at `.github/pull_request_template.md`:
- **Always start with** `Closes #<issue_number>` on the first line
- Include Description, New Dependencies, and Testing Instructions sections
- The "Closes #N" syntax automatically links and closes the issue when the PR is merged

**Before creating a PR**, consult `.claude/PR_CHECKLIST.md` for a helpful reminder list of common things to verify (documentation updates, testing, git hygiene, etc.). This checklist is not mandatory but helps catch oversights.

### Commit Messages

Follow this format for commits:
```
Brief summary of changes (50 chars or less)

More detailed explanation if needed. Explain what and why, not how.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Data Access Policies

cruijff_kit implements a three-tier data access system for Claude Code, using a traffic light metaphor:

**⚠️ EXPERIMENTAL FEATURE**: This data access system has not been extensively tested. Researchers should:
- **Avoid using the red tier** until this feature has been validated
- **Use the yellow tier with caution** and verify Claude Code's behavior
- Report any unexpected access patterns or behavior

### 🔴 Red Tier (`data/red/`) - No Access

**Claude Code MUST NOT access any files in this directory.**

This tier contains highly sensitive data:
- Raw data with personally identifiable information (PII)
- Data under strict IRB protocols or ethical restrictions
- Proprietary datasets with legal restrictions
- Confidential information

**Behavior**: If asked to access files in `data/red/`, refuse the request and explain that these files are marked as highly sensitive.

### 🟡 Yellow Tier (`data/yellow/`) - Permission Required

**Claude Code MAY access files ONLY with explicit user permission.**

This tier contains research data with moderate privacy considerations:
- De-identified social science data
- Datasets with usage agreements or restrictions
- Research collaborator datasets
- Data requiring case-by-case access decisions

**Behavior**: Before accessing any file in `data/yellow/`, ask the user for explicit permission, explain what you need to do with the data, and wait for confirmation.

**Per-dataset permissions**: Datasets may include a `README.md` or `PERMISSIONS.md` file documenting standing permissions.

### 🟢 Green Tier (`data/green/`) - Full Access

**Claude Code has FULL ACCESS to all files in this directory.**

This tier contains data safe for AI assistance:
- Synthetic data generated for testing
- Publicly available datasets
- Test fixtures and example data
- Educational datasets
- Generated word lists and bit sequences

**Behavior**: You may freely read, analyze, process, and work with files in `data/green/` without asking for permission.

### Implementation

Each tier directory contains a `CLAUDE.md` file with detailed access rules. See:
- `data/red/CLAUDE.md`
- `data/yellow/CLAUDE.md`
- `data/green/CLAUDE.md`

## Project Status

**Alpha**: Core workflows are functional, but breaking changes may still occur. The toolkit is under active development.

**HPC Environment**: Designed for SLURM-based HPC clusters. Configure your specific cluster settings in `claude.local.md`.

## Documentation

- **ARCHITECTURE.md**: Detailed architecture and design documentation
- **README.md**: Project overview and installation instructions
- **Experiment-specific READMEs**: Located in `experiments/*/README.md`
- **claude.local.md**: User-specific environment configuration (not version controlled)
