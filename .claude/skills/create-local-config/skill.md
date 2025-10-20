# create-local-config

Create a personalized `claude.local.md` configuration file for cruijff_kit development.

## Description

This skill generates a `claude.local.md` file in the repository root that contains personal customizations and environment-specific settings for working with cruijff_kit. This file should be added to `.gitignore` to keep personal configurations private.

## Usage

When invoked, this skill will:
1. Check if `claude.local.md` already exists and warn if it does
2. Gather user-specific information (HPC username, common paths, preferences)
3. Generate a `claude.local.md` file with personal settings
4. Ensure `claude.local.md` is in `.gitignore`

## What to Include

The generated file should contain:
- HPC environment settings (cluster name, documentation link, username, scratch paths, model directories)
- Common command shortcuts and aliases
- Preferred SLURM parameters (account, partition, constraint for GPU VRAM requirements)
- Weights & Biases configuration (project names, entity)
- Personal notes and workflow preferences
- Common dataset locations
- Conda environment names if different from default
- Recent experiments and current work notes

## Example Structure

```markdown
# Personal Configuration for cruijff_kit

## HPC Environment
- Cluster: Della (Princeton Research Computing)
- Documentation: https://researchcomputing.princeton.edu/systems/della
- Username: `username`
- Group: `GROUPNAME`
- Scratch directory: `/scratch/gpfs/GROUPNAME/username`
- Models directory: `/scratch/gpfs/GROUPNAME/pretrained-llms`
- Working directory: `/home/username/cruijff_kit`

## SLURM Defaults
- Account: `groupname`
- Partition: `gpu`
- Constraint: `gpu80` (80GB VRAM GPUs only)
- Default conda environment: `ttenv`

## Weights & Biases
- Entity: `username`
- Default project: `my-research-project`

## Common Paths
- Data storage: `/scratch/gpfs/GROUPNAME/username/`
- Recent experiments: [list recent experiment directories]

## Quick Commands
[Include common fine-tuning, evaluation, and job management commands]

## Personal Workflow Notes
[Add your workflow notes, experiments in progress, etc.]
```

## Implementation

1. **Check if file exists**: Warn the user if `claude.local.md` already exists before overwriting
2. **Detect environment information**:
   - Get username from `$USER`
   - Detect working directory
   - Check for existing scratch directory structure
   - List recent experiments in scratch directory
3. **Ask the user for**:
   - HPC cluster name (default: Della)
   - HPC cluster documentation URL (default: https://researchcomputing.princeton.edu/systems/della)
   - Scratch directory group name (can often be detected from path)
   - SLURM account name (usually lowercase group name)
   - Preferred SLURM partition (default: `gpu`)
   - GPU constraint for VRAM requirements (e.g., `gpu80` for 80GB, `gpu40` for 40GB)
   - Weights & Biases entity/username
   - Default W&B project name
   - Conda environment name (default: `ttenv`)
   - Any other custom paths or preferences
4. **Generate the file** with:
   - All detected and user-provided information
   - Common command templates for fine-tuning, evaluation, and job management
   - Section for personal workflow notes
   - Links to recent experiments
5. **Update `.gitignore`**: Ensure `claude.local.md` is in `.gitignore` if not already present

The generated file should be comprehensive enough to serve as a quick reference for the user's specific HPC environment and workflow.
