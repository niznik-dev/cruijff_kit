# cruijff_kit

<p align="center">
  <img src="assets/cruijff_kit_logo.png" alt="cruijff_kit logo" width="250">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="#-alpha-release-"><img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: Alpha"></a>
</p>

cruijff_kit is a toolkit for doing research with social data and LLMs. It is designed for software agents guided by humans, but supports fully manual operation. The toolkit emphasizes the values of science: correctness, provenance of results, and continual learning and improvement.

cruijff_kit is named after Dutch footballer and philosopher [Johan Cruijff](https://en.wikipedia.org/wiki/Johan_Cruyff). Many of these ideas were developed while we were doing research in Amsterdam, the city of his birth.

We are grateful to the following funders and supporters: [Princeton AI Lab](https://ai.princeton.edu/ai-lab), [Princeton Precision Health](https://pph.princeton.edu/), [Princeton Research Computing](https://researchcomputing.princeton.edu/), and the [Center for Information Technology Policy](https://citp.princeton.edu/) at Princeton University.

## What cruijff_kit Does

cruijff_kit lets you **design**, **scaffold**, **run**, and **analyze** LLM experiments on HPC clusters.

- **Design** - Plan runs with specific models, datasets, and hyperparameters
- **Scaffold** - Auto-generate torchtune configs, inspect-ai tasks, and SLURM scripts
- **Run** - Submit fine-tuning and evaluation jobs with dependency management
- **Analyze** - Collect metrics, generate visualizations, compare across runs

## ⚠ Alpha Release ⚠

This project is under active development. The core workflows are functional, but you may encounter bugs or breaking changes between updates. We'd love to collaborate - your feedback and bug reports are valuable!

See [CHANGELOG.md](CHANGELOG.md) for release history and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for current limitations.

## Quick Start

**Prerequisites**: See [PREREQUISITES.md](PREREQUISITES.md) for required skills and accounts.

```bash
# Clone the repository
git clone https://github.com/niznik-dev/cruijff_kit.git
cd cruijff_kit

# Create and activate environment
conda create -n cruijff python=3.13 -y && conda activate cruijff

# Install cruijff_kit with all dependencies
make install
```

For contributors (adds pytest, pytest-cov, and GitHub CLI):
```bash
make install-dev
```

Verify installation:
```bash
python -c "import cruijff_kit; print('cruijff_kit installed successfully')"
```

**Installation time**: Approximately 5-10 minutes depending on network speed.

## Configuration: claude.local.md

cruijff_kit uses a `claude.local.md` file to store environment-specific settings. This file is git-ignored and stays on your machine.

```bash
cp claude.local.md.template claude.local.md
```

Key settings to configure:
- **Cluster details** - hostname, username, scratch directory
- **SLURM defaults** - account, partition, constraint, time limits
- **Conda environment** - name and module load commands
- **Model paths** - shared model directory location

Without this file, you'll need to specify these values manually in each experiment configuration. See the template for a complete example.

## Downloading a Model

You'll need a model to fine-tune and evaluate. Here's how to get one via torchtune:

**Step 1**: Request access on HuggingFace (if required). For Meta models, navigate to `https://huggingface.co/meta-llama/<model_name>`, log in, and agree to the license. Wait for the confirmation email before proceeding.

**Step 2**: Download the model:

```bash
tune download meta-llama/<model_name> --output-dir <model_dir> --hf-token <hf-token>
```

Models we've worked with:
- Llama-3.2-1B-Instruct (most common)
- Llama-3.1-8B-Instruct
- Llama-3.3-70B-Instruct

**Never commit your HuggingFace token to a repository.**

## Running Experiments

### With Claude Code (Recommended)

If you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code), use the built-in skills to automate the full pipeline:

| Step | Skill | What it does |
|------|-------|-------------|
| 1 | `/design-experiment` | Plan runs, create `experiment_summary.yaml` |
| 2 | `/scaffold-experiment` | Generate configs, SLURM scripts, directory structure |
| 3 | `/run-experiment` | Submit jobs, monitor progress, validate outputs |
| 4 | `/summarize-experiment` | Collect key metrics (loss, accuracy) into summary |

Additional skills: `/create-inspect-task` for custom evaluations, `/analyze-experiment` for visualizations.

See `.claude/skills/*/SKILL.md` for detailed documentation on each skill.

### Without Claude Code

See the [Workflow Guide](WORKFLOW_GUIDE.md) for step-by-step manual instructions covering single-run and multi-run experiments.

## Further Reading

| Document | Description |
|----------|-------------|
| [PREREQUISITES.md](PREREQUISITES.md) | Required skills, accounts, and tutorials |
| [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) | Manual workflow instructions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, skill diagrams |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Current limitations and workarounds |
| [SKILLS_ARCHITECTURE_SUMMARY.md](SKILLS_ARCHITECTURE_SUMMARY.md) | How skills are organized |
| `experiments/*/README.md` | Experiment-specific instructions |

## How to Cite

If you use cruijff_kit in your research, please cite our package. But remeber we are still in alpha.  If you are using cruijff_kit in your research please contact us.