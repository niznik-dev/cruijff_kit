# Changelog

All notable changes to cruijff_kit will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-03-11

### Added

#### Skills & Workflows
- `analyze-experiment` skill with inspect-viz integration for interactive HTML plots (#274, @shelby-tisdale; #287)
- `analyze-to-pdf` skill for converting reports to PDF via pandoc (#304)
- Per-sample risk plots: ROC curves, calibration curves, prediction histograms (#310)
- Provenance metadata and `inspect view` commands in analysis reports (#308, #309)
- Future directions section in analysis reports
- Eval SLURM template and `setup_inspect.py` renderer (#328)
- Eval metadata flags (epoch, finetuned, source_model) in scaffold-inspect (#275)
- HF datasets cache pre-building to prevent eval race conditions (#322)
- `HF_HUB_OFFLINE=1` in eval SLURM template (#354)

#### Evaluation & Metrics
- `risk_scorer` — softmax probability extraction from logprobs (#293, @sarahepedersen)
- ECE, Brier Score, and AUC calibration metrics (#296)
- `risk_calibration_error` (R-ECE) metric immune to inspect-ai Score.value bug (#307)
- Balanced accuracy and F1 score in `summary_binary.py` (#266)
- Consolidated ACS eval tasks into single file with aliases (#253)
- `text_completion` data format for base model evaluation (#277, @sarahepedersen)
- Qwen 2.5 support and refactored model-specific behavior (#277, @sarahepedersen)
- Mistral tokenizer support (#292)
- `SUPPORTED_MODELS.md` with all 7 supported models (#291)
- Torchtune base recipe support (#260)

#### Observability & Testing
- GPU metrics capture with dual-source (nvidia-smi + sacct) (#319)
- CPU metrics via jobstats integration (#331)
- MIG detection and sacct time limit fallback (#319)
- Training step guard with 3x warmup threshold check (#334)
- Ruff linting + formatting with CI integration and Makefile targets (#343)
- Unit tests for 5 previously untested modules (#340)
- 12 new `parse_eval_log` tests + `setup_finetune main()` tests (#340)
- Coverage badge via shields.io + gist (#340)
- Pre-commit linting guidance in CLAUDE.md

#### Documentation & Data
- Refactored README: extracted prerequisites, manual workflow, added diagrams (#290)
- `PREREQUISITES.md` and `WORKFLOW_GUIDE.md` as standalone docs
- `ACS_EXAMPLE.md` onboarding walkthrough for ACS experiments (#348)
- Claude Code installation instructions (#268)
- `claude.local.md` validation and cleaned-up template (#291)
- `ARTIFACT_LOCATIONS.md` for canonical experiment directory layout (#341)
- `--balanced` flag for equal class sampling in ACS extraction (#279)
- Folktexts experiment infrastructure and ACS data prep scripts (#253)
- ML baseline script (catboost) for comparison
- Pinned folktexts HuggingFace dataset revision (#342)

### Changed

- Standardized experiment artifact locations (#341)
- Removed cluster-specific GPU constraints from model configs — now cluster-agnostic (#335)
- Removed hardcoded Della paths from setup_finetune, design-experiment, templates (#301, #312, #345, #353)
- Replaced hardcoded step threshold with dynamic 3x warmup check (#334)
- Reduced SLURM CPU allocation to 1 per GPU for all models (#331)
- Default 1B model to gpu80 partition (#319)
- Replaced `TQDM_MININTERVAL` env var with `tqdm_miniters` config option (#269)
- Removed `base_recipe` from default experiment workflow (#333)
- Removed baseline separation from analysis reports (#306)
- Pinned dependency version floors: torch, torchtune, torchao (#344)
- Pinned inspect-ai>=0.3.163 and transformers<5 (#296)
- Pinned torchao<0.15 — v0.15 removed `int4_weight_only` needed by torchtune nightly (#355)

### Fixed

- ECE metric handling of post-reduction float Score.value (#307)
- nvidia-smi field parsing crash with edge cases (#331)
- Boolean normalization bug in setup_inspect (#328)
- pandas NA handling in parse_eval_metadata (#274)
- inspect-viz slash bug: sanitize `/` in metric names for DuckDB (#296)
- Skip diagonal heatmaps when model×task is 1:1 (#296)
- PDF figure ordering with `-implicit_figures` pandoc flag (#304)
- Fix recipe caching (#271)

### Deprecated

- `llm_utils` quarantined as experimental with mask bug fix (#346)

## [0.1.1] - 2026-01-08

### Changed

- **Descriptive model names in eval**: Evaluation scripts now use descriptive model names (e.g., `hf/1B_normal_epoch_4`, `hf/3B_base`) instead of generic `hf/local` (#251)

## [0.1.0] - 2026-01-07

Initial alpha release.

### What Works

- **Core workflow**: design → scaffold → run → summarize experiments
- **Fine-tuning**: LoRA fine-tuning via torchtune with validation loss tracking
- **Evaluation**: inspect-ai integration for model evaluation
- **Multi-model support**: Llama 1B, 3B, 8B, and 70B with model-aware SLURM allocation
- **Claude Code skills**: Automated workflows for experiment design, scaffolding, and execution
- **Installation**: Automated setup via `make install`

### Known Limitations

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for current limitations and workarounds.
