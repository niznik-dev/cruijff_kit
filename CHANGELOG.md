# Changelog

All notable changes to cruijff_kit will be documented in this file.

## [Unreleased]

### Added

- `create-quiz` skill — turn one or two completed experiments into a self-contained, self-grading HTML quiz that tests a recipient's intuition about the results. (#453)

### Changed

- `python-markdown` added as a runtime dependency (used by the quiz renderer for intro / prompt / explanation / write-up markdown).

## [0.2.2] - 2026-04-23

### Added

#### Skills & Workflows
- `tabular-to-text` workflow — convert tabular data (CSV, Stata, Parquet) to textual representations for LLM fine-tuning, with dictionary, narrative, and LLM-generated formats (#414, @sarahepedersen)
- `model_organisms` framework — composable synthetic experiments on input × rule × format × design axes; unified inspect task replaces per-task scripts (#423)
- Seed plumbing from `experiment_summary.yaml` through to torchtune configs for full reproducibility (#427)

#### Evaluation & Metrics
- `split` arg in `TASK_ARG_KEYS` for parameterized inspect-ai task invocation (#420)
- Validation field fallback in `prebuild_cache.py` for datasets missing an explicit validation split (#422)

#### Observability & Testing
- Warning when `epochs_to_save` configuration will produce excessive checkpoints (#426)

### Changed

- `shuffle_acs_variables.py` generalized to any variable count, with per-sample variant added (#425, #432)
- Config-file values routed through argparse type converters to handle quoted YAML strings (#433)
- `check-release` skill updated with lessons from v0.2.1 release (#413)
- Adopt stricter `inspect-ai` version pinning to prevent CI/local drift after being bitten mid-release; update `DEFAULT_SCORERS` aliasing for the newer API (#423)
- Bump `actions/upload-artifact` 7.0.0 → 7.0.1 (#429)

### Fixed

- Evaluation template supports multi-GPU inspect eval (#410, @sarahepedersen)
- `report_generator` respects bundled `extra_splits` from train sidecars (#414)

### Removed

- Pre-framework sanity-check scripts superseded by `model_organisms`: `sanity_checks/{bit_sequences, predictable_or_not, bernoulli, count_digits, majority}` (#439)

## [0.2.1] - 2026-04-03

### Added

#### Skills & Workflows
- `archive-experiment` skill and CLI tool (#400)
- `check-release` skill for weekly release workflow (#404)

#### Observability & Testing
- GPU smoke test workflow for self-hosted della runner (#383)
- Nightly torchtune integration smoke test (#393)
- Nightly inspect-ai eval smoke test (#394)
- Extended GPU smoke test with model loading and scaffolding verification (#392)
- Tracked test fixtures for nightly GPU smoke tests (#396)
- Skill output checkpoint fixtures and tests (#377)

#### Documentation & Data
- Contributors and acknowledgments sections in README (#376)
- Maintainer contact information in README (#381)

### Changed

- SHA-pin and upgrade all GitHub Actions (#405, #411)
- Add Dependabot for GitHub Actions (#405)
- Bump actions/checkout 4→6 (#406), schneegans/dynamic-badges-action 1.7.0→1.8.0 (#409)
- Rewrite `spot_check` to use inspect-ai-style HF model loading (#397)
- Remove orphaned files and references (#397)

### Fixed

- Route SLURM output logs into experiment directories, fix double-slash in output path (#361)
- Include `vis_label` in dedup key to prevent cross-evaluation collapse (#390)
- `header_only=True` in `deduplicate_eval_files()` for ~300x speedup (#403)

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
