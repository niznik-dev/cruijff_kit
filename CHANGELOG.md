# Changelog

All notable changes to cruijff_kit will be documented in this file.

## [Unreleased]

### Added

#### Evaluation & Metrics
- `continuous_scorer` and ACS continuous-target tasks (`acs_age`, `acs_income_continuous`, `acs_hours`, `acs_commute`) for regression evals; reports `mae` / `rmse` / `r_squared` / `parse_rate` (#508, @EmnetSy, @sarahepedersen)

### Changed

- **Per-cell evaluation layout** — every `(run, task, epoch)` eval now lives in its own cell directory at `{run}/eval/{cell_name}/`, replacing the previous flat-eval layout where multiple cells shared a single `eval_config.yaml`. Each cell contains its own `eval_config.yaml`, `cell.slurm`, and `logs/`. This unblocks heterogeneous runs — two cells in the same run can carry different per-task overrides (e.g. `system_prompt`, `assistant_prefix`) without colliding. `submit_inspect` globs `*/eval/*/cell.slurm`; `setup_inspect.py` defaults its output to `cell.slurm`. **Breaking** for tooling that hard-coded the old `{run}/eval/{task}_epoch{N}.slurm` shape; in-repo callers (archive-experiment, scaffold-inspect agent, run-experiment evaluators) have been updated. (#498)
- **Per-task `system_prompt` / `assistant_prefix` overrides** in `experiment_summary.yaml` under `evaluation.tasks[]` — when set, override the experiment-wide values for cells produced from that task. Enables cue-presence ablations and other prompt-variation experiments. (#498)
- **Recipe patching policy** documented in `CLAUDE.md` and `.claude/PR_CHECKLIST.md` — location-based rule for cruijff_kit divergences from torchtune: outside-`train()` patches are fine where they are; inside-`train()` patches require defensive guards (init validation + unit test). (#465)
- `calculate_custom_metrics` is now wrapped in `try/except` with auto-disable in all 3 recipes; a buggy metric function logs a warning and disables custom metrics for the rest of the run instead of crashing training. (#465)
- `epochs_to_save` is now validated at recipe init via the new `validate_epochs_to_save()` helper — misformatted values (out-of-range indices, empty lists, wrong types) raise `ValueError` with a clear message instead of silently producing zero-checkpoint runs. (#465)

### Removed

- **Embeddings extraction from the nightly recipe** (`get_embeddings`, `_get_embeddings`, `embeddings_output_path`) — the feature was broken (`.item()` on multi-element tensors, train mode never restored, double forward pass, never wrote to disk) and unused in any current config. #517 tracks a corrected reimplementation; original feature came from #54 / #77. (#465)

### Fixed

- `report_generator` no longer emits an empty "Models evaluated: 0" report for `risk_scorer`-only experiments. When `*_accuracy` columns are absent but supplementary risk metrics exist, the report now renders a "Risk Metrics" section, derives `model_count` from the calibration results, and picks a best performer by AUC (or Brier) in the executive summary. `extract_calibration_metrics` also groups by `task_name` so per-task variation (e.g. cue vs no-cue) renders as separate rows. (#504)
- `scaffold-inspect` now recognizes `assistant_prefix` in `eval_config.yaml` and renders it as a `-T assistant_prefix=...` flag in the generated SLURM script. Previously, the key was warned as unknown and dropped — base / Instruct models that needed a prefill to emit option tokens (`"0"` / `"1"`) for `risk_scorer` would silently produce all-NaN risk metrics. Values are emitted with strict YAML-double-quoted inside shell-single-quoted form so common cases like `"Answer: "` survive inspect-ai's per-value YAML parse. (#511)
- `scaffold-inspect` no longer silently improvises when an experiment has per-cell overrides that don't fit the shared `eval_config.yaml` model. The COLING-cue trigger case (9 runs × 2 prompt cues per run = 18 cells) previously got a nested layout `submit_inspect` couldn't see, producing `ALL_COMPLETE: {}` on zero work. The scaffolder is now deterministic — each cell is materialized into a canonical `eval/{cell_name}/` directory. (#498)

## [0.3.1] - 2026-05-14

### Added

#### Skills & Workflows
- **Callable `run-experiment` submitter platform** — replaces the prose-only execution path the skill used to rely on:
  - `src/tools/run/submit_torchtune.py` and `submit_inspect.py` — callable submitters with QoS-aware drip-feed, resume-safe state, and canonical `SUBMIT_JOB:` / `SUBMIT_EVAL:` logging (#451)
  - `src/tools/run/status.py` — read-only snapshot that refreshes in-flight entries from `squeue`/`sacct` and prints a table or `--json` (#479)
  - Detach mechanisms — SIGINT, SIGTERM, and a sticky `<exp_dir>/logs/.detach` sentinel; all paths flush state and exit cleanly while SLURM jobs continue (#479)
  - `--resume-monitor` flag on both submitters — re-attaches a watcher to an existing state file (#479)
  - Two-layer submitter config for `poll_sec` / `stagger_sec` / `max_submit` — per-checkout defaults at `<repo>/.config/config.json`, live mid-run overrides at `<exp_dir>/logs/monitor.json`, plus matching CLI flags (#480)
  - OOM auto-retry — detected OOM fine-tunes resubmit with halved `batch_size`, up to 3 times per run; `--no-retry` to disable (#254)
- `--batch_size_val` flag on `setup_finetune.py`, honored by `_single_device_nightly` for a larger validation DataLoader (#450)
- `--cpus_per_task` flag on `setup_finetune.py`, overriding the per-model SLURM default (#449)

#### Evaluation & Metrics
- `evaluation.max_connections` plumbed through scaffold-inspect to inspect-ai; default 32 (raising it can slow variable-length workloads — opt in with evidence) (#318)
- `assistant_prefix` field in the ACS inspect task to autofill the start of the model's response (#485, @sarahepedersen)
- `top_logprobs` exposed in the ACS inspect task (#485, @sarahepedersen)

#### Observability & Testing
- **Throughput-based wall-time estimation** — `estimate_compute` now predicts `total_tokens / tps_gpu × margin` instead of linear scaling; tps parsed from prior runs' slurm-out via the new `throughput_parsers.py` (single-GPU only in v1) (#473)
- `inspect eval` wrapped in `/usr/bin/time -p` for a format-stable wall-time anchor (#473)
- `THROUGHPUT_STATUS` mid-run heartbeat that warns when fine-tune tps falls below 70% of predicted (#473)
- `eval_dataset_size` field added to `compute_summary.build_summary()` (#473)
- `harvest_jids_from_run_logs()` in `compute_metrics.py` — single helper for analyze-experiment to extract job IDs from run logs (#451)
- `log_all_complete()` is idempotent — at most one `ALL_COMPLETE` block per per-tool log (#479)

### Changed

- `estimate_compute.scale_finetune_time` / `scale_eval_time` signatures swapped to throughput-based inputs; `estimate_from_prior` gains a required `new_seq_len` parameter (#473)
- `scaffold-torchtune` defaults single-GPU runs to `_single_device_nightly` so `validation_during_training` is no longer silently dropped (#471)
- `setup_finetune.py` picks a GPU-aware default for `--custom_recipe`, anchoring the choice in code rather than agent prose (#471)
- `run-experiment` skill invokes the callable submitters; skill docs collapse to schema descriptions (#451)
- `analyze-experiment` surfaces missing run logs as a stderr warning + `report.md` note instead of silently skipping the Compute Utilization section (#451)

### Fixed

- `setup_finetune.py` errors at scaffold time when GPU-count auto-switch resolves to a missing recipe, instead of failing late at SLURM runtime (#471)
- `setup_finetune.py` warns when `validation_during_training` is requested for a multi-GPU run (#471)
- Eval-side submitter no longer collapses distinct runs onto a single state-file key when their eval slurm filenames match; key now includes the relative path (#451)
- `setup_finetune.py` ignores `new_system_prompt` with a warning when `dataset_type` is `text_completion` (base models have no chat template) (#485, @sarahepedersen)

### Removed

- `MAX_SUBMIT` / `POLL_SEC` / `STAGGER_SEC` env vars, in favor of `<repo>/.config/config.json` (#480)

## [0.3.0] - 2026-05-07

### Added

#### Skills & Workflows
- `/setup` skill — interactive first-time walkthrough of `claude.local.md.template`, or health check for an existing config (#458)
- `create-quiz` skill — turn one or two completed experiments into a self-grading HTML quiz that tests a recipient's intuition (#454, @msalganik)
- Compute estimates in `design-experiment` based on previous runs (#349, @sarahepedersen)

#### Evaluation & Metrics
- `weighted_sum` and `weighted_sum_binary` rules in the `model_organisms` framework — linear-DGP outputs (`w·x + intercept`) with optional Bernoulli noise; analytical or 20k-sample Monte Carlo Bayes-accuracy reporting (#462, @msalganik)
- `Rule.prepare` hook for dataset-level state (resolved weights, format width) and `Rule.supports_ood` flag for rules to opt out of OOD designs (#462, @msalganik)

### Changed

- **Breaking:** Folder reorganization — flat `src/` layout, `projects/` → `blueprints/`, all artifacts unified under `ck-projects/{project}/{experiment_name}/`. The `experiments/`, `data/`, and `synthetic_twins/` directories are retired; input data is now user-provisioned via `{ck_data_dir}`. `experiment_summary.yaml` schema gains `experiment.project` and drops `type`. (#441)
- **Breaking:** `model_organism` inspect task replaces `calibration` flag with `logprobs` (capability switch) and `top_logprobs`. Logprobs auto-enable when a configured scorer declares `requires_logprobs`. Existing `-T calibration=true` configs must migrate to placing `risk_scorer` in `scorer:` (for calibration metrics) or `-T logprobs=true` (raw logprobs only). (#463, @msalganik)
- **Breaking:** Renamed `blueprints/model_organism/` → `blueprints/model_organisms/` for consistency with the `src/tools/model_organisms/` module. Existing `experiment_summary.yaml` files with `experiment.project: model_organism` must update to `model_organisms`. (#462)
- `design-experiment` defers `claude.local.md` validation to `/setup` (#458)
- Workflow test specs converted from YAML to plaintext briefs (#445)
- Dropped legacy "synthetic" and "sanity check" framing from model-organisms documentation, blueprint READMEs, and the `design-experiment` skill — described as sequence-labeling tasks with known ground-truth rules instead (#462)
- `python-markdown` added as a runtime dependency (used by `create-quiz` for markdown rendering)

### Fixed

- `extract_loss` regex now captures scientific-notation losses (#456)
- `--experiment_name` passed in GPU smoke test scaffolding (#469)

### Removed

- Parquet support in fine-tune/eval workflow, subsumed by folder reorg (#441)
- Duplicative `workflows/` directory in run-experiment skill (#468)
- Vestigial `sanity_check vs research experiment` dichotomy in the `design-experiment` skill — was unused post-folder-reorg (#441), with the `experiment.type` field already removed from the schema (#462)

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
