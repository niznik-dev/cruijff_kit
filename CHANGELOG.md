# Changelog

All notable changes to cruijff_kit will be documented in this file.

## [Unreleased]

### Added

- **Evaluation-only experiments**: `experiment_summary.yaml` gains a third run type, `eval-only`, for evaluating a pre-existing checkpoint (via `parameters.checkpoint_path`) without retraining. `scaffold-experiment` skips `scaffold-torchtune` entirely when no run is `fine-tuned`, so an experiment can be made up of base models and/or pre-existing checkpoints. (#478)

### Changed

#### Schema-key renames (`experiment_summary.yaml` / `eval_config.yaml`)

Each requires migrating existing config files; see the migration note per item.

- **`experiment.directory` → `experiment.dir`** — the lone spelled-out `directory` in a codebase that otherwise uses `dir`/`_dir` (and sits beside torchtune's `output_dir`/`checkpoint_dir`). Local `*_directory` identifiers collapse to `*_dir` too. **Migration:** rename `directory:` to `dir:` under `experiment`; `archive_experiment` fails loudly and names the legacy key on an un-migrated file. (#372)
- **`evaluation.scorer` → `evaluation.scorers`** — the key holds a list, so the plural matches its sibling `tasks:`. **Migration:** rename `scorer:` to `scorers:`. A leftover singular is silently ignored at runtime (falls back to default `match` + `includes`); `setup_inspect.py` warns about the unknown key at scaffold time. (#372)
- **`finetuned` → `is_finetuned`** (eval-config metadata) — aligns with our other `is_`-style booleans (`use_chat_template`, `emit_source_parquet`). **Migration:** rename in `eval_config.yaml`; eval logs written before this change leave the `is_finetuned` column null. (#372)
- **`data_generation.split` → `split_ratio`** (model-organism; a float train fraction) — matches the tabular generator and frees `split` to mean a *split name* everywhere. **Migration:** rename under `data.data_generation`. A stale `split:` is ignored (falls back to 0.8) but `prepare_data.py` warns when it sees it. (#372)
- **`data.training.label` → `dataset_label`** — the schema now matches the downstream name; bare `label` read as a class label. **Migration:** rename under `data.training`. (#372)
- **`controls.dataset_type` is now required** and used uniformly by training and eval — no longer inferred from the model name or a `MODEL_CONFIGS` default. An absent value is a hard error at every layer, preventing a silent chat-vs-text mismatch from corrupting train/eval parity. **Migration:** add `controls.dataset_type` (`chat_completion` | `text_completion`). (#478)
- **`output.base_directory` retired; `experiment.dir` is the single experiment root.** The unused `output.checkpoint_pattern` is dropped too. **Migration:** remove `output.base_directory` from existing files. (#442)
- **`experiment.seed` dropped** — training and eval each resolve an independent run-time seed via `resolve_seed`, which rejects non-integer seeds at scaffold time. **Migration:** remove `experiment.seed`. (#500)

#### Renames (modules / skills)

- **`compute_metrics.py` → `compute_gpu_metrics.py`** — disambiguates GPU/`seff` *hardware* metrics from the eval-accuracy `compute_metrics()` functions. Module-only. (#372)
- **`compute_metrics.json` → `compute_utilization.json`** (the compute-utilization summary artifact in `{experiment_dir}/exploration/`) — matches the `## Compute Utilization` report header it's documented under and ends the one-underscore clash with the unrelated raw-telemetry `gpu_metrics.csv`. The `COMPUTE_METRICS` log marker is renamed `COMPUTE_UTILIZATION` to match; `gpu_metrics.csv` keeps its name (it accurately labels the raw nvidia-smi dump). **Clean break:** `design-experiment` / `run-experiment` glob and read only the new name, so compute estimates won't seed from experiment dirs written before this change. (#571)
- **`analyze-to-pdf` skill → `md-to-pdf`** — it's a generic markdown→PDF pandoc wrapper, not an "analysis" step. **Migration:** invoke `/md-to-pdf`. (#372)
- **`output_*` path variables renamed to match the `artifacts/` layout** (and archive dedup). (#443)

#### Structure & workflow

- **Post-run flow restructured**: `summarize-experiment` is now the required post-run step, and the optional `analyze-experiment` skill is renamed `explore-experiment` (output dir `analysis/` → `exploration/`). `report_generator.py` is retired (renamed `pdf_preprocess.py`) — `explore-experiment` authors `report.md` directly. (#539)
- **Deterministic `experiment_summary.yaml` propagation** — `propagate_*_fields()` is a mandatory propagate-first step in scaffolding, so generated `eval_config.yaml` / `setup_finetune.yaml` derive from one verified source. (#502)
- Evaluation scaffolding (`scaffold-inspect`) no longer reads `setup_finetune.yaml` — `prompt` and `dataset_type` are propagated from `experiment_summary.yaml`, decoupling eval from the training artifact. (#478)
- **`utils/` namespace audit** — single-domain scripts moved out of `src/utils/` into their domain folders; the `utils/` charter is documented in `docs/ARCHITECTURE.md`. (#535)
- **`create-inspect-task` skill** updated to the current `eval_config.yaml` / per-cell architecture. (#566)

### Fixed

- `summary_binary` reads eval logs via inspect-ai's `read_eval_log()` API instead of hand-unzipping `.eval` archives. (#317)
- Nightly GPU smoke-test scaffolding now receives the required `dataset_type`. (#570)

### Documentation

- **Naming conventions + external-contract boundary written down** (`CLAUDE.md`, no renames): the `dir`-over-`directory` preference and the `dir`-vs-`path` (folder-vs-file) distinction, plus the names off-limits to renames because they belong to an external contract — torchtune recipe keys and inspect-ai's `@task`/`@scorer`/`@metric` registry names. Also records why `ck-setup` keeps its prefix (a bare `setup` would collide with a built-in skill). (#372)
- **Tool-domain folder convention + `inspect/`-shadows-stdlib hazard** documented (`docs/ARCHITECTURE.md`, `src/tools/inspect/__init__.py`) and locked by a guard test (`tests/unit/test_inspect_no_stdlib_shadow.py`). Stale `ck-experiments/` paths fixed in the `create-quiz` docs and a `summarize` test fixture. (#372)

## [0.3.3] - 2026-06-04

### Added

#### Skills & Workflows
- `summarize-experiment` is now continuous-scorer aware: it detects `continuous_scorer` runs and reports regression metrics (mae / rmse / r_squared / parse_rate) instead of silently falling back to accuracy. (#544)

#### Documentation & Data
- Sixth project principle, **Wrapper-only**, documented in `CLAUDE.md`: cruijff_kit integrates with external tools (torchtune, inspect-ai) by wrapping them, not modifying their internals. Capstone of the milestone-wide return to wrapper-only recipes. (#465)
- `continuous_scorer` is now documented in the scorers reference. (#543)

### Changed

- inspect-ai evaluations now default to `do_sample=false` (greedy decoding); `temperature` is optional and gated on `do_sample`. Empirically verified safe on Llama-3.2-1B-Instruct with chat template + logprobs. (#546)
- Risk scorer `METRIC_NAMES` are hoisted into a canonical module and shared via a public helper rather than reaching into inspect-ai's private `_METRICS`. (#540)
- Swept stale issue/PR-number references out of `src/` code comments, per the code-comment convention. (#542)

### Fixed

- `report_generator` no longer emits the misleading `## Risk Metrics` heading + C-ECE/R-ECE footnote on `continuous_scorer`-only experiments. The section now renders as `## Regression Metrics` with an mae / rmse / r_squared / parse_rate footnote. Mixed risk + regression experiments emit two sub-sections, one per category. (#519)
- GPU smoke test updated for the `cell.slurm` filename (per-cell evaluation layout). (#522)

### Removed

- `epochs_to_save` / `save_last_epoch_only` config keys and the `validate_epochs_to_save` helper from custom recipes and `setup_finetune.py`. Upstream torchtune's save-every-epoch behavior is restored; combine with `save_adapter_weights_only: True` (the default since 0.3.2) to keep disk usage modest. Part of the milestone-wide return to wrapper-only recipes. (#465, #525)
- Custom-metrics framework: `src/utils/finetune_custom_metrics.py`, the recipe-side `calculate_custom_metrics` integration in `_loss_step` / the training loop, and the try/except auto-disable guard. No known consumers in this repo; if needed, a wrapper-layer reimplementation can come back later. Part of the milestone-wide return to wrapper-only recipes. (#465, #526)
- `tqdm_miniters` config key and the recipe-side progress-bar override; the `tqdm` progress bars revert to upstream torchtune defaults. The override existed to keep SLURM logs short enough that log-reading wouldn't choke on chatty `tqdm` output; that constraint is obsolete now that large / carriage-return-heavy logs paginate cleanly (verified on a 626 KB / 10k-step log). Logs are more verbose without it (a controlled A/B measured ~25× more progress-bar lines/epoch at fast step-speed) but remain fully readable. Part of the milestone-wide return to wrapper-only recipes. (#465, #527)
- `num_workers` / `persistent_workers` DataLoader knobs: the `--num_workers` / `--persistent_workers` CLI flags on `setup_finetune.py`, their `finetune.yaml` propagation, and the recipe-side DataLoader wiring in all 3 custom recipes. Measured throughput gains were minimal; users needing parallel data loading can reintroduce it via a wrapper or recipe fork. Part of the milestone-wide return to wrapper-only recipes. (#465, #528)
- Dead `setup_logger` import block from the three custom recipes — it bound a module-level `logger` that was never used (the recipes log via torchtune's `log`). `src/utils/logger.py` itself is unchanged (still used by the inspect subtree). Part of the milestone-wide return to wrapper-only recipes. (#465, #529)

## [0.3.2] - 2026-05-21

### Added

#### Skills & Workflows
- `--training_samples` actually slices the training dataset; the existing step-count guard fired but the slice never happened (#494)
- `--num_workers` / `--persistent_workers` flags on `setup_finetune.py`, propagated through `finetune.yaml` to the DataLoader in all 3 custom recipes (#515)
- `port_cruijff_adapter` utility — restores adapter portability across machines after the local-base-path rewrite (#495)

#### Evaluation & Metrics
- `continuous_scorer` and ACS continuous-target tasks (`acs_age`, `acs_income_continuous`, `acs_hours`, `acs_commute`) for regression evals (#508, @EmnetSy, @sarahepedersen)
- `evaluation.temperature` and `evaluation.max_tokens` now thread from `experiment_summary.yaml` through `scaffold-inspect` to the inspect-ai task; a hard-coded `1e-7` was previously winning silently (#496)

#### Observability & Testing
- `scripts/check_env.py` preflight that compares installed versions against `==` pins in `pyproject.toml` (#507)
- `setup_finetune.py` warns on unknown keys in `setup_finetune.yaml` so typos no longer fall through to defaults (#505)
- `evaluation.temperature: 0` is explicitly rejected with a pointer to the underlying `do_sample` / `top_logprobs` mechanism (#496)

### Changed

- **Per-cell evaluation layout** — each `(run, task, epoch)` cell lives in `{run}/eval/{cell_name}/` with its own `eval_config.yaml`, `cell.slurm`, and `logs/`. **Breaking** for tooling hard-coded to the old `{task}_epoch{N}.slurm` shape; in-repo callers updated. (#498)
- Per-task `system_prompt` / `assistant_prefix` overrides in `experiment_summary.yaml` under `evaluation.tasks[]` (#498)
- **Adapter-only LoRA saves are now the default** (`save_adapter_weights_only=True`); `adapter_config.json`'s base path is rewritten to the local absolute path so adapters self-load via transformers' native PEFT auto-detection on offline compute (#495, #99)
- **Recipe patching policy** documented in `CLAUDE.md` and `.claude/PR_CHECKLIST.md` — outside-`train()` patches are fine; inside-`train()` patches require defensive guards (#465)
- `calculate_custom_metrics` wrapped in `try/except` with auto-disable; a buggy metric no longer crashes the run (#465)
- `epochs_to_save` validated at recipe init via new `validate_epochs_to_save()` helper; misformatted values raise `ValueError` instead of silently producing zero-checkpoint runs (#465)
- Adopt stricter `==` pinning for `inspect-viz` at `0.3.5` to match the `scores_*` import path used by `analyze-experiment` (#507, #503)

### Removed

- Embeddings extraction from the nightly recipe — broken and unused; #517 tracks a corrected reimplementation. Original feature came from #54 / #77. (#465)

### Fixed

- `report_generator` no longer emits an empty "Models evaluated: 0" report for `risk_scorer`-only experiments; renders a "Risk Metrics" section keyed by task instead (#504)
- `scaffold-inspect` recognizes `assistant_prefix` in `eval_config.yaml` and renders it as `-T assistant_prefix=...` instead of warning-and-dropping (#511)
- `scaffold-inspect` no longer silently improvises when per-cell overrides don't fit the shared layout — each cell now materializes deterministically to `eval/{cell_name}/` (#498)

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
