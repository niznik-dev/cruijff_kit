# Changelog

All notable changes to cruijff_kit will be documented in this file.

## [Unreleased]

### Added

- `evaluation.max_connections` field in `experiment_summary.yaml`, plumbed through `scaffold-inspect` → `eval_config.yaml` → `setup_inspect.py` as `--max-connections N` on the generated `inspect eval` command. Default matches inspect-ai upstream (32); a sensitivity test on the 318 branch showed that raising it on variable-length workloads (verbose ACS prompts) can actually *slow* evals down (mc=256 ran 23% slower at 0% GPU util versus mc=8 at 100% GPU util on Llama-3.2-3B-Instruct + ACS employment, due to padding waste at large batch sizes). Users can opt into higher values per experiment when they have evidence those help their workload. (#318)
- **OOM auto-retry in `run-experiment`** (#254 PR A) — fine-tune jobs detected as OOM are now automatically resubmitted with `batch_size` halved in `finetune.yaml`, up to 3 times per run. Detection covers both SLURM `OUT_OF_MEMORY` (host RAM cgroup kill) and SLURM `FAILED` jobs whose `slurm-<jid>.out` shows `torch.OutOfMemoryError` / "CUDA out of memory" — the latter is the common case for fine-tunes since CUDA OOMs are Python exceptions, not cgroup events. Each retry appends a `{"batch_size": N}` entry to a per-run `retry_history` list in the state file (forward-compat with multi-knob strategies), and the entry records `oom_detected_via` as `"slurm_state"` or `"cuda_log"` for forensics. Logs new `OOM_RETRY` blocks per attempt and a loud `OOM_EXHAUSTED` block when the limit is hit. Default ON; pass `--no-retry` to either submitter to disable. That gate also covers future non-OOM retry strategies. Active escalation (email/terminal prompt on exhaustion) is deferred to a follow-up PR.
- `--batch_size_val` flag on `setup_finetune.py`, wired through `scaffold-torchtune` as `controls.batch_size_val` (also overrideable per-run via `runs[].parameters.batch_size_val`). Honored by the `_single_device_nightly` recipe (the default since #475), which uses it to construct a larger-batch validation DataLoader; recipes that don't read the field silently ignore it. Absent the flag, the recipe falls back to `cfg.batch_size`. (#450)
- `--cpus_per_task` flag on `setup_finetune.py`, wired through `scaffold-torchtune` as `runs[].compute.cpus_per_task`. Overrides `MODEL_CONFIGS[model]["slurm"]["cpus"]` for the generated `#SBATCH --cpus-per-task=`. Useful for workloads with long sequences or heavy preprocessing where tokenizer threading and OpenMP can give a small GPU-util bump even at `num_workers=0`. Does *not* parallelize data loading itself — that's recipe-level and tracked separately. (#449)
- `src/tools/run/submit_torchtune.py` and `submit_inspect.py` — callable submitters for `run-experiment`. Drip-feed against the gpu-test QoS cap (default `MAX_SUBMIT=25`), 5-second stagger, resume-safe JSON state file, canonical `SUBMIT_JOB:` / `SUBMIT_EVAL:` log emission. Replaces the prose-only execution path the skill used to rely on. (#451)
- `harvest_jids_from_run_logs()` in `src/tools/slurm/compute_metrics.py` — single helper that `analyze-experiment` calls to extract job IDs from `run-torchtune.log` / `run-inspect.log` and surface loud warnings when those logs are missing or malformed. (#451)
- Detach mechanisms for `run-experiment` submitters — SIGINT, SIGTERM, and a sticky `<exp_dir>/logs/.detach` sentinel file. Any path flushes state, emits a canonical `MONITOR_DETACHED` log block, prints a re-attach hint to stderr, and exits cleanly; SLURM jobs continue running. (#479)
- `--resume-monitor` flag on `submit_torchtune` and `submit_inspect` — skips the submit phase and re-attaches a watcher to the existing state file. Idempotent; safe to invoke repeatedly. (#479)
- `src/tools/run/status.py` — consolidated read-only snapshot command. Reads both state files, refreshes in-flight entries from `squeue`/`sacct`, prints a table (or `--json`). Emits a per-tool `ALL_COMPLETE` block when refresh first observes all-terminal so the pipeline log closes cleanly without a follow-up `--resume-monitor`. Composes with `/loop`, cron, or interactive use. (#479)
- `log_all_complete()` is now idempotent — at most one `ALL_COMPLETE` block per per-tool log regardless of how many submitters or `status` calls emit it. (#479)
- Two layers of submitter-knob configuration for `poll_sec`, `stagger_sec`, `max_submit` (#480):
  - **Per-checkout defaults** at `<repo>/.config/config.json`, a tracked file shipping with the built-in values visible. Power users edit it to change defaults across all future experiments; `git update-index --skip-worktree .config/config.json` keeps local edits out of `git status`.
  - **Live mid-run overrides** at `<exp_dir>/logs/monitor.json`. The watcher re-reads it on every poll iteration. Same filesystem-as-control-plane shape as the `.detach` sentinel.
  - New `--poll-sec` / `--stagger-sec` CLI flags on both submitters (alongside the existing `--max-submit`).
  - Precedence: `monitor.json` > CLI flag > `<repo>/.config/config.json` > built-in default. Bad values warn-and-skip; missing files are silent no-ops. Each applied `monitor.json` change emits a `MONITOR_CONFIG` block to the per-tool log.
  - Removed the previously-supported `MAX_SUBMIT` / `POLL_SEC` / `STAGGER_SEC` env vars in favor of the user-config file. None had been documented or set anywhere in the repo.

### Changed

- `scaffold-torchtune` defaults single-GPU runs to `_single_device_nightly` so `validation_during_training` is no longer silently dropped (#471)
- `setup_finetune.py` now picks a GPU-aware default for `--custom_recipe` (single-GPU → `_single_device_nightly`, multi-GPU → `_distributed_stable`), anchoring the recipe choice in code rather than agent prose (#471)
- `run-experiment` skill now invokes the callable submitters instead of executing prose-by-prose recipes; the skill docs collapse to schema descriptions (#451)
- `analyze-experiment` no longer silently skips the Compute Utilization section when run logs are missing; a `WARNING:` is printed to stderr and the absence is surfaced as a visible note in `report.md` (#451)

### Fixed

- `setup_finetune.py` now errors at scaffold time when the GPU-count auto-switch resolves to a non-existent recipe (e.g. `_distributed_nightly`), instead of failing late at SLURM runtime. Multi-GPU val support tracked in #474. (#471)
- `setup_finetune.py` now warns when `validation_during_training` is requested for a multi-GPU run, in addition to the agent-side warning (#471)
- Eval-side submitter no longer collapses distinct runs onto a single state-file key when each run has the same eval slurm filename (the bug that bit the `lostmiddle_kxp_3B` experiment, issue #451 comment #2). State key now includes the relative path. (#451)

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
