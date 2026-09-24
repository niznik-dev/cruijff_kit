# Known Issues

Limitations at the final release. Repository is archived; none of these will be fixed here.

## Installation

- **torchtune nightly, unpinned.** `make install` runs `pip install --pre --upgrade torchtune` with no version and takes the newest nightly. torchtune is deprecated upstream. Last known-good stack: torch 2.10.0+cu126, torchtune 0.7.0.dev20250929, torchao 0.14.1, transformers 4.57.6, inspect-ai 0.3.209, inspect-viz 0.3.5. Pin torchtune if imports fail. [#359](https://github.com/niznik-dev/cruijff_kit/issues/359)

## Scaffolding

- **`run_val_every_n_steps: 50` default.** ~3× training time on a 5k validation set; torchtune checkpoints per epoch only, so a mid-epoch timeout loses the run. Target ~5 validations per run. [#59](https://github.com/niznik-dev/cruijff_kit/issues/59)
- **No validation loss on multi-GPU.** The distributed recipe has no validation loop. `setup_finetune.py` warns. [#474](https://github.com/niznik-dev/cruijff_kit/issues/474)
- **`estimate_compute.py` is single-GPU, same-model only.** Refuses distributed runs; no cross-model multiplier; batch-size recommendation uses the prior run's GPU memory, not the target's. [#489](https://github.com/niznik-dev/cruijff_kit/issues/489), [#490](https://github.com/niznik-dev/cruijff_kit/issues/490), [#492](https://github.com/niznik-dev/cruijff_kit/issues/492), [#466](https://github.com/niznik-dev/cruijff_kit/issues/466)

## Running

- **Eval `--time` default is `0:10:00`.** 8B+ models can exceed it during load; job dies with no `.eval` file. Raise `--time`. [#589](https://github.com/niznik-dev/cruijff_kit/issues/589)
- **`max_connections` > 32 is slower.** 256 ran ~2× slower than 32 on a 3B eval; cause unprofiled. Keep the default. [#318](https://github.com/niznik-dev/cruijff_kit/issues/318), [#487](https://github.com/niznik-dev/cruijff_kit/issues/487)
- **Qwen3 emits `<think>` before the answer.** Breaks first-token logprob scoring and exact match. Set `evaluation.enable_thinking: false` or use `reasoning_risk_scorer`. Qwen3-32B and Qwen3 Base are not in `MODEL_CONFIGS`. [#600](https://github.com/niznik-dev/cruijff_kit/pull/600)
- **GGS blueprint: synthetic data only.** Requires output from [`ggs-hh-dk-synthetic`](https://github.com/niznik-dev/ggs-hh-dk-synthetic); baseline requires `catboost`, absent from the `cruijff` env. [#601](https://github.com/niznik-dev/cruijff_kit/pull/601)

## Results

- **`experiment_summary.yaml` is intent, not as-run.** OOM auto-retry halves `batch_size` in `finetune.yaml` (up to 3×) without updating the summary. Read `finetune.yaml` and grep the run log for `OOM_RETRY`. Re-scaffolding reverts the drift. [#588](https://github.com/niznik-dev/cruijff_kit/issues/588)
- **8B risk scores differ ~1e-5 across GPU nodes.** Bitwise identical on one node; up to 2.7e-5 apart on another. Text output identical; Brier/ECE/AUC shift. Pin the node for reproducibility. Not seen below 8B or on short sequences. [#547](https://github.com/niznik-dev/cruijff_kit/issues/547)
- **Unarchive never tested.** `archive-experiment` keeps `experiment_summary.yaml` and summaries and deletes checkpoints; rebuilding from that was never exercised. [#398](https://github.com/niznik-dev/cruijff_kit/issues/398)
