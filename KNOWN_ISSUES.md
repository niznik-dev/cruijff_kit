# Known Issues

cruijff_kit is archived. This is the list of limitations known at the final release. Issue numbers link to the full threads, which stay readable on GitHub.

## Installation

**The dependency stack is frozen.** ([#359](https://github.com/niznik-dev/cruijff_kit/issues/359)) torchtune is deprecated upstream; the kit uses a nightly build for validation-loss support. `make install` pulls torchtune with `pip install --pre --upgrade` and no version, so it installs whatever the newest nightly is. The stack that ran every experiment through September 2026:

| package | version |
|---|---|
| torch | 2.10.0+cu126 |
| torchtune | 0.7.0.dev20250929 |
| torchao | 0.14.1 |
| transformers | 4.57.6 |
| inspect-ai | 0.3.209 |
| inspect-viz | 0.3.5 |

`pyproject.toml` pins torchao `<0.15` and transformers `<5`; inspect-ai and inspect-viz are pinned exactly. If an install misbehaves, pin torchtune to the version above.

## Scaffolding

**Validation every 50 steps is the default and it is expensive.** ([#59](https://github.com/niznik-dev/cruijff_kit/issues/59)) With `validation_during_training` on, the scaffold writes `run_val_every_n_steps: 50`. Over a ~5k validation set that is roughly 3× the training time, and torchtune checkpoints only per epoch, so a run that hits its time limit mid-epoch leaves nothing. Set the step count to validate about five times per run.

**Multi-GPU fine-tunes have no mid-training validation loss.** ([#474](https://github.com/niznik-dev/cruijff_kit/issues/474)) The distributed recipe has no validation loop. Any run on more than one GPU (70B by default) trains blind between epochs. `setup_finetune.py` warns at scaffold time.

**The wall-time predictor covers one case.** ([#489](https://github.com/niznik-dev/cruijff_kit/issues/489), [#490](https://github.com/niznik-dev/cruijff_kit/issues/490), [#492](https://github.com/niznik-dev/cruijff_kit/issues/492), [#466](https://github.com/niznik-dev/cruijff_kit/issues/466)) `estimate_compute.py` predicts from a prior run's throughput and is reliable only for the same model on a single GPU of the same class. It refuses distributed runs, emits a warning instead of a multiplier for cross-model estimates, and sizes batch recommendations from the prior run's GPU memory, not the target's.

## Running

**The eval SLURM time limit defaults to 10 minutes.** ([#589](https://github.com/niznik-dev/cruijff_kit/issues/589)) `setup_inspect.py --time` defaults to `0:10:00`. An 8B or larger model loaded from disk can exceed that during model load alone; the job dies before writing a `.eval` file. Raise `--time` for large models.

**`max_connections` above 32 slows evals down.** ([#318](https://github.com/niznik-dev/cruijff_kit/issues/318), [#487](https://github.com/niznik-dev/cruijff_kit/issues/487), [#488](https://github.com/niznik-dev/cruijff_kit/issues/488)) On a 3B ACS eval, 256 connections ran ~2× slower than 32. The cause was never profiled and does not appear to be GPU-side. Leave `evaluation.max_connections` at the default.

**Qwen3 instruct models emit a `<think>` block before answering.** ([#600](https://github.com/niznik-dev/cruijff_kit/pull/600)) This breaks first-token logprob scoring and exact-match accuracy. Either set `enable_thinking: false` under `evaluation` in `eval.yaml`, or use `reasoning_risk_scorer`, which locates the answer after `</think>`. Qwen3-32B and the Qwen3 Base checkpoints download but are not in `MODEL_CONFIGS`.

**The GGS blueprint is validated on synthetic data only.** ([#601](https://github.com/niznik-dev/cruijff_kit/pull/601)) `blueprints/ggs/` expects the books-of-life JSON from the separate [`ggs-hh-dk-synthetic`](https://github.com/niznik-dev/ggs-hh-dk-synthetic) generator. Its CatBoost baseline needs an environment with `catboost`, which the `cruijff` env does not include.

## Reading results

**`experiment_summary.yaml` records design intent, not what ran.** ([#588](https://github.com/niznik-dev/cruijff_kit/issues/588)) The OOM auto-retry halves `batch_size` in the run's `finetune.yaml` and resubmits, up to three times, without updating the summary. Treat the per-run `finetune.yaml` as the as-run record and check the run log for `OOM_RETRY` lines before quoting a batch size. Do not re-scaffold a drifted run; scaffold regenerates `finetune.yaml` from `controls` and reverts the change.

**8B risk scores reproduce only to ~1e-5 across GPU nodes.** ([#547](https://github.com/niznik-dev/cruijff_kit/issues/547)) Same model, data, and greedy decoding: scores are bitwise identical on the same node and differ by up to 2.7e-5 on a different node. Generated text is identical; aggregate calibration metrics (Brier, ECE, AUC) move with the scores. Pin the SLURM node if you need bitwise reproducibility. Not observed below 8B or on short-sequence tasks.

**The unarchive round-trip was never tested.** ([#398](https://github.com/niznik-dev/cruijff_kit/issues/398)) `archive-experiment` deletes checkpoints and keeps `experiment_summary.yaml` and the summaries on the claim that the metadata is enough to rebuild the run. Nobody ever rebuilt one. Expect `experiment_summary.yaml` → scaffold → run to work in principle and to differ from the original at the ~1e-5 level (#547) in practice.
