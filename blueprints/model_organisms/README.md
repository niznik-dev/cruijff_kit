# Model Organisms

Sequence-labeling tasks with known ground-truth rules — useful for verifying that fine-tuning and evaluation workflows produce expected results. A "model organism" experiment is defined by four choices:

| Axis | Options | Role |
|---|---|---|
| **input_type** | `bits`, `digits`, `letters` | What the model sees in each prompt |
| **rule** | `parity`, `first`, `last`, `nth`, `length`, `constant`, `coin`, `majority`, `min`, `max`, `weighted_sum`, `weighted_sum_binary` | How the target is computed from the input |
| **fmt** (format) | `spaced`, `dense`, `comma`, `tab`, `pipe` | How the sequence is rendered as text |
| **design** | `memorization`, `in_distribution`, `ood` | What train/test split you want |

Any `input_type × rule × fmt × design` combination produces a self-contained dataset with known ground truth, making it trivial to detect silent workflow regressions.

## How this project is different

Unlike `blueprints/capitalization/` and `blueprints/folktexts/`, this folder does **not** contain its own `inspect_task.py` or `generate_data.py`. The code that generates data and evaluates model performance is fully generic — it composes primitives from `src/tools/model_organisms/` (`inputs.py`, `rules.py`, `formats.py`) and does not need per-project customization. In that sense, the "project" here is a conceptual label, not a distinct code unit; all implementation lives in the library.

## Code locations

| What | Where |
|---|---|
| Primitives (inputs, rules, formats) | `src/tools/model_organisms/` |
| Dataset generator CLI | `cruijff_kit.tools.model_organisms.generate` |
| Inspect-ai evaluation task | `cruijff_kit.tools.model_organisms.inspect_task` |

## How to use

Add a `data_generation` block to your `experiment_summary.yaml`:

```yaml
data:
  data_generation:
    tool: model_organism
    name: my_experiment_name
    input_type: bits          # bits | digits | letters
    rule: parity              # see rule registry in src/tools/model_organisms/rules.py
    k: 8                      # sequence length
    N: 1000                   # number of samples
    design: in_distribution   # memorization | in_distribution | ood
    split_ratio: 0.8          # train fraction (for in_distribution / ood)
    output_path: dataset.json
    # optional:
    fmt: spaced               # defaults to spaced
    rule_kwargs: {p: 0.7}     # extra params for rules like coin
    ood_tests: [...]          # required when design: ood
```

The `scaffold-experiment` skill reads this block and invokes the generator automatically before scaffolding the fine-tuning runs.

### Linear DGP rules (`weighted_sum`, `weighted_sum_binary`)

These rules apply to `bits` and `digits` and define a linear data-generating process `output = w · x + intercept`. They share these `rule_kwargs`:

```yaml
rule_kwargs:
  # Either give weights explicitly...
  weights: [1, 2, -1, 3, 0, 1, -2, 1]
  # ...or let them be drawn:
  weight_max: 3                # magnitudes uniform on {-W,…,-1,1,…,W}
  sparsity: 0.0                # P(drawn weight masked to 0)
  weight_seed: 42              # default = dataset seed

  intercept: balanced          # int, or "balanced" (round(-sum(w)·E[x_i]))

  # weighted_sum_binary only:
  noise_scale: 0.0             # 0 → deterministic z>0 threshold;
                               # >0 → σ-Bernoulli sample (deterministic
                               # per (weight_seed, sequence))
```

`weighted_sum` formats outputs as a spaced + signed + zero-padded integer string (e.g. `+ 4 2`); width is computed once per dataset from the resolved weights and intercept. `weighted_sum_binary` outputs `"0"` or `"1"`.

Top-level metadata captures `resolved_weights`, `intercept`, `format_width`, `weight_seed`, and (for binary stochastic) `bayes_accuracy` — the optimal-classifier upper bound.

To run the generator manually (outside the skill workflow):

```bash
python -m cruijff_kit.tools.model_organisms.generate \
    --name my_dataset \
    --input_type bits \
    --rule parity \
    --k 8 \
    --N 1000 \
    --design in_distribution \
    --output dataset.json
```

Evaluation is configured by setting `inspect_task` to `cruijff_kit.tools.model_organisms.inspect_task` in your evaluation config.
