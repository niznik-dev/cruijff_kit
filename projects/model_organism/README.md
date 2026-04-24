# Model Organism

Synthetic sequence-labeling tasks for sanity-checking fine-tuning and evaluation workflows. A "model organism" experiment is defined by four choices:

| Axis | Options | Role |
|---|---|---|
| **input_type** | `bits`, `digits`, `letters` | What the model sees in each prompt |
| **rule** | `parity`, `first`, `last`, `nth`, `length`, `constant`, `coin`, `majority`, `min`, `max` | How the target is computed from the input |
| **fmt** (format) | `spaced`, `dense`, `comma`, `tab`, `pipe` | How the sequence is rendered as text |
| **design** | `memorization`, `in_distribution`, `ood` | What train/test split you want |

Any `input_type × rule × fmt × design` combination produces a self-contained dataset with known ground truth, making it trivial to detect silent workflow regressions.

## How this project is different

Unlike `projects/capitalization/` and `projects/folktexts/`, this folder does **not** contain its own `inspect_task.py` or `generate_data.py`. The code that generates data and evaluates model performance is fully generic — it composes primitives from `src/tools/model_organisms/` (`inputs.py`, `rules.py`, `formats.py`) and does not need per-project customization. In that sense, the "project" here is a conceptual label, not a distinct code unit; all implementation lives in the library.

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
    split: 0.8                # train fraction (for in_distribution / ood)
    output_path: dataset.json
    # optional:
    fmt: spaced               # defaults to spaced
    rule_kwargs: {p: 0.7}     # extra params for rules like coin
    ood_tests: [...]          # required when design: ood
```

The `scaffold-experiment` skill reads this block and invokes the generator automatically before scaffolding the fine-tuning runs.

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
