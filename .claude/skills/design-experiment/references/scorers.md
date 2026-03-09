# Available Scorers

Reference for inspect-ai scorers available in cruijff_kit experiments.

## Built-in Scorers (inspect-ai)

### match

Exact string match between model output and target.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `location` | `"begin"` | Where to match: `"exact"`, `"begin"`, `"end"`, `"any"` |
| `ignore_case` | `True` | Case-insensitive matching |

**Typical usage:** Binary classification tasks where the model should output exactly "0" or "1".

```yaml
- name: "match"
  params:
    location: "exact"
    ignore_case: false
```

### includes

Checks if the target string appears anywhere in the model output.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ignore_case` | `True` | Case-insensitive matching |

**Typical usage:** Tasks where the answer may appear within a longer response.

```yaml
- name: "includes"
  params:
    ignore_case: false
```

## Custom Scorers (tools/inspect/scorers/)

### risk_scorer

Extracts risk scores from logprobs of the first generated token. Computes normalized probabilities over specified option tokens using softmax.

**Source:** `tools/inspect/scorers/risk_scorer.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `option_tokens` | `["0", "1"]` | Target answer tokens to extract probabilities for |

**Metrics produced:**
- `risk_score`: P(first option token) — only for binary tasks (2 option tokens)
- `option_probs`: Normalized probability distribution over all option tokens
- Standard `CORRECT`/`INCORRECT` based on exact match with target

**Requirements:**
- Task must set `GenerateConfig(logprobs=True, top_logprobs=20)`
- PYTHONPATH must include cruijff_kit repo root (handled by scaffold-inspect)

**Typical usage:** Binary prediction tasks (e.g., ACS income >$50k yes/no).

```yaml
- name: "risk_scorer"
  params:
    option_tokens: ["0", "1"]
```

**Multiclass example** (e.g., multiple-choice):

```yaml
- name: "risk_scorer"
  params:
    option_tokens: ["A", "B", "C", "D"]
```

Note: For multiclass tasks, `risk_score` will be `null` — use `option_probs` instead.

### numeric_risk_scorer

Parses a numeric probability from the model's text output (e.g., "0.73"). For tasks where the model directly outputs a risk score as text rather than via logprobs.

**Source:** `tools/inspect/scorers/numeric_risk_scorer.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `labels` | `("0", "1")` | The ground-truth target values in the dataset: (negative, positive). Used to determine correctness and compute calibration metrics. |

**How it works:**
1. Parses the model's text output as a float in [0, 1] — this is the `risk_score` (probability of the positive class)
2. Builds synthetic `option_probs`: `{negative_label: 1-risk, positive_label: risk}`
3. Thresholds at 0.5 to determine correctness: if risk >= 0.5, the predicted label is `labels[-1]` (positive); this is compared to `target.text`

**Metrics produced:**
- `risk_score`: The parsed probability value
- `option_probs`: Synthetic distribution over the two ground-truth labels
- `CORRECT`/`INCORRECT` based on thresholding at 0.5

**Requirements:**
- Model must output a single float in [0, 1]
- Does NOT require logprobs (works with any model, including API models)
- PYTHONPATH must include cruijff_kit repo root (handled by scaffold-inspect)

**Typical usage:** Binary prediction tasks where the prompt asks for a numeric probability.

```yaml
- name: "numeric_risk_scorer"
  params:
    labels: ["0", "1"]
```

**When to use `numeric_risk_scorer` vs `risk_scorer`:**
- Use `risk_scorer` when you want probabilities derived from logprobs (first token)
- Use `numeric_risk_scorer` when the model outputs a probability as text
- Both produce compatible metadata, so all calibration metrics (ECE, Brier, AUC) and visualization code work with either

## Design-Time Considerations

When a user selects scorers during experiment design, follow these guidelines:

- **Multiple scorers can be combined** — each runs independently and all results are stored in the eval log.
- **Scorer parameters** (e.g., `option_tokens`) are specified in experiment_summary.yaml and flow through to eval_config.yaml at scaffold time. Task files read them at runtime.
- **If `risk_scorer` is selected:** Ask which tokens represent the answer classes (default: `["0", "1"]` for binary). The task file will automatically request logprobs when `risk_scorer` is in the scorer list.
- **If `numeric_risk_scorer` is selected:** Ask what the ground-truth target values are in the dataset (default: `["0", "1"]` for binary). The model should be prompted to output a single probability as text.
- **If `match` is selected:** Ask whether matching should be exact, case-sensitive, etc. Defaults (`location: "exact"`, `ignore_case: false`) work well for constrained-output tasks.

## Common Scorer Combinations

### Binary classification with risk analysis
```yaml
scorer:
  - name: "match"
    params:
      location: "exact"
      ignore_case: false
  - name: "includes"
    params:
      ignore_case: false
  - name: "risk_scorer"
    params:
      option_tokens: ["0", "1"]
```

### Simple exact match
```yaml
scorer:
  - name: "match"
```
