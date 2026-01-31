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

## Design-Time Considerations

When a user selects scorers during experiment design, follow these guidelines:

- **Multiple scorers can be combined** — each runs independently and all results are stored in the eval log.
- **Scorer parameters** (e.g., `option_tokens`) are specified in experiment_summary.yaml and flow through to eval_config.yaml at scaffold time. Task files read them at runtime.
- **If `risk_scorer` is selected:** Ask which tokens represent the answer classes (default: `["0", "1"]` for binary). The task file will automatically request logprobs when `risk_scorer` is in the scorer list.
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
