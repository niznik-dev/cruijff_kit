# Question Catalog

Question types the renderer supports, split into **default** (use these) and **deprecated** (only when the user explicitly asks).

A good 6–8 question quiz uses the default mix:
- 3 multiple_choice
- 2 numerical_estimate
- 1 ranking (only if the metric isn't degenerate; otherwise replace with a 4th multiple_choice)

Don't repeat the same type back-to-back. Don't generate every type just to check a box.

---

## Default types

### multiple_choice

**Use when:** the experiment's results contain a clearly correct fact and the wrong answers can be made plausibly tempting (not just absurd distractors).

Three sub-uses worth knowing:
- **Concrete-lookup variant** — "what was X's accuracy at Y?"; tests whether the recipient read the intro carefully.
- **Hypothesis-check variant** — "the experiment hypothesised X. Did the data confirm?"; the four choices encode {confirmed, partly confirmed, contradicted, inconclusive}.
- **Synthesis variant** — "which of these statements is best supported by the results?"; harder, picks one true claim from three plausible-but-wrong ones.

**Spec:**
```json
{
  "type": "multiple_choice",
  "prompt": "...",
  "choices": ["A", "B", "C", "D"],
  "answer_index": 2,
  "explanation": "..."
}
```

**Good example (`length_sweep_nth14`):**

> The hypothesis stated: "k=50 and k=100 are 'hard but learnable' — at least one of N=500 or N=2000 reaches ≥0.95. If k=100 fails to saturate even at N=2000, that becomes the headline." Did the data confirm this?
>
> A) Yes — both k=50 and k=100 reached ≥0.95 already at N=500.
> B) Partial — k=50 saturated, but k=100 needed N=2000.
> C) No — k=100 failed to saturate even at N=2000 (the hypothesised "headline" finding).
> D) The data is inconclusive.
>
> **Answer: A.** k=50 hit 0.987 at N=500 and 1.000 at N=2000; k=100 hit 1.000 at N=500 and 1.000 at N=2000. Both saturated at N=500, so the worried "headline" outcome did not occur.

**Weak example:** "How many runs were in the experiment?" → trivia, not intuition.

---

### numerical_estimate

**Use when:** the experiment produced a continuous metric (accuracy, loss, training time) and the recipient should be able to predict it within a tolerance from neighbouring data.

**Default conventions:**
- **Use percentage form (0–100), not decimal (0.0–1.0).** Easier to enter, less likely to be off by an order of magnitude.
- **Set `min`, `max`, `step`.** The renderer applies them as HTML5 input attributes and rejects out-of-range submissions before grading. For accuracy: `min=0, max=100, step=0.1`.
- **Tolerance discipline:** pick a tolerance an experienced researcher would land in but a guess wouldn't. ±5 percentage points is reasonable for accuracy; tighten for stable metrics, loosen for noisy ones.

**Spec:**
```json
{
  "type": "numerical_estimate",
  "prompt": "...",
  "answer": 98.7,
  "tolerance": 5,
  "min": 0,
  "max": 100,
  "step": 0.1,
  "unit": "%",
  "explanation": "..."
}
```

**Good example:** "Predict the validation accuracy for the fine-tuned configuration at k=50, N=500. Answer as a percentage (0–100), within ±5 percentage points." Answer: 98.7. Explanation: "Per the per-k summary table, the k=50 / N=500 cell was 0.987 (i.e. 98.7%). It was the only N=500 cell that didn't reach exactly 100%."

**Weak example:** "What was the final training loss to 4 decimal places?" → memorisation, not intuition.

---

### ranking

**Use when:** there are at least 3 conditions on a single metric, the *order* (rather than absolute values) carries the lesson, and the conditions actually differ on that metric. If everything ties, the question is degenerate — replace it.

**Spec:**
```json
{
  "type": "ranking",
  "prompt": "Order from {hardest|easiest|first-saturating|...} to ...",
  "items": ["k=15", "k=30", "k=50", "k=100"],
  "answer_order": ["k=15", "k=100", "k=50", "k=30"],
  "explanation": "..."
}
```

The renderer shows draggable items with up/down arrows.

**Good example (`length_sweep_nth14`):** "At N=100, order k ∈ {15, 30, 50, 100} from *easiest to learn* (highest accuracy) to *hardest*. The answer is not what 'longer ⇒ harder' would predict." Answer: `k=15, k=100, k=50, k=30` (accuracies 0.996, 0.743, 0.640, 0.581). The non-monotonic pattern at low N is the lesson.

**Weak example:** Ranking 6 conditions when 5 of them are tied at 1.000. The question only differentiates 1 conditions vs. the other 5 — that's a 2-way distinction, not a ranking.

---

## Deprecated types (renderer supports them; default mix doesn't)

These are kept in the schema and renderer for the cases where the user explicitly asks ("I want a question about the SLURM bug" or "I have a way to collect free-text responses, enable that question type"). Do not include them in the default question portfolio.

### image_read — deprecated

**Why deprecated:** the recipient typically doesn't have access to the report or the figure context. A graph alone, without its surrounding caption and methodology, often misleads more than it teaches. If a finding is real, it can be summarised in a table cell or a sentence — and that's where the question should go.

**Rare valid use:** a graph that shows a *shape* (e.g. a learning curve, a ROC curve) which can't be conveyed in a single table cell, AND the recipient can be expected to read it correctly without context.

### equation_or_baseline — deprecated by default

**Why deprecated:** chance baselines belong in the intro as stated facts ("random-guessing accuracy is 1/k = 10%"). Asking the recipient to compute them is busywork that distracts from results.

**Rare valid use:** the baseline derivation is itself the point of the experiment — e.g. an experiment specifically designed to test whether a non-obvious baseline holds.

### anomaly_hunt — split

**For *operational* anomalies** (SLURM failures, config bugs, compute issues): **deprecated.** Out of scope. The quiz is about scientific results, not about how the experiment was run.

**For *results* anomalies** (a counter-example to the hypothesis, an unexpected outlier in the data): land it as a `multiple_choice` question. Don't use the dedicated `anomaly_hunt` type.

### free_text_intuition — deprecated

**Why deprecated:** there is no auto-grade and no feedback channel back to the experimenter. The recipient self-grading on reveal is honest but not useful — nobody collects the answers.

**Rare valid use:** the user explicitly says they have a way to collect responses (a Slack channel, a Google Form, an email thread) — *and* mentions it in the intro so the recipient knows where to send their answer.

---

## Question ordering

Order questions roughly:

1. **Anchor** — concrete lookup, answerable from the intro alone. Builds confidence.
2. **Predict** — numerical estimate, derivable from the data and reasoning.
3. **Twist** — the most surprising / non-monotonic finding (the lesson the experiment teaches that intuition wouldn't predict).
4. **Hypothesis check** — did the data confirm what the experiment set out to test?
5. **Predict (harder)** — second numerical, or a comparison.
6. **Synthesis** — which conclusion is best supported?

Don't lead with anything that gives away a later answer.
