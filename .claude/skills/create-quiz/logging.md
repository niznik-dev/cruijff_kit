# Logging

See [shared/logging_spec.md](../shared/logging_spec.md) for the format and general guidelines.

## Log file location

```
{experiment_dir}/logs/create-quiz.log
```

If `logs/` doesn't exist, create it. Append to the log on re-runs (don't truncate) so the audit trail across regeneration is preserved.

## Action types

| Type | When | Required fields |
|------|------|-----------------|
| `LOCATE_EXPERIMENT` | One per experiment directory found | Path, presence of report.md and YAML |
| `PARSE_INPUTS` | After reading all inputs for one experiment | Counts: runs, report rows, PNGs |
| `PLAN_QUESTIONS` | After deciding the question portfolio | Total count + breakdown by type |
| `WRITE_SPEC` | After writing quiz.json | File path, byte size |
| `RENDER_HTML` | After running render_quiz | File path, byte size, image count |
| `VALIDATE` | After acceptance checks pass | Which checks ran |
| `WARN` | Non-fatal anomalies (missing optional input) | What was missing, what was substituted |
| `ERROR` | Fatal errors | What failed, what the user should do |

## Example

```
[2026-05-03 16:42:08] LOCATE_EXPERIMENT: length_sweep_nth14
Details: /scratch/gpfs/MSALGANIK/mjs3/ck-experiments/length_sweep_nth14
Result: experiment_summary.yaml present, analysis/report.md present, summary.md absent

[2026-05-03 16:42:09] PARSE_INPUTS: length_sweep_nth14
Details: 22 runs parsed, 22 result rows in report.md, 1 PNG (headline_accuracy_vs_N_by_k.png)

[2026-05-03 16:42:14] PLAN_QUESTIONS: 8 questions
Details: multiple_choice=2, ranking=1, numerical_estimate=1, image_read=1, equation_or_baseline=1, anomaly_hunt=1, free_text_intuition=1

[2026-05-03 16:42:14] WRITE_SPEC: quiz.json
Details: /scratch/gpfs/.../length_sweep_nth14/quiz/quiz.json (12,481 bytes)

[2026-05-03 16:42:15] RENDER_HTML: quiz.html
Command: python -m cruijff_kit.tools.quiz.render_quiz --spec ... --out ...
Result: 184,322 bytes; 1 image embedded

[2026-05-03 16:42:15] VALIDATE: all checks passed
Details: ≥6 questions ✓, every q has explanation ✓, no http refs except MathJax ✓, every PNG path resolved ✓
```
