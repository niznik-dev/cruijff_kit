# Logging - summarize-experiment

**See [shared/logging_spec.md](../shared/logging_spec.md) for format specification.**

---

## Log File Location

```
{experiment_dir}/logs/summarize-experiment.log
```

---

## Action Types

| Action Type | Purpose |
|-------------|---------|
| `DISCOVER_EXPERIMENT` | Locate and validate experiment directory |
| `PARSE_STATUS` | Read run status from experiment_summary.yaml |
| `EXTRACT_LOSS` | Extract final loss from SLURM stdout |
| `EXTRACT_ACCURACY` | Extract accuracy from .eval files using parse_eval_log.py |
| `GENERATE_SUMMARY` | Create summary.md file |
| `COMPLETE` | Summarization finished |

---

## When to Log

Create log entries at these points:

1. **Start:** Log experiment discovery and validation
2. **Status parsing:** Log what runs were found and their states
3. **Per-run extraction:** Log each loss and accuracy extraction (success or failure)
4. **Summary generation:** Log the output file creation
5. **End:** Log completion with summary statistics

---

## Example Log Entries

```
[2025-12-05 14:00:00] DISCOVER_EXPERIMENT
Details: Found experiment at /path/to/experiment
Result: 4 runs identified (3 fine-tuned, 1 base model)

[2025-12-05 14:00:01] PARSE_STATUS
Details: Reading experiment_summary.yaml status tables
Result: 3 fine-tuned COMPLETED, 1 base N/A for training, 4 evaluations COMPLETED

[2025-12-05 14:00:02] EXTRACT_LOSS: rank4_lr1e-5
Details: Parsing slurm-12345678.out
Result: Final loss = 0.234 (epoch 2, step 500)

[2025-12-05 14:00:03] EXTRACT_LOSS: rank8_lr1e-5
Details: Parsing slurm-12345679.out
Result: Final loss = 0.198 (epoch 2, step 500)

[2025-12-05 14:00:04] EXTRACT_ACCURACY: rank4_lr1e-5/capitalization/epoch0
Command: python tools/inspect/parse_eval_log.py {path}
Result: Accuracy = 0.85, samples = 100, scorer = exact_match

[2025-12-05 14:00:05] EXTRACT_ACCURACY: rank8_lr1e-5/capitalization/epoch0
Command: python tools/inspect/parse_eval_log.py {path}
Result: Accuracy = 0.91, samples = 100, scorer = exact_match

[2025-12-05 14:00:06] EXTRACT_ACCURACY: base_model/capitalization
Command: python tools/inspect/parse_eval_log.py {path}
Result: Accuracy = 0.45, samples = 100, scorer = exact_match

[2025-12-05 14:00:10] GENERATE_SUMMARY
Details: Creating summary.md
Result: Written to /path/to/experiment/summary.md (4 runs, 3 eval results)

[2025-12-05 14:00:10] COMPLETE
Summary: 2 loss values extracted, 3 accuracy values extracted
Duration: 10 seconds
```

---

## Error Logging

When extraction fails, log with sufficient detail for debugging:

```
[2025-12-05 14:00:03] EXTRACT_LOSS: rank16_lr1e-5
Details: No SLURM output file found
Result: N/A (run may have failed or not started)

[2025-12-05 14:00:05] EXTRACT_ACCURACY: rank16_lr1e-5/capitalization/epoch0
Command: python tools/inspect/parse_eval_log.py {path}
Result: ERROR - File not found: {path}
Explanation: Evaluation may not have completed
```
