# Test Fixtures

Reference artifacts at each pipeline stage for testing skill contracts in isolation.

## Path Placeholder Convention

All fixture files use placeholder tokens instead of real paths:

| Token | Replaced with | Represents |
|-------|--------------|------------|
| `__SCRATCH__` | `tmp_path / "scratch"` | User's scratch directory (e.g., `/scratch/gpfs/GROUP/user`) |
| `__REPO__` | `tmp_path / "repo"` | The cruijff_kit repository root |

The shared helper in `conftest.py` provides `resolve_placeholders()` and the `resolved_fixture` pytest fixture for automatic replacement at test time.

## Directory Layout

```
fixtures/
├── conftest.py                     # Shared path-resolution helpers
├── design/
│   └── experiment_summary.yaml     # Stage 1 output → Stage 2 input
├── scaffold/
│   ├── torchtune/
│   │   ├── rank4/
│   │   │   └── setup_finetune.yaml # Per-run scaffold-torchtune input
│   │   └── rank8/
│   │       └── setup_finetune.yaml
│   └── inspect/
│       ├── rank4/
│       │   └── eval/
│       │       └── eval_config.yaml # Per-run scaffold-inspect input
│       └── rank8/
│           └── eval/
│               └── eval_config.yaml
└── summarize/
    ├── slurm_training_output.txt    # Synthetic SLURM stdout with loss lines
    └── eval_result.json             # Synthetic parse_eval_log output
```

## Fixture Sources

| Fixture | Based on | Purpose |
|---------|----------|---------|
| `design/experiment_summary.yaml` | Template + workflow_test.yaml | Reference design; validates schema |
| `scaffold/torchtune/*/setup_finetune.yaml` | test_setup_finetune_main.py pattern | Input to setup_finetune.py |
| `scaffold/inspect/*/eval/eval_config.yaml` | test_setup_inspect.py pattern | Input to setup_inspect.py |
| `summarize/slurm_training_output.txt` | Synthetic (uses `__SCRATCH__` placeholders) | Loss regex extraction testing |
| `summarize/eval_result.json` | parse_eval_log output format | Eval parsing without .eval files |

## Test Files

| Test | Location | What it validates |
|------|----------|-------------------|
| `test_experiment_summary_schema.py` | `tests/unit/` | Structure of experiment_summary.yaml |
| `test_summarize_fixtures.py` | `tests/unit/` | Loss extraction module + eval JSON parsing |

## Manual Usage

To use fixtures for manual skill testing, copy them to a working directory and replace the placeholders:

```bash
sed 's|__SCRATCH__|/scratch/gpfs/GROUP/user|g; s|__REPO__|/path/to/cruijff_kit|g' \
  tests/fixtures/design/experiment_summary.yaml > /tmp/experiment_summary.yaml
```
