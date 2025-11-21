"""
Load and prepare data from the Model × Task experiment.

Experiment: cap_model_wordlen_generalization_2025-11-13
- 2 models (1B, 3B) × 2 conditions (fine-tuned, base) × 3 word lengths (4L, 5L, 6L)
- 12 total evaluations
"""

from inspect_ai.log import read_eval_log
import os
import pandas as pd

EXPERIMENT_DIR = "/scratch/gpfs/MSALGANIK/st0898/ck-experiments/cap_model_wordlen_generalization_2025-11-13"
RUNS = [
    "Llama-3.2-1B-Instruct_base",
    "Llama-3.2-3B-Instruct_base",
    "Llama-3.2-1B-Instruct_5L",
    "Llama-3.2-3B-Instruct_5L",
]


def load_eval_data() -> pd.DataFrame:
    """Load all evaluation results into a DataFrame."""
    data = []

    for run in RUNS:
        eval_dir = os.path.join(EXPERIMENT_DIR, run, "eval/logs")
        if not os.path.exists(eval_dir):
            continue

        for f in os.listdir(eval_dir):
            if not f.endswith('.eval'):
                continue

            log = read_eval_log(os.path.join(eval_dir, f))

            # Extract word length from sample input
            if log.samples:
                sample_input = log.samples[0].input
                if isinstance(sample_input, str):
                    input_len = len(sample_input.strip())
                    word_len = f"{input_len}L"
                else:
                    word_len = "unknown"
            else:
                word_len = "unknown"

            # Extract model and condition
            model = "1B" if "1B" in run else "3B"
            condition = "Fine-tuned" if run.endswith("_5L") else "Base"

            # Get match accuracy score
            for score in log.results.scores:
                if score.name == "match":
                    acc = score.metrics['accuracy'].value
                    stderr = score.metrics['stderr'].value
                    data.append({
                        'model': model,
                        'condition': condition,
                        'word_length': word_len,
                        'accuracy': acc,
                        'stderr': stderr,
                        'run': run,
                        'eval_file': f,
                    })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = load_eval_data()
    print(df.to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"\nWord lengths found: {df['word_length'].unique()}")
