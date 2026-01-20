"""
Load and prepare data from Experiment: cap_wordlen_2025-12-05

Experiment: Model × Word Length Generalization
- 2 models (1B, 3B) × 3 word lengths (4L, 5L, 6L) × 2 epochs (0, 1)
- 12 total evaluations

This experiment tests how model size affects generalization from one word length
to other word lengths in capitalization tasks.
"""

from inspect_ai.log import read_eval_log
from inspect_ai.analysis import log_viewer, prepare
import os
import pandas as pd

EXPERIMENT_DIR = "/scratch/gpfs/MSALGANIK/st0898/ck-experiments/cap_wordlen_2025-12-05"
# Only 2 training runs (both trained on 5L), but each evaluated on 4L, 5L, 6L at epoch 0 and 1
RUNS = [
    "Llama-3.2-1B-Instruct_5L",
    "Llama-3.2-3B-Instruct_5L",
]

# Use bundled viewer with relative URLs
# Requires serving the HTML via HTTP server (see CLICKABLE_LINKS_SETUP.md)
# To serve: cd viz_examples/html && python -m http.server 8000
# Then open: http://localhost:8000/bar_plot.html
USE_LOCALHOST = False

if USE_LOCALHOST:
    # Map log directories to localhost inspect view server
    LOG_DIRS = {
        os.path.join(EXPERIMENT_DIR, "Llama-3.2-1B-Instruct_5L/eval/logs"): "http://localhost:7575",
        os.path.join(EXPERIMENT_DIR, "Llama-3.2-3B-Instruct_5L/eval/logs"): "http://localhost:7575",
    }
else:
    # Map log directories to bundled viewer (for deployed visualizations)
    # These are relative URLs that will work when served via HTTP
    LOG_DIRS = {
        os.path.join(EXPERIMENT_DIR, "Llama-3.2-1B-Instruct_5L/eval/logs"): "http://localhost:8000/cap_wordlen_logs_viewer/",
        os.path.join(EXPERIMENT_DIR, "Llama-3.2-3B-Instruct_5L/eval/logs"): "http://localhost:8000/cap_wordlen_logs_viewer/",
    }


def load_eval_data() -> pd.DataFrame:
    """Load all evaluation results into a DataFrame."""
    data = []

    for run in RUNS:
        eval_dir = os.path.join(EXPERIMENT_DIR, run, "eval/logs")
        if not os.path.exists(eval_dir):
            print(f"Warning: {eval_dir} does not exist")
            continue

        for f in os.listdir(eval_dir):
            if not f.endswith('.eval'):
                continue

            log_path = os.path.join(eval_dir, f)
            try:
                log = read_eval_log(log_path)

                # Skip if no results
                if log.results is None:
                    continue
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

            # Extract word length from samples (most reliable)
            word_len = "unknown"
            if log.samples and len(log.samples) > 0:
                sample_input = log.samples[0].input
                if isinstance(sample_input, str):
                    input_len = len(sample_input.strip())
                    word_len = f"{input_len}L"

            # Extract model size from run name
            model = "1B" if "1B" in run else "3B"

            # Extract epoch by checking the model_path in the log
            epoch = "unknown"
            if log.eval and hasattr(log.eval, 'model_args') and log.eval.model_args:
                model_path = log.eval.model_args.get('model_path', '')
                if 'epoch_0' in model_path:
                    epoch = "epoch_0"
                elif 'epoch_1' in model_path:
                    epoch = "epoch_1"

            # Fallback: Check task args
            if epoch == "unknown" and log.eval and hasattr(log.eval, 'task_args') and log.eval.task_args:
                if 'epoch' in log.eval.task_args:
                    epoch = f"epoch_{log.eval.task_args['epoch']}"

            # Get match accuracy score
            for score in log.results.scores:
                if score.name == "match":
                    acc = score.metrics['accuracy'].value
                    stderr = score.metrics['stderr'].value
                    data.append({
                        'model': model,
                        'word_length': word_len,
                        'epoch': epoch,
                        'accuracy': acc,
                        'stderr': stderr,
                        'run': run,
                        'eval_file': f,
                        'log': log_path,  # Required by prepare() - must be named 'log'
                        'timestamp': f.split('_cap-task')[0],  # Extract full timestamp
                    })

    df = pd.DataFrame(data)

    # Keep only the most recent evaluation for each (model, word_length, epoch) combination
    # This filters out failed/outdated runs
    # Sort by timestamp (desc) then accuracy (desc) to prefer recent non-zero results
    if len(df) > 0:
        df = df.sort_values(['timestamp', 'accuracy'], ascending=[False, False])
        df = df.drop_duplicates(subset=['model', 'word_length', 'epoch'], keep='first')
        df = df.drop(columns=['timestamp'])  # Remove helper column
        df = df.sort_values(['model', 'epoch', 'word_length'])  # Sort for display

        # Add log_viewer URLs using inspect-ai's prepare() function
        # This creates clickable links to the bundled log viewer
        df = prepare(df, log_viewer('evals', LOG_DIRS))

    return df


if __name__ == "__main__":
    df = load_eval_data()
    print(df.to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"\nModels: {df['model'].unique()}")
    print(f"Word lengths: {df['word_length'].unique()}")
    print(f"Epochs: {df['epoch'].unique()}")
    print(f"\nExample log_viewer URL:")
    print(df['log_viewer'].iloc[0])
