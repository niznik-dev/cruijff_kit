"""
Bar plot comparing eval scores across models and word lengths with confidence intervals.

Visualization #1 from Issue #214.

Uses cap_wordlen_2025-12-05 experiment:
- 2 models (1B, 3B) × 3 word lengths (4L, 5L, 6L) × 2 epochs (0, 1)
- 12 total evaluations
"""

import pandas as pd
from inspect_viz import Data
from inspect_viz.plot import plot, write_html, write_png, legend
from inspect_viz.mark import bar_y, grid_y, title, rule_x
from inspect_viz.transform import ci_bounds

from load_cap_wordlen_data import load_eval_data


def prepare_data() -> pd.DataFrame:
    """Load and prepare data for plotting."""
    df = load_eval_data()

    # Create combined label for coloring (model + epoch)
    df = df.copy()
    df['Model/Epoch'] = df['model'] + ' ' + df['epoch'].str.replace('epoch_', 'epoch ')

    # Create full model name for display
    df['Model'] = df['model'].map({
        '1B': 'Llama-3.2-1B-Instruct',
        '3B': 'Llama-3.2-3B-Instruct'
    })

    # Rename word_length values for display
    df['Task'] = df['word_length'].map({
        '4L': '4-letters',
        '5L': '5-letters',
        '6L': '6-letters'
    })

    # Calculate 95% CI bounds (using 1.96 * stderr)
    df['ci_low'] = df['accuracy'] - 1.96 * df['stderr']
    df['ci_high'] = df['accuracy'] + 1.96 * df['stderr']

    # Format accuracy as percentage for display
    df['Accuracy (%)'] = (df['accuracy'] * 100).round(2)

    # Extract epoch number for display
    df['Epoch'] = df['epoch'].str.replace('epoch_', '')

    # Keep log_viewer column - it's already been created by load_eval_data()
    # inspect-viz will automatically detect and render it as a clickable link
    cols_to_keep = ['Model/Epoch', 'Model', 'Task', 'Epoch', 'Accuracy (%)', 'log_viewer',
                    'accuracy', 'ci_low', 'ci_high', 'stderr']
    df = df[cols_to_keep]

    return df.sort_values(['Model/Epoch'])


def create_bar_plot():
    """Create grouped bar plot comparing accuracy by model/epoch and word length."""
    df = prepare_data()
    data = Data(df)

    # Calculate confidence interval bounds (95% CI)
    ci_lower, ci_upper = ci_bounds(
        score='accuracy',
        level=0.95,
        stderr='stderr'
    )

    # Create the plot with side-by-side bars using fx faceting
    # - fx='Task' creates separate columns for each word length
    # - fill='Model/Epoch' colors bars by model/epoch combination
    # - grid_y() adds horizontal gridlines
    # - rule_x creates vertical error bars using ci bounds
    # - legend shows color key for Model/Epoch
    # - Remove x-axis labels (x_ticks=[]), rely on legend instead
    p = plot(
        grid_y(),
        bar_y(
            data,
            x='Model/Epoch',
            y='accuracy',
            fill='Model/Epoch',  # Color by model/epoch instead of word length
            fx='Task',
            tip=True,
            channels={
                'Model': 'Model',
                'Epoch': 'Epoch',
                'Accuracy (%)': 'Accuracy (%)',
                'Task': 'Task',
                'log_viewer': 'log_viewer',  # Explicitly include for tooltip
            }
        ),
        rule_x(
            data,
            x='Model/Epoch',
            y1=ci_lower,
            y2=ci_upper,
            fx='Task',
            stroke='black',
            marker='tick-x',
        ),
        legend=legend('color', frame_anchor='bottom'),
        title=title('Capitalization Accuracy by Model and Word Length', margin_top=40),
        x_label=None,  # Remove x-axis label
        x_ticks=[],  # Remove x-axis tick labels
        y_label='Accuracy',
        fx_label=None,  # Hide "Word Length" label above facets
        color_label='Model / Epoch',  # Label for the color legend
        width=900,
        height=500,
        y_domain=[0, 1],  # Y-axis from 0 to 1
    )

    return p


if __name__ == "__main__":
    p = create_bar_plot()

    # Save outputs
    write_html('../html/bar_plot.html', p)
    write_png('../png/bar_plot.png', p)

    print("✓ Saved to bar_plot.html and bar_plot.png")

    # Also print data summary
    df = prepare_data()
    print("\nData summary:")
    print(df[['Model/Epoch', 'Task', 'Accuracy (%)']].to_string(index=False))
