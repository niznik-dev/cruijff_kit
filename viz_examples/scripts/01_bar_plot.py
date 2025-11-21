"""
Bar plot comparing eval scores across models and tasks with confidence intervals.

Visualization #1 from Issue #214.
"""

import pandas as pd
from inspect_viz import Data
from inspect_viz.plot import plot, write_html, write_png
from inspect_viz.mark import bar_y, grid_y, title

from load_model_x_task_data import load_eval_data


def prepare_data() -> pd.DataFrame:
    """Load and prepare data for plotting."""
    df = load_eval_data()

    # Create combined label for x-axis grouping
    df = df.copy()
    df['model_condition'] = df['model'] + ' ' + df['condition']

    # Rename word_length values for display
    df['Word Length'] = df['word_length'].map({
        '4L': '4-letters',
        '5L': '5-letters',
        '6L': '6-letters'
    })

    # Calculate CI bounds (using 1.96 * stderr for 95% CI)
    df['ci_low'] = df['accuracy'] - 1.96 * df['stderr']
    df['ci_high'] = df['accuracy'] + 1.96 * df['stderr']

    return df.sort_values(['model', 'condition', 'word_length'])


def create_bar_plot():
    """Create grouped bar plot comparing accuracy by model/condition and word length."""
    df = prepare_data()
    data = Data(df)

    # Create the plot with side-by-side bars using fx faceting
    # - fx='word_length' creates separate columns for each word length (side-by-side)
    # - grid_y() adds horizontal gridlines
    # - tip=True enables tooltips on hover
    p = plot(
        grid_y(),
        bar_y(
            data,
            x='model_condition',
            y='accuracy',
            fill='Word Length',
            fx='Word Length',
            tip=True,
        ),
        title=title('Capitalization Accuracy by Model and Word Length', margin_top=40),
        x_label='Model / Condition',
        y_label='Accuracy',
        fx_label=None,  # Hide "Word Length" label above facets
        width=900,
        height=500,
        y_domain=[0, 1],  # Y-axis from 0 to 1
    )

    return p


if __name__ == "__main__":
    p = create_bar_plot()
    write_html('../html/bar_plot.html', p)
    write_png('../png/bar_plot.png', p)
    print("Saved to bar_plot.html and bar_plot.png")

    # Also print data summary
    df = prepare_data()
    print("\nData summary:")
    print(df[['model_condition', 'word_length', 'accuracy']].to_string(index=False))
