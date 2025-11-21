"""
Heatmap visualizing scores across model × task grid.

Visualization #5 from Issue #214.
"""

import pandas as pd
from inspect_viz import Data
from inspect_viz.plot import plot, write_html, write_png
from inspect_viz.mark import cell, text, title

from load_model_x_task_data import load_eval_data


def prepare_data() -> pd.DataFrame:
    """Load and prepare data for heatmap."""
    df = load_eval_data()

    # Create row label (model + condition)
    df = df.copy()
    df['model_condition'] = df['model'] + ' ' + df['condition']

    # Rename word_length for display
    df['Word Length'] = df['word_length'].map({
        '4L': '4-letters',
        '5L': '5-letters',
        '6L': '6-letters'
    })

    # Convert accuracy to percentage for display
    df['accuracy_pct'] = (df['accuracy'] * 100).round(0).astype(int)
    df['label'] = df['accuracy_pct'].astype(str) + '%'

    # Text color: black for low accuracy (light background), white for high
    df['text_color'] = df['accuracy'].apply(lambda x: 'black' if x < 0.15 else 'white')

    return df


def create_heatmap():
    """Create heatmap of accuracy by model/condition and word length."""
    df = prepare_data()
    data = Data(df)

    # Create heatmap using cell mark with text labels
    # Swapped axes: model_condition on x, Word Length on y
    p = plot(
        cell(
            data,
            x='model_condition',
            y='Word Length',
            fill='accuracy',
            tip=True,
        ),
        text(
            data,
            x='model_condition',
            y='Word Length',
            text='label',
            fill='text_color',
        ),
        title=title('Accuracy Heatmap: Model × Word Length', margin_top=10),
        x_label='Model / Condition',
        y_label='Word Length',
        width=600,
        height=320,
        margin_left=80,  # More space for y-axis labels
        color_scheme='blues',  # Continuous color scale
    )

    return p


if __name__ == "__main__":
    p = create_heatmap()
    write_html('../html/heatmap.html', p)
    write_png('../png/heatmap.png', p)
    print("Saved to heatmap.html and heatmap.png")

    # Print pivot table
    df = prepare_data()
    pivot = df.pivot(index='model_condition', columns='Word Length', values='accuracy_pct')
    print("\nAccuracy (%)")
    print(pivot.to_string())
