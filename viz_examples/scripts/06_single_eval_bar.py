"""
Bar plot comparing model scores on a single evaluation task.

Visualization #6 from Issue #214.
This shows accuracy for a single word length (5L - the training length).
"""

import pandas as pd
from inspect_viz import Data
from inspect_viz.plot import plot, write_html, write_png
from inspect_viz.mark import rule_x, dot, grid_x, title, error_bar_x

from load_model_x_task_data import load_eval_data


def prepare_data(word_length: str = '5L') -> pd.DataFrame:
    """Load and prepare data for a single word length."""
    df = load_eval_data()

    # Filter to single word length
    df = df[df['word_length'] == word_length].copy()

    # Create model/condition label
    df['model_condition'] = df['model'] + ' ' + df['condition']

    # Calculate 95% CI bounds
    df['ci_low'] = (df['accuracy'] - 1.96 * df['stderr']).clip(lower=0)
    df['ci_high'] = (df['accuracy'] + 1.96 * df['stderr']).clip(upper=1)

    return df.sort_values(['model', 'condition'])


def create_single_eval_bar(word_length: str = '5L'):
    """Create lollipop chart comparing models on a single evaluation."""
    df = prepare_data(word_length)
    data = Data(df)

    word_len_display = {'4L': '4-letter', '5L': '5-letter', '6L': '6-letter'}[word_length]

    # Lollipop chart with CI: thin bar for CI range, line+dot for accuracy
    from inspect_viz.mark import rule_y

    p = plot(
        grid_x(),
        # CI bar (thin, lighter)
        rule_y(
            data,
            y='model_condition',
            x1='ci_low',
            x2='ci_high',
            stroke='model_condition',
            stroke_opacity=0.4,
            stroke_width=12,
        ),
        # Accuracy line (thin)
        rule_y(
            data,
            y='model_condition',
            x1=0,
            x2='accuracy',
            stroke='model_condition',
            stroke_width=3,
        ),
        # Dot at accuracy value
        dot(
            data,
            y='model_condition',
            x='accuracy',
            fill='model_condition',
            r=6,
            tip=True,
        ),
        title=title(f'Capitalization Accuracy on {word_len_display} Words', margin_top=30),
        x_label='Accuracy',
        y_label='Model / Condition',
        width=550,
        height=350,
        x_domain=[0, 1],
        margin_left=100,
    )

    return p


if __name__ == "__main__":
    # Generate for the training word length (5L)
    p = create_single_eval_bar('5L')
    write_html('../html/single_eval_bar.html', p)
    write_png('../png/single_eval_bar.png', p)
    print("Saved to single_eval_bar.html and single_eval_bar.png")

    # Print data
    df = prepare_data('5L')
    print("\n5-letter word accuracy:")
    print(df[['model_condition', 'accuracy']].to_string(index=False))
