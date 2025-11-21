"""
Radar plot comparing multiple models across headline metrics from different evals.

Visualization #7 from Issue #214.
Shows model performance across 4L, 5L, and 6L word lengths.
"""

import pandas as pd
import numpy as np
from inspect_viz import Data
from inspect_viz._core.selection import Selection
from inspect_viz.mark import circle, line, text, title
from inspect_viz.plot import plot, write_html, write_png
from inspect_viz.plot._legend import legend
from inspect_viz.view.beta import LabelStyles
from inspect_viz.view.beta._scores_radar import (
    axes_coordinates,
    grid_circles_coordinates,
    labels_coordinates,
)

from load_model_x_task_data import load_eval_data


def prepare_radar_data() -> pd.DataFrame:
    """Load and prepare data for radar plot with x,y coordinates."""
    df = load_eval_data()

    # Create model/condition label
    df = df.copy()
    df['model_condition'] = df['model'] + ' ' + df['condition']

    # Rename word lengths for display
    df['task'] = df['word_length'].map({
        '4L': '4-letter words',
        '5L': '5-letter words',
        '6L': '6-letter words'
    })

    # Get unique tasks and models
    tasks = ['4-letter words', '5-letter words', '6-letter words']
    models = df['model_condition'].unique()

    # Calculate radar coordinates for each point
    # Match axes_coordinates: 0°, 120°, -120° (right, upper-left, lower-left)
    num_axes = len(tasks)
    angles = [0, 2*np.pi/3, -2*np.pi/3]  # 0°, 120°, -120° in radians

    rows = []
    for model in models:
        model_df = df[df['model_condition'] == model]
        for i, task in enumerate(tasks):
            task_row = model_df[model_df['task'] == task]
            if len(task_row) > 0:
                value = task_row['accuracy'].values[0]
                angle = angles[i]
                # Convert polar to cartesian
                x = value * np.cos(angle)
                y = value * np.sin(angle)
                rows.append({
                    'model_condition': model,
                    'task': task,
                    'accuracy': value,
                    'x': x,
                    'y': y,
                })

    # Close the polygon by adding first point at end for each model
    for model in models:
        model_rows = [r for r in rows if r['model_condition'] == model]
        if model_rows:
            first = model_rows[0].copy()
            rows.append(first)

    return pd.DataFrame(rows)


def create_radar_plot():
    """Create radar plot comparing models across word lengths."""
    df = prepare_radar_data()
    data = Data(df)

    tasks = ['4-letter words', '5-letter words', '6-letter words']

    # Get coordinates for axes, grid circles, and labels
    axes = axes_coordinates(num_axes=len(tasks))
    grid_circles = grid_circles_coordinates()
    labels = labels_coordinates(labels=tasks)

    model_selection = Selection.single()

    channels = {
        "Model": "model_condition",
        "Task": "task",
        "Accuracy": "accuracy",
    }

    elements = [
        # Grid circles (background)
        *[line(x=gc["x"], y=gc["y"], stroke="#e0e0e0")
          for gc in grid_circles],
        # Axes lines
        line(x=axes["x"], y=axes["y"], stroke="#ddd"),
        # Model lines (highlighted on selection)
        line(data, x="x", y="y", stroke="model_condition",
             filter_by=model_selection, tip=True, channels=channels),
        # Model lines (faded background)
        line(data, x="x", y="y", stroke="model_condition",
             stroke_opacity=0.4, tip=False),
        # Points at each vertex
        circle(data, x="x", y="y", r=6, fill="model_condition",
               stroke="white", filter_by=model_selection, tip=False),
        # Axis labels
        *[text(x=label["x"], y=label["y"], text=label["label"],
               frame_anchor=label["frame_anchor"],
               styles=LabelStyles(line_width=8))
          for label in labels],
    ]

    p = plot(
        elements,
        title=title('Model Performance Across Word Lengths', margin_top=40),
        margin=70,
        x_axis=False,
        y_axis=False,
        width=500,
        height=500,
        legend=legend("color", target=model_selection),
    )

    return p


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_dir = os.path.join(script_dir, '..', 'html')
    png_dir = os.path.join(script_dir, '..', 'png')
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    p = create_radar_plot()
    write_html(os.path.join(html_dir, 'radar_plot.html'), p)
    write_png(os.path.join(png_dir, 'radar_plot.png'), p)
    print("Saved to radar_plot.html and radar_plot.png")

    # Print data summary
    df = prepare_radar_data()
    pivot = df.drop_duplicates(['model_condition', 'task']).pivot(
        index='model_condition', columns='task', values='accuracy'
    )
    print("\nAccuracy by model and task:")
    print(pivot.to_string())
