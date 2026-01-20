# Clickable Log Viewer Links in Inspect-Viz Visualizations

## Summary

The bar plot visualization now includes clickable links in tooltips that open the Inspect AI log viewer to show detailed evaluation results for each data point.

## How It Works

### Current Implementation: Bundled Viewer via HTTP Server ✅

The data loader (`load_cap_wordlen_data.py`) uses URLs pointing to a bundled log viewer:

```python
from inspect_ai.analysis import log_viewer, prepare

# Map log directories to bundled viewer via HTTP server
LOG_DIRS = {
    "/path/to/logs/": "http://localhost:8000/cap_wordlen_logs_viewer/",
}

# Apply to dataframe
df = prepare(df, log_viewer('evals', LOG_DIRS))
```

This creates URLs like:
```
http://localhost:8000/cap_wordlen_logs_viewer/#/logs/2025-12-05T14-42-32-05-00_cap-task_TE83zdWqnXbU2mzj.eval
```

**Requirements:**
- Serve the HTML directory via HTTP: `python -m http.server 8000`
- Open visualization at: `http://localhost:8000/bar_plot.html`
- URLs start with `http://` so tooltips recognize them as clickable links

### Bundled Log Viewer

The bundled viewer was created using:
```bash
inspect view bundle --log-dir <logs> --output-dir cap_wordlen_logs_viewer
```

This creates a directory with:
- `index.html` - The log viewer interface
- `logs/` - All evaluation log files (11 .eval files)
- `assets/` - CSS/JS for the viewer

### Alternative: inspect view Server

To use `inspect view` server instead, set `USE_LOCALHOST = True` in `load_cap_wordlen_data.py` and start the server with `inspect view`

### 3. Tooltip Integration

The visualization maps the `log_viewer` column to the tooltip channels:

```python
plot(
    bar_y(
        data,
        channels={
            'Log Viewer': 'log_viewer',  # Maps to clickable link
            ...
        }
    )
)
```

## File Structure

```
viz_examples/
├── html/
│   ├── bar_plot.html                 # Visualization with clickable links
│   └── cap_wordlen_logs_viewer/      # Bundled log viewer
│       ├── index.html                # Viewer interface
│       ├── logs/                     # 11 eval log files
│       └── assets/                   # CSS/JS
└── scripts/
    ├── load_cap_wordlen_data.py      # Data loader with log_viewer URLs
    └── 01_bar_plot_updated.py        # Bar plot visualization script
```

## How to Use

### Opening the Visualization

**Important:** The visualization must be served via HTTP (not opened as `file://`) for links to work.

1. **Start a local HTTP server**:
   ```bash
   cd viz_examples/html
   python -m http.server 8000
   ```

2. **Open in browser**:
   ```
   http://localhost:8000/bar_plot.html
   ```

3. **Use the tooltips**:
   - Hover over any bar to see the tooltip
   - Click on the "log_viewer" link in the tooltip
   - The Inspect AI log viewer opens in a new tab showing detailed results

**Note:** Opening `bar_plot.html` directly as a file (file://) won't work because the bundled viewer URLs are configured for http://localhost:8000/

### Adding Links to New Visualizations

To add clickable log links to a new visualization:

1. **Use the data loader** that already has log_viewer URLs:
   ```python
   from load_cap_wordlen_data import load_eval_data
   df = load_eval_data()  # Already has 'log_viewer' column
   ```

2. **Map the column in your plot**:
   ```python
   plot(
       bar_y(
           data,
           channels={'Log Viewer': 'log_viewer'},  # Add this
           tip=True,  # Enable tooltips
       )
   )
   ```

3. **Save to the same directory** as the bundled viewer:
   ```python
   write_html('../html/my_plot.html', p)
   ```

### Creating a New Bundled Viewer

For a different experiment:

```bash
inspect view bundle \
  --log-dir /path/to/experiment/run1/eval/logs \
  --log-dir /path/to/experiment/run2/eval/logs \
  --output-dir viz_examples/html/experiment_logs_viewer
```

Then update your data loader to map to the new viewer location.

## Benefits

- ✅ **Self-contained** - Viewer and logs bundled together
- ✅ **Shareable** - Can deploy to any static web host
- ✅ **Integrated** - Links appear automatically in tooltips
- ✅ **Detailed inspection** - Click through to see full evaluation transcripts
- ✅ **No `inspect view` needed** - Uses bundled viewer instead of running server

## Alternative: Local Inspect Server

For development, you can also run a local server:

```bash
inspect view  # Starts server at http://localhost:7575
```

Then use `http://localhost:7575/log?file=...` URLs instead of bundled viewer URLs. However, this requires the server to be running, so bundled viewer is better for sharing.

## References

- [Inspect-Viz Components: Links](https://meridianlabs-ai.github.io/inspect_viz/components-links.html)
- [Inspect AI Log Viewer](https://inspect.ai-safety-institute.org.uk/log-viewer.html)
