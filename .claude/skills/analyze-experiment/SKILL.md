# Analyze Experiment (PLANNED)

**STATUS: This skill is planned for future implementation. Documentation below describes the intended functionality.**

You will help users interpret and compare experimental results by analyzing training metrics, evaluation scores, and generating comparison visualizations.

## Your Task (Future)

Read experiment results from:
- Fine-tuning SLURM logs (training metrics)
- inspect-ai evaluation logs (.eval files)
- experiment_summary.md (experiment design)

Then generate:
- Comparison tables across runs
- Performance visualization plots
- Statistical summaries
- Formatted reports

## Workflow (Planned)

1. **Locate experiment** - Find the experiment directory
2. **Read experiment_summary.md** - Understand experimental design
3. **Extract training metrics** - Parse SLURM logs for:
   - Training loss curves
   - Validation loss (if available)
   - Training time
   - GPU utilization
4. **Extract evaluation metrics** - Parse inspect-ai logs for:
   - Accuracy scores
   - Per-sample results
   - Task-specific metrics
5. **Generate comparisons** - Create:
   - Tables comparing all runs
   - Line plots showing metric trends
   - Bar charts for final performance
   - Heatmaps for parameter sweeps
6. **Create report** - Write markdown summary with:
   - Best performing configurations
   - Key findings
   - Visualizations embedded
   - Statistical significance (if applicable)
7. **Create log** - Document analysis process in `analyze-experiment.log`
8. **Present findings** - Show user summary and offer to dig deeper

## Data Sources (Planned)

### Training Metrics

**Source:** SLURM logs (`rank*/slurm-*.out`)

**Extract:**
- Loss per step/epoch
- Learning rate schedule
- Training time
- GPU memory usage
- Convergence indicators

**Parsing approach:**
- Search for torchtune output patterns
- Extract numeric values with regex
- Build time series data structures

### Evaluation Metrics

**Source:** inspect-ai logs (`rank*/eval/logs/*.eval`)

**Extract:**
- Overall accuracy
- Per-sample predictions and scores
- Task-specific metrics
- Evaluation time

**Parsing approach:**
- Use inspect-ai's programmatic API
- Load .eval files as JSON
- Aggregate across runs

### Experimental Design

**Source:** experiment_summary.md

**Extract:**
- Variables being tested (LoRA rank, learning rate, etc.)
- Run naming conventions
- Scientific question being answered
- Expected outcomes

## Outputs (Planned)

### 1. Comparison Tables

**Fine-tuning comparison:**
```markdown
| Run Name | LoRA Rank | LR | Final Loss | Train Time | Converged |
|----------|-----------|-----|-----------|------------|-----------|
| rank8_lr1e-5 | 8 | 1e-5 | 0.234 | 8m | Yes |
| rank16_lr5e-5 | 16 | 5e-5 | 0.189 | 9m | Yes |
...
```

**Evaluation comparison:**
```markdown
| Run Name | Accuracy | Exact Match | F1 Score | Notes |
|----------|----------|-------------|----------|-------|
| rank8_lr1e-5 | 0.85 | 0.82 | 0.87 | Good |
| rank16_lr5e-5 | 0.91 | 0.89 | 0.92 | Best |
...
```

### 2. Visualization Plots

**Training curves:**
- Loss vs. step for all runs (overlaid line plot)
- Learning rate schedule
- GPU utilization over time

**Performance comparisons:**
- Bar chart: Final accuracy per run
- Heatmap: Accuracy by (LoRA rank × learning rate)
- Box plots: Score distribution across samples

**File format:** PNG or SVG, embedded in markdown report

### 3. Statistical Analysis

**Compare runs:**
- Mean and standard deviation
- Best/worst performers
- Statistical significance tests (if multiple seeds)
- Confidence intervals

**Identify patterns:**
- Which hyperparameters matter most?
- Diminishing returns analysis
- Cost-benefit analysis (accuracy vs. training time)

### 4. Markdown Report

**Structure:**
```markdown
# Experiment Analysis: {experiment_name}

## Overview
- Research question
- Variables tested
- Number of runs
- Date completed

## Training Results

### Loss Curves
![Training loss comparison](plots/training_loss.png)

### Convergence
- 7/8 runs converged successfully
- rank64_lr5e-5 showed oscillation

### Training Time
- Average: 10 minutes
- Range: 8-12 minutes
- Faster with lower rank (as expected)

## Evaluation Results

### Accuracy Comparison
![Accuracy by configuration](plots/accuracy_heatmap.png)

### Best Performers
1. rank16_lr5e-5: 91% accuracy
2. rank32_lr1e-5: 90% accuracy
3. rank8_lr5e-5: 87% accuracy

### Key Findings
- LoRA rank 16-32 optimal for this task
- Learning rate 5e-5 slightly better than 1e-5
- Diminishing returns above rank 32

## Statistical Summary

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | 0.88 | 0.03 | 0.85 | 0.91 |
| F1 Score | 0.89 | 0.02 | 0.87 | 0.92 |

## Recommendations

1. **Production config:** rank16, lr=5e-5
   - Best accuracy-cost tradeoff
   - Reliable convergence
   - 9 minute training time

2. **Further exploration:**
   - Test rank 24 (between 16 and 32)
   - Try warmup schedule
   - Evaluate on held-out generalization set

## Next Steps

1. Share these results with team
2. Run production fine-tuning with best config
3. Deploy model for inference testing

---

*Generated by analyze-experiment skill*
*Timestamp: 2025-10-24 01:00:00*
```

## Technical Implementation (Planned)

### Dependencies

**Python libraries:**
- pandas (data manipulation)
- matplotlib/seaborn (plotting)
- scipy (statistical tests)
- inspect-ai (evaluation log parsing)

**Skills integration:**
- Reads from outputs of run-torchtune and run-inspect
- Can invoke iteratively to refine analysis

### Log Parsing Strategies

**SLURM logs:**
```python
# Extract loss values
loss_pattern = r"Loss: ([0-9.]+)"
losses = re.findall(loss_pattern, slurm_output)

# Extract timing
time_pattern = r"Epoch [0-9]+ completed in ([0-9.]+)s"
times = re.findall(time_pattern, slurm_output)
```

**inspect-ai logs:**
```python
from inspect_ai import read_eval_log

# Load evaluation results
eval_log = read_eval_log("path/to/result.eval")
accuracy = eval_log.results.scores["accuracy"].value
samples = eval_log.samples
```

### Plot Generation

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Training loss comparison
fig, ax = plt.subplots(figsize=(10, 6))
for run_name, losses in all_losses.items():
    ax.plot(losses, label=run_name)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Comparison")
ax.legend()
fig.savefig("plots/training_loss.png", dpi=300)
```

## User Interaction (Planned)

**Initial analysis:**
- Automatically generate standard report
- Present summary to user

**Follow-up questions:**
- "Would you like me to explore {specific_aspect} in more detail?"
- "Should I generate a presentation-ready figure for {metric}?"
- "Would you like statistical significance tests?"

**Export options:**
- "I can export this data as CSV for further analysis"
- "Would you like me to create a LaTeX table for your paper?"

## Integration Points (Planned)

**Input from:**
- run-experiment (completion trigger)
- experiment_summary.md (design context)
- SLURM logs (training metrics)
- inspect-ai logs (evaluation metrics)

**Output for:**
- Research papers (formatted tables, figures)
- Presentations (high-quality plots)
- Team discussions (markdown reports)
- Further experiments (insights drive next design)

## Error Handling (Planned)

**If logs missing:**
- Warn user which runs have incomplete data
- Proceed with available data
- Note limitations in report

**If parsing fails:**
- Log which files couldn't be parsed
- Try alternative parsing strategies
- Ask user to check log format

**If no clear winner:**
- Report that results are similar
- Suggest additional experiments
- Provide confidence intervals

## Future Enhancements

Potential additions:
- Interactive plots (plotly instead of matplotlib)
- Automatic outlier detection
- Cost analysis (GPU hours × $/hour)
- Compare against baseline/literature
- Export to Weights & Biases for team sharing
- Automatic experiment retrospective generation

## Important Notes

- This skill is **planned for future implementation**
- Current workflow ends after run-experiment
- Users must manually analyze results for now
- This documentation serves as specification for future development
- Feedback welcome on what analyses would be most valuable

## Workarounds Until Implementation

**Manual analysis:**
```bash
# View evaluation results interactively
inspect view

# Export evaluation data
for dir in rank*/eval/logs; do
  inspect log export $dir/*.eval --format csv >> results.csv
done

# View in spreadsheet or Jupyter notebook
```

**Check training logs:**
```bash
# Find final loss values
grep "Final loss" rank*/slurm-*.out

# Compare training times
grep "Total time" rank*/slurm-*.out
```

**Create quick comparison:**
```python
import pandas as pd
from inspect_ai import read_eval_log
from pathlib import Path

# Load all eval logs
results = []
for eval_file in Path(".").glob("rank*/eval/logs/*.eval"):
    log = read_eval_log(str(eval_file))
    run_name = eval_file.parts[-3]
    accuracy = log.results.scores["accuracy"].value
    results.append({"run": run_name, "accuracy": accuracy})

df = pd.DataFrame(results)
print(df.sort_values("accuracy", ascending=False))
```
