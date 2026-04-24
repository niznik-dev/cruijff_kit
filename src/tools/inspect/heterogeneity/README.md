# Heterogeneity Analysis Tools

Tools for detecting performance heterogeneity in model predictions across demographic or experimental groups.

## Purpose

These tools analyze pre-computed model predictions to identify systematic performance differences (bias) across groups. This is critical for:
- Fairness evaluation (demographic groups)
- Understanding model behavior across experimental conditions
- Identifying which subpopulations a model predicts well/poorly

## Files

- **`heterogeneity_report.py`** - Standalone analysis script with statistical tests and visualization
- **`heterogeneity_eval.py`** - Inspect AI wrapper for standardized evaluation pipeline integration

## When to Use

Use heterogeneity analysis when you have:
- Binary classification predictions with group labels
- Concerns about fairness or differential performance
- Need to understand which groups your model serves well/poorly

## Input Format

Both tools require a CSV file with these columns:

| Column | Description | Type |
|--------|-------------|------|
| `INPUT` | The prompt/input text | string |
| `TRUE_LABEL` | Ground truth label | 0 or 1 |
| `PREDICTION` | Model's predicted label | 0 or 1 |
| `P(TRUE LABEL)` | Model's probability for true label | float [0,1] |
| `GROUP` or custom | Group identifier | string/numeric |

**Example CSV:**
```csv
INPUT,TRUE_LABEL,PREDICTION,P(TRUE LABEL),GROUP
"Is this positive?",1,1,0.85,female
"Is this positive?",0,0,0.72,male
"Is this positive?",1,0,0.45,female
```

## Usage

### Option 1: Standalone Analysis

Run directly with Python for quick analysis:

```bash
python heterogeneity_report.py \
  --input_file results.csv \
  --group_column GROUP \
  --output_dir heterogeneity_results
```

**Arguments:**
- `--input_file` (required) - Path to predictions CSV
- `--group_column` (required) - Name of the column containing group identifiers
- `--output_dir` (optional) - Output directory (default: `results`)

**Outputs:**
- `heterogeneity_report.json` - Statistical test results and identified outlier groups
- `group_performance.png` - Visualization of accuracy and AUC by group

### Option 2: Inspect AI Integration

Run as an Inspect AI evaluation task for standardized metrics:

```bash
inspect eval tools/inspect/heterogeneity/heterogeneity_eval.py \
  -- --input_file results.csv --group_column GROUP
```

**Arguments:**
- `--input_file` (required) - Path to predictions CSV
- `--group_column` (required) - Name of the column containing group identifiers

**Outputs:**
- Standard Inspect AI logs with custom metrics
- `inspect_results/heterogeneity_report.json` - Detailed analysis
- `inspect_results/group_performance.png` - Visualizations

## Statistical Methods

### Accuracy Heterogeneity
- **Method:** One-way ANOVA F-test
- **Null hypothesis:** All groups have equal accuracy
- **Threshold:** p < 0.05 indicates heterogeneity

### AUC Heterogeneity
- **Method:** Variance-based detection
- **Thresholds:**
  - Standard deviation > 0.1, OR
  - Range (max - min) > 0.3
- Both indicate heterogeneity

### Outlier Groups
Identifies specific problematic groups using:
- **Z-score threshold:** < -1.5 (more than 1.5 SD below mean)
- **AUC absolute threshold:** < 0.6 (worse than moderate performance)

## Output Interpretation

### heterogeneity_report.json Structure

```json
{
  "summary": {
    "total_samples": 1000,
    "num_groups": 10,
    "heterogeneity_found": true
  },
  "heterogeneity_analysis": {
    "accuracy_heterogeneity": {
      "heterogeneity": true,
      "p_value": 0.003,
      "f_statistic": 3.45
    },
    "auc_heterogeneity": {
      "heterogeneity": true,
      "std": 0.15,
      "range": 0.42,
      "mean": 0.78
    }
  },
  "identified_groups": {
    "accuracy_outliers": [...],
    "auc_outliers": [...],
    "outlying_in_both": [...]
  },
  "group_metrics": {
    "group_name": {
      "accuracy": 0.85,
      "auc": 0.92,
      "n_samples": 100
    }
  }
}
```

### Key Metrics

- **`heterogeneity_found`** - Overall detection (true if found in either accuracy or AUC)
- **`p_value`** - Statistical significance of accuracy differences (lower = more significant)
- **`outlying_in_both`** - Groups with poor accuracy AND AUC (highest concern)

## Example Workflow

```bash
# 1. Generate predictions from your model (using your evaluation pipeline)
#    Ensure output CSV has required columns

# 2. Run heterogeneity analysis
python tools/inspect/heterogeneity/heterogeneity_report.py \
  --input_file predictions.csv \
  --group_column demographic_group \
  --output_dir fairness_analysis

# 3. Review results
cat fairness_analysis/heterogeneity_report.json
open fairness_analysis/group_performance.png

# 4. If heterogeneity found, investigate problematic groups
#    - Check data distribution (sample sizes)
#    - Examine input characteristics
#    - Consider group-specific interventions
```

## Integration with cruijff_kit

While these tools work standalone, they're designed to integrate with the evaluation pipeline:

1. **After fine-tuning:** Evaluate your model and save predictions to CSV
2. **Run heterogeneity analysis:** Check for group-level performance issues
3. **Iterate:** Use insights to improve training data or model design

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Accuracy and AUC metrics
- `scipy` - Statistical tests (ANOVA)
- `matplotlib` - Visualization
- `inspect_ai` - (for heterogeneity_eval.py only)

All dependencies are included in the `cruijff` conda environment.

## History

Created in PR #84 (issue #53) by @DonggyuBan to enable fairness evaluation in the prediction pipeline.

## Notes

- **Minimum groups:** Statistical tests require at least 2 groups
- **Sample size:** Small groups may have unstable AUC calculations
- **Binary classification only:** Currently supports binary (0/1) predictions
- **Group column flexibility:** Can use any column name, not just 'GROUP'
