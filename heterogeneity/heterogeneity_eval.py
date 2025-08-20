from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer, scorer, Score, metric
import pandas as pd
import os

from heterogeneity_report import (
    calculate_group_metrics,
    find_heterogeneity,
    identify_groups,
    create_visualizations,
    generate_report
)

accumulated_data = []
EXPECTED_SAMPLES = None

def load_samples_for_inspect(csv_path, group_column='GROUP'):
    """Convert CSV to Inspect AI Samples - works with ANY group column"""
    global EXPECTED_SAMPLES
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    EXPECTED_SAMPLES = len(df)
    
    print(f"Loaded {len(df)} rows.")
    print(f"Using group column: {group_column}")
    
    unique_groups = df[group_column].unique()
    n_groups = len(unique_groups)
    
    print(f"Found {n_groups} unique groups")
    if n_groups <= 10:
        print(f"Groups: {sorted(unique_groups)}")
    else:
        print(f"Groups: {sorted(unique_groups)[:5]}... and {n_groups-5} more")
    
    samples = []
    for index, row in df.iterrows():
        sample = Sample(
            input=str(row['INPUT']),
            target=str(int(row['TRUE_LABEL'])),
            metadata={
                'prediction': int(row['PREDICTION']),
                'probability': float(row['P(TRUE LABEL)']),
                'group': str(row[group_column])  # Convert to string for consistency
            }
        )
        samples.append(sample)
        
        if (index + 1) % 1000 == 0:
            print(f"Converted {index + 1} samples...")
    
    print(f"Converted {len(samples)} samples.")
    return samples

@metric
def heterogeneity_metric():
    def compute(scores):
        global accumulated_data, EXPECTED_SAMPLES
        
        new_scores = scores[len(accumulated_data):]
        
        for score in new_scores:
            if score.metadata:
                accumulated_data.append(score.metadata)
        
        total_samples = len(accumulated_data)
        
        if total_samples % 1000 == 0 or total_samples == EXPECTED_SAMPLES:
            print(f"Accumulated {total_samples}/{EXPECTED_SAMPLES} samples...")
        
        if total_samples >= EXPECTED_SAMPLES:
            print("\n=== RUNNING HETEROGENEITY ANALYSIS ===")
            
            data_for_df = []
            for item in accumulated_data:
                data_for_df.append({
                    'TRUE_LABEL': item['TRUE_LABEL'],
                    'PREDICTION': item['PREDICTION'],
                    'P(TRUE LABEL)': item['P(TRUE LABEL)'],
                    'GROUP': str(item['group'])  # Ensure string type
                })
            
            df = pd.DataFrame(data_for_df)
            
            n_groups = len(df['GROUP'].unique())
            print(f"Analyzing {n_groups} groups...")
            
            group_metrics = calculate_group_metrics(df)
            heterogeneity_results = find_heterogeneity(df, group_metrics)
            identified_groups = identify_groups(group_metrics)
            
            output_dir = 'inspect_results'
            os.makedirs(output_dir, exist_ok=True)
            
            create_visualizations(group_metrics, output_dir)
            report = generate_report(df, group_metrics, heterogeneity_results, 
                                   identified_groups, output_dir)
            
            accumulated_data = []
            
            metrics_dict = {
                'heterogeneity_found': float(report['summary']['heterogeneity_found']),
                'n_groups': float(n_groups),
                'accuracy_p_value': float(heterogeneity_results['accuracy_heterogeneity']['p_value']),
                'auc_std': float(heterogeneity_results['auc_heterogeneity']['std']),
                'auc_range': float(heterogeneity_results['auc_heterogeneity']['range']),
                'n_accuracy_outliers': float(len(identified_groups['accuracy_outliers'])),
                'n_auc_outliers': float(len(identified_groups['auc_outliers'])),
                'n_outlying_in_both': float(len(identified_groups['outlying_in_both']))
            }
            
            return metrics_dict
        
        else:
            return {'samples_processed': float(total_samples)}
    
    return compute

@scorer(metrics=[heterogeneity_metric()])
def heterogeneity_scorer():
    
    async def score(state, target, **kwargs):
        if hasattr(target, 'target'):
            target_value = target.target[0] if isinstance(target.target, list) else target.target
        else:
            target_value = str(target)
        
        return Score(
            value=1.0,
            answer=str(state.metadata.get("prediction", "")),
            metadata={
                'TRUE_LABEL': int(target_value),
                'PREDICTION': state.metadata.get("prediction"),
                'P(TRUE LABEL)': state.metadata.get("probability"),
                'group': state.metadata.get("group")
            }
        )
    
    return score

@task
def heterogeneity_task():

    input_file = os.environ.get('INPUT_FILE', 'predictions.csv')
    group_column = os.environ.get('GROUP_COLUMN', 'GROUP')
    
    print(f"Running heterogeneity task with:")
    print(f"  Input file: {input_file}")
    print(f"  Group column: {group_column}")
    
    return Task(
        dataset=load_samples_for_inspect(input_file, group_column),
        solver=[],
        scorer=heterogeneity_scorer()
    )