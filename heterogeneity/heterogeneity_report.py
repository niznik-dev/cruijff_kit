import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score 
from scipy.stats import f_oneway

def load_data(file_path, group_column='GROUP'):
    """Load CSV data for analysis"""
    df = pd.read_csv(file_path)
    
    if group_column != 'GROUP' and group_column in df.columns:
        df['GROUP'] = df[group_column]
    
    print(f"Loaded {len(df)} samples from {len(df['GROUP'].unique())} groups")
    return df

def calculate_group_metrics(df):
    """Calculate accuracy and AUC for each group"""
    metrics = {}
    
    for group in df['GROUP'].unique():
        group_df = df[df['GROUP'] == group]
        
        accuracy = accuracy_score(
            group_df['TRUE_LABEL'],
            group_df['PREDICTION']
        )
        
        try:
            auc = roc_auc_score(
                group_df['TRUE_LABEL'],
                group_df['P(TRUE LABEL)']
            )
        except:
            auc = np.nan
        
        metrics[group] = {
            'accuracy': accuracy,
            'auc': auc,
            'n_samples': len(group_df)
        }
    
    return metrics

def find_heterogeneity(df, group_metrics):
    """Find heterogeneity using ANOVA and variance methods"""
    results = {}
    
    # Accuracy heterogeneity (ANOVA)
    group_accuracy_scores = []
    groups = sorted(df['GROUP'].unique())
    
    for group in groups:
        group_df = df[df['GROUP'] == group]
        correct = (group_df['PREDICTION'] == group_df['TRUE_LABEL']).astype(int)
        group_accuracy_scores.append(correct.values)
    
    if len(groups) > 1:
        f_stat, p_value = f_oneway(*group_accuracy_scores)
    else:
        f_stat, p_value = 0, 1.0
    
    results['accuracy_heterogeneity'] = {
        'heterogeneity': bool(p_value < 0.05),
        'p_value': p_value,
        'f_statistic': f_stat
    }
    
    # AUC heterogeneity (variance-based)
    aucs = [m['auc'] for m in group_metrics.values() if not np.isnan(m['auc'])]
    
    if len(aucs) > 1:
        auc_std = np.std(aucs)
        auc_range = max(aucs) - min(aucs)
        heterogeneity_found = bool(auc_std > 0.1 or auc_range > 0.3)
    else:
        auc_std = 0
        auc_range = 0
        heterogeneity_found = False
    
    results['auc_heterogeneity'] = {
        'heterogeneity': heterogeneity_found,
        'std': auc_std,
        'range': auc_range,
        'mean': np.mean(aucs) if aucs else 0
    }
    
    return results

def identify_groups(group_metrics, z_threshold=-1.5, auc_threshold=0.6):
    """Identify which specific groups show heterogeneity"""

    accuracies = [m['accuracy'] for m in group_metrics.values()]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
    
    aucs = [m['auc'] for m in group_metrics.values() if not np.isnan(m['auc'])]
    mean_auc = np.mean(aucs) if aucs else 0
    std_auc = np.std(aucs) if len(aucs) > 1 else 0
    
    identified_groups = {
        'accuracy_outliers': [],
        'auc_outliers': [],
        'outlying_in_both': []
    }
    
    for group, metrics in group_metrics.items():
        acc_outlier = False
        auc_outlier = False
        
        if std_acc > 0:
            z_score_acc = (metrics['accuracy'] - mean_acc) / std_acc
            if z_score_acc < z_threshold:
                identified_groups['accuracy_outliers'].append({
                    'group': group,
                    'accuracy': metrics['accuracy'],
                    'z_score': z_score_acc
                })
                acc_outlier = True
        
        if not np.isnan(metrics['auc']):
            if std_auc > 0:
                z_score_auc = (metrics['auc'] - mean_auc) / std_auc
                if z_score_auc < z_threshold or metrics['auc'] < auc_threshold:
                    identified_groups['auc_outliers'].append({
                        'group': group,
                        'auc': metrics['auc'],
                        'z_score': z_score_auc
                    })
                    auc_outlier = True
            elif metrics['auc'] < auc_threshold:
                identified_groups['auc_outliers'].append({
                    'group': group,
                    'auc': metrics['auc'],
                    'z_score': 0
                })
                auc_outlier = True
        
        if acc_outlier and auc_outlier:
            identified_groups['outlying_in_both'].append(group)
    
    return identified_groups

def visualizations(group_metrics, output_dir='results'):
    """Create performance visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    group_metrics_str = {str(k): v for k, v in group_metrics.items()}
    sorted_groups = sorted(group_metrics_str.keys(), 
                          key=lambda x: float(x) if x.replace('.','').isdigit() else x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    accuracies = [group_metrics_str[g]['accuracy'] for g in sorted_groups]
    x_pos = np.arange(len(sorted_groups))
    
    ax1.bar(x_pos, accuracies, color='skyblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_groups, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Group')
    ax1.set_title('Accuracy by Group')
    ax1.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # AUC plot
    aucs = [group_metrics_str[g]['auc'] if not np.isnan(group_metrics_str[g]['auc']) else 0 
            for g in sorted_groups]
    
    ax2.bar(x_pos, aucs, color='lightcoral', edgecolor='darkred')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_groups, rotation=45, ha='right')
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('Group')
    ax2.set_title('AUC by Group')
    
    valid_aucs = [a for a in aucs if a > 0]
    if valid_aucs:
        ax2.axhline(y=np.mean(valid_aucs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(valid_aucs):.3f}')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Random')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/group_performance.png")

def generate_report(df, group_metrics, heterogeneity_results, identified_groups, output_dir='results'):
    """Generate a heterogeneity report"""

    os.makedirs(output_dir, exist_ok=True)
    
    group_metrics_json = {str(k): v for k, v in group_metrics.items()}
    
    report = {
        'summary': {
            'total_samples': int(len(df)),
            'num_groups': len(group_metrics_json),
            'heterogeneity_found': bool(heterogeneity_results['accuracy_heterogeneity']['heterogeneity'] or 
                                 heterogeneity_results['auc_heterogeneity']['heterogeneity']),
        },
        'heterogeneity_analysis': heterogeneity_results,
        'identified_groups': identified_groups,
        'group_metrics': group_metrics_json
    }
    
    with open(f'{output_dir}/heterogeneity_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nHeterogeneity found: {report['summary']['heterogeneity_found']}")
    if identified_groups['outlying_in_both']:
        print(f"Groups outlying in both: {identified_groups['outlying_in_both']}")
    
    return report

def run_analysis(input_file, group_column='GROUP', output_dir='results'):
    """Main analysis pipeline"""
    df = load_data(input_file, group_column)
    
    group_metrics = calculate_group_metrics(df)
    heterogeneity_results = find_heterogeneity(df, group_metrics)
    identified_groups = identify_groups(group_metrics)
    
    visualizations(group_metrics, output_dir)
    report = generate_report(df, group_metrics, heterogeneity_results, identified_groups, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions for heterogeneity")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--group_column', type=str, default='GROUP')
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    run_analysis(args.input_file, args.group_column, args.output_dir)