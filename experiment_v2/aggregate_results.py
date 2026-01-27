"""
Result Aggregation Script

Merges results from multiple experiment runs:
1. Ablation Run 1 (E0, A1, A2)
2. Ablation Run 2 (A3-A7)
3. Baseline Run (B2-B7)

Generates final consolidated report and visualization data.
"""

import os
import pandas as pd
import glob
from experiment_v2.ablation_experiment import generate_ablation_report

def aggregate_results(output_dir='final_results'):
    """
    Aggregate all experiment result CSVs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List of files to merge
    files = [
        'ablation_results/ablation_results_run1.csv',      # E0, A1, A2
        'ablation_results/ablation_results_final.csv',     # A3-A7
        'baseline_results/baseline_results_final.csv'      # B2-B7
    ]
    
    dfs = []
    for f in files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            dfs.append(pd.read_csv(f))
        else:
            print(f"Warning: File not found: {f}")
    
    if not dfs:
        print("No results found!")
        return
    
    # Merge
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by variant name
    full_df = full_df.sort_values(['variant', 'dataset'])

    # Parse string dictionaries
    import ast
    dict_cols = ['routing_distribution', 'latency_breakdown', 'tier_calls', 'tier_tokens', 'tier1_recall_metrics']
    
    for col in dict_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Save merged
    final_csv = f'{output_dir}/comprehensive_results.csv'
    full_df.to_csv(final_csv, index=False)
    print(f"Merged {len(full_df)} records to {final_csv}")
    
    # Generate unified report
    generate_ablation_report(full_df, output_dir)
    print(f"Generated unified report in {output_dir}/")
    
    return full_df

if __name__ == "__main__":
    aggregate_results()
