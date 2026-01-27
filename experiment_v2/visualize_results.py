"""
Visualization Script for Ablation Study

Generates charts:
1. Accuracy vs Cost (Scatter Plot)
2. Token Consumption by Component (Stacked Bar)
3. Latency vs Accuracy (Scatter Plot)
4. Routing Distribution (Stacked Bar)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_accuracy_vs_cost(df, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Calculate means
    summary = df.groupby('variant').agg({
        'accuracy': 'mean',
        'cost_usd': 'mean',
        'latency_total': 'mean'
    }).reset_index()
    
    sns.scatterplot(data=summary, x='cost_usd', y='accuracy', hue='variant', s=100)
    
    # Add labels
    for i, row in summary.iterrows():
        plt.text(row['cost_usd'], row['accuracy'], row['variant'], 
                 fontsize=9, ha='left', va='bottom')
        
    plt.title('Accuracy vs Cost Trade-off')
    plt.xlabel('Average Cost per Query ($)')
    plt.ylabel('Average Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_cost.png')
    plt.close()

def plot_routing_distribution(df, output_dir):
    # Parse routing distribution column if necessary
    # Assuming metrics are already in columns like 'fast_pass_rate' etc from aggregation?
    # Actually aggregate script just concats CSVs.
    # The CSVs have 'routing_distribution' as a string dict representation.
    # We need to parse it.
    
    import ast
    
    def parse_routing(row):
        try:
            d = ast.literal_eval(row['routing_distribution'])
            return pd.Series(d)
        except:
            return pd.Series({})

    routing_df = df.apply(parse_routing, axis=1)
    df = pd.concat([df, routing_df], axis=1)
    
    # Fill NaN with 0
    cols = ['fast_pass', 'escalate', 'tier1_hit', 'discard']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
        
    # Group by variant
    summary = df.groupby('variant')[cols].mean()
    
    summary.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Average Routing Distribution by Variant')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/routing_distribution.png')
    plt.close()

def generate_visualizations(csv_path='final_results/comprehensive_results.csv', output_dir='final_results/plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    plot_accuracy_vs_cost(df, output_dir)
    plot_routing_distribution(df, output_dir)
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    generate_visualizations()
