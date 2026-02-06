
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Color palette
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

def load_data():
    # Load ablation results
    ablation_df = pd.read_csv('c:/good/11/ablation_results/ablation_results_final.csv')
    # Load baseline results
    baseline_df = pd.read_csv('c:/good/11/baseline_results/baseline_results_final.csv')
    # Load original results
    original_df = pd.read_csv('c:/good/11/multi_dataset_results.csv')
    
    return ablation_df, baseline_df, original_df

def get_tokens(row):
    if 'total_tokens' in row:
        return row['total_tokens']
    if 'tokens' in row:
        return row['tokens']
    return 0

def plot_ablation_study(ablation_df, original_df):
    dataset = 'WebQSP' # Using WebQSP as the representative dataset
    
    # Filter data for WebQSP
    original_tf = original_df[(original_df['dataset'] == dataset) & (original_df['name'] == 'Tiered-Forest')].iloc[0]
    
    # Ablation variants mappings
    # A3: NoDynamic, A4: TwoTier (No Tier 1), A5: Tier1LLM (No Tier 2), A6: SingleThreshold
    variants = {
        'Full System': original_tf,
        'No Tier 1': ablation_df[(ablation_df['dataset'] == dataset) & (ablation_df['variant'] == 'A4')].iloc[0],
        'No Tier 2': ablation_df[(ablation_df['dataset'] == dataset) & (ablation_df['variant'] == 'A5')].iloc[0],
        'No Dynamic': ablation_df[(ablation_df['dataset'] == dataset) & (ablation_df['variant'] == 'A3')].iloc[0],
        'Single Threshold': ablation_df[(ablation_df['dataset'] == dataset) & (ablation_df['variant'] == 'A6')].iloc[0]
    }
    
    labels = list(variants.keys())
    accuracies = [v['accuracy'] * 100 for v in variants.values()]
    tokens = [get_tokens(v) for v in variants.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar for Accuracy
    rects1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#55A868', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Bar for Tokens
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, tokens, width, label='Token Count', color='#DD8452', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Token Consumption', fontweight='bold')
    # Use log scale for tokens as the difference is huge
    ax2.set_yscale('log')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title(f'Ablation Study: Component Impact on {dataset}', pad=20)
    
    # Legend
    lines, labels_l = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_l + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig('c:/good/11/figures/ablation_study_chart.png', dpi=300, bbox_inches='tight')
    print("Saved ablation_study_chart.png")

def plot_fair_baseline_comparison(baseline_df, original_df):
    dataset = 'WebQSP'
    
    # Data for comparison
    tog = original_df[(original_df['dataset'] == dataset) & (original_df['name'] == 'ToG')].iloc[0]
    frugal_fair = baseline_df[(baseline_df['dataset'] == dataset) & (baseline_df['variant'] == 'B2')].iloc[0]
    tf_full = original_df[(original_df['dataset'] == dataset) & (original_df['name'] == 'Tiered-Forest')].iloc[0]
    
    methods = ['Standard ToG', 'FrugalGPT (Fair)', 'Tiered-Forest']
    accuracies = [tog['accuracy']*100, frugal_fair['accuracy']*100, tf_full['accuracy']*100]
    tokens = [get_tokens(tog), get_tokens(frugal_fair), get_tokens(tf_full)]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Bar for Accuracy
    ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#4C72B0', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Bar for Tokens
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, tokens, width, label='Tokens', color='#C44E52', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Token Consumption', fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_title('Fair Baseline Comparison (WebQSP)', pad=20)
    
    # Add labels on top of bars
    for i, acc in enumerate(accuracies):
        ax1.text(i - width/2, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, t in enumerate(tokens):
        ax2.text(i + width/2, t + 100, str(int(t)), ha='center', va='bottom', fontsize=10, color='#C44E52', fontweight='bold')

    # Legend
    lines, labels_l = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_l + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('c:/good/11/figures/fair_comparison_chart.png', dpi=300)
    print("Saved fair_comparison_chart.png")

def plot_token_savings_matrix(baseline_df, original_df):
    # Calculate savings across all datasets
    datasets = ['MetaQA', 'WebQSP', 'Logistics']
    results = []
    
    for ds in datasets:
        tog_tokens = get_tokens(original_df[(original_df['dataset'] == ds) & (original_df['name'] == 'ToG')].iloc[0])
        tf_tokens = get_tokens(original_df[(original_df['dataset'] == ds) & (original_df['name'] == 'Tiered-Forest')].iloc[0])
        savings = (tog_tokens - tf_tokens) / tog_tokens * 100
        results.append(savings)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(datasets, results, color='#8172B3', alpha=0.7, edgecolor='black', width=0.5)
    ax.set_ylabel('Token Savings (%)', fontweight='bold')
    ax.set_title('Token Savings Relative to Standard ToG', pad=15)
    ax.set_ylim(0, 100)
    
    for i, v in enumerate(results):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('c:/good/11/figures/token_savings_by_dataset.png', dpi=300)
    print("Saved token_savings_by_dataset.png")

if __name__ == "__main__":
    if not os.path.exists('c:/good/11/figures'):
        os.makedirs('c:/good/11/figures')
        
    try:
        a_df, b_df, o_df = load_data()
        plot_ablation_study(a_df, o_df)
        plot_fair_baseline_comparison(b_df, o_df)
        plot_token_savings_matrix(b_df, o_df)
        print("All charts generated successfully.")
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
