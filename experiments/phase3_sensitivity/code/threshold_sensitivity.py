"""
Threshold Sensitivity Analysis Script

Tests different tau_high values to find optimal balance between
accuracy and cost efficiency.
"""

import os
import pandas as pd
from experiment_v2.ablation_models import TieredForestFull
from experiment_v2.data_loaders import load_all_datasets
from experiment_v2.simulation import CostMonitor
from tqdm import tqdm


def evaluate_with_threshold(model, dataset, tau_high, tau_low=0.2):
    """
    Evaluate model with specific threshold values
    """
    # Update router thresholds
    model.router.tau_high = tau_high
    model.router.tau_low = tau_low
    model.router.dynamic = False  # Static thresholds for fair comparison
    
    model.reset_metrics()
    monitor = CostMonitor()
    monitor.reset()
    
    correct = 0
    for item in tqdm(dataset, desc=f"tau_high={tau_high}"):
        try:
            pred = model.solve(item['question'], ground_truth=item['ground_truth'])
            if item['ground_truth'].lower() in pred.lower():
                correct += 1
        except Exception as e:
            print(f"Error: {e}")
    
    metrics = model.get_metrics()
    cost_stats = monitor.get_session_stats()
    
    return {
        'tau_high': tau_high,
        'tau_low': tau_low,
        'accuracy': correct / len(dataset) if dataset else 0,
        'total_tokens': cost_stats.get('tokens_total', 0),
        'cost_usd': cost_stats.get('cost_usd', 0),
        'latency': metrics.get('latency_total', 0),
        'fast_pass_rate': metrics.get('routing_distribution', {}).get('fast_pass', 0),
        'escalate_rate': metrics.get('routing_distribution', {}).get('escalate', 0),
        'tier1_hit_rate': metrics.get('routing_distribution', {}).get('tier1_hit', 0)
    }


def run_threshold_sensitivity(datasets, tau_high_values, output_dir='threshold_sensitivity'):
    """
    Run threshold sensitivity analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    print("="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Testing tau_high values: {tau_high_values}")
    print(f"Datasets: {list(datasets.keys())}")
    print("="*60)
    
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (n={len(dataset)})")
        print(f"{'='*60}")
        
        for tau_high in tau_high_values:
            print(f"\nTesting tau_high = {tau_high}")
            
            # Create fresh model instance
            model = TieredForestFull()
            
            # Evaluate
            result = evaluate_with_threshold(model, dataset, tau_high)
            result['dataset'] = dataset_name
            
            all_results.append(result)
            
            # Print summary
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Tokens: {result['total_tokens']}")
            print(f"  Fast-Pass: {result['fast_pass_rate']:.1%}")
            print(f"  Escalate: {result['escalate_rate']:.1%}")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f'{output_dir}/threshold_sensitivity_results.csv', index=False)
    
    # Generate report
    generate_sensitivity_report(df, output_dir)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return df


def generate_sensitivity_report(df, output_dir):
    """
    Generate markdown report
    """
    report_path = f'{output_dir}/threshold_sensitivity_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Threshold Sensitivity Analysis Report\n\n")
        f.write(f"**Tested tau_high values**: {sorted(df['tau_high'].unique())}\n\n")
        f.write(f"**Datasets**: {', '.join(df['dataset'].unique())}\n\n")
        f.write("---\n\n")
        
        for dataset in df['dataset'].unique():
            f.write(f"## Dataset: {dataset}\n\n")
            
            subset = df[df['dataset'] == dataset].sort_values('tau_high')
            
            f.write("| tau_high | Accuracy | Tokens | Cost ($) | Fast-Pass% | Escalate% |\n")
            f.write("|----------|----------|--------|----------|------------|----------|\n")
            
            for _, row in subset.iterrows():
                f.write(f"| {row['tau_high']:.2f} | {row['accuracy']:.1%} | "
                       f"{row['total_tokens']:,} | ${row['cost_usd']:.4f} | "
                       f"{row['fast_pass_rate']:.1%} | {row['escalate_rate']:.1%} |\n")
            
            f.write("\n")
            
            # Find optimal
            best_acc = subset.loc[subset['accuracy'].idxmax()]
            best_cost = subset.loc[subset['total_tokens'].idxmin()]
            
            f.write("### Recommendations\n\n")
            f.write(f"- **Best Accuracy**: tau_high={best_acc['tau_high']:.2f} "
                   f"({best_acc['accuracy']:.1%})\n")
            f.write(f"- **Best Cost**: tau_high={best_cost['tau_high']:.2f} "
                   f"({best_cost['total_tokens']:,} tokens)\n\n")
            
            f.write("---\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        summary = df.groupby('tau_high').agg({
            'accuracy': 'mean',
            'total_tokens': 'mean',
            'cost_usd': 'mean',
            'fast_pass_rate': 'mean'
        }).round(4)
        
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        f.write("---\n\n")
        f.write(f"**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    # Load datasets
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    all_data = load_all_datasets(base_dir, limit_per_dataset=15)
    
    # Group by dataset
    datasets = {}
    for item in all_data:
        ds_name = item['dataset']
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(item)
    
    # Test threshold values
    tau_high_values = [0.7, 0.8, 0.85, 0.9]
    
    # Run analysis
    results_df = run_threshold_sensitivity(
        datasets=datasets,
        tau_high_values=tau_high_values,
        output_dir='threshold_sensitivity'
    )
    
    print("\nThreshold sensitivity analysis complete!")
    print(f"Results shape: {results_df.shape}")
