"""
Ablation Experiment Execution Script

Runs all ablation variants (E0, A1-A7) across multiple datasets
and generates comprehensive analysis reports.
"""

import os
import ast
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from experiment_v2.data_loaders import load_all_datasets
from experiment_v2.metrics_tracker import MetricsTracker
from experiment_v2.simulation import CostMonitor
# Import from new location
from experiments.phase1_ablation.code.ablation_models import create_ablation_model


def evaluate_model_detailed(model, dataset: List[Dict], experiment_name: str) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset with detailed metrics
    
    Args:
        model: Model instance with solve() method
        dataset: List of {question, ground_truth, dataset} dicts
        experiment_name: Name for logging
        
    Returns:
        Dictionary with comprehensive results
    """
    model.reset_metrics()
    monitor = CostMonitor()
    monitor.reset()
    
    correct_count = 0
    results_detail = []
    
    print(f"\nEvaluating {experiment_name} on {len(dataset)} samples...")
    
    for item in tqdm(dataset, desc=experiment_name):
        try:
            question = item['question']
            ground_truth = item['ground_truth']
            
            # Get prediction
            prediction = model.solve(question, ground_truth=ground_truth)
            
            # Check correctness (keyword match)
            is_correct = ground_truth.lower() in prediction.lower()
            if is_correct:
                correct_count += 1
            
            results_detail.append({
                'question': question[:100],  # Truncate for storage
                'ground_truth': ground_truth,
                'prediction': prediction[:100],
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"\nError on question: {item.get('question', 'N/A')[:50]}")
            print(f"Error: {e}")
            results_detail.append({
                'question': item.get('question', 'N/A')[:100],
                'ground_truth': item.get('ground_truth', 'N/A'),
                'prediction': f"ERROR: {str(e)}",
                'correct': False
            })
    
    # Get metrics from model
    model_metrics = model.get_metrics()
    
    # Get cost metrics from monitor
    cost_stats = monitor.get_session_stats()
    
    # Compute accuracy
    accuracy = correct_count / len(dataset) if dataset else 0.0
    
    # Compile results
    results = {
        'experiment': experiment_name,
        'model_name': model.get_name(),
        'accuracy': accuracy,
        'total_samples': len(dataset),
        'correct_count': correct_count,
        
        # Cost metrics
        'total_tokens': cost_stats.get('tokens_total', 0),
        'cost_usd': cost_stats.get('cost_usd', 0.0),
        
        # Latency metrics
        'latency_total': model_metrics.get('latency_total', 0.0),
        'latency_breakdown': model_metrics.get('latency_breakdown', {}),
        
        # Routing metrics
        'routing_distribution': model_metrics.get('routing_distribution', {}),
        'tier_calls': model_metrics.get('tier_calls', {}),
        'tier_tokens': model_metrics.get('tier_tokens', {}),
        
        # Tier 1 recall
        'tier1_recall': model_metrics.get('tier1_recall_metrics', {}).get('recall', 0.0),
        'tier1_precision': model_metrics.get('tier1_recall_metrics', {}).get('precision', 0.0),
        'tier1_hit_rate': model_metrics.get('tier1_recall_metrics', {}).get('tier1_hit_rate', 0.0),
        
        # Threshold info
        'threshold_adjustments': model_metrics.get('threshold_adjustments', 0),
        
        # Detailed results
        'details': results_detail
    }
    
    return results


def run_ablation_experiment(datasets: Dict[str, List[Dict]], 
                            output_dir: str = 'experiments/phase1_ablation/results',
                            variants: List[str] = None) -> pd.DataFrame:
    """
    Run complete ablation experiment
    
    Args:
        datasets: Dict of {dataset_name: list of samples}
        output_dir: Directory to save results
        variants: List of variant names to run (default: all E0, A1-A7)
        
    Returns:
        DataFrame with all results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default to all variants
    if variants is None:
        variants = ['E0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    all_results = []
    
    print("="*60)
    print("ABLATION STUDY EXPERIMENT")
    print("="*60)
    print(f"Variants: {', '.join(variants)}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running Variant: {variant}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = create_ablation_model(variant)
            
            # Run on each dataset
            for dataset_name, dataset in datasets.items():
                print(f"\nDataset: {dataset_name} (n={len(dataset)})")
                
                # Evaluate
                results = evaluate_model_detailed(
                    model, 
                    dataset, 
                    f"{variant}_{dataset_name}"
                )
                
                # Add dataset info
                results['dataset'] = dataset_name
                results['variant'] = variant
                
                all_results.append(results)
                
                # Print summary
                print(f"  Accuracy: {results['accuracy']:.1%}")
                print(f"  Tokens: {results['total_tokens']}")
                print(f"  Cost: ${results['cost_usd']:.4f}")
                print(f"  Latency: {results['latency_total']:.2f}s")
                
                # Save intermediate results
                df_partial = pd.DataFrame(all_results)
                df_partial.to_csv(f'{output_dir}/ablation_results_partial.csv', index=False)
        
        except Exception as e:
            print(f"\nERROR in variant {variant}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final results
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(f'{output_dir}/ablation_results_final.csv', index=False)
    
    # Generate summary report
    generate_ablation_report(df_final, output_dir)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return df_final


def generate_ablation_report(df: pd.DataFrame, output_dir: str):
    """
    Generate markdown analysis report
    
    Args:
        df: Results DataFrame
        output_dir: Output directory
    """
    report_path = f'{output_dir}/ablation_analysis.md'
    
    with open(report_path, 'w') as f:
        f.write("# Ablation Study Analysis Report\n\n")
        f.write(f"**Total Experiments**: {len(df)}\n\n")
        f.write(f"**Variants Tested**: {', '.join(df['variant'].unique())}\n\n")
        f.write(f"**Datasets**: {', '.join(df['dataset'].unique())}\n\n")
        
        f.write("---\n\n")
        
        # Per-dataset analysis
        for dataset in df['dataset'].unique():
            f.write(f"## Dataset: {dataset}\n\n")
            
            subset = df[df['dataset'] == dataset].copy()
            
            # Sort by accuracy descending
            subset = subset.sort_values('accuracy', ascending=False)
            
            # Create comparison table
            f.write("| Variant | Accuracy | Tokens | Cost ($) | Latency (s) | Tier1 Hit% | Fast-Pass% |\n")
            f.write("|---------|----------|--------|----------|-------------|------------|------------|\n")
            
            for _, row in subset.iterrows():
                variant = row['variant']
                acc = row['accuracy']
                tokens = row['total_tokens']
                cost = row['cost_usd']
                latency = row['latency_total']
                
                # Extract routing stats
                routing = row.get('routing_distribution', {})
                if isinstance(routing, str):
                    try:
                        routing = ast.literal_eval(routing)
                    except:
                        routing = {}
                
                tier1_hit = routing.get('tier1_hit', 0.0) * 100
                fast_pass = routing.get('fast_pass', 0.0) * 100
                
                f.write(f"| **{variant}** | {acc:.1%} | {tokens:,} | ${cost:.4f} | {latency:.2f} | {tier1_hit:.0f}% | {fast_pass:.0f}% |\n")
            
            f.write("\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            # Find baseline (E0)
            baseline = subset[subset['variant'] == 'E0']
            if not baseline.empty:
                baseline_tokens = baseline.iloc[0]['total_tokens']
                baseline_acc = baseline.iloc[0]['accuracy']
                
                f.write(f"**Baseline (E0)**: {baseline_acc:.1%} accuracy, {baseline_tokens:,} tokens\n\n")
                
                # Compare each variant to baseline
                for _, row in subset.iterrows():
                    if row['variant'] == 'E0':
                        continue
                    
                    variant = row['variant']
                    token_change = ((row['total_tokens'] - baseline_tokens) / baseline_tokens) * 100
                    acc_change = (row['accuracy'] - baseline_acc) * 100
                    
                    f.write(f"- **{variant}**: ")
                    f.write(f"Tokens {token_change:+.0f}%, ")
                    f.write(f"Accuracy {acc_change:+.1f}pp\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        
        # Aggregate by variant across datasets
        variant_summary = df.groupby('variant').agg({
            'accuracy': 'mean',
            'total_tokens': 'mean',
            'cost_usd': 'mean',
            'latency_total': 'mean'
        }).round(4)
        
        f.write("### Average Performance Across Datasets\n\n")
        f.write(variant_summary.to_markdown())
        f.write("\n\n")
        
        f.write("---\n\n")
        f.write("**Report Generated**: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    
    print(f"\nAnalysis report saved to: {report_path}")


if __name__ == "__main__":
    # Example usage
    from experiment_v2.data_loaders import load_all_datasets
    
    # Load datasets
    base_dir = "data"
    all_data = load_all_datasets(base_dir, limit_per_dataset=15)
    
    # Group by dataset
    datasets = {}
    for item in all_data:
        ds_name = item['dataset']
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(item)
    
    # Run experiment
    results_df = run_ablation_experiment(
        datasets=datasets,
        output_dir='experiments/phase1_ablation/results',
        variants=['E0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']  # Run all variants
    )
    
    print("\nExperiment complete!")
    print(f"Results shape: {results_df.shape}")
