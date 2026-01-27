"""
Fair Baseline Experiment Execution Script

Runs fair baseline variants (B2-B7) across multiple datasets
and generates comparison reports.
"""

import os
import pandas as pd
from typing import List, Dict
from experiment_v2.baseline_models import create_baseline_model
from experiment_v2.ablation_experiment import evaluate_model_detailed, generate_ablation_report
from experiment_v2.data_loaders import load_all_datasets

def run_baseline_experiment(datasets: Dict[str, List[Dict]], 
                            output_dir: str = 'baseline_results',
                            variants: List[str] = None) -> pd.DataFrame:
    """
    Run fair baseline experiment
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default variants if not specified
    if variants is None:
        variants = ['B2', 'B3', 'B5', 'B6', 'B7']
    
    all_results = []
    
    print("="*60)
    print("FAIR BASELINE EXPERIMENT")
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
            model = create_baseline_model(variant)
            
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
                df_partial.to_csv(f'{output_dir}/baseline_results_partial.csv', index=False)
        
        except Exception as e:
            print(f"\nERROR in variant {variant}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final results
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(f'{output_dir}/baseline_results_final.csv', index=False)
    
    # Generate report (reuse ablation reporting since metrics are same)
    generate_ablation_report(df_final, output_dir)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return df_final

if __name__ == "__main__":
    # Load datasets
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    # Same limit as ablation run (n=15)
    all_data = load_all_datasets(base_dir, limit_per_dataset=15)
    
    # Group by dataset
    datasets = {}
    for item in all_data:
        ds_name = item['dataset']
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(item)
    
    # Run experiment
    metrics = run_baseline_experiment(
        datasets=datasets,
        output_dir='baseline_results'
    )
