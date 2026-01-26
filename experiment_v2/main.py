
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from experiment_v2 import config
from experiment_v2.simulation import CostMonitor
from experiment_v2.models import StandardToG, FrugalGPT, TieredForest
from experiment_v2.models import StandardToG, FrugalGPT, TieredForest
from experiment_v2.data_loaders import load_all_datasets
from experiment_v2 import plot_results

def evaluate_model_on_dataset(model, dataset, model_name="Model"):
    monitor = CostMonitor()
    monitor.reset()
    
    results = []
    print(f"\n--- Evaluating {model_name} (N={len(dataset)}) ---")
    
    correct_count = 0
    for item in tqdm(dataset):
        try:
            prediction = model.solve(item['question'])
        except Exception as e:
            print(f"Error: {e}")
            prediction = ""
            
        # Accuracy Check (Keyword Match)
        # Ground truth might be short ("Moderate Risk") vs long explanation
        hit = item['ground_truth'].lower() in prediction.lower()
        if hit: correct_count += 1
        results.append({
            "dataset": item.get("dataset", "Unknown"),
            "question_id": item.get("id", 0),
            "correct": hit,
            "prediction": prediction[:50] + "..."
        })
        
    stats = monitor.get_session_stats()
    accuracy = correct_count / len(dataset) if dataset else 0
    
    print(f"Acc: {accuracy:.2%}, Cost: ${stats['cost_usd']:.4f}, Tokens: {stats['tokens_total']}")
    
    return {
        "name": model_name,
        "accuracy": accuracy,
        "tokens": stats['tokens_total'],
        "cost": stats['cost_usd'],
        "latency": stats['latency_avg'],
        "details": results
    }

def main():
    # 1. Load Data
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    # Load 15 samples from each dataset
    dataset = load_all_datasets(base_dir, limit_per_dataset=15)
    
    if not dataset:
        print("No data loaded. Check paths.")
        return

    # Group by dataset for detailed analysis
    datasets = {}
    for d in dataset:
        ds_name = d['dataset']
        if ds_name not in datasets: datasets[ds_name] = []
        datasets[ds_name].append(d)

    final_report = []

    # 2. Run Experiments Per Dataset
    for ds_name, ds_data in datasets.items():
        print(f"\n\n==========================================")
        print(f"  Running Benchmark on: {ds_name}")
        print(f"==========================================")
        
        # Models
        tog = StandardToG()
        frugal = FrugalGPT(thresholds=(0.8,))
        # Use generic optimized thresholds for now, or per-dataset?
        # Let's use the robust ones found earlier (Drop 0.2, Pass 0.7)
        # Or optimize on the fly per dataset (better for paper)
        
        # Optimize Tiered-Forest?
        # We'll skip per-dataset optimization loop to save time/tokens unless critical.
        # Using "Robust" parameters.
        tf = TieredForest(t_drop=0.2, t_pass=0.7)
        
        # Eval
        res_tog = evaluate_model_on_dataset(tog, ds_data, "ToG")
        res_frugal = evaluate_model_on_dataset(frugal, ds_data, "FrugalGPT")
        res_tf = evaluate_model_on_dataset(tf, ds_data, "Tiered-Forest")
        
        # Store
        for r in [res_tog, res_frugal, res_tf]:
            r['dataset'] = ds_name
            final_report.append(r)

    # 3. Aggregate & Save
    df = pd.DataFrame(final_report)
    df.to_csv("multi_dataset_results.csv", index=False)
    print("\nSaved multi_dataset_results.csv")
    
    # 4. Generate Markdown Report
    with open("experiment_analysis_multi.md", "w") as f:
        f.write("# Experiment Report: Multi-Dataset Benchmark\n\n")
        f.write("Datasets tested: MetaQA, WebQSP, Logistics (Supply Chain).\n\n")
        
        for ds_name in datasets.keys():
            subset = df[df['dataset'] == ds_name]
            f.write(f"## Dataset: {ds_name}\n")
            f.write(subset[['name', 'accuracy', 'tokens', 'cost', 'latency']].to_markdown(index=False))
            f.write("\n\n")
            
        f.write("## Overall Analysis\n")
        f.write("ToG vs FrugalGPT vs Tiered-Forest across distinct domains.\n")
        
    # 5. Generate Plots
    print("\nGenerating Plots...")
    try:
        plot_results.main()
        print("Plots generated successfully in experiment_v2/")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
