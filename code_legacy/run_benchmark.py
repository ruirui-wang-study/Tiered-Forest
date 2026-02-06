import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulation import DataGenerator, CostManager, PROFILES
from models import StandardToG, FrugalGPT, TieredForest

def run_benchmark():
    print("--- Starting Multi-Dataset Benchmark for Tiered-Forest ---")
    
    all_results = []
    
    # Iterate over datasets
    for profile_name, profile in PROFILES.items():
        print(f"\n>>> Simulating Dataset: {profile_name}")
        print(f"    (Ambiguity: {profile.avg_ambiguity}, Complexity: {profile.complexity_ratio:.0%})")
        
        generator = DataGenerator(profile, seed=101)
        samples = generator.generate_samples(n=100)
        
        models = [StandardToG(), FrugalGPT(), TieredForest()]
        
        for model in models:
            cm = CostManager()
            total_acc = 0.0
            
            for sample in samples:
                acc = model.solve(sample, cm)
                total_acc += acc
                
            # Compile Stats
            stats = cm.get_stats()
            res = {
                "Dataset": profile_name,
                "Model": model.name,
                "Accuracy": total_acc / len(samples),
                "Total Tokens": stats["tokens"] / len(samples),
                "Avg Latency": stats["latency"] / len(samples),
                "Symbolic Time": stats["latency_breakdown"]["symbolic"] / len(samples),
                "Small Time": stats["latency_breakdown"]["small"] / len(samples),
                "Large Time": stats["latency_breakdown"]["large"] / len(samples)
            }
            all_results.append(res)
            print(f"    [{model.name}] Acc: {res['Accuracy']:.1%} | Toks: {res['Total Tokens']:.0f} | Lat: {res['Avg Latency']:.2f}s")
            
    df = pd.DataFrame(all_results)
    print("\n--- Final Aggregated Results ---")
    print(df.groupby(["Dataset", "Model"])[["Accuracy", "Total Tokens"]].mean())
    
    # Visualization
    plot_pareto_multi(df)
    plot_latency_multi(df)

def plot_pareto_multi(df):
    datasets = df["Dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    
    if len(datasets) == 1: axes = [axes]
    
    markers = {"Standard ToG": "s", "FrugalGPT": "^", "Tiered-Forest": "o"}
    colors = {"Standard ToG": "#e74c3c", "FrugalGPT": "#3498db", "Tiered-Forest": "#27ae60"}
    
    for ax, ds in zip(axes, datasets):
        subset = df[df["Dataset"] == ds]
        
        for _, row in subset.iterrows():
            ax.scatter(
                row["Total Tokens"], 
                row["Accuracy"], 
                s=150, 
                label=row["Model"],
                marker=markers.get(row["Model"], "o"),
                color=colors.get(row["Model"], "gray"),
                edgecolors="black"
            )
            ax.text(
                row["Total Tokens"] * 1.1, 
                row["Accuracy"], 
                f"{row['Model']}",
                fontsize=8
            )
            
        ax.set_xscale("log")
        ax.set_xlabel("Tokens (Log)")
        ax.set_title(f"Pareto Frontier: {ds}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if ds == datasets[0]: ax.set_ylabel("Accuracy")

    # Unified Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig("pareto_multi_dataset.png", dpi=300, bbox_inches="tight")
    print("Saved pareto_multi_dataset.png")

def plot_latency_multi(df):
    datasets = df["Dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1: axes = [axes]
    
    for ax, ds in zip(axes, datasets):
        subset = df[df["Dataset"] == ds]
        models = subset["Model"]
        
        # Stacked Data
        t_sym = subset["Symbolic Time"].values
        t_sml = subset["Small Time"].values
        t_lrg = subset["Large Time"].values
        
        ax.bar(models, t_sym, color="#95a5a6", label="Symbolic")
        ax.bar(models, t_sml, bottom=t_sym, color="#f1c40f", label="Small Model")
        ax.bar(models, t_lrg, bottom=t_sym+t_sml, color="#2c3e50", label="Large Model")
        
        ax.set_title(f"Latency: {ds}")
        ax.set_ylabel("Seconds")
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig("latency_multi_dataset.png", dpi=300, bbox_inches="tight")
    print("Saved latency_multi_dataset.png")

if __name__ == "__main__":
    run_benchmark()
