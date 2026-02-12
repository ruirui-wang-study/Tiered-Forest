#!/usr/bin/env python3
"""
Visualize Benchmark Results: Naive LLM vs ToG vs Tiered-Forest
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_comprehensive_comparison():
    """Generate comprehensive comparison charts"""
    print("Reading results...")
    try:
        df = pd.read_csv("results/benchmark_summary.csv")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create output directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Define methods and colors
    methods = ['Naive LLM', 'ToG', 'Tiered-Forest']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    # Filter only relevant agents if others exist
    # If a method is not in csv, handle it gracefully
    existing_methods = [m for m in methods if m in df['agent'].values]
    if not existing_methods:
        print("No matching methods found in CSV.")
        return
        
    df = df[df['agent'].isin(existing_methods)]
    
    # Sort df by methods order
    df['agent'] = pd.Categorical(df['agent'], categories=existing_methods, ordered=True)
    df = df.sort_values('agent')
    
    colors = [dict(zip(methods, colors)).get(x, '#CCCCCC') for x in df['agent']]

    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['agent'], df['accuracy'], color=colors)
    plt.title('Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (0-1)', fontsize=12)
    plt.ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12)
        
    plt.savefig('results/plots/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved accuracy_comparison.png")
    
    # 2. Total Cost Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['agent'], df['total_cost_usd'], color=colors)
    plt.title('Total Cost Comparison (50 Questions)', fontsize=16)
    plt.ylabel('Cost (USD)', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.5f}', ha='center', va='bottom', fontsize=10)
        
    plt.savefig('results/plots/cost_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved cost_comparison.png")

    # 3. Average Latency Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['agent'], df['avg_latency_s'], color=colors)
    plt.title('Average Latency per Question', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    plt.savefig('results/plots/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved latency_comparison.png")

    # 4. LLM Calls Comparison
    if 'avg_llm_calls' in df.columns:
        plt.figure(figsize=(10, 6))
        # Fill NaN with 1 for Naive (approx) for plotting purposes if missing but we know Naive usually is 1
        # But let's check values in df
        calls_values = []
        for _, row in df.iterrows():
            val = row['avg_llm_calls']
            if pd.isna(val) and row['agent'] == 'Naive LLM':
                val = 1.0
            elif pd.isna(val):
                val = 0.0
            calls_values.append(val)
        
        bars = plt.bar(df['agent'], calls_values, color=colors)
        plt.title('Average LLM Calls per Question', fontsize=16)
        plt.ylabel('Number of Calls', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=12)
            
        plt.savefig('results/plots/llm_calls_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved llm_calls_comparison.png")

    # 5. Radar Chart (normalized)
    categories = ['Accuracy', 'Cost (Inv)', 'Speed (Inv)', 'LLM Calls (Inv)']
    
    df_norm = df.copy()
    
    # Accuracy
    max_acc = df_norm['accuracy'].max()
    df_norm['norm_acc'] = df_norm['accuracy'] / (max_acc + 1e-9)
    
    # Cost (Inverse)
    max_cost = df_norm['total_cost_usd'].max()
    df_norm['norm_cost'] = 1 - (df_norm['total_cost_usd'] / (max_cost + 1e-9))
    
    # Latency (Inverse)
    max_lat = df_norm['avg_latency_s'].max()
    df_norm['norm_lat'] = 1 - (df_norm['avg_latency_s'] / (max_lat + 1e-9))

    # LLM Calls (Inverse)
    # Fix avg_llm_calls for Naive
    if 'avg_llm_calls' in df_norm.columns:
         df_norm.loc[df_norm['agent'] == 'Naive LLM', 'avg_llm_calls'] = 1.0
         max_calls = df_norm['avg_llm_calls'].max()
         if pd.notna(max_calls) and max_calls > 0:
             df_norm['norm_calls'] = 1 - (df_norm['avg_llm_calls'].fillna(0) / max_calls)
         else:
             df_norm['norm_calls'] = 0.5
    else:
        df_norm['norm_calls'] = 0.5

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, row in df_norm.iterrows():
        try:
            values = [row['norm_acc'], row['norm_cost'], row['norm_lat'], row['norm_calls']]
            values += values[:1]
            agent_name = row['agent']
            color = dict(zip(methods, colors)).get(agent_name, '#CCCCCC')
            
            ax.plot(angles, values, label=agent_name, linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        except Exception as e:
            print(f"Error plotting radar for {row['agent']}: {e}")
            continue
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    plt.title('Comprehensive Comparison (Normalized)', fontsize=16, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig('results/plots/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved radar_comparison.png")

if __name__ == "__main__":
    plot_comprehensive_comparison()
