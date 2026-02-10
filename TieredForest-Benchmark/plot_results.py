#!/usr/bin/env python3
"""
生成 Benchmark 可视化图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_accuracy_cost():
    """准确率 vs 成本对比"""
    # 读取数据
    df = pd.read_csv('results/benchmark_summary.csv')
    
    # 估算 Naive LLM 的真实成本（无缓存）
    naive_estimated_cost = 0.0051
    
    # 准备数据
    agents = df['agent'].tolist()
    accuracy = df['accuracy'].tolist()
    costs = df['total_cost_usd'].tolist()
    
    # 替换 Naive LLM 的成本为估算值
    costs[0] = naive_estimated_cost
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 准确率对比
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars1 = ax1.bar(agents, [a*100 for a in accuracy], color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 70)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 成本对比
    bars2 = ax2.bar(agents, [c*1000 for c in costs], color=colors, alpha=0.8)
    ax2.set_ylabel('Total Cost (×$0.001)', fontsize=12, fontweight='bold')
    ax2.set_title('Cost Comparison (50 questions)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/accuracy_cost_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: results/accuracy_cost_comparison.png")
    plt.close()

def plot_tier_distribution():
    """Tiered-Forest 的层分布"""
    df = pd.read_csv('results/benchmark_summary.csv')
    
    # 获取 Tiered-Forest 的数据
    tf_row = df[df['agent'] == 'Tiered-Forest'].iloc[0]
    
    tiers = ['Tier 1\n(Symbolic)', 'Tier 2\n(Semantic)', 'Tier 3\n(LLM)']
    percentages = [tf_row['tier1_pct'], tf_row['tier2_pct'], tf_row['tier3_pct']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(tiers, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Usage Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tiered-Forest: Layer Usage Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 添加成本标注
    cost_labels = ['$0', '~$0.0001', '~$0.002']
    for i, (bar, label) in enumerate(zip(bars, cost_labels)):
        ax.text(bar.get_x() + bar.get_width()/2., 5,
                f'Cost: {label}',
                ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('results/tier_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: results/tier_distribution.png")
    plt.close()

def plot_pareto_frontier():
    """Pareto 前沿图"""
    df = pd.read_csv('results/benchmark_summary.csv')
    
    # 估算 Naive LLM 的真实成本
    naive_estimated_cost = 0.0051
    
    agents = df['agent'].tolist()
    accuracy = df['accuracy'].tolist()
    costs = df['total_cost_usd'].tolist()
    costs[0] = naive_estimated_cost
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制散点
    colors_map = {'Naive LLM': '#3498db', 'FrugalGPT': '#e74c3c', 'Tiered-Forest': '#2ecc71'}
    markers = {'Naive LLM': 'o', 'FrugalGPT': 's', 'Tiered-Forest': '^'}
    
    for agent, acc, cost in zip(agents, accuracy, costs):
        ax.scatter(cost*1000, acc*100, 
                  s=300, 
                  color=colors_map[agent], 
                  marker=markers[agent],
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=2,
                  label=agent)
    
    # 添加标签
    for agent, acc, cost in zip(agents, accuracy, costs):
        offset_x = 0.1 if agent != 'FrugalGPT' else -0.3
        offset_y = 2 if agent == 'Naive LLM' else -3
        ax.annotate(agent, 
                   (cost*1000, acc*100),
                   xytext=(cost*1000 + offset_x, acc*100 + offset_y),
                   fontsize=11,
                   fontweight='bold')
    
    ax.set_xlabel('Total Cost (×$0.001)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Accuracy vs Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 添加 Pareto 前沿线（Tiered-Forest 和 Naive LLM）
    pareto_x = [costs[2]*1000, costs[0]*1000]
    pareto_y = [accuracy[2]*100, accuracy[0]*100]
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.3, linewidth=2, label='Pareto Frontier')
    
    plt.tight_layout()
    plt.savefig('results/pareto_frontier.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存: results/pareto_frontier.png")
    plt.close()

def main():
    print("="*80)
    print("生成 Benchmark 可视化图表")
    print("="*80)
    
    # 检查数据文件
    if not os.path.exists('results/benchmark_summary.csv'):
        print("✗ 错误: 未找到 benchmark_summary.csv")
        print("  请先运行: python run_benchmark.py")
        return
    
    print("\n生成图表...")
    
    # 生成图表
    plot_accuracy_cost()
    plot_tier_distribution()
    plot_pareto_frontier()
    
    print("\n" + "="*80)
    print("所有图表已生成!")
    print("="*80)
    print("\n生成的文件:")
    print("  1. results/accuracy_cost_comparison.png")
    print("  2. results/tier_distribution.png")
    print("  3. results/pareto_frontier.png")

if __name__ == "__main__":
    main()
