"""
Visualization for Tiered-Forest Benchmark Results
Generate publication-quality figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

# Benchmark results from telecom dataset
results = {
    'DeepSeek-only': {
        'accepted_paths': 1,
        'total_tokens': 2842,
        'total_time': 56.30,
        'tier1_discard': 0,
        'tier2_discard': 0,
        'tier2_fastpass': 0,
        'tier3_eval': 20  # All paths go through Tier 3
    },
    'Tiered-Forest': {
        'accepted_paths': 13,
        'total_tokens': 1528,
        'total_time': 18.19,
        'tier1_discard': 0,
        'tier2_discard': 0,
        'tier2_fastpass': 13,
        'tier3_eval': 7
    }
}

# ------------------------------
# Figure 1: Token & Time Comparison
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

methods = ['DeepSeek-only', 'Tiered-Forest']
tokens = [results[m]['total_tokens'] for m in methods]
times = [results[m]['total_time'] for m in methods]

# Subplot 1: Token Consumption
bars1 = ax1.bar(methods, tokens, color=['#E74C3C', '#27AE60'], 
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_ylabel('Total Token Consumption', fontsize=11, fontweight='bold')
ax1.set_title('(a) Token Cost Comparison', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, tokens)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add reduction percentage
reduction_pct = 100 * (1 - tokens[1] / tokens[0])
ax1.annotate(f'↓ {reduction_pct:.1f}%', 
             xy=(0.5, max(tokens) * 0.6), 
             fontsize=14, ha='center',
             color='#27AE60', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#27AE60', linewidth=2))

# Subplot 2: Time Cost
bars2 = ax2.bar(methods, times, color=['#E74C3C', '#27AE60'], 
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax2.set_ylabel('Total Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Time Cost Comparison', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, times)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add reduction percentage
time_reduction_pct = 100 * (1 - times[1] / times[0])
ax2.annotate(f'↓ {time_reduction_pct:.1f}%', 
             xy=(0.5, max(times) * 0.6), 
             fontsize=14, ha='center',
             color='#27AE60', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#27AE60', linewidth=2))

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: benchmark_comparison.png")
plt.show()

# ------------------------------
# Figure 2: Pipeline Flow Breakdown
# ------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Tiered-Forest pipeline breakdown
tier_labels = ['Tier 1\nDiscarded', 'Tier 2\nDiscarded', 'Tier 2\nFast-Pass', 'Tier 3\nEvaluated']
tier_values = [
    results['Tiered-Forest']['tier1_discard'],
    results['Tiered-Forest']['tier2_discard'],
    results['Tiered-Forest']['tier2_fastpass'],
    results['Tiered-Forest']['tier3_eval']
]
tier_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']

# Create horizontal stacked bar
left = 0
bars = []
for label, value, color in zip(tier_labels, tier_values, tier_colors):
    bar = ax.barh(0, value, left=left, color=color, edgecolor='black', 
                  linewidth=1.5, alpha=0.85, height=0.5)
    bars.append(bar)
    
    # Add text label
    if value > 0:
        ax.text(left + value/2, 0, f'{value}\npaths', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    left += value

# DeepSeek-only baseline (all go to Tier 3)
ax.barh(1, 20, color='#E74C3C', edgecolor='black', 
        linewidth=1.5, alpha=0.85, height=0.5, label='All paths → LLM')
ax.text(10, 1, '20 paths\n(All evaluated by LLM)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.set_yticks([0, 1])
ax.set_yticklabels(['Tiered-Forest\n(Ours)', 'DeepSeek-only\n(Baseline)'], fontsize=11, fontweight='bold')
ax.set_xlabel('Number of Paths', fontsize=11, fontweight='bold')
ax.set_title('Pipeline Flow Distribution (20 Candidate Paths)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 21)
ax.grid(axis='x', linestyle='--', alpha=0.4)

# Add legend
legend_elements = [
    mpatches.Patch(color='#E74C3C', label='Discarded', alpha=0.85),
    mpatches.Patch(color='#F39C12', label='Tier 2 Discarded', alpha=0.85),
    mpatches.Patch(color='#27AE60', label='Fast-Pass (No LLM)', alpha=0.85),
    mpatches.Patch(color='#3498DB', label='Tier 3 LLM', alpha=0.85)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.savefig('pipeline_flow.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pipeline_flow.png")
plt.show()

# ------------------------------
# Figure 3: Efficiency Metrics
# ------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ['Token\nReduction', 'Time\nReduction', 'Speedup\n(Tokens)', 'Accepted\nPaths']
values = [
    reduction_pct,
    time_reduction_pct,
    tokens[0] / tokens[1],
    results['Tiered-Forest']['accepted_paths']
]
colors_metrics = ['#27AE60', '#3498DB', '#9B59B6', '#F39C12']

bars = ax.bar(metrics, values, color=colors_metrics, 
              edgecolor='black', linewidth=1.5, alpha=0.85)

ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Tiered-Forest Performance Metrics', fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Add value labels
for bar, val, metric in zip(bars, values, metrics):
    height = bar.get_height()
    if 'Reduction' in metric:
        label = f'{val:.1f}%'
    elif 'Speedup' in metric:
        label = f'{val:.2f}x'
    else:
        label = f'{int(val)}'
    
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('efficiency_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: efficiency_metrics.png")
plt.show()

# ------------------------------
# Summary Statistics
# ------------------------------
print("\n" + "="*60)
print("BENCHMARK VISUALIZATION SUMMARY")
print("="*60)
print(f"\nDataset: 电信运营支撑领域术语库 (1000 terms)")
print(f"Candidate paths evaluated: 20")
print(f"\nKey Findings:")
print(f"  • Token consumption reduced by {reduction_pct:.1f}%")
print(f"  • Inference time reduced by {time_reduction_pct:.1f}%")
print(f"  • {tokens[0] / tokens[1]:.2f}x speedup in token efficiency")
print(f"  • {results['Tiered-Forest']['tier2_fastpass']} paths fast-passed (no LLM call)")
print(f"  • Only {results['Tiered-Forest']['tier3_eval']} paths required LLM evaluation")
print(f"\nGenerated 3 publication-ready figures:")
print(f"  1. benchmark_comparison.png")
print(f"  2. pipeline_flow.png")
print(f"  3. efficiency_metrics.png")
print("="*60)
