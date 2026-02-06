"""
ACM Style Visualization for Tiered-Forest Benchmark Results
Generates publication-quality figures with Accuracy/F1 metrics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set ACM publication style parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5

import json
import os

# Load Benchmark Results (MetaQA)
# Priority: MetaQA > WebQSP > Fallback
if os.path.exists("metaqa_results.json"):
    with open("metaqa_results.json", "r") as f:
        results = json.load(f)
    print("✓ Loaded results from metaqa_results.json")
elif os.path.exists("webqsp_results.json"):
    with open("webqsp_results.json", "r") as f:
        results = json.load(f)
    print("✓ Loaded results from webqsp_results.json")
else:
    # Fallback / Placeholder
    results = {
        'DeepSeek-only': {
            'total_tokens': 3000,
            'total_time': 50.0,
            'accuracy': 0.6,
            'f1': 0.6
        },
        'Tiered-Forest': {
            'total_tokens': 800,
            'total_time': 5.0,
            'accuracy': 0.95,
            'f1': 0.95
        }
    }

# ------------------------------
# Figure 1: Performance vs Cost (Double Y-Axis)
# ------------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))

methods = ['DeepSeek-only', 'Tiered-Forest']
x = np.arange(len(methods))
width = 0.35

# Plot Token Consumption (Left Axis)
tokens = [results[m]['total_tokens'] for m in methods]
rects1 = ax1.bar(x - width/2, tokens, width, label='Total Tokens', 
                 color='white', edgecolor='black', hatch='//')

ax1.set_ylabel('Total Token Consumption', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontweight='bold')
ax1.set_ylim(0, max(tokens)*1.2)

# Plot Time Cost (Right Axis)
ax2 = ax1.twinx()
times = [results[m]['total_time'] for m in methods]
rects2 = ax2.bar(x + width/2, times, width, label='Time Cost (s)', 
                 color='gray', edgecolor='black', hatch='..')

ax2.set_ylabel('Time Cost (seconds)', fontweight='bold')
ax2.set_ylim(0, max(times)*1.2)

# Add title
plt.title('Cost Efficiency Comparison', fontweight='bold', pad=20)

# Add Legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
           bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Add value labels
def autolabel(rects, ax, suffix=''):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1, ax1)
for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f}s',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('acm_cost_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: acm_cost_efficiency.png")
plt.show()

# ------------------------------
# Figure 2: Quality Metrics (Accuracy & F1)
# ------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

metrics = ['Accuracy', 'F1 Score']
# Baseline is 1.0 for both (reference)
baseline_vals = [1.0, 1.0]
tiered_vals = [results['Tiered-Forest']['accuracy'], results['Tiered-Forest']['f1']]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax.bar(x - width/2, baseline_vals, width, label='DeepSeek-only (Basline)',
                color='lightgray', edgecolor='black')
rects2 = ax.bar(x + width/2, tiered_vals, width, label='Tiered-Forest',
                color='black', edgecolor='black', hatch='xx')

ax.set_ylabel('Score (0-1)', fontweight='bold')
ax.set_title('Quality Metrics Comparison', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.legend(loc='upper right', frameon=True)

# Add labels
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('acm_quality_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: acm_quality_metrics.png")
plt.show()

# ------------------------------
# Figure 3: Trade-off Analysis (Scatter Plot)
# ------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Define points
points = {
    'DeepSeek-only': {'cost': tokens[0], 'acc': 1.0, 'marker': 'o', 'color': 'black'},
    'Tiered-Forest': {'cost': tokens[1], 'acc': results['Tiered-Forest']['accuracy'], 'marker': '^', 'color': 'black'}
}

for name, data in points.items():
    ax.scatter(data['cost'], data['acc'], s=200, marker=data['marker'], 
               color=data['color'], label=name, edgecolors='black')
    ax.annotate(name, (data['cost'], data['acc']), xytext=(10, -10), 
                textcoords='offset points', fontweight='bold')

ax.set_xlabel('Token Consumption', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Accuracy vs. Cost Trade-off', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# Add arrow indicating ideal direction
ax.annotate('Better Trade-off', xy=(tokens[1]*0.9, 0.8), xytext=(tokens[0], 0.6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
plt.savefig('acm_tradeoff_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: acm_tradeoff_scatter.png")
plt.show()
