"""
Visualization for Comparison of Tier 2 Models
"""
import matplotlib.pyplot as plt
import json
import numpy as np

# Load Results
with open("comparison_results.json", "r") as f:
    results = json.load(f)

models = list(results.keys())
accuracy = [results[m]['accuracy']*100 for m in models]
tokens = [results[m]['tokens'] for m in models]

# ------------------------------
# Visualization
# ------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

# Plot Accuracy (Bar)
bars = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#3498DB', alpha=0.9, edgecolor='black')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2C3E50')
ax1.set_ylim(0, 110)
ax1.set_title('Tier 2 Scoring Strategy Comparison (MetaQA)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add values
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot Token Cost (Line)
ax2 = ax1.twinx()
line = ax2.plot(x, tokens, label='Token Cost', color='#E74C3C', marker='o', linewidth=3, markersize=8)
ax2.set_ylabel('Token Consumption', fontsize=12, fontweight='bold', color='#C0392B')
ax2.set_ylim(0, max(tokens)*1.2)

# Add token values
for i, v in enumerate(tokens):
    ax2.text(i + 0.1, v, str(int(v)), color='#C0392B', fontweight='bold', fontsize=10)

# Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("✓ Saved model_comparison.png")
