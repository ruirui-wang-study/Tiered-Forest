
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Data Configuration
datasets = ['MetaQA', 'WebQSP', 'Logistics']
models = ['ToG', 'FrugalGPT', 'Tiered-Forest']

# Load Data from CSV
csv_path = "multi_dataset_results.csv"
# Check if file exists in current dir or subdir
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "multi_dataset_results.csv")

data = {}
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        for ds in datasets:
            data[ds] = {'Accuracy': [], 'Tokens': [], 'Cost': [], 'Latency': []}
            ds_subset = df[df['dataset'] == ds]
            for model in models:
                # Assuming model names in CSV match: ToG, FrugalGPT, Tiered-Forest
                # Note: main.py uses "ToG", "FrugalGPT", "Tiered-Forest"
                row = ds_subset[ds_subset['name'] == model]
                if not row.empty:
                    data[ds]['Accuracy'].append(row.iloc[0]['accuracy'] * 100)
                    data[ds]['Tokens'].append(row.iloc[0]['tokens'])
                    data[ds]['Cost'].append(row.iloc[0]['cost'])
                    data[ds]['Latency'].append(row.iloc[0]['latency'])
                else:
                    print(f"Warning: No data for {model} in {ds}")
                    data[ds]['Accuracy'].append(0)
                    data[ds]['Tokens'].append(0)
                    data[ds]['Cost'].append(0)
                    data[ds]['Latency'].append(0)
        print("Loaded data from CSV successfully.")
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        data = {}
else:
    print(f"CSV file not found at {csv_path}. Using placeholder data.")
    # Fallback to hardcoded for testing/safety
    data = {
        'MetaQA': {
            'Accuracy': [20.0, 30.0, 10.0],
            'Tokens': [2924, 2549, 374],
            'Cost': [0.0115, 0.0187, 0.0019],
            'Latency': [8.05, 8.78, 1.46]
        },
        'WebQSP': {
            'Accuracy': [90.0, 70.0, 75.0],
            'Tokens': [2764, 1200, 800],
            'Cost': [0.0106, 0.0048, 0.0031],
            'Latency': [7.30, 8.10, 2.15]
        },
        'Logistics': {
            'Accuracy': [70.0, 70.0, 70.0],
            'Tokens': [2105, 398, 322],
            'Cost': [0.0068, 0.0026, 0.0018],
            'Latency': [6.49, 2.11, 1.57]
        }
    }


# ACM Paper Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
})

# Academic Palette (Colorblind friendly / High Contrast)
# ToG (Baseline 1): Dark Grey/Blue
# Pastel Palette (Light & Professional)
# ToG: Pastel Blue
# FrugalGPT: Pastel Orange/Peach
# Tiered-Forest: Pastel Green
colors = ['#AEC7E8', '#FFBB78', '#98DF8A'] # Pastel versions of Blue, Orange, Green
hatches = ['//', '..', 'xx'] # For B&W printing compatibility

def plot_grouped_bar(metric, ylab, title, filename, log_scale=False):
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5)) # Standard single-column width approx
    
    for i, model in enumerate(models):
        vals = [data[ds][metric][i] for ds in datasets]
        rects = ax.bar(x + (i-1)*width, vals, width, label=model, 
                       color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5,
                       hatch=hatches[i] * 2) # Dense hatch
        
        # Label values
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                val_text = f'{height:.2f}' if isinstance(height, float) and height < 10 else f'{int(height)}'
                ax.annotate(val_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel(ylab)
    # ACM usually puts titles in captions, but we keep it for self-contained figs
    ax.set_title(title, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    
    if log_scale:
        ax.set_yscale('log')
        
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_pareto_frontier():
    # Gather all points
    all_tokens = []
    all_acc = []
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = ['o', 's', '^'] # Circle, Square, Triangle for datasets
    
    for d_idx, ds in enumerate(datasets):
        for m_idx, model in enumerate(models):
            tokens = data[ds]['Tokens'][m_idx]
            acc = data[ds]['Accuracy'][m_idx]
            
            ax.scatter(tokens, acc, color=colors[m_idx], marker=markers[d_idx], s=180, 
                       label=f"{model} ({ds})" if m_idx == 0 else "", 
                       alpha=0.9, edgecolors='black', linewidth=1.0, zorder=10)

    # Custom Legend - ACM Style (Boxed)
    from matplotlib.lines import Line2D
    
    # Dataset Markers
    legend_ds = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='MetaQA', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='WebQSP', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='Logistics', markersize=10, markeredgecolor='black'),
    ]
    
    # Model Colors
    legend_models = [
        Line2D([0], [0], color=colors[0], lw=4, label='ToG'),
        Line2D([0], [0], color=colors[1], lw=4, label='FrugalGPT'),
        Line2D([0], [0], color=colors[2], lw=4, label='Tiered-Forest'),
    ]
    
    # Combine or separate based on space. Let's combine.
    # We place legend outside to keep plot clean.
    
    ax.set_xscale('log')
    ax.set_xlabel('Token Consumption (Log Scale)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Cost-Efficiency Pareto Frontier', pad=15)
    
    # Add light annotations for Tiered-Forest points to highlight them
    # (Optional, but effective for papers)
    
    ax.grid(True, which="both", ls=":", alpha=0.4)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend1 = ax.legend(handles=legend_ds + legend_models, bbox_to_anchor=(1.05, 1), loc='upper left', 
                        borderaxespad=0., title="Legend", frameon=True, edgecolor='black')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'multi_dataset_pareto.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def main():
    # 1. Accuracy Plot
    plot_grouped_bar('Accuracy', 'Accuracy (%)', 'Model Accuracy by Dataset', 'bar_accuracy.png')
    
    # 2. Token Plot
    plot_grouped_bar('Tokens', 'Total Tokens', 'Token Consumption by Dataset (Lower is Better)', 'bar_tokens.png')
    
    # 3. Latency Plot
    plot_grouped_bar('Latency', 'Latency (s)', 'Inference Latency by Dataset (Lower is Better)', 'bar_latency.png')
    
    # 4. Pareto
    plot_pareto_frontier()

if __name__ == "__main__":
    main()
