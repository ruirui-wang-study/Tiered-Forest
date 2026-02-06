
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from experiments.phase1_ablation.code.ablation_experiment import generate_ablation_report

def main():
    results_dir = 'experiments/phase2_baseline/results'
    csv_path = os.path.join(results_dir, 'baseline_results_final.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading results from {csv_path}...")
        df = pd.read_csv(csv_path)
        generate_ablation_report(df, results_dir)
        print("Report regenerated successfully.")
    else:
        print(f"Error: Results file not found at {csv_path}")

if __name__ == "__main__":
    main()
