import csv
import sys
import ast

csv.field_size_limit(sys.maxsize)

try:
    with open('c:/good/11/final_results/comprehensive_results.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['experiment'] == 'A2_Logistics' or row['experiment'] == 'E0_MetaQA':
                try:
                    print(f"--- Experiment: {row['experiment']} ---")
                    details = ast.literal_eval(row['details'])
                    for i, d in enumerate(details):
                        if not d.get('correct', False):
                            print(f"Sample {i}:")
                            print(f"Q: {d.get('question', 'N/A')}")
                            print(f"GT: {d.get('ground_truth', 'N/A')}")
                            print(f"Pred: {d.get('prediction', 'N/A')}")
                            print("-" * 10)
                            if i >= 2: break 
                except Exception as e:
                    print(f"Error: {e}")
except Exception as e:
    print(e)
