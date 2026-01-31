import csv
import sys
import ast

csv.field_size_limit(sys.maxsize)

try:
    with open('c:/good/11/baseline_results/baseline_results_final.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['experiment'] == 'B2_MetaQA':
                try:
                    details = ast.literal_eval(row['details'])
                    with open('c:/good/11/debug_failed_sample.txt', 'w', encoding='utf-8') as out:
                        out.write(f"Total samples: {len(details)}\n")
                        for i, d in enumerate(details):
                            if not d.get('correct', False):
                                out.write(f"Sample {i}:\n")
                                out.write(f"Question: {d.get('question', 'N/A')}\n")
                                out.write(f"Ground Truth: {d.get('ground_truth', 'N/A')}\n")
                                out.write(f"Prediction: {d.get('prediction', 'N/A')}\n")
                                out.write(f"Full Dict: {d}\n")
                                out.write("-" * 20 + "\n")
                                if i >= 4: break 
                except Exception as e:
                    with open('c:/good/11/debug_failed_sample.txt', 'w', encoding='utf-8') as out:
                        out.write(f"Error parsing details: {e}\n")
                        out.write(f"Raw start: {row['details'][:500]}")
                break
except Exception as e:
    with open('c:/good/11/debug_failed_sample.txt', 'w') as out:
        out.write(str(e))
