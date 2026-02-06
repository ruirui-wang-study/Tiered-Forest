
import os
import json
import csv
import random
import pandas as pd
from typing import List, Dict, Any

class BaseLoader:
    def load(self, limit=20) -> List[Dict[str, Any]]:
        raise NotImplementedError

class MetaQALoader(BaseLoader):
    def __init__(self, base_dir):
        # qa_test.txt is missing, use qa_dev.txt instead
        self.qa_path = os.path.join(base_dir, "MetaQA", "1-hop", "vanilla", "qa_dev.txt")
        
    def load(self, limit=20):
        questions = []
        if os.path.exists(self.qa_path):
            with open(self.qa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(questions) >= limit: break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        questions.append({
                            "dataset": "MetaQA",
                            "question": parts[0],
                            "ground_truth": parts[1].split('|')[0] # Take first valid answer
                        })
        return questions

class WebQSPLoader(BaseLoader):
    def __init__(self, base_dir):
        self.path = os.path.join(base_dir, "WebQSP", "data", "WebQSP.test.json")
        
    def load(self, limit=20):
        questions = []
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Random sample to avoid bias if ordered
                # But for reproducibility let's take first N unique
                candidates = data['Questions']
                
                for item in candidates:
                    if len(questions) >= limit: break
                    q_text = item['RawQuestion']
                    # Get answers
                    answers = []
                    for parse in item['Parses']:
                        for ans in parse['Answers']:
                            if ans['AnswerType'] == 'Entity':
                                answers.append(ans['EntityName'])
                            elif ans['AnswerType'] == 'Value':
                                answers.append(ans['AnswerArgument'])
                    
                    if answers:
                        questions.append({
                            "dataset": "WebQSP",
                            "question": q_text,
                            "ground_truth": answers[0] # Take first valid
                        })
        return questions

class LogisticsLoader(BaseLoader):
    def __init__(self, base_dir):
        # Folder is named 'Logistics Dataset'
        self.path = os.path.join(base_dir, "Logistics Dataset", "dynamic_supply_chain_logistics_dataset.csv")
        
    def load(self, limit=20):
        questions = []
        if os.path.exists(self.path):
            df = pd.read_csv(self.path)
            # Sample random rows
            sample_df = df.sample(n=limit, random_state=42)
            
            for _, row in sample_df.iterrows():
                # Formulate a Risk Prediction Question
                # "What is the risk classification for a shipment with fuel_consumption=X, weather_severity=Y?"
                # To make it "Logical", we include some key features.
                
                q_text = (f"What is the risk classification for a shipment with "
                          f"weather severity {row['weather_condition_severity']:.2f}, "
                          f"traffic congestion {row['traffic_congestion_level']:.2f}, "
                          f"and supplier reliability {row['supplier_reliability_score']:.2f}?")
                
                questions.append({
                    "dataset": "Logistics",
                    "question": q_text,
                    "ground_truth": str(row['risk_classification'])
                })
        return questions

def load_all_datasets(base_dir, limit_per_dataset=10):
    loaders = [
        MetaQALoader(base_dir),
        WebQSPLoader(base_dir),
        LogisticsLoader(base_dir)
    ]
    
    all_data = []
    for loader in loaders:
        try:
            data = loader.load(limit=limit_per_dataset)
            all_data.extend(data)
            print(f"Loaded {len(data)} samples from {loader.__class__.__name__}")
        except Exception as e:
            print(f"Error loading {loader.__class__.__name__}: {e}")
            
    return all_data
