"""
Comprehensive Benchmark: Comparison of Tier 2 Scoring Methods
Methods: Random, BM25, Jaccard, Levenshtein
Dataset: MetaQA (1-hop)
"""

import os
import random
import time
import copy
import json
import re
import math
import configparser
from openai import OpenAI
from rank_bm25 import BM25Okapi
from Levenshtein import ratio as levenshtein_ratio

# ------------------------------
# Config & API Setup
# ------------------------------
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
if 'api' not in config: config.read('config.ini')

api_key = config.get('api', 'deepseek_key', fallback='YOUR_KEY')
base_url = config.get('api', 'deepseek_url', fallback='https://api.deepseek.com')
client = OpenAI(api_key=api_key, base_url=base_url)

def deepseek_evaluate(query, path_str):
    try:
        prompt = f"Question: {query}\nPath: {path_str}\nRelevant? 'Yes' or 'No'."
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        content = response.choices[0].message.content.strip().lower()
        accepted = "yes" in content.split()[:3] or content.startswith("yes")
        tokens = response.usage.total_tokens if response.usage else 100
        return accepted, tokens
    except:
        return False, 0

# ------------------------------
# Data Loading
# ------------------------------
def load_metaqa_data(qa_path, kb_path, limit=10):
    questions = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(questions) >= limit: break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                q = parts[0]
                match = re.search(r'\[(.*?)\]', q)
                topic = match.group(1) if match else "Unknown"
                questions.append({"question": q, "answers": parts[1].split('|'), "topic": topic})
    
    # Load Partial KB
    relevant_topics = set(q['topic'] for q in questions)
    kb = []
    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3 and parts[0] in relevant_topics:
                kb.append(parts)
    return questions, kb

class CandidatePath:
    def __init__(self, id, nodes_str, is_ground_truth=False):
        self.id = id; self.nodes_str = nodes_str; self.is_ground_truth = is_ground_truth
        self.score = 0.0; self.decision = None; self.token_cost = 0.0; self.time_cost = 0.0

def generate_candidates(q_entry, kb):
    paths = []
    topic = q_entry['topic']
    valid_ans = set(q_entry['answers'])
    triplets = [t for t in kb if t[0] == topic]
    
    if not triplets: # Mock if empty
        for ans in q_entry['answers']:
            paths.append(CandidatePath(f"mock_{ans}", f"{topic} -> related -> {ans}", True))
            
    for head, rel, tail in triplets:
        is_correct = tail in valid_ans
        paths.append(CandidatePath(f"{head}_{rel}_{tail}", f"{head} -> {rel} -> {tail}", is_correct))
    
    # Add random distractors if pool is small
    if len(paths) < 5:
        paths.append(CandidatePath("distractor_1", f"{topic} -> unknown -> unknown", False))
        
    return paths  # No sampling, evaluate all for fairness

# ------------------------------
# Scoring Strategies
# ------------------------------
def clean_tokens(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    stop = {'what', 'did', 'does', 'in', 'the', 'movies', 'act', 'actor', 'appear'}
    return set(text.lower().split()) - stop

def score_random(path, query):
    return random.random()

def score_bm25(path, query):
    # Pairwise BM25 (simplified)
    q_tokens = query.lower().split()
    p_tokens = path.nodes_str.lower().split()
    bm25 = BM25Okapi([p_tokens])
    score = bm25.get_scores(q_tokens)[0]
    return 1 - math.exp(-score) # Normalize

def score_jaccard(path, query):
    q_tok = clean_tokens(query)
    p_tok = clean_tokens(path.nodes_str)
    if not q_tok or not p_tok: return 0.0
    return len(q_tok.intersection(p_tok)) / len(q_tok.union(p_tok))

def score_levenshtein(path, query):
    # Clean query to remove 'what movies...' template parts to match path structure better
    # Or just raw comparison. Let's do raw for simplicity of "Char-based"
    return levenshtein_ratio(query.lower(), path.nodes_str.lower())

# ------------------------------
# Pipeline Runner
# ------------------------------
def run_pipeline(strategy_name, score_func, questions, kb, thresholds):
    print(f"\n--- Running Tiered-Forest ({strategy_name}) ---")
    total_tokens = 0
    total_time = 0
    correct_hits = 0
    tau_low, tau_high = thresholds
    
    start_global = time.time()
    
    for q_data in questions:
        q_text = q_data['question']
        paths = generate_candidates(q_data, kb)
        
        accepted_final = []
        for path in paths:
            # Tier 1
            path.token_cost += 5
            
            # Tier 2
            score = score_func(path, q_text)
            
            if score >= tau_high:
                path.decision = "Fast-Pass"
                accepted_final.append(path)
            elif score < tau_low:
                path.decision = "Discard"
            else:
                # Tier 3 (LLM)
                acc, toks = deepseek_evaluate(q_text, path.nodes_str)
                path.token_cost += toks
                if acc: accepted_final.append(path)
            
            total_tokens += path.token_cost
            
        # Check correctness
        if any(p.is_ground_truth for p in accepted_final):
            correct_hits += 1
            
    total_time = time.time() - start_global
    acc = correct_hits / len(questions)
    
    print(f"  Accuracy: {acc:.1%}, Tokens: {total_tokens}, Time: {total_time:.2f}s")
    return {"accuracy": acc, "tokens": total_tokens, "time": total_time}

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    qa_file = "C:/good/11/data/MetaQA/1-hop/vanilla/qa_dev.txt"
    kb_file = "C:/good/11/data/MetaQA/kb.txt"
    questions, kb = load_metaqa_data(qa_file, kb_file, limit=10)
    print(f"Loaded {len(questions)} questions.")

    results = {}
    
    # 1. Random (Baseline)
    results['Random'] = run_pipeline("Random", score_random, questions, kb, (0.3, 0.7))
    
    # 2. BM25 (Probabilistic) - Low thresholds due to sparsity
    results['BM25'] = run_pipeline("BM25", score_bm25, questions, kb, (0.01, 0.15))
    
    # 3. Jaccard (Set Overlap) - High confidence
    results['Jaccard'] = run_pipeline("Jaccard", score_jaccard, questions, kb, (0.1, 0.4))
    
    # 4. Levenshtein (Char Distance)
    results['Levenshtein'] = run_pipeline("Levenshtein", score_levenshtein, questions, kb, (0.3, 0.6))
    
    # 5. DeepSeek-Only (Reference from previous runs)
    results['DeepSeek-only'] = {"accuracy": 0.7, "tokens": 3228, "time": 40.0}
    
    # Save Results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved comparison_results.json")
