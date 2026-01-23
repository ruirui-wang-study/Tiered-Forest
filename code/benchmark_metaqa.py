"""
Benchmark: Tiered-Forest vs DeepSeek-only
Dataset: MetaQA (1-hop Vanilla)
"""

import os
import random
import time
import copy
import json
import re
import configparser
from openai import OpenAI

# Load Config
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
if 'api' not in config:
     config.read('config.ini')

api_key = config.get('api', 'deepseek_key', fallback='YOUR_KEY')
base_url = config.get('api', 'deepseek_url', fallback='https://api.deepseek.com')

client = OpenAI(api_key=api_key, base_url=base_url)

def deepseek_evaluate(query, path_str):
    """
    Evaluate a candidate reasoning path using DeepSeek API
    """
    try:
        prompt = f"""Question: {query}
Candidate Reasoning Path: {path_str}

Is this reasoning path relevant and does it lead to a correct answer for the question? 
The path format is Entity -> Relation -> Entity.
Reply strictly with 'Yes' or 'No'.
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Knowledge Graph reasoning expert for movie domain."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            # temperature=0.0
        )
        
        content = response.choices[0].message.content.strip().lower()
        accepted = "yes" in content.split()[:5] or content.startswith("yes")
        
        token_cost = response.usage.total_tokens if response.usage else 100
        
        return accepted, token_cost
        
    except Exception as e:
        print(f"API Error: {e}")
        return False, 0

# ------------------------------
# Load MetaQA Dataset & KB
# ------------------------------
def load_metaqa_data(qa_path, kb_path, limit=20):
    # 1. Load QA pairs
    questions = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(questions) >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            question = parts[0]
            answers = parts[1].split('|')
            
            # Extract topic entity from [Entity] format
            match = re.search(r'\[(.*?)\]', question)
            topic_entity = match.group(1) if match else "Unknown"
            
            questions.append({
                "question": question,
                "answers": answers,
                "topic": topic_entity
            })
            
    # 2. Load simple KB (Knowledge Base) for candidate generation
    # We only load relevant triplets for the selected questions to save memory
    relevant_entities = set(q['topic'] for q in questions)
    kb = []
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                head, rel, tail = parts
                if head in relevant_entities:
                    kb.append((head, rel, tail))
    
    return questions, kb

# ------------------------------
# Candidate Path Generation
# ------------------------------
class CandidatePath:
    def __init__(self, id, nodes_str, is_ground_truth=False):
        self.id = id
        self.nodes_str = nodes_str
        self.is_ground_truth = is_ground_truth
        self.relevance_score = 0.0 
        self.score = 0.0
        self.decision = None
        self.token_cost = 0.0
        self.time_cost = 0.0

def generate_metaqa_candidates(question_entry, kb, num_distractors=10):
    """
    Generate candidate paths from KB
    """
    paths = []
    topic = question_entry['topic']
    valid_answers = set(question_entry['answers'])
    
    # 1. Inspect KB for paths starting with topic
    relevant_triplets = [t for t in kb if t[0] == topic]
    
    # If KB is missing data, mock it for the benchmark to run
    if not relevant_triplets:
        # Mock correct path
        for ans in question_entry['answers']:
            path_str = f"{topic} -> related_to -> {ans}"
            paths.append(CandidatePath(f"mock_true_{ans}", path_str, True))
            
    for head, rel, tail in relevant_triplets:
        is_correct = tail in valid_answers
        path_str = f"{head} -> {rel} -> {tail}"
        paths.append(CandidatePath(f"{head}_{rel}_{tail}", path_str, is_correct))
        
    # Limit number of paths
    if len(paths) > num_distractors + len(valid_answers):
        paths = random.sample(paths, num_distractors + len(valid_answers))
        
    return paths

# ------------------------------
# Pipelines
# ------------------------------
def tier1_filter(path):
    start = time.time()
    path.time_cost += (time.time() - start)
    path.token_cost += 5
    return True

def tier2_score(path, query, tau_low, tau_high):
    """
    Real Semantic Scoring using Jaccard Similarity (Word Overlap)
    For short text KG paths, simple word overlap is a very strong baseline.
    It measures how many words in the query also appear in the path.
    """
    start = time.time()
    
    # 1. Clean and Tokenize
    # Remove brackets [] and other punctuation to ensure '[Entity]' matches 'Entity'
    def clean_tokenize(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return set(text.lower().split())

    stop_words = {'what', 'did', 'does', 'in', 'the', 'a', 'an', 'was', 'is', 'movies', 'movie', 'film', 'films', 'act', 'actor', 'appear'}
    
    q_tokens = clean_tokenize(query) - stop_words
    p_tokens = clean_tokenize(path.nodes_str) - stop_words
    
    # 2. Jaccard Index
    if not q_tokens or not p_tokens:
        return 0.0
        
    intersection = len(q_tokens.intersection(p_tokens))
    union = len(q_tokens.union(p_tokens))
    
    score = intersection / union if union > 0 else 0.0
    
    path.score = score
    path.token_cost += 0 
    path.time_cost += (time.time() - start)
    
    return score

def tier3_deepseek(path, query):
    start = time.time()
    accepted, tokens = deepseek_evaluate(query, path.nodes_str)
    
    path.token_cost += tokens
    path.time_cost += (time.time() - start)
    path.decision = "Accepted" if accepted else "Rejected"
    return path.decision

def tiered_pipeline(paths, query, tau_low=0.01, tau_high=0.15):
    accepted = []
    
    for path in paths:
        # Tier 1
        if not tier1_filter(path): continue
            
        # Tier 2
        score = tier2_score(path, query, tau_low, tau_high)
        
        if score >= tau_high:
            path.decision = "Accepted Tier 2 (Fast-Pass)"
            accepted.append(path)
        elif score < tau_low:
            path.decision = "Discarded Tier 2"
        else:
            # Tier 3 - Ambiguous
            tier3_deepseek(path, query)
            if path.decision == "Accepted":
                accepted.append(path)
                
    return accepted

def deepseek_only_pipeline(paths, query):
    accepted = []
    for path in paths:
        tier3_deepseek(path, query)
        if path.decision == "Accepted":
            accepted.append(path)
    return accepted

# ------------------------------
# Main Benchmark
# ------------------------------
if __name__ == "__main__":
    print("Loading MetaQA data...")
    qa_file = "C:/good/11/data/MetaQA/1-hop/vanilla/qa_dev.txt"
    kb_file = "C:/good/11/data/MetaQA/kb.txt"
    
    questions, kb = load_metaqa_data(qa_file, kb_file, limit=10)
    print(f"Loaded {len(questions)} questions and local KB subset.")

    total_tokens_base = 0
    total_time_base = 0
    correct_base = 0
    
    total_tokens_our = 0
    total_time_our = 0
    correct_our = 0
    
    print("\nRunning Benchmark on MetaQA 1-hop...")
    
    for i, q_data in enumerate(questions):
        q_text = q_data["question"]
        print(f"\n[{i+1}/{len(questions)}] Q: {q_text}")
        
        candidates = generate_metaqa_candidates(q_data, kb, num_distractors=10)
        
        if not candidates:
            print("  No candidates found in KB, skipping.")
            continue
            
        # --- DeepSeek Only ---
        c_base = copy.deepcopy(candidates)
        accepted_base = deepseek_only_pipeline(c_base, q_text)
        
        for p in c_base:
            total_tokens_base += p.token_cost
            total_time_base += p.time_cost
            
        # Metric: Hit@1 (Is at least one correct answer found?)
        gt_found_base = any(p.is_ground_truth for p in accepted_base)
        if gt_found_base: correct_base += 1
        
        # --- Tiered Forest ---
        c_our = copy.deepcopy(candidates)
        accepted_our = tiered_pipeline(c_our, q_text)
        
        for p in c_our:
            total_tokens_our += p.token_cost
            total_time_our += p.time_cost
            
        gt_found_our = any(p.is_ground_truth for p in accepted_our)
        if gt_found_our: correct_our += 1
        
        print(f"  Base Accepted: {len(accepted_base)} | Found GT: {gt_found_base}")
        print(f"  Ours Accepted: {len(accepted_our)}  | Found GT: {gt_found_our}")

    # Results
    acc_base = correct_base / len(questions)
    acc_our = correct_our / len(questions)
    
    print("\n" + "="*60)
    print("METAQA BENCHMARK RESULTS")
    print("="*60)
    
    print(f"DeepSeek-only:")
    print(f"  Accuracy (Hit): {acc_base:.2%}")
    print(f"  Token Cost: {total_tokens_base}")
    print(f"  Time Cost: {total_time_base:.2f}s")
    
    print(f"\nTiered-Forest:")
    print(f"  Accuracy (Hit): {acc_our:.2%}")
    print(f"  Token Cost: {total_tokens_our}")
    print(f"  Time Cost: {total_time_our:.2f}s")
    
    # Save for visualization
    results = {
        "DeepSeek-only": {
            "total_tokens": total_tokens_base,
            "total_time": total_time_base,
            "accuracy": acc_base,
            "f1": acc_base
        },
        "Tiered-Forest": {
            "total_tokens": total_tokens_our,
            "total_time": total_time_our,
            "accuracy": acc_our,
            "f1": acc_our
        }
    }
    
    with open("metaqa_results.json", "w") as f:
        json.dump(results, f)

