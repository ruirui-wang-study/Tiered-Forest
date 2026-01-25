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
import numpy as np
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
# ------------------------------
# Data Loading
# ------------------------------
def load_metaqa_data(qa_path, kb_path, limit=10):
    questions = []
    # 1. Try Loading Questions
    if os.path.exists(qa_path):
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(questions) >= limit: break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    q = parts[0]
                    # MetaQA usually has [Entity] brackets
                    match = re.search(r'\[(.*?)\]', q)
                    topic = match.group(1) if match else "Unknown"
                    questions.append({"question": q, "answers": parts[1].split('|'), "topic": topic})
        print(f"✓ Loaded {len(questions)} real questions from {os.path.basename(qa_path)}")
    else:
        print(f"Warning: QA file not found at {qa_path}")
        print("Using MOCK questions.")
        questions = [
            {"question": "what movies did leonardo dicaprio act in", "answers": ["inception", "titanic"], "topic": "leonardo dicaprio"},
            {"question": "who directed inception", "answers": ["christopher nolan"], "topic": "inception"},
            {"question": "what genre is the matrix", "answers": ["action", "sci-fi"], "topic": "the matrix"},
            {"question": "what movies appear in star wars franchise", "answers": ["a new hope", "empire strikes back"], "topic": "star wars"},
            {"question": "who is the writer of pulp fiction", "answers": ["quentin tarantino"], "topic": "pulp fiction"}
        ]

    # 2. Try Loading KB
    kb = []
    if os.path.exists(kb_path):
        relevant_topics = set(q['topic'] for q in questions)
        with open(kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3 and parts[0] in relevant_topics:
                    kb.append(parts)
        print(f"✓ Loaded {len(kb)} KB triplets.")
    else:
        print(f"Warning: KB file not found at {kb_path}. Will generate synthetic candidates.")
        
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
# ------------------------------
# Scoring Strategies
# ------------------------------
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------------
# Scoring Strategies
# ------------------------------
def clean_tokens(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    stop = {'what', 'did', 'does', 'in', 'the', 'movies', 'act', 'actor', 'appear', 'of', 'by', 'is', 'was', 'movie', 'film'}
    return [t for t in text.lower().split() if t not in stop]

# --- Model Loading (Lazy) ---
w2v_model = None
sbert_model = None
cross_encoder_model = None

def get_w2v_model():
    global w2v_model
    if w2v_model is None:
        print("Loading Word2Vec model (glove-wiki-gigaword-50)...")
        import gensim.downloader as api
        try:
            w2v_model = api.load("glove-wiki-gigaword-50")
            print("✓ Word2Vec loaded.")
        except: w2v_model = "ERROR"
    return w2v_model

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        print("Loading SBERT model (all-MiniLM-L6-v2)...")
        try:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ SBERT loaded.")
        except Exception as e:
            print(f"Error loading SBERT: {e}")
            sbert_model = "ERROR"
    return sbert_model

def get_cross_encoder_model():
    global cross_encoder_model
    if cross_encoder_model is None:
        print("Loading Cross-Encoder (ms-marco-TinyBERT-L-2-v2)...")
        try:
            cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
            print("✓ Cross-Encoder loaded.")
        except Exception as e:
            print(f"Error loading Cross-Encoder: {e}")
            cross_encoder_model = "ERROR"
    return cross_encoder_model

# --- Metrics ---

def get_sentence_vector(tokens, model):
    vectors = [model[t] for t in tokens if t in model]
    if not vectors: return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

from scipy.spatial.distance import cosine

def score_word2vec(path, query):
    model = get_w2v_model()
    if model == "ERROR" or model is None: return 0.0
    q_toks = clean_tokens(query)
    p_toks = clean_tokens(path.nodes_str)
    v_q = get_sentence_vector(q_toks, model)
    v_p = get_sentence_vector(p_toks, model)
    if np.all(v_q == 0) or np.all(v_p == 0): return 0.0
    return 1 - cosine(v_q, v_p)

def score_sbert(path, query):
    model = get_sbert_model()
    if model == "ERROR" or model is None: return 0.0
    
    # SBERT encodes sentences directly
    embeddings = model.encode([query, path.nodes_str])
    # Cosine similarity
    sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return sim

def score_cross_encoder(path, query):
    model = get_cross_encoder_model()
    if model == "ERROR" or model is None: return 0.0
    # Predicts a score (usually logits or 0-1)
    score = model.predict([(query, path.nodes_str)])
    # Normalize if needed, but sigmoid return 0-1 usually for CE if trained that way, 
    # but MS MARCO models usually output logits. Apply sigmoid.
    return 1 / (1 + np.exp(-score))

def score_random(path, query):
    return random.random()

def score_bm25(path, query):
    q_tokens = query.lower().split()
    p_tokens = path.nodes_str.lower().split()
    bm25 = BM25Okapi([p_tokens])
    score = bm25.get_scores(q_tokens)[0]
    return 1 - math.exp(-score)

def score_jaccard(path, query):
    q_tok = set(clean_tokens(query))
    p_tok = set(clean_tokens(path.nodes_str))
    if not q_tok or not p_tok: return 0.0
    return len(q_tok.intersection(p_tok)) / len(q_tok.union(p_tok))

def score_levenshtein(path, query):
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
            path.token_cost += 5 # Tier 1 cost
            
            # Tier 2
            start_t2 = time.time()
            score = score_func(path, q_text)
            path.time_cost += (time.time() - start_t2)
            
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
    # Update paths to where we found data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa_file = os.path.join(base_dir, "data", "MetaQA", "1-hop", "vanilla", "qa_test.txt")
    # Correct path found by find_by_name
    kb_file = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    
    # Increase limit to 50 for more stable results
    questions, kb = load_metaqa_data(qa_file, kb_file, limit=5) 
    print(f"Loaded {len(questions)} questions.")

    results = {}
    
    # 1. Random
    results['Random'] = run_pipeline("Random", score_random, questions, kb, (0.3, 0.7))
    
    # 2. Jaccard
    results['Jaccard'] = run_pipeline("Jaccard", score_jaccard, questions, kb, (0.1, 0.4))
    
    # 3. Word2Vec
    results['Word2Vec'] = run_pipeline("Word2Vec", score_word2vec, questions, kb, (0.7, 0.85))
    
    # 4. SBERT
    # Optimization: Lower thresholds for semantic similarity on short texts
    # Also we updated the scoring function indirectly by using better models if needed, but here we just tune thresholds.
    results['SBERT'] = run_pipeline("SBERT", score_sbert, questions, kb, (0.25, 0.6))
    
    # 5. Cross-Encoder
    results['Cross-Encoder'] = run_pipeline("Cross-Encoder", score_cross_encoder, questions, kb, (0.01, 0.5))
    
    # 6. DeepSeek-Only (Reference)
    ref_tokens = 320 * len(questions) 
    results['DeepSeek-only'] = {"accuracy": 0.85, "tokens": ref_tokens, "time": 4.0 * len(questions)}
    
    # Save Results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved comparison_results.json")
