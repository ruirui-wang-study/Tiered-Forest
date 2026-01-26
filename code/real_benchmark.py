
import os
import time
import json
import configparser
import numpy as np
import pandas as pd
from openai import OpenAI
import httpx

# ---------- Config & Setup ----------
config = configparser.ConfigParser()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.ini')
config.read(config_path)

if 'api' not in config:
    print(f"Error: 'api' section not found in {config_path}")
    exit(1)

api_key = config.get('api', 'deepseek_key')
base_url = config.get('api', 'deepseek_url')

# Self-signed cert workaround
http_client = httpx.Client(verify=False)
client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

# ---------- Data Loading (MetaQA Wrapper) ----------
def load_metaqa_data(limit=10):
    qa_path = os.path.join(base_dir, "data", "MetaQA", "1-hop", "vanilla", "qa_dev.txt")
    kb_path = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    
    questions = []
    if os.path.exists(qa_path):
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(questions) >= limit: break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    q = parts[0]
                    # Extract topic [Entity]
                    import re
                    match = re.search(r'\[(.*?)\]', q)
                    topic = match.group(1).strip() if match else "Unknown"
                    questions.append({"question": q, "answers": parts[1].split('|'), "topic": topic})
    
    kb = []
    if os.path.exists(kb_path):
        relevant_topics = set(q['topic'] for q in questions)
        with open(kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                # Check Head or Tail
                if len(parts) == 3:
                    if parts[0] in relevant_topics or parts[2] in relevant_topics:
                        kb.append(parts)

    print(f"Loaded {len(kb)} KB triplets.")
    return questions, kb

def get_candidates(q_data, kb):
    topic = q_data['topic']
    candidates = []
    for h, r, t in kb:
        if h == topic:
            candidates.append(f"{h} -> {r} -> {t}")
        elif t == topic:
            candidates.append(f"{t} <- {r} <- {h}")
    return list(set(candidates))

# ---------- Scorers ----------

def get_jaccard_score(query, candidates):
    if not candidates: return []
    scores = []
    
    def clean(text):
        import re
        text = re.sub(r'[^\w\s]', '', text)
        return set(text.lower().split())
        
    q_toks = clean(query)
    for c in candidates:
        c_toks = clean(c)
        if not q_toks or not c_toks:
            scores.append(0.0)
            continue
        intersection = len(q_toks.intersection(c_toks))
        union = len(q_toks.union(c_toks))
        scores.append(intersection / union)
    return scores

def call_llm_batch(query, candidates):
    # Determine the most plausible answer from a list
    if not candidates: return None, 0, 0
    
    # Cost optimization: Batch selection
    prompt = f"Question: {query}\nCandidates:\n"
    for i, c in enumerate(candidates):
        prompt += f"{i}. {c}\n"
    prompt += "\nSelect the index of the correct path. Return only the number. If none, return -1."
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        # Parse index
        import re
        match = re.search(r'-?\d+', content)
        idx = int(match.group()) if match else -1
        
        latency = time.time() - start
        tokens = response.usage.total_tokens
        
        if 0 <= idx < len(candidates):
            return candidates[idx], tokens, latency
        return None, tokens, latency
    except Exception as e:
        print(f"LLM Error: {e}")
        return None, 0, 0

def check_answer(pred_path, gold_answers):
    if not pred_path: return False
    # Path: Head -> Rel -> Tail or Tail <- Rel <- Head
    parts = pred_path.split(' -> ') if ' -> ' in pred_path else pred_path.split(' <- ')
    for p in parts:
        clean_p = p.strip()
        # Loose check
        for g in gold_answers:
            if clean_p.lower() == g.strip().lower() or g.strip().lower() in clean_p.lower():
                return True
    return False

# ---------- Strategies ----------

def run_tog_real(questions, kb):
    # ToG Baseline: "Verify All" (Expensive)
    print("\n>>> Running Real ToG (Verify All)...")
    stats = {"acc": 0, "tokens": 0, "latency": 0.0}
    
    for i, q in enumerate(questions):
        print(f"  Q{i+1}...", end="\r")
        cands = get_candidates(q, kb)
        if not cands: continue
        
        # Limit to 10 for speed
        if len(cands) > 10: cands = cands[:10]
        
        ans, toks, lat = call_llm_batch(q["question"], cands)
        stats["tokens"] += toks
        stats["latency"] += lat
        
        if check_answer(ans, q["answers"]):
             stats["acc"] += 1
                 
    return stats

def run_frugal_real(questions, kb):
    # FrugalGPT: Small Model (Jaccard) -> Threshold -> LLM
    print("\n>>> Running Real FrugalGPT (Jaccard + LLM)...")
    stats = {"acc": 0, "tokens": 0, "latency": 0.0}
    
    for i, q in enumerate(questions):
        print(f"  Q{i+1}...", end="\r")
        cands = get_candidates(q, kb)
        if not cands: continue
        
        t0 = time.time()
        scores = get_jaccard_score(q["question"], cands)
        stats["latency"] += (time.time() - t0)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Threshold 0.3
        if best_score > 0.3: 
            ans = cands[best_idx]
            if check_answer(ans, q["answers"]): stats["acc"] += 1
        else:
            # Stage 2: Rerank Top 5
            top_indices = np.argsort(scores)[-5:]
            top_cands = [cands[i] for i in top_indices]
            
            ans, toks, lat = call_llm_batch(q["question"], top_cands)
            stats["tokens"] += toks
            stats["latency"] += lat
            
            if check_answer(ans, q["answers"]): stats["acc"] += 1
                 
    return stats

def run_tiered_forest_real(questions, kb):
    print("\n>>> Running Real Tiered-Forest (Jaccard + LLM)...")
    stats = {"acc": 0, "tokens": 0, "latency": 0.0}
    
    for i, q in enumerate(questions):
        print(f"  Q{i+1}...", end="\r")
        cands = get_candidates(q, kb)
        if not cands: continue
        
        t0 = time.time()
        scores = get_jaccard_score(q["question"], cands)
        stats["latency"] += (time.time() - t0)
        
        pass_cands = []
        ambiguous_cands = []
        
        for idx, s in enumerate(scores):
            if s > 0.3: pass_cands.append(cands[idx])
            elif s > 0.1: ambiguous_cands.append(cands[idx])
            
        final_ans = None
        
        if pass_cands:
            # Fast Pass
            best_in_pass = np.argmax([scores[cands.index(c)] for c in pass_cands])
            final_ans = pass_cands[best_in_pass]
        elif ambiguous_cands:
            # Ambiguity Zone
            ans, toks, lat = call_llm_batch(q["question"], ambiguous_cands)
            stats["tokens"] += toks
            stats["latency"] += lat
            final_ans = ans
            
        if check_answer(final_ans, q["answers"]): stats["acc"] += 1
             
    return stats

# ---------- Main Execution ----------
if __name__ == "__main__":
    limit = 20
    print(f"Loading {limit} questions from MetaQA...")
    questions, kb = load_metaqa_data(limit=limit)
    print(f"Loaded {len(questions)} questions.")
    
    results = []
    
    # 1. ToG
    res_tog = run_tog_real(questions, kb)
    results.append({"Model": "Standard ToG", **res_tog})
    print(f"ToG: {res_tog}")

    # 2. FrugalGPT
    res_frugal = run_frugal_real(questions, kb)
    results.append({"Model": "FrugalGPT", **res_frugal})
    print(f"Frugal: {res_frugal}")

    # 3. Tiered-Forest
    res_tf = run_tiered_forest_real(questions, kb)
    results.append({"Model": "Tiered-Forest", **res_tf})
    print(f"TF: {res_tf}")
    
    # Save
    df = pd.DataFrame(results)
    df["Accuracy"] = df["acc"] / limit
    df["Avg Tokens"] = df["tokens"] / limit
    df["Avg Latency"] = df["latency"] / limit
    
    print("\n--- Real Benchmark Results ---")
    print(df[["Model", "Accuracy", "Avg Tokens", "Avg Latency"]])
    df.to_csv("real_benchmark_results.csv", index=False)
