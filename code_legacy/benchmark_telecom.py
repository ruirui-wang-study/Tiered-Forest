"""
Benchmark: Tiered-Forest vs DeepSeek-only
Using Real Telecom Domain Terminology Dataset
"""

import random
import time
import copy
import pandas as pd
import numpy as np
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

def deepseek_evaluate(query, path_nodes):
    """
    Evaluate a candidate reasoning path using DeepSeek API
    """
    try:
        path_str = " -> ".join(path_nodes)
        prompt = f"""Query: {query}
Reasoning Path: {path_str}

This is a telecom domain knowledge graph reasoning task. 
Evaluate if this path is semantically coherent and relevant to answer the query.
Answer with 'Yes' or 'No' and provide a brief rationale (1-2 sentences)."""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in telecom operations and knowledge graph reasoning."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        content = response.choices[0].message.content.strip().lower()
        accepted = "yes" in content.split()[:3] or content.startswith("yes")
        
        token_cost = response.usage.total_tokens if response.usage else 150
        rationale = response.choices[0].message.content
        
        # Extract score from response (0-1)
        score = 0.8 if accepted else 0.3
        
        return accepted, score, rationale, token_cost
        
    except Exception as e:
        print(f"API Error: {e}")
        return False, 0.0, f"Error: {str(e)}", 0

# ------------------------------
# Load Telecom Terminology Dataset
# ------------------------------
def load_telecom_terms(filepath='C:/good/11/data/1.xlsx', top_n=200):
    """Load top N telecom terms based on final score"""
    df = pd.read_excel(filepath)
    # Sort by final score and take top N
    df_sorted = df.sort_values('最终得分', ascending=False).head(top_n)
    terms = df_sorted['关键词'].tolist()
    scores = df_sorted['最终得分'].tolist()
    return terms, scores

# ------------------------------
# Candidate Path Class
# ------------------------------
class CandidatePath:
    def __init__(self, path_id, nodes, relevance_score=None):
        self.id = path_id
        self.nodes = nodes
        self.relevance_score = relevance_score  # Based on term scores
        self.score = None  # Tier 2 embedding score
        self.decision = None
        self.token_cost = 0.0
        self.time_cost = 0.0
        self.rationale = ""

# ------------------------------
# Generate Realistic Paths from Telecom Terms
# ------------------------------
def generate_telecom_paths(terms, term_scores, num_paths=30, path_length=4):
    """
    Generate candidate reasoning paths using real telecom terms
    Paths with higher-scored terms are more likely to be relevant
    """
    paths = []
    for i in range(num_paths):
        # Sample terms with probability weighted by their scores
        weights = np.array(term_scores)
        weights = weights / weights.sum()
        
        selected_indices = np.random.choice(
            len(terms), 
            size=path_length, 
            replace=False,
            p=weights
        )
        
        nodes = [terms[idx] for idx in selected_indices]
        avg_score = np.mean([term_scores[idx] for idx in selected_indices])
        
        paths.append(CandidatePath(i, nodes, relevance_score=avg_score))
    
    return paths

# ------------------------------
# Tier 1: Symbolic Filter
# ------------------------------
def tier1_filter(path):
    """
    Symbolic/structural filter
    - Check for duplicate nodes
    - Check path length constraints
    """
    start_time = time.time()
    token_cost = 5
    
    # Rule 1: No duplicate nodes
    if len(set(path.nodes)) < len(path.nodes):
        path.time_cost += time.time() - start_time
        path.token_cost += token_cost
        return False
    
    # Rule 2: Path length should be reasonable
    if len(path.nodes) < 2 or len(path.nodes) > 6:
        path.time_cost += time.time() - start_time
        path.token_cost += token_cost
        return False
    
    path.time_cost += time.time() - start_time
    path.token_cost += token_cost
    return True

# ------------------------------
# Tier 2: Embedding-based Scoring
# ------------------------------
def tier2_score(path, query):
    """
    Mock embedding-based semantic scoring
    In practice, this would use actual embeddings
    Here we use the relevance_score from the dataset as a proxy
    """
    start_time = time.time()
    token_cost = random.randint(15, 35)
    
    # Use the pre-computed relevance score with some noise
    base_score = path.relevance_score if path.relevance_score else 0.5
    noise = random.uniform(-0.1, 0.1)
    score = max(0.0, min(1.0, base_score + noise))
    
    path.score = score
    path.time_cost += time.time() - start_time
    path.token_cost += token_cost
    
    return score

# ------------------------------
# Tier 3: DeepSeek LLM Evaluation
# ------------------------------
def tier3_deepseek(path, query):
    """Call DeepSeek API for deep reasoning validation"""
    start_time = time.time()
    
    accepted, score, rationale, token_cost = deepseek_evaluate(query, path.nodes)
    
    path.token_cost += token_cost
    path.time_cost += time.time() - start_time
    path.decision = "Accepted" if accepted else "Rejected"
    path.rationale = rationale
    
    return path.decision

# ------------------------------
# Tiered-Forest Pipeline
# ------------------------------
def tiered_forest_pipeline(paths, query, tau_low=0.85, tau_high=0.96):
    """
    Three-tier cascading pipeline
    Note: Adjusted thresholds for telecom dataset (scores are generally high)
    """
    accepted = []
    total_token = 0
    total_time = 0
    
    tier1_discarded = 0
    tier2_discarded = 0
    tier2_fastpass = 0
    tier3_evaluated = 0
    
    for path in paths:
        # --- Tier 1: Symbolic Filter ---
        if not tier1_filter(path):
            path.decision = "Discarded Tier 1"
            tier1_discarded += 1
            total_token += path.token_cost
            total_time += path.time_cost
            continue
        
        # --- Tier 2: Embedding Scoring ---
        score = tier2_score(path, query)
        
        if score >= tau_high:
            # Fast-Pass: High confidence
            path.decision = "Accepted Tier 2 (Fast-Pass)"
            tier2_fastpass += 1
            accepted.append(path)
        elif score < tau_low:
            # Discard: Low confidence
            path.decision = "Discarded Tier 2"
            tier2_discarded += 1
        else:
            # Escalate to Tier 3: Ambiguous
            tier3_evaluated += 1
            tier3_deepseek(path, query)
            if path.decision == "Accepted":
                accepted.append(path)
        
        total_token += path.token_cost
        total_time += path.time_cost
    
    stats = {
        'tier1_discarded': tier1_discarded,
        'tier2_discarded': tier2_discarded,
        'tier2_fastpass': tier2_fastpass,
        'tier3_evaluated': tier3_evaluated
    }
    
    return accepted, total_token, total_time, stats

# ------------------------------
# DeepSeek-Only Baseline
# ------------------------------
def deepseek_only_pipeline(paths, query):
    """
    Baseline: Every path is evaluated by DeepSeek
    """
    accepted = []
    total_token = 0
    total_time = 0
    
    for path in paths:
        tier3_deepseek(path, query)
        if path.decision == "Accepted":
            accepted.append(path)
        total_token += path.token_cost
        total_time += path.time_cost
    
    return accepted, total_token, total_time

# ------------------------------
# Main Benchmark
# ------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Telecom Domain KG Reasoning Benchmark")
    print("Dataset: 电信运营支撑领域术语库")
    print("=" * 60)
    
    # Load telecom terminology
    print("\n[1] Loading telecom terminology dataset...")
    terms, term_scores = load_telecom_terms(top_n=200)
    print(f"    Loaded {len(terms)} terms")
    print(f"    Sample terms: {terms[:5]}")
    
    # Generate candidate paths
    print("\n[2] Generating candidate reasoning paths...")
    query = "如何优化电信网络的数据传输性能？"
    num_paths = 20
    candidate_paths = generate_telecom_paths(terms, term_scores, num_paths=num_paths, path_length=4)
    print(f"    Generated {num_paths} candidate paths")
    print(f"    Sample path: {' -> '.join(candidate_paths[0].nodes)}")
    
    # Deep copy for fair comparison
    paths_for_baseline = copy.deepcopy(candidate_paths)
    paths_for_tiered = copy.deepcopy(candidate_paths)
    
    # Run DeepSeek-only baseline
    print("\n[3] Running DeepSeek-only baseline...")
    baseline_acc, baseline_tokens, baseline_time = deepseek_only_pipeline(paths_for_baseline, query)
    
    # Run Tiered-Forest
    print("\n[4] Running Tiered-Forest pipeline...")
    tiered_acc, tiered_tokens, tiered_time, stats = tiered_forest_pipeline(paths_for_tiered, query)
    
    # Calculate Accuracy and F1 (Assuming DeepSeek-only is Ground Truth)
    print("\n[5] Calculating Performance Metrics (vs DeepSeek Baseline)...")
    
    baseline_decisions = {p.id: p.decision for p in baseline_acc} # Accepted paths
    # Note: paths not in baseline_acc were rejected.
    
    tiered_decisions = {p.id: p.decision for p in tiered_acc}
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for path in paths_for_baseline:
        p_id = path.id
        # Ground Truth
        gt_accepted = p_id in baseline_decisions
        
        # Prediction (Check if accepted in tiered)
        pred_accepted = p_id in tiered_decisions
        
        if gt_accepted and pred_accepted:
            tp += 1
        elif not gt_accepted and pred_accepted:
            fp += 1
        elif gt_accepted and not pred_accepted:
            fn += 1
        else:
            tn += 1
            
    accuracy = (tp + tn) / num_paths
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- Model Performance Metrics ---")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Print Results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\nQuery: {query}")
    print(f"Total candidate paths: {num_paths}")
    
    print("\n--- DeepSeek-Only Baseline ---")
    print(f"  Accepted paths: {len(baseline_acc)}")
    print(f"  Total tokens: {baseline_tokens}")
    print(f"  Total time: {baseline_time:.2f}s")
    
    print("\n--- Tiered-Forest (Ours) ---")
    print(f"  Accepted paths: {len(tiered_acc)}")
    print(f"  Total tokens: {tiered_tokens}")
    print(f"  Total time: {tiered_time:.2f}s")
    print(f"\n  Pipeline Statistics:")
    print(f"    - Tier 1 discarded: {stats['tier1_discarded']}")
    print(f"    - Tier 2 discarded: {stats['tier2_discarded']}")
    print(f"    - Tier 2 fast-pass: {stats['tier2_fastpass']}")
    print(f"    - Tier 3 evaluated: {stats['tier3_evaluated']}")
    
    # Calculate improvements
    if baseline_tokens > 0:
        token_reduction = 100 * (1 - tiered_tokens / baseline_tokens)
        time_reduction = 100 * (1 - tiered_time / baseline_time)
        
        print("\n--- Performance Improvement ---")
        print(f"  Token reduction: {token_reduction:.1f}%")
        print(f"  Time reduction: {time_reduction:.1f}%")
        print(f"  Efficiency gain: {baseline_tokens / tiered_tokens:.2f}x faster (tokens)")
        
    print("\n--- Quality Metrics (ACM Style) ---")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    

    
    # Show sample accepted paths
    print("\n--- Sample Accepted Paths (Tiered-Forest) ---")
    for i, path in enumerate(tiered_acc[:3]):
        print(f"\n  Path {path.id}: {' -> '.join(path.nodes)}")
        print(f"    Decision: {path.decision}")
        print(f"    Tokens: {path.token_cost}, Time: {path.time_cost:.2f}s")
        if path.rationale:
            print(f"    Rationale: {path.rationale[:100]}...")
    
    print("\n" + "=" * 60)
