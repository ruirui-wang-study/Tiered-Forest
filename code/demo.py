"""
Tiered-Forest: Multi-Level Cascading KG Reasoning
Python prototype with Token and Time Cost Simulation
"""

import random
import numpy as np
import time

# ------------------------------
# Utilities: Mock KG Paths and Query
# ------------------------------

class CandidatePath:
    """Represents a reasoning path in KG"""
    def __init__(self, path_id, nodes):
        self.id = path_id
        self.nodes = nodes
        self.score_tier2 = None
        self.final_decision = None
        self.token_cost = 0.0
        self.time_cost = 0.0

def generate_mock_paths(num_paths=10, path_length=3):
    """Generate dummy candidate paths"""
    paths = []
    for i in range(num_paths):
        nodes = [f"Entity_{random.randint(1, 20)}" for _ in range(path_length)]
        paths.append(CandidatePath(i, nodes))
    return paths

# ------------------------------
# Tier 1: Symbolic / Structural Filter
# ------------------------------

def tier1_filter(path):
    """
    Simple example:
    - Prune paths with repeated nodes
    - Penalize paths with hub nodes (Entity_1 ~ Entity_3)
    """
    start_time = time.time()
    token_cost = 5  # arbitrary small cost for symbolic check

    if len(set(path.nodes)) < len(path.nodes):
        path.time_cost += time.time() - start_time
        path.token_cost += token_cost
        return False
    hub_nodes = {"Entity_1", "Entity_2", "Entity_3"}
    if any(n in hub_nodes for n in path.nodes):
        path.time_cost += time.time() - start_time
        path.token_cost += token_cost
        return False
    path.time_cost += time.time() - start_time
    path.token_cost += token_cost
    return True

# ------------------------------
# Tier 2: Embedding-based Semantic Scoring
# ------------------------------

def tier2_score(query, path):
    """Mock embedding scoring"""
    start_time = time.time()
    score = random.uniform(0, 1)
    token_cost = random.randint(10, 30)  # simulate embedding computation tokens
    path.score_tier2 = score
    path.time_cost += time.time() - start_time
    path.token_cost += token_cost
    return score

def tier2_decision(path, tau_low=0.3, tau_high=0.7):
    if path.score_tier2 >= tau_high:
        return "Fast-Pass"
    elif path.score_tier2 < tau_low:
        return "Discard"
    else:
        return "Escalate"

# ------------------------------
# Tier 3: LLM-based Logical Validation
# ------------------------------

from openai import OpenAI

client = OpenAI(api_key="sk-121ee6eebcc5460e91e367135db5b1c9", base_url="https://api.deepseek.com")

def tier3_evaluate(query, path):
    """Real LLM evaluation with Deepseek API"""
    start_time = time.time()
    
    # Construct a prompt for the path validation
    path_str = " -> ".join(path.nodes)
    prompt = f"Query: {query}\nPath: {path_str}\nIs this reasoning path valid and relevant for the query? Answer 'Yes' or 'No' and explain briefly."
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant evaluating reasoning paths in a knowledge graph."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        content = response.choices[0].message.content.strip().lower()
        # Simple heuristic: if it starts with yes or contains yes as a distinct word
        accepted = "yes" in content.split() or content.startswith("yes")
        
        if response.usage:
             path.token_cost += response.usage.total_tokens
        else:
             path.token_cost += 50
             
    except Exception as e:
        print(f"API Error: {e}")
        accepted = False
        path.token_cost += 0

    path.final_decision = "Accepted" if accepted else "Rejected"
    path.time_cost += time.time() - start_time
    return path.final_decision

# ------------------------------
# Tiered-Forest Pipeline
# ------------------------------

def tiered_forest_pipeline(query, candidate_paths, tau_low=0.3, tau_high=0.7):
    final_paths = []
    total_token_cost = 0
    total_time_cost = 0

    for path in candidate_paths:
        # --- Tier 1 ---
        if not tier1_filter(path):
            path.final_decision = "Discarded Tier 1"
            total_token_cost += path.token_cost
            total_time_cost += path.time_cost
            continue

        # --- Tier 2 ---
        score = tier2_score(query, path)
        decision = tier2_decision(path, tau_low, tau_high)

        if decision == "Fast-Pass":
            path.final_decision = "Accepted Tier 2"
            final_paths.append(path)
        elif decision == "Discard":
            path.final_decision = "Discarded Tier 2"
        else:
            # --- Tier 3 ---
            tier3_evaluate(query, path)
            if path.final_decision == "Accepted":
                final_paths.append(path)

        total_token_cost += path.token_cost
        total_time_cost += path.time_cost

    return final_paths, total_token_cost, total_time_cost

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    query = "Which carriers delivered packages on time?"
    candidate_paths = generate_mock_paths(num_paths=20, path_length=4)

    accepted_paths, total_tokens, total_time = tiered_forest_pipeline(
        query, candidate_paths, tau_low=0.3, tau_high=0.7
    )

    print("Accepted Reasoning Paths:")
    for path in accepted_paths:
        print(f"Path ID: {path.id}, Nodes: {path.nodes}, "
              f"Tier2 Score: {path.score_tier2:.2f}, Decision: {path.final_decision}, "
              f"Tokens: {path.token_cost}, Time: {path.time_cost:.3f}s")

    print("\nSummary:")
    print(f"Total candidate paths: {len(candidate_paths)}")
    print(f"Accepted paths: {len(accepted_paths)}")
    discarded = [p for p in candidate_paths if "Discard" in p.final_decision]
    print(f"Discarded paths: {len(discarded)}")
    print(f"Total Token Consumption: {total_tokens}")
    print(f"Total Time Cost: {total_time:.3f} seconds")
