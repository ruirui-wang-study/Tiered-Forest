import random
import time
import copy

# ------------------------------
# DeepSeek API Setup
# ------------------------------
# pip install deepseek-sdk (假设有官方 SDK)
# from deepseek import DeepSeekClient

# client = DeepSeekClient(api_key="YOUR_API_KEY")

# Mock DeepSeek call for demo
def deepseek_evaluate(query, path_nodes):
    """
    Evaluate a candidate path using DeepSeek
    Replace this with actual DeepSeek API call
    """
    # Example DeepSeek API call (pseudo-code)
    # response = client.query(
    #     prompt=f"Question: {query}\nPath: {path_nodes}\nCheck if path leads to correct answer. Return score [0-1].",
    #     model="deepseek-v3"
    # )
    # score = response["score"]
    # rationale = response["rationale"]

    # For demo, simulate score
    score = random.uniform(0,1)
    accepted = score >= 0.5
    rationale = "Simulated rationale."
    return accepted, score, rationale

# ------------------------------
# Candidate Path Class
# ------------------------------

class CandidatePath:
    def __init__(self, path_id, nodes):
        self.id = path_id
        self.nodes = nodes
        self.score = None
        self.decision = None
        self.token_cost = 0
        self.time_cost = 0

def generate_mock_paths(num_paths=15, path_length=4):
    paths = []
    for i in range(num_paths):
        nodes = [f"Entity_{random.randint(1, 20)}" for _ in range(path_length)]
        paths.append(CandidatePath(i, nodes))
    return paths

# ------------------------------
# Tiered-Forest Pipeline with DeepSeek
# ------------------------------

def tier3_deepseek(path, query):
    """Call DeepSeek for Tier 3 evaluation"""
    start_time = time.time()
    accepted, score, rationale = deepseek_evaluate(query, path.nodes)
    
    # Estimate token cost (DeepSeek API cost)
    token_cost = 150  # approximate
    time.sleep(random.uniform(0.05,0.1))  # simulate latency
    
    path.token_cost += token_cost
    path.time_cost += time.time() - start_time
    path.decision = "Accepted" if accepted else "Rejected"
    path.score = score
    return path.decision

def tiered_forest_deepseek(paths, query, tau_low=0.3, tau_high=0.7):
    accepted = []
    total_token = 0
    total_time = 0

    for path in paths:
        # Tier 1
        token_cost_t1 = 5
        start_time = time.time()
        if len(set(path.nodes)) < len(path.nodes) or any(n in {"Entity_1","Entity_2","Entity_3"} for n in path.nodes):
            path.decision = "Discarded Tier 1"
            path.token_cost += token_cost_t1
            path.time_cost += time.time() - start_time
            total_token += path.token_cost
            total_time += path.time_cost
            continue
        path.token_cost += token_cost_t1
        path.time_cost += time.time() - start_time

        # Tier 2
        token_cost_t2 = random.randint(10,30)
        start_time = time.time()
        score = random.uniform(0,1)
        path.score = score
        path.token_cost += token_cost_t2
        path.time_cost += time.time() - start_time

        if score >= tau_high:
            path.decision = "Accepted Tier 2"
            accepted.append(path)
        elif score < tau_low:
            path.decision = "Discarded Tier 2"
        else:
            # Tier 3: DeepSeek API
            tier3_deepseek(path, query)
            if path.decision == "Accepted":
                accepted.append(path)

        total_token += path.token_cost
        total_time += path.time_cost

    return accepted, total_token, total_time

# ------------------------------
# LLM-only benchmark with DeepSeek
# ------------------------------

def deepseek_only(paths, query):
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
    query = "Which carriers delivered packages on time?"
    candidate_paths = generate_mock_paths(num_paths=20, path_length=4)

    paths_for_llm = copy.deepcopy(candidate_paths)
    paths_for_tiered = copy.deepcopy(candidate_paths)

    # Run DeepSeek-only (every path calls DeepSeek)
    llm_acc, llm_tokens, llm_time = deepseek_only(paths_for_llm, query)

    # Run Tiered-Forest with DeepSeek only on ambiguous paths
    tier_acc, tier_tokens, tier_time = tiered_forest_deepseek(paths_for_tiered, query)

    print("=== Benchmark with DeepSeek ===")
    print("DeepSeek-only:")
    print(f"  Accepted paths: {len(llm_acc)}")
    print(f"  Total Token consumption: {llm_tokens}")
    print(f"  Total Time: {llm_time:.2f} s")

    print("\nTiered-Forest + DeepSeek:")
    print(f"  Accepted paths: {len(tier_acc)}")
    print(f"  Total Token consumption: {tier_tokens}")
    print(f"  Total Time: {tier_time:.2f} s")

    print("\nEfficiency Gain:")
    token_reduction = 100 * (1 - tier_tokens/llm_tokens)
    time_reduction = 100 * (1 - tier_time/llm_time)
    print(f"  Token Reduction: {token_reduction:.1f}%")
    print(f"  Time Reduction: {time_reduction:.1f}%")



import matplotlib.pyplot as plt
import numpy as np

# 假设 benchmark 结果
models = ['DeepSeek-only', 'Tiered-Forest + DeepSeek']
accepted_paths = [len(llm_acc), len(tier_acc)]
token_consumption = [llm_tokens, tier_tokens]
time_cost = [llm_time, tier_time]

x = np.arange(len(models))
width = 0.35

# ------------------------------
# 创建论文风格图
# ------------------------------
fig, ax1 = plt.subplots(figsize=(8,5))

# 柱状图：Accepted Paths（黑白灰 + 纹理填充）
bar_patterns = ['/', '\\']
colors = ['lightgray', 'gray']
bar1 = ax1.bar(x - width/2, accepted_paths, width,
               label='Accepted Paths', color=colors[0], hatch=bar_patterns[0], edgecolor='black')

ax1.set_ylabel('Accepted Paths', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_title('Tiered-Forest vs DeepSeek-only Benchmark', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# 柱状图数据标注
for rect in bar1:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., height + 0.1,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)

# ------------------------------
# 折线图：Token & Time (双 y 轴)
# ------------------------------
ax2 = ax1.twinx()
line1 = ax2.plot(x + width/2, token_consumption, color='black', marker='o', linestyle='-', label='Token Consumption')
line2 = ax2.plot(x + width/2, time_cost, color='dimgray', marker='s', linestyle='--', label='Time Cost (s)')
ax2.set_ylabel('Token / Time', fontsize=11)

# 折线图数据标注
for i, v in enumerate(token_consumption):
    ax2.text(i + width/2, v + 5, str(v), color='black', ha='center', fontsize=9)
for i, v in enumerate(time_cost):
    ax2.text(i + width/2, v + 0.02, f"{v:.2f}", color='dimgray', ha='center', fontsize=9)

# ------------------------------
# 图例
# ------------------------------
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()
