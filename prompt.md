# Role
Act as a Principal AI Research Engineer targeting top-tier conferences (CIKM/NeurIPS). Your objective is to implement a rigorous comparative benchmark for a new KGQA framework called **"Tiered-Forest"**.

# Experiment Goal
Prove that "Tiered-Forest" occupies the optimal position on the **Cost-Accuracy Pareto Frontier** compared to state-of-the-art baselines: **Standard ToG (Think-on-Graph)** and **FrugalGPT**.

# 1. System Components & Cost Models
Please implement a simulation environment in Python. Since we don't have the live LLM APIs connected, you must use **Cost Simulation** based on the following distinct profiles:

### A. Cost & Latency Profiles (Crucial)
Define a `CostManager` to track usage:
-   **Symbolic Operation** (Tier 1 Rules): 0 Tokens, 0.001s Latency.
-   **Small Model** (BERT/Cross-Encoder/LLaMA-7B): 10 Tokens, 0.05s Latency.
-   **Large Model** (DeepSeek/GPT-4): 100 Tokens, 1.5s Latency.

### B. Models to Implement
Create a unified interface `BaseReasoner` and implement three subclasses:

1.  **`StandardToG` (Baseline 1 - High Cost/High Acc)**
    -   **Logic**: Simulates a "Think-on-Graph" process. It performs a beam search (width=3, depth=3).
    -   **Cost**: For *every* hop and *every* candidate, it calls the **Large Model**.
    -   **Simulation**: `Total Cost = 3 * 3 * Large_Model_Cost`. High accuracy probability (e.g., 0.95).

2.  **`FrugalGPT` (Baseline 2 - Adaptive)**
    -   **Logic**: A 2-stage cascade.
        -   *Stage 1*: Score with **Small Model** (LLaMA-7B).
        -   *Stage 2*: If confidence < threshold (e.g., 0.7), call **Large Model**.
    -   **Simulation**: Randomly assign confidence. If low confidence, incur both Small + Large model costs.

3.  **`TieredForest` (Ours - 3-Tier Cascade)**
    -   **Logic**:
        -   *Tier 1 (Symbolic)*: Filter out 50% of obvious noise (0 cost).
        -   *Tier 2 (Semantic)*: Use **Cross-Encoder** (Small Model Cost). Implement the **Dual-Threshold** logic (`Drop` < 0.2, `Pass` > 0.8, `Escalate` between).
        -   *Tier 3 (LLM)*: Call **Large Model** *only* for escalated paths.
    -   **Advantage**: Most paths should be resolved at Tier 1 or 2.

# 2. Data Simulation (Mocking)
Create a `DataGenerator` that produces 100 synthetic query samples. Each sample should have:
-   `complexity`: "Simple" (1-hop) vs "Complex" (multi-hop).
-   `ambiguity`: A score (0.0-1.0) representing how hard it is to distinguish the correct path.
    -   *High ambiguity* forces FrugalGPT and Tiered-Forest to use the Large Model.
    -   *Low ambiguity* allows early exits.

# 3. Metrics Calculation
Implement the following metrics in your evaluation loop:
1.  **Total Token Consumption**: Sum of all tokens used per query.
2.  **Accuracy**: Simulate correctness based on model probability and query difficulty.
3.  **Token Efficiency Ratio (TER)**: Formula: `Base_Tokens / Method_Tokens` (normalized to same accuracy).
4.  **Latency Breakdown**: Record time spent in "Reasoning" (Model) vs "Overhead" (Rules).

# 4. Execution & Visualization
Write a script `run_benchmark.py` that:
1.  Runs all 3 models on the same 100 samples.
2.  Generates a **Pareto Frontier Plot** (`pareto_plot.png`):
    -   X-axis: Total Token Cost (Log Scale).
    -   Y-axis: Accuracy.
    -   Markers: Distinct shapes for each model.
3.  Generates a **Latency Stacked Bar Chart** (`latency_breakdown.png`):
    -   Showing "Symbolic Time", "Small Model Time", "Large Model Time".

# Artifacts Required
-   `models.py`: Class implementations.
-   `simulation.py`: Cost tracking and data generation.
-   `benchmark.py`: Main execution script.
-   `requirements.txt`: `matplotlib`, `numpy`, `pandas`.

**Constraints:**
-   Ensure the code is modular.
-   Use `numpy.random` with a fixed seed for reproducibility.
-   Add comments explaining where the "Cost Savings" come from in Tiered-Forest code.