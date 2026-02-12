# ToG Baseline Benchmark Report

## 📊 Executive Summary

This report analyzes the performance of **Think-on-Graph (ToG)** as a new baseline compared to **Tiered-Forest** and **Naive LLM** on the MetaQA (1-hop) dataset.

### Key Findings

1.  **ToG is the Accuracy Champion**: ToG achieved **96% accuracy**, significantly outperforming Naive LLM (64%) and Tiered-Forest (48%). This validates ToG's deep reasoning capabilities on Knowledge Graphs.
2.  **Tiered-Forest is the Efficiency Champion**: Tiered-Forest is **~180x cheaper** and **~13x faster** than ToG. It offers a viable low-cost alternative for high-volume applications where 48-60% accuracy is acceptable.
3.  **Trade-off Established**: There is a clear trade-off between accuracy (ToG) and efficiency (Tiered-Forest).

---

## 📈 Quantitative Results

| Agent | Accuracy | Cost (USD) | Cost/Query | Latency (s) | LLM Calls/Query |
|-------|----------|------------|------------|-------------|-----------------|
| **Naive LLM** | 64.00% | $0.000000* | $0.000000* | 0.00* | 1.0 (est) |
| **ToG** | **96.00%** | $0.042308 | **$0.000846** | 2.87 | **2.18** |
| **Tiered-Forest**| 48.00% | **$0.000235**| **$0.000005**| **0.22** | 0.62 |

*\*Naive LLM results were cached, so cost/latency are near zero. Real-world cost would be ~$0.049 and latency ~2-3s.*

### Relative Performance (Tiered-Forest vs ToG)

*   **Cost Reduction**: **99.4%** cheaper (1/180th of ToG cost)
*   **Speedup**: **13x** faster
*   **Accuracy Gap**: -48% points (significant room for improvement in Tiered-Forest)

---

## 🧠 Analysis

### 1. ToG Performance (The "Smart" Baseline)
ToG demonstrated exceptional performance (96%) by effectively utilizing the Knowledge Graph.
*   **Mechanism**: It identifies entities, searches for relations, and verifies answers using the LLM.
*   **Why it works**: For MetaQA (1-hop), the answer is directly linked in the KG. ToG's search strategy (Depth=1/2) is perfect for finding these connections.
*   **Cost**: The multi-step process (Entity extraction -> Relation search -> LLM verification) incurs high token usage (~2.2 calls per query).

### 2. Tiered-Forest Performance (The "Fast" Approach)
Tiered-Forest prioritized speed and cost.
*   **Tier 1 (Symbolic)**: Handled ~6% of queries instantly.
*   **Tier 2 (Semantic)**: Handled ~32% of queries using small models.
*   **Tier 3 (LLM)**: Handled ~62% of queries.
*   **Issue**: The accuracy (48%) is lower than Naive LLM (64%). This suggests that **Tier 1 or Tier 2 is answering incorrectly** for ~16% of questions that the LLM would have gotten right.
    *   *Hypothesis*: The small model (Tier 2) might be retrieving irrelevant contexts or matching incorrectly.

### 3. Comparison with Naive LLM
*   **Naive LLM (64%)** is a strong baseline itself. It relies on internal knowledge (or hallucinations that happen to be correct).
*   **ToG (96%)** beats Naive LLM by grounding answers in the KG.
*   **Tiered-Forest (48%)** underperforms Naive LLM in accuracy but is far cheaper.

---

## 🖼️ Visualizations

All plots are saved in `results/plots/`:
*   `accuracy_comparison.png`: Shows the accuracy gap.
*   `cost_comparison.png`: Shows the massive cost difference.
*   `radar_comparison.png`: Highlights the trade-offs (Area of shape).

---

## 📝 Recommendations for Paper

1.  **Position ToG as the "Upper Bound"**: Use ToG to demonstrate what is possible with unlimited budget/time.
2.  **Position Tiered-Forest for "Scale"**: Emphasize that Tiered-Forest enables **massive throughput** at negligible cost.
3.  **Future Work**: Explicitly mention that optimizing Tier 2 (Semantic Layer) is the key to closing the accuracy gap with Naive LLM/ToG while maintaining efficiency.

---

## 📂 Artifacts

*   **Implementation**: `src/agents/tog_agent.py`
*   **Benchmark Script**: `run_benchmark.py`
*   **Visualization**: `plot_benchmark_results.py`
*   **Raw Results**: `results/benchmark_summary.csv`
