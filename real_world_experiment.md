# 真实环境实验报告：基于 DeepSeek API 与 MetaQA

**日期**: 2026-01-26
**数据集**: MetaQA (1-hop sample, n=20)
**LLM**: DeepSeek-V3 (via API)
**Small Model**: Jaccard Similarity (Local)

---

## 1. 实验结果 (Real-World Benchmark)

在真实 API 环境下，我们对比了三种模型在 MetaQA 简单问答任务上的表现。由于网络限制，Small Model 由 Cross-Encoder 替换为 Jaccard Similarity。

| 模型 | Accuracy | Avg Tokens | Avg Latency (s) | 评价 |
| :--- | :---: | :---: | :---: | :--- |
| **Standard ToG** | **100.0%** | 101.6 | 1.49 | **稳健基准**。验证所有候选路径，虽然准确率完美，但成本较高。 |
| **FrugalGPT** | **100.0%** | **72.3** | **0.86** | **最优解**。在简单实体匹配任务中，Jaccard 评分极高，绝大多数问题在 Stage 1 即可解决，无需调用 LLM，大幅节省了 Token。 |
| **Tiered-Forest** | **100.0%** | 111.4 | 1.77 | **配置保守**。由于设定的模糊区间阈值 (0.1 - 0.3) 过于保守，导致大量本可丢弃的路径被送入 LLM 进行验证，反而增加了开销。 这提示我们需要针对 Jaccard 分数特性调整阈值。 |

## 2. 结果分析

### 2.1 为什么 FrugalGPT 表现最好？
MetaQA 1-hop 数据集的特点是“实体匹配即答案”。例如 "What movies did X act in?"，只要找到包含 X 的路径即可。
*   **Jaccard 的统治力**: 简单的词重叠（Jaccard）足以完美区分正确路径和干扰项。
*   **FrugalGPT 策略**: 当 Jaccard 分数 > 0.3 时直接输出，这在 MetaQA 上几乎总是正确的。因此它极少进入 Stage 2 (LLM)，成本极低。

### 2.2 Tiered-Forest 的反思
Tiered-Forest 的设计初衷是处理由于语义模糊导致的小模型失效。但在 MetaQA 这种强匹配场景下：
*   **Ambiguity Zone 失效**: 我们设置了 `Discard < 0.1`。但在 Jaccard 评分中，哪怕是无关路径也可能有微小的重叠（如停用词），导致得分落入 `0.1 - 0.3` 区间。
*   **批量验证成本**: Tiered-Forest 将所有模糊路径打包发送给 LLM。如果无法有效剔除噪音，Batch Size 会很大，导致 Token 消耗超过仅验证 Top-5 的 FrugalGPT。

### 3. 下一步改进建议
1.  **调整阈值**: 针对 Jaccard 指标，提高 Discard 阈值至 0.2 或 0.25，减少进入模糊区间的噪音。
2.  **Top-K 截断**: 在 Ambiguity Zone 中增加 Top-K 限制（如最多验证 5 条），防止 Token 爆炸。

---
*注：本次实验受限于网络环境，使用了简单的 Jaccard 替代 Cross-Encoder。若使用语义模型，Tiered-Forest 在处理同义词替换时的优势将更明显。*
