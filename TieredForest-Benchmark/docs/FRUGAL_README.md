# FrugalGPT 简化版实现

## 📖 概述

基于论文 **"FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"** (Chen et al., Stanford 2023) 的简化版实现。

FrugalGPT 通过 **LLM 级联策略** 在保持准确率的同时大幅降低推理成本。

## 🎯 核心思想

1. **LLM Cascade（级联路由）**: 按成本从低到高的顺序调用 LLM
2. **Generation Scoring（答案评分）**: 评估每个 LLM 答案的质量
3. **Early Stopping（提前停止）**: 如果答案质量足够好，立即返回，不调用更贵的模型

## 🏗️ 架构设计

```
用户问题
    ↓
┌─────────────────────────────────────┐
│  FrugalGPT Agent                    │
├─────────────────────────────────────┤
│  1. 调用小模型 (便宜)                │
│     ↓                               │
│  2. 评分函数评估答案质量             │
│     ↓                               │
│  3. 质量 >= 阈值？                  │
│     ├─ 是 → 返回答案 ✓              │
│     └─ 否 → 调用大模型 (贵)         │
│         ↓                           │
│     4. 评分函数评估答案质量          │
│         ↓                           │
│     5. 返回答案                     │
└─────────────────────────────────────┘
```

## 📁 文件结构

```
src/agents/
├── frugal_agent.py      # FrugalGPT 主实现
└── frugal_scorer.py     # 简化版评分函数

test_frugal.py           # 测试脚本
FRUGAL_README.md         # 本文档
```

## 🚀 快速开始

### 1. 运行测试

```bash
# 测试评分函数和 FrugalGPT Agent
python test_frugal.py
```

### 2. 使用示例

```python
from src.agents.frugal_agent import FrugalGPTAgent
from src.cost_monitor import CostMonitor
from src.utils.cache_manager import LLMCache

# 初始化
monitor = CostMonitor()
cache = LLMCache(cache_file="data/cache/llm_cache.json")

# 创建 FrugalGPT Agent
agent = FrugalGPTAgent(
    monitor=monitor,
    cache_manager=cache,
    thresholds=[0.7, 0.5]  # [小模型阈值, 大模型阈值]
)

# 回答问题
answer = agent.solve("Who directed Inception?")
print(f"Answer: {answer}")

# 查看统计
stats = agent.get_frugal_stats()
print(f"Small model usage: {stats['small_model_usage_rate']:.1%}")
print(f"Avg LLMs called: {stats['avg_llms_called']:.2f}")
```

## ⚙️ 配置说明

### LLM 级联配置

默认配置（从便宜到贵）：
1. **Small-Model**: Qwen2.5-7B (~$0.20/1M tokens)
2. **Large-Model**: DeepSeek-Chat (~$2/1M input, ~$8/1M output)

### 阈值配置

```python
thresholds = [0.7, 0.5]
```

- `thresholds[0]`: 小模型的质量阈值（建议 0.7-0.8）
- `thresholds[1]`: 大模型的质量阈值（建议 0.5-0.6）

**调优建议**:
- **降低成本**: 提高小模型阈值 → 更多问题用小模型
- **提高准确率**: 降低小模型阈值 → 更多问题用大模型

## 📊 评分函数

### 简化版实现

使用 **启发式规则** 评估答案质量（0-1分）：

| 规则 | 分数变化 | 说明 |
|------|---------|------|
| 包含错误词 | -0.4 | "error", "failed", "不知道" |
| 包含不确定词 | -0.2 | "maybe", "possibly", "可能" |
| 包含积极词 | +0.2 | "the answer is", "根据" |
| 答案长度合理 | +0.1 | 3-50 词 |
| 包含实体/数字 | +0.2 | 人名、地名、数字 |
| 直接回答问题 | +0.1 | 匹配问题类型 |

### 完整版实现（未实现）

在论文中，使用 **DistilBERT** 训练评分模型：
- 输入: `[CLS] question [SEP] answer [SEP]`
- 输出: 回归分数 [0, 1]
- 训练数据: (question, answer, correctness_label)

## 📈 实验结果

### 测试案例

```
问题 1: Who directed the movie Inception?
  [Small-Model] Score: 0.800 ≥ 0.800 ✓
  答案: Christopher Nolan directed the movie Inception.
  成本: $0.000010
  
问题 2: What is 2+2?
  [Small-Model] Score: 0.500 < 0.800 ✗
  [Large-Model] Score: 0.800 ≥ 0.500 ✓
  答案: 2 + 2 = 4
  成本: $0.000101
  
统计:
  总查询数: 3
  小模型使用率: 33.3%
  平均调用 LLM 数: 1.67
```

## 🔬 与 Tiered-Forest 的对比

| 维度 | FrugalGPT | Tiered-Forest |
|------|-----------|---------------|
| **第一层** | 小模型 (有成本) | 符号推理 (零成本) |
| **路由依据** | 答案质量评分 | 问题复杂度 + 图结构 |
| **适用场景** | 通用 QA | 知识图谱 QA |
| **优点** | 简单、通用 | 成本更低、准确率高 |
| **缺点** | 需要评分模型 | 依赖知识图谱 |

## 🎓 论文要点

### 三大策略

1. **Prompt Adaptation**: 优化提示词，减少 token 使用
2. **LLM Approximation**: 使用缓存和模型微调
3. **LLM Cascade**: 动态选择最合适的 LLM（核心）

### 实验结果

- **成本节省**: 高达 98%
- **准确率**: 与 GPT-4 持平，或提高 4%

### 论文信息

- **标题**: FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
- **作者**: Lingjiao Chen, Matei Zaharia, James Zou
- **机构**: Stanford University
- **年份**: 2023
- **链接**: https://arxiv.org/abs/2305.05176

## 🛠️ 扩展方向

### 1. 训练评分模型

```python
# 使用 DistilBERT 训练评分函数
from transformers import DistilBertForSequenceClassification

# 准备训练数据
train_data = [
    {"question": "...", "answer": "...", "label": 1.0},
    ...
]

# 训练模型
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# ... 训练代码
```

### 2. 优化级联顺序

```python
# 在验证集上搜索最优 LLM 顺序
def optimize_cascade_order(validation_data):
    best_order = None
    best_score = 0
    
    for order in all_permutations(llm_list):
        score = evaluate_cascade(order, validation_data)
        if score > best_score:
            best_order = order
            best_score = score
    
    return best_order
```

### 3. 添加更多 LLM

```python
# 添加中等成本的 LLM
cascade = [
    LLMConfig("Tiny-Model", ...),    # 最便宜
    LLMConfig("Small-Model", ...),   # 便宜
    LLMConfig("Medium-Model", ...),  # 中等
    LLMConfig("Large-Model", ...),   # 贵
]
```

### 4. 实现缓存策略

```python
# 使用语义相似度匹配缓存
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []
    
    def get(self, query):
        # 查找语义相似的缓存
        query_emb = self.encoder.encode(query)
        for cached_query, cached_answer in self.cache:
            similarity = cosine_similarity(query_emb, cached_query)
            if similarity > 0.95:
                return cached_answer
        return None
```

## 📚 参考资料

1. **论文**: [FrugalGPT (arXiv)](https://arxiv.org/abs/2305.05176)
2. **官方代码**: [stanford-futuredata/FrugalGPT](https://github.com/stanford-futuredata/FrugalGPT)
3. **相关工作**:
   - Cascade of LLMs
   - Model Routing
   - Cost-Performance Trade-off

## 📝 TODO

- [ ] 训练 DistilBERT 评分模型
- [ ] 实现级联顺序优化
- [ ] 添加语义缓存
- [ ] 集成到 Benchmark 对比实验
- [ ] 实现 Prompt Adaptation 策略
- [ ] 支持更多 LLM API

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License

---

**最后更新**: 2026-02-10  
**实现状态**: ✅ 简化版完成，核心功能可用
