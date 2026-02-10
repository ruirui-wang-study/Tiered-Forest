# 训练 FrugalGPT 评分模型

## 📋 概述

本指南将帮助你训练一个基于 **DistilBERT** 的评分模型，用于评估 LLM 答案的质量。

训练好的模型将替换简化版的 `SimpleScorer`，提供更准确的答案质量评估。

---

## 🎯 目标

训练一个模型来预测 LLM 答案的质量分数 [0, 1]:
- **输入**: (question, answer)
- **输出**: 质量分数 (0 = 错误, 1 = 正确)

---

## 📦 依赖安装

```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

或者使用 requirements:

```bash
pip install -r requirements_training.txt
```

---

## 🚀 快速开始

### 方法 1: 一键运行（推荐）

```bash
# 运行完整的训练流程
python run_training_pipeline.py --max-samples 200 --epochs 3
```

这将自动完成:
1. ✅ 收集训练数据
2. ✅ 训练 DistilBERT 模型
3. ✅ 测试模型
4. ✅ 保存模型

### 方法 2: 分步运行

#### 步骤 1: 收集训练数据

```bash
python collect_training_data.py --max-samples 200 --output data/training/scorer_training_data.json
```

**说明**:
- 从 MetaQA 数据集加载问题
- 使用 3 个 LLM (Small-Model, Kimi, DeepSeek) 生成答案
- 评估答案正确性
- 保存为训练数据

**输出**:
- `data/training/scorer_training_data.json` - 训练数据
- 约 600 个样本 (200 问题 × 3 LLM)

**成本估算**:
- 200 问题 × 3 LLM × ~50 tokens = ~30,000 tokens
- 成本: ~$0.05

#### 步骤 2: 训练模型

```bash
python train_scorer.py \
  --data data/training/scorer_training_data.json \
  --output models/frugal_scorer \
  --epochs 3 \
  --batch-size 16
```

**说明**:
- 使用 DistilBERT 作为基础模型
- 训练回归模型预测质量分数
- 自动划分训练集和验证集 (80/20)
- 保存最佳模型

**输出**:
- `models/frugal_scorer/` - 训练好的模型
- `models/frugal_scorer/training_curves.png` - 训练曲线
- `models/frugal_scorer/training_history.json` - 训练历史

**训练时间**:
- CPU: ~30-60 分钟
- GPU: ~5-10 分钟

#### 步骤 3: 测试模型

```bash
python src/agents/trained_scorer.py models/frugal_scorer
```

**输出**:
```
Q: Who directed Inception?
A: Christopher Nolan directed Inception.
Score: 0.912

Q: Who directed Inception?
A: I don't know.
Score: 0.123
```

---

## 📊 训练数据格式

```json
[
  {
    "question": "Who directed Inception?",
    "answer": "Christopher Nolan directed Inception.",
    "label": 1.0,
    "llm": "DeepSeek",
    "ground_truth": "Christopher Nolan"
  },
  {
    "question": "Who directed Inception?",
    "answer": "I'm not sure.",
    "label": 0.0,
    "llm": "Small-Model",
    "ground_truth": "Christopher Nolan"
  }
]
```

---

## 🔧 配置参数

### 数据收集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | 200 | 收集的最大问题数 |
| `--output` | `data/training/scorer_training_data.json` | 输出文件路径 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | - | 训练数据文件路径 (必需) |
| `--output` | `models/frugal_scorer` | 模型输出目录 |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 2e-5 | 学习率 |
| `--test-size` | 0.2 | 验证集比例 |

---

## 📈 评估指标

训练完成后会显示以下指标:

- **MSE** (Mean Squared Error): 均方误差，越小越好
- **MAE** (Mean Absolute Error): 平均绝对误差，越小越好
- **Accuracy**: 分类准确率 (阈值 0.5)，越高越好

**目标**:
- MSE < 0.1
- MAE < 0.2
- Accuracy > 85%

---

## 🔄 集成到 FrugalGPT

### 方法 1: 修改 frugal_agent.py

```python
# 原来
from .frugal_scorer import SimpleScorer

# 修改为
from .trained_scorer import TrainedScorer

# 在 __init__ 中
# self.scorer = SimpleScorer()
self.scorer = TrainedScorer("models/frugal_scorer")
```

### 方法 2: 创建新的 Agent

```python
from src.agents.frugal_agent import FrugalGPTAgent
from src.agents.trained_scorer import TrainedScorer
from src.cost_monitor import CostMonitor
from src.utils.cache_manager import LLMCache

# 创建 Agent
monitor = CostMonitor()
cache = LLMCache(cache_file="data/cache/llm_cache.json")

agent = FrugalGPTAgent(monitor, cache)

# 替换评分器
agent.scorer = TrainedScorer("models/frugal_scorer")

# 使用
answer = agent.solve("Who directed Inception?")
```

---

## 🧪 测试训练好的模型

```bash
# 测试 FrugalGPT (使用训练好的评分器)
python test_frugal.py

# 运行完整 Benchmark
python run_benchmark.py
```

---

## 📝 训练流程详解

### 1. 数据收集

```
MetaQA 数据集
    ↓
加载 200 个问题
    ↓
对每个问题:
  ├─ Small-Model 生成答案
  ├─ Kimi 生成答案
  └─ DeepSeek 生成答案
    ↓
评估答案正确性
    ↓
保存训练数据 (600 样本)
```

### 2. 模型训练

```
训练数据 (600 样本)
    ↓
划分训练集/验证集 (480/120)
    ↓
加载 DistilBERT
    ↓
训练 3 个 epoch
    ↓
评估并保存最佳模型
    ↓
生成训练曲线
```

### 3. 模型推理

```
输入: (question, answer)
    ↓
Tokenize: "question [SEP] answer"
    ↓
DistilBERT 编码
    ↓
线性层 + Sigmoid
    ↓
输出: 质量分数 [0, 1]
```

---

## 🎓 模型架构

```
Input: "Who directed Inception? [SEP] Christopher Nolan"
    ↓
DistilBERT Tokenizer
    ↓
[CLS] who directed inception [SEP] christopher nolan [SEP] [PAD] ...
    ↓
DistilBERT Encoder (6 layers, 768 dim)
    ↓
[CLS] token representation
    ↓
Linear Layer (768 → 1)
    ↓
Sigmoid
    ↓
Score: 0.912
```

**模型大小**: ~250MB  
**推理速度**: ~50ms/query (CPU)  
**参数量**: ~66M

---

## 💰 成本估算

### 数据收集成本

| 样本数 | LLM 数 | 总 tokens | 成本 |
|--------|--------|-----------|------|
| 100 | 3 | ~15,000 | ~$0.025 |
| 200 | 3 | ~30,000 | ~$0.050 |
| 500 | 3 | ~75,000 | ~$0.125 |

### 训练成本

- **CPU**: 免费（时间较长）
- **GPU**: 如果使用云 GPU，约 $0.50/小时

---

## 🐛 常见问题

### 1. 缺少依赖

```bash
pip install torch transformers scikit-learn tqdm matplotlib
```

### 2. 内存不足

减少批次大小:
```bash
python train_scorer.py --batch-size 8
```

### 3. 训练时间太长

- 使用 GPU
- 减少样本数: `--max-samples 100`
- 减少 epoch: `--epochs 2`

### 4. 模型准确率低

- 增加训练数据: `--max-samples 500`
- 增加训练轮数: `--epochs 5`
- 调整学习率: `--lr 3e-5`

---

## 📚 参考资料

1. **FrugalGPT 论文**: https://arxiv.org/abs/2305.05176
2. **DistilBERT**: https://huggingface.co/distilbert-base-uncased
3. **Transformers 文档**: https://huggingface.co/docs/transformers

---

## 🎯 下一步

1. ✅ 收集训练数据
2. ✅ 训练评分模型
3. ✅ 测试模型
4. ⬜ 集成到 FrugalGPT
5. ⬜ 运行 Benchmark 对比
6. ⬜ 优化阈值
7. ⬜ 发布论文

---

**最后更新**: 2026-02-10  
**状态**: ✅ 训练脚本完成，可以开始训练
