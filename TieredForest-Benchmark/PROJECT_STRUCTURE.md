# 项目文件结构

## 📁 根目录

```
TieredForest-Benchmark/
├── README.md                    # 项目说明文档 ⭐
├── BENCHMARK_REPORT.md          # 详细实验分析报告
├── PROJECT_SUMMARY.md           # 项目总结
├── requirements.txt             # Python 依赖
│
├── run_benchmark.py             # 主实验脚本 ⭐
├── test_small_scale.py          # 小规模测试
├── plot_results.py              # 可视化生成
│
├── src/                         # 源代码
├── data/                        # 数据目录
├── results/                     # 实验结果
├── logs/                        # 日志文件
└── lib/                         # 第三方库
```

---

## 💻 源代码 (src/)

### Agents (src/agents/)
```
agents/
├── __init__.py
├── base_agent.py          # Agent 基类
├── forest_agent.py        # Tiered-Forest 实现 ⭐
├── naive_agent.py         # Naive LLM Baseline
├── frugal_agent.py        # FrugalGPT Baseline
└── agent.py               # 旧版 Agent（保留兼容）
```

### Tiers (src/tiers/)
```
tiers/
├── __init__.py
├── tier1_pruner.py            # 符号层（规则引擎）
├── tier2_ranker.py            # 语义层（CrossEncoder）
├── tier3_reasoner.py          # LLM 层
└── small_model_generator.py   # 小模型生成器
```

### Utils (src/utils/)
```
utils/
├── __init__.py
├── cache_manager.py       # LLM 缓存管理
├── logger.py              # 日志工具
└── retry_handler.py       # 重试机制
```

### Core Modules (src/)
```
src/
├── graph_engine.py        # 图谱引擎（MultiDiGraph）
├── cost_monitor.py        # 成本监控
├── data_loader.py         # 数据加载器
└── config.py              # 配置管理
```

---

## 💾 数据 (data/)

```
data/
├── MetaQA/                # 知识图谱
│   └── kb.txt             # 三元组数据
│
├── metaqa/                # MetaQA 数据集
│   ├── qa_dev.txt         # 验证集
│   └── qa_test.txt        # 测试集
│
├── processed/             # 缓存文件
│   ├── graph.pkl          # 图谱缓存
│   └── entity_index.pkl   # 实体索引
│
└── cache/                 # LLM 缓存
    └── llm_responses.json # 响应缓存
```

---

## 📊 结果 (results/)

```
results/
├── benchmark_summary.csv              # 汇总结果 ⭐
├── benchmark_detailed.csv             # 详细结果
│
├── accuracy_cost_comparison.png       # 准确率vs成本对比图
├── tier_distribution.png              # Tier 分布图
└── pareto_frontier.png                # Pareto 前沿图
```

---

## 📝 日志 (logs/)

```
logs/
├── benchmark.log          # Benchmark 日志
└── (其他运行日志)
```

---

## 🔑 核心文件说明

### 1. 实验脚本

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `run_benchmark.py` | 完整 Benchmark（50 问题） | ⭐⭐⭐ |
| `test_small_scale.py` | 小规模测试（20 问题） | ⭐⭐ |
| `plot_results.py` | 生成可视化图表 | ⭐⭐ |

### 2. 核心代码

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `src/agents/forest_agent.py` | Tiered-Forest 主实现 | ⭐⭐⭐ |
| `src/tiers/tier1_pruner.py` | 符号层（13 个规则） | ⭐⭐⭐ |
| `src/tiers/tier2_ranker.py` | 语义层（CrossEncoder） | ⭐⭐⭐ |
| `src/tiers/tier3_reasoner.py` | LLM 层（DeepSeek） | ⭐⭐⭐ |
| `src/graph_engine.py` | 图谱引擎（MultiDiGraph） | ⭐⭐⭐ |
| `src/cost_monitor.py` | 成本监控 | ⭐⭐ |

### 3. Baseline 实现

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `src/agents/naive_agent.py` | Naive LLM Baseline | ⭐⭐ |
| `src/agents/frugal_agent.py` | FrugalGPT Baseline | ⭐⭐ |

### 4. 结果文件

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `results/benchmark_summary.csv` | 汇总结果（用于论文） | ⭐⭐⭐ |
| `results/accuracy_cost_comparison.png` | 对比图表 | ⭐⭐⭐ |
| `results/tier_distribution.png` | Tier 分布 | ⭐⭐⭐ |
| `results/pareto_frontier.png` | Pareto 前沿 | ⭐⭐⭐ |

### 5. 文档

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `README.md` | 项目说明 | ⭐⭐⭐ |
| `BENCHMARK_REPORT.md` | 详细分析报告 | ⭐⭐⭐ |
| `PROJECT_SUMMARY.md` | 项目总结 | ⭐⭐ |

---

## 🗑️ 已删除的文件

### 调试脚本（已完成任务）
- ❌ `debug_candidate.py`
- ❌ `debug_graph.py`
- ❌ `debug_tier1.py`

### 旧的测试脚本
- ❌ `quick_test.py`
- ❌ `test_tier_routing.py`
- ❌ `test_tier1_enhanced.py`

### 旧的可视化脚本
- ❌ `visualize_custom.py`
- ❌ `visualize_graph.py`
- ❌ `visualize_graph_simple.py`

### 旧的优化脚本
- ❌ `optimize_thresholds.py`

### 旧的结果文件
- ❌ `results/graph_*.png` (图谱可视化，已不需要)
- ❌ `results/threshold_search.csv` (旧的阈值搜索)
- ❌ `results/small_test_results.csv` (旧的小规模测试)

### 旧的文档
- ❌ `PROJECT_STATUS.md` (已被 PROJECT_SUMMARY.md 替代)

---

## 📦 文件统计

### 总计
- **Python 文件**: 19 个
- **文档文件**: 3 个
- **实验脚本**: 3 个
- **结果文件**: 5 个
- **数据文件**: 4+ 个

### 代码行数（估算）
- **核心代码**: ~2,000 行
- **实验脚本**: ~500 行
- **文档**: ~1,500 行

---

## 🎯 使用指南

### 运行实验
```bash
# 完整 Benchmark
python run_benchmark.py

# 小规模测试
python test_small_scale.py

# 生成可视化
python plot_results.py
```

### 查看结果
```bash
# 汇总结果
cat results/benchmark_summary.csv

# 详细结果
cat results/benchmark_detailed.csv

# 图表
open results/accuracy_cost_comparison.png
open results/tier_distribution.png
open results/pareto_frontier.png
```

---

**最后更新**: 2026-02-10  
**项目状态**: ✅ 清理完成，结构清晰
