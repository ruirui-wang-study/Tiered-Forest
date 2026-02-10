# 项目清理和整理计划

## 📋 当前状态分析

### 文档文件 (10个)
1. FRUGAL_README.md - ✅ 保留 (主要使用文档)
2. FRUGAL_FINAL_SUMMARY.md - ✅ 保留 (最终总结)
3. FRUGAL_IMPLEMENTATION_PLAN.md - ⚠️ 归档 (已完成，仅供参考)
4. FRUGAL_SUMMARY.md - ❌ 删除 (被 FINAL_SUMMARY 替代)
5. KIMI_CONFIG.md - ✅ 保留 (Kimi配置文档)
6. TRAINING_GUIDE.md - ✅ 保留 (训练指南)
7. TRAINING_ANALYSIS_V2.md - ✅ 保留 (最终分析)
8. TRAINING_PLAN_V2.md - ❌ 删除 (临时文件)
9. TRAINING_RESULTS.md - ❌ 删除 (被 ANALYSIS_V2 替代)
10. TRAINING_STATUS.md - ❌ 删除 (临时状态文件)

### Python 脚本 (7个)
1. test_frugal.py - ✅ 保留 (测试脚本)
2. test_kimi.py - ✅ 保留 (Kimi测试)
3. visualize_frugal.py - ✅ 保留 (可视化)
4. collect_training_data.py - ⚠️ 归档 (训练相关，不常用)
5. train_scorer.py - ⚠️ 归档 (训练相关，不常用)
6. run_training_pipeline.py - ⚠️ 归档 (训练相关，不常用)
7. monitor_training.py - ❌ 删除 (临时工具)

### 训练数据和模型
1. data/training/scorer_training_data.json (150样本) - ⚠️ 归档
2. data/training/scorer_training_data_v2.json (600样本) - ⚠️ 归档
3. models/frugal_scorer/ (v1模型) - ❌ 删除 (性能差)
4. models/frugal_scorer_v2/ (v2模型) - ❌ 删除 (性能差)

---

## 🎯 整理方案

### 创建归档目录
```
docs/archive/          # 归档文档
scripts/archive/       # 归档脚本
data/archive/          # 归档数据
```

### 保留的核心文件
```
TieredForest-Benchmark/
├── docs/
│   ├── FRUGAL_README.md              # FrugalGPT使用文档
│   ├── FRUGAL_FINAL_SUMMARY.md       # 最终总结
│   ├── KIMI_CONFIG.md                # Kimi配置
│   ├── TRAINING_GUIDE.md             # 训练指南
│   └── TRAINING_ANALYSIS_V2.md       # 训练分析
│
├── src/agents/
│   ├── frugal_agent.py               # FrugalGPT Agent
│   ├── frugal_scorer.py              # SimpleScorer
│   └── trained_scorer.py             # TrainedScorer (备用)
│
├── test_frugal.py                    # 测试脚本
├── test_kimi.py                      # Kimi测试
└── visualize_frugal.py               # 可视化
```

### 归档的文件
```
docs/archive/
├── FRUGAL_IMPLEMENTATION_PLAN.md
└── (其他历史文档)

scripts/archive/
├── collect_training_data.py
├── train_scorer.py
└── run_training_pipeline.py

data/archive/
├── scorer_training_data.json
└── scorer_training_data_v2.json
```

### 删除的文件
```
- FRUGAL_SUMMARY.md
- TRAINING_PLAN_V2.md
- TRAINING_RESULTS.md
- TRAINING_STATUS.md
- monitor_training.py
- models/frugal_scorer/
- models/frugal_scorer_v2/
```

---

## 📝 执行步骤

### 1. 创建目录结构
```bash
mkdir -p docs/archive
mkdir -p scripts/archive
mkdir -p data/archive
```

### 2. 移动文档到 docs/
```bash
mv FRUGAL_*.md docs/
mv KIMI_CONFIG.md docs/
mv TRAINING_*.md docs/
```

### 3. 归档文件
```bash
# 归档文档
mv docs/FRUGAL_IMPLEMENTATION_PLAN.md docs/archive/

# 归档脚本
mv collect_training_data.py scripts/archive/
mv train_scorer.py scripts/archive/
mv run_training_pipeline.py scripts/archive/

# 归档数据
mv data/training/*.json data/archive/
```

### 4. 删除无效文件
```bash
# 删除重复文档
rm docs/FRUGAL_SUMMARY.md
rm docs/TRAINING_PLAN_V2.md
rm docs/TRAINING_RESULTS.md
rm docs/TRAINING_STATUS.md

# 删除临时脚本
rm monitor_training.py

# 删除无效模型
rm -rf models/frugal_scorer
rm -rf models/frugal_scorer_v2
```

### 5. 创建索引文档
```bash
# 在 docs/ 创建 README.md
```

---

## ✅ 清理后的结构

```
TieredForest-Benchmark/
├── docs/
│   ├── README.md                     # 文档索引
│   ├── FRUGAL_README.md              # FrugalGPT使用文档
│   ├── FRUGAL_FINAL_SUMMARY.md       # 最终总结
│   ├── KIMI_CONFIG.md                # Kimi配置
│   ├── TRAINING_GUIDE.md             # 训练指南
│   ├── TRAINING_ANALYSIS_V2.md       # 训练分析
│   └── archive/
│       └── FRUGAL_IMPLEMENTATION_PLAN.md
│
├── scripts/
│   └── archive/
│       ├── collect_training_data.py
│       ├── train_scorer.py
│       └── run_training_pipeline.py
│
├── data/
│   └── archive/
│       ├── scorer_training_data.json
│       └── scorer_training_data_v2.json
│
├── src/agents/
│   ├── frugal_agent.py
│   ├── frugal_scorer.py
│   └── trained_scorer.py
│
├── test_frugal.py
├── test_kimi.py
└── visualize_frugal.py
```

---

## 📊 清理统计

### 删除
- 文档: 4个
- 脚本: 1个
- 模型: 2个目录 (~500MB)

### 归档
- 文档: 1个
- 脚本: 3个
- 数据: 2个

### 保留
- 文档: 5个
- 脚本: 3个
- 源代码: 3个

---

**执行时间**: 预计 5 分钟
**释放空间**: ~500MB
**状态**: 待执行
