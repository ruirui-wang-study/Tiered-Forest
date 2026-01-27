# 消融实验执行日志

## 实验信息

**实验名称**: Tiered-Forest 消融实验 - 扩展样本测试  
**执行日期**: 2026-01-27  
**执行者**: Antigravity AI  
**目标**: 验证三层架构各组件的独立贡献

---

## 实验配置

### 基础参数
- **样本数**: n=15 per dataset (总计 45 样本)
- **数据集**: MetaQA, WebQSP, Logistics
- **测试变体**: E0 (完整系统), A1 (无Tier1), A2 (无Tier2)
- **评估指标**: Accuracy, Token消耗, 成本, 延迟, 路由分布

### 模型配置
- **Tier 1**: 符号层 (规则匹配)
- **Tier 2**: Cross-Encoder (ms-marco-TinyBERT-L-2-v2)
- **Tier 3**: DeepSeek-V3
- **Small Model**: Qwen2.5-7B-Instruct (via SiliconFlow)

### 路由参数
- **tau_low**: 0.2 (丢弃阈值)
- **tau_high**: 0.7 (Fast-Pass阈值)
- **动态调整**: 启用

---

## 实验假设

### H1: Tier 1 的价值
**假设**: 移除 Tier 1 会导致 Token 消耗增加 20-30%  
**原因**: 失去零成本的规则过滤层

### H2: Tier 2 的价值
**假设**: 移除 Tier 2 会导致 Token 消耗增加 200-400%  
**原因**: 所有问题直接调用昂贵的 LLM

### H3: Fast-Pass 机制
**假设**: E0 和 A1 应该有 60-80% 的 Fast-Pass 率  
**原因**: Cross-Encoder 能有效识别简单问题

---

## 实验执行记录

### Run 1: 初步测试 (n=5)
**时间**: 2026-01-27 19:50 - 20:10  
**状态**: ✅ 完成  
**发现**: Cross-Encoder 数组索引 bug  
**行动**: 修复 `component_library.py` 中的 score 提取逻辑

### Run 2: Bug修复验证 (n=5)
**时间**: 2026-01-27 20:12 - 20:14  
**状态**: ✅ 完成  
**关键结果**:
- E0: 433 tokens, 0.84s, 80-100% Fast-Pass
- A1: 441 tokens (+2%), 0.85s, 100% Fast-Pass
- A2: 1,050 tokens (+143%), 7.28s, 0% Fast-Pass

**问题**: 准确率偏低（E0/A1 为 0%），Fast-Pass 过于激进

### Run 3: 扩展样本测试 (n=15)
**时间**: 2026-01-27 20:24 - [进行中]  
**状态**: 🔄 执行中  
**目标**: 
1. 验证 n=5 结果的稳定性
2. 获得更可靠的统计数据
3. 观察准确率是否随样本增加而改善

---

## 预期成本估算

### Token 消耗预测
- **E0**: ~433 tokens/sample × 15 samples × 3 datasets = ~19,485 tokens
- **A1**: ~441 tokens/sample × 15 samples × 3 datasets = ~19,845 tokens
- **A2**: ~1,050 tokens/sample × 15 samples × 3 datasets = ~47,250 tokens
- **总计**: ~86,580 tokens

### 成本预测
- DeepSeek 定价: $0.002/1k input, $0.008/1k output
- 预估成本: ~$0.50 - $1.00

### 时间预测
- E0/A1: ~1s/sample × 45 samples = ~45s per variant
- A2: ~8s/sample × 45 samples = ~360s
- 总计: ~8-10 分钟

---

## 待观察指标

### 性能指标
- [ ] Accuracy 是否提升
- [ ] Token 消耗是否稳定
- [ ] 延迟分布是否一致

### 路由指标
- [ ] Tier 1 命中率
- [ ] Fast-Pass 率
- [ ] Escalate 率
- [ ] Discard 率

### 成本效率
- [ ] Cost per correct answer
- [ ] Token savings vs baseline
- [ ] Latency reduction

---

## 下一步计划

### 如果 Run 3 成功
1. 运行完整变体集 (A3-A7)
2. 进行参数敏感性分析
3. 测试公平基线 (B2-B7)

### 如果准确率仍然偏低
1. 调整 tau_high 到 0.8-0.9
2. 优化 Tier 1 规则
3. 检查 Small Model 输出质量

### 如果 Token 消耗异常
1. 检查路由决策逻辑
2. 验证 CostMonitor 统计
3. 分析异常样本

---

## 实验日志更新

### [2026-01-27 20:24] 开始 Run 3
- 修改 `ablation_experiment.py`: limit_per_dataset=5 → 15
- 清理旧结果目录
- 启动实验执行

### [2026-01-27 20:33] Run 3 完成 ✅
**总耗时**: ~9 分钟  
**实际成本**: $0.0725 (~$0.024 per variant)  
**总样本**: 45 samples (15 × 3 datasets)

**关键发现**:

#### 准确率对比
| 变体 | MetaQA | WebQSP | Logistics | 平均 |
|------|--------|--------|-----------|------|
| E0 | 0.0% | **46.7%** | **26.7%** | 24.4% |
| A1 | 0.0% | **60.0%** | 13.3% | 24.4% |
| A2 | **20.0%** | **80.0%** | 0.0% | 33.3% |

#### Token 消耗对比
| 变体 | 平均 Tokens | vs E0 | 平均延迟 | vs E0 |
|------|------------|-------|---------|-------|
| E0 | 1,448 | baseline | 1.03s | baseline |
| A1 | 1,560 | +8% | 1.44s | +40% |
| A2 | 3,302 | **+128%** | 7.22s | **+602%** |

#### 路由行为分析
- **E0**: 93-100% Fast-Pass，Tier 1 命中率 0-7%
- **A1**: 93-100% Fast-Pass（无 Tier 1）
- **A2**: 0% Fast-Pass（直接调用 LLM）

**重要观察**:
1. **Tier 2 的价值得到验证**: A2 vs E0 显示 Token 增加 128%，延迟增加 602%
2. **Tier 1 价值有限**: A1 vs E0 仅增加 8% Token，说明当前规则覆盖不足
3. **准确率-成本权衡**: A2 准确率最高但成本最高；E0/A1 成本低但准确率受限
4. **Fast-Pass 过于激进**: E0/A1 在 MetaQA 上 100% Fast-Pass 但准确率为 0%

**问题诊断**:
- Small Model (Qwen-7B) 在 MetaQA 实体识别上表现不佳
- Cross-Encoder 对错误答案给予过高评分
### [2026-01-27 20:41] 阈值敏感性分析完成 ✅
**测试范围**: tau_high ∈ {0.7, 0.8, 0.85, 0.9}

**关键结果**:
- **MetaQA**: 全线崩溃 (0.0% 准确率, 100% Fast-Pass)，即使在 0.9 阈值下。
- **WebQSP**: 准确率随阈值提升显著
  - 0.7: 40.0%
  - 0.9: 60.0% (建议值)
- **Logistics**: 表现平稳 (13-20%)，阈值影响不大。

**结论**:
- Fast-Pass 在 MetaQA 上仍然过度自信。
- WebQSP 受益于更高的阈值 (0.9)。
- 建议后续实验主要关注架构差异，MetaQA 问题可能源于 Qwen-7B 的知识盲区。

### [2026-01-27 20:53] 开始剩余实验
1. **剩余消融变体**: A3 (静态), A4 (二级), A5 (Tier1+LLM), A6 (单阈值), A7 (激进)
2. **公平基线**: B2-B7
预计样本数: n=15 per dataset

**中间结果 (进行中)**:
- **A5 (Tier1+LLM)**: 
  - WebQSP: 73.3% 准确率 (与 A2 80% 相当)
  - Logistics: 13.3% 准确率 (与 A2 0% 相比有提升，可能因 LLM 随机性)
- **B3 (FrugalGPT-SBERT)**:
  - WebQSP: 53.3% 准确率 (高于 E0 46.7%)
- **B5 (Tiered-Forest-LLaMA)**:
  - MetaQA: 6.7% 准确率
  - WebQSP: 40.0% 准确率 (Token: 2955, Cost: $0.0038). 显著低于 A2 (80%)。
  - Logistics: 13.3% 准确率 (Token: 7444, Latency: 19.53s). 极度低效。

- **B7 (Tiered-Forest-Jaccard)**:
  - MetaQA: 运行极慢 (~8s/sample)，表明 Jaccard 相似度无法有效通过 Tier 2 阈值，导致频繁升级到 LLM。这验证了"弱语义组件导致高延迟"的假设。

**观察**:
- B3 (单阈值 + SBERT) 在 WebQSP 上表现不俗。
- A5 和 A2 表现基本一致 (预期内)，确认为高成本高准确率变体。
- B5 (LLaMA评分) 严重失败：不仅准确率低，而且成本和延迟极高。
- B7 (Jaccard) 虽然零参数成本，但无法有效过滤，导致整体系统退化为昂贵的 LLM 调用。
