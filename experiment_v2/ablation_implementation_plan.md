# Tiered-Forest 消融实验与公平对比实现方案

> **文档类型**: Implementation Plan  
> **创建日期**: 2026-01-27  
> **目标**: 基于现有 `experiment_v2` 框架，实现完整的消融实验和公平基线对比

---

## 📋 实现总览

### 核心目标

1. **消融实验 (Ablation Study)**: 实现 8 个变体配置 (E0, A1-A7)
2. **公平基线对比 (Fair Baseline)**: 实现 8 个基线配置 (B0-B7)
3. **参数敏感性分析**: 阈值网格搜索 + 动态参数优化
4. **召回率分析**: Tier 1 性能监控和误杀分析
5. **问题类型分层**: 按问题类型评估架构适用性

### 实现原则

- **最小侵入**: 基于现有 `experiment_v2` 框架扩展，不破坏原有代码
- **模块化设计**: 每个实验组独立实现，便于维护和复用
- **统一接口**: 所有模型变体继承统一基类，确保评估一致性
- **详细追踪**: 记录完整的路由决策、Token 消耗、延迟分解

---

## 🏗️ 架构设计

### 1. 目录结构扩展

```
experiment_v2/
├── models.py                    # 现有：StandardToG, FrugalGPT, TieredForest
├── ablation_models.py          # 新增：消融实验变体 (A1-A7)
├── baseline_models.py          # 新增：公平基线变体 (B0-B7)
├── component_library.py        # 新增：可替换组件库 (SBERT, Jaccard, etc.)
├── routing_strategies.py       # 新增：路由策略抽象
├── metrics_tracker.py          # 新增：详细指标追踪
├── ablation_experiment.py      # 新增：消融实验执行脚本
├── fair_baseline_experiment.py # 新增：公平对比执行脚本
├── sensitivity_analysis.py     # 新增：参数敏感性分析
└── visualization/              # 新增：可视化模块
    ├── ablation_plots.py
    ├── sensitivity_heatmap.py
    └── pareto_frontier.py
```

### 2. 核心类设计

#### 2.1 统一基类 `BaseRoutingModel`

```python
class BaseRoutingModel:
    """
    所有路由模型的统一基类
    """
    def __init__(self, config: dict):
        self.config = config
        self.metrics = MetricsTracker()
    
    def solve(self, question: str, **kwargs) -> str:
        """统一推理接口"""
        raise NotImplementedError
    
    def get_metrics(self) -> dict:
        """获取详细指标"""
        return self.metrics.export()
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics.reset()
```

#### 2.2 组件抽象 `Tier2Component`

```python
class Tier2Component:
    """
    Tier 2 组件的统一接口
    """
    def score(self, question: str, candidate: str) -> float:
        """
        评分接口
        返回: 0-1 之间的置信度分数
        """
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError
```

#### 2.3 路由策略 `RoutingStrategy`

```python
class RoutingStrategy:
    """
    路由决策策略抽象
    """
    def decide(self, score: float, context: dict) -> str:
        """
        路由决策
        返回: "fast_pass" | "escalate" | "discard"
        """
        raise NotImplementedError
```

---

## 🔬 实验一：消融实验实现方案

### 阶段 1: 组件库实现 (`component_library.py`)

#### 1.1 Tier 2 组件实现

```python
# 组件 1: Cross-Encoder (现有)
class CrossEncoderComponent(Tier2Component):
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        from sentence_transformers import CrossEncoder
        self.encoder = CrossEncoder(model_name)
    
    def score(self, question, candidate):
        pred = self.encoder.predict([(question, candidate)])
        return 1 / (1 + np.exp(-pred))  # Sigmoid

# 组件 2: SBERT
class SBERTComponent(Tier2Component):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def score(self, question, candidate):
        embeddings = self.model.encode([question, candidate])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return (similarity + 1) / 2  # Normalize to [0, 1]

# 组件 3: Jaccard
class JaccardComponent(Tier2Component):
    def score(self, question, candidate):
        q_tokens = set(question.lower().split())
        c_tokens = set(candidate.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        intersection = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        return intersection / union if union > 0 else 0.0

# 组件 4: LLaMA Self-Confidence
class LLaMaSelfConfidenceComponent(Tier2Component):
    def score(self, question, candidate):
        # 调用 LLaMA 进行自评分
        prompt = f"Question: {question}\nAnswer: {candidate}\nRate your confidence (0-1):"
        response = call_small_model(prompt, max_tokens=10)
        try:
            score = float(re.search(r'0\.\d+|1\.0', response).group())
            return score
        except:
            return 0.5  # 默认中等置信度
```

#### 1.2 路由策略实现 (`routing_strategies.py`)

```python
# 策略 1: 双阈值动态路由 (完整系统)
class DualThresholdDynamicRouting(RoutingStrategy):
    def __init__(self, tau_low=0.2, tau_high=0.7, dynamic=True):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.dynamic = dynamic
        self.load_history = []
    
    def decide(self, score, context):
        # 动态调整阈值
        if self.dynamic:
            self._adjust_thresholds(context)
        
        if score > self.tau_high:
            return "fast_pass"
        elif score < self.tau_low:
            return "discard"
        else:
            return "escalate"
    
    def _adjust_thresholds(self, context):
        # 根据负载和预算动态调整
        load_factor = context.get('load_factor', 1.0)
        budget_factor = context.get('budget_factor', 1.0)
        
        # 简化版动态调整逻辑
        self.tau_high = min(0.9, 0.7 + 0.1 * (load_factor - 1))
        self.tau_low = max(0.1, 0.2 - 0.05 * (budget_factor - 1))

# 策略 2: 单阈值路由 (A6)
class SingleThresholdRouting(RoutingStrategy):
    def __init__(self, tau=0.5):
        self.tau = tau
    
    def decide(self, score, context):
        return "fast_pass" if score > self.tau else "escalate"

# 策略 3: 激进 Fast-Pass (A7)
class AggressiveRouting(RoutingStrategy):
    def __init__(self, tau_low=0.2, tau_high=0.5):
        self.tau_low = tau_low
        self.tau_high = tau_high
    
    def decide(self, score, context):
        if score > self.tau_high:
            return "fast_pass"
        elif score < self.tau_low:
            return "discard"
        else:
            return "escalate"
```

### 阶段 2: 消融变体实现 (`ablation_models.py`)

#### 2.1 E0: Full System (完整系统)

```python
class TieredForestFull(BaseRoutingModel):
    """
    E0: 完整的 Tiered-Forest 系统
    """
    def __init__(self):
        super().__init__({
            'tier1_enabled': True,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'dual_threshold_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = CrossEncoderComponent()
        self.router = DualThresholdDynamicRouting()
    
    def solve(self, question, **kwargs):
        self.metrics.start_timer('total')
        
        # Tier 1: Symbolic
        self.metrics.start_timer('tier1')
        sym_ans = self.symbolic.check(question)
        self.metrics.stop_timer('tier1')
        
        if sym_ans:
            self.metrics.record_route('tier1_hit')
            self.metrics.stop_timer('total')
            return sym_ans
        
        # Tier 1.5: Candidate Generation
        self.metrics.start_timer('candidate_gen')
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        self.metrics.stop_timer('candidate_gen')
        
        # Tier 2: Semantic Scoring
        self.metrics.start_timer('tier2')
        score = self.tier2.score(question, candidate)
        self.metrics.record_score('tier2_score', score)
        self.metrics.stop_timer('tier2')
        
        # Routing Decision
        decision = self.router.decide(score, kwargs.get('context', {}))
        self.metrics.record_route(decision)
        
        if decision == "fast_pass":
            self.metrics.stop_timer('total')
            return candidate
        elif decision == "discard":
            # 可选：返回空或调用 LLM 兜底
            pass
        
        # Tier 3: LLM
        self.metrics.start_timer('tier3')
        answer = call_llm(f"Explain step by step: {question}")
        self.metrics.stop_timer('tier3')
        self.metrics.stop_timer('total')
        
        return answer
```

#### 2.2 A1: No-Tier1 (移除符号层)

```python
class TieredForestNoTier1(BaseRoutingModel):
    """
    A1: 移除 Tier 1 符号层
    """
    def __init__(self):
        super().__init__({
            'tier1_enabled': False,
            'tier2_component': 'cross_encoder',
            'routing_strategy': 'dual_threshold_dynamic'
        })
        self.tier2 = CrossEncoderComponent()
        self.router = DualThresholdDynamicRouting()
    
    def solve(self, question, **kwargs):
        # 跳过 Tier 1，直接进入 Tier 2
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        score = self.tier2.score(question, candidate)
        decision = self.router.decide(score, kwargs.get('context', {}))
        
        if decision == "fast_pass":
            return candidate
        else:
            return call_llm(f"Explain step by step: {question}")
```

#### 2.3 A2: No-Tier2 (移除语义层)

```python
class TieredForestNoTier2(BaseRoutingModel):
    """
    A2: 移除 Tier 2 语义层
    """
    def __init__(self):
        super().__init__({
            'tier1_enabled': True,
            'tier2_component': None,
            'routing_strategy': None
        })
        self.symbolic = SymbolicLayer()
    
    def solve(self, question, **kwargs):
        # Tier 1 检查
        sym_ans = self.symbolic.check(question)
        if sym_ans:
            return sym_ans
        
        # 直接跳到 Tier 3
        return call_llm(f"Explain step by step: {question}")
```

#### 2.4 其他变体 (A3-A7)

```python
# A3: No-Dynamic (静态阈值)
class TieredForestNoDynamic(TieredForestFull):
    def __init__(self):
        super().__init__()
        self.router = DualThresholdDynamicRouting(
            tau_low=0.2, 
            tau_high=0.7, 
            dynamic=False  # 关闭动态调整
        )

# A4: Two-Tier-Only (二级级联)
class TieredForestTwoTier(TieredForestNoTier1):
    pass  # 与 A1 相同

# A5: Tier1+LLM (跳过 Tier 2)
class TieredForestTier1LLM(TieredForestNoTier2):
    pass  # 与 A2 相同

# A6: Single-Threshold (单阈值)
class TieredForestSingleThreshold(TieredForestFull):
    def __init__(self):
        super().__init__()
        self.router = SingleThresholdRouting(tau=0.5)

# A7: Aggressive-Pass (激进 Fast-Pass)
class TieredForestAggressive(TieredForestFull):
    def __init__(self):
        super().__init__()
        self.router = AggressiveRouting(tau_low=0.2, tau_high=0.5)
```

### 阶段 3: 指标追踪系统 (`metrics_tracker.py`)

```python
class MetricsTracker:
    """
    详细的指标追踪系统
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timers = {}
        self.active_timers = {}
        self.routes = []
        self.scores = {}
        self.tier_tokens = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        self.tier_calls = {'tier1': 0, 'tier2': 0, 'tier3': 0}
    
    def start_timer(self, name):
        self.active_timers[name] = time.time()
    
    def stop_timer(self, name):
        if name in self.active_timers:
            elapsed = time.time() - self.active_timers[name]
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(elapsed)
            del self.active_timers[name]
    
    def record_route(self, decision):
        self.routes.append(decision)
    
    def record_score(self, name, value):
        if name not in self.scores:
            self.scores[name] = []
        self.scores[name].append(value)
    
    def export(self):
        return {
            'latency_breakdown': {
                k: sum(v) / len(v) if v else 0 
                for k, v in self.timers.items()
            },
            'routing_distribution': {
                route: self.routes.count(route) / len(self.routes) if self.routes else 0
                for route in set(self.routes)
            },
            'average_scores': {
                k: sum(v) / len(v) if v else 0
                for k, v in self.scores.items()
            },
            'tier_tokens': self.tier_tokens,
            'tier_calls': self.tier_calls
        }
```

---

## 🎯 实验二：公平基线对比实现方案

### 阶段 1: 基线模型实现 (`baseline_models.py`)

#### B2: FrugalGPT-Fair (公平版 FrugalGPT)

```python
class FrugalGPTFair(BaseRoutingModel):
    """
    B2: 使用 Cross-Encoder 的公平版 FrugalGPT
    """
    def __init__(self, threshold=0.6):
        super().__init__({
            'architecture': 'two_tier',
            'tier2_component': 'cross_encoder'
        })
        self.tier2 = CrossEncoderComponent()
        self.threshold = threshold
    
    def solve(self, question, **kwargs):
        # Stage 1: Cross-Encoder 评分
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        score = self.tier2.score(question, candidate)
        
        # 单阈值决策
        if score > self.threshold:
            return candidate
        else:
            # Stage 2: LLM
            return call_llm(f"Explain step by step: {question}")
```

#### B5: Tiered-Forest-LLaMA (反向验证)

```python
class TieredForestLLaMA(BaseRoutingModel):
    """
    B5: 使用 LLaMA-7B 作为 Tier 2 的 Tiered-Forest
    """
    def __init__(self):
        super().__init__({
            'tier1_enabled': True,
            'tier2_component': 'llama_self_confidence',
            'routing_strategy': 'dual_threshold_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = LLaMaSelfConfidenceComponent()
        self.router = DualThresholdDynamicRouting()
    
    def solve(self, question, **kwargs):
        # Tier 1
        sym_ans = self.symbolic.check(question)
        if sym_ans:
            return sym_ans
        
        # Tier 2: LLaMA 生成 + 自评分
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        score = self.tier2.score(question, candidate)
        
        decision = self.router.decide(score, kwargs.get('context', {}))
        
        if decision == "fast_pass":
            return candidate
        else:
            return call_llm(f"Explain step by step: {question}")
```

#### B7: Tiered-Forest-Jaccard (最轻量配置)

```python
class TieredForestJaccard(BaseRoutingModel):
    """
    B7: 使用 Jaccard 相似度的 Tiered-Forest
    """
    def __init__(self):
        super().__init__({
            'tier1_enabled': True,
            'tier2_component': 'jaccard',
            'routing_strategy': 'dual_threshold_dynamic'
        })
        self.symbolic = SymbolicLayer()
        self.tier2 = JaccardComponent()
        self.router = DualThresholdDynamicRouting()
    
    def solve(self, question, **kwargs):
        # 与 E0 相同的流程，但使用 Jaccard 评分
        sym_ans = self.symbolic.check(question)
        if sym_ans:
            return sym_ans
        
        candidate = call_small_model(f"Answer briefly: {question}", max_tokens=100)
        score = self.tier2.score(question, candidate)
        decision = self.router.decide(score, kwargs.get('context', {}))
        
        if decision == "fast_pass":
            return candidate
        else:
            return call_llm(f"Explain step by step: {question}")
```

---

## 📊 实验三：参数敏感性分析实现方案

### 阶段 1: 网格搜索实现 (`sensitivity_analysis.py`)

```python
def threshold_grid_search(dataset, tau_low_range, tau_high_range):
    """
    阈值网格搜索
    """
    results = []
    
    for tau_low in tau_low_range:
        for tau_high in tau_high_range:
            if tau_high <= tau_low + 0.2:
                continue  # 确保模糊区存在
            
            # 创建模型实例
            model = TieredForestFull()
            model.router.tau_low = tau_low
            model.router.tau_high = tau_high
            model.router.dynamic = False  # 静态阈值
            
            # 评估
            metrics = evaluate_model(model, dataset)
            
            results.append({
                'tau_low': tau_low,
                'tau_high': tau_high,
                'accuracy': metrics['accuracy'],
                'tokens': metrics['tokens'],
                'cost': metrics['cost'],
                'fast_pass_rate': metrics['fast_pass_rate']
            })
    
    return pd.DataFrame(results)

def plot_sensitivity_heatmap(results_df):
    """
    绘制敏感性热力图
    """
    pivot_acc = results_df.pivot(
        index='tau_low', 
        columns='tau_high', 
        values='accuracy'
    )
    pivot_tokens = results_df.pivot(
        index='tau_low', 
        columns='tau_high', 
        values='tokens'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy Heatmap
    sns.heatmap(pivot_acc, annot=True, fmt='.2%', ax=axes[0], cmap='YlGnBu')
    axes[0].set_title('Accuracy Heatmap')
    
    # Token Cost Heatmap
    sns.heatmap(pivot_tokens, annot=True, fmt='.0f', ax=axes[1], cmap='YlOrRd')
    axes[1].set_title('Token Cost Heatmap')
    
    plt.tight_layout()
    plt.savefig('sensitivity_heatmap.png', dpi=300)
```

---

## 📈 实验四：召回率分析实现方案

### 阶段 1: 召回率监控 (`metrics_tracker.py` 扩展)

```python
class RecallTracker:
    """
    Tier 1 召回率追踪
    """
    def __init__(self):
        self.tier1_decisions = []
        self.ground_truths = []
    
    def record(self, tier1_hit, tier1_answer, final_answer, ground_truth):
        self.tier1_decisions.append({
            'tier1_hit': tier1_hit,
            'tier1_answer': tier1_answer,
            'final_answer': final_answer,
            'ground_truth': ground_truth
        })
    
    def compute_recall(self):
        """
        计算 Tier 1 召回率
        """
        total_correct = 0
        tier1_preserved_correct = 0
        tier1_discarded_correct = 0
        
        for record in self.tier1_decisions:
            is_correct = record['final_answer'] == record['ground_truth']
            tier1_hit = record['tier1_hit']
            
            if is_correct:
                total_correct += 1
                if tier1_hit:
                    tier1_preserved_correct += 1
                else:
                    tier1_discarded_correct += 1
        
        recall = tier1_preserved_correct / total_correct if total_correct > 0 else 0
        false_negative_rate = tier1_discarded_correct / total_correct if total_correct > 0 else 0
        
        return {
            'tier1_recall': recall,
            'tier1_false_negative_rate': false_negative_rate,
            'tier1_precision': tier1_preserved_correct / len([r for r in self.tier1_decisions if r['tier1_hit']]) if any(r['tier1_hit'] for r in self.tier1_decisions) else 0
        }
```

---

## 🔄 实验执行脚本设计

### 主执行脚本 (`ablation_experiment.py`)

```python
def run_ablation_experiment(datasets, output_dir='ablation_results'):
    """
    运行完整的消融实验
    """
    # 定义实验组
    experiments = {
        'E0_Full': TieredForestFull(),
        'A1_NoTier1': TieredForestNoTier1(),
        'A2_NoTier2': TieredForestNoTier2(),
        'A3_NoDynamic': TieredForestNoDynamic(),
        'A4_TwoTier': TieredForestTwoTier(),
        'A5_Tier1LLM': TieredForestTier1LLM(),
        'A6_SingleThreshold': TieredForestSingleThreshold(),
        'A7_Aggressive': TieredForestAggressive()
    }
    
    all_results = []
    
    for exp_name, model in experiments.items():
        print(f"\n{'='*50}")
        print(f"Running: {exp_name}")
        print(f"{'='*50}")
        
        for dataset_name, dataset in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # 评估
            results = evaluate_model_detailed(model, dataset, exp_name)
            results['experiment'] = exp_name
            results['dataset'] = dataset_name
            
            all_results.append(results)
            
            # 保存中间结果
            pd.DataFrame(all_results).to_csv(
                f'{output_dir}/ablation_results_partial.csv', 
                index=False
            )
    
    # 保存最终结果
    df = pd.DataFrame(all_results)
    df.to_csv(f'{output_dir}/ablation_results_final.csv', index=False)
    
    # 生成分析报告
    generate_ablation_report(df, output_dir)
    
    return df
```

---

## 📊 可视化方案

### 消融实验对比图 (`visualization/ablation_plots.py`)

```python
def plot_ablation_comparison(results_df, output_dir):
    """
    生成消融实验对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy Comparison
    sns.barplot(
        data=results_df, 
        x='experiment', 
        y='accuracy', 
        hue='dataset', 
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    
    # 2. Token Consumption
    sns.barplot(
        data=results_df, 
        x='experiment', 
        y='tokens', 
        hue='dataset', 
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Token Consumption')
    axes[0, 1].set_ylabel('Total Tokens')
    
    # 3. Cost Efficiency
    results_df['cost_efficiency'] = results_df['accuracy'] / results_df['cost']
    sns.barplot(
        data=results_df, 
        x='experiment', 
        y='cost_efficiency', 
        hue='dataset', 
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Cost Efficiency (Accuracy / Cost)')
    
    # 4. Latency
    sns.barplot(
        data=results_df, 
        x='experiment', 
        y='latency', 
        hue='dataset', 
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Latency Comparison')
    axes[1, 1].set_ylabel('Latency (s)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_comparison.png', dpi=300)
```

---

## ✅ 实施步骤总结

### Phase 1: 基础设施 (1-2 天)

1. **创建新文件**:
   - `component_library.py`: 实现 Cross-Encoder, SBERT, Jaccard, LLaMA 组件
   - `routing_strategies.py`: 实现各种路由策略
   - `metrics_tracker.py`: 实现详细指标追踪

2. **扩展现有文件**:
   - `models.py`: 添加 `BaseRoutingModel` 基类
   - `simulation.py`: 扩展 `CostMonitor` 支持分层统计

### Phase 2: 消融实验 (2-3 天)

1. **创建 `ablation_models.py`**: 实现 E0, A1-A7 所有变体
2. **创建 `ablation_experiment.py`**: 实现实验执行脚本
3. **运行实验**: 在 MetaQA, WebQSP, Logistics 上运行
4. **生成报告**: 自动生成 Markdown 分析报告

### Phase 3: 公平基线 (2-3 天)

1. **创建 `baseline_models.py`**: 实现 B0-B7 所有基线
2. **创建 `fair_baseline_experiment.py`**: 实现对比实验脚本
3. **运行实验**: 在相同数据集上运行
4. **生成对比报告**: 公平对比分析

### Phase 4: 参数敏感性 (1-2 天)

1. **创建 `sensitivity_analysis.py`**: 实现网格搜索
2. **运行敏感性分析**: 25 组阈值组合
3. **生成热力图**: Accuracy 和 Token Cost 热力图

### Phase 5: 可视化与报告 (1 天)

1. **创建可视化模块**: `visualization/` 目录
2. **生成所有图表**: 对比图、热力图、Pareto 前沿
3. **整合报告**: 生成完整的实验报告

---

## 🎯 预期输出

### 文件输出

```
ablation_results/
├── ablation_results_final.csv          # 消融实验数据
├── fair_baseline_results.csv           # 公平对比数据
├── sensitivity_grid_results.csv        # 敏感性分析数据
├── ablation_comparison.png             # 消融对比图
├── sensitivity_heatmap.png             # 敏感性热力图
├── pareto_frontier.png                 # Pareto 前沿图
├── ablation_analysis.md                # 消融分析报告
└── fair_baseline_analysis.md           # 公平对比报告
```

### 关键发现 (预期)

1. **Tier 1 贡献**: 30% Token 节省
2. **Tier 2 贡献**: 避免 288% 成本增长
3. **动态阈值贡献**: 额外 5% 节省
4. **架构优势**: 三级比二级节省 8%
5. **组件优势**: Cross-Encoder 比 LLaMA 提升 4% 准确率

---

## 📝 注意事项

1. **API 成本控制**: 
   - 每个实验组先在小样本 (n=10) 上测试
   - 确认无误后再扩展到 n=50

2. **代码复用**:
   - 尽量复用现有的 `evaluate_model_on_dataset` 函数
   - 只需扩展指标收集部分

3. **错误处理**:
   - 所有 API 调用都需要 try-except
   - 记录失败的样本，便于后续分析

4. **可复现性**:
   - 固定随机种子
   - 记录所有超参数配置
   - 保存完整的实验日志

---

**下一步**: 等待用户确认后，开始逐步实现各个模块。
