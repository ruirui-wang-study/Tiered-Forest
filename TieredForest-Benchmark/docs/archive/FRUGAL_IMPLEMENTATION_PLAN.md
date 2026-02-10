# FrugalGPT 完整实现计划

## ✅ 已完成（简化版）

### 1. 核心组件
- [x] `FrugalGPTAgent`: LLM 级联路由实现
- [x] `SimpleScorer`: 基于规则的评分函数
- [x] LLM 配置管理
- [x] 成本监控集成
- [x] 缓存系统集成

### 2. 测试验证
- [x] 评分函数单元测试
- [x] FrugalGPT Agent 功能测试
- [x] 成本统计验证

### 3. 文档
- [x] README 文档
- [x] 代码注释
- [x] 使用示例

## 🚧 待完成（完整版）

### 阶段 1: 训练评分模型（2-3天）

#### 1.1 数据收集
```python
# 收集训练数据
def collect_training_data(dataset, llm_list):
    """
    对数据集中的每个问题，调用所有 LLM 生成答案
    
    输出格式:
    [
        {
            "question": "Who directed Inception?",
            "answer": "Christopher Nolan",
            "llm": "gpt-3.5-turbo",
            "label": 1,  # 1=正确, 0=错误
            "ground_truth": "Christopher Nolan"
        },
        ...
    ]
    """
    training_data = []
    
    for question, ground_truth in dataset:
        for llm in llm_list:
            answer = llm.generate(question)
            label = evaluate_answer(answer, ground_truth)
            
            training_data.append({
                "question": question,
                "answer": answer,
                "llm": llm.name,
                "label": label,
                "ground_truth": ground_truth
            })
    
    return training_data
```

**数据需求**:
- 训练集: 500-1000 个问题
- 验证集: 100-200 个问题
- 每个问题调用 3-5 个 LLM
- 总样本数: 1500-5000 条

#### 1.2 模型训练
```python
# 使用 DistilBERT 训练评分模型
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    Trainer, 
    TrainingArguments
)

def train_scorer_model(training_data, validation_data):
    """
    训练 DistilBERT 评分模型
    """
    # 1. 加载预训练模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1  # 回归任务
    )
    
    # 2. 准备数据
    def tokenize_function(examples):
        # 拼接 question 和 answer
        texts = [
            f"{q} [SEP] {a}" 
            for q, a in zip(examples['question'], examples['answer'])
        ]
        return tokenizer(texts, padding='max_length', truncation=True)
    
    train_dataset = Dataset.from_dict(training_data).map(tokenize_function)
    val_dataset = Dataset.from_dict(validation_data).map(tokenize_function)
    
    # 3. 训练配置
    training_args = TrainingArguments(
        output_dir='./models/frugal_scorer',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )
    
    # 4. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # 5. 保存模型
    model.save_pretrained('./models/frugal_scorer')
    tokenizer.save_pretrained('./models/frugal_scorer')
    
    return model, tokenizer
```

**文件**: `train_scorer.py`

#### 1.3 集成训练好的模型
```python
# 替换 SimpleScorer
class TrainedScorer:
    def __init__(self, model_path):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def score(self, question: str, answer: str) -> float:
        # 编码输入
        text = f"{question} [SEP] {answer}"
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        return score
```

**文件**: `src/agents/trained_scorer.py`

---

### 阶段 2: 优化级联顺序（1-2天）

#### 2.1 搜索最优顺序
```python
def optimize_cascade_order(llm_list, validation_data, cost_weight=0.5):
    """
    在验证集上搜索最优 LLM 级联顺序
    
    优化目标: minimize (cost_weight * cost + (1 - cost_weight) * error_rate)
    """
    from itertools import permutations
    
    best_order = None
    best_score = float('inf')
    
    # 遍历所有可能的顺序
    for order in permutations(llm_list):
        # 评估当前顺序
        total_cost = 0
        total_errors = 0
        
        for question, ground_truth in validation_data:
            answer, cost = cascade_inference(question, order)
            total_cost += cost
            if answer != ground_truth:
                total_errors += 1
        
        # 计算综合得分
        avg_cost = total_cost / len(validation_data)
        error_rate = total_errors / len(validation_data)
        score = cost_weight * avg_cost + (1 - cost_weight) * error_rate
        
        if score < best_score:
            best_score = score
            best_order = order
    
    return best_order
```

**文件**: `optimize_cascade.py`

#### 2.2 优化阈值
```python
def optimize_thresholds(llm_order, validation_data, cost_weight=0.5):
    """
    为每个 LLM 优化停止阈值
    
    使用网格搜索或贝叶斯优化
    """
    from sklearn.model_selection import ParameterGrid
    
    # 定义搜索空间
    param_grid = {
        f'threshold_{i}': [0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(llm_order))
    }
    
    best_thresholds = None
    best_score = float('inf')
    
    for params in ParameterGrid(param_grid):
        thresholds = [params[f'threshold_{i}'] for i in range(len(llm_order))]
        
        # 评估当前阈值
        score = evaluate_cascade(llm_order, thresholds, validation_data, cost_weight)
        
        if score < best_score:
            best_score = score
            best_thresholds = thresholds
    
    return best_thresholds
```

**文件**: `optimize_thresholds.py`

---

### 阶段 3: 实现语义缓存（1天）

#### 3.1 语义缓存实现
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    """
    基于语义相似度的缓存系统
    """
    def __init__(self, similarity_threshold=0.95):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []  # [(query_embedding, answer, metadata)]
        self.similarity_threshold = similarity_threshold
    
    def get(self, query: str) -> Optional[str]:
        """查找语义相似的缓存"""
        if not self.cache:
            return None
        
        # 编码查询
        query_emb = self.encoder.encode([query])[0]
        
        # 计算相似度
        for cached_emb, cached_answer, metadata in self.cache:
            similarity = cosine_similarity(
                query_emb.reshape(1, -1), 
                cached_emb.reshape(1, -1)
            )[0][0]
            
            if similarity >= self.similarity_threshold:
                print(f"Cache hit! Similarity: {similarity:.3f}")
                return cached_answer
        
        return None
    
    def set(self, query: str, answer: str, metadata: dict = None):
        """添加到缓存"""
        query_emb = self.encoder.encode([query])[0]
        self.cache.append((query_emb, answer, metadata))
        
        # 限制缓存大小
        if len(self.cache) > 1000:
            self.cache.pop(0)
```

**文件**: `src/utils/semantic_cache.py`

---

### 阶段 4: 实现 Prompt Adaptation（1天）

#### 4.1 Prompt 优化
```python
class PromptOptimizer:
    """
    优化提示词以减少 token 使用
    """
    def optimize(self, question: str, context: str = None) -> str:
        """
        优化策略:
        1. 移除冗余词汇
        2. 使用简洁的指令
        3. 合并相似的上下文
        """
        # 简化问题
        optimized_question = self._simplify_question(question)
        
        # 构建简洁的 prompt
        if context:
            prompt = f"Context: {self._compress_context(context)}\nQ: {optimized_question}\nA:"
        else:
            prompt = f"Q: {optimized_question}\nA:"
        
        return prompt
    
    def _simplify_question(self, question: str) -> str:
        """简化问题表述"""
        # 移除礼貌用语
        question = question.replace("Could you please", "")
        question = question.replace("I would like to know", "")
        
        # 移除多余空格
        question = " ".join(question.split())
        
        return question.strip()
    
    def _compress_context(self, context: str) -> str:
        """压缩上下文"""
        # 提取关键信息
        # 可以使用 extractive summarization
        return context[:200]  # 简单截断
```

**文件**: `src/utils/prompt_optimizer.py`

---

### 阶段 5: 集成到 Benchmark（1天）

#### 5.1 更新 run_benchmark.py
```python
# 添加 FrugalGPT 到 baseline 对比
agents = {
    "Tiered-Forest": ForestAgent(...),
    "Naive-LLM": NaiveLLMAgent(...),
    "FrugalGPT": FrugalGPTAgent(...),
    "FrugalGPT-Optimized": FrugalGPTAgent(
        ..., 
        thresholds=optimized_thresholds
    ),
}
```

#### 5.2 生成对比图表
```python
# 在 plot_results.py 中添加 FrugalGPT
def plot_pareto_frontier():
    """
    绘制 Pareto 前沿
    """
    agents = {
        "Tiered-Forest": (accuracy_tf, cost_tf),
        "FrugalGPT": (accuracy_fg, cost_fg),
        "Naive-LLM": (accuracy_naive, cost_naive),
    }
    
    plt.scatter(costs, accuracies)
    # ...
```

---

## 📊 实验计划

### 实验 1: 评分函数对比
- **简化版 (SimpleScorer)** vs **训练版 (TrainedScorer)**
- 指标: AUC-ROC, 准确率, 校准误差

### 实验 2: 级联顺序优化
- **固定顺序** vs **优化顺序**
- 指标: 成本节省率, 准确率

### 实验 3: 阈值优化
- **固定阈值** vs **优化阈值**
- 指标: 成本-准确率 Pareto 前沿

### 实验 4: 完整对比
- **Tiered-Forest** vs **FrugalGPT** vs **Naive LLM**
- 数据集: MetaQA, WebQSP
- 指标: 准确率, 成本, 延迟, Pareto 前沿

---

## 📅 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| 1 | 数据收集 | 1天 |
| 1 | 模型训练 | 1天 |
| 1 | 模型集成 | 0.5天 |
| 2 | 级联顺序优化 | 1天 |
| 2 | 阈值优化 | 1天 |
| 3 | 语义缓存 | 1天 |
| 4 | Prompt 优化 | 1天 |
| 5 | Benchmark 集成 | 1天 |
| 5 | 实验运行 | 1天 |
| 5 | 结果分析 | 0.5天 |
| **总计** | | **9天** |

---

## 🎯 优先级

### P0 (必须完成)
1. ✅ 简化版实现（已完成）
2. 训练评分模型
3. Benchmark 集成

### P1 (重要)
4. 级联顺序优化
5. 阈值优化

### P2 (可选)
6. 语义缓存
7. Prompt 优化

---

## 📝 注意事项

### 1. 成本控制
- 数据收集阶段会产生 API 成本
- 估算: 1000 问题 × 3 LLM × $0.001 = **$3**
- 建议: 先用小规模数据集测试

### 2. 模型大小
- DistilBERT: ~250MB
- 推理速度: ~50ms/query
- 可以接受的开销

### 3. 缓存策略
- 语义缓存会增加内存使用
- 建议: 限制缓存大小 (1000-5000 条)

---

**最后更新**: 2026-02-10  
**当前进度**: 简化版完成 (30%)
