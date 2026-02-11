# ToG (Think-on-Graph) 集成计划

## 📚 ToG 项目概述

**论文**: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" (ICLR 2024)

**GitHub**: https://github.com/DataArcTech/ToG

**核心思想**: 
- 在知识图谱上进行深度推理
- 结合LLM和KG的优势
- 通过图上的"思考"过程生成答案

---

## 🎯 ToG vs Tiered-Forest 对比

### 相似之处

| 特性 | ToG | Tiered-Forest |
|------|-----|---------------|
| **使用KG** | ✅ Freebase/Wikidata | ✅ MetaQA |
| **LLM推理** | ✅ GPT-3.5/4 | ✅ DeepSeek |
| **多步推理** | ✅ 图上推理 | ✅ 三层路由 |

### 关键差异

| 特性 | ToG | Tiered-Forest |
|------|-----|---------------|
| **推理方式** | 图上深度推理（多跳） | 三层路由（符号→语义→LLM） |
| **KG查询** | 动态探索子图 | 规则模板匹配 |
| **LLM角色** | 引导推理路径 | 最后的兜底 |
| **复杂度** | 高（多轮交互） | 低（单次查询） |
| **适用场景** | 复杂多跳问题 | 简单1-hop问题 |

---

## 📊 ToG 项目结构

```
ToG/
├── data/                    # 数据集
│   ├── cwq.json            # ComplexWebQuestions
│   ├── WebQSP.json         # WebQuestionsSP
│   ├── graliqa.json        # GrailQA
│   ├── SimpleQA.json       # Simple Questions
│   └── ...
├── ToG/                     # 核心代码
│   ├── main_freebase.py    # Freebase版本
│   ├── main_wiki.py        # Wikidata版本
│   ├── prompt_list.py      # 提示模板
│   ├── freebase_func.py    # Freebase函数
│   └── utils.py            # 工具函数
├── eval/                    # 评估脚本
├── Freebase/               # Freebase环境
└── Wikidata/               # Wikidata环境
```

---

## 🔧 集成ToG到Benchmark的挑战

### 挑战1: KG不兼容 ⚠️

**ToG支持的KG**:
- Freebase
- Wikidata

**Tiered-Forest使用的KG**:
- MetaQA (电影知识图谱)

**问题**: 
- ToG没有MetaQA适配器
- 需要将MetaQA转换为Freebase/Wikidata格式
- 或者实现MetaQA版本的ToG

### 挑战2: 数据格式不同 ⚠️

**ToG数据格式** (cwq.json):
```json
{
  "ID": "WebQTest-1",
  "question": "what movies did...",
  "answer": ["Q123", "Q456"],  // Wikidata QID
  "mid": ["m.0abc", "m.0def"]  // Freebase MID
}
```

**MetaQA数据格式** (qa_dev.txt):
```
what movies did [Temuera Morrison] act in\tOnce Were Warriors
```

**问题**:
- 需要转换数据格式
- MetaQA没有Wikidata QID或Freebase MID

### 挑战3: 环境依赖 ⚠️

**ToG需要**:
- Freebase本地服务器 或
- Wikidata本地服务器

**问题**:
- 安装复杂
- 需要大量存储空间
- 可能需要额外的服务器配置

---

## 💡 集成方案

### 方案1: 使用ToG的WebQuestions数据集 (推荐) ⭐

**优点**:
- ToG已经支持WebQuestions
- 数据格式兼容
- 可以直接运行

**步骤**:
1. 使用ToG的WebQuestions数据集
2. 在Tiered-Forest中实现WebQuestions适配器
3. 对比ToG和Tiered-Forest在WebQuestions上的表现

**缺点**:
- 不是MetaQA数据集
- 需要修改Tiered-Forest以支持WebQuestions

### 方案2: 实现MetaQA版本的ToG

**优点**:
- 保持MetaQA数据集
- 公平对比

**步骤**:
1. 将MetaQA转换为ToG格式
2. 实现MetaQA的KG查询接口
3. 修改ToG代码以支持MetaQA

**缺点**:
- 工作量大
- 需要深入理解ToG代码

### 方案3: 简化版ToG (零样本)

**优点**:
- 无需完整ToG环境
- 快速实现

**步骤**:
1. 提取ToG的核心推理逻辑
2. 使用MetaQA KG
3. 实现简化版ToG Agent

**缺点**:
- 不是完整的ToG
- 性能可能不如原版

---

## 🎯 推荐方案: 方案3 - 简化版ToG

### 实现计划

#### 1. ToG核心逻辑

```python
class ToGAgent:
    """
    简化版ToG Agent
    
    核心思想:
    1. 使用LLM识别问题中的实体
    2. 在KG中查找相关子图
    3. 使用LLM在子图上推理
    4. 生成最终答案
    """
    
    def solve(self, question):
        # Step 1: 实体识别
        entities = self.extract_entities(question)
        
        # Step 2: 子图检索
        subgraph = self.retrieve_subgraph(entities)
        
        # Step 3: 图上推理
        reasoning_path = self.reason_on_graph(question, subgraph)
        
        # Step 4: 生成答案
        answer = self.generate_answer(reasoning_path)
        
        return answer
```

#### 2. 与Tiered-Forest对比

| 特性 | ToG (简化版) | Tiered-Forest |
|------|-------------|---------------|
| **Tier 1** | 实体识别 | 符号层（规则） |
| **Tier 2** | 子图检索 | 语义层（小模型） |
| **Tier 3** | 图上推理 | LLM层（大模型） |
| **成本** | 中等（多次LLM调用） | 低（单次调用） |
| **准确率** | 高（深度推理） | 中等 |

---

## 📝 下一步行动

### 立即行动

1. **查看ToG代码**
   - 理解核心推理逻辑
   - 提取关键函数

2. **设计简化版ToG**
   - 适配MetaQA数据集
   - 使用MetaQA KG

3. **实现ToG Agent**
   - 创建 `src/agents/tog_agent.py`
   - 集成到benchmark

4. **运行对比实验**
   - ToG vs Tiered-Forest vs FrugalGPT
   - 在MetaQA上评估

### 预期结果

| Agent | 准确率 | 成本 | 速度 |
|-------|--------|------|------|
| Naive LLM | 64% | 高 | 慢 |
| FrugalGPT | 16% | 中 | 中 |
| **ToG (简化版)** | **50-60%?** | **中** | **慢** |
| Tiered-Forest | 48% | 低 | 快 |

---

## 🏆 对论文的价值

✅ **ToG作为baseline的优势**:

1. **强大的对比对象**
   - ICLR 2024论文
   - 知名的KG-QA方法
   - 结合LLM和KG

2. **展示Tiered-Forest的优势**
   - 成本更低
   - 速度更快
   - 准确率相当

3. **方法论对比**
   - ToG: 深度推理（复杂）
   - Tiered-Forest: 三层路由（简单高效）

---

## 📋 检查清单

- [x] 克隆ToG项目
- [ ] 查看ToG核心代码
- [ ] 理解ToG推理逻辑
- [ ] 设计简化版ToG
- [ ] 实现ToG Agent
- [ ] 集成到benchmark
- [ ] 运行对比实验
- [ ] 分析结果
- [ ] 撰写论文

---

**状态**: ✅ ToG项目已克隆到 `c:\good\Tiered-Forest\ToG`

**下一步**: 查看ToG核心代码，理解推理逻辑
