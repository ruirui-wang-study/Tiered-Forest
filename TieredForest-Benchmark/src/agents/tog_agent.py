"""
ToG (Think-on-Graph) Agent - 简化版

基于论文: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" (ICLR 2024)

简化版实现，适配MetaQA数据集
核心思想: 在知识图谱上进行多跳推理
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from src.agents.base_agent import BaseAgent
from src.cost_monitor import CostMonitor
from src.graph_engine import MetaQAGraphEngine
from src.kg import BaseKGBackend, MetaQABackend
from src.utils.cache_manager import LLMCache
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME
from openai import OpenAI


class ToGAgent(BaseAgent):
    """
    ToG Agent - 图上推理
    
    核心流程:
    1. 实体识别 (Entity Recognition)
    2. 多跳搜索 (Multi-hop Search, depth=1-3)
       - 关系搜索 (Relation Search)
       - 实体搜索 (Entity Search)
       - LLM剪枝 (LLM Pruning)
    3. 推理判断 (Reasoning)
    4. 答案生成 (Answer Generation)
    """
    
    def __init__(self, 
                 monitor: CostMonitor,
                 graph_engine: Optional[MetaQAGraphEngine],
                 cache_manager: LLMCache,
                 kg_backend: Optional[BaseKGBackend] = None,
                 depth: int = 2,
                 width: int = 3,
                 temperature: float = 0.0):
        """
        初始化ToG Agent
        
        Args:
            monitor: 成本监控器
            graph_engine: 图谱引擎
            cache_manager: LLM缓存管理器
            kg_backend: 可插拔KG后端（若不提供则使用graph_engine包装MetaQA后端）
            depth: 搜索深度（最多几跳）
            width: 搜索宽度（每次保留多少实体）
            temperature: LLM温度
        """
        super().__init__(monitor)
        if kg_backend is not None:
            self.graph = kg_backend
        else:
            if graph_engine is None:
                raise ValueError("Either kg_backend or graph_engine must be provided.")
            self.graph = MetaQABackend(graph_engine)
        self.cache = cache_manager
        self.depth = depth
        self.width = width
        self.temperature = temperature
        
        # 初始化LLM客户端
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "stopped_at_depth": {1: 0, 2: 0, 3: 0},
            "avg_depth": 0.0,
            "llm_calls": 0
        }
    
    def solve(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        使用ToG方法求解问题
        
        Args:
            query: 问题文本，或包含question/topic_entity/qid_topic_entity的样本
            
        Returns:
            答案文本
        """
        self.stats["total_queries"] += 1
        question, seeded_entities = self._normalize_query_input(query)
        
        try:
            # 步骤1: 实体识别
            entities = seeded_entities if seeded_entities else self._extract_entities(question)
            
            if not entities:
                # 如果没有识别到实体，直接用LLM回答
                return self._generate_directly(question)
            
            # 步骤2: 多跳搜索
            reasoning_paths = []
            current_entities = entities
            
            for d in range(1, self.depth + 1):
                print(f"  [ToG] Depth {d}: 搜索中...")
                
                # 关系搜索
                entity_relations = self._search_relations(current_entities, question)
                
                if not entity_relations:
                    print(f"  [ToG] Depth {d}: 未找到相关关系，停止")
                    break
                
                # 实体搜索
                new_entities_with_paths = self._search_entities(entity_relations)
                
                if not new_entities_with_paths:
                    print(f"  [ToG] Depth {d}: 未找到新实体，停止")
                    break
                
                # 保存推理路径
                reasoning_paths.extend(new_entities_with_paths)
                
                # 判断是否找到答案
                has_answer, answer = self._evaluate_paths(question, reasoning_paths)
                
                if has_answer:
                    print(f"  [ToG] Depth {d}: 找到答案!")
                    self.stats["stopped_at_depth"][d] += 1
                    return answer
                
                # LLM剪枝：选择最相关的实体继续搜索
                current_entities = self._prune_entities(question, new_entities_with_paths)
                
                if not current_entities:
                    print(f"  [ToG] Depth {d}: 剪枝后无实体，停止")
                    break
            
            # 步骤3: 生成最终答案
            if reasoning_paths:
                answer = self._generate_answer(question, reasoning_paths)
            else:
                answer = self._generate_directly(question)
            
            return answer
            
        except Exception as e:
            print(f"  [ToG] 错误: {e}")
            return f"Error: {e}"

    def _normalize_query_input(self, query: Union[str, Dict[str, Any]]) -> Tuple[str, List[str]]:
        if isinstance(query, str):
            return query, []

        question = str(query.get("question", "")).strip()
        seeds: List[str] = []

        qid_topic_entity = query.get("qid_topic_entity", {})
        if isinstance(qid_topic_entity, dict):
            # Prefer ids for Wikidata backend.
            seeds.extend(str(k) for k in qid_topic_entity.keys())
            seeds.extend(str(v) for v in qid_topic_entity.values())

        topic_entity = query.get("topic_entity", {})
        if isinstance(topic_entity, dict):
            seeds.extend(str(k) for k in topic_entity.keys())
            seeds.extend(str(v) for v in topic_entity.values())

        deduped = []
        seen = set()
        for seed in seeds:
            text = seed.strip()
            norm = text.lower()
            if text and norm not in seen:
                seen.add(norm)
                deduped.append(text)

        return question, deduped[: self.width]
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        从问题中提取实体
        
        对于MetaQA，实体通常在方括号中，如: [Temuera Morrison]
        
        Args:
            question: 问题文本
            
        Returns:
            实体列表
        """
        # 提取方括号中的实体
        entities = re.findall(r'\[([^\]]+)\]', question)
        
        if entities:
            print(f"  [ToG] 识别到实体: {entities}")
            return entities
        
        # 如果没有方括号，尝试用LLM识别
        print(f"  [ToG] 未找到方括号实体，使用LLM识别")
        return self._extract_entities_llm(question)
    
    def _extract_entities_llm(self, question: str) -> List[str]:
        """
        使用LLM提取实体
        
        Args:
            question: 问题文本
            
        Returns:
            实体列表
        """
        prompt = f"""Extract the main entities from the following question. Return only the entity names, separated by commas.

Question: {question}

Entities:"""
        
        response = self._call_llm(prompt, max_tokens=50)
        self.stats["llm_calls"] += 1
        
        # 解析实体
        entities = [e.strip() for e in response.split(',') if e.strip()]
        print(f"  [ToG] LLM识别到实体: {entities}")
        
        return entities[:self.width]  # 限制数量
    
    def _search_relations(self, entities: List[str], question: str) -> List[Dict]:
        """
        搜索实体的相关关系
        
        Args:
            entities: 实体列表
            question: 问题文本
            
        Returns:
            实体-关系对列表
        """
        entity_relations = []
        
        for entity in entities:
            # 从图谱中获取实体的所有关系
            relations = self.graph.get_entity_relations(entity)
            
            if relations:
                # 对于MetaQA，关系通常是固定的（如acted_in, directed_by等）
                # 我们选择最相关的关系
                for relation in relations[:3]:  # 限制关系数量
                    entity_relations.append({
                        'entity': entity,
                        'relation': relation
                    })
        
        return entity_relations
    
    def _search_entities(self, entity_relations: List[Dict]) -> List[Dict]:
        """
        沿着关系搜索新实体
        
        Args:
            entity_relations: 实体-关系对列表
            
        Returns:
            新实体及其路径
        """
        new_entities_with_paths = []
        
        for er in entity_relations:
            entity = er['entity']
            relation = er['relation']
            
            # 在图谱中查询
            results = self.graph.query_relation(entity, relation)
            
            for result in results:
                new_entities_with_paths.append({
                    'entity': result,
                    'path': f"{entity} -> {relation} -> {result}"
                })
        
        return new_entities_with_paths
    
    def _prune_entities(self, question: str, entities_with_paths: List[Dict]) -> List[str]:
        """
        使用LLM剪枝，选择最相关的实体
        
        Args:
            question: 问题文本
            entities_with_paths: 实体及其路径
            
        Returns:
            剪枝后的实体列表
        """
        if len(entities_with_paths) <= self.width:
            return [e['entity'] for e in entities_with_paths]
        
        # 构建提示
        entities_str = "\n".join([f"{i+1}. {e['entity']} (Path: {e['path']})" 
                                  for i, e in enumerate(entities_with_paths)])
        
        prompt = f"""Given the question and the candidate entities with their paths, select the top {self.width} most relevant entities.

Question: {question}

Candidate Entities:
{entities_str}

Please return only the entity names of the top {self.width} most relevant ones, separated by commas.

Selected Entities:"""
        
        response = self._call_llm(prompt, max_tokens=100)
        self.stats["llm_calls"] += 1
        
        # 解析选中的实体
        selected = [e.strip() for e in response.split(',') if e.strip()]
        
        return selected[:self.width]
    
    def _evaluate_paths(self, question: str, paths: List[Dict]) -> Tuple[bool, str]:
        """
        评估推理路径是否足以回答问题
        
        Args:
            question: 问题文本
            paths: 推理路径列表
            
        Returns:
            (是否找到答案, 答案文本)
        """
        # 构建知识三元组
        triplets = "\n".join([p['path'] for p in paths[:10]])  # 限制数量
        
        prompt = f"""Given a question and the associated retrieved knowledge graph paths, determine if it's sufficient to answer the question.

Question: {question}

Knowledge Paths:
{triplets}

Can you answer the question with these paths? Reply with "Yes" or "No", followed by the answer if yes.

Answer:"""
        
        response = self._call_llm(prompt, max_tokens=150)
        self.stats["llm_calls"] += 1
        
        # 解析响应
        if response.strip().lower().startswith('yes'):
            # 提取答案
            answer = response.replace('Yes', '').replace('yes', '').strip()
            # 提取花括号中的答案
            match = re.search(r'\{([^}]+)\}', answer)
            if match:
                return True, match.group(1)
            return True, answer
        
        return False, ""
    
    def _generate_answer(self, question: str, paths: List[Dict]) -> str:
        """
        基于推理路径生成答案
        
        Args:
            question: 问题文本
            paths: 推理路径列表
            
        Returns:
            答案文本
        """
        triplets = "\n".join([p['path'] for p in paths[:10]])
        
        prompt = f"""Given a question and the associated retrieved knowledge graph paths, please answer the question.

Question: {question}

Knowledge Paths:
{triplets}

Please provide a concise answer. If the answer is an entity name, put it in curly braces like {{answer}}.

Answer:"""
        
        response = self._call_llm(prompt, max_tokens=200)
        self.stats["llm_calls"] += 1
        
        # 提取花括号中的答案
        match = re.search(r'\{([^}]+)\}', response)
        if match:
            return match.group(1)
        
        return response.strip()
    
    def _generate_directly(self, question: str) -> str:
        """
        直接使用LLM生成答案（无图谱推理）
        
        Args:
            question: 问题文本
            
        Returns:
            答案文本
        """
        prompt = f"""Answer the following question concisely.

Question: {question}

Answer:"""
        
        response = self._call_llm(prompt, max_tokens=200)
        self.stats["llm_calls"] += 1
        
        return response.strip()
    
    def _call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        """
        调用LLM
        
        Args:
            prompt: 提示文本
            max_tokens: 最大token数
            
        Returns:
            LLM响应
        """
        # 检查缓存
        cached = self.cache.get(prompt, MODEL_NAME)
        if cached:
            return cached
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        
        result = response.choices[0].message.content
        
        # 记录成本
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.monitor.record_usage(prompt_tokens, completion_tokens, 0, "large")
        
        # 缓存结果
        self.cache.set(prompt, MODEL_NAME, result)
        
        return result
    
    def get_stats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        total = self.stats["total_queries"]
        if total > 0:
            avg_depth = sum(d * count for d, count in self.stats["stopped_at_depth"].items()) / total
            self.stats["avg_depth"] = avg_depth
        
        return self.stats
