from typing import Any, Dict, List, Optional, Tuple, Union

from .base_agent import BaseAgent
from ..cost_monitor import CostMonitor
from ..graph_engine import MetaQAGraphEngine
from ..kg import BaseKGBackend, MetaQABackend
from ..tiers.tier1_pruner import SymbolicPruner
from ..tiers.tier2_ranker import SemanticRanker
from ..tiers.tier3_reasoner import LLMReasoner
from ..tiers.small_model_generator import SmallModelGenerator
from ..utils.cache_manager import LLMCache

class TieredForestAgent(BaseAgent):
    """
    Tiered-Forest Agent
    三层路由架构:
    - Tier 1: Symbolic (规则/图谱)
    - Tier 2: Semantic (CrossEncoder)
    - Tier 3: LLM (DeepSeek)
    """
    
    def __init__(self, 
                 monitor: CostMonitor,
                 graph_engine: Optional[MetaQAGraphEngine],
                 cache_manager: LLMCache,
                 kg_backend: Optional[BaseKGBackend] = None,
                 t_drop: float = 0.2,
                 t_pass: float = 0.7,
                 enable_small_model: bool = False):
        """
        初始化 Tiered-Forest Agent
        
        Args:
            monitor: 成本监控器
            graph_engine: 图谱引擎
            cache_manager: LLM 缓存管理器
            kg_backend: 可插拔KG后端（若不提供则使用graph_engine包装MetaQA后端）
            t_drop: 丢弃阈值（低于此分数拒绝候选）
            t_pass: 通过阈值（高于此分数直接返回）
        """
        super().__init__(monitor)
        self.t_drop = t_drop
        self.t_pass = t_pass
        self.enable_small_model = enable_small_model

        if kg_backend is not None:
            self.kg = kg_backend
        else:
            if graph_engine is None:
                raise ValueError("Either kg_backend or graph_engine must be provided.")
            self.kg = MetaQABackend(graph_engine)
        
        # 初始化三层
        self.tier1 = SymbolicPruner(self.kg)
        self.tier2 = SemanticRanker()
        self.tier3 = LLMReasoner(cache_manager)
        
        # 初始化小模型生成器（用于 Tier 2 候选生成）
        self.small_model = SmallModelGenerator() if self.enable_small_model else None
        
        # 统计各层的使用情况
        self.tier_usage = {"tier1": 0, "tier2": 0, "tier3": 0}
    
    def solve(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        三层路由求解
        
        Args:
            query: 问题文本，或包含question/topic_entity/qid_topic_entity的样本
            
        Returns:
            答案字符串
        """
        question, entity_hints = self._normalize_query_input(query)

        # --- Tier 1: Symbolic Layer ---
        answer, confidence = self.tier1.check(question, entity_hints=entity_hints)
        if answer and confidence > 0.8:
            # Tier 1 成功，零成本
            self.monitor.record_usage(0, 0, 0.001, "symbolic")
            self.tier_usage["tier1"] += 1
            return answer
        
        # --- Tier 2: Semantic Layer ---
        # 生成候选答案（图谱或小模型）
        candidate, source = self._generate_candidate(question, entity_hints=entity_hints)
        
        if candidate:
            # 使用 CrossEncoder 打分
            score = self.tier2.score_candidate(question, candidate)
            
            # 路由决策
            can_direct_pass = score > self.t_pass
            if source == "small_model":
                can_direct_pass = (
                    can_direct_pass
                    and score >= 0.95
                    and self._is_short_answer(candidate)
                )

            if can_direct_pass:
                # 高置信度，直接返回
                # 如果候选来自图谱，成本为0；如果来自小模型，已经在生成时记录
                if source == "graph":
                    self.monitor.record_usage(0, 0, 0.01, "small")  # CrossEncoder 成本可忽略
                self.tier_usage["tier2"] += 1
                return candidate
            elif score < self.t_drop:
                # 低置信度，候选无效，进入 Tier 3
                pass
            else:
                # 中等置信度，也进入 Tier 3 但可以使用候选作为上下文
                pass
        
        # --- Tier 3: LLM Reasoning ---
        try:
            context = f"Candidate answer: {candidate}" if candidate else None
            choices = self._extract_candidate_options(candidate, source)
            answer, prompt_tokens, completion_tokens, latency = self.tier3.reason(
                question,
                context=context,
                choices=choices if choices else None,
            )
            self.monitor.record_usage(prompt_tokens, completion_tokens, latency, "large")
            self.tier_usage["tier3"] += 1
            return answer
        except Exception as e:
            print(f"Tier 3 failed: {e}")
            return "Error: Unable to answer"
    
    def _generate_candidate(self, question: str, entity_hints: Optional[List[str]] = None) -> tuple:
        """
        生成候选答案（增强版）
        
        策略（按优先级）:
        1. 图谱查询（零成本）
        2. 小模型生成（低成本）
        
        Returns:
            (candidate_answer, source) 或 (None, None)
        """
        # 策略1: 尝试从图谱获取候选
        graph_candidate = self._graph_candidate(question, entity_hints=entity_hints)
        if graph_candidate:
            return graph_candidate, "graph"
        
        # 策略2: 使用小模型生成候选
        if self.small_model and self.small_model.is_available():
            try:
                candidate, p_tokens, c_tokens, latency = self.small_model.generate_candidate(question)
                if candidate:
                    # 记录小模型的成本
                    self.monitor.record_usage(p_tokens, c_tokens, latency, "small")
                    return candidate, "small_model"
            except Exception as e:
                print(f"Small model generation failed: {e}")
        
        # 所有策略都失败
        return None, None
    
    def _graph_candidate(self, question: str, entity_hints: Optional[List[str]] = None) -> Optional[str]:
        """
        从图谱生成候选答案（改进版）
        
        策略:
        1. 分析问题类型，确定目标关系
        2. 提取问题中的实体
        3. 查询图谱获取特定关系的邻居
        4. 返回最相关的邻居
        """
        question_lower = question.lower()
        
        # 根据问题模式确定目标关系类型（优先MetaQA关系名，也兼容Wikidata常见label）
        target_relations: List[str] = []
        if "act in" in question_lower or "acted in" in question_lower or "starred in" in question_lower:
            target_relations = ["starred_actors", "cast member", "performer"]
            search_direction = "in"  # 查询入边
        elif "directed" in question_lower:
            target_relations = ["directed_by", "director"]
            search_direction = "out"
        elif "genre" in question_lower:
            target_relations = ["has_genre", "genre"]
            search_direction = "out"
        elif "written by" in question_lower or "wrote" in question_lower:
            target_relations = ["written_by", "author", "screenwriter"]
            search_direction = "out"
        else:
            # 默认：双向查询
            target_relations = []
            search_direction = "both"
        
        # 提取实体（优先使用样本中给定的 topic entity）
        candidate_entities: List[str] = []
        if entity_hints:
            candidate_entities.extend(entity_hints[:6])
        else:
            words = question_lower.split()
            stop_words = {
                "what", "who", "when", "where", "which", "how", "did", "does",
                "is", "are", "was", "were", "the", "a", "an", "in", "on", "at", "of",
            }
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    candidate_entities.append(word)
                if len(candidate_entities) >= 4:
                    break

        # 去重并尝试查询
        dedup_entities = []
        seen = set()
        for entity_text in candidate_entities:
            key = entity_text.lower().strip()
            if key and key not in seen:
                seen.add(key)
                dedup_entities.append(entity_text)

        for entity_text in dedup_entities:
            entity = self.kg.find_entity(entity_text)
            if not entity:
                continue

            # 先按关系检索
            for relation in target_relations:
                neighbors = self.kg.get_neighbors(entity, relation=relation, direction=search_direction)
                if neighbors:
                    return self._join_neighbors(neighbors, limit=5)

            # 再做无关系约束检索（保底）
            neighbors = self.kg.get_neighbors(entity, relation=None, direction=search_direction)
            if neighbors:
                return self._join_neighbors(neighbors, limit=5)
        
        return None

    def _join_neighbors(self, neighbors: List[Tuple[str, str]], limit: int = 5) -> str:
        values: List[str] = []
        seen = set()
        for item in neighbors:
            if not item or not item[0]:
                continue
            text = str(item[0]).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            values.append(text)
            if len(values) >= limit:
                break
        return ", ".join(values)

    def _extract_candidate_options(self, candidate: Optional[str], source: Optional[str]) -> List[str]:
        if not candidate or source != "graph":
            return []

        parts = [p.strip() for p in str(candidate).split(",") if p.strip()]
        options: List[str] = []
        seen = set()
        for part in parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            options.append(part)
            if len(options) >= 8:
                break
        return options

    def _normalize_query_input(self, query: Union[str, Dict[str, Any]]) -> Tuple[str, List[str]]:
        if isinstance(query, str):
            return query, []

        question = str(query.get("question", "")).strip()
        hints: List[str] = []
        for key in ("qid_topic_entity", "topic_entity"):
            value = query.get(key, {})
            if isinstance(value, dict):
                hints.extend(str(k) for k in value.keys())
                hints.extend(str(v) for v in value.values())

        deduped = []
        seen = set()
        for hint in hints:
            text = hint.strip()
            norm = text.lower()
            if text and norm not in seen:
                seen.add(norm)
                deduped.append(text)

        return question, deduped

    def _is_short_answer(self, text: str) -> bool:
        words = text.strip().split()
        return 1 <= len(words) <= 10

    def get_tier_usage(self):
        """获取各层的使用统计"""
        return self.tier_usage
