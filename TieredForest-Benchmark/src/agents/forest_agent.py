from .base_agent import BaseAgent
from ..cost_monitor import CostMonitor
from ..graph_engine import MetaQAGraphEngine
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
                 graph_engine: MetaQAGraphEngine,
                 cache_manager: LLMCache,
                 t_drop: float = 0.2,
                 t_pass: float = 0.7):
        """
        初始化 Tiered-Forest Agent
        
        Args:
            monitor: 成本监控器
            graph_engine: 图谱引擎
            cache_manager: LLM 缓存管理器
            t_drop: 丢弃阈值（低于此分数拒绝候选）
            t_pass: 通过阈值（高于此分数直接返回）
        """
        super().__init__(monitor)
        self.t_drop = t_drop
        self.t_pass = t_pass
        
        # 初始化三层
        self.tier1 = SymbolicPruner(graph_engine)
        self.tier2 = SemanticRanker()
        self.tier3 = LLMReasoner(cache_manager)
        
        # 初始化小模型生成器（用于 Tier 2 候选生成）
        self.small_model = SmallModelGenerator()
        
        # 统计各层的使用情况
        self.tier_usage = {"tier1": 0, "tier2": 0, "tier3": 0}
    
    def solve(self, question: str) -> str:
        """
        三层路由求解
        
        Args:
            question: 问题文本
            
        Returns:
            答案字符串
        """
        # --- Tier 1: Symbolic Layer ---
        answer, confidence = self.tier1.check(question)
        if answer and confidence > 0.8:
            # Tier 1 成功，零成本
            self.monitor.record_usage(0, 0, 0.001, "symbolic")
            self.tier_usage["tier1"] += 1
            return answer
        
        # --- Tier 2: Semantic Layer ---
        # 生成候选答案（图谱或小模型）
        candidate, source = self._generate_candidate(question)
        
        if candidate:
            # 使用 CrossEncoder 打分
            score = self.tier2.score_candidate(question, candidate)
            
            # 路由决策
            if score > self.t_pass:
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
            answer, prompt_tokens, completion_tokens, latency = self.tier3.reason(question)
            self.monitor.record_usage(prompt_tokens, completion_tokens, latency, "large")
            self.tier_usage["tier3"] += 1
            return answer
        except Exception as e:
            print(f"Tier 3 failed: {e}")
            return "Error: Unable to answer"
    
    def _generate_candidate(self, question: str) -> tuple:
        """
        生成候选答案（增强版）
        
        策略（按优先级）:
        1. 图谱查询（零成本）
        2. 小模型生成（低成本）
        
        Returns:
            (candidate_answer, source) 或 (None, None)
        """
        # 策略1: 尝试从图谱获取候选
        graph_candidate = self._graph_candidate(question)
        if graph_candidate:
            return graph_candidate, "graph"
        
        # 策略2: 使用小模型生成候选
        if self.small_model.is_available():
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
    
    def _graph_candidate(self, question: str) -> str:
        """
        从图谱生成候选答案（改进版）
        
        策略:
        1. 分析问题类型，确定目标关系
        2. 提取问题中的实体
        3. 查询图谱获取特定关系的邻居
        4. 返回最相关的邻居
        """
        question_lower = question.lower()
        
        # 根据问题模式确定目标关系类型
        target_relation = None
        if "act in" in question_lower or "acted in" in question_lower or "starred in" in question_lower:
            target_relation = "starred_actors"  # 反向查询
            search_direction = "in"  # 查询入边
        elif "directed" in question_lower:
            target_relation = "directed_by"
            search_direction = "out"
        elif "genre" in question_lower:
            target_relation = "has_genre"
            search_direction = "out"
        elif "written by" in question_lower or "wrote" in question_lower:
            target_relation = "written_by"
            search_direction = "out"
        else:
            # 默认：双向查询
            target_relation = None
            search_direction = "both"
        
        # 提取实体
        words = question_lower.split()
        stop_words = {'what', 'who', 'when', 'where', 'which', 'how', 'did', 'does', 
                     'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'of'}
        
        # 尝试多词组合（优先，更准确）
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if words[i] not in stop_words and words[i+1] not in stop_words:
                entity = self.tier1.graph.find_entity(phrase)
                if entity:
                    neighbors = self.tier1.graph.get_neighbors(entity, relation=target_relation, direction=search_direction)
                    if neighbors:
                        # 返回前几个邻居
                        top_neighbors = [n[0] for n in neighbors[:3]]
                        return ", ".join(top_neighbors)
        
        # 尝试单词
        for word in words:
            if len(word) > 3 and word not in stop_words:
                entity = self.tier1.graph.find_entity(word)
                if entity:
                    neighbors = self.tier1.graph.get_neighbors(entity, relation=target_relation, direction=search_direction)
                    if neighbors:
                        # 返回前几个邻居
                        top_neighbors = [n[0] for n in neighbors[:3]]
                        return ", ".join(top_neighbors)
        
        return None
    
    def get_tier_usage(self):
        """获取各层的使用统计"""
        return self.tier_usage
