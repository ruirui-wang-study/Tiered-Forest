import re
import time
from typing import Optional, Tuple, List
from ..graph_engine import MetaQAGraphEngine

class SymbolicPruner:
    """
    Tier 1: 符号层 / 规则引擎（增强版）
    零成本的快速过滤层
    
    策略:
    1. 精确匹配缓存的问答对
    2. 基于模板的规则匹配（扩展）
    3. 图谱直接查询（关系感知）
    """
    
    def __init__(self, graph_engine: MetaQAGraphEngine):
        self.graph = graph_engine
        self.rules = self._load_rules()
        self.cache = {}  # 简单的问答缓存
    
    def _load_rules(self):
        """
        加载规则模板（大幅扩展）
        
        每个规则包含:
        - pattern: 正则表达式模板
        - relation: 对应的图谱关系
        - direction: 查询方向 (in/out)
        """
        return [
            # 导演相关
            {
                "pattern": re.compile(r"who directed (.+?)[\?]?$", re.IGNORECASE),
                "relation": "directed_by",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"who (?:is|was) the director of (.+?)[\?]?$", re.IGNORECASE),
                "relation": "directed_by",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies) did (.+?) direct[\?]?$", re.IGNORECASE),
                "relation": "directed_by",
                "direction": "in",
                "entity_group": 1
            },
            
            # 演员相关
            {
                "pattern": re.compile(r"who (?:acted in|starred in|was in) (.+?)[\?]?$", re.IGNORECASE),
                "relation": "starred_actors",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies) did (.+?) (?:act in|star in|appear in)[\?]?$", re.IGNORECASE),
                "relation": "starred_actors",
                "direction": "in",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies|roles) (?:did|has) (.+?) (?:acted|starred|appeared) in[\?]?$", re.IGNORECASE),
                "relation": "starred_actors",
                "direction": "in",
                "entity_group": 1
            },
            
            # 类型相关
            {
                "pattern": re.compile(r"what (?:is the )?genre (?:of |is )?(.+?)[\?]?$", re.IGNORECASE),
                "relation": "has_genre",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what type of (?:film|movie) is (.+?)[\?]?$", re.IGNORECASE),
                "relation": "has_genre",
                "direction": "out",
                "entity_group": 1
            },
            
            # 编剧相关
            {
                "pattern": re.compile(r"who wrote (.+?)[\?]?$", re.IGNORECASE),
                "relation": "written_by",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"who (?:is|was) the writer of (.+?)[\?]?$", re.IGNORECASE),
                "relation": "written_by",
                "direction": "out",
                "entity_group": 1
            },
            
            # 发行年份相关
            {
                "pattern": re.compile(r"when was (.+?) released[\?]?$", re.IGNORECASE),
                "relation": "release_year",
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what year (?:was|did) (.+?) (?:come out|release)[\?]?$", re.IGNORECASE),
                "relation": "release_year",
                "direction": "out",
                "entity_group": 1
            },
            
            # 语言相关
            {
                "pattern": re.compile(r"what language is (.+?) in[\?]?$", re.IGNORECASE),
                "relation": "in_language",
                "direction": "out",
                "entity_group": 1
            },
        ]
    
    def check(self, question: str) -> Tuple[Optional[str], float]:
        """
        检查是否可以通过符号层回答
        
        Args:
            question: 问题文本
            
        Returns:
            (answer, confidence): 答案和置信度，如果无法回答则返回 (None, 0.0)
        """
        # 策略1: 缓存查询
        if question in self.cache:
            return self.cache[question], 1.0
        
        # 策略2: 规则匹配 + 图谱查询
        rule_answer = self._match_rules(question)
        if rule_answer:
            self.cache[question] = rule_answer
            return rule_answer, 0.95  # 高置信度
        
        # 策略3: 图谱简单查询（fallback）
        graph_answer = self.graph.query_simple(question)
        if graph_answer:
            self.cache[question] = graph_answer
            return graph_answer, 0.85  # 中等置信度
        
        # 无法回答
        return None, 0.0
    
    def _match_rules(self, question: str) -> Optional[str]:
        """
        基于规则模板匹配（增强版）
        
        流程:
        1. 遍历所有规则模板
        2. 匹配问题模式
        3. 提取实体
        4. 查询图谱获取答案
        """
        question_clean = question.strip()
        
        for rule in self.rules:
            match = rule["pattern"].match(question_clean)
            if match:
                # 提取实体名称
                entity_text = match.group(rule["entity_group"]).strip()
                
                # 清理实体文本（去除冠词等）
                entity_text = self._clean_entity(entity_text)
                
                # 在图谱中查找实体
                entity = self.graph.find_entity(entity_text)
                if not entity:
                    # 尝试多词组合
                    entity = self._find_entity_fuzzy(entity_text)
                
                if entity:
                    # 根据规则查询图谱
                    neighbors = self.graph.get_neighbors(
                        entity,
                        relation=rule["relation"],
                        direction=rule["direction"]
                    )
                    
                    if neighbors:
                        # 返回前几个答案（用逗号分隔）
                        answers = [n[0] for n in neighbors[:3]]
                        return ", ".join(answers)
        
        return None
    
    def _clean_entity(self, text: str) -> str:
        """清理实体文本"""
        # 去除常见的冠词和介词
        articles = ['the', 'a', 'an']
        words = text.lower().split()
        cleaned = [w for w in words if w not in articles]
        return " ".join(cleaned)
    
    def _find_entity_fuzzy(self, text: str) -> Optional[str]:
        """
        模糊查找实体
        
        策略:
        1. 尝试完整文本
        2. 尝试去除冠词
        3. 尝试首字母大写
        4. 尝试全小写
        """
        # 尝试1: 原文
        entity = self.graph.find_entity(text)
        if entity:
            return entity
        
        # 尝试2: 清理后的文本
        cleaned = self._clean_entity(text)
        entity = self.graph.find_entity(cleaned)
        if entity:
            return entity
        
        # 尝试3: 首字母大写（人名/电影名）
        capitalized = text.title()
        entity = self.graph.find_entity(capitalized)
        if entity:
            return entity
        
        # 尝试4: 全小写
        lowercased = text.lower()
        entity = self.graph.find_entity(lowercased)
        if entity:
            return entity
        
        return None
