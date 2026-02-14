import re
import time
from typing import List, Optional, Tuple

from ..kg.base_backend import BaseKGBackend

class SymbolicPruner:
    """
    Tier 1: 符号层 / 规则引擎（增强版）
    零成本的快速过滤层
    
    策略:
    1. 精确匹配缓存的问答对
    2. 基于模板的规则匹配（扩展）
    3. 图谱直接查询（关系感知）
    """
    
    def __init__(self, graph_engine: BaseKGBackend):
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
                "relations": ["directed_by", "director", "P57"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"who (?:is|was) the director of (.+?)[\?]?$", re.IGNORECASE),
                "relations": ["directed_by", "director", "P57"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies) did (.+?) direct[\?]?$", re.IGNORECASE),
                "relations": ["directed_by", "director", "P57"],
                "direction": "in",
                "entity_group": 1
            },
            
            # 演员相关
            {
                "pattern": re.compile(r"who (?:acted in|starred in|was in) (.+?)[\?]?$", re.IGNORECASE),
                "relations": ["starred_actors", "cast member", "performer", "P161", "P175"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies) did (.+?) (?:act in|star in|appear in)[\?]?$", re.IGNORECASE),
                "relations": ["starred_actors", "cast member", "performer", "P161", "P175"],
                "direction": "in",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what (?:films|movies|roles) (?:did|has) (.+?) (?:acted|starred|appeared) in[\?]?$", re.IGNORECASE),
                "relations": ["starred_actors", "cast member", "performer", "P161", "P175"],
                "direction": "in",
                "entity_group": 1
            },
            
            # 类型相关
            {
                "pattern": re.compile(r"what (?:is the )?genre (?:of |is )?(.+?)[\?]?$", re.IGNORECASE),
                "relations": ["has_genre", "genre", "P136"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what type of (?:film|movie) is (.+?)[\?]?$", re.IGNORECASE),
                "relations": ["has_genre", "genre", "P136"],
                "direction": "out",
                "entity_group": 1
            },
            
            # 编剧相关
            {
                "pattern": re.compile(r"who wrote (.+?)[\?]?$", re.IGNORECASE),
                "relations": ["written_by", "author", "screenwriter", "P50", "P58"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"who (?:is|was) the writer of (.+?)[\?]?$", re.IGNORECASE),
                "relations": ["written_by", "author", "screenwriter", "P50", "P58"],
                "direction": "out",
                "entity_group": 1
            },
            
            # 发行年份相关
            {
                "pattern": re.compile(r"when was (.+?) released[\?]?$", re.IGNORECASE),
                "relations": ["release_year", "publication date", "date of publication", "inception", "P577"],
                "direction": "out",
                "entity_group": 1
            },
            {
                "pattern": re.compile(r"what year (?:was|did) (.+?) (?:come out|release)[\?]?$", re.IGNORECASE),
                "relations": ["release_year", "publication date", "date of publication", "inception", "P577"],
                "direction": "out",
                "entity_group": 1
            },
            
            # 语言相关
            {
                "pattern": re.compile(r"what language is (.+?) in[\?]?$", re.IGNORECASE),
                "relations": ["in_language", "original language of film or TV show", "language of work or name", "P364"],
                "direction": "out",
                "entity_group": 1
            },
        ]
    
    def check(self, question: str, entity_hints: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
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
        rule_answer = self._match_rules(question, entity_hints=entity_hints)
        if rule_answer:
            self.cache[question] = rule_answer
            return rule_answer, 0.95  # 高置信度

        # 策略2.5: entity hint + 关键词关系映射（适配WebQSP非模板问句）
        hinted_answer = self._hinted_relation_lookup(question, entity_hints=entity_hints)
        if hinted_answer:
            self.cache[question] = hinted_answer
            return hinted_answer, 0.82
        
        # 策略3: 图谱简单查询（fallback）
        graph_answer = self.graph.query_simple(question)
        if graph_answer:
            self.cache[question] = graph_answer
            return graph_answer, 0.85  # 中等置信度
        
        # 无法回答
        return None, 0.0
    
    def _match_rules(self, question: str, entity_hints: Optional[List[str]] = None) -> Optional[str]:
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
            if not match:
                continue

            # 候选实体：优先外部hint，再用规则抽取实体
            candidates: List[str] = []
            if entity_hints:
                candidates.extend(h.strip() for h in entity_hints if h and h.strip())

            raw_entity_text = match.group(rule["entity_group"]).strip()
            cleaned_entity_text = self._clean_entity(raw_entity_text)
            if raw_entity_text:
                candidates.append(raw_entity_text)
            if cleaned_entity_text and cleaned_entity_text != raw_entity_text:
                candidates.append(cleaned_entity_text)

            # 去重
            deduped_candidates: List[str] = []
            seen = set()
            for text in candidates:
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    deduped_candidates.append(text)

            for entity_text in deduped_candidates:
                entity = self.graph.find_entity(entity_text)
                if not entity:
                    entity = self._find_entity_fuzzy(entity_text)
                if not entity:
                    continue

                for relation in self._rule_relations(rule):
                    neighbors = self.graph.get_neighbors(
                        entity,
                        relation=relation,
                        direction=rule["direction"],
                    )
                    if neighbors:
                        answers = [n[0] for n in neighbors[:3] if n and n[0]]
                        if answers:
                            return ", ".join(answers)

        return None

    def _hinted_relation_lookup(
        self, question: str, entity_hints: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Controlled symbolic fallback using dataset-provided entity hints.
        Only runs when clear relation keywords exist in question.
        """
        if not entity_hints:
            return None

        question_lower = question.lower()
        relation_candidates: List[str] = []
        direction = "out"

        # Existing relation families (MetaQA + Wikidata label/PID aliases).
        if any(k in question_lower for k in ["act in", "acted in", "starred in", "appear in"]):
            relation_candidates.extend(["starred_actors", "cast member", "performer", "P161", "P175"])
            if "what" in question_lower and " did " in question_lower:
                direction = "in"
        if "direct" in question_lower:
            relation_candidates.extend(["directed_by", "director", "P57"])
            if "what" in question_lower and " did " in question_lower:
                direction = "in"
        if any(k in question_lower for k in ["genre", "type of"]):
            relation_candidates.extend(["has_genre", "genre", "P136"])
        if any(k in question_lower for k in ["wrote", "writer", "written by", "author"]):
            relation_candidates.extend(["written_by", "author", "screenwriter", "P50", "P58"])
            if "what" in question_lower and " did " in question_lower:
                direction = "in"
        if any(k in question_lower for k in ["released", "release year", "what year"]):
            relation_candidates.extend(["release_year", "publication date", "date of publication", "inception", "P577"])
        if any(k in question_lower for k in ["language", "speak", "write in"]):
            relation_candidates.extend(
                [
                    "in_language",
                    "original language of film or TV show",
                    "language of work or name",
                    "P364",
                ]
            )

        # No confident relation signal -> don't force Tier1.
        if not relation_candidates:
            return None

        dedup_relations: List[str] = []
        seen_rel = set()
        for rel in relation_candidates:
            key = rel.lower()
            if key not in seen_rel:
                seen_rel.add(key)
                dedup_relations.append(rel)

        for hint in entity_hints:
            entity = self.graph.find_entity(hint)
            if not entity:
                continue
            for relation in dedup_relations:
                neighbors = self.graph.get_neighbors(entity, relation=relation, direction=direction)
                if not neighbors:
                    continue
                answers = [n[0] for n in neighbors[:3] if n and n[0]]
                if answers:
                    return ", ".join(answers)

        return None

    def _rule_relations(self, rule: dict) -> List[str]:
        relations = rule.get("relations")
        if isinstance(relations, list):
            return [str(r).strip() for r in relations if str(r).strip()]

        legacy = rule.get("relation")
        if isinstance(legacy, str) and legacy.strip():
            return [legacy.strip()]

        return []
    
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
