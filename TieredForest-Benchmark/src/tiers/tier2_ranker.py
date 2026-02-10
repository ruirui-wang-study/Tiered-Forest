import time
import numpy as np
from typing import List, Tuple
from sentence_transformers import CrossEncoder

class SemanticRanker:
    """
    Tier 2: 语义打分层
    使用 CrossEncoder 对候选答案进行打分
    成本: 本地推理，可忽略
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        """
        初始化 CrossEncoder
        
        Args:
            model_name: HuggingFace 模型名称
        """
        try:
            self.encoder = CrossEncoder(model_name)
            print(f"Loaded CrossEncoder: {model_name}")
        except Exception as e:
            print(f"Warning: Failed to load CrossEncoder: {e}")
            self.encoder = None
    
    def score_candidate(self, question: str, candidate_answer: str) -> float:
        """
        对单个候选答案打分
        
        Args:
            question: 问题文本
            candidate_answer: 候选答案
            
        Returns:
            置信度分数 [0, 1]
        """
        if self.encoder is None:
            # Fallback: 简单的启发式评分
            return self._heuristic_score(question, candidate_answer)
        
        try:
            # CrossEncoder 预测
            score = self.encoder.predict([(question, candidate_answer)])[0]
            # Sigmoid 归一化到 [0, 1]
            normalized_score = 1 / (1 + np.exp(-score))
            return float(normalized_score)
        except Exception as e:
            print(f"Error in scoring: {e}")
            return 0.5
    
    def rank_candidates(self, question: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        对多个候选答案排序
        
        Args:
            question: 问题文本
            candidates: 候选答案列表
            
        Returns:
            [(candidate, score), ...] 按分数降序排列
        """
        if not candidates:
            return []
        
        scores = [self.score_candidate(question, cand) for cand in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked
    
    def _heuristic_score(self, question: str, answer: str) -> float:
        """
        启发式评分 (当 CrossEncoder 不可用时)
        
        简单策略:
        - 答案长度适中: +0.2
        - 答案包含问题中的关键词: +0.3
        - 基础分: 0.5
        """
        score = 0.5
        
        # 答案长度检查
        answer_len = len(answer.split())
        if 2 <= answer_len <= 10:
            score += 0.2
        
        # 关键词重叠
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words)
        if overlap > 0:
            score += min(0.3, overlap * 0.1)
        
        return min(1.0, score)
