"""
Simple Scoring Function for FrugalGPT

简化版评分函数，使用启发式规则评估LLM答案的质量
不需要训练，基于以下规则:
1. 答案长度合理性
2. 答案完整性（是否包含实体/数字）
3. 答案置信度（是否包含不确定词汇）
4. 答案格式（是否包含错误信息）

在完整版实现中，应该使用训练好的DistilBERT模型
"""

import re
from typing import Tuple


class SimpleScorer:
    """
    简化版答案质量评分器
    
    返回0-1之间的分数，表示答案的可靠性:
    - 1.0: 非常可靠
    - 0.5: 中等可靠
    - 0.0: 不可靠
    """
    
    def __init__(self):
        # 不确定性词汇（降低分数）
        self.uncertainty_words = [
            "maybe", "perhaps", "possibly", "might", "could be",
            "i think", "i'm not sure", "unclear", "uncertain",
            "may be", "probably", "likely", "seems like",
            "可能", "也许", "大概", "不确定", "不清楚", "似乎"
        ]
        
        # 错误指示词（大幅降低分数）
        self.error_indicators = [
            "error", "failed", "cannot", "unable to", "don't know",
            "no information", "not found", "sorry",
            "错误", "失败", "无法", "不知道", "没有信息", "找不到", "抱歉"
        ]
        
        # 积极指示词（提高分数）
        self.positive_indicators = [
            "the answer is", "based on", "according to",
            "specifically", "exactly", "definitely",
            "答案是", "根据", "确切地", "具体来说", "肯定"
        ]
    
    def score(self, question: str, answer: str) -> float:
        """
        评估答案质量
        
        Args:
            question: 问题文本
            answer: LLM生成的答案
            
        Returns:
            质量分数 [0.0, 1.0]
        """
        if not answer or len(answer.strip()) == 0:
            return 0.0
        
        answer_lower = answer.lower()
        score = 0.5  # 基础分数
        
        # 规则1: 检查错误指示词（-0.4）
        if any(indicator in answer_lower for indicator in self.error_indicators):
            score -= 0.4
        
        # 规则2: 检查不确定性词汇（-0.2）
        uncertainty_count = sum(
            1 for word in self.uncertainty_words if word in answer_lower
        )
        score -= min(0.2, uncertainty_count * 0.1)
        
        # 规则3: 检查积极指示词（+0.2）
        if any(indicator in answer_lower for indicator in self.positive_indicators):
            score += 0.2
        
        # 规则4: 答案长度合理性
        answer_length = len(answer.split())
        if answer_length < 2:
            # 答案太短（可能是"不知道"）
            score -= 0.2
        elif 3 <= answer_length <= 50:
            # 答案长度合理
            score += 0.1
        elif answer_length > 100:
            # 答案太长（可能包含解释而非直接答案）
            score -= 0.1
        
        # 规则5: 检查是否包含实体或数字（+0.2）
        if self._contains_entity_or_number(answer):
            score += 0.2
        
        # 规则6: 检查答案是否直接回答问题（+0.1）
        if self._is_direct_answer(question, answer):
            score += 0.1
        
        # 限制分数范围在 [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _contains_entity_or_number(self, answer: str) -> bool:
        """
        检查答案是否包含实体或数字
        
        Args:
            answer: 答案文本
            
        Returns:
            是否包含实体或数字
        """
        # 检查数字
        if re.search(r'\d+', answer):
            return True
        
        # 检查大写开头的词（可能是实体）
        words = answer.split()
        capitalized_words = [w for w in words if w and w[0].isupper()]
        
        # 如果有多个大写词（排除句首），可能包含实体
        if len(capitalized_words) >= 2:
            return True
        
        return False
    
    def _is_direct_answer(self, question: str, answer: str) -> bool:
        """
        检查答案是否直接回答问题
        
        简单启发式:
        - 如果问题是"who"，答案应该包含人名
        - 如果问题是"when"，答案应该包含时间
        - 如果问题是"where"，答案应该包含地点
        - 如果问题是"how many"，答案应该包含数字
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            是否直接回答
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # "how many" 问题应该包含数字
        if "how many" in question_lower or "多少" in question:
            if re.search(r'\d+', answer):
                return True
        
        # "when" 问题应该包含时间词
        if "when" in question_lower or "什么时候" in question or "何时" in question:
            time_words = ["year", "month", "day", "年", "月", "日", "时"]
            if any(word in answer_lower for word in time_words):
                return True
        
        # "where" 问题应该包含地点词
        if "where" in question_lower or "哪里" in question or "何处" in question:
            location_words = ["in", "at", "on", "在", "于"]
            if any(word in answer_lower for word in location_words):
                return True
        
        # 默认返回False（保守策略）
        return False
    
    def explain_score(self, question: str, answer: str) -> Tuple[float, dict]:
        """
        评估答案质量并返回详细解释
        
        Args:
            question: 问题文本
            answer: LLM生成的答案
            
        Returns:
            (score, explanation_dict)
        """
        score = self.score(question, answer)
        
        explanation = {
            "final_score": score,
            "has_error": any(ind in answer.lower() for ind in self.error_indicators),
            "has_uncertainty": any(word in answer.lower() for word in self.uncertainty_words),
            "has_positive": any(ind in answer.lower() for ind in self.positive_indicators),
            "answer_length": len(answer.split()),
            "has_entity_or_number": self._contains_entity_or_number(answer),
            "is_direct": self._is_direct_answer(question, answer)
        }
        
        return score, explanation


# 示例用法
if __name__ == "__main__":
    scorer = SimpleScorer()
    
    # 测试案例
    test_cases = [
        ("Who directed Inception?", "Christopher Nolan directed Inception."),
        ("Who directed Inception?", "I'm not sure, but it might be Christopher Nolan."),
        ("Who directed Inception?", "Error: Cannot find information."),
        ("How many Oscars did Titanic win?", "Titanic won 11 Oscars."),
        ("How many Oscars did Titanic win?", "It won many awards."),
    ]
    
    print("=== FrugalGPT Scorer Test ===\n")
    for question, answer in test_cases:
        score, explanation = scorer.explain_score(question, answer)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Score: {score:.3f}")
        print(f"Details: {explanation}")
        print()
