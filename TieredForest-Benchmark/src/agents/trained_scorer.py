"""
训练好的评分器 - 使用 DistilBERT 模型

替换 SimpleScorer，提供更准确的答案质量评估
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Tuple
import os


class TrainedScorer:
    """
    训练好的 DistilBERT 评分模型
    
    输入: (question, answer)
    输出: 质量分数 [0, 1]
    """
    
    def __init__(self, model_path: str):
        """
        初始化评分器
        
        Args:
            model_path: 训练好的模型路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型路径不存在: {model_path}\n"
                f"请先运行训练脚本:\n"
                f"  python collect_training_data.py\n"
                f"  python train_scorer.py --data data/training/scorer_training_data.json"
            )
        
        print(f"加载训练好的评分模型: {model_path}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式
        
        # 检查是否有 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"  设备: {self.device}")
        print(f"  模型加载完成")
    
    def score(self, question: str, answer: str) -> float:
        """
        评估答案质量
        
        Args:
            question: 问题文本
            answer: LLM生成的答案
            
        Returns:
            质量分数 [0.0, 1.0]
        """
        # 拼接输入
        text = f"{question} [SEP] {answer}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 sigmoid 将输出转换为 [0, 1]
            score = torch.sigmoid(outputs.logits).item()
        
        # 限制范围
        score = max(0.0, min(1.0, score))
        
        return score
    
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
        
        # 简单的解释（可以扩展）
        explanation = {
            "final_score": score,
            "confidence": "high" if score > 0.7 or score < 0.3 else "medium",
            "recommendation": "accept" if score >= 0.5 else "reject",
            "model_type": "DistilBERT (trained)"
        }
        
        return score, explanation
    
    def batch_score(self, questions: list, answers: list) -> list:
        """
        批量评分（更高效）
        
        Args:
            questions: 问题列表
            answers: 答案列表
            
        Returns:
            分数列表
        """
        assert len(questions) == len(answers), "问题和答案数量必须相同"
        
        # 拼接输入
        texts = [f"{q} [SEP] {a}" for q, a in zip(questions, answers)]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().tolist()
        
        # 如果只有一个样本，转换为列表
        if isinstance(scores, float):
            scores = [scores]
        
        # 限制范围
        scores = [max(0.0, min(1.0, s)) for s in scores]
        
        return scores


# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python trained_scorer.py <model_path>")
        print("示例: python trained_scorer.py models/frugal_scorer")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # 创建评分器
    scorer = TrainedScorer(model_path)
    
    # 测试案例
    test_cases = [
        ("Who directed Inception?", "Christopher Nolan directed Inception."),
        ("Who directed Inception?", "I'm not sure, but it might be Christopher Nolan."),
        ("Who directed Inception?", "Error: Cannot find information."),
        ("What is 2+2?", "4"),
        ("What is 2+2?", "The answer is unclear."),
    ]
    
    print("\n" + "=" * 60)
    print("测试训练好的评分器")
    print("=" * 60)
    
    for question, answer in test_cases:
        score, explanation = scorer.explain_score(question, answer)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Score: {score:.3f}")
        print(f"Recommendation: {explanation['recommendation']}")
    
    # 测试批量评分
    print("\n" + "=" * 60)
    print("测试批量评分")
    print("=" * 60)
    
    questions = [tc[0] for tc in test_cases]
    answers = [tc[1] for tc in test_cases]
    
    scores = scorer.batch_score(questions, answers)
    
    for q, a, s in zip(questions, answers, scores):
        print(f"\nQ: {q}")
        print(f"A: {a}")
        print(f"Score: {s:.3f}")
