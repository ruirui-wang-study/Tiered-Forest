import os
from typing import List, Dict, Any

class MetaQALoader:
    """MetaQA 数据集加载器"""
    
    def __init__(self, data_dir: str = "data/metaqa"):
        self.data_dir = data_dir
        self.qa_dev_path = os.path.join(data_dir, "qa_dev.txt")
        self.qa_test_path = os.path.join(data_dir, "qa_test.txt")
    
    def load_dev(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        加载验证集
        
        Args:
            limit: 最大加载数量
            
        Returns:
            问题列表
        """
        return self._load_qa_file(self.qa_dev_path, limit, dataset_name="MetaQA-Dev")
    
    def load_test(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        加载测试集
        
        Args:
            limit: 最大加载数量
            
        Returns:
            问题列表
        """
        return self._load_qa_file(self.qa_test_path, limit, dataset_name="MetaQA-Test")
    
    def _load_qa_file(self, file_path: str, limit: int, dataset_name: str) -> List[Dict[str, Any]]:
        """
        从文件加载问答对
        
        格式: question\tanswer1|answer2|...
        """
        questions = []
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found!")
            return questions
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if len(questions) >= limit:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                question = parts[0].strip()
                # 答案可能有多个，用 | 分隔，我们取第一个作为 ground truth
                answers = parts[1].split('|')
                ground_truth = answers[0].strip()
                
                questions.append({
                    "id": idx,
                    "dataset": dataset_name,
                    "question": question,
                    "ground_truth": ground_truth,
                    "all_answers": [a.strip() for a in answers]  # 保留所有答案用于评估
                })
        
        print(f"Loaded {len(questions)} questions from {dataset_name}")
        return questions
