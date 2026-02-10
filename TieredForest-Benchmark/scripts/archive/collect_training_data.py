"""
收集 FrugalGPT 评分模型的训练数据

工作流程:
1. 从 MetaQA 数据集加载问题和答案
2. 使用多个 LLM 生成答案
3. 评估答案的正确性
4. 保存为训练数据集

输出格式:
{
    "question": "Who directed Inception?",
    "answer": "Christopher Nolan",
    "label": 1.0,  # 1.0 = 正确, 0.0 = 错误
    "llm": "gpt-3.5-turbo",
    "ground_truth": "Christopher Nolan"
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.cost_monitor import CostMonitor
from src.utils.cache_manager import LLMCache
from openai import OpenAI
import json
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
import re


class DataCollector:
    """训练数据收集器"""
    
    def __init__(self, cache_file: str = "data/cache/training_data_cache.json"):
        self.cache = LLMCache(cache_file=cache_file)
        self.monitor = CostMonitor()
        
        # 配置 LLM 列表
        self.llms = self._setup_llms()
        
    def _setup_llms(self) -> List[Dict]:
        """设置要使用的 LLM 列表"""
        llms = []
        
        # Small Model
        if config.SMALL_MODEL_API_KEY != "EMPTY":
            llms.append({
                "name": "Small-Model",
                "api_key": config.SMALL_MODEL_API_KEY,
                "base_url": config.SMALL_MODEL_BASE_URL,
                "model": config.SMALL_MODEL_NAME,
                "tier": "small"
            })
        
        # Kimi
        if config.KIMI_API_KEY != "EMPTY":
            llms.append({
                "name": "Kimi",
                "api_key": config.KIMI_API_KEY,
                "base_url": config.KIMI_BASE_URL,
                "model": config.KIMI_MODEL_NAME,
                "tier": "kimi"
            })
        
        # DeepSeek
        llms.append({
            "name": "DeepSeek",
            "api_key": config.DEEPSEEK_API_KEY,
            "base_url": config.DEEPSEEK_BASE_URL,
            "model": config.MODEL_NAME,
            "tier": "large"
        })
        
        return llms
    
    def load_metaqa_data(self, max_samples: int = 200) -> List[Tuple[str, str]]:
        """
        加载 MetaQA 数据集
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            [(question, answer), ...]
        """
        data = []
        
        # 读取测试集
        test_file = "data/MetaQA/qa_test.txt"
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # MetaQA 格式: question\tanswer1|answer2|...
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        question = parts[0].strip()
                        # 取第一个答案作为 ground truth
                        answers = parts[1].split('|')
                        answer = answers[0].strip()
                        
                        data.append((question, answer))
                    
                    if len(data) >= max_samples:
                        break
        
        print(f"加载了 {len(data)} 个问题")
        return data
    
    def call_llm(self, llm_config: Dict, question: str) -> str:
        """
        调用 LLM 生成答案
        
        Args:
            llm_config: LLM 配置
            question: 问题
            
        Returns:
            答案字符串
        """
        prompt = f"Answer this question concisely: {question}"
        
        # 检查缓存
        cache_key = f"{llm_config['model']}:{prompt}"
        cached = self.cache.get(cache_key, llm_config['model'])
        if cached:
            return cached
        
        # 调用 API
        client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url']
        )
        
        try:
            response = client.chat.completions.create(
                model=llm_config['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            # 记录成本
            self.monitor.record_usage(
                usage.prompt_tokens,
                usage.completion_tokens,
                0.0,
                llm_config['tier']
            )
            
            # 写入缓存
            self.cache.set(cache_key, llm_config['model'], answer)
            
            # 避免频繁调用
            time.sleep(0.5)
            
            return answer
            
        except Exception as e:
            print(f"LLM 调用失败 ({llm_config['name']}): {e}")
            return ""
    
    def evaluate_answer(self, answer: str, ground_truth: str) -> float:
        """
        评估答案的正确性
        
        Args:
            answer: LLM 生成的答案
            ground_truth: 正确答案
            
        Returns:
            1.0 (正确) 或 0.0 (错误)
        """
        # 标准化
        answer = answer.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # 移除标点符号
        answer = re.sub(r'[^\w\s]', '', answer)
        ground_truth = re.sub(r'[^\w\s]', '', ground_truth)
        
        # 检查是否包含正确答案
        if ground_truth in answer:
            return 1.0
        
        # 检查词重叠
        answer_words = set(answer.split())
        truth_words = set(ground_truth.split())
        
        if len(truth_words) == 0:
            return 0.0
        
        overlap = len(answer_words & truth_words) / len(truth_words)
        
        # 如果重叠度 >= 0.8，认为正确
        return 1.0 if overlap >= 0.8 else 0.0
    
    def collect(self, max_samples: int = 200) -> List[Dict]:
        """
        收集训练数据
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            训练数据列表
        """
        print("=" * 60)
        print("开始收集训练数据")
        print("=" * 60)
        
        # 加载问题
        qa_pairs = self.load_metaqa_data(max_samples)
        
        training_data = []
        
        print(f"\n使用 {len(self.llms)} 个 LLM:")
        for llm in self.llms:
            print(f"  - {llm['name']}")
        
        print(f"\n开始生成答案...")
        
        # 对每个问题，使用所有 LLM 生成答案
        for question, ground_truth in tqdm(qa_pairs, desc="收集数据"):
            for llm in self.llms:
                # 生成答案
                answer = self.call_llm(llm, question)
                
                if not answer:
                    continue
                
                # 评估正确性
                label = self.evaluate_answer(answer, ground_truth)
                
                # 添加到训练数据
                training_data.append({
                    "question": question,
                    "answer": answer,
                    "label": label,
                    "llm": llm['name'],
                    "ground_truth": ground_truth
                })
        
        print(f"\n收集完成！")
        print(f"  总样本数: {len(training_data)}")
        print(f"  正确样本: {sum(1 for d in training_data if d['label'] == 1.0)}")
        print(f"  错误样本: {sum(1 for d in training_data if d['label'] == 0.0)}")
        
        # 显示成本
        stats = self.monitor.get_session_stats()
        print(f"\n成本统计:")
        print(f"  总 Token: {stats['tokens_total']}")
        print(f"  总成本: ${stats['cost_usd']:.4f}")
        
        return training_data
    
    def save(self, data: List[Dict], output_file: str):
        """保存训练数据"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 训练数据已保存到: {output_file}")
        
        # 保存缓存
        self.cache.save()
        print(f"✓ 缓存已保存")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="收集 FrugalGPT 评分模型训练数据")
    parser.add_argument("--max-samples", type=int, default=200,
                       help="最大样本数 (默认: 200)")
    parser.add_argument("--output", type=str, default="data/training/scorer_training_data.json",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    try:
        # 创建收集器
        collector = DataCollector()
        
        # 收集数据
        training_data = collector.collect(max_samples=args.max_samples)
        
        # 保存数据
        collector.save(training_data, args.output)
        
        print("\n" + "=" * 60)
        print("数据收集完成！")
        print("=" * 60)
        print(f"\n下一步: 运行训练脚本")
        print(f"  python train_scorer.py --data {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n数据收集被用户中断")
    except Exception as e:
        print(f"\n数据收集失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
