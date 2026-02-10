#!/usr/bin/env python3
"""
小规模测试 - 验证改进后的三层路由
测试 20 个问题，分析 Tier 使用分布和成本
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cost_monitor import CostMonitor
from src.graph_engine import MetaQAGraphEngine
from src.data_loader import MetaQALoader
from src.utils.cache_manager import LLMCache
from src.utils.logger import setup_logger
from src.agents.forest_agent import TieredForestAgent

def evaluate_accuracy(prediction: str, ground_truths: list) -> bool:
    """评估准确性"""
    pred_lower = prediction.lower().strip()
    for gt in ground_truths:
        if gt.lower().strip() in pred_lower:
            return True
    return False

def run_small_test():
    """运行小规模测试"""
    print("="*80)
    print("小规模测试 - 改进后的三层路由")
    print("="*80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "metaqa")
    kb_path = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    cache_dir = os.path.join(base_dir, "data", "processed")
    llm_cache_path = os.path.join(base_dir, "data", "cache", "llm_responses.json")
    
    # 设置日志
    logger = setup_logger("small_test", "logs/small_test.log")
    
    # 加载数据
    print("\n1. 加载数据...")
    loader = MetaQALoader(data_dir)
    val_dataset = loader.load_dev()[:20]  # 只测试 20 个问题
    print(f"   ✓ 加载了 {len(val_dataset)} 个问题")
    
    # 加载组件
    print("\n2. 加载组件...")
    graph_engine = MetaQAGraphEngine(kb_path, cache_dir)
    cache_manager = LLMCache(llm_cache_path)
    
    # 测试不同的阈值组合
    threshold_configs = [
        {"t_drop": 0.2, "t_pass": 0.5, "name": "宽松 (0.2, 0.5)"},
        {"t_drop": 0.3, "t_pass": 0.6, "name": "中等 (0.3, 0.6)"},
        {"t_drop": 0.4, "t_pass": 0.7, "name": "严格 (0.4, 0.7)"},
    ]
    
    all_results = []
    
    for config in threshold_configs:
        print(f"\n{'='*80}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*80}")
        
        # 初始化 Agent
        monitor = CostMonitor()
        agent = TieredForestAgent(
            monitor=monitor,
            graph_engine=graph_engine,
            cache_manager=cache_manager,
            t_drop=config['t_drop'],
            t_pass=config['t_pass']
        )
        
        # 运行测试
        correct_count = 0
        results = []
        
        for item in tqdm(val_dataset, desc=f"处理问题"):
            question = item['question']
            ground_truths = item['all_answers']
            
            try:
                # 求解
                prediction = agent.solve(question)
                is_correct = evaluate_accuracy(prediction, ground_truths)
                
                if is_correct:
                    correct_count += 1
                
                # 记录结果
                results.append({
                    'question': question,
                    'prediction': prediction[:100],  # 截断
                    'ground_truth': ground_truths[0],
                    'correct': is_correct
                })
                
            except Exception as e:
                logger.error(f"Error on question '{question}': {e}")
                results.append({
                    'question': question,
                    'prediction': f"ERROR: {e}",
                    'ground_truth': ground_truths[0],
                    'correct': False
                })
        
        # 统计
        accuracy = correct_count / len(val_dataset)
        tier_usage = agent.get_tier_usage()
        stats = monitor.get_session_stats()
        
        # 打印结果
        print(f"\n结果:")
        print(f"  准确率: {accuracy:.2%} ({correct_count}/{len(val_dataset)})")
        print(f"\nTier 使用分布:")
        print(f"  Tier 1 (Symbolic): {tier_usage['tier1']} ({tier_usage['tier1']/len(val_dataset)*100:.1f}%)")
        print(f"  Tier 2 (Semantic): {tier_usage['tier2']} ({tier_usage['tier2']/len(val_dataset)*100:.1f}%)")
        print(f"  Tier 3 (LLM): {tier_usage['tier3']} ({tier_usage['tier3']/len(val_dataset)*100:.1f}%)")
        print(f"\n成本统计:")
        print(f"  总 Token: {stats['tokens_total']}")
        print(f"  总成本: ${stats['cost_usd']:.6f}")
        print(f"  平均成本/问题: ${stats['cost_usd']/len(val_dataset):.6f}")
        print(f"  平均延迟: {stats['latency_avg']:.2f}s")
        
        # 保存到总结果
        all_results.append({
            'config': config['name'],
            't_drop': config['t_drop'],
            't_pass': config['t_pass'],
            'accuracy': accuracy,
            'tier1_pct': tier_usage['tier1']/len(val_dataset)*100,
            'tier2_pct': tier_usage['tier2']/len(val_dataset)*100,
            'tier3_pct': tier_usage['tier3']/len(val_dataset)*100,
            'total_cost': stats['cost_usd'],
            'avg_cost': stats['cost_usd']/len(val_dataset),
            'total_tokens': stats['tokens_total'],
            'avg_latency': stats['latency_avg']
        })
    
    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    output_path = "results/small_test_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ 结果已保存: {output_path}")
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")
    print(f"\n{'配置':<20} {'准确率':<10} {'Tier1%':<10} {'Tier2%':<10} {'Tier3%':<10} {'总成本':<12} {'平均成本':<12}")
    print("-" * 90)
    for result in all_results:
        print(f"{result['config']:<20} "
              f"{result['accuracy']:<10.2%} "
              f"{result['tier1_pct']:<10.1f} "
              f"{result['tier2_pct']:<10.1f} "
              f"{result['tier3_pct']:<10.1f} "
              f"${result['total_cost']:<11.6f} "
              f"${result['avg_cost']:<11.6f}")
    
    # 找出最佳配置
    best_config = max(all_results, key=lambda x: x['accuracy'] / (x['total_cost'] + 0.001))
    print(f"\n最佳配置 (Pareto 最优):")
    print(f"  {best_config['config']}")
    print(f"  准确率: {best_config['accuracy']:.2%}")
    print(f"  总成本: ${best_config['total_cost']:.6f}")
    print(f"  Tier 2 使用率: {best_config['tier2_pct']:.1f}%")
    
    # 保存缓存
    cache_manager.save()
    cache_stats = cache_manager.get_stats()
    print(f"\n缓存统计: 命中率={cache_stats['hit_rate']:.2%}, 大小={cache_stats['cache_size']}")
    
    print(f"\n{'='*80}")
    print("测试完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_small_test()
