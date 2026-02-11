#!/usr/bin/env python3
"""
完整 Benchmark - 对比 Tiered-Forest vs Baselines
测试 50 个问题，对比不同方法的性能
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import time

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cost_monitor import CostMonitor
from src.graph_engine import MetaQAGraphEngine
from src.data_loader import MetaQALoader
from src.utils.cache_manager import LLMCache
from src.utils.logger import setup_logger
from src.agents.forest_agent import TieredForestAgent
from src.agents.naive_agent import NaiveLLMAgent

def evaluate_accuracy(prediction: str, ground_truths: list) -> bool:
    """评估准确性"""
    pred_lower = prediction.lower().strip()
    for gt in ground_truths:
        if gt.lower().strip() in pred_lower:
            return True
    return False

def run_benchmark():
    """运行完整 Benchmark"""
    print("="*80)
    print("完整 Benchmark - Tiered-Forest vs Baselines")
    print("="*80)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "MetaQA", "1-hop", "vanilla")
    kb_path = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    cache_dir = os.path.join(base_dir, "data", "processed")
    llm_cache_path = os.path.join(base_dir, "data", "cache", "llm_responses.json")
    
    # 设置日志
    logger = setup_logger("benchmark", "logs/benchmark.log")
    
    # 加载数据
    print("\n1. 加载数据...")
    loader = MetaQALoader(data_dir)
    test_dataset = loader.load_dev()[:50]  # 使用 50 个问题
    print(f"   ✓ 加载了 {len(test_dataset)} 个问题")
    
    # 加载组件
    print("\n2. 加载组件...")
    graph_engine = MetaQAGraphEngine(kb_path, cache_dir)
    cache_manager = LLMCache(llm_cache_path)
    
    # 定义 Agent 配置
    agents_config = [
        {
            "name": "Naive LLM",
            "class": NaiveLLMAgent,
            "params": {
                "monitor": None,  # 会在循环中创建
                "cache_manager": cache_manager
            },
            "description": "直接使用大模型"
        },
        {
            "name": "Tiered-Forest",
            "class": TieredForestAgent,
            "params": {
                "monitor": None,
                "graph_engine": graph_engine,
                "cache_manager": cache_manager,
                "t_drop": 0.3,
                "t_pass": 0.6
            },
            "description": "三层路由（符号→语义→LLM）"
        }
    ]
    
    all_results = []
    detailed_results = []
    
    for agent_config in agents_config:
        print(f"\n{'='*80}")
        print(f"测试: {agent_config['name']}")
        print(f"描述: {agent_config['description']}")
        print(f"{'='*80}")
        
        # 创建 Agent
        monitor = CostMonitor()
        params = agent_config['params'].copy()
        params['monitor'] = monitor
        agent = agent_config['class'](**params)
        
        # 运行测试
        correct_count = 0
        start_time = time.time()
        
        for item in tqdm(test_dataset, desc=f"处理问题"):
            question = item['question']
            ground_truths = item['all_answers']
            
            try:
                # 求解
                prediction = agent.solve(question)
                is_correct = evaluate_accuracy(prediction, ground_truths)
                
                if is_correct:
                    correct_count += 1
                
                # 记录详细结果
                detailed_results.append({
                    'agent': agent_config['name'],
                    'question': question,
                    'prediction': prediction[:100],
                    'ground_truth': ground_truths[0],
                    'correct': is_correct
                })
                
            except Exception as e:
                logger.error(f"Error on question '{question}': {e}")
                detailed_results.append({
                    'agent': agent_config['name'],
                    'question': question,
                    'prediction': f"ERROR: {e}",
                    'ground_truth': ground_truths[0],
                    'correct': False
                })
        
        total_time = time.time() - start_time
        
        # 统计
        accuracy = correct_count / len(test_dataset)
        stats = monitor.get_session_stats()
        
        # 打印结果
        print(f"\n结果:")
        print(f"  准确率: {accuracy:.2%} ({correct_count}/{len(test_dataset)})")
        print(f"  总成本: ${stats['cost_usd']:.6f}")
        print(f"  平均成本/问题: ${stats['cost_usd']/len(test_dataset):.6f}")
        print(f"  总 Token: {stats['tokens_total']}")
        print(f"  平均延迟: {stats['latency_avg']:.2f}s")
        print(f"  总时间: {total_time:.1f}s")
        
        # 特殊统计
        if agent_config['name'] == "Tiered-Forest":
            tier_usage = agent.get_tier_usage()
            print(f"\n  Tier 使用分布:")
            print(f"    Tier 1 (Symbolic): {tier_usage['tier1']} ({tier_usage['tier1']/len(test_dataset)*100:.1f}%)")
            print(f"    Tier 2 (Semantic): {tier_usage['tier2']} ({tier_usage['tier2']/len(test_dataset)*100:.1f}%)")
            print(f"    Tier 3 (LLM): {tier_usage['tier3']} ({tier_usage['tier3']/len(test_dataset)*100:.1f}%)")
        
        # 保存到总结果
        result = {
            'agent': agent_config['name'],
            'description': agent_config['description'],
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_questions': len(test_dataset),
            'total_cost_usd': stats['cost_usd'],
            'avg_cost_usd': stats['cost_usd']/len(test_dataset),
            'total_tokens': stats['tokens_total'],
            'avg_tokens': stats['tokens_total']/len(test_dataset) if len(test_dataset) > 0 else 0,
            'avg_latency_s': stats['latency_avg'],
            'total_time_s': total_time
        }
        
        # 添加特殊字段
        if agent_config['name'] == "Tiered-Forest":
            result['tier1_pct'] = tier_usage['tier1']/len(test_dataset)*100
            result['tier2_pct'] = tier_usage['tier2']/len(test_dataset)*100
            result['tier3_pct'] = tier_usage['tier3']/len(test_dataset)*100
        
        all_results.append(result)
    
    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果")
    print(f"{'='*80}")
    
    # 汇总结果
    df_summary = pd.DataFrame(all_results)
    summary_path = "results/benchmark_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ 汇总结果已保存: {summary_path}")
    
    # 详细结果
    df_detailed = pd.DataFrame(detailed_results)
    detailed_path = "results/benchmark_detailed.csv"
    df_detailed.to_csv(detailed_path, index=False)
    print(f"✓ 详细结果已保存: {detailed_path}")
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")
    print(f"\n{'Agent':<20} {'准确率':<12} {'总成本':<14} {'平均成本':<14} {'总Token':<12}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['agent']:<20} "
              f"{result['accuracy']:<12.2%} "
              f"${result['total_cost_usd']:<13.6f} "
              f"${result['avg_cost_usd']:<13.6f} "
              f"{result['total_tokens']:<12.0f}")
    
    # 计算成本节省
    print(f"\n{'='*80}")
    print("成本效率分析")
    print(f"{'='*80}")
    
    naive_cost = next(r['total_cost_usd'] for r in all_results if r['agent'] == 'Naive LLM')
    
    if naive_cost > 0:
        for result in all_results:
            if result['agent'] != 'Naive LLM':
                cost_reduction = (naive_cost - result['total_cost_usd']) / naive_cost * 100
                print(f"\n{result['agent']}:")
                print(f"  成本降低: {cost_reduction:.1f}%")
                print(f"  准确率差异: {(result['accuracy'] - all_results[0]['accuracy'])*100:+.1f}%")
    else:
        print("\n注意: Naive LLM 成本为 0（可能全部命中缓存），无法计算成本降低率")
    
    # 保存缓存
    cache_manager.save()
    cache_stats = cache_manager.get_stats()
    print(f"\n缓存统计: 命中率={cache_stats['hit_rate']:.2%}, 大小={cache_stats['cache_size']}")
    
    print(f"\n{'='*80}")
    print("Benchmark 完成!")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_benchmark()
