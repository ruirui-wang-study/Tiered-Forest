#!/usr/bin/env python3
"""
快速测试ToG Agent
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cost_monitor import CostMonitor
from src.graph_engine import MetaQAGraphEngine
from src.utils.cache_manager import LLMCache
from src.agents.tog_agent import ToGAgent

def test_tog():
    """测试ToG Agent"""
    print("="*80)
    print("ToG Agent 快速测试")
    print("="*80)
    
    # 初始化
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, "data", "MetaQA", "kb.txt")
    cache_dir = os.path.join(base_dir, "data", "processed")
    llm_cache_path = os.path.join(base_dir, "data", "cache", "tog_test.json")
    
    print("\n加载组件...")
    graph_engine = MetaQAGraphEngine(kb_path, cache_dir)
    cache_manager = LLMCache(llm_cache_path)
    monitor = CostMonitor()
    
    print("\n创建ToG Agent...")
    agent = ToGAgent(
        monitor=monitor,
        graph_engine=graph_engine,
        cache_manager=cache_manager,
        depth=2,
        width=3,
        temperature=0.0
    )
    
    # 测试问题
    test_questions = [
        "what movies did [Temuera Morrison] act in",
        "who directed [Once Were Warriors]",
        "what movies did [Patsha Bay] act in",
    ]
    
    print(f"\n开始测试 {len(test_questions)} 个问题...")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 80)
        
        try:
            # 重置监控器
            monitor.reset()
            
            # 求解
            answer = agent.solve(question)
            
            # 获取统计
            stats = monitor.get_session_stats()
            
            print(f"\n答案: {answer}")
            print(f"成本: ${stats['cost_usd']:.6f}")
            print(f"Token: {stats['tokens_total']}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 总体统计
    print("\n" + "="*80)
    print("ToG 统计")
    print("="*80)
    
    tog_stats = agent.get_stats()
    print(f"\n总查询数: {tog_stats['total_queries']}")
    print(f"平均搜索深度: {tog_stats['avg_depth']:.2f}")
    print(f"LLM调用次数: {tog_stats['llm_calls']}")
    print(f"平均LLM调用/问题: {tog_stats['llm_calls']/tog_stats['total_queries']:.2f}")
    
    print("\n" + "="*80)
    print("✓ 测试完成!")
    print("="*80)
    
    # 保存缓存
    cache_manager.save()

if __name__ == "__main__":
    test_tog()
