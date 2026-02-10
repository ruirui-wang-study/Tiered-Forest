"""
测试 FrugalGPT 实现

验证:
1. FrugalGPT 级联策略是否正常工作
2. 评分函数是否合理
3. 成本监控是否正确
4. 与 Naive LLM 的对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.frugal_agent import FrugalGPTAgent
from src.agents.naive_agent import NaiveLLMAgent
from src.cost_monitor import CostMonitor
from src.utils.cache_manager import LLMCache
from src.agents.frugal_scorer import SimpleScorer


def test_scorer():
    """测试评分函数"""
    print("=" * 60)
    print("测试 1: 评分函数")
    print("=" * 60)
    
    scorer = SimpleScorer()
    
    test_cases = [
        ("Who directed Inception?", "Christopher Nolan directed Inception.", "高质量答案"),
        ("Who directed Inception?", "I'm not sure, but it might be Christopher Nolan.", "不确定答案"),
        ("Who directed Inception?", "Error: Cannot find information.", "错误答案"),
        ("How many Oscars did Titanic win?", "Titanic won 11 Oscars.", "包含数字"),
        ("How many Oscars did Titanic win?", "It won many awards.", "模糊答案"),
        ("What is the capital of France?", "Paris", "简短答案"),
    ]
    
    for question, answer, description in test_cases:
        score, explanation = scorer.explain_score(question, answer)
        print(f"\n{description}:")
        print(f"  Q: {question}")
        print(f"  A: {answer}")
        print(f"  Score: {score:.3f}")
        print(f"  Details: {explanation}")
    
    print("\n✓ 评分函数测试完成\n")


def test_frugal_agent():
    """测试 FrugalGPT Agent"""
    print("=" * 60)
    print("测试 2: FrugalGPT Agent")
    print("=" * 60)
    
    # 初始化
    monitor = CostMonitor()
    cache = LLMCache(cache_file="data/cache/frugal_test_cache.json")
    
    # 创建 FrugalGPT Agent（使用较高的阈值来测试级联）
    frugal_agent = FrugalGPTAgent(
        monitor=monitor,
        cache_manager=cache,
        thresholds=[0.8, 0.5]  # 小模型需要0.8，大模型需要0.5
    )
    
    # 测试问题
    test_questions = [
        "Who directed the movie Inception?",
        "What is 2+2?",
        "Who is the president of the United States in 2024?",
    ]
    
    print(f"\nFrugalGPT 配置:")
    print(f"  LLM 数量: {len(frugal_agent.llm_cascade)}")
    print(f"  LLM 列表: {[llm.name for llm in frugal_agent.llm_cascade]}")
    print(f"  阈值: {frugal_agent.thresholds}")
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 60)
        
        try:
            answer = frugal_agent.solve(question)
            print(f"答案: {answer}")
            
            # 显示成本统计
            stats = monitor.get_session_stats()
            print(f"\n成本统计:")
            print(f"  总 Token: {stats['tokens_total']}")
            print(f"  总成本: ${stats['cost_usd']:.6f}")
            print(f"  延迟: {stats['latency_avg']:.3f}s")
            
            # 重置监控器以便下一个问题
            monitor.reset()
            
        except Exception as e:
            print(f"错误: {e}")
    
    # 显示 FrugalGPT 统计
    frugal_stats = frugal_agent.get_frugal_stats()
    print("\n" + "=" * 60)
    print("FrugalGPT 统计:")
    print("=" * 60)
    print(f"  总查询数: {frugal_stats['total_queries']}")
    print(f"  在小模型停止: {frugal_stats['stopped_at_small']}")
    print(f"  在大模型停止: {frugal_stats['stopped_at_large']}")
    print(f"  平均调用 LLM 数: {frugal_stats['avg_llms_called']:.2f}")
    print(f"  小模型使用率: {frugal_stats['small_model_usage_rate']:.1%}")
    
    print("\n✓ FrugalGPT Agent 测试完成\n")


def compare_agents():
    """对比 FrugalGPT 和 Naive LLM"""
    print("=" * 60)
    print("测试 3: FrugalGPT vs Naive LLM")
    print("=" * 60)
    
    # 测试问题
    test_questions = [
        "Who directed Inception?",
        "What is the capital of France?",
    ]
    
    # 初始化两个 Agent
    monitor_frugal = CostMonitor()
    monitor_naive = CostMonitor()
    cache = LLMCache(cache_file="data/cache/frugal_compare_cache.json")
    
    frugal_agent = FrugalGPTAgent(monitor_frugal, cache, thresholds=[0.7, 0.5])
    naive_agent = NaiveLLMAgent(monitor_naive, cache)
    
    print("\n对比结果:")
    print("-" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        
        # FrugalGPT
        monitor_frugal.reset()
        try:
            answer_frugal = frugal_agent.solve(question)
            stats_frugal = monitor_frugal.get_session_stats()
            print(f"  [FrugalGPT] 答案: {answer_frugal[:50]}...")
            print(f"  [FrugalGPT] 成本: ${stats_frugal['cost_usd']:.6f}")
        except Exception as e:
            print(f"  [FrugalGPT] 错误: {e}")
        
        # Naive LLM
        monitor_naive.reset()
        try:
            answer_naive = naive_agent.solve(question)
            stats_naive = monitor_naive.get_session_stats()
            print(f"  [Naive LLM] 答案: {answer_naive[:50]}...")
            print(f"  [Naive LLM] 成本: ${stats_naive['cost_usd']:.6f}")
        except Exception as e:
            print(f"  [Naive LLM] 错误: {e}")
    
    print("\n✓ 对比测试完成\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("FrugalGPT 简化版实现 - 测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 测试1: 评分函数
        test_scorer()
        
        # 测试2: FrugalGPT Agent
        test_frugal_agent()
        
        # 测试3: 对比测试
        # compare_agents()  # 需要API密钥才能运行
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
