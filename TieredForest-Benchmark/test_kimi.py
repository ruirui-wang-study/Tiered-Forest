"""
测试 Kimi API 配置

验证 Kimi API 是否正确配置并可用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from openai import OpenAI
import time


def test_kimi_api():
    """测试 Kimi API 连接"""
    print("=" * 60)
    print("测试 Kimi API 配置")
    print("=" * 60)
    
    # 检查配置
    print(f"\nKimi API Key: {config.KIMI_API_KEY[:20]}..." if config.KIMI_API_KEY != "EMPTY" else "未配置")
    print(f"Kimi Base URL: {config.KIMI_BASE_URL}")
    print(f"Kimi Model: {config.KIMI_MODEL_NAME}")
    print(f"Kimi Price: ${config.PRICE_KIMI}/1K tokens")
    
    if config.KIMI_API_KEY == "EMPTY":
        print("\n❌ Kimi API Key 未配置")
        return False
    
    # 测试 API 调用
    print("\n测试 API 调用...")
    try:
        client = OpenAI(
            api_key=config.KIMI_API_KEY,
            base_url=config.KIMI_BASE_URL
        )
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=config.KIMI_MODEL_NAME,
            messages=[
                {"role": "user", "content": "你好，请简单介绍一下你自己"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        latency = time.time() - start_time
        
        answer = response.choices[0].message.content
        usage = response.usage
        
        print(f"\n✅ API 调用成功！")
        print(f"\n回答: {answer}")
        print(f"\nToken 使用:")
        print(f"  - 输入: {usage.prompt_tokens}")
        print(f"  - 输出: {usage.completion_tokens}")
        print(f"  - 总计: {usage.total_tokens}")
        print(f"\n延迟: {latency:.2f}s")
        
        # 计算成本
        cost = usage.total_tokens / 1000 * config.PRICE_KIMI
        print(f"成本: ${cost:.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frugal_with_kimi():
    """测试 FrugalGPT 与 Kimi 的集成"""
    print("\n" + "=" * 60)
    print("测试 FrugalGPT 与 Kimi 集成")
    print("=" * 60)
    
    from src.agents.frugal_agent import FrugalGPTAgent
    from src.cost_monitor import CostMonitor
    from src.utils.cache_manager import LLMCache
    
    # 初始化
    monitor = CostMonitor()
    cache = LLMCache(cache_file="data/cache/kimi_test_cache.json")
    
    # 创建 FrugalGPT Agent
    agent = FrugalGPTAgent(
        monitor=monitor,
        cache_manager=cache,
        thresholds=[0.8, 0.7, 0.5]  # [小模型, Kimi, 大模型]
    )
    
    print(f"\nFrugalGPT 配置:")
    print(f"  LLM 数量: {len(agent.llm_cascade)}")
    print(f"  LLM 列表: {[llm.name for llm in agent.llm_cascade]}")
    print(f"  阈值: {agent.thresholds}")
    
    # 测试问题
    test_questions = [
        "What is the capital of France?",
        "Who directed the movie Inception?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 60)
        
        try:
            answer = agent.solve(question)
            print(f"答案: {answer}")
            
            # 显示成本统计
            stats = monitor.get_session_stats()
            print(f"\n成本统计:")
            print(f"  总 Token: {stats['tokens_total']}")
            print(f"  总成本: ${stats['cost_usd']:.6f}")
            print(f"  延迟: {stats['latency_avg']:.3f}s")
            
            # 重置监控器
            monitor.reset()
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 显示 FrugalGPT 统计
    frugal_stats = agent.get_frugal_stats()
    print("\n" + "=" * 60)
    print("FrugalGPT 统计:")
    print("=" * 60)
    print(f"  总查询数: {frugal_stats['total_queries']}")
    print(f"  在小模型停止: {frugal_stats['stopped_at_small']}")
    print(f"  在 Kimi 停止: {frugal_stats['stopped_at_kimi']}")
    print(f"  在大模型停止: {frugal_stats['stopped_at_large']}")
    print(f"  平均调用 LLM 数: {frugal_stats['avg_llms_called']:.2f}")
    print(f"  小模型使用率: {frugal_stats['small_model_usage_rate']:.1%}")
    print(f"  Kimi 使用率: {frugal_stats['kimi_model_usage_rate']:.1%}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Kimi API 配置测试")
    print("=" * 60 + "\n")
    
    try:
        # 测试1: Kimi API
        if test_kimi_api():
            # 测试2: FrugalGPT 集成
            test_frugal_with_kimi()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
