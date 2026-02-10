import time
from dataclasses import dataclass, field
from typing import Dict, List
from . import config

@dataclass
class CostStats:
    """统计单个 Tier 的成本数据"""
    tokens_input: int = 0
    tokens_output: int = 0
    latency: float = 0.0
    cost_usd: float = 0.0
    calls: int = 0

class CostMonitor:
    """
    核心成本追踪器
    追踪每个 Tier 的 Token 消耗、延迟和成本
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置所有统计数据"""
        self.stats = {
            "symbolic": CostStats(),  # Tier 1: 规则层
            "small": CostStats(),     # Tier 2: 小模型/CrossEncoder
            "large": CostStats(),     # Tier 3: 大模型
            "total": CostStats()
        }
        
    def record_usage(self, prompt_tokens, completion_tokens, latency, tier="large"):
        """
        记录单次 API 调用的使用情况
        
        Args:
            prompt_tokens: 输入 token 数
            completion_tokens: 输出 token 数
            latency: 延迟时间 (秒)
            tier: 层级 ("symbolic", "small", "large")
        """
        if tier not in self.stats:
            tier = "large"
            
        # 计算成本
        cost = 0.0
        if tier == "symbolic":
            cost = 0.0  # Tier 1 零成本
        elif tier == "small":
            cost = (prompt_tokens + completion_tokens) / 1000 * config.PRICE_SMALL_MODEL
        elif tier == "kimi":
            cost = (prompt_tokens + completion_tokens) / 1000 * config.PRICE_KIMI
        elif tier == "large":
            cost = (prompt_tokens / 1000 * config.PRICE_LARGE_INPUT) + \
                   (completion_tokens / 1000 * config.PRICE_LARGE_OUTPUT)
        
        # 更新 Tier 统计
        self.stats[tier].tokens_input += prompt_tokens
        self.stats[tier].tokens_output += completion_tokens
        self.stats[tier].latency += latency
        self.stats[tier].cost_usd += cost
        self.stats[tier].calls += 1
        
        # 更新总计
        self.stats["total"].tokens_input += prompt_tokens
        self.stats["total"].tokens_output += completion_tokens
        self.stats["total"].latency += latency
        self.stats["total"].cost_usd += cost
        self.stats["total"].calls += 1

    def get_session_stats(self):
        """
        获取当前会话的统计信息
        
        Returns:
            dict: 包含总计和分层统计的字典
        """
        return {
            "tokens_total": self.stats["total"].tokens_input + self.stats["total"].tokens_output,
            "cost_usd": self.stats["total"].cost_usd,
            "latency_avg": self.stats["total"].latency / max(1, self.stats["total"].calls),
            "calls_total": self.stats["total"].calls,
            "breakdown": {
                tier: {
                    "tokens_input": stats.tokens_input,
                    "tokens_output": stats.tokens_output,
                    "tokens_total": stats.tokens_input + stats.tokens_output,
                    "cost_usd": stats.cost_usd,
                    "latency": stats.latency,
                    "calls": stats.calls
                }
                for tier, stats in self.stats.items()
            }
        }
    
    def export_to_dict(self):
        """导出为字典格式，用于保存到 JSON"""
        return self.get_session_stats()