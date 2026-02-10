from .base_agent import BaseAgent
from ..cost_monitor import CostMonitor
from ..tiers.tier3_reasoner import LLMReasoner
from ..utils.cache_manager import LLMCache

class NaiveLLMAgent(BaseAgent):
    """
    Naive LLM Agent - Baseline 1
    
    策略: 直接使用大模型回答所有问题
    优点: 准确率高
    缺点: 成本高，所有问题都调用昂贵的 LLM
    """
    
    def __init__(self, monitor: CostMonitor, cache_manager: LLMCache):
        """
        初始化 Naive LLM Agent
        
        Args:
            monitor: 成本监控器
            cache_manager: LLM 缓存管理器
        """
        super().__init__(monitor)
        self.llm = LLMReasoner(cache_manager)
    
    def solve(self, question: str) -> str:
        """
        使用大模型直接回答问题
        
        Args:
            question: 问题文本
            
        Returns:
            答案字符串
        """
        try:
            # 直接调用大模型
            answer, prompt_tokens, completion_tokens, latency = self.llm.reason(question)
            
            # 记录成本
            self.monitor.record_usage(prompt_tokens, completion_tokens, latency, "large")
            
            return answer
        except Exception as e:
            print(f"Naive LLM failed: {e}")
            return f"Error: {e}"
