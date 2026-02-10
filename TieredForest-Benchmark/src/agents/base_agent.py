from abc import ABC, abstractmethod
from typing import Any, Dict
from ..cost_monitor import CostMonitor

class BaseAgent(ABC):
    """
    Agent 抽象基类
    所有 Agent 必须实现 solve 方法
    """
    
    def __init__(self, monitor: CostMonitor):
        """
        初始化 Agent
        
        Args:
            monitor: 成本监控器
        """
        self.monitor = monitor
    
    @abstractmethod
    def solve(self, question: str) -> str:
        """
        回答问题
        
        Args:
            question: 问题文本
            
        Returns:
            答案字符串
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前会话的统计信息"""
        return self.monitor.get_session_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.monitor.reset()
