"""
FrugalGPT Agent - Simplified Implementation

基于论文: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"
作者: Lingjiao Chen, Matei Zaharia, James Zou (Stanford, 2023)

简化版实现:
1. LLM Cascade: 从便宜到贵的级联调用
2. Generation Scoring: 使用启发式规则评估答案质量
3. 固定阈值: 不需要训练评分模型
"""

from .base_agent import BaseAgent
from ..cost_monitor import CostMonitor
from ..utils.cache_manager import LLMCache
from .frugal_scorer import SimpleScorer
from openai import OpenAI
from .. import config
import time
from typing import List, Tuple, Optional


class LLMConfig:
    """LLM配置类"""
    def __init__(self, name: str, api_key: str, base_url: str, 
                 model_name: str, price_input: float, price_output: float, tier: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.price_input = price_input  # per 1k tokens
        self.price_output = price_output
        self.tier = tier  # "small" or "large"


class FrugalGPTAgent(BaseAgent):
    """
    FrugalGPT Agent - 简化版实现
    
    策略:
    1. 按成本从低到高的顺序调用LLM
    2. 使用评分函数评估每个答案的质量
    3. 如果答案质量超过阈值，立即返回
    4. 否则继续调用下一个更贵的LLM
    
    优点:
    - 平衡成本和准确率
    - 简单问题用便宜模型，复杂问题用贵模型
    
    缺点:
    - 可能需要多次API调用
    - 评分函数的准确性影响性能
    """
    
    def __init__(self, monitor: CostMonitor, cache_manager: LLMCache, 
                 thresholds: Optional[List[float]] = None):
        """
        初始化 FrugalGPT Agent
        
        Args:
            monitor: 成本监控器
            cache_manager: LLM 缓存管理器
            thresholds: 每个LLM的质量阈值列表 [threshold_small, threshold_large]
                       如果答案质量 >= threshold，则接受答案
                       默认: [0.7, 0.5] (小模型要求更高的置信度)
        """
        super().__init__(monitor)
        self.cache = cache_manager
        self.scorer = SimpleScorer()
        
        # 默认阈值
        if thresholds is None:
            thresholds = [0.7, 0.6, 0.5]  # [小模型阈值, Kimi阈值, 大模型阈值]
        self.thresholds = thresholds
        
        # 配置LLM级联（从便宜到贵）
        self.llm_cascade = self._setup_llm_cascade()
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "stopped_at_small": 0,
            "stopped_at_kimi": 0,
            "stopped_at_large": 0,
            "avg_llms_called": 0.0
        }
    
    def _setup_llm_cascade(self) -> List[LLMConfig]:
        """
        设置LLM级联顺序（从便宜到贵）
        
        Returns:
            LLM配置列表
        """
        cascade = []
        
        # LLM 1: 小模型 (Qwen2.5-7B) - 最便宜
        if config.SMALL_MODEL_API_KEY and config.SMALL_MODEL_API_KEY != "EMPTY":
            cascade.append(LLMConfig(
                name="Small-Model",
                api_key=config.SMALL_MODEL_API_KEY,
                base_url=config.SMALL_MODEL_BASE_URL,
                model_name=config.SMALL_MODEL_NAME,
                price_input=config.PRICE_SMALL_MODEL,
                price_output=config.PRICE_SMALL_MODEL,
                tier="small"
            ))
        
        # LLM 2: Kimi (Moonshot AI) - 中等价格
        if config.KIMI_API_KEY and config.KIMI_API_KEY != "EMPTY":
            cascade.append(LLMConfig(
                name="Kimi-Model",
                api_key=config.KIMI_API_KEY,
                base_url=config.KIMI_BASE_URL,
                model_name=config.KIMI_MODEL_NAME,
                price_input=config.PRICE_KIMI,
                price_output=config.PRICE_KIMI,
                tier="kimi"
            ))
        
        # LLM 3: 大模型 (DeepSeek-Chat) - 较贵
        cascade.append(LLMConfig(
            name="Large-Model",
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            model_name=config.MODEL_NAME,
            price_input=config.PRICE_LARGE_INPUT,
            price_output=config.PRICE_LARGE_OUTPUT,
            tier="large"
        ))
        
        return cascade
    
    def _call_llm(self, llm_config: LLMConfig, question: str) -> Tuple[str, int, int, float]:
        """
        调用单个LLM
        
        Args:
            llm_config: LLM配置
            question: 问题文本
            
        Returns:
            (answer, prompt_tokens, completion_tokens, latency)
        """
        # 构建prompt
        prompt = f"Answer this question concisely: {question}"
        
        # 检查缓存
        cache_key = f"{llm_config.model_name}:{prompt}"
        cached_response = self.cache.get(cache_key, llm_config.model_name)
        if cached_response:
            return cached_response, 0, 0, 0.0
        
        # 调用API
        client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url
        )
        
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=llm_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
                stream=False
            )
            latency = time.time() - start_time
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            
            # 写入缓存
            self.cache.set(cache_key, llm_config.model_name, answer)
            
            return answer, prompt_tokens, completion_tokens, latency
            
        except Exception as e:
            print(f"LLM API Error ({llm_config.name}): {e}")
            raise
    
    def solve(self, question: str) -> str:
        """
        使用FrugalGPT策略回答问题
        
        工作流程:
        1. 从最便宜的LLM开始
        2. 调用LLM生成答案
        3. 使用评分函数评估答案质量
        4. 如果质量 >= 阈值，返回答案
        5. 否则，调用下一个更贵的LLM
        6. 如果所有LLM都调用完，返回最后一个答案
        
        Args:
            question: 问题文本
            
        Returns:
            答案字符串
        """
        self.stats["total_queries"] += 1
        llms_called = 0
        final_answer = ""
        
        try:
            # 级联调用LLM
            for i, llm_config in enumerate(self.llm_cascade):
                llms_called += 1
                
                # 调用LLM
                answer, prompt_tokens, completion_tokens, latency = self._call_llm(
                    llm_config, question
                )
                
                # 记录成本
                self.monitor.record_usage(
                    prompt_tokens, completion_tokens, latency, llm_config.tier
                )
                
                # 评估答案质量
                score = self.scorer.score(question, answer)
                
                # 获取当前LLM的阈值
                threshold = self.thresholds[i] if i < len(self.thresholds) else 0.5
                
                # 调试信息
                print(f"  [{llm_config.name}] Score: {score:.3f}, Threshold: {threshold:.3f}")
                
                # 如果质量足够好，或者是最后一个LLM，返回答案
                if score >= threshold or i == len(self.llm_cascade) - 1:
                    final_answer = answer
                    
                    # 更新统计
                    if llm_config.tier == "small":
                        self.stats["stopped_at_small"] += 1
                    elif llm_config.tier == "kimi":
                        self.stats["stopped_at_kimi"] += 1
                    else:
                        self.stats["stopped_at_large"] += 1
                    
                    break
            
            # 更新平均调用次数
            self.stats["avg_llms_called"] = (
                (self.stats["avg_llms_called"] * (self.stats["total_queries"] - 1) + llms_called) 
                / self.stats["total_queries"]
            )
            
            return final_answer
            
        except Exception as e:
            print(f"FrugalGPT failed: {e}")
            return f"Error: {e}"
    
    def get_frugal_stats(self) -> dict:
        """
        获取FrugalGPT特定的统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_queries": self.stats["total_queries"],
            "stopped_at_small": self.stats["stopped_at_small"],
            "stopped_at_kimi": self.stats["stopped_at_kimi"],
            "stopped_at_large": self.stats["stopped_at_large"],
            "avg_llms_called": self.stats["avg_llms_called"],
            "small_model_usage_rate": (
                self.stats["stopped_at_small"] / max(1, self.stats["total_queries"])
            ),
            "kimi_model_usage_rate": (
                self.stats["stopped_at_kimi"] / max(1, self.stats["total_queries"])
            ),
            "cascade_config": {
                "num_llms": len(self.llm_cascade),
                "llm_names": [llm.name for llm in self.llm_cascade],
                "thresholds": self.thresholds
            }
        }
