import time
from openai import OpenAI
from typing import Optional
from ..utils.cache_manager import LLMCache
from ..utils.retry_handler import retry_on_failure
from .. import config

class LLMReasoner:
    """
    Tier 3: 大模型推理层
    使用 DeepSeek API 进行复杂推理
    成本: 高 (按 token 计费)
    """
    
    def __init__(self, cache_manager: LLMCache):
        """
        初始化 LLM 推理器
        
        Args:
            cache_manager: LLM 缓存管理器
        """
        self.cache = cache_manager
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )
        self.model_name = config.MODEL_NAME
    
    @retry_on_failure(max_retries=3, backoff_base=2.0)
    def reason(self, question: str, context: Optional[str] = None, max_tokens: int = 200) -> tuple:
        """
        使用 LLM 进行推理
        
        Args:
            question: 问题文本
            context: 可选的上下文信息
            max_tokens: 最大生成 token 数
            
        Returns:
            (answer, prompt_tokens, completion_tokens, latency)
        """
        # 构建 Prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nThink step by step and provide a concise answer:"
        else:
            prompt = f"Question: {question}\n\nThink step by step and provide a concise answer:"
        
        # 检查缓存
        cached_response = self.cache.get(prompt, self.model_name)
        if cached_response:
            # 缓存命中，返回缓存的响应（token 计数为 0）
            return cached_response, 0, 0, 0.0
        
        # 调用 API
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False
            )
            latency = time.time() - start_time
            
            # 提取响应
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            
            # 写入缓存
            self.cache.set(prompt, self.model_name, answer)
            
            return answer, prompt_tokens, completion_tokens, latency
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            # 重试机制会自动处理，如果所有重试都失败，则抛出异常
            raise
    
    def reason_simple(self, question: str) -> tuple:
        """
        简化版推理（不使用 CoT）
        
        Args:
            question: 问题文本
            
        Returns:
            (answer, prompt_tokens, completion_tokens, latency)
        """
        prompt = f"Answer this question concisely: {question}"
        
        # 检查缓存
        cached_response = self.cache.get(prompt, self.model_name)
        if cached_response:
            return cached_response, 0, 0, 0.0
        
        # 调用 API
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
                stream=False
            )
            latency = time.time() - start_time
            
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            # 写入缓存
            self.cache.set(prompt, self.model_name, answer)
            
            return answer, usage.prompt_tokens, usage.completion_tokens, latency
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise
