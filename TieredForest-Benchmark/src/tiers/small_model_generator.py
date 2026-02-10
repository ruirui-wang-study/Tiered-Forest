import time
from openai import OpenAI
from typing import Optional
from ..utils.retry_handler import retry_on_failure
from .. import config

class SmallModelGenerator:
    """
    小模型候选生成器
    使用 Qwen2.5-7B 等小模型快速生成候选答案
    成本: 低（~$0.0002 per question）
    """
    
    def __init__(self):
        """初始化小模型客户端"""
        try:
            self.client = OpenAI(
                api_key=config.SMALL_MODEL_API_KEY,
                base_url=config.SMALL_MODEL_BASE_URL
            )
            self.model_name = config.SMALL_MODEL_NAME
            print(f"Initialized SmallModelGenerator: {self.model_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize small model: {e}")
            self.client = None
    
    @retry_on_failure(max_retries=2, backoff_base=1.5)
    def generate_candidate(self, question: str, max_tokens: int = 50) -> tuple:
        """
        生成候选答案
        
        Args:
            question: 问题文本
            max_tokens: 最大生成 token 数
            
        Returns:
            (candidate_answer, prompt_tokens, completion_tokens, latency)
        """
        if self.client is None:
            return None, 0, 0, 0.0
        
        # 构建简洁的 prompt
        prompt = f"Answer this question briefly in one sentence: {question}"
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,  # 低温度，更确定性
                stream=False
            )
            latency = time.time() - start_time
            
            # 提取响应
            answer = response.choices[0].message.content.strip()
            usage = response.usage
            
            return answer, usage.prompt_tokens, usage.completion_tokens, latency
            
        except Exception as e:
            print(f"Small model generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """检查小模型是否可用"""
        return self.client is not None
