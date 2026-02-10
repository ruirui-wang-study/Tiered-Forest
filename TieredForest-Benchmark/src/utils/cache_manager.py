import json
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, Any

class LLMCache:
    """
    LLM 响应缓存管理器
    使用 MD5 哈希作为 key，避免重复 API 调用
    """
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.hits = 0
        self.misses = 0
        
    def load_cache(self) -> Dict[str, Any]:
        """从磁盘加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return {}
        return {}
    
    def get_cache_key(self, prompt: str, model_name: str) -> str:
        """
        生成缓存 key
        
        Args:
            prompt: 输入提示词
            model_name: 模型名称
            
        Returns:
            MD5 哈希字符串
        """
        content = f"{model_name}::{prompt}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, model_name: str) -> Optional[str]:
        """
        查询缓存
        
        Args:
            prompt: 输入提示词
            model_name: 模型名称
            
        Returns:
            缓存的响应，如果不存在则返回 None
        """
        key = self.get_cache_key(prompt, model_name)
        if key in self.cache:
            self.hits += 1
            return self.cache[key].get('response')
        self.misses += 1
        return None
    
    def set(self, prompt: str, model_name: str, response: str):
        """
        写入缓存
        
        Args:
            prompt: 输入提示词
            model_name: 模型名称
            response: LLM 响应
        """
        key = self.get_cache_key(prompt, model_name)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'model': model_name
        }
    
    def save(self):
        """持久化缓存到磁盘"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
