import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, backoff_base: float = 2.0):
    """
    装饰器：API 调用失败时自动重试
    
    Args:
        max_retries: 最大重试次数
        backoff_base: 指数退避基数 (秒)
    
    Example:
        @retry_on_failure(max_retries=3, backoff_base=2.0)
        def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_base ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed.")
            
            # 所有重试都失败，抛出最后一个异常
            raise last_exception
        
        return wrapper
    return decorator
