
import os
import configparser

# Load Config
# Set Hub Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
config.read(config_path)

# API Constants
DEEPSEEK_API_KEY = config.get('api', 'deepseek_key', fallback=os.environ.get("DEEPSEEK_API_KEY"))
DEEPSEEK_BASE_URL = config.get('api', 'deepseek_url', fallback="https://api.deepseek.com")
MODEL_NAME = "deepseek-chat"

# Small Model Configuration (e.g. SiliconFlow, Groq, or Local)
# Defaults to SiliconFlow (Qwen2.5-7B) for high-performance open/cheap inference
SMALL_MODEL_API_KEY = config.get('api', 'small_model_key', fallback=os.environ.get("SILICONFLOW_API_KEY", "EMPTY"))
SMALL_MODEL_BASE_URL = config.get('api', 'small_model_url', fallback="https://api.siliconflow.cn/v1")
SMALL_MODEL_NAME = config.get('api', 'small_model_name', fallback="Qwen/Qwen2.5-7B-Instruct")

# Pricing (Per 1k Tokens)
PRICE_ZERO = 0.00
PRICE_SMALL_MODEL = 0.0002
PRICE_LARGE_INPUT = 0.002
PRICE_LARGE_OUTPUT = 0.008

# Experiment Settings
TIMEOUT = 30
MAX_RETRIES = 3
