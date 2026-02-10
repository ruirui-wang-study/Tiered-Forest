import os
import configparser

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # TieredForest-Benchmark/
ROOT_DIR = os.path.dirname(BASE_DIR) # Tiered-Forest/

# Load Config
config = configparser.ConfigParser()
config_path = os.path.join(ROOT_DIR, 'config.ini')
config.read(config_path)

# API Constants
DEEPSEEK_API_KEY = config.get('api', 'deepseek_key', fallback=os.environ.get("DEEPSEEK_API_KEY"))
DEEPSEEK_BASE_URL = config.get('api', 'deepseek_url', fallback="https://api.deepseek.com")
MODEL_NAME = "deepseek-chat"

# Small Model Configuration
SMALL_MODEL_API_KEY = config.get('api', 'small_model_key', fallback=os.environ.get("SILICONFLOW_API_KEY", "EMPTY"))
SMALL_MODEL_BASE_URL = config.get('api', 'small_model_url', fallback="https://api.siliconflow.cn/v1")
SMALL_MODEL_NAME = config.get('api', 'small_model_name', fallback="Qwen/Qwen2.5-7B-Instruct")

# Kimi API Configuration (Moonshot AI)
KIMI_API_KEY = config.get('api', 'kimi_key', fallback=os.environ.get("KIMI_API_KEY", "EMPTY"))
KIMI_BASE_URL = config.get('api', 'kimi_url', fallback="https://api.moonshot.cn/v1")
KIMI_MODEL_NAME = config.get('api', 'kimi_model', fallback="moonshot-v1-8k")

# Pricing (Per 1k Tokens) - Approx
PRICE_ZERO = 0.00
PRICE_SMALL_MODEL = 0.0002 # ~$0.20 / 1M tokens
PRICE_KIMI = 0.0012        # Kimi: ~$1.20 / 1M tokens (8k context)
PRICE_LARGE_INPUT = 0.002  # DeepSeek V3 pricing
PRICE_LARGE_OUTPUT = 0.008

# Experiment Settings
TIMEOUT = 30
MAX_RETRIES = 3
