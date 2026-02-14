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

# KG Backend Configuration
KG_BACKEND = config.get('kg', 'backend', fallback=os.environ.get("KG_BACKEND", "metaqa"))

# Comma-separated URLs, e.g. http://127.0.0.1:23546,http://127.0.0.1:23547
_wikidata_server_urls_raw = config.get(
    'kg',
    'wikidata_server_urls',
    fallback=os.environ.get("WIKIDATA_SERVER_URLS", "")
)
WIKIDATA_SERVER_URLS = [
    url.strip() for url in _wikidata_server_urls_raw.split(",") if url.strip()
]

WIKIDATA_SERVER_URLS_FILE = config.get(
    'kg',
    'wikidata_server_urls_file',
    fallback=os.environ.get(
        "WIKIDATA_SERVER_URLS_FILE",
        os.path.join(ROOT_DIR, "ToG", "ToG", "server_urls.txt"),
    ),
)

FREEBASE_SPARQL_ENDPOINT = config.get(
    'kg',
    'freebase_sparql_endpoint',
    fallback=os.environ.get("FREEBASE_SPARQL_ENDPOINT", "http://localhost:8890/sparql"),
)
