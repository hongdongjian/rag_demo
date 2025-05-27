# config.py
import os

# 模型配置
EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_PATH = "./models/ms-marco-MiniLM-L-6-v2"
# LLM_MODEL = "deepseek-r1:1.5b"
# LLM_URL = "http://localhost:11434/v1"
LLM_MODEL = "doubao-1.5-thinking-pro-m-250415"
LLM_URL = "https://ark.cn-beijing.volces.com/api/v3"
LLM_API_KEY = os.environ.get("LLM_API_KEY")  # 从环境变量获取API密钥
LLM_TEMPERATURE = 0.3

# Milvus配置
MILVUS_DB_PATH = "milvus/milvus_demo.db"
MILVUS_COLLECTION = "langchain_docs"
EMBEDDING_DIMENSION = 384

# 文档处理配置
DATA_PATH = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 性能配置
BATCH_SIZE = 64
NUM_THREADS = 8

# 环境设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# mcp
MCP_SERVER_SCRIPT = "./mcp_server.py"  # MCP服务器脚本路径
MCP_SERVER_URL = "http://127.0.0.1:8000/sse"