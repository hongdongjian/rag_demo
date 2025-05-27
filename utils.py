# utils.py
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from config import (
    EMBEDDING_MODEL_PATH, MILVUS_DB_PATH, BATCH_SIZE,
    NUM_THREADS, MILVUS_COLLECTION
)

def setup_warnings():
    """配置并忽略常见警告"""
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=Warning, module="langchain")
    warnings.filterwarnings("ignore", message="Some weights of BertForSequenceClassification were not initialized")

def create_embeddings_model():
    """创建并返回嵌入模型实例"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": BATCH_SIZE,
            "num_threads": NUM_THREADS,
        }
    )

def create_milvus_client():
    """创建并返回Milvus客户端实例"""
    return MilvusClient(MILVUS_DB_PATH)

def handle_exceptions(func):
    """装饰器：处理并记录函数执行期间的异常"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"执行 {func.__name__} 时出错: {e}")
            if func.__name__ == "store_to_milvus_lite":
                raise
            return None
    return wrapper