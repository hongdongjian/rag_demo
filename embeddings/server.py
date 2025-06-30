import asyncio
import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from embeddings.flag_mbedder import FlagEmbedder
from embeddings.model import EmbeddingResponse, EmbeddingRequest

app = FastAPI(title="向量检索服务", description="提供向量检索API服务")

# 安全验证
security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "jina_04afb1e9a75e41d9acde9357d01f2c6fTAk8UbRt3usi9AA8y0NCLq1-Kr9l")

# 模型配置
MODEL_CONFIG = {
    "bge-code-v1": {
        "model_name": "BAAI/bge-code-v1",
        "model_path": "/Users/hongdongjian/Documents/workspace/python/rag_demo/models/models--BAAI--bge-code-v1/snapshots/bd67852057c5d7ddcc7b8234d9d6c410117ed851",
        # "model_path": "/export/model/bge-code-v1",
        "default_dimensions": 1024
    }
}

# 验证API密钥
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    return credentials.credentials


# 加载模型（单例模式）
model_instances = {}

def get_model(model_name: str):
    if model_name not in model_instances:
        if model_name not in MODEL_CONFIG:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {model_name}")

        config = MODEL_CONFIG[model_name]
        if model_name == "bge-code-v1":
            model_instances[model_name] = FlagEmbedder(model_name_or_path=config["model_path"])
    return model_instances[model_name]


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
        request: EmbeddingRequest,
        api_key: str = Depends(verify_api_key)
):
    try:
        model = get_model(request.model)
        return model.encode(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成嵌入时出错: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 启动FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run("embeddings.server:app", host="0.0.0.0", port=8000, workers=1)