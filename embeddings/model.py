from pydantic import BaseModel
from typing import List, Optional, Union

# 请求模型
class EmbeddingRequest(BaseModel):
    model: str
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    input: List[str]


# 响应模型
class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]


class EmbeddingUsage(BaseModel):
    total_tokens: int
    prompt_tokens: int


class EmbeddingResponse(BaseModel):
    model: str
    object: str = "list"
    usage: EmbeddingUsage
    data: List[EmbeddingData]