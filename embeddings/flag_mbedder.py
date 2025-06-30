import base64

import numpy as np
from FlagEmbedding import FlagLLMModel
from transformers import AutoTokenizer

from embeddings.model import EmbeddingResponse, EmbeddingRequest, EmbeddingUsage, EmbeddingData

MODEL_NAME = "bge-code-v1"
MODEL_FULL_NAME = "BAAI/bge-code-v1"


class FlagEmbedder:
    """FlagEmbedding 封装类，用于生成文本嵌入向量"""

    def __init__(
            self,
            model_name_or_path: str,
            query_instruction_format: str = "<instruct>{}\n<query>{}",
            query_instruction_for_retrieval: str = "Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
            trust_remote_code: bool = True,
            use_fp16: bool = False,
    ):
        self.model = FlagLLMModel(
            model_name_or_path,
            query_instruction_format=query_instruction_format,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            trust_remote_code=trust_remote_code,
            use_fp16=use_fp16,
        )
        self.model.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, request: EmbeddingRequest):
        if request.dimensions and request.dimensions > 1536:
            raise ValueError(f"请求的维度({request.dimensions})超过了模型支持的最大维度({1536})")
        if request.encoding_format not in ["float", "base64"]:
            raise ValueError(f"不支持的编码格式: {request.encoding_format}. 仅支持 'float' 或 'base64'")
        # 计算输入的token数量
        inputs = self.tokenizer(request.input, return_tensors="np")
        attention_mask = inputs["attention_mask"]
        prompt_tokens = 0
        for mask in attention_mask:
            prompt_tokens += len(mask)
        # 生成嵌入向量
        embedding_data = self.model.encode_queries(request.input)
        # 根据request.dimensions调整嵌入向量的维度
        if request.dimensions is None:
            request.dimensions = 1536
        processed_embeddings = []
        for emb in embedding_data:
            truncated = emb[:request.dimensions]
            if request.encoding_format == "base64":
                # 将向量转换为Base64字符串
                truncated = base64.b64encode(truncated.tobytes()).decode()
            processed_embeddings.append(truncated)
        embedding_data = processed_embeddings
        return EmbeddingResponse(
            model=MODEL_NAME,
            usage=EmbeddingUsage(
                total_tokens=prompt_tokens,
                prompt_tokens=prompt_tokens
            ),
            data=[
                EmbeddingData(
                    object="embedding",
                    index=i,
                    embedding=embedding
                ) for i, embedding in enumerate(embedding_data)
            ]
        )
