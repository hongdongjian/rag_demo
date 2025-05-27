import numpy as np
from typing import List

# 导入配置和工具
from config import (
    CROSS_ENCODER_MODEL_PATH, MILVUS_COLLECTION,
    LLM_MODEL, LLM_TEMPERATURE
)
from utils import (
    setup_warnings, create_embeddings_model,
    create_milvus_client, handle_exceptions
)

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 设置警告
setup_warnings()


class HybridRetriever(BaseRetriever):
    """混合检索器，结合向量检索和交叉编码器重排序"""

    def __init__(self):
        """初始化混合检索器"""
        super().__init__()
        # 延迟初始化，避免重复创建
        self._embeddings = None
        self._milvus_client = None
        self._cross_encoder = None

    @property
    def embeddings(self):
        """懒加载嵌入模型"""
        if self._embeddings is None:
            self._embeddings = create_embeddings_model()
        return self._embeddings

    @property
    def milvus_client(self):
        """懒加载Milvus客户端"""
        if self._milvus_client is None:
            self._milvus_client = create_milvus_client()
        return self._milvus_client

    @property
    def cross_encoder(self):
        """懒加载交叉编码器"""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_PATH)
        return self._cross_encoder

    @handle_exceptions
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """根据查询获取相关文档"""
        # 向量检索
        query_emb = self.embeddings.embed_query(query)
        vector_results = self.milvus_client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_emb],
            limit=2,
            output_fields=["content"]
        )

        if not vector_results or not vector_results[0]:
            return []

        # 结果格式化
        docs = [
            Document(page_content=hit.get("entity", {}).get("content", ""))
            for hit in vector_results[0]
        ]

        # 元数据过滤
        filtered_docs = [doc for doc in docs if doc.metadata.get("valid", True)]

        if not filtered_docs:
            return []

        # 交叉编码器重排序
        return self._rerank_results(filtered_docs, query)[:1]

    def _rerank_results(self, docs: List[Document], query: str) -> List[Document]:
        """使用交叉编码器重排序文档"""
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        return [docs[i] for i in sorted_indices]