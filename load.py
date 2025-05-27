# load.py
import time
from tqdm import tqdm
from typing import List, Optional

# 导入配置和工具
from config import (
    DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_DIMENSION, MILVUS_COLLECTION
)
from utils import (
    setup_warnings, create_embeddings_model,
    create_milvus_client, handle_exceptions
)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pymilvus import MilvusClient

# 设置警告
setup_warnings()


@handle_exceptions
def load_documents() -> List[Document]:
    """加载目录中的文本文档"""
    print("正在加载文档...")
    loader = DirectoryLoader(
        path=DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    return loader.load()


@handle_exceptions
def split_text(documents: List[Document]) -> List[Document]:
    """将文档分割成更小的块"""
    print("正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(documents)


@handle_exceptions
def generate_embeddings(texts: List[Document]) -> List[List[float]]:
    """为文档生成嵌入向量"""
    print("正在生成嵌入向量...")
    start_time = time.time()

    # 使用共享的嵌入模型创建函数
    embeddings_model = create_embeddings_model()

    # 批量处理优化
    batch_size = 16  # 更小的批量可以减少内存消耗
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="批量向量化进度"):
        batch = texts[i:i + batch_size]
        batch_contents = [doc.page_content for doc in batch]
        batch_embeddings = embeddings_model.embed_documents(batch_contents)
        embeddings.extend(batch_embeddings)

    elapsed_time = time.time() - start_time
    print(f"向量化完成，耗时: {elapsed_time:.2f}秒，平均每个文档 {elapsed_time / len(texts):.4f} 秒")
    return embeddings


@handle_exceptions
def store_to_milvus_lite(texts: List[Document], embeddings: List[List[float]]) -> Optional[MilvusClient]:
    """将文档及其嵌入向量存储到Milvus Lite"""
    print("正在存储到Milvus Lite...")

    # 连接Milvus Lite
    client = create_milvus_client()

    # 如果集合已存在，先删除
    if client.has_collection(MILVUS_COLLECTION):
        client.drop_collection(MILVUS_COLLECTION)
        print(f"已删除现有集合: {MILVUS_COLLECTION}")

    # 创建新集合
    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        dimension=EMBEDDING_DIMENSION
    )
    print(f"已创建集合: {MILVUS_COLLECTION}")

    # 准备数据并批量插入
    total = len(texts)
    batch_size = 100  # 更大的批量可以加快插入速度

    for i in tqdm(range(0, total, batch_size), desc="数据存储进度"):
        end_idx = min(i + batch_size, total)
        batch_texts = texts[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]

        batch_entities = [
            {
                "id": i + idx,
                "content": doc.page_content,
                "vector": emb
            }
            for idx, (doc, emb) in enumerate(zip(batch_texts, batch_embeddings))
        ]

        client.insert(
            collection_name=MILVUS_COLLECTION,
            data=batch_entities
        )

    print(f"成功插入所有数据到Milvus集合")
    return client


def main():
    """主程序流程"""
    try:
        # 步骤1: 加载文档
        docs = load_documents()
        if not docs:
            print("未找到文档，程序终止")
            return
        print(f"已加载 {len(docs)} 个文档")

        # 步骤2: 分割文本
        splits = split_text(docs)
        if not splits:
            print("文档分割失败，程序终止")
            return
        print(f"已分割为 {len(splits)} 个文本块")

        # 步骤3: 生成向量
        embeddings = generate_embeddings(splits)
        if not embeddings:
            print("向量生成失败，程序终止")
            return

        if len(embeddings) != len(splits):
            print(f"警告: 嵌入向量数量 ({len(embeddings)}) 与文本块数量 ({len(splits)}) 不匹配")
            return

        # 步骤4: 存入Milvus
        client = store_to_milvus_lite(splits, embeddings)
        if client:
            print(f"成功将 {len(splits)} 个向量存储到Milvus集合中")
            print("\n知识库已准备就绪，可以运行 test.py 开始对话！")

    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()