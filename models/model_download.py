from sentence_transformers import SentenceTransformer

# 通过官方库自动下载并保存
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("./all-MiniLM-L6-v2")  # 指定本地存储路径

model = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2")
model.save("./ms-marco-MiniLM-L-6-v2")  # 指定本地存储路径