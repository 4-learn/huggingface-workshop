"""
Workshop 解答：任務 3 - 使用 Sentence Transformers 計算相似度

執行方式：
python 03_embeddings.py

預期輸出：
=== 語義相似度測試 ===

查詢: 沒有戴安全帽

相似法規:
  [0.89] 職業安全衛生設施規則 第 281 條：雇主應使勞工確實使用安全帽
  [0.76] 營造安全衛生設施標準 第 11-1 條：進入營造工地應配戴安全帽
  [0.45] 職業安全衛生設施規則 第 277 條：有車輛出入場所應穿著反光衣
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# 載入模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 法規資料庫
regulations = [
    "職業安全衛生設施規則 第 281 條：雇主應使勞工確實使用安全帽",
    "營造安全衛生設施標準 第 11-1 條：進入營造工地應配戴安全帽",
    "職業安全衛生設施規則 第 277 條：有車輛出入場所應穿著反光衣",
    "職業安全衛生設施規則 第 225 條：高度二公尺以上作業應使用安全帶",
    "職業安全衛生設施規則 第 33 條：緊急出口應保持暢通不得堆積物品",
]

# 計算法規的 embeddings
regulation_embeddings = model.encode(regulations)

# 查詢
query = "沒有戴安全帽"
query_embedding = model.encode(query)

# 計算相似度
similarities = np.dot(regulation_embeddings, query_embedding)

# 排序
sorted_indices = np.argsort(similarities)[::-1]

print("=== 語義相似度測試 ===\n")
print(f"查詢: {query}\n")
print("相似法規:")
for idx in sorted_indices[:3]:
    print(f"  [{similarities[idx]:.2f}] {regulations[idx]}")
