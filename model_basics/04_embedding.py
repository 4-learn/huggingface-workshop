"""
Workshop 解答：Embedding 概念

題目：
1. 把 violations 換成你自己想的 5 個違規描述
2. 用 3 個不同的查詢句子搜尋，觀察結果

執行前先安裝：
pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util

# 載入模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# === 1. 換成你自己的違規描述 ===
violations = [
    "未穿著反光背心在車道行走",
    "高空作業未繫安全帶",
    "倉庫通道堆放雜物",
    "機台旁邊有積水未清理",
    "施工人員安全帽缺失",
]

# 轉成 Embedding
embeddings = model.encode(violations)

# === 2. 用 3 個不同的查詢搜尋 ===
queries = [
    "有人在馬路上沒穿反光衣",
    "地上濕濕的很危險",
    "工人沒有戴頭部護具",
]

for query in queries:
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, embeddings)[0]

    results = sorted(zip(violations, scores), key=lambda x: x[1], reverse=True)

    print(f"查詢：「{query}」")
    for i, (text, score) in enumerate(results[:3], 1):
        print(f"  {i}. {text}（相似度：{score:.2f}）")
    print()
