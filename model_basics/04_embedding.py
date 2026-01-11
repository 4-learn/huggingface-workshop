"""
Workshop 解答：Embedding 概念

題目：
1. 準備 10 個違規描述
2. 將它們轉換成 Embedding
3. 實作搜尋功能：輸入新描述，找出最相似的 3 個

執行方式：
python 04_embedding.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    print("=== Embedding 相似度搜尋 ===\n")

    # 載入模型
    print("載入模型...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("模型載入完成\n")

    # === 1. 準備 10 個違規描述 ===
    violations = [
        "工人未配戴頭部防護具",
        "員工沒有穿戴安全帽",
        "施工人員安全帽缺失",
        "高空作業未戴頭盔",
        "未穿著反光背心在車道行走",
        "緊急出口被貨物阻擋",
        "倉庫通道堆放雜物",
        "電線裸露在地面上",
        "機台旁邊有積水",
        "員工在禁煙區吸煙",
    ]

    print("1. 違規描述資料庫：")
    for i, v in enumerate(violations, 1):
        print(f"   {i}. {v}")

    # === 2. 將描述轉換成 Embedding ===
    print("\n2. 計算 Embeddings...")
    embeddings = model.encode(violations)
    print(f"   Embedding 維度: {embeddings.shape}")
    print(f"   每個向量大小: {embeddings[0].shape}")

    # === 3. 實作搜尋功能 ===
    def search_similar(query: str, top_k: int = 3):
        """搜尋最相似的違規描述"""
        # 計算查詢的 embedding
        query_embedding = model.encode(query)

        # 計算餘弦相似度
        similarities = np.dot(embeddings, query_embedding)

        # 正規化（轉換成 0-1 範圍）
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = similarities / norms

        # 排序取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": violations[idx],
                "score": float(similarities[idx]),
            })
        return results

    # === 測試搜尋 ===
    test_queries = [
        "工人沒戴安全帽在高處作業",
        "通道被東西擋住了",
        "地上有水很滑",
    ]

    print("\n3. 搜尋測試：")
    for query in test_queries:
        print(f"\n   輸入：{query}")
        results = search_similar(query, top_k=3)
        print("   最相似：")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['text']}（相似度：{r['score']:.2f}）")

    # === 額外：展示 Embedding 的特性 ===
    print("\n" + "=" * 50)
    print("\n4. Embedding 特性展示：")

    # 計算兩個相似描述的相似度
    similar_pair = ["工人未配戴頭部防護具", "員工沒有穿戴安全帽"]
    emb1 = model.encode(similar_pair[0])
    emb2 = model.encode(similar_pair[1])
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\n   相似描述對比：")
    print(f"   「{similar_pair[0]}」")
    print(f"   「{similar_pair[1]}」")
    print(f"   相似度：{sim:.3f}")

    # 計算兩個不相似描述的相似度
    diff_pair = ["工人未配戴頭部防護具", "員工在禁煙區吸煙"]
    emb1 = model.encode(diff_pair[0])
    emb2 = model.encode(diff_pair[1])
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\n   不相似描述對比：")
    print(f"   「{diff_pair[0]}」")
    print(f"   「{diff_pair[1]}」")
    print(f"   相似度：{sim:.3f}")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
