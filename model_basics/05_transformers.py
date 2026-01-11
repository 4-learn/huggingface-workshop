"""
Workshop 解答：Hugging Face Transformers

題目：
1. 比較 Hugging Face 模型和 API 型 LLM 的差異
2. 分析各自的優缺點
3. 考慮使用情境選擇合適的方案

執行方式：
python 05_transformers.py
"""


def main():
    print("=== Hugging Face Transformers vs API LLM 比較分析 ===\n")

    # === 1. 基本差異比較 ===

    comparison = """
1. 基本差異比較：

┌─────────────────┬─────────────────────────┬─────────────────────────┐
│ 面向            │ Hugging Face 模型       │ API 型 LLM              │
│                 │ (本地/自建)              │ (OpenAI, Claude 等)      │
├─────────────────┼─────────────────────────┼─────────────────────────┤
│ 部署方式        │ 本地安裝、自建伺服器      │ 雲端 API 呼叫            │
│ 模型控制        │ 完全控制                 │ 供應商控制               │
│ 資料隱私        │ 資料不離開內網           │ 資料需傳送到外部          │
│ 硬體需求        │ 需要 GPU（大模型）        │ 無需特殊硬體             │
│ 成本模式        │ 硬體 + 電費 + 維護       │ 按 Token 計費            │
│ 更新維護        │ 需自行更新               │ 供應商自動更新           │
└─────────────────┴─────────────────────────┴─────────────────────────┘
"""
    print(comparison)

    # === 2. 優缺點分析 ===

    pros_cons = """
2. 優缺點分析：

【Hugging Face 模型優點】
✓ 資料隱私：敏感資料不需離開公司內網
✓ 無使用限制：沒有 rate limit，可無限呼叫
✓ 可客製化：可以 fine-tune 調整模型
✓ 離線運作：不需要網路連線
✓ 長期成本：大量使用時成本較低
✓ 模型選擇：數萬個開源模型可選

【Hugging Face 模型缺點】
✗ 初始投資高：需要 GPU 硬體
✗ 技術門檻：需要 ML 專業知識
✗ 維護成本：需自行處理更新、安全
✗ 模型效能：開源模型通常不及頂級 API 模型
✗ 部署複雜：需處理擴展、負載平衡

【API 型 LLM 優點】
✓ 即開即用：註冊即可使用
✓ 頂級效能：使用最強大的模型
✓ 持續更新：自動獲得最新功能
✓ 無維護負擔：供應商處理一切
✓ 彈性擴展：自動處理高流量
✓ 快速原型：適合 POC 和快速開發

【API 型 LLM 缺點】
✗ 資料外洩風險：資料需傳送到外部
✗ 使用成本：按量計費，大量使用昂貴
✗ 供應商依賴：受限於服務條款變更
✗ 網路依賴：需要穩定網路連線
✗ Rate Limit：有呼叫頻率限制
"""
    print(pros_cons)

    # === 3. 使用情境建議 ===

    scenarios = """
3. 使用情境建議：

【選擇 Hugging Face 本地模型】
- 處理敏感資料（醫療、金融、個資）
- 需要離線運作的環境
- 大量持續性的推論需求
- 需要客製化訓練的場景
- 有 GPU 資源且有 ML 團隊

【選擇 API 型 LLM】
- 快速原型開發、POC
- 需要最強模型能力的場景
- 團隊缺乏 ML 專業
- 使用量不確定或變動大
- 開發資源有限

【混合方案（推薦）】
- 敏感資料 → 本地 Hugging Face 模型
- 複雜推理 → API 型 LLM
- 簡單分類 → 輕量本地模型
- 開發測試 → API，正式環境 → 本地
"""
    print(scenarios)

    # === 4. 工安系統應用建議 ===

    recommendation = """
4. 工安系統應用建議：

┌─────────────────────┬─────────────────────────────────────────┐
│ 功能                │ 建議方案                                 │
├─────────────────────┼─────────────────────────────────────────┤
│ 即時影像辨識        │ YOLO + 本地輕量模型（速度優先）          │
│ 違規描述生成        │ API LLM（品質優先）或本地 LLaMA          │
│ 文件 OCR           │ Hugging Face 本地模型（資料隱私）         │
│ 語意搜尋           │ sentence-transformers 本地 Embedding     │
│ 複雜決策判斷        │ API LLM（需要強推理能力）                 │
│ 批量報告生成        │ 本地模型（成本考量）                      │
└─────────────────────┴─────────────────────────────────────────┘

成本估算範例（每月 10 萬次推論）：
- API：約 $500-2000（依模型而定）
- 本地：GPU 租賃約 $200-500 + 電費

結論：建議採用混合架構
- 前端偵測：本地 YOLO + Hugging Face
- 後端分析：API LLM（或自建 LLaMA）
- 資料儲存：完全本地化
"""
    print(recommendation)

    # === 5. 實際程式碼範例對比 ===

    code_comparison = """
5. 程式碼範例對比：

【Hugging Face 本地模型】
```python
from transformers import pipeline

# 載入本地模型（一次性）
classifier = pipeline("text-classification",
                      model="bert-base-chinese")

# 推論（無網路需求）
result = classifier("工人沒戴安全帽")
print(result)
# [{'label': 'VIOLATION', 'score': 0.95}]
```

【API 型 LLM（OpenAI）】
```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

# 每次呼叫都需要網路
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user",
               "content": "分類這個違規：工人沒戴安全帽"}]
)
print(response.choices[0].message.content)
```

【執行時間比較】
- 本地模型首次載入：5-30 秒
- 本地模型推論：10-100 毫秒
- API 呼叫：200-2000 毫秒
"""
    print(code_comparison)

    print("=== 完成 ===")


if __name__ == "__main__":
    main()
