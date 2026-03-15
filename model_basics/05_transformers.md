# Workshop 解答：Hugging Face Transformers 進階控制

## 題目

針對以下 5 個需求，選擇合適的 Pipeline task 和參數設定。

---

## 解答

| # | 需求 | Pipeline task | 建議參數 | 理由 |
|---|------|--------------|---------|------|
| 1 | 判斷客戶留言是正面還是負面 | `sentiment-analysis` | 預設即可 | 二元分類，預設模型就夠用 |
| 2 | 自動產生違規報告的摘要 | `summarization` | `max_length=80, min_length=20` | 摘要不能太短也不能太長，需控制長度 |
| 3 | 預測句子中缺少的關鍵字 | `fill-mask` | 預設即可 | BERT 類模型的核心能力 |
| 4 | 用一句提示讓模型寫出完整段落 | `text-generation` | `max_length=100, temperature=0.7` | 中等 temperature 兼顧品質與多樣性 |
| 5 | 需要非常精確、不能亂講的報告 | `text-generation` | `temperature=0.1` | 低 temperature → 保守、穩定的輸出 |

---

## 判斷原則

| 參數 | 低值效果 | 高值效果 |
|------|---------|---------|
| **temperature** | 保守、重複、精確 | 多樣、創意、可能離題 |
| **max_length** | 輸出短，適合標題/摘要 | 輸出長，適合段落/文章 |
| **top_k** | 只從最可能的幾個詞選 | 從更多候選詞中選 |

### 選模型的時機

| 情境 | 做法 |
|------|------|
| 快速測試 | 用預設模型：`pipeline("task")` |
| 需要中文支援 | 指定模型：`pipeline("task", model="bert-base-chinese")` |
| 需要更好效果 | 去 Hub 搜尋該 task 的高分模型 |
