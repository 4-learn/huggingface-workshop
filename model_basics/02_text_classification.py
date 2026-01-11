"""
Workshop 解答：任務 2 - 使用 Pipeline 進行文本分類

執行方式：
python 02_text_classification.py

預期輸出：
=== 零樣本分類測試 ===

事件: 工人沒有配戴安全帽
分類結果:
  safety_violation: 92.3%
  equipment_issue: 5.1%
  normal_operation: 2.6%

事件: 機台運作正常
分類結果:
  normal_operation: 88.7%
  ...
"""

from transformers import pipeline

# 建立零樣本分類器
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 定義分類標籤
labels = [
    "safety_violation",    # 安全違規
    "equipment_issue",     # 設備問題
    "normal_operation",    # 正常操作
]

# 測試案例
test_events = [
    "工人沒有配戴安全帽",
    "機台運作正常",
    "倉庫緊急出口被雜物阻擋",
]

print("=== 零樣本分類測試 ===\n")

for event in test_events:
    result = classifier(event, labels)

    print(f"事件: {event}")
    print("分類結果:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.1%}")
    print()
