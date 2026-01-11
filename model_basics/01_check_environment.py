"""
Workshop 解答：任務 1 - 確認 Hugging Face 環境可以運作

執行方式：
python 01_check_environment.py

預期輸出：
模型資訊：
  名稱: distilbert-base-uncased
  類型: DistilBertModel
  參數量: 66,362,880

Pipeline 測試：
[{'label': 'POSITIVE', 'score': 0.9998...}]
"""

from transformers import AutoModel, AutoTokenizer, pipeline

# 1. 測試載入模型
model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("模型資訊：")
print(f"  名稱: {model_name}")
print(f"  類型: {type(model).__name__}")
print(f"  參數量: {sum(p.numel() for p in model.parameters()):,}")

# 2. 測試 Pipeline
print("\nPipeline 測試：")
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
print(result)
