"""
Workshop 解答：Hugging Face - 模型互動

題目：
1. 載入中文 BERT 模型
2. 實作 tokenization 流程
3. 執行推論並解析結果

執行方式：
python 11_hf_interaction.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main():
    print("=== Hugging Face 模型互動實作 ===\n")

    # === 1. 載入中文 BERT 模型 ===

    print("1. 載入模型和 Tokenizer...")
    print("-" * 50)

    model_name = "bert-base-chinese"

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   Tokenizer: {model_name}")

    # 載入模型（這裡用預訓練模型示範）
    # 實際分類任務需要微調過的模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # 假設 3 個分類：輕微、中等、嚴重
    )
    print(f"   Model: {model_name}")
    print(f"   分類數量: 3 (輕微/中等/嚴重)")

    # === 2. 實作 Tokenization 流程 ===

    print("\n2. Tokenization 流程")
    print("-" * 50)

    text = "工人違規操作機台"
    print(f"   輸入文字: {text}")

    # Tokenize
    tokens = tokenizer.tokenize(text)
    print(f"   Tokens: {tokens}")

    # 轉換為 IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"   Token IDs: {token_ids}")

    # 完整的編碼（包含特殊 token）
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    print(f"\n   完整編碼結果:")
    print(f"   - input_ids: {encoded['input_ids'].tolist()}")
    print(f"   - attention_mask: {encoded['attention_mask'].tolist()}")
    print(f"   - token_type_ids: {encoded['token_type_ids'].tolist()}")

    # 解碼驗證
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"\n   解碼驗證: {decoded}")

    # === 3. 執行推論並解析結果 ===

    print("\n3. 執行推論")
    print("-" * 50)

    # 設定為評估模式
    model.eval()

    # 執行推論
    with torch.no_grad():
        outputs = model(**encoded)

    # 取得 logits
    logits = outputs.logits
    print(f"   原始輸出 (logits): {logits.tolist()[0]}")

    # 轉換為機率
    probabilities = torch.softmax(logits, dim=1)
    print(f"   機率分布: {[f'{p:.3f}' for p in probabilities.tolist()[0]]}")

    # 取得預測類別
    predicted_class = torch.argmax(probabilities, dim=1).item()
    class_names = ["輕微", "中等", "嚴重"]
    confidence = probabilities[0][predicted_class].item()

    print(f"\n   預測結果:")
    print(f"   - 類別: {class_names[predicted_class]}")
    print(f"   - 信心度: {confidence:.2%}")

    # === 完整範例函數 ===

    print("\n" + "=" * 50)
    print("完整分類函數示範")
    print("=" * 50)

    def classify_violation(text: str) -> dict:
        """
        違規嚴重程度分類函數

        Args:
            text: 違規描述文字

        Returns:
            dict: 包含預測類別和信心度
        """
        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # 推論
        model.eval()
        with torch.no_grad():
            outputs = model(**encoded)

        # 解析結果
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        class_names = ["輕微", "中等", "嚴重"]

        return {
            "text": text,
            "predicted_class": class_names[predicted_class],
            "confidence": probabilities[0][predicted_class].item(),
            "all_probabilities": {
                name: prob.item()
                for name, prob in zip(class_names, probabilities[0])
            }
        }

    # 測試多個案例
    test_cases = [
        "工人違規操作機台",
        "員工未配戴安全帽",
        "地上有少許垃圾",
        "電線裸露在走道上",
        "高空作業未繫安全帶"
    ]

    print("\n測試結果:")
    print("-" * 50)

    for text in test_cases:
        result = classify_violation(text)
        print(f"\n輸入: {result['text']}")
        print(f"預測: {result['predicted_class']} (信心度: {result['confidence']:.2%})")
        print(f"機率分布: ", end="")
        for name, prob in result['all_probabilities'].items():
            print(f"{name}={prob:.2%} ", end="")
        print()

    # === 注意事項 ===

    print("\n" + "=" * 50)
    print("注意事項")
    print("=" * 50)

    notes = """
⚠️ 重要說明：
1. 上述範例使用的是預訓練 BERT，未經微調
2. 實際應用需要用標註資料微調模型
3. 預測結果僅供示範，不代表實際分類能力

📝 實際應用步驟：
1. 準備標註資料（違規描述 + 嚴重程度標籤）
2. 微調 BERT 模型
3. 評估模型效能
4. 部署到生產環境

💡 建議：
- 至少準備 1000+ 筆標註資料
- 使用交叉驗證評估
- 考慮使用更適合中文的模型如 chinese-roberta-wwm
"""
    print(notes)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
