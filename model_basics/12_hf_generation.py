"""
Workshop 解答：Hugging Face - 生成文本

題目：
1. 使用相同的 prompt，測試不同的 temperature
2. 比較 do_sample=True 和 False 的差異
3. 生成 5 個告警訊息，選出最好的

執行方式：
python 12_hf_generation.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
    print("=== Hugging Face 文本生成參數測試 ===\n")

    # 載入模型（使用較小的 GPT-2 示範）
    print("載入模型中...")
    model_name = "gpt2"  # 實際應用可換成中文模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 設定 pad token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"模型: {model_name}")
    print("-" * 60)

    # === 1. Temperature 測試 ===

    print("\n1. Temperature 參數測試")
    print("=" * 60)

    prompt = "Safety alert: Worker without helmet detected in"

    temperatures = [0.1, 0.5, 0.7, 1.0, 1.5]

    print(f"\nPrompt: {prompt}\n")

    for temp in temperatures:
        print(f"--- Temperature = {temp} ---")

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"輸出: {generated}\n")

    # === Temperature 分析 ===

    temp_analysis = """
Temperature 分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────┬────────────────────────────────────────────────┐
│ Temperature │ 特性                                           │
├─────────────┼────────────────────────────────────────────────┤
│ 0.1         │ 非常確定，幾乎確定性輸出，最安全但最無聊        │
│ 0.5         │ 較確定，有輕微變化，適合正式文件               │
│ 0.7         │ 平衡，有適度創意，適合一般應用                 │
│ 1.0         │ 標準隨機，較有創意，可能偏離主題               │
│ 1.5         │ 高度隨機，非常創意但可能產生無意義內容          │
└─────────────┴────────────────────────────────────────────────┘
"""
    print(temp_analysis)

    # === 2. do_sample 比較 ===

    print("\n2. do_sample 參數比較")
    print("=" * 60)

    print(f"\nPrompt: {prompt}\n")

    # do_sample = False (Greedy)
    print("--- do_sample = False (Greedy Decoding) ---")
    for i in range(3):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"第 {i+1} 次: {generated}")

    print("\n--- do_sample = True (Sampling) ---")
    for i in range(3):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"第 {i+1} 次: {generated}")

    # === do_sample 分析 ===

    sample_analysis = """
do_sample 分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【do_sample = False (Greedy Decoding)】
- 每次選擇機率最高的 token
- 輸出完全確定，每次都一樣
- 適合需要一致性的場景
- 缺點：可能產生重複內容

【do_sample = True (Sampling)】
- 根據機率分布隨機選擇 token
- 輸出每次可能不同
- 需要搭配 temperature 使用
- 適合需要多樣性的場景
"""
    print(sample_analysis)

    # === 3. 生成告警訊息 ===

    print("\n3. 生成 5 個告警訊息並評選")
    print("=" * 60)

    alert_prompt = "SAFETY VIOLATION ALERT: A worker was detected without proper PPE."

    print(f"\nPrompt: {alert_prompt}\n")
    print("生成 5 個告警訊息：\n")

    generated_alerts = []

    for i in range(5):
        inputs = tokenizer(alert_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_alerts.append(generated)
        print(f"【版本 {i+1}】")
        print(f"{generated}\n")

    # === 評選標準 ===

    selection_criteria = """
評選標準
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

選擇最佳告警訊息的標準：

1. 【清晰度】訊息是否清楚說明違規內容？
   - 明確指出違規類型
   - 說明發生地點
   - 包含時間資訊

2. 【完整度】是否包含必要資訊？
   - 違規描述
   - 風險說明
   - 處理建議

3. 【專業度】用語是否專業適當？
   - 使用正式用語
   - 避免口語化表達
   - 符合工安術語

4. 【可行動性】是否包含行動指引？
   - 告知下一步
   - 指出責任人
   - 提供時限

5. 【無幻覺】是否包含虛假資訊？
   - 不編造數據
   - 不虛構條款
   - 不臆測原因
"""
    print(selection_criteria)

    # === 推薦配置 ===

    recommendations = """
工安系統推薦配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【生成告警訊息】
- do_sample: True
- temperature: 0.5
- top_p: 0.9
- max_new_tokens: 100

【分類任務】
- do_sample: False
- temperature: 不適用
- max_new_tokens: 10

【摘要生成】
- do_sample: True
- temperature: 0.3
- top_p: 0.95
- max_new_tokens: 200

【創意內容（如安全標語）】
- do_sample: True
- temperature: 0.8
- top_p: 0.9
- max_new_tokens: 50
- 生成多個版本供選擇

範例程式碼：
```python
def generate_alert(violation_text, num_versions=1):
    results = []
    for _ in range(num_versions):
        output = model.generate(
            prompt,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            max_new_tokens=100
        )
        results.append(output)
    return results
```
"""
    print(recommendations)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
