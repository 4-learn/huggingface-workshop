"""
Workshop 解答：Hugging Face 模型基礎

題目要求：
1. 實作文本分類器（使用預訓練模型）
2. 實作文本生成器（Inference）
3. 實作語義搜尋（Embedding）
4. 整合成工安違規分析器

預期輸出：
=== 工安違規分析器 ===

輸入事件：no_helmet detected at construction zone (confidence: 0.87)

1. 違規分類
   類別: safety_violation
   信心度: 95.3%

2. 建議處理方式
   立即通知現場人員配戴安全帽...

3. 相關法規搜尋
   [0.89] 職業安全衛生設施規則 第 281 條：雇主對於...
   [0.76] 營造安全衛生設施標準 第 11-1 條：...

分析完成
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)


# === 資料結構 ===

@dataclass
class ViolationEvent:
    """違規事件"""
    object: str
    confidence: float
    zone: str
    timestamp: datetime


@dataclass
class AnalysisResult:
    """分析結果"""
    event: ViolationEvent
    classification: dict
    suggestion: str
    related_laws: list


# === 工具函數 ===

def mean_pooling(model_output, attention_mask):
    """Mean Pooling 取得句子向量"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# === 文本分類器 ===

class ViolationClassifier:
    """
    違規事件分類器

    使用零樣本分類（Zero-shot Classification）
    不需要針對特定任務訓練
    """

    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.labels = [
            "safety_violation",    # 安全違規
            "equipment_issue",     # 設備問題
            "normal_operation",    # 正常操作
        ]

    def classify(self, event: ViolationEvent) -> dict:
        """分類違規事件"""
        # 建立描述文字
        text = f"{event.object} detected at {event.zone} zone with confidence {event.confidence:.0%}"

        # 零樣本分類
        result = self.classifier(text, self.labels)

        return {
            "label": result["labels"][0],
            "confidence": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"]))
        }


# === 文本生成器 ===

class SuggestionGenerator:
    """
    建議生成器

    使用 GPT-2 生成處理建議
    """

    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, event: ViolationEvent) -> str:
        """生成處理建議"""
        # 建立提示詞
        prompt = self._create_prompt(event)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 解碼
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 只取生成的部分
        generated = full_text[len(prompt):].strip()

        return generated

    def _create_prompt(self, event: ViolationEvent) -> str:
        """建立提示詞"""
        violation_type = event.object.replace("_", " ")
        return f"Safety recommendation for {violation_type} in {event.zone} area:"


# === 語義搜尋 ===

class LawSearcher:
    """
    法規搜尋器

    使用 Sentence Embedding 進行語義搜尋
    """

    # 模擬法規資料庫
    LAWS = [
        {
            "id": "law_001",
            "title": "職業安全衛生設施規則 第 281 條",
            "content": "雇主對於在高度二公尺以上之處所作業，勞工有墜落之虞者，應使勞工確實使用安全帽。"
        },
        {
            "id": "law_002",
            "title": "營造安全衛生設施標準 第 11-1 條",
            "content": "雇主使勞工於營造工作場所作業，應使勞工配戴安全帽。"
        },
        {
            "id": "law_003",
            "title": "職業安全衛生設施規則 第 21 條",
            "content": "雇主對於勞工工作場所之通道、地板、階梯，應保持不致使勞工跌倒之安全狀態。"
        },
        {
            "id": "law_004",
            "title": "職業安全衛生設施規則 第 277 條",
            "content": "雇主對於有車輛出入之場所，應設置反光標誌或反光衣著。"
        },
        {
            "id": "law_005",
            "title": "職業安全衛生設施規則 第 286 條",
            "content": "雇主對於高壓電路之檢查、修理等有接近高壓電路之作業，應使勞工戴用絕緣用防護具。"
        },
    ]

    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # 預先計算法規 embeddings
        self._index_laws()

    def _index_laws(self):
        """建立法規向量索引"""
        # 用標題+內容作為搜尋文本
        texts = [f"{law['title']} {law['content']}" for law in self.LAWS]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        self.law_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        self.law_embeddings = F.normalize(self.law_embeddings, p=2, dim=1)

    def search(self, query: str, top_k: int = 2) -> list:
        """搜尋相關法規"""
        # 取得查詢 embedding
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs, inputs['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # 計算相似度
        similarities = torch.matmul(query_embedding, self.law_embeddings.T)[0]

        # 取得 top-k
        top_indices = torch.topk(similarities, k=min(top_k, len(self.LAWS))).indices

        results = []
        for idx in top_indices:
            idx = idx.item()
            results.append({
                "law": self.LAWS[idx],
                "score": similarities[idx].item()
            })

        return results


# === 整合分析器 ===

class ViolationAnalyzer:
    """
    違規分析器

    整合分類、生成、搜尋功能
    """

    def __init__(self):
        print("載入模型中...")
        self.classifier = ViolationClassifier()
        self.generator = SuggestionGenerator()
        self.searcher = LawSearcher()
        print("模型載入完成\n")

    def analyze(self, event: ViolationEvent) -> AnalysisResult:
        """完整分析違規事件"""
        # 1. 分類
        classification = self.classifier.classify(event)

        # 2. 生成建議
        suggestion = self.generator.generate(event)

        # 3. 搜尋法規
        query = self._build_search_query(event)
        related_laws = self.searcher.search(query, top_k=2)

        return AnalysisResult(
            event=event,
            classification=classification,
            suggestion=suggestion,
            related_laws=related_laws
        )

    def _build_search_query(self, event: ViolationEvent) -> str:
        """建立搜尋查詢"""
        # 根據違規類型建立查詢
        queries = {
            "no_helmet": "安全帽 頭部防護 營造工地",
            "no_vest": "反光背心 高可見度 車輛作業",
            "no_gloves": "防護手套 手部防護",
            "no_glasses": "護目鏡 眼睛防護",
        }
        return queries.get(event.object, event.object)


# === 顯示結果 ===

def display_result(result: AnalysisResult):
    """顯示分析結果"""
    event = result.event

    print("=" * 50)
    print("工安違規分析結果")
    print("=" * 50)

    print(f"\n輸入事件:")
    print(f"  物件: {event.object}")
    print(f"  信心度: {event.confidence:.0%}")
    print(f"  區域: {event.zone}")
    print(f"  時間: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n1. 違規分類")
    print(f"   類別: {result.classification['label']}")
    print(f"   信心度: {result.classification['confidence']:.1%}")

    print(f"\n2. 建議處理方式")
    print(f"   {result.suggestion}")

    print(f"\n3. 相關法規")
    for item in result.related_laws:
        law = item['law']
        score = item['score']
        print(f"   [{score:.2f}] {law['title']}")
        print(f"         {law['content'][:50]}...")

    print(f"\n分析完成")


# === 簡易版（不用載入大模型）===

class SimpleViolationAnalyzer:
    """
    簡易版分析器（用規則取代模型）

    適合快速測試或資源有限的環境
    """

    VIOLATION_RULES = {
        "no_helmet": {
            "classification": "safety_violation",
            "suggestion": "立即通知現場人員配戴安全帽，並記錄違規情況",
            "laws": ["職業安全衛生設施規則 第 281 條", "營造安全衛生設施標準 第 11-1 條"]
        },
        "no_vest": {
            "classification": "safety_violation",
            "suggestion": "提醒人員穿著反光背心，尤其在車輛作業區域",
            "laws": ["職業安全衛生設施規則 第 277 條"]
        },
    }

    def analyze(self, event: ViolationEvent) -> dict:
        """分析違規事件"""
        rule = self.VIOLATION_RULES.get(event.object, {
            "classification": "unknown",
            "suggestion": "請人工確認違規類型",
            "laws": []
        })

        return {
            "event": {
                "object": event.object,
                "confidence": event.confidence,
                "zone": event.zone,
            },
            "classification": rule["classification"],
            "suggestion": rule["suggestion"],
            "related_laws": rule["laws"],
        }


# === 主程式 ===

def main():
    print("=" * 50)
    print("Hugging Face Workshop: 違規分析器")
    print("=" * 50)
    print()

    # 測試事件
    events = [
        ViolationEvent(
            object="no_helmet",
            confidence=0.87,
            zone="construction",
            timestamp=datetime.now(timezone.utc)
        ),
        ViolationEvent(
            object="no_vest",
            confidence=0.75,
            zone="entrance",
            timestamp=datetime.now(timezone.utc)
        ),
    ]

    # 使用簡易版（避免下載大型模型）
    print("使用簡易版分析器（規則式）\n")
    simple_analyzer = SimpleViolationAnalyzer()

    for event in events:
        result = simple_analyzer.analyze(event)
        print(f"事件: {event.object} @ {event.zone}")
        print(f"  分類: {result['classification']}")
        print(f"  建議: {result['suggestion']}")
        print(f"  法規: {result['related_laws']}")
        print()

    print("-" * 50)
    print()

    # 詢問是否使用完整版
    print("是否執行完整版分析器？（需下載模型，約 1-2GB）")
    print("輸入 'yes' 執行，其他跳過")
    user_input = input("> ").strip().lower()

    if user_input == "yes":
        print()
        analyzer = ViolationAnalyzer()

        for event in events:
            result = analyzer.analyze(event)
            display_result(result)
            print()
    else:
        print("\n跳過完整版測試")

    print("\nWorkshop 完成！")


if __name__ == "__main__":
    main()
