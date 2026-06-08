"""個人化提問單模組。

純程式碼生成，不呼叫 LLM。資料來源完全是基線統計。
"""

from __future__ import annotations

from typing import Any

from .baseline import high_frequency_fields


# 基礎欄位——若統計不足，fallback 成這些通用面向
FALLBACK_FIELDS = [
    ("goal", "目標（你這次想完成什麼？）"),
    ("budget", "預算"),
    ("time", "時間／檔期"),
    ("venue", "場地／地點"),
    ("headcount", "人數／規模"),
]

# 類別到中文提示的對照表（對應 baseline.ALLOWED_GAP_CATEGORIES）
CATEGORY_LABELS = {
    "budget": "預算",
    "time": "時間／檔期",
    "deadline": "截止日",
    "venue": "場地／地點",
    "headcount": "人數／規模",
    "audience": "對象／受眾",
    "stakeholder": "相關人員",
    "deliverable": "交付物／成品",
    "quality": "品質要求",
    "risk": "風險／備案",
    "motivation": "動機／目的",
    "resource": "資源／人力",
    "constraint": "其他硬限制",
}


def should_generate(completed_events: int) -> bool:
    """規格書要求累積 5 筆端點後才自動生成提問單。"""
    return completed_events >= 5


def generate_form(baseline: dict[str, Any]) -> list[dict[str, str]]:
    """回傳欄位列表。每個欄位是 {key, label}。

    欄位來源：
    1. 高頻 gap_categories（top_k = 6, threshold = 30%）
    2. 若高頻欄位不足 3 個，補上常見限制條件
    3. 永遠把 goal 放在第一個
    """
    high = high_frequency_fields(baseline, ratio_threshold=0.3, top_k=6)

    fields: list[dict[str, str]] = [
        {"key": "goal", "label": "目標（你這次想完成什麼？）"}
    ]

    seen_keys = {"goal"}
    for cat in high:
        if cat in seen_keys:
            continue
        label = CATEGORY_LABELS.get(cat, cat)
        fields.append({"key": cat, "label": label})
        seen_keys.add(cat)

    # 不足的話，用 fallback 補齊到至少 4 個
    if len(fields) < 4:
        for key, label in FALLBACK_FIELDS:
            if key in seen_keys:
                continue
            fields.append({"key": key, "label": label})
            seen_keys.add(key)
            if len(fields) >= 4:
                break

    return fields


def render_form(fields: list[dict[str, str]]) -> str:
    """把提問單渲染成 CLI 可見的字串，不含使用者輸入框。"""
    lines = [
        "[TOFU] 根據你過去的互動，以下是你通常會考慮的面向：",
    ]
    for i, f in enumerate(fields, start=1):
        lines.append(f"  {i}. {f['label']}：______")
    lines.append("請填寫（逐題輸入），或輸入 /skip 直接用自然語言描述需求。")
    return "\n".join(lines)


def compose_natural_input(
    answers: dict[str, str],
    fields: list[dict[str, str]],
) -> str:
    """把逐題填答結果組合成一段自然語言，等同『使用者的初始輸入』。

    這段自然語言會丟回主流程，重新走一次復述確認。
    """
    parts: list[str] = []
    goal = answers.get("goal", "").strip()
    if goal:
        parts.append(goal)
    detail_bits: list[str] = []
    for f in fields:
        key = f["key"]
        if key == "goal":
            continue
        val = answers.get(key, "").strip()
        if val:
            detail_bits.append(f"{f['label']}：{val}")
    if detail_bits:
        parts.append("，".join(detail_bits))
    return "。".join([p for p in parts if p]) or "（使用者未填寫任何欄位）"
