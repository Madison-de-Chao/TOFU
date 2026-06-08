"""v4.0 P1：情緒三軸模型——偵測層。

依 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P1。
理論依據：《人機協作 AI 與人類雙向教育系統 v1.1》第九章——中醫七情 ×
偏好軸 × 心情軸。

兩層偵測：
- 規則式（本模組 `detect_emotion_rule_based`）：低成本、快速，信心度 0.6。
- LLM 輔助（claude_client 的 generate_restate 在 prompt 裡搭車）：
  不額外打 API，精度較高但依賴 LLM 輸出結構。

輸出結構（emotion_state）：
{
    "seven_emotion": "喜|怒|憂|思|悲|恐|驚|中性",
    "preference_axis": "+|-|中性",
    "mood_axis": "好|差|中性",
    "confidence": float,    # 0.0-1.0
    "detection_source": "rule_based" | "llm_inferred",
}
"""

from __future__ import annotations

import re
from typing import Any


# 七情關鍵字表（繁中 + 英文）
SEVEN_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "喜": [
        "開心", "高興", "興奮", "哈哈", "太好了", "讚", "爽",
        "happy", "excited", "love it", "awesome", "yay",
    ],
    "怒": [
        "生氣", "氣死", "幹", "靠", "煩死", "氣炸", "惹毛",
        "angry", "pissed", "wtf", "pissed off", "fuck",
    ],
    "憂": [
        "擔心", "煩惱", "焦慮", "怕來不及", "壓力",
        "worried", "anxious", "concerned", "stressed",
    ],
    "思": [
        "思考", "想想", "考慮", "評估", "盤算", "權衡",
        "thinking", "considering", "weighing",
    ],
    "悲": [
        "難過", "失望", "沮喪", "哭", "心碎", "低落",
        "sad", "disappointed", "down", "heartbroken",
    ],
    "恐": [
        "害怕", "恐懼", "不敢", "怕怕", "嚇死",
        "scared", "afraid", "terrified", "frightened",
    ],
    "驚": [
        "嚇到", "驚訝", "天啊", "不會吧", "哇靠",
        "shocked", "surprised", "oh my god", "no way",
    ],
}

# 偏好軸——正面
PREFERENCE_POSITIVE = [
    "喜歡", "愛", "讚", "不錯", "好棒", "蠻好", "挺好",
    "like", "love", "great", "awesome", "enjoy",
]

# 偏好軸——負面
PREFERENCE_NEGATIVE = [
    "討厭", "爛", "糟", "不要", "受不了", "不喜歡",
    "hate", "dislike", "awful", "terrible", "can't stand",
]

# 心情軸——正向信號
_MOOD_POSITIVE_MARKERS = [
    ":)", ":-)", ":D", ":-D",
    "😊", "😄", "🥰", "❤️", "👍",
    "haha", "哈哈", "呵呵", "嘿嘿",
]

# 心情軸——髒話（觸發「差」）
_PROFANITY = [
    "幹", "靠", "靠北", "他媽", "她媽", "shit", "fuck", "damn", "bullshit",
]

# 七情優先順序（用於同段同時命中多種情緒時的仲裁）。原則：
# 負向優於正向（因為誤放正向的風險 < 誤放負向）；強烈的先於弱烈的。
_SEVEN_EMOTION_PRIORITY = ["怒", "悲", "恐", "憂", "驚", "喜", "思"]


# ----------------------------------------------------------------------
# v4.0 P1：社交收尾語偵測
# ----------------------------------------------------------------------
# CASE_05 對話 35 實測發現：「happy knitting」「hope your beanie turns out」這類
# 社交收尾語會觸發 "happy" 關鍵字被判為「喜」，但語境上是禮貌性收尾、不是
# 實際情緒表達。為避免誤判進而影響冷卻模式邏輯，在規則式偵測前先跑一層
# 社交收尾語偵測——命中即強制判為「中性 / 中性 / 中性」。
#
# 偵測策略：同時命中兩個以上「社交動詞 + 受益者語氣」片語才算社交收尾。
# 單獨一個「happy」不算，避免誤傷真實的喜悅表達。
_SOCIAL_FAREWELL_MARKERS = [
    # 英文
    "feel free to ask", "feel free to reach",
    "hope your", "hope it", "hope you", "hope the",
    "happy knitting", "happy learning", "happy coding", "happy reading",
    "glad i could help", "glad i could assist",
    "wish you the best", "wish you luck", "good luck",
    "turns out wonderful", "turns out great", "turns out well",
    "if you have any more", "if you need further",
    # 中文
    "有任何問題", "如果還有需要", "隨時歡迎", "祝你",
    "聊得很愉快", "聊得開心", "謝謝你的幫助", "感謝你",
    "祝福你", "祝你成功", "祝你順利",
]


def is_social_farewell(text: str) -> bool:
    """判斷一段文字是否為社交收尾語。

    需要同時命中 ≥ 2 個社交片語、或單一強烈收尾片語（例如「happy knitting」）
    才算社交收尾，避免誤傷真實情緒表達。
    """
    if not text:
        return False
    lowered = text.lower()
    hits = [m for m in _SOCIAL_FAREWELL_MARKERS if m in lowered]
    if len(hits) >= 2:
        return True
    # 單一強烈片語：happy + 動名詞（knitting/learning/coding…）
    strong = [m for m in hits if m.startswith("happy ") and m != "happy"]
    if strong:
        return True
    return False


def detect_mood(text: str) -> str:
    """偵測心情軸：好 / 差 / 中性。

    標點符號同時計入半形（ASCII `!` / `?`）與全形（`！` U+FF01 / `？` U+FF1F）；
    任一方向連續 ≥ 3 個視為高情緒強度信號。
    """
    if not text:
        return "中性"
    exclamations = text.count("!") + text.count("！")
    questions = text.count("?") + text.count("？")
    has_profanity = any(w in text for w in _PROFANITY)
    has_repeat = bool(re.search(r"(.)\1{3,}", text))  # 連續 4 個以上相同字
    if exclamations >= 3 or questions >= 3 or has_profanity or has_repeat:
        return "差"
    if any(m in text.lower() for m in (mm.lower() for mm in _MOOD_POSITIVE_MARKERS)):
        return "好"
    return "中性"


def detect_seven_emotion(text: str) -> str:
    """偵測七情。同時命中多種時依優先序仲裁。"""
    if not text:
        return "中性"
    lowered = text.lower()
    hit: set[str] = set()
    for emotion, keywords in SEVEN_EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lowered:
                hit.add(emotion)
                break
    if not hit:
        return "中性"
    for candidate in _SEVEN_EMOTION_PRIORITY:
        if candidate in hit:
            return candidate
    # 理論上不會到這：hit 非空但不在優先序 → 隨便挑一個
    return next(iter(hit))


def detect_preference_axis(text: str) -> str:
    """偵測偏好軸：+ / - / 中性。負面優先（避免誤判）。"""
    if not text:
        return "中性"
    lowered = text.lower()
    if any(k.lower() in lowered for k in PREFERENCE_NEGATIVE):
        return "-"
    if any(k.lower() in lowered for k in PREFERENCE_POSITIVE):
        return "+"
    return "中性"


def detect_emotion_rule_based(text: str) -> dict[str, Any]:
    """規則式情緒三軸偵測。低成本、快速，信心度 0.6。

    v4.0 新增：社交收尾語命中時強制回中性 / 中性 / 中性，避免「happy knitting」
    這類禮貌語氣誤觸「喜」。
    """
    if is_social_farewell(text):
        return {
            "seven_emotion": "中性",
            "preference_axis": "中性",
            "mood_axis": "中性",
            "confidence": 0.7,  # 社交收尾判斷較確定、信心度略高
            "detection_source": "rule_based_social_farewell",
        }
    return {
        "seven_emotion": detect_seven_emotion(text),
        "preference_axis": detect_preference_axis(text),
        "mood_axis": detect_mood(text),
        "confidence": 0.6,
        "detection_source": "rule_based",
    }


def normalize_emotion_state(raw: Any) -> dict[str, Any] | None:
    """把 LLM 回傳的 emotion_state 正規化為標準結構。

    LLM 可能給出不完整或不合法的欄位；本函式過濾回合法集合，
    不合法的替回「中性」/「中性」/「中性」，信心度 0.5。
    None 或無法解析時回 None（caller 決定退回規則式偵測）。
    """
    if not isinstance(raw, dict):
        return None

    valid_seven = {"喜", "怒", "憂", "思", "悲", "恐", "驚", "中性"}
    valid_pref = {"+", "-", "中性"}
    valid_mood = {"好", "差", "中性"}

    seven = str(raw.get("seven_emotion", "中性"))
    if seven not in valid_seven:
        seven = "中性"
    pref = str(raw.get("preference_axis", "中性"))
    if pref not in valid_pref:
        pref = "中性"
    mood = str(raw.get("mood_axis", "中性"))
    if mood not in valid_mood:
        mood = "中性"

    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    return {
        "seven_emotion": seven,
        "preference_axis": pref,
        "mood_axis": mood,
        "confidence": conf,
        "detection_source": str(raw.get("detection_source", "llm_inferred")),
    }
