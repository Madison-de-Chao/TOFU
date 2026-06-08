"""v4.0 P1：冷卻模式。

依 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P1。

判斷使用者是否進入「高情緒狀態」，若是，本輪只做事實釐清、不進入給
建議模式——避免語義倒置（把使用者的邏輯質問當成情緒訴求處理）。

觸發條件（任一滿足）：
1. mood_axis == "差" AND seven_emotion in ("怒", "悲")
2. last_emotion_sample 最後 3 筆全是同一個負向情緒（怒/悲/憂/恐）
3. preference_axis 從 + 切到 -，且 mood_axis 從好切到差

冷卻模式下：
- ATL-3 前驗證閘門不強制交付（has_deliverable 可為 False）
- 回應走 COOLING_MODE_PROMPT_ADDENDUM，只做釐清
- 進入端點寫入時跳過 zone_history（避免污染 ATL-4 歷史）
"""

from __future__ import annotations

from typing import Any


# 冷卻模式下要注入到 execute_task 的 extra_context 前綴。
# 明確列出「做什麼」和「不做什麼」，避免 prompt 模糊導致 LLM 自己解讀。
COOLING_MODE_PROMPT_ADDENDUM = (
    "## 冷卻模式啟動（v4.0）\n\n"
    "使用者當前處於高情緒狀態。在這一輪回覆中：\n\n"
    "只做：\n"
    "- 釐清事實（使用者講的事情，我理解對不對）\n"
    "- 問具體的澄清問題（地點、時間、對象）\n"
    "- 用中性語氣回應\n\n"
    "不做：\n"
    "- 給建議、給方案、給解決辦法\n"
    "- 用「我理解你的感受」「這個確實很困難」這類安撫句型\n"
    "- 加任何情感共鳴的詞彙\n"
    "- 假設情緒會影響判斷、進而「引導」使用者往某個方向\n\n"
    "結束時問一句：「這樣的理解符合你的原意嗎？」\n\n"
    "目的：避免語義倒置（把使用者的邏輯質問當成情緒訴求處理）。\n"
)


_NEGATIVE_EMOTIONS = {"怒", "悲", "憂", "恐"}


def should_enter_cooling_mode(
    emotion_state: dict[str, Any] | None,
    profile_baseline: dict[str, Any] | None,
) -> bool:
    """判斷是否進入冷卻模式。

    三個觸發條件任一滿足即啟動。輸入容錯：任一參數為 None / 空 dict 都不觸發。
    """
    if not emotion_state:
        return False

    mood = emotion_state.get("mood_axis")
    seven = emotion_state.get("seven_emotion")
    pref = emotion_state.get("preference_axis")

    # 條件 1：壞心情 + 怒/悲
    if mood == "差" and seven in ("怒", "悲"):
        return True

    if not profile_baseline:
        profile_baseline = {}

    recent = list(profile_baseline.get("last_emotion_sample", []) or [])

    # 條件 2：最近 3 筆都是同一個負向情緒
    if len(recent) >= 3:
        last3 = recent[-3:]
        last3_emotions = [r.get("seven_emotion") for r in last3 if isinstance(r, dict)]
        if (
            len(last3_emotions) == 3
            and all(e in _NEGATIVE_EMOTIONS for e in last3_emotions)
            and len(set(last3_emotions)) == 1
        ):
            return True

    # 條件 3：上一輪 + / 好 → 本輪 - / 差
    if recent:
        last = recent[-1] if isinstance(recent[-1], dict) else {}
        if (
            last.get("preference_axis") == "+"
            and pref == "-"
            and last.get("mood_axis") == "好"
            and mood == "差"
        ):
            return True

    return False


def append_emotion_sample(
    profile_baseline: dict[str, Any],
    emotion_state: dict[str, Any],
    max_samples: int = 10,
) -> dict[str, Any]:
    """把本輪 emotion_state 附加到 profile_baseline.last_emotion_sample。

    max_samples 為滾動視窗上限（spec §P1 預設 10）。
    """
    if not isinstance(profile_baseline, dict):
        profile_baseline = {}
    samples = list(profile_baseline.get("last_emotion_sample", []) or [])
    samples.append(emotion_state)
    profile_baseline["last_emotion_sample"] = samples[-max_samples:]
    return profile_baseline
