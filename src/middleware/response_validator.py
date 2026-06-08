"""v0.6 PR-G step 2：LLM 回應的迴避句型偵測 + 重試 hint 產生器。

MOMO 2026-04-16 bug report：逗福在 default / free / propose_final 模式下
會給「方法論」而非「答案」——典型句型：

- 「你可以思考一下...」「你可以想想看...」
- 「你可以在搜尋引擎輸入...」「你可以 google 看看」
- 「建議你先確認 X 然後再決定」「請你再補充 Y」
- 「我建議你先從 Z 中選出至少三種」（偽裝成提案的反問）

Prompt 層禁令（PR-F）不完全可靠——這裡加 runtime 層偵測：
1. 每次 execute_task 回應後，掃「最終提案 / 一般回應」文字
2. 偵測到迴避句型 → 呼叫端決定是否重試一次（需 guardrails 同意）
3. 重試仍迴避 → 呼叫端可以在結果前面加「[逗福警告] 本回應可能未完整…」

本模組只提供**偵測 + hint 建構**，不自行觸發重試——retry 的決策在
main.py 的 run_one_interaction / run_propose_mode 裡，因為那裡才能
access client / guardrails。

本模組不呼叫 LLM，純字串處理；測試不需要網路或 API key。
"""

from __future__ import annotations


# 迴避句型清單。順序會影響 detect 回傳的順序（但不影響「是否偵測到」布林）。
# 以「明顯且常見」為選擇標準——寧可漏掉邊緣案例，也不要誤判正常輸出。
# 排除可能誤判的短句（例如「思考」「搜尋」本身可能出現在合法內容中）。
EVASION_PATTERNS: tuple[str, ...] = (
    # 「你可以 ___」推諉句型
    "你可以思考一下",
    "你可以想想看",
    "你可以想想",
    "你可以在搜尋引擎",
    "你可以上網搜尋",
    "你可以上網查",
    "你可以 google",
    "你可以谷歌",
    "你可以先思考",
    # 「建議你先 ___」推延句型
    "建議你先確認",
    "建議你先想",
    "建議你先從",
    "建議你先思考",
    "建議你再補",
    "建議你再提供",
    "建議你再想",
    # 「請你 ___」把工作丟回使用者
    "請你再補",
    "請你再提供",
    "請你再確認",
    "請你先補",
    "請你先確認",
    "請你補充",
    # 「我建議你先從 X 選出…」偽裝成提案的反問（MOMO 原始觀察）
    "我建議你先從",
)


# 這些模式即使 mode 是 default / free / propose_final，也允許出現（正向使用）。
# 例如：「若你不確定，建議你先查 X」——這在 prompt 層本來就是允許 fallback。
# 暫時不做白名單（keep simple），但標記位置方便未來加。
ALLOWED_EVASION_CONTEXTS: tuple[str, ...] = ()


# 哪些 mode 要跑偵測。/risk 模式輸出「觸發條件 / 建議應對」本來就會出現
# 部分類似句型（例如「建議你先確認場地容量」），跑偵測會誤判，排除。
ENFORCED_MODES: frozenset[str] = frozenset({"default", "free", "propose_final"})


# 重試後若仍迴避，在回應前加上此前綴，讓使用者知道這份結果被降級。
DEGRADED_WARNING_PREFIX = (
    "[逗福Tofu 警告] 本回應連續兩次偵測到迴避句型，可能未完整處理你的需求。"
    "建議補充更具體的資訊後重跑，或改用 /propose 模式。\n\n"
)


def detect_evasion_phrases(text: str) -> list[str]:
    """掃描文字中是否包含 EVASION_PATTERNS 中任何句型。

    回傳命中的句型清單（依 EVASION_PATTERNS 原順序、去重）。空 list 表示
    沒偵測到迴避。大小寫不敏感（Google/google 都會命中）。
    """
    if not text:
        return []
    lowered = text.lower()
    hits: list[str] = []
    for pattern in EVASION_PATTERNS:
        if pattern.lower() in lowered and pattern not in hits:
            hits.append(pattern)
    return hits


def is_evasive_response(text: str, mode: str) -> bool:
    """Policy：特定 mode 下若命中任一迴避句型即視為 evasive。

    只對 `default / free / propose_final` 生效；其他 mode（risk / propose
    收資訊輪 / check）一律回 False，避免誤判正常輸出。
    """
    if mode not in ENFORCED_MODES:
        return False
    return bool(detect_evasion_phrases(text))


def build_retry_hint(matched_phrases: list[str]) -> str:
    """產生附在下一次 execute_task extra_context 前面的重試 hint。

    目的：讓 Haiku 清楚看到「上一次輸出哪裡違規」，並要求用具體內容重寫。
    只列出最多 3 個命中句型（避免 prompt 過長 / 給太多負面範例）。
    """
    if not matched_phrases:
        return ""
    shown = matched_phrases[:3]
    quoted = "、".join(f"「{p}」" for p in shown)
    return (
        "## ⚠ 上一次輸出偵測到迴避句型，本次必須重寫\n\n"
        f"上一次輸出出現了以下禁止的句型：{quoted}。\n"
        "這代表你把思考 / 查詢 / 補充資訊的工作丟回給使用者了。\n\n"
        "**本次重寫必須**：\n"
        "- 直接給具體名稱、地點、價格、時段、店家或產品名\n"
        "- 不要用「你可以思考」「建議你先確認」「請你再補充」這類推諉句型\n"
        "- 確實不知道具體答案時，直接說「這個我不確定，建議查 X」，"
        "**不要**包裝成「你可以從三個方向思考」這種方法論\n"
        "- 使用者要的是「幫我做」，不是「教我怎麼找答案」\n\n"
        "---\n\n"
    )


def mark_degraded(text: str) -> str:
    """在原回應前面加降級警告前綴，讓使用者知道連續兩次都失敗。"""
    return DEGRADED_WARNING_PREFIX + text


__all__ = [
    "EVASION_PATTERNS",
    "ENFORCED_MODES",
    "DEGRADED_WARNING_PREFIX",
    "detect_evasion_phrases",
    "is_evasive_response",
    "build_retry_hint",
    "mark_degraded",
]
