"""邏輯基線模組。

**全部由純程式碼完成，不呼叫 LLM。**

從歷史端點統計：
1. 高頻目標關鍵詞
2. 高頻限制條件（按原字串與關鍵詞雙維度統計）
3. 高頻被補位的面向（gap_categories）
4. 使用者修正的常見類型

規格書要求「3 筆端點後」基線就開始影響補位提問，因此門檻故意設得很低。

v2.0 P0-1：斷詞改為呼叫 tokenizer.tokenize()，不再自行做 2-gram。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.utils.tokenizer import tokenize as _tokenize

# 規格書允許 MVP 用固定 gap_categories 清單，這裡定義主要維度。
# LLM 只能從此清單中選，才能做出乾淨的頻率統計。
ALLOWED_GAP_CATEGORIES = [
    "budget",       # 預算
    "time",         # 時間／檔期
    "deadline",     # 截止日
    "venue",        # 場地／地點
    "headcount",    # 人數／規模
    "audience",     # 對象／受眾
    "stakeholder",  # 相關人員
    "deliverable",  # 交付物／成品
    "quality",      # 品質要求
    "risk",         # 風險／備案
    "motivation",   # 動機／目的
    "resource",     # 資源／人力
    "constraint",   # 其他硬限制
]


def compute_baseline(
    endpoints: list[dict[str, Any]],
    topic_filter: str | None = None,
) -> dict[str, Any]:
    """輸入：原始端點列表（start 與 end 混雜），輸出：基線 dict。

    只看 start 類型端點。空輸入會回傳全空的基線。

    P0-4：topic_filter 指定時，只統計該 topic 的 start 端點。
    用於防止不同事件的基線互相污染（CBP：Case Boundary Protocol）。

    v0.6 PR-B：排除 ``mode == 'check'`` 的端點。/check 模式是事實查核子系統，
    其內容（使用者貼的詐騙訊息、新聞等）不是「使用者的需求」，
    不應該污染使用者畫像與 gap_categories 統計。舊端點沒有 mode 欄位時
    視為 ``default``，全部納入統計。
    """
    starts = [e for e in endpoints if e.get("type") == "start" and e.get("start_data")]
    # v0.6 PR-B：排除 /check 模式端點
    starts = [
        e for e in starts
        if (e.get("start_data") or {}).get("mode", "default") != "check"
    ]
    if topic_filter:
        starts = [
            e for e in starts
            if (e.get("start_data") or {}).get("topic") == topic_filter
        ]
    total = len(starts)

    goal_kw_counter: Counter[str] = Counter()
    constraint_raw_counter: Counter[str] = Counter()
    constraint_kw_counter: Counter[str] = Counter()
    gap_cat_counter: Counter[str] = Counter()
    modification_counter: Counter[str] = Counter()
    modified_n = 0

    for row in starts:
        sd = row["start_data"]
        goal_kw_counter.update(_tokenize(sd.get("goal") or sd.get("user_input") or ""))
        for c in sd.get("constraints") or []:
            if not c:
                continue
            constraint_raw_counter[c.strip()] += 1
            constraint_kw_counter.update(_tokenize(c))
        for cat in sd.get("gap_categories") or []:
            if cat:
                gap_cat_counter[cat.strip().lower()] += 1
        if (sd.get("user_confirmation") or "").lower() == "modified":
            modified_n += 1
            mod_text = sd.get("user_modifications") or ""
            modification_counter.update(_tokenize(mod_text))

    return {
        "total_starts": total,
        # P0-4：total_events 作為 total_starts 的別名，方便 topic 過濾場景
        # 以比較自然的語意判斷「這個 topic 的樣本夠不夠」。
        "total_events": total,
        "topic_filter": topic_filter,
        "modified_ratio": (modified_n / total) if total else 0.0,
        "top_goal_keywords": [kw for kw, _ in goal_kw_counter.most_common(8)],
        "top_constraints": [c for c, _ in constraint_raw_counter.most_common(6)],
        "top_constraint_keywords": [
            kw for kw, _ in constraint_kw_counter.most_common(8)
        ],
        "top_gap_categories": [cat for cat, _ in gap_cat_counter.most_common(6)],
        "top_corrections": [kw for kw, _ in modification_counter.most_common(6)],
        "_raw": {
            "goal_kw": dict(goal_kw_counter),
            "constraint_raw": dict(constraint_raw_counter),
            "constraint_kw": dict(constraint_kw_counter),
            "gap_cat": dict(gap_cat_counter),
            "modifications": dict(modification_counter),
        },
    }


def is_mature(baseline: dict[str, Any], threshold: int = 3) -> bool:
    """規格書要求 3 筆端點後基線開始影響復述。"""
    return baseline.get("total_starts", 0) >= threshold


def high_frequency_fields(
    baseline: dict[str, Any],
    ratio_threshold: float = 0.3,
    top_k: int = 6,
) -> list[str]:
    """回傳『高頻欄位』——提問單會用到。

    條件：在全部 start 端點中出現比例 >= ratio_threshold，
    取 top_k 個最常見者。
    """
    total = baseline.get("total_starts", 0)
    if total == 0:
        return []

    gap_raw = baseline.get("_raw", {}).get("gap_cat", {})
    scored = sorted(gap_raw.items(), key=lambda kv: kv[1], reverse=True)
    result = [
        cat for cat, cnt in scored
        if (cnt / total) >= ratio_threshold
    ]
    return result[:top_k]


def frequently_missed_fields(
    baseline: dict[str, Any],
    ratio_threshold: float = 0.3,
    top_k: int = 6,
) -> list[str]:
    """使用者經常『漏掉』的面向。

    MVP 的近似：被 TOFU 補位的次數 / 全部 start 端點數。
    TOFU 每次都會針對某些面向發問，若這些面向重複出現，
    就代表使用者『習慣性忽略』這些欄位。
    """
    return high_frequency_fields(baseline, ratio_threshold, top_k)


# gap_category 的中文顯示名（用於人類可讀摘要）
_CATEGORY_ZH = {
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


# ----------------------------------------------------------------------
# v0.5：畫像線策略摘要（補位策略配置）
# ----------------------------------------------------------------------
# 依 20260416_端點檢索與元動機偵測_邏輯鏈_v0_5.md 第二塊定義。
#
# 來源：baseline 對 `gap_categories` 的全量統計（掃全量端點，不受 top-k 限制）
# 翻譯：把「自帶率」翻成四級策略
#   自帶率 > 80%  → 使用者自己會想到 → 逗福跳過
#   自帶率 30-80% → 有時會忘 → 相關場景才帶
#   自帶率 < 30%  → 盲區 → 在決策／規劃／購買／行動類場景主動帶
#   自帶率 < 5%   → 重度盲區 → 同上，但優先級更高、提示可更強
#   純資訊查詢不觸發上述主動補位
#
# 自帶率定義（v0.5+ 修正）：
#   自帶率 = 此類別在 gap_categories 中出現的次數 / 總 start 端點數
#
# 設計語意：gap_categories 記錄的是每一輪互動中「使用者實際談到／涉入」的
# 面向標籤。某類別在全量端點中完全沒出現 → 使用者從未涉入過這個面向
# → 視為盲區（自帶率 0%）。某類別頻繁出現 → 使用者經常主動談到
# → 視為強項（自帶率接近 100%）。
#
# 舊實作（v0.5）把公式寫成 `1 - hit/total`，導致「沒出現」被誤判為
# 「自帶率 100% 的強項」——與使用者的心智模型完全相反。本次修正
# 將公式改為 `hit/total`，以「沒出現 = 0% 自帶 = 盲區」為真。
# 例如 total_starts = 10，motivation 出現 9 次 → 使用者自帶率 = 90%
# ----------------------------------------------------------------------

# 策略翻譯門檻（比例）
_STRAT_SKIP = 0.80            # 自帶率 > 80%
_STRAT_CONTEXTUAL = 0.30      # 自帶率 30-80%
_STRAT_HEAVY_BLIND = 0.05     # 自帶率 < 5%

# 行為節奏對應的中文節奏句（由畫像線 decision_style.pace 決定）
_PACE_BRIEF: dict[str, str] = {
    "fast": "此使用者行為節奏：急性子。先給答案再補位，盲區提醒夾在答案裡。",
    "deliberate": "此使用者行為節奏：要求準確。先確認邊界再給答案。",
    "avoidant": "此使用者行為節奏：反覆斟酌。幫他收斂選項，盲區提醒用獨立提問。",
}

STRATEGY_BRIEF_MAX_TOKENS = 100  # 注入量上限，約 3-5 句中文


def _category_self_mention_rate(
    baseline: dict[str, Any],
) -> dict[str, float]:
    """回傳 {gap_category: 自帶率}。

    自帶率 = 此類別在 baseline 統計中出現的次數 / total_starts。

    設計原則（v0.5+ 修正）：
    - 沒出現 → 自帶率 0% → 盲區（使用者從未涉入過這個面向）
    - 每次都出現 → 自帶率 100% → 強項（使用者每次都自己帶到）

    舊版（v0.5）用 `1 - hit/total`，會把「完全沒出現」判成「自帶率
    100% 的強項」，策略摘要誤寫「此使用者強項：預算（自帶率 100%），
    不需要問」。這與實際情況相反——沒出現代表沒證據，應視為盲區。

    total_starts 為 0 時回空 dict。
    """
    total = int(baseline.get("total_starts") or 0)
    if total <= 0:
        return {}
    gap_raw = baseline.get("_raw", {}).get("gap_cat", {}) or {}
    if not isinstance(gap_raw, dict):
        return {}
    rates: dict[str, float] = {}
    for cat in ALLOWED_GAP_CATEGORIES:
        hit = int(gap_raw.get(cat, 0))
        rate = hit / total
        # 夾到 [0, 1]
        if rate < 0:
            rate = 0.0
        if rate > 1:
            rate = 1.0
        rates[cat] = rate
    return rates


def compute_strategy_brief(
    baseline: dict[str, Any],
    user_profile: dict[str, Any] | None = None,
) -> str:
    """把 baseline 全量統計 + 畫像線的 pace 翻成 3-5 句策略摘要。

    輸出設計為 ≈ 80-100 tokens 的中文短段落，注入 system prompt。
    冷啟動（total_starts == 0）時回一句「冷啟動」告訴 LLM 走通用面向。
    """
    total = int(baseline.get("total_starts") or 0)
    if total <= 0:
        return (
            "此使用者尚無基線：目前是冷啟動，使用七個提問模式的預設優先序，"
            "優先補位通用面向（time/budget/venue/stakeholder）。"
        )

    rates = _category_self_mention_rate(baseline)
    if not rates:
        return "此使用者基線尚未成熟，優先補位通用面向（time/budget/venue）。"

    strengths: list[tuple[str, float]] = []   # 自帶率 > 80%
    heavy_blind: list[tuple[str, float]] = [] # 自帶率 < 5%
    blind: list[tuple[str, float]] = []       # 自帶率 < 30%（含 heavy）
    contextual: list[tuple[str, float]] = []  # 30-80%

    for cat, rate in rates.items():
        if rate > _STRAT_SKIP:
            strengths.append((cat, rate))
        elif rate < _STRAT_HEAVY_BLIND:
            heavy_blind.append((cat, rate))
            blind.append((cat, rate))
        elif rate < _STRAT_CONTEXTUAL:
            blind.append((cat, rate))
        else:
            contextual.append((cat, rate))

    # 依自帶率升序（盲區最嚴重的在前）
    heavy_blind.sort(key=lambda kv: kv[1])
    blind.sort(key=lambda kv: kv[1])
    # 強項依自帶率降序（最強的在前）
    strengths.sort(key=lambda kv: kv[1], reverse=True)

    def _fmt(entries: list[tuple[str, float]], max_n: int = 3) -> str:
        items: list[str] = []
        for cat, rate in entries[:max_n]:
            zh = _CATEGORY_ZH.get(cat, cat)
            items.append(f"{zh}（自帶率 {rate * 100:.0f}%）")
        return "、".join(items)

    lines: list[str] = []
    if blind:
        # 盲區優先寫出。v0.5+ 修正：把原本「回答中必須主動包含...，
        # 即使使用者沒問」改成條件觸發——只有在使用者的問題涉及決策、
        # 規劃、購買或行動時才帶出盲區提醒，純粹的資訊查詢或推薦（無
        # 資源投入）不觸發，避免 LLM 在聊天或查詢情境硬塞時間/預算提醒。
        lines.append(
            f"此使用者盲區：{_fmt(blind, max_n=3)}。\n"
            "當使用者的問題涉及決策、規劃、購買或行動時，"
            "回答中主動包含這些維度的提醒。\n"
            "純粹的資訊查詢或推薦（無資源投入）不觸發盲區提醒。"
        )
    if heavy_blind:
        lines.append(
            f"其中重度盲區：{_fmt(heavy_blind, max_n=2)}——不只帶，還要強調。"
        )
    if strengths:
        lines.append(
            f"此使用者強項：{_fmt(strengths, max_n=2)}，不需要問這些面向。"
        )
    if contextual and not blind:
        lines.append(
            f"此使用者中等關注：{_fmt(contextual, max_n=2)}，相關場景才帶。"
        )

    # 節奏句（從畫像線 decision_style.pace 取；沒畫像就預設 deliberate）
    pace = "deliberate"
    if isinstance(user_profile, dict):
        ds = user_profile.get("decision_style") or {}
        if isinstance(ds, dict):
            p = ds.get("pace")
            if isinstance(p, str) and p in _PACE_BRIEF:
                pace = p
    lines.append(_PACE_BRIEF[pace])

    brief = "\n".join(lines)

    # 寬鬆的長度夾持：避免極端情況下衝破 100 tokens（≈ 200 中文字元）
    # 保險上限設 260 字元，超過就截到最後一個換行
    if len(brief) > 260:
        brief = brief[:260].rsplit("\n", 1)[0]
    return brief


def summarize_baseline(baseline: dict[str, Any]) -> str:
    """輸出人類可讀的基線摘要（用於 /baseline 指令）。

    涵蓋：互動次數、高頻目標類型、高頻限制、高頻被補位面向、修正率。
    """
    total = baseline.get("total_starts", 0)
    if total == 0:
        return (
            "目前基線：尚無任何互動紀錄。\n"
            "先輸入幾個需求讓 TOFU 建立你的基線吧。"
        )

    modified_ratio = baseline.get("modified_ratio", 0.0)
    top_goals = baseline.get("top_goal_keywords", [])
    top_constraints = baseline.get("top_constraints", [])
    top_gaps = baseline.get("top_gap_categories", [])

    def _fmt_list(xs: list[str], empty: str = "（尚無資料）") -> str:
        if not xs:
            return empty
        return "、".join(xs)

    gap_display = [_CATEGORY_ZH.get(c, c) for c in top_gaps]

    mature_flag = "已建立" if is_mature(baseline) else "建立中"
    lines = [
        f"目前基線（{mature_flag}）",
        f"- 互動次數：{total}",
        f"- 高頻目標關鍵詞：{_fmt_list(top_goals)}",
        f"- 常見限制條件：{_fmt_list(top_constraints)}",
        f"- 常被補位的面向：{_fmt_list(gap_display)}",
        f"- 修正率：{modified_ratio * 100:.1f}%",
    ]
    return "\n".join(lines)
