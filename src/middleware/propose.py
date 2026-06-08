"""`/propose` 模式的純程式碼工具：meta 區塊解析 + 歷史格式化。

依 `docs/specs/逗福_v0_6_開發_Spec_v2.md` §1.5。

/propose 流程要解析 LLM 每輪輸出結尾的 `[PROPOSE_META]` JSON 區塊，判斷
`submit_now` 訊號以決定是否提前進第 6 輪（交卷）。解析器要**寬鬆但安全**：
- 寬鬆：可以容忍 `[PROPOSE_META]` 後有 code fence、空白、其他文字
- 安全：欄位型別不對（submit_now 不是 bool 等）一律視為「沒找到」，
  caller 會 fallback 成 `submit_now=false`，繼續下一輪

本模組不呼叫任何 LLM，全部由程式碼完成；測試不需要網路或 API key。
"""

from __future__ import annotations

import json
import re
from typing import Any


# 公開給測試用：同時允許 `[PROPOSE_META]` 與小寫變體，避免 LLM 偶爾把標記寫錯
_META_MARKERS = ("[PROPOSE_META]", "[propose_meta]", "[Propose_Meta]")


def parse_propose_meta(text: str) -> dict[str, Any] | None:
    """從 LLM 的 /propose 輪輸出抓 `[PROPOSE_META]` JSON 區塊。

    期望格式（allow variations）::

        本輪焦點：...
        缺的關鍵問題：...

        [PROPOSE_META]
        {"submit_now": false, "round": 2, "reason": "need budget"}

    或者 JSON 在 markdown code fence 裡也接受::

        [PROPOSE_META]
        ```json
        {"submit_now": true, "round": 3, "reason": "enough info"}
        ```

    回傳結構（成功）::

        {"submit_now": bool, "round": int, "reason": str}

    失敗條件（任一觸發即回傳 ``None``）：

    - 找不到 `[PROPOSE_META]` 標記
    - 標記後找不到完整的 ``{...}`` JSON 物件（括號不平衡、被截斷）
    - JSON 無法解析（語法錯誤）
    - 缺 ``submit_now``、``round``、``reason`` 任一欄位
    - ``submit_now`` 不是 bool、``round`` 不是 int、``reason`` 不是 str

    caller 預期拿到 ``None`` 時 fallback 成 ``submit_now=false``（繼續下一輪），
    不要把解析失敗當成錯誤中斷整個 propose session。
    """
    if not text:
        return None

    # 找第一個 marker 位置
    marker_idx = -1
    for marker in _META_MARKERS:
        idx = text.find(marker)
        if idx >= 0 and (marker_idx < 0 or idx < marker_idx):
            marker_idx = idx
    if marker_idx < 0:
        return None

    # 從 marker 之後找第一個平衡的 {...}
    after = text[marker_idx:]
    json_str = _extract_first_json_object(after)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    submit_now = data.get("submit_now")
    round_num = data.get("round")
    reason = data.get("reason")

    if not isinstance(submit_now, bool):
        return None
    if not isinstance(round_num, int) or isinstance(round_num, bool):
        # isinstance(True, int) 是 True，要額外排除
        return None
    if not isinstance(reason, str):
        return None

    return {
        "submit_now": submit_now,
        "round": round_num,
        "reason": reason,
    }


def _extract_first_json_object(text: str) -> str | None:
    """以平衡括號匹配抓出 ``text`` 中第一個完整的 ``{...}`` 子字串。

    考慮字串內的 `{` `}`——遇到雙引號時忽略括號計數，直到字串結束。
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _strip_meta_block(text: str) -> str:
    """移除文字中的 `[PROPOSE_META]` 區塊（含之後的 JSON），回傳主體文字。"""
    marker_idx = -1
    for marker in _META_MARKERS:
        idx = text.find(marker)
        if idx >= 0 and (marker_idx < 0 or idx < marker_idx):
            marker_idx = idx
    if marker_idx >= 0:
        return text[:marker_idx].rstrip()
    return text


# 擷取「本輪焦點」和「暫定假設」的 regex。允許 markdown 粗體 / 編號 / 全形冒號。
# 粗體：`**本輪焦點**`、`**本輪焦點：**`
# 編號：`1. 本輪焦點`、`1) **本輪焦點**`
# 前綴：`### 本輪焦點`、`- 本輪焦點`
_FOCUS_RE = re.compile(
    r"(?:^|\n)[\s#\-*>\d.)）]*\**\s*本輪焦點\**\s*[:：]?\s*\**\s*([^\n]+)",
    re.MULTILINE,
)
_ASSUMPTION_RE = re.compile(
    r"(?:^|\n)[\s#\-*>\d.)）]*\**\s*暫定假設\**\s*[:：]?\s*\**\s*([^\n]+)",
    re.MULTILINE,
)


def _extract_round_summary(text: str) -> dict[str, Any]:
    """從一輪 /propose LLM 輸出中抽取「焦點 + 暫定假設 bullet」。

    - ``focus``：本輪焦點的一句話（若抓不到回空字串）
    - ``assumptions``：所有「暫定假設：...」的清單（若無回空 list）

    parser **不嚴格**——抓不到就 fallback 成 `focus=""` / `assumptions=[]`，
    由 caller 決定是否用截斷原文補位。
    """
    body = _strip_meta_block(text or "")
    focus = ""
    assumptions: list[str] = []

    match = _FOCUS_RE.search(body)
    if match:
        candidate = match.group(1).strip()
        # 去除左右殘留的 markdown 粗體星號
        candidate = candidate.strip("*").strip()
        focus = candidate

    for m in _ASSUMPTION_RE.finditer(body):
        assumption = m.group(1).strip().strip("*").strip()
        if assumption and assumption not in assumptions:
            assumptions.append(assumption)

    return {"focus": focus, "assumptions": assumptions}


# 當 parser 抓不到焦點 / 假設時，退回顯示原文的前 N 字以維持最低資訊量。
# 長度選 200 字：夠保留 1-2 個句子、又不會讓 5 輪累積變成 1k+ tokens。
_FALLBACK_EXCERPT_LEN = 200


def format_propose_history(rounds: list[dict[str, Any]]) -> str:
    """把 /propose 前幾輪的紀錄格式化成 execute_task 的 ``extra_context``。

    v0.6 PR-G：**壓縮版**。不再傳每輪的原文，只傳「本輪焦點一句 + 暫定假設
    bullet」。原因：5 輪 × 全文 → Haiku 注意力稀釋、容易重複問同樣的問題。
    原文仍透過 `store.append_black_box()` 存進端點，事後可追溯。

    ``rounds`` 是 dict list，每筆必須含：

    - ``round``：輪次編號（1 起算）
    - ``text``：該輪 LLM 輸出的完整文字（用來抽取焦點 + 假設）
    - ``meta`` (optional)：解析出的 meta 物件（可為空 dict）

    回傳結構（每輪一段）::

        ### 第 N 輪
        - 焦點：<一句話>
        - 暫定假設：
          - <假設 1>
          - <假設 2>

    Parser 抓不到焦點時，退回顯示原文前 200 字（避免完全失去資訊）。

    空 list 回傳空字串（由 caller 決定是否送 extra_context=None）。
    """
    if not rounds:
        return ""
    parts: list[str] = []
    for row in rounds:
        round_num = row.get("round")
        body = (row.get("text") or "").strip()
        if not body:
            continue
        summary = _extract_round_summary(body)
        focus = summary["focus"]
        assumptions = summary["assumptions"]

        # 若焦點和假設都抓不到 → 回退成截斷的原文摘要（顯示前 200 字，
        # 並標註「parser 未抓到結構」，讓下一輪 Haiku 知道格式不標準）
        if not focus and not assumptions:
            stripped = _strip_meta_block(body)
            excerpt = stripped[:_FALLBACK_EXCERPT_LEN].rstrip()
            if len(stripped) > _FALLBACK_EXCERPT_LEN:
                excerpt += "…"
            parts.append(
                f"### 第 {round_num} 輪（未解析到結構，僅摘要）\n\n"
                f"{excerpt}"
            )
            continue

        lines = [f"### 第 {round_num} 輪", f"- 焦點：{focus or '（未抓到）'}"]
        if assumptions:
            lines.append("- 暫定假設：")
            lines.extend(f"  - {a}" for a in assumptions)
        else:
            lines.append("- 暫定假設：（本輪未提出）")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


__all__ = [
    "parse_propose_meta",
    "format_propose_history",
]
