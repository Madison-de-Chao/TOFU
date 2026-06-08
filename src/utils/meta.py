"""逗福Tofu meta 問題關鍵詞的單一來源。

Meta 問題：使用者在問逗福Tofu 本身（「你是什麼」「你跟別的 AI 有什麼不同」），
而不是要辦一件事。這種輸入不應該走復述補位流程（會問他預算和時間），
應該直接用身份資訊回答。

偵測邏輯放在程式碼層（關鍵詞比對），由 `main.py` 與 `claude_client.py` 共用，
確保兩處不會因為清單漂移而失去攔截。

v2.0：加入四個比較型關鍵詞（你跟其他 / 你跟別的 / 你的差別 / 你的不同），
涵蓋「你跟別的 AI 有什麼不同」這類提問。
"""

from __future__ import annotations


META_KEYWORDS: tuple[str, ...] = (
    "你是什麼", "你是誰", "你能做什麼", "你幹嘛的", "你是幹嘛",
    "怎麼用", "如何使用", "你做什麼", "你的功能", "介紹一下",
    "what are you", "what do you do", "how to use", "who are you",
    "你跟其他", "你跟別的", "你的差別", "你的不同",
)


def is_meta_query(text: str) -> bool:
    """偵測使用者是否在問逗福Tofu 本身。"""
    lower = (text or "").strip().lower()
    if not lower:
        return False
    return any(kw in lower for kw in META_KEYWORDS)
