"""v0.6 PR-E：Haiku 詞性標記器 + WordTaggerGuardrail 成本護欄。

依 Spec v2 §3.5 / §3.11 / §3.12，加上 MOMO 的三條 fallback 規則：

1. **即時寫入時 Haiku 失敗** → 端點本體照寫，詞索引留空（tagged_at: null），
   不阻塞使用者當前互動。
2. **Haiku 格式錯誤**（不符合 [類別] 格式）→ 重試一次；仍錯則當空。
   不做更複雜的容錯。
3. **背景補標記 job（可選，不進 PR-E）**。PR-E 只留門，不實作。

## 架構

```
WordTaggerGuardrail ── $1/day 成本護欄（UTC 日期 reset）
     │
     └─ HaikuWordTagger ── jieba 斷詞 → Haiku API → 解析 [類別] 標記
          │
          └─ tag_and_index_endpoint() ── 協調：取文字 → 分詞 → 標記 → 寫 WordIndexStore
```

## 成本估算

Haiku 4.5：input ≈ $0.8/M tokens；output ≈ $4/M tokens。
一次標記 ~200 tokens in + ~200 tokens out ≈ $0.0008 / 次。
$1/day 護欄 ≈ 1,250 次 / 天，正常使用綽綽有餘。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

try:
    import jieba  # type: ignore

    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

try:
    from anthropic import Anthropic  # type: ignore
    from anthropic import APIError, APITimeoutError  # type: ignore

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    _ANTHROPIC_AVAILABLE = False

from src.middleware.word_index import (
    KIND_ADJ,
    KIND_NOUN,
    KIND_TIME,
    KIND_VERB,
    NOUN_CATEGORIES,
    ADJ_CATEGORIES,
    WordIndexStore,
)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Haiku 的 input + output 成本估算（$/token）——用來更新護欄累計。
# 2026-04 Haiku 4.5 定價：input $0.8/M, output $4/M。
_COST_PER_INPUT_TOKEN = 0.8 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 4.0 / 1_000_000

# 護欄 JSON 狀態檔的預設名稱（存在 base_dir 旁邊）
_GUARDRAIL_FILENAME = "_guardrail.json"

# Haiku 標記的 system prompt（依 Spec v2 §3.5）
_TAGGER_SYSTEM_PROMPT = """\
你是詞性分類機器人。只做一件事：依照下面的規則，在詞串的實質詞彙後面加 [類別] 標記。

規則：
1. 實質名詞加 topic 標記，類別從以下選一：食、衣、住、行、育、樂、工作、個人、其他
   例：拉麵[食]、電影[樂]、台北[行]、書[育]、會議[工作]

2. 指向性動詞加 topic 標記（類別同名詞）
   例：吃[食]、看[樂]、買[衣]、去[行]

3. 動詞+補語複合詞：只對動詞部分標記，補語保留原樣
   例：吃完 → 吃[食] 完；買了 → 買[衣] 了

4. 口語時間詞還原並加 [時間]
   例：昨兒個[時間]、前陣子[時間]、這兩天[時間]

5. 形容詞加屬性標記，類別從以下選一：顏色、溫度、天氣、評價、情緒
   例：紅色[顏色]、熱[溫度]、好吃[評價]、開心[情緒]

6. 其他詞（代詞、助詞、連詞、一般非指向性動詞、虛詞）保持原樣，不加標記。

輸出格式：用「 / 」分隔，詞和標記之間沒有空格，例如：
  輸入：我 / 昨天 / 吃完 / 拉麵 / 覺得 / 好吃 / 去 / 看 / 電影
  輸出：我 / 昨天[時間] / 吃[食] 完 / 拉麵[食] / 覺得 / 好吃[評價] / 去[行] / 看[樂] / 電影[樂]

不要解釋、不要加其他文字，直接輸出標記後的詞串。\
"""


def _now_utc_date() -> str:
    """回傳 UTC 今天的 yyyy-mm-dd 字串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# WordTaggerGuardrail
# ---------------------------------------------------------------------------


class WordTaggerGuardrail:
    """$1/day 成本護欄，UTC 日期 reset。

    狀態持久化到 ``<words_dir>/_guardrail.json``。每次標記完成後
    呼叫 ``record_cost(cost_usd)`` 累積今天的花費。
    ``check_before_tag()`` 在護欄超限時回傳 False，呼叫方應跳過標記。
    """

    def __init__(
        self,
        state_file: str,
        max_cost_per_day: float = 1.0,
    ) -> None:
        self.state_file = state_file
        self.max_cost_per_day = max_cost_per_day
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._state = {}

    def _save(self) -> None:
        tmp_path = self.state_file + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False)
            os.replace(tmp_path, self.state_file)
        except OSError as exc:  # pragma: no cover - 防禦
            print(
                f"[逗福Tofu 警告] WordTaggerGuardrail 狀態存檔失敗：{exc}",
                file=sys.stderr,
                flush=True,
            )

    def _today_cost(self) -> float:
        """今天（UTC）累計的成本。若日期改變，重置為 0。"""
        today = _now_utc_date()
        if self._state.get("date") != today:
            self._state = {"date": today, "cost_usd": 0.0}
            self._save()
        return float(self._state.get("cost_usd") or 0.0)

    def check_before_tag(self) -> bool:
        """呼叫標記前先確認護欄。超限回傳 False，呼叫方應跳過。"""
        return self._today_cost() < self.max_cost_per_day

    def record_cost(self, cost_usd: float) -> None:
        """記錄一次標記的花費。"""
        today = _now_utc_date()
        if self._state.get("date") != today:
            self._state = {"date": today, "cost_usd": 0.0}
        self._state["cost_usd"] = float(self._state.get("cost_usd") or 0.0) + max(0.0, cost_usd)
        self._save()

    @property
    def today_cost_usd(self) -> float:
        return self._today_cost()


# ---------------------------------------------------------------------------
# 解析 Haiku 回傳的標記詞串
# ---------------------------------------------------------------------------

# 合法的類別集合
_VALID_NOUN_CATS = set(NOUN_CATEGORIES)
_VALID_ADJ_CATS = set(ADJ_CATEGORIES)
_ALL_VALID_CATS = _VALID_NOUN_CATS | _VALID_ADJ_CATS | {"時間"}

# 預定義動詞類別：動詞的 topic 類別和名詞相同，
# 但時間詞是 kind=time 而不是 kind=verb
_VERB_NOUN_CATS = _VALID_NOUN_CATS


def _parse_tagger_output(
    output: str,
) -> list[dict[str, str]]:
    """解析 Haiku 回傳的標記詞串，回傳 [{word, kind, category}] 列表。

    只回傳有標記的詞。未加標記的詞（代詞、助詞等）不列入。
    無法解析的標記 → 跳過（不引發例外）。

    格式範例：
        我 / 昨天[時間] / 吃[食] 完 / 拉麵[食] / 好吃[評價]

    回傳格式：
        [
          {"word": "昨天", "kind": "time",   "category": "時間"},
          {"word": "吃",   "kind": "verb",   "category": "食"},
          {"word": "拉麵", "kind": "noun",   "category": "食"},
          {"word": "好吃", "kind": "adj",    "category": "評價"},
        ]
    """
    import re

    results: list[dict[str, str]] = []
    seen: set[str] = set()

    # 把詞串分成 token（依 / 切）
    # 每個 token 可能是 "詞[類別]" 或 "動詞[類別] 補語" 等
    tokens = [t.strip() for t in output.split("/") if t.strip()]

    # 逐 token 找 [...] 標記
    bracket_pat = re.compile(r"^(.+?)\[([^\]]+)\]")
    for token in tokens:
        m = bracket_pat.match(token)
        if not m:
            continue
        word_raw = m.group(1).strip()
        cat = m.group(2).strip()

        if not word_raw or not cat:
            continue
        if cat not in _ALL_VALID_CATS:
            # 不認識的類別 → 跳過
            continue

        # 決定 kind
        if cat == "時間":
            kind = KIND_TIME
        elif cat in _VALID_ADJ_CATS:
            kind = KIND_ADJ
        elif cat in _VALID_NOUN_CATS:
            # 可能是名詞也可能是動詞——如果 token 是「動詞[食] 補語」形式
            # 通常「動詞」部分會是單字動詞，但我們沒有詞典判斷，
            # 用 jieba POS 代價太高；簡化為：只有一個漢字的詞視為動詞
            if len(word_raw) == 1 and word_raw[0] >= "\u4e00":
                # 單字漢字 → 動詞（「吃」「看」「買」「去」）
                kind = KIND_VERB
            else:
                kind = KIND_NOUN
        else:
            continue

        # 去重（同一個詞在同一份輸出只取第一次）
        key = f"{word_raw}:{cat}"
        if key in seen:
            continue
        seen.add(key)

        results.append({"word": word_raw, "kind": kind, "category": cat})

    return results


# ---------------------------------------------------------------------------
# HaikuWordTagger
# ---------------------------------------------------------------------------


class HaikuWordTagger:
    """呼叫 Haiku 4.5 做詞性標記。

    - ``tag(text)``：jieba 斷詞 → 送 Haiku → 解析 → 回傳標記列表。
    - 格式錯誤時重試一次（Rule 2）。
    - API 失敗或重試後仍格式錯誤 → 回傳 None（Rule 1：呼叫方寫 tagged_at:null）。
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("CLAUDE_API_KEY")
        self._client = None
        self._offline = True

        if key and _ANTHROPIC_AVAILABLE and Anthropic is not None:
            try:
                self._client = Anthropic(api_key=key)
                self._offline = False
            except Exception:  # pragma: no cover - 防禦
                self._offline = True

    @property
    def offline(self) -> bool:
        """True 代表沒有 API client（fallback 模式）。"""
        return self._offline

    def _jieba_tokenize(self, text: str) -> list[str]:
        """用 jieba 斷詞；jieba 未安裝時退回空白切割。"""
        if _JIEBA_AVAILABLE:
            return list(jieba.cut(text, cut_all=False))
        # 退回空格切割（ASCII 文字的粗略分詞）
        return text.split()

    def _call_haiku(self, token_string: str) -> str | None:
        """呼叫 Haiku 一次，回傳原始 text；失敗回 None。"""
        if self._client is None:
            return None
        try:
            resp = self._client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=512,
                system=_TAGGER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": token_string}],
            )
            return resp.content[0].text.strip() if resp.content else None
        except (APIError, APITimeoutError, Exception):
            return None

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * _COST_PER_INPUT_TOKEN
            + output_tokens * _COST_PER_OUTPUT_TOKEN
        )

    def tag(
        self, text: str
    ) -> list[dict[str, str]] | None:
        """主要入口：給一段文字，回傳 [{word, kind, category}] 列表。

        Fallback 規則（MOMO Rule 1-2）：
        - Rule 1：API 呼叫失敗 → 回 None（呼叫方應寫 tagged_at:null）。
        - Rule 2：格式錯誤（parse 出 0 個詞）→ 重試一次；仍 0 → 回 None。

        回傳空列表（[]）表示文字合法但沒有可標記的詞（純代詞/虛詞）。
        回傳 None 表示標記過程失敗，呼叫方應跳過。
        """
        if not text or not text.strip():
            return []

        if self._offline:
            return None  # 離線模式 → 當 API 失敗處理

        tokens = self._jieba_tokenize(text)
        if not tokens:
            return []

        token_string = " / ".join(t for t in tokens if t.strip())
        if not token_string:
            return []

        # 第一次呼叫
        raw = self._call_haiku(token_string)
        if raw is None:
            return None  # Rule 1：API 失敗

        parsed = _parse_tagger_output(raw)

        # Rule 2：格式錯誤（0 個有效標記但輸入詞串不為空）→ 重試一次
        if not parsed and len(tokens) > 1:
            raw2 = self._call_haiku(token_string)
            if raw2 is None:
                return None  # Rule 1 on retry
            parsed = _parse_tagger_output(raw2)
            if not parsed:
                return None  # Rule 2：重試仍 0 → 當空

        return parsed

    def estimate_cost_for_text(self, text: str) -> float:
        """粗估一次標記的成本（用於護欄 check，不需要真實 token 計數）。"""
        # 估算：jieba 斷詞約 1-2 tokens/詞，回應約等長
        tokens = len(self._jieba_tokenize(text)) if text else 0
        est_input = len(_TAGGER_SYSTEM_PROMPT.split()) + tokens * 2
        est_output = tokens * 2
        return self._estimate_cost(est_input, est_output)


# ---------------------------------------------------------------------------
# tag_and_index_endpoint：整合協調
# ---------------------------------------------------------------------------


def tag_and_index_endpoint(
    endpoint_id: str,
    text: str,
    tagger: HaikuWordTagger,
    guardrail: WordTaggerGuardrail,
    word_store: WordIndexStore,
    now: str | None = None,
) -> dict[str, Any]:
    """標記一筆端點的文字並寫入詞索引。

    呼叫方（通常是 main.py 的 append_end 流程）提供：
    - endpoint_id：端點的 UUID（用來做詞條的 endpoint_ids 引用）
    - text：user_input + '\\n' + result 拼接的文字

    回傳：
    - {"tagged": True,  "count": n} — 成功，寫了 n 個有標記的詞
    - {"tagged": False, "reason": "..."} — 跳過或失敗

    tagged_at 欄位由呼叫方自行更新端點的 row（endpoint.py 不在本函式裡）。
    """
    if not text or not text.strip():
        return {"tagged": False, "reason": "empty_text"}

    if not guardrail.check_before_tag():
        return {"tagged": False, "reason": "cost_limit_reached"}

    tagged_words = tagger.tag(text)
    if tagged_words is None:
        return {"tagged": False, "reason": "tagger_failed"}

    now_iso = now or _now_iso()
    count = 0
    for entry in tagged_words:
        word = entry.get("word", "")
        kind = entry.get("kind", "")
        category = entry.get("category", "")
        if not word or not kind:
            continue
        try:
            word_store.upsert_word(
                word=word,
                kind=kind,
                category=category or None,
                endpoint_id=endpoint_id,
                now=now_iso,
            )
            count += 1
        except Exception as exc:  # pragma: no cover - 防禦
            print(
                f"[逗福Tofu 警告] 詞條寫入失敗（{word}/{kind}/{category}）：{exc}",
                file=sys.stderr,
                flush=True,
            )

    # 記錄估算成本
    estimated = tagger.estimate_cost_for_text(text)
    guardrail.record_cost(estimated)

    return {"tagged": True, "count": count}


__all__ = [
    "HaikuWordTagger",
    "WordTaggerGuardrail",
    "tag_and_index_endpoint",
]
