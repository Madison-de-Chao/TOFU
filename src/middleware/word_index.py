"""v0.6 PR-E：詞性分類檔（word index）儲存。

依 Spec v2 §3.3 / §3.4。詞按詞性（noun / verb / adj / time）+ 類別分檔，
每行一個詞條（JSONL）：

    {
      "word": "拉麵",
      "categories": ["食"],
      "endpoint_ids": ["E001", "E045"],
      "last_seen": "2026-04-16T...",
      "count": 3
    }

設計原則：
- 端點本體**不分類**，只存 ``data/endpoints.jsonl`` 一份；
  詞索引建在 ``data/words/`` 底下，依詞性 × 類別分檔。
- 動詞統一存 ``verbs.jsonl``，``categories`` 累積（同一個「看」可同時屬樂、育、工作）；
  名詞依類別分檔（``noun_食.jsonl`` / ``noun_衣.jsonl`` / …）；
  形容詞依屬性分檔（``adj_顏色.jsonl`` / …）；時間詞單檔 ``time.jsonl``。
- upsert 行為：詞已存在 → 累積 ``endpoint_ids``（dedup）、bump ``count``、
  更新 ``last_seen``、累積 ``categories``；詞不存在 → 新增一行。
- 寫入用 filelock + 整檔重寫（每個分類檔通常很小，重寫成本低）。

不包含：Haiku 標記呼叫、jieba 斷詞——那些在 ``word_tagger.py``。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

from src.middleware.endpoint import _acquire_path_lock


# ---------------------------------------------------------------------------
# 詞性分類常數（Spec v2 §3.3）
# ---------------------------------------------------------------------------

NOUN_CATEGORIES: tuple[str, ...] = (
    "食", "衣", "住", "行", "育", "樂", "工作", "個人", "其他",
)

ADJ_CATEGORIES: tuple[str, ...] = (
    "顏色", "溫度", "天氣", "評價", "情緒",
)

# 詞性類型
KIND_NOUN = "noun"
KIND_VERB = "verb"
KIND_ADJ = "adj"
KIND_TIME = "time"

VALID_KINDS: tuple[str, ...] = (KIND_NOUN, KIND_VERB, KIND_ADJ, KIND_TIME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_for(kind: str, category: str | None) -> str:
    """回傳該詞條應該寫入的檔名（不含路徑）。

    - noun/adj：依類別分檔（noun_食.jsonl / adj_顏色.jsonl）
    - verb：統一 verbs.jsonl，category 寫到詞條的 categories 欄位
    - time：統一 time.jsonl，無 category
    """
    if kind == KIND_NOUN:
        if not category or category not in NOUN_CATEGORIES:
            category = "其他"
        return f"noun_{category}.jsonl"
    if kind == KIND_ADJ:
        if not category or category not in ADJ_CATEGORIES:
            # 不認識的形容詞屬性 → 退回「評價」當泛用桶
            category = "評價"
        return f"adj_{category}.jsonl"
    if kind == KIND_VERB:
        return "verbs.jsonl"
    if kind == KIND_TIME:
        return "time.jsonl"
    raise ValueError(f"unknown kind: {kind!r}")


class WordIndexStore:
    """管理 ``data/words/*.jsonl`` 詞性分類檔。

    所有寫入透過 ``upsert_word()``。讀取以分類檔為單位
    （``read_category(kind, category)``）。
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 內部：單一分類檔 read / write
    # ------------------------------------------------------------------
    def _path_for(self, kind: str, category: str | None) -> str:
        return os.path.join(self.base_dir, _filename_for(kind, category))

    def _read_file(self, path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows: list[dict[str, Any]] = []
        bad_lines = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        bad_lines += 1
        except (UnicodeDecodeError, OSError):
            return []
        if bad_lines:
            print(
                f"[逗福Tofu 警告] {os.path.basename(path)} 跳過 "
                f"{bad_lines} 行損壞紀錄。",
                file=sys.stderr,
                flush=True,
            )
        return rows

    def _write_file_nolock(
        self,
        path: str,
        rows: Iterable[dict[str, Any]],
    ) -> None:
        """原子覆寫（無鎖版本）：必須由呼叫方持有 _acquire_path_lock(path) 再呼叫。"""
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        os.replace(tmp_path, path)

    def _write_file_atomic(
        self,
        path: str,
        rows: Iterable[dict[str, Any]],
    ) -> None:
        """原子覆寫：tmp + rename，filelock 保護。供外部（無鎖上下文）直接呼叫。"""
        with _acquire_path_lock(path):
            self._write_file_nolock(path, rows)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def read_category(
        self,
        kind: str,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """讀某個詞性 × 類別分類檔的所有詞條。"""
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        return self._read_file(self._path_for(kind, category))

    def list_categories_for_kind(self, kind: str) -> list[str]:
        """回傳該詞性下實際已存在的類別清單（依檔案存在判斷）。"""
        if kind == KIND_NOUN:
            cats = NOUN_CATEGORIES
            prefix = "noun_"
        elif kind == KIND_ADJ:
            cats = ADJ_CATEGORIES
            prefix = "adj_"
        elif kind == KIND_VERB:
            return ["verbs"] if os.path.exists(self._path_for(kind, None)) else []
        elif kind == KIND_TIME:
            return ["time"] if os.path.exists(self._path_for(kind, None)) else []
        else:
            raise ValueError(f"unknown kind: {kind!r}")

        out: list[str] = []
        for cat in cats:
            path = os.path.join(self.base_dir, f"{prefix}{cat}.jsonl")
            if os.path.exists(path):
                out.append(cat)
        return out

    def upsert_word(
        self,
        word: str,
        kind: str,
        category: str | None,
        endpoint_id: str,
        now: str | None = None,
    ) -> dict[str, Any]:
        """新增或更新一個詞條。

        - 詞已存在：累積 ``endpoint_ids``（dedup）、bump ``count``、
          更新 ``last_seen``、把新 ``category`` 加入 ``categories``（dedup）。
        - 詞不存在：寫一行新詞條。

        對動詞而言，``upsert_word("看", "verb", "樂", "E001")`` 與
        ``upsert_word("看", "verb", "育", "E067")`` 會合併為單一條：
        ``categories=["樂","育"]``、``endpoint_ids=["E001","E067"]``。

        對名詞而言，每個類別有獨立檔案——同一個詞如果出現在兩個類別，
        會在兩個檔案各有一條（這是 spec 設計）。

        回傳寫入後的詞條 dict。
        """
        if not word:
            raise ValueError("word must be non-empty")
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        if not endpoint_id:
            raise ValueError("endpoint_id must be non-empty")

        now_iso = now or _now_iso()
        path = self._path_for(kind, category)

        with _acquire_path_lock(path):
            rows = self._read_file(path)
            target: dict[str, Any] | None = None
            for row in rows:
                if row.get("word") == word:
                    target = row
                    break

            if target is None:
                cats = [category] if category and kind in (KIND_VERB,) else (
                    [category] if category else []
                )
                # noun/adj 已經以類別分檔，categories 仍記一份方便交叉查
                if kind in (KIND_NOUN, KIND_ADJ) and category:
                    cats = [category]
                target = {
                    "word": word,
                    "kind": kind,
                    "categories": cats,
                    "endpoint_ids": [endpoint_id],
                    "last_seen": now_iso,
                    "count": 1,
                }
                rows.append(target)
            else:
                # 累積 categories（dedup，保序）
                if category:
                    cats = list(target.get("categories") or [])
                    if category not in cats:
                        cats.append(category)
                    target["categories"] = cats
                # 累積 endpoint_ids（dedup，保序）
                eids = list(target.get("endpoint_ids") or [])
                if endpoint_id not in eids:
                    eids.append(endpoint_id)
                target["endpoint_ids"] = eids
                target["last_seen"] = now_iso
                target["count"] = int(target.get("count") or 0) + 1
                target.setdefault("kind", kind)

            # 在同一個 lock context 裡寫入（避免巢狀 lock 死鎖）
            self._write_file_nolock(path, rows)
            return dict(target)

    def find_endpoint_ids(
        self,
        kind: str,
        category: str | None,
        words: Iterable[str],
    ) -> list[str]:
        """給一組查詢詞，回傳該分類檔裡命中那些詞的端點 ID 列表（dedup、保序）。

        檢索流程（Spec v2 §3.6 步驟 5-6）的純查表部分。
        """
        rows = self.read_category(kind, category)
        target_words = set(words)
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if row.get("word") in target_words:
                for eid in row.get("endpoint_ids") or []:
                    if eid and eid not in seen:
                        out.append(eid)
                        seen.add(eid)
        return out


__all__ = [
    "NOUN_CATEGORIES",
    "ADJ_CATEGORIES",
    "KIND_NOUN",
    "KIND_VERB",
    "KIND_ADJ",
    "KIND_TIME",
    "VALID_KINDS",
    "WordIndexStore",
]
