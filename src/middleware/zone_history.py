"""v4.0 P0-2：ATL-4 跨輪一致性檢測。

依 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P0-2。

逗福 v0.6 之前每一輪互動是獨立處理的——同一個使用者問類似問題，可能在
不同輪被分到不同的 Zone（CASE_05 對話 20/22/26/35 的根本原因）。

本模組新增第四個 ATL：每次互動寫入 `zone_history.jsonl`，下一次進來時
比對 signature，發現跨輪漂移就把警告寫進 end 端點（不覆寫當前 zone，
警告交給 MOMO 事後分析）。

signature 設計：
- 納入：關鍵名詞前 5 個、meta_motivation 方向（self/other/both）、topic
- 不納入：具體實體名稱（「Nikon D750」與「Sony A7R IV」走同一 signature）

使用方式：
    store = ZoneHistoryStore("data/zone_history.jsonl")
    features = extract_question_features(user_input, meta_motivation, topic)
    signature = compute_question_signature(features)
    result = atl4_consistency_check(signature, proposed_zone, store)
    # 寫入 end_data["atl4_consistency_check"] = result
    store.append({
        "signature": signature,
        "question_features": features,
        "zone_assignment": proposed_zone,
        "endpoint_id": endpoint_id,
        "timestamp": _now_iso(),
        "confidence": 0.85,
    })
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from src.utils.tokenizer import tokenize as _tokenize

try:
    from filelock import FileLock as _FileLock  # type: ignore
    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover
    _FileLock = None  # type: ignore
    _HAS_FILELOCK = False


# 模組內退路鎖：filelock 缺席時用
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _acquire_lock(path: str, timeout: float = 10.0):
    if _HAS_FILELOCK:
        return _FileLock(path + ".lock", timeout=timeout)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(path, threading.Lock())
    return lock


# 具體實體名稱的啟發式過濾——避免「Nikon D750」與「Sony A7R IV」分到不同
# signature。命中任一條件即視為具體實體名稱，不納入 signature。
#
# 注意：`src.utils.tokenizer.tokenize()` 會把英文 token 正規化為小寫
# （見 src/utils/tokenizer.py），因此下列 regex 一律對小寫生效
# （Copilot review r3105635896）。
_BRAND_LIKE_RE = __import__("re").compile(
    r"(?:"
    r"^[a-z][a-z]+\d+[a-z]*$"           # nikon / d750 / a7r4 形式
    r"|^[a-z]{2,}\d+"                    # ibm360 / m365
    r"|^iphone\d+"
    r"|^ps\d+"
    r")"
)

# 純字母品牌名（無數字）——單獨出現時過濾。搭配 `_is_adjacent_brand_token`
# 偵測 `sony a7r4` 這類「品牌 + 型號」組合、避免品牌 token 本身進入 signature。
_PURE_BRAND_TOKENS: set[str] = {
    "nikon", "sony", "canon", "leica", "olympus", "panasonic", "fujifilm",
    "apple", "samsung", "google", "pixel", "iphone", "ipad", "ps",
    "xbox", "playstation", "nintendo", "switch",
}


def _is_specific_entity(token: str) -> bool:
    """判斷一個 token 是不是具體品牌/型號名稱。"""
    if not token:
        return False
    if _BRAND_LIKE_RE.match(token):
        return True
    # 含數字且長度 >= 3 → 很可能是型號（d750、a7r4、m365）
    if any(c.isdigit() for c in token) and len(token) >= 3:
        return True
    return False


def extract_question_features(
    user_input: str,
    meta_motivation_direction: str | None = None,
    topic: str | None = None,
    mode: str = "default",
    top_keywords: int = 5,
) -> dict[str, Any]:
    """從使用者輸入抽取 signature 用的問題特徵。

    - keywords：斷詞後取前 N 個非具體實體、非純品牌名的 token
    - meta_motivation_direction：self / other / both / None
    - topic：confirmation.infer_topic() 推斷的主題
    - mode：default / free / risk / propose / propose_final

    過濾規則（Copilot review r3105635896 強化）：
    1. 具體實體（D750 / A7R4 / IBM360）→ 丟棄
    2. 純品牌 token（nikon / sony / iphone）且鄰接一個具體實體 → 丟棄
       （確保「nikon d750」與「sony a7r4」走同一 signature）
    3. 純品牌 token 不鄰接具體實體 → 仍保留（例如使用者只寫「nikon」時）
    """
    tokens = _tokenize(user_input) if user_input else []

    def _is_adjacent_brand_token(index: int) -> bool:
        """若 token 是已知純品牌名、且下一個 token 是具體實體 → 視為品牌前綴丟棄。"""
        tok = tokens[index]
        if tok not in _PURE_BRAND_TOKENS:
            return False
        if index + 1 >= len(tokens):
            return False
        return _is_specific_entity(tokens[index + 1])

    generic_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        if _is_specific_entity(tok):
            continue
        if _is_adjacent_brand_token(i):
            continue
        generic_tokens.append(tok)

    keywords = generic_tokens[:top_keywords]

    return {
        "keywords": keywords,
        "meta_motivation_direction": meta_motivation_direction or "",
        "topic": topic or "",
        "mode": mode,
    }


def compute_question_signature(features: dict[str, Any]) -> str:
    """依問題特徵計算 16-hex signature。

    signature = first16(sha256( sorted(keywords[:5]) | direction | topic ))

    不納入 mode——同一個問題在 default / free 之間不應被當成不同 signature。
    """
    keywords = sorted((features.get("keywords") or [])[:5])
    direction = features.get("meta_motivation_direction") or ""
    topic = features.get("topic") or ""
    raw = "|".join(keywords + [direction, topic])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ZoneHistoryStore:
    """Append-only JSONL 儲存 signature → zone 的歷史。

    並發安全：每次 append/query 都獲取 filelock（缺席時退回 thread lock）。
    """

    def __init__(self, path: str | Path = "data/zone_history.jsonl"):
        self.path = Path(path)
        self._path_str = str(self.path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        """Append 一筆 zone history 紀錄。"""
        self._ensure_parent()
        line = json.dumps(record, ensure_ascii=False)
        with _acquire_lock(self._path_str):
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _iter_records(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return []
        results: list[dict[str, Any]] = []
        with _acquire_lock(self._path_str):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 壞行跳過，不讓單筆損壞污染整個檢查
                        continue
        return results

    def query(
        self,
        signature: str,
        lookback_days: int = 30,
    ) -> list[dict[str, Any]]:
        """回傳該 signature 在 lookback_days 內的所有紀錄（新 → 舊不保證）。"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        results: list[dict[str, Any]] = []
        for rec in self._iter_records():
            if rec.get("signature") != signature:
                continue
            ts_raw = rec.get("timestamp")
            if ts_raw:
                try:
                    ts = datetime.fromisoformat(ts_raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                except ValueError:
                    pass
            results.append(rec)
        return results


def atl4_consistency_check(
    current_signature: str,
    proposed_zone: str,
    store: ZoneHistoryStore,
    *,
    min_history: int = 3,
    consistency_threshold: float = 0.80,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """ATL-4 跨輪一致性檢測。

    回傳：
    {
        "is_consistent": bool,
        "historical_zone_distribution": {"A": n, "B": n, "C": n, "unknown": n},
        "dominant_zone": str | None,
        "consistency_score": float,
        "history_size": int,
        "warning": str | None,
    }

    邏輯：
    1. 查 signature 的歷史
    2. < min_history 筆 → 不判定（資料不足）
    3. 統計分布，找主導 zone
    4. 主導一致性 >= threshold 且 proposed != dominant → 警告
    5. 主導一致性 < threshold → 歷史本身就不穩，不警告
    """
    history = store.query(current_signature, lookback_days=lookback_days)

    if len(history) < min_history:
        return {
            "is_consistent": True,
            "historical_zone_distribution": {},
            "dominant_zone": None,
            "consistency_score": 0.0,
            "history_size": len(history),
            "warning": None,
        }

    distribution: dict[str, int] = {"A": 0, "B": 0, "C": 0, "unknown": 0}
    for rec in history:
        z = rec.get("zone_assignment", "unknown")
        if z not in distribution:
            distribution[z] = 0
        distribution[z] += 1

    dominant = max(distribution, key=lambda k: distribution[k])
    consistency = distribution[dominant] / len(history) if history else 0.0

    if consistency < consistency_threshold:
        return {
            "is_consistent": True,
            "historical_zone_distribution": distribution,
            "dominant_zone": None,
            "consistency_score": consistency,
            "history_size": len(history),
            "warning": None,
        }

    if proposed_zone != dominant:
        return {
            "is_consistent": False,
            "historical_zone_distribution": distribution,
            "dominant_zone": dominant,
            "consistency_score": consistency,
            "history_size": len(history),
            "warning": (
                f"本次判 {proposed_zone}、過去 {len(history)} 次中 "
                f"{consistency:.0%} 判為 {dominant}。"
                "可能是語境壓力下的分類漂移。"
            ),
        }

    return {
        "is_consistent": True,
        "historical_zone_distribution": distribution,
        "dominant_zone": dominant,
        "consistency_score": consistency,
        "history_size": len(history),
        "warning": None,
    }
