"""v4.0 P0-2：ATL-4 跨輪一致性檢測的單元測試。

對應規格 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md §P0-2 的 12 項
驗收測試清單。
"""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.zone_history import (
    ZoneHistoryStore,
    atl4_consistency_check,
    compute_question_signature,
    extract_question_features,
)


def _make_record(
    signature: str,
    zone: str,
    ts: datetime | None = None,
    endpoint_id: str = "ep_test",
) -> dict:
    return {
        "signature": signature,
        "question_features": {"keywords": [], "meta_motivation_direction": "", "topic": ""},
        "zone_assignment": zone,
        "endpoint_id": endpoint_id,
        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
        "confidence": 0.85,
    }


class ZoneHistoryStoreTests(unittest.TestCase):
    def test_zone_history_append_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            store.append(_make_record("sig1", "A"))
            store.append(_make_record("sig1", "B"))
            store.append(_make_record("sig2", "C"))

            sig1 = store.query("sig1")
            sig2 = store.query("sig2")
            self.assertEqual(len(sig1), 2)
            self.assertEqual(len(sig2), 1)
            self.assertEqual(sig2[0]["zone_assignment"], "C")

    def test_zone_history_lookback_filter(self):
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            old = datetime.now(timezone.utc) - timedelta(days=60)
            recent = datetime.now(timezone.utc) - timedelta(days=5)
            store.append(_make_record("sig1", "A", ts=old))
            store.append(_make_record("sig1", "B", ts=recent))

            within_30 = store.query("sig1", lookback_days=30)
            self.assertEqual(len(within_30), 1)
            self.assertEqual(within_30[0]["zone_assignment"], "B")

            within_90 = store.query("sig1", lookback_days=90)
            self.assertEqual(len(within_90), 2)

    def test_zone_history_concurrent_append(self):
        """模擬多執行緒 concurrent append，檢查所有紀錄都寫入、無交錯毀損。"""
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "zh.jsonl")
            N = 20

            def worker(i: int) -> None:
                store.append(_make_record(f"sig{i % 3}", "A"))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # 檢查：所有紀錄都寫入了（不會少寫也不會毀損）
            total = sum(len(store.query(f"sig{k}")) for k in range(3))
            self.assertEqual(total, N)

    def test_zone_history_handles_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "zh.jsonl"
            store = ZoneHistoryStore(path)
            store.append(_make_record("sig1", "A"))
            # 直接塞壞行到檔案
            with open(path, "a", encoding="utf-8") as f:
                f.write("not-a-json-line\n")
            store.append(_make_record("sig1", "B"))
            res = store.query("sig1")
            self.assertEqual(len(res), 2)

    def test_zone_history_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            store = ZoneHistoryStore(Path(td) / "never_created.jsonl")
            self.assertEqual(store.query("sig1"), [])


class QuestionSignatureTests(unittest.TestCase):
    def test_signature_identical_for_same_features(self):
        f1 = {
            "keywords": ["舞蹈", "教室", "推薦"],
            "meta_motivation_direction": "self",
            "topic": "樂",
        }
        f2 = {
            "keywords": ["教室", "推薦", "舞蹈"],  # 不同順序
            "meta_motivation_direction": "self",
            "topic": "樂",
        }
        self.assertEqual(
            compute_question_signature(f1),
            compute_question_signature(f2),
        )

    def test_signature_different_for_different_topics(self):
        f1 = {
            "keywords": ["舞蹈", "教室"],
            "meta_motivation_direction": "self",
            "topic": "樂",
        }
        f2 = {
            "keywords": ["舞蹈", "教室"],
            "meta_motivation_direction": "self",
            "topic": "學習",
        }
        self.assertNotEqual(
            compute_question_signature(f1),
            compute_question_signature(f2),
        )

    def test_extract_features_skips_brand_model_tokens(self):
        # 輸入同時含「相機」跟「D750」——只 D750 該被過濾
        features = extract_question_features(
            "請推薦適合初學者的相機 D750 或 A7R4",
            meta_motivation_direction="self",
            topic="creative",
        )
        joined = " ".join(features["keywords"])
        self.assertNotIn("D750", joined)

    def test_extract_features_keywords_limited_to_5(self):
        features = extract_question_features(
            "預算 時間 地點 人數 目的 其他 備註",
            meta_motivation_direction="",
            topic="",
        )
        self.assertLessEqual(len(features["keywords"]), 5)


class Atl4ConsistencyCheckTests(unittest.TestCase):
    def _store_with(self, zones: list[str]) -> ZoneHistoryStore:
        td = tempfile.mkdtemp()
        store = ZoneHistoryStore(Path(td) / "zh.jsonl")
        for z in zones:
            store.append(_make_record("sig1", z))
        return store

    def test_insufficient_history_no_check(self):
        store = self._store_with(["A", "A"])  # < 3 筆
        result = atl4_consistency_check("sig1", "C", store)
        self.assertTrue(result["is_consistent"])
        self.assertIsNone(result["dominant_zone"])
        self.assertEqual(result["history_size"], 2)
        self.assertIsNone(result["warning"])

    def test_consistent_case_no_warning(self):
        store = self._store_with(["C", "C", "C", "C"])
        result = atl4_consistency_check("sig1", "C", store)
        self.assertTrue(result["is_consistent"])
        self.assertEqual(result["dominant_zone"], "C")
        self.assertIsNone(result["warning"])

    def test_inconsistent_case_with_warning(self):
        # 4 筆 C、1 次判 A → 主導 C 一致性 80%，當前 A 不同 → 警告
        store = self._store_with(["C", "C", "C", "C"])
        result = atl4_consistency_check("sig1", "A", store)
        self.assertFalse(result["is_consistent"])
        self.assertEqual(result["dominant_zone"], "C")
        self.assertIsNotNone(result["warning"])
        self.assertIn("A", result["warning"])
        self.assertIn("C", result["warning"])

    def test_unstable_history_no_false_alarm(self):
        # 分布均勻 → 主導一致性 < 80% → 不警告
        store = self._store_with(["A", "B", "C"])
        result = atl4_consistency_check("sig1", "A", store)
        self.assertTrue(result["is_consistent"])
        self.assertIsNone(result["dominant_zone"])
        self.assertIsNone(result["warning"])

    def test_warning_does_not_override_current_zone(self):
        """ATL-4 只記錄警告，不改變當前 zone——回傳值中本來就不帶 new_zone。"""
        store = self._store_with(["C", "C", "C"])
        result = atl4_consistency_check("sig1", "A", store)
        self.assertFalse(result["is_consistent"])
        # 回傳結構沒有任何 zone 覆寫欄位
        self.assertNotIn("override_zone", result)

    def test_consistency_threshold_honored(self):
        # 3 筆 A、1 筆 B → A 一致性 75%，低於預設 80% → 不警告
        store = self._store_with(["A", "A", "A", "B"])
        result = atl4_consistency_check("sig1", "C", store)
        self.assertTrue(result["is_consistent"])
        self.assertIsNone(result["warning"])


if __name__ == "__main__":
    unittest.main()
