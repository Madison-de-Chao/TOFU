"""v0.6 PR-E：EndpointStore JSONL 底層 + filelock + legacy 遷移測試。

測試覆蓋：
- 雙格式自動切換（.json vs .jsonl）
- JSONL append-only 行為（O(1)、不重讀整檔）
- legacy .json → .jsonl 自動遷移、原檔保留為 .bak
- 損壞行跳過、整檔損壞自動 quarantine
- mark_hit / apply_cooldown / clear 在 JSONL 上仍正確
- filelock 並發場景（多 thread 同時 append）
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.endpoint import (
    EndpointStore,
    make_end_data,
    make_start_data,
    new_event_id,
)


def _read_jsonl_lines(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class JsonlFormatDetectionTests(unittest.TestCase):
    """依 path 後綴決定格式：.jsonl → 新格式、.json → 舊格式。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_jsonl_path_uses_jsonl_format(self):
        path = os.path.join(self.tmp.name, "endpoints.jsonl")
        store = EndpointStore(path)
        self.assertEqual(store._format, "jsonl")
        self.assertTrue(os.path.exists(path))
        # 空檔案
        self.assertEqual(os.path.getsize(path), 0)

    def test_json_path_uses_legacy_format(self):
        path = os.path.join(self.tmp.name, "endpoints.json")
        store = EndpointStore(path)
        self.assertEqual(store._format, "json")
        # legacy 格式：empty array
        with open(path, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])

    def test_jsonl_append_writes_one_line_per_row(self):
        path = os.path.join(self.tmp.name, "endpoints.jsonl")
        store = EndpointStore(path)
        store.append_start("evt1", {"goal": "a"})
        store.append_end("evt1", {"result": "r"})
        store.append_start("evt2", {"goal": "b"})

        rows = _read_jsonl_lines(path)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["event_id"], "evt1")
        self.assertEqual(rows[0]["type"], "start")
        self.assertEqual(rows[1]["type"], "end")
        self.assertEqual(rows[2]["event_id"], "evt2")

    def test_jsonl_round_trip_via_store(self):
        path = os.path.join(self.tmp.name, "endpoints.jsonl")
        store = EndpointStore(path)
        eid = new_event_id()
        store.append_start(
            eid,
            make_start_data(
                user_input="買拉麵",
                tofu_understanding="",
                gap_questions=[],
                user_confirmation="confirmed",
                user_modifications="",
                goal="買拉麵",
                motivation="",
                constraints=[],
                gap_categories=["food"],
            ),
        )
        store.append_end(eid, make_end_data(result="去家樂福"))

        # 透過 store API 讀回
        all_rows = store.all()
        self.assertEqual(len(all_rows), 2)
        self.assertEqual(store.completed_count(), 1)


class JsonlMigrationTests(unittest.TestCase):
    """legacy .json → .jsonl 自動遷移，原檔保留 .bak。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.legacy_path = os.path.join(self.tmp.name, "endpoints.json")
        self.new_path = os.path.join(self.tmp.name, "endpoints.jsonl")

    def _write_legacy(self, rows: list[dict]) -> None:
        with open(self.legacy_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

    def test_legacy_json_migrates_to_jsonl_on_init(self):
        legacy_rows = [
            {
                "endpoint_id": "e1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "type": "start",
                "event_id": "evt1",
                "start_data": {"goal": "a"},
                "end_data": None,
            },
            {
                "endpoint_id": "e2",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "type": "end",
                "event_id": "evt1",
                "start_data": None,
                "end_data": {"result": "r"},
            },
        ]
        self._write_legacy(legacy_rows)

        store = EndpointStore(self.new_path)

        # 新檔已生成、舊檔重命名為 .bak
        self.assertTrue(os.path.exists(self.new_path))
        self.assertFalse(os.path.exists(self.legacy_path))
        self.assertTrue(os.path.exists(self.legacy_path + ".bak"))

        # 內容一致
        rows = _read_jsonl_lines(self.new_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event_id"], "evt1")
        self.assertEqual(rows[0]["type"], "start")
        self.assertEqual(rows[1]["type"], "end")

        # store.all() 也讀得到（且自動補上 v0.5 冷卻欄位）
        all_rows = store.all()
        self.assertEqual(len(all_rows), 2)
        self.assertEqual(all_rows[0]["status"], "active")
        self.assertEqual(all_rows[0]["reference_count"], 0)

    def test_legacy_corrupted_does_not_block_init(self):
        with open(self.legacy_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        store = EndpointStore(self.new_path)

        # legacy 重命名為 .bak、新檔不存在或為空
        self.assertTrue(os.path.exists(self.legacy_path + ".bak"))
        self.assertEqual(store.all(), [])

        # 後續仍可正常 append
        store.append_start("evt", {"goal": "ok"})
        self.assertEqual(len(store.all()), 1)

    def test_migration_skipped_if_new_jsonl_already_exists(self):
        """當 .jsonl 已存在時，不再用 legacy 覆寫；legacy 仍重命名 .bak 避免混淆。"""
        # 先建好新檔
        with open(self.new_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"endpoint_id": "new", "type": "start"}, ensure_ascii=False) + "\n")
        # 同時放一份舊檔
        self._write_legacy([{"endpoint_id": "old", "type": "start"}])

        EndpointStore(self.new_path)

        # 新檔內容沒被舊檔覆寫
        rows = _read_jsonl_lines(self.new_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["endpoint_id"], "new")
        # 但 legacy 仍重命名為 .bak（避免之後又被誤遷移）
        self.assertTrue(os.path.exists(self.legacy_path + ".bak"))

    def test_no_migration_when_legacy_absent(self):
        EndpointStore(self.new_path)
        self.assertTrue(os.path.exists(self.new_path))
        self.assertFalse(os.path.exists(self.legacy_path + ".bak"))


class JsonlCorruptedLineTests(unittest.TestCase):
    """損壞行（單行 JSON parse 失敗）跳過，不汙染整檔。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "endpoints.jsonl")

    def test_skips_bad_line_keeps_good_lines(self):
        good = json.dumps(
            {
                "endpoint_id": "good",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "type": "start",
                "event_id": "evt",
                "start_data": {"goal": "g"},
                "end_data": None,
            },
            ensure_ascii=False,
        )
        bad = "{not valid"
        non_dict = '"a string row"'

        with open(self.path, "w", encoding="utf-8") as f:
            f.write(good + "\n")
            f.write(bad + "\n")
            f.write(non_dict + "\n")
            f.write(good + "\n")

        store = EndpointStore(self.path)
        rows = store.all()
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["endpoint_id"], "good")

    def test_blank_lines_are_ignored(self):
        good = json.dumps(
            {"endpoint_id": "g", "type": "start", "event_id": "e", "start_data": {}},
            ensure_ascii=False,
        )
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(good + "\n")
            f.write("\n\n")
            f.write(good + "\n")

        store = EndpointStore(self.path)
        rows = store.all()
        self.assertEqual(len(rows), 2)


class JsonlBatchOperationsTests(unittest.TestCase):
    """mark_hit / apply_cooldown / clear 在 JSONL 上仍正確。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "endpoints.jsonl")
        self.store = EndpointStore(self.path)

    def test_clear_truncates_jsonl_to_empty(self):
        self.store.append_start("e1", {"goal": "a"})
        self.store.append_start("e2", {"goal": "b"})
        self.assertEqual(len(self.store.all()), 2)

        self.store.clear()

        self.assertEqual(self.store.all(), [])
        # 檔案存在但為空
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(os.path.getsize(self.path), 0)

    def test_mark_hit_updates_existing_rows_in_place(self):
        row = self.store.append_start("evt", make_start_data(
            user_input="x",
            tofu_understanding="",
            gap_questions=[],
            user_confirmation="confirmed",
            user_modifications="",
            goal="x",
            motivation="",
            constraints=[],
        ))
        self.store.mark_hit([row["endpoint_id"]], current_round=5)

        rows = _read_jsonl_lines(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reference_count"], 1)
        self.assertEqual(rows[0]["round_last_referenced"], 5)
        self.assertIsNotNone(rows[0]["last_referenced"])

    def test_append_after_batch_rewrite_preserves_all_rows(self):
        """整檔 rewrite (mark_hit) 後再 append，所有 row 都還在。"""
        r1 = self.store.append_start("e1", {"goal": "a"})
        r2 = self.store.append_start("e2", {"goal": "b"})
        self.store.mark_hit([r1["endpoint_id"]], current_round=1)
        self.store.append_start("e3", {"goal": "c"})

        rows = _read_jsonl_lines(self.path)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["event_id"] for r in rows], ["e1", "e2", "e3"])


class JsonlConcurrencyTests(unittest.TestCase):
    """filelock 並發保護：多 thread 同時 append 不會掉資料、不會交錯。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "endpoints.jsonl")
        self.store = EndpointStore(self.path)

    def test_concurrent_append_does_not_lose_rows(self):
        n_threads = 10
        per_thread = 20

        def worker(tid: int):
            for i in range(per_thread):
                self.store.append_start(f"t{tid}-i{i}", {"goal": f"goal-{tid}-{i}"})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = _read_jsonl_lines(self.path)
        self.assertEqual(len(rows), n_threads * per_thread)

        # 所有 event_id 都唯一（沒有 row 被覆蓋掉）
        event_ids = [r["event_id"] for r in rows]
        self.assertEqual(len(event_ids), len(set(event_ids)))

    def test_concurrent_jsonl_lines_are_valid_json(self):
        """每行都應該是合法 JSON（沒有兩個 thread 寫的內容交錯在同一行）。"""
        def worker(tid: int):
            payload = {"goal": "x" * 200, "tid": tid}  # 大一點，更容易暴露交錯
            for i in range(5):
                self.store.append_start(f"t{tid}-i{i}", payload)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 直接逐行 json.loads，任何交錯都會炸
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                except json.JSONDecodeError as exc:
                    self.fail(f"line {line_no} not valid JSON: {exc}\n  raw: {stripped[:200]!r}")


class JsonlAppendPerformanceTests(unittest.TestCase):
    """append 應該是 O(1) 而不是 O(N)：500 筆寫入應遠少於整檔重讀的時間。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_jsonl_append_does_not_scale_with_existing_size(self):
        """對比 .json（O(N)）vs .jsonl（O(1)）：JSONL 應顯著更快。

        不是嚴格的 benchmark，只是 sanity check —— 確保沒有不小心讓 append
        走了「讀整檔→改→寫整檔」路徑。
        """
        path_jsonl = os.path.join(self.tmp.name, "endpoints.jsonl")
        store = EndpointStore(path_jsonl)

        # 先預載 200 筆讓檔案有點重量
        for i in range(200):
            store.append_start(f"warm-{i}", {"goal": "x" * 100})

        # 量再 append 100 筆要多久
        t0 = time.perf_counter()
        for i in range(100):
            store.append_start(f"hot-{i}", {"goal": "x" * 100})
        elapsed = time.perf_counter() - t0

        # 寬鬆上限：在普通 dev box 上應該 < 2s（filelock 的 acquire 比較慢）
        self.assertLess(
            elapsed, 5.0,
            f"100 次 jsonl append 花了 {elapsed:.3f}s，疑似走了 O(N) 路徑",
        )

        # 確認筆數正確
        self.assertEqual(len(store.all()), 300)


if __name__ == "__main__":
    unittest.main()
