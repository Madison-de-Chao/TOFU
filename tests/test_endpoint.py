"""endpoint 模組的單元測試——JSON 讀寫與事件配對。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.endpoint import (
    EndpointStore,
    make_end_data,
    make_start_data,
    new_event_id,
)


class EndpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_store_is_initialized(self):
        store = EndpointStore(self.path)
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])
        self.assertEqual(store.completed_count(), 0)

    def test_start_end_pairing(self):
        store = EndpointStore(self.path)
        event_id = new_event_id()
        start_data = make_start_data(
            user_input="辦派對",
            tofu_understanding="你想辦派對。預算、時間？",
            gap_questions=["預算", "時間"],
            user_confirmation="confirmed",
            user_modifications="",
            goal="辦派對",
            motivation="慶生",
            constraints=["預算一萬"],
            gap_categories=["budget", "time"],
        )
        store.append_start(event_id, start_data)
        self.assertEqual(store.completed_count(), 0)  # 還沒寫 end

        end_data = make_end_data(result="建議先找場地", deviation_from_goal="")
        store.append_end(event_id, end_data)
        self.assertEqual(store.completed_count(), 1)

        pairs = store.completed_events()
        self.assertEqual(len(pairs), 1)
        start_row, end_row = pairs[0]
        self.assertEqual(start_row["event_id"], end_row["event_id"])
        self.assertEqual(start_row["type"], "start")
        self.assertEqual(end_row["type"], "end")

    def test_clear_empties_store_and_keeps_valid_json(self):
        """P1-3：clear() 後 store.all() 回空列表，檔案仍為合法 JSON list。"""
        store = EndpointStore(self.path)
        store.append_start(new_event_id(), {"goal": "a"})
        store.append_end("evt", {"result": "r"})
        self.assertEqual(len(store.all()), 2)

        store.clear()

        self.assertEqual(store.all(), [])
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(data, [])

    def test_start_data_schema(self):
        start_data = make_start_data(
            user_input="a",
            tofu_understanding="b",
            gap_questions=["c"],
            user_confirmation="confirmed",
            user_modifications="",
            goal="g",
            motivation="m",
            constraints=["x"],
            gap_categories=["budget"],
        )
        for key in [
            "user_input",
            "tofu_understanding",
            "gap_questions",
            "user_confirmation",
            "user_modifications",
            "goal",
            "motivation",
            "constraints",
            "gap_categories",
        ]:
            self.assertIn(key, start_data)


if __name__ == "__main__":
    unittest.main()
