"""修法 B：跨題延續偵測（detect_continuation）與整合測試。

- detect_continuation 從嚴：只認明確指代詞，純省略主詞回 False（靠 A 兜）。
- 整合：問 A（auto_confirm）→ 寫端點 → 延續式追問 B →
  encoded_top_k 非空且含 A 的 goal 關鍵詞。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware import confirmation as confirm_mod  # noqa: E402
from src.middleware.endpoint import EndpointStore  # noqa: E402


class DetectContinuationTests(unittest.TestCase):
    def test_marker_with_completed_returns_true(self):
        """測試 6：含「那／這／它」且 completed_count>=1 → True。"""
        for text in ("那線上的呢", "這樣可以嗎", "它的成本呢"):
            self.assertTrue(
                confirm_mod.detect_continuation(text, completed_count=1),
                msg=text,
            )

    def test_pure_ellipsis_returns_false(self):
        """測試 7：純省略「線上的呢」（無指代詞）→ False（從嚴）。"""
        self.assertFalse(
            confirm_mod.detect_continuation("線上的呢", completed_count=3)
        )

    def test_no_completed_returns_false(self):
        """測試 8：completed_count=0 → False。"""
        self.assertFalse(
            confirm_mod.detect_continuation("那線上的呢", completed_count=0)
        )

    def test_empty_input_returns_false(self):
        self.assertFalse(confirm_mod.detect_continuation("", completed_count=2))


class _RecordingClient:
    """記錄 generate_restate 每次呼叫 kwargs 的 mock client。"""

    fallback_mode = False

    def __init__(self) -> None:
        self.restate_calls: list[dict] = []

    def generate_restate(self, user_input, baseline_summary, allowed_categories, **kwargs):
        self.restate_calls.append({"user_input": user_input, **kwargs})
        # auto_confirm 走單輪：第一輪就回非空 gap_questions，
        # 由 auto_confirm 強制收斂（info_sufficient）。
        return {
            "restate_text": f"你想「{user_input}」。",
            "gap_questions": ["細節？"],
            "gap_categories": ["scope"],
            "inferred_goal": user_input,
            "inferred_motivation": "",
            "inferred_constraints": [],
        }

    def merge_confirmation(self, original_input, restate_text, user_confirmation_text):
        return {
            "goal": original_input,
            "motivation": "",
            "constraints": [],
            "answered_categories": ["scope"],
        }

    def execute_task(self, goal, motivation, constraints, user_profile=None):
        return f"針對 {goal}：建議改成線上採直播形式。"


class ContinuationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)
        self.client = _RecordingClient()

    def tearDown(self):
        self.tmp.cleanup()

    def test_followup_recovers_prev_endpoint_and_frames(self):
        """測試 9：問 A → 延續式追問 B → encoded_top_k 非空且含 A 的 goal。"""
        from src.main import run_one_interaction

        # 問 A（auto_confirm）→ 寫端點
        ok_a = run_one_interaction(
            "林口新品牌發表會要注意什麼",
            self.store,
            self.client,
            ask_satisfaction=False,
            auto_confirm=True,
        )
        self.assertTrue(ok_a)
        self.assertEqual(self.store.completed_count(), 1)

        # 延續式追問 B（含指代詞「那」，名詞與 A 幾乎不重疊）
        self.client.restate_calls.clear()
        ok_b = run_one_interaction(
            "那如果改成線上的呢",
            self.store,
            self.client,
            ask_satisfaction=False,
            auto_confirm=True,
        )
        self.assertTrue(ok_b)

        # B 的第一輪 restate 呼叫
        first_call = self.client.restate_calls[0]
        encoded = first_call.get("encoded_top_k") or ""
        # A：近因保底讓上一筆端點出現在 encoded_top_k（資料在）
        self.assertTrue(encoded, "encoded_top_k 不應為空（近因保底失效）")
        self.assertIn("林口", encoded)
        # B：延續框定讓 LLM 知道在延續上一題（明說框定）
        prior = first_call.get("prior_rounds_context") or ""
        self.assertIn("本題延續上一題", prior)
        self.assertIn("林口", prior)


if __name__ == "__main__":
    unittest.main()
