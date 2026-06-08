"""修法 A：retrieve_top_k_endpoints 近因保底（recent_floor_n）測試。

- recent_floor_n 預設 0 時行為與現狀逐筆相同（保護既有測試）。
- recent_floor_n>0 時，最近 N 筆 start 端點無條件進入 top_k，
  補住延續式追問（低名詞重疊）的漏撈。
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.endpoint import (  # noqa: E402
    _ensure_cooldown_fields,
    make_start_data,
    new_endpoint_id,
    new_event_id,
    retrieve_top_k_endpoints,
)


def _mk_start(goal: str, gap_categories=None) -> dict:
    sd = make_start_data(
        user_input=goal,
        tofu_understanding="",
        gap_questions=[],
        user_confirmation="confirmed",
        user_modifications="",
        goal=goal,
        motivation="",
        constraints=[],
        gap_categories=list(gap_categories or []),
    )
    row = {
        "endpoint_id": new_endpoint_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type": "start",
        "event_id": new_event_id(),
        "start_data": sd,
        "end_data": None,
    }
    _ensure_cooldown_fields(row)
    return row


class RecentFloorTests(unittest.TestCase):
    def test_default_zero_identical_to_current(self):
        """測試 1：recent_floor_n=0（或不帶參數）→ 結果與現狀逐筆相同。"""
        rows = [
            _mk_start("辦派對 30 人 預算 1 萬", gap_categories=["budget"]),
            _mk_start("學攝影 買相機", gap_categories=["budget"]),
            _mk_start("寫報告 交期", gap_categories=["deadline"]),
        ]
        no_param, no_param_known = retrieve_top_k_endpoints(
            rows, "辦派對 預算 一萬", top_k=3,
        )
        explicit_zero, explicit_known = retrieve_top_k_endpoints(
            rows, "辦派對 預算 一萬", top_k=3, recent_floor_n=0,
        )
        self.assertEqual(
            [r["endpoint_id"] for r in no_param],
            [r["endpoint_id"] for r in explicit_zero],
        )
        self.assertEqual(no_param_known, explicit_known)

    def test_low_overlap_continuation_recovered(self):
        """測試 2：低重疊延續 query + recent_floor_n=2 → 撈回最近端點。"""
        rows = [
            _mk_start("林口新品牌發表會要注意什麼", gap_categories=["scope"]),
        ]
        # 延續式追問：完全不含名詞重疊
        baseline, _ = retrieve_top_k_endpoints(rows, "那如果改成線上的呢", top_k=5)
        self.assertEqual(baseline, [])  # 現狀撈不回（重現根因）

        floored, _ = retrieve_top_k_endpoints(
            rows, "那如果改成線上的呢", top_k=5, recent_floor_n=2,
        )
        self.assertEqual(len(floored), 1)
        self.assertIn("林口", floored[0]["start_data"]["goal"])

    def test_superseded_recent_not_floored(self):
        """測試 3：superseded 的近端點不被補入。"""
        a = _mk_start("舊題目 A", gap_categories=[])
        b = _mk_start("舊題目 B", gap_categories=[])
        b["status"] = "superseded"
        rows = [a, b]
        floored, _ = retrieve_top_k_endpoints(
            rows, "完全無關的延續追問", top_k=5, recent_floor_n=2,
        )
        ids = {r["endpoint_id"] for r in floored}
        self.assertIn(a["endpoint_id"], ids)
        self.assertNotIn(b["endpoint_id"], ids)

    def test_no_duplicate_when_already_hit(self):
        """測試 4：已被相似度命中的端點不重複補入（去重）。"""
        rows = [
            _mk_start("辦派對 預算 一萬", gap_categories=["budget"]),
        ]
        floored, _ = retrieve_top_k_endpoints(
            rows, "辦派對 預算 一萬", top_k=5, recent_floor_n=2,
        )
        self.assertEqual(len(floored), 1)

    def test_floor_added_after_topk_truncation(self):
        """測試 5：top_k 截斷後再補近因，最終長度可超過 top_k；近因排最前。"""
        # 5 筆高分端點 + 1 筆無重疊的最近端點
        high = [_mk_start(f"派對 預算 場地 {i}", gap_categories=[]) for i in range(5)]
        recent_unrelated = _mk_start("毫不相干的最新題目", gap_categories=[])
        rows = high + [recent_unrelated]

        floored, _ = retrieve_top_k_endpoints(
            rows, "派對 預算 場地", top_k=3, recent_floor_n=1,
        )
        # 3（截斷）+ 1（近因補入）= 4，超過 top_k
        self.assertEqual(len(floored), 4)
        # 最近的未命中端點插在最前 = 最高優先
        self.assertEqual(
            floored[0]["endpoint_id"], recent_unrelated["endpoint_id"],
        )


if __name__ == "__main__":
    unittest.main()
