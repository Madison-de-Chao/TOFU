"""v0.5 端點檢索與元動機偵測邏輯鏈的測試。

對應 `docs/specs/20260416_逗福Tofu_端點檢索與元動機偵測_邏輯鏈_v0_5.md`
以及 DESIGN.md 第 17 節。

涵蓋：
- 端點冷卻欄位與冷卻規則（apply_cooldown / mark_hit）
- 第四層強制查表 Gate（retrieve_top_k_endpoints）
- 密碼表編碼（encode_retrieved_endpoints）
- 畫像線策略摘要（compute_strategy_brief）
- 首輪方向偵測（detect_first_round_direction）
- 三塊注入（fallback 模式下的 _v05_context）
- 腳本護欄（GuardrailTracker）
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.claude_client import (
    CODEBOOK,
    LLMClient,
    _fallback_execute_task,
    _fallback_generate_restate,
)
from src.middleware.baseline import (
    compute_baseline,
    compute_strategy_brief,
)
from src.middleware.confirmation import (
    detect_first_round_direction,
    FIRST_ROUND_GENERAL_THRESHOLD,
)
from src.middleware.endpoint import (
    COOLDOWN_DAYS,
    COOLDOWN_ROUNDS,
    DEFAULT_TOP_K,
    EndpointStore,
    _detect_noun_status,
    _ensure_cooldown_fields,
    encode_endpoint,
    encode_retrieved_endpoints,
    extract_key_nouns,
    extract_key_nouns_with_status,
    make_end_data,
    make_start_data,
    new_endpoint_id,
    new_event_id,
    retrieve_top_k_endpoints,
)
from src.middleware.guardrails import (
    GuardrailTracker,
    GuardrailTripped,
    MAX_COST_USD,
    MAX_SESSIONS,
)


def _mk_start(
    goal: str,
    constraints=None,
    gap_categories=None,
    topic: str = "general",
) -> dict:
    sd = make_start_data(
        user_input=goal,
        tofu_understanding="",
        gap_questions=[],
        user_confirmation="confirmed",
        user_modifications="",
        goal=goal,
        motivation="",
        constraints=list(constraints or []),
        gap_categories=list(gap_categories or []),
    )
    sd["topic"] = topic
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


# ----------------------------------------------------------------------
# Layer 1 + 2：冷卻欄位 + 強制查表 Gate
# ----------------------------------------------------------------------
class CooldownFieldsTests(unittest.TestCase):
    def test_legacy_row_gets_default_cooldown_fields(self):
        row = {
            "endpoint_id": "e1",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "type": "start",
            "event_id": "evt",
            "start_data": {"goal": "g"},
            "end_data": None,
        }
        _ensure_cooldown_fields(row)
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["reference_count"], 0)
        self.assertIsNone(row["last_referenced"])
        self.assertIsNone(row["round_last_referenced"])

    def test_append_start_initializes_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "endpoints.json")
            store = EndpointStore(path)
            sd = make_start_data(
                user_input="辦派對 預算一萬 30人",
                tofu_understanding="",
                gap_questions=[],
                user_confirmation="confirmed",
                user_modifications="",
                goal="辦派對",
                motivation="",
                constraints=["預算一萬"],
                gap_categories=["budget"],
            )
            row = store.append_start("evt1", sd)
            self.assertEqual(row["status"], "active")
            self.assertEqual(row["reference_count"], 0)

    def test_cold_legacy_file_gets_lazy_migration_on_read(self):
        """直接把舊格式寫入磁碟，_read() 讀回來時補欄位。"""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "endpoints.json")
            legacy_rows = [
                {
                    "endpoint_id": "e1",
                    "timestamp": "2020-01-01T00:00:00+00:00",
                    "type": "start",
                    "event_id": "evt",
                    "start_data": {"goal": "g"},
                    "end_data": None,
                }
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(legacy_rows, f)
            store = EndpointStore(path)
            rows = store.all()
            self.assertEqual(rows[0]["status"], "active")
            self.assertEqual(rows[0]["reference_count"], 0)


class CooldownRulesTests(unittest.TestCase):
    def _make_store(self) -> EndpointStore:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "endpoints.json")
        return EndpointStore(path)

    def test_mark_hit_increments_and_restores_pending(self):
        store = self._make_store()
        row = store.append_start(
            "evt",
            make_start_data(
                user_input="學 Premiere Pro",
                tofu_understanding="",
                gap_questions=[],
                user_confirmation="confirmed",
                user_modifications="",
                goal="學 Premiere Pro",
                motivation="",
                constraints=[],
                gap_categories=[],
            ),
        )
        # 手動把 status 設為 pending_confirmation 模擬冷卻發生
        data = store._read()
        data[0]["status"] = "pending_confirmation"
        store._write(data)

        store.mark_hit([row["endpoint_id"]], current_round=7)
        refreshed = store.all()
        self.assertEqual(refreshed[0]["status"], "active")
        self.assertEqual(refreshed[0]["reference_count"], 1)
        self.assertEqual(refreshed[0]["round_last_referenced"], 7)
        self.assertIsNotNone(refreshed[0]["last_referenced"])

    def test_apply_cooldown_rounds_threshold_marks_pending(self):
        store = self._make_store()
        row = store.append_start("e1", make_start_data(
            user_input="a", tofu_understanding="", gap_questions=[],
            user_confirmation="confirmed", user_modifications="",
            goal="a", motivation="", constraints=[], gap_categories=[],
        ))
        # 先 mark_hit 一次，讓 round_last_referenced 有值（例如 round 1）。
        # 這樣 apply_cooldown 的輪數門檻才會生效。
        # （round_last_referenced 為 None 的端點只走時間門檻，不走輪數門檻。）
        store.mark_hit([row["endpoint_id"]], current_round=1)
        changed = store.apply_cooldown(current_round=1 + COOLDOWN_ROUNDS + 5)
        self.assertEqual(changed, 1)
        rows = store.all()
        self.assertEqual(rows[0]["status"], "pending_confirmation")

    def test_apply_cooldown_never_hit_only_uses_time_threshold(self):
        """round_last_referenced 為 None → 只走時間門檻，不走輪數門檻。

        修正前的 bug：round_last_referenced 為 None 時退回 0，
        導致 global round 一超過 50，所有從未被 hit 的新端點立刻被降級。
        """
        store = self._make_store()
        store.append_start("e1", make_start_data(
            user_input="a", tofu_understanding="", gap_questions=[],
            user_confirmation="confirmed", user_modifications="",
            goal="a", motivation="", constraints=[], gap_categories=[],
        ))
        # round_last_referenced 為 None，即使 global round 遠超門檻也不觸發
        changed = store.apply_cooldown(current_round=COOLDOWN_ROUNDS + 999)
        self.assertEqual(changed, 0)

    def test_apply_cooldown_days_threshold_marks_pending(self):
        store = self._make_store()
        row = store.append_start("e1", make_start_data(
            user_input="a", tofu_understanding="", gap_questions=[],
            user_confirmation="confirmed", user_modifications="",
            goal="a", motivation="", constraints=[], gap_categories=[],
        ))
        # 把 timestamp 退回 100 天前
        data = store._read()
        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=COOLDOWN_DAYS + 10)
        ).isoformat(timespec="seconds")
        data[0]["timestamp"] = old_ts
        store._write(data)
        changed = store.apply_cooldown(current_round=1)
        self.assertEqual(changed, 1)
        rows = store.all()
        self.assertEqual(rows[0]["status"], "pending_confirmation")

    def test_apply_cooldown_does_not_delete_endpoints(self):
        store = self._make_store()
        for i in range(3):
            store.append_start(f"e{i}", make_start_data(
                user_input=f"a{i}", tofu_understanding="", gap_questions=[],
                user_confirmation="confirmed", user_modifications="",
                goal=f"a{i}", motivation="", constraints=[], gap_categories=[],
            ))
        store.apply_cooldown(current_round=COOLDOWN_ROUNDS + 999)
        self.assertEqual(len(store.all()), 3)

    def test_apply_cooldown_skips_already_pending(self):
        store = self._make_store()
        store.append_start("e1", make_start_data(
            user_input="a", tofu_understanding="", gap_questions=[],
            user_confirmation="confirmed", user_modifications="",
            goal="a", motivation="", constraints=[], gap_categories=[],
        ))
        data = store._read()
        data[0]["status"] = "pending_confirmation"
        store._write(data)
        # 不應再被標記（已經是 pending）
        changed = store.apply_cooldown(current_round=COOLDOWN_ROUNDS + 999)
        self.assertEqual(changed, 0)


class StartEndJoinTests(unittest.TestCase):
    """start↔end 的 in-memory join 測試。

    append_start / append_end 分別存成獨立 row，start row 的 end_data
    永遠是 None。retrieval 和 encode 需要把 end row 的 end_data 接回去，
    才能讓 result 文字參與搜索計分和密碼表 RESULT 欄位產出。
    """

    def test_retrieve_matches_on_end_data_result_terms(self):
        """end_data.result 裡的詞也能讓端點被查表命中。"""
        event_id = "evt-join-test"
        start_row = _mk_start("學影片剪輯", gap_categories=["time"])
        start_row["event_id"] = event_id
        # start_data 裡完全沒提到 Premiere Pro
        end_row = {
            "endpoint_id": new_endpoint_id(),
            "timestamp": start_row["timestamp"],
            "type": "end",
            "event_id": event_id,
            "start_data": None,
            "end_data": {"result": "推薦 Premiere Pro 和 DaVinci Resolve"},
        }
        all_rows = [start_row, end_row]

        # 用 Premiere Pro 查表 → 應該命中這個端點
        top, _ = retrieve_top_k_endpoints(all_rows, "Premiere Pro", top_k=5)
        self.assertGreaterEqual(len(top), 1)
        self.assertEqual(top[0]["event_id"], event_id)

    def test_encode_populates_result_line_from_joined_end_data(self):
        """encode 後密碼表的 RESULT 欄位應該含 end_data.result 的內容。"""
        event_id = "evt-encode-join"
        start_row = _mk_start("學攝影", gap_categories=[])
        start_row["event_id"] = event_id
        end_row = {
            "endpoint_id": new_endpoint_id(),
            "timestamp": start_row["timestamp"],
            "type": "end",
            "event_id": event_id,
            "start_data": None,
            "end_data": {"result": "推薦 Lightroom Classic 做後製"},
        }
        all_rows = [start_row, end_row]

        # retrieve 會把 end_data 接到 start row
        top, _ = retrieve_top_k_endpoints(all_rows, "攝影", top_k=5)
        self.assertGreaterEqual(len(top), 1)

        encoded = encode_retrieved_endpoints(top)
        self.assertIn("RESULT:", encoded)
        self.assertIn("Lightroom Classic", encoded)


class ForcedLookupGateTests(unittest.TestCase):
    def test_retrieve_returns_empty_on_empty_store(self):
        top, known = retrieve_top_k_endpoints([], "辦派對")
        self.assertEqual(top, [])
        self.assertEqual(known, [])

    def test_retrieve_scores_by_overlap(self):
        rows = [
            _mk_start("辦派對 30 人 預算 1 萬", gap_categories=["budget", "headcount"]),
            _mk_start("學攝影 買相機", gap_categories=["budget"]),
            _mk_start("寫報告 交期", gap_categories=["deadline"]),
        ]
        top, known = retrieve_top_k_endpoints(rows, "辦派對 預算 一萬", top_k=3)
        self.assertGreaterEqual(len(top), 1)
        # 最相關的派對端點應排第一
        self.assertIn("派對", top[0]["start_data"]["goal"])
        self.assertIn("budget", known)

    def test_retrieve_respects_top_k_limit(self):
        rows = [_mk_start(f"goal {i} 派對 預算") for i in range(10)]
        top, _ = retrieve_top_k_endpoints(rows, "派對 預算", top_k=3)
        self.assertEqual(len(top), 3)

    def test_retrieve_penalizes_pending_confirmation(self):
        active = _mk_start("辦派對 預算 1 萬", gap_categories=["budget"])
        pending = _mk_start("辦派對 預算 2 萬", gap_categories=["budget"])
        pending["status"] = "pending_confirmation"
        # 兩者 token 重疊一樣大，active 應該在 pending 之前
        top, _ = retrieve_top_k_endpoints([pending, active], "辦派對 預算", top_k=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["status"], "active")

    def test_retrieve_excludes_superseded(self):
        active = _mk_start("辦派對 預算 1 萬", gap_categories=["budget"])
        superseded = _mk_start("辦派對 預算 2 萬", gap_categories=["budget"])
        superseded["status"] = "superseded"
        top, _ = retrieve_top_k_endpoints([active, superseded], "辦派對 預算", top_k=5)
        ids = [r["status"] for r in top]
        self.assertNotIn("superseded", ids)

    def test_retrieve_default_top_k_is_30(self):
        rows = [_mk_start(f"辦派對 預算 #{i}") for i in range(50)]
        top, _ = retrieve_top_k_endpoints(rows, "派對 預算")
        self.assertLessEqual(len(top), DEFAULT_TOP_K)


class EncodeRetrievedTests(unittest.TestCase):
    def test_encode_empty(self):
        self.assertEqual(encode_retrieved_endpoints([]), "")

    def test_encode_marks_pending_status(self):
        row = _mk_start("辦派對 30 人", gap_categories=["headcount"])
        row["status"] = "pending_confirmation"
        text = encode_retrieved_endpoints([row])
        self.assertIn("STATUS: pending_confirmation", text)

    def test_encode_produces_e001_header(self):
        row = _mk_start("辦派對")
        text = encode_retrieved_endpoints([row])
        self.assertIn("[E001]", text)


# ----------------------------------------------------------------------
# Layer 3：畫像線策略摘要
# ----------------------------------------------------------------------
class StrategyBriefTests(unittest.TestCase):
    def test_cold_start_returns_default(self):
        brief = compute_strategy_brief(compute_baseline([]))
        self.assertIn("冷啟動", brief)

    def test_detects_blind_spot_time(self):
        # v0.5+ 自帶率定義：出現次數 / 總數
        # 10 筆端點、budget 每筆都出現 → 自帶率 100% → 強項
        # 其他完全沒出現的類別 → 自帶率 0% → 重度盲區
        rows = [_mk_start("a", gap_categories=["budget"]) for _ in range(10)]
        baseline = compute_baseline(rows)
        brief = compute_strategy_brief(baseline)
        # budget 是強項（自帶率 100%）
        self.assertIn("預算", brief)
        self.assertIn("強項", brief)
        self.assertIn("100%", brief)
        # 其他沒出現的類別是盲區
        self.assertIn("盲區", brief)
        self.assertIn("0%", brief)

    def test_empty_gap_categories_are_all_blind(self):
        """使用者 bug report：跑一個全空 gap_categories 的端點集合
        → 每個類別的自帶率都是 0%，全部應判為盲區，不是強項。"""
        rows = [_mk_start("a", gap_categories=[]) for _ in range(5)]
        brief = compute_strategy_brief(compute_baseline(rows))
        self.assertIn("盲區", brief)
        # 不應該寫成「強項」，因為沒有資料能佐證使用者自帶
        self.assertNotIn("強項", brief)
        # 自帶率應該是 0%，不是 100%
        self.assertIn("0%", brief)
        self.assertNotIn("100%", brief)

    def test_honors_decision_style_pace(self):
        baseline = compute_baseline([_mk_start("a", gap_categories=["budget"])])
        fast_brief = compute_strategy_brief(
            baseline,
            user_profile={"decision_style": {"pace": "fast"}},
        )
        slow_brief = compute_strategy_brief(
            baseline,
            user_profile={"decision_style": {"pace": "deliberate"}},
        )
        self.assertIn("急性子", fast_brief)
        self.assertIn("要求準確", slow_brief)

    def test_length_bounded(self):
        # 製造很多 gap category 進來，確認輸出不會超過 260 字元
        rows = []
        cats = ["budget", "time", "venue", "stakeholder", "deliverable"]
        for c in cats:
            for _ in range(3):
                rows.append(_mk_start("a", gap_categories=[c]))
        brief = compute_strategy_brief(compute_baseline(rows))
        self.assertLessEqual(len(brief), 260)

    def test_blind_spot_uses_conditional_trigger(self):
        """v0.5+ 修正：盲區提醒改為條件觸發，不再絕對。

        - 必須出現「當使用者的問題涉及決策、規劃、購買或行動時」這種
          條件語句
        - 必須出現「純粹的資訊查詢或推薦（無資源投入）不觸發」這種
          反例條款
        - **不得**出現原本的「必須主動包含」「即使使用者沒問」這種
          絕對語氣——那會讓 LLM 在純查詢情境硬塞預算/時間提醒
        """
        # 10 筆端點，gap_categories 全空 → 所有類別自帶率 0% → 全部是盲區
        rows = [_mk_start("a", gap_categories=[]) for _ in range(10)]
        brief = compute_strategy_brief(compute_baseline(rows))
        # 條件觸發的關鍵字
        self.assertIn("決策", brief)
        self.assertIn("規劃", brief)
        self.assertIn("購買", brief)
        self.assertIn("行動", brief)
        # 反例條款
        self.assertIn("純粹的資訊查詢", brief)
        self.assertIn("不觸發", brief)
        # 舊的絕對語氣必須消失
        self.assertNotIn("必須主動包含", brief)
        self.assertNotIn("即使使用者沒問", brief)


# ----------------------------------------------------------------------
# Layer 5：首輪方向偵測
# ----------------------------------------------------------------------
class FirstRoundDirectionTests(unittest.TestCase):
    def test_general_established_returns_none(self):
        self.assertIsNone(
            detect_first_round_direction("我想學攝影", completed_rounds=10)
        )

    def test_self_only_flags_when_immature(self):
        result = detect_first_round_direction("我想學攝影", completed_rounds=0)
        self.assertEqual(result, "self_only")

    def test_mentions_other_returns_none(self):
        result = detect_first_round_direction(
            "我想幫我媽辦一場生日派對", completed_rounds=0,
        )
        self.assertIsNone(result)

    def test_pure_event_description_flagged_as_self(self):
        # 沒有人稱代名詞但也沒提到他人 → 視為 self_only
        self.assertEqual(
            detect_first_round_direction("辦派對", completed_rounds=0),
            "self_only",
        )

    def test_threshold_respected(self):
        self.assertIsNone(
            detect_first_round_direction(
                "我想學攝影",
                completed_rounds=FIRST_ROUND_GENERAL_THRESHOLD,
            )
        )


# ----------------------------------------------------------------------
# Layer 4：三塊注入（透過 fallback 檢驗管線完整性）
# ----------------------------------------------------------------------
class ThreeBlockInjectionTests(unittest.TestCase):
    def test_codebook_has_v05_fields(self):
        for key in ("PREF_EXP", "PREF_IMP", "AVOID", "KEY", "SRC"):
            self.assertIn(key, CODEBOOK)

    def test_fallback_records_v05_context(self):
        out = _fallback_generate_restate(
            "辦派對",
            baseline_summary={},
            allowed_categories=["budget", "time"],
            strategy_brief="盲區：時間",
            encoded_top_k="[E001] ...",
            first_round_direction_hint="self_only",
        )
        ctx = out.get("_v05_context")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["strategy_brief"], "盲區：時間")
        self.assertEqual(ctx["first_round_direction_hint"], "self_only")
        self.assertGreater(ctx["encoded_top_k_chars"], 0)

    def test_fallback_self_only_hints_adds_other_question(self):
        out_without = _fallback_generate_restate(
            "辦派對",
            baseline_summary={},
            allowed_categories=[],
        )
        out_with = _fallback_generate_restate(
            "辦派對",
            baseline_summary={},
            allowed_categories=[],
            first_round_direction_hint="self_only",
        )
        # self_only 情況下應該多追加一個「別人方向」的反問
        self.assertGreater(
            len(out_with["gap_questions"]),
            len(out_without["gap_questions"]),
        )
        joined = " ".join(out_with["gap_questions"])
        self.assertTrue("誰" in joined or "別人" in joined)

    def test_fallback_execute_task_echoes_brief(self):
        text = _fallback_execute_task(
            goal="辦派對",
            motivation="",
            constraints=[],
            strategy_brief="盲區：時間（自帶率 4%）。回答時主動帶出時間提醒。",
        )
        self.assertIn("盲區", text)


# ----------------------------------------------------------------------
# Layer 6：腳本護欄
# ----------------------------------------------------------------------
class GuardrailsTests(unittest.TestCase):
    def test_fresh_tracker_passes_check(self):
        tracker = GuardrailTracker()
        tracker.check_before_call()

    def test_cost_trip(self):
        tracker = GuardrailTracker(cost_per_call_usd=1.0)
        tracker.cumulative_cost_usd = MAX_COST_USD - 0.1
        with self.assertRaises(GuardrailTripped) as ctx:
            tracker.check_before_call()
        self.assertEqual(ctx.exception.field_name, "cost")

    def test_sessions_trip(self):
        tracker = GuardrailTracker()
        tracker.sessions_completed = MAX_SESSIONS
        with self.assertRaises(GuardrailTripped) as ctx:
            tracker.check_before_call()
        self.assertEqual(ctx.exception.field_name, "sessions")

    def test_runtime_trip_uses_time_provider(self):
        clock = [0.0]

        def _fake_time() -> float:
            return clock[0]

        tracker = GuardrailTracker(time_provider=_fake_time)
        # 模擬時間推進到超過 120 分鐘
        clock[0] = 120.0 * 60 + 1.0
        with self.assertRaises(GuardrailTripped) as ctx:
            tracker.check_before_call()
        self.assertEqual(ctx.exception.field_name, "runtime")

    def test_record_call_and_session(self):
        tracker = GuardrailTracker()
        tracker.record_call(cost_usd=0.01)
        tracker.record_session()
        self.assertAlmostEqual(tracker.cumulative_cost_usd, 0.01)
        self.assertEqual(tracker.sessions_completed, 1)

    def test_status_line_contains_numbers(self):
        tracker = GuardrailTracker()
        line = tracker.status_line()
        self.assertIn("cost=", line)
        self.assertIn("sessions=", line)
        self.assertIn("runtime=", line)


# ----------------------------------------------------------------------
# B：KEY 名詞的動詞方向偵測
# ----------------------------------------------------------------------
class NounStatusDetectionTests(unittest.TestCase):
    def test_acquired_english(self):
        self.assertEqual(
            _detect_noun_status("I just bought a Nikon D750 yesterday.", "Nikon D750"),
            "acquired",
        )

    def test_acquired_chinese(self):
        self.assertEqual(
            _detect_noun_status("我上週買了一台 Rolleiflex 2.8F。", "Rolleiflex"),
            "acquired",
        )

    def test_active_english(self):
        self.assertEqual(
            _detect_noun_status(
                "I'm still using my Nikon D750 for portraits.", "Nikon D750"
            ),
            "active",
        )

    def test_active_chinese(self):
        self.assertEqual(
            _detect_noun_status("我現在用 Rolleiflex 拍底片。", "Rolleiflex"),
            "active",
        )

    def test_divested_english(self):
        self.assertEqual(
            _detect_noun_status("I sold my Nikon D750 last month.", "Nikon D750"),
            "divested",
        )

    def test_divested_chinese(self):
        self.assertEqual(
            _detect_noun_status("上個月把 Rolleiflex 賣掉了。", "Rolleiflex"),
            "divested",
        )

    def test_abandoned_english(self):
        self.assertEqual(
            _detect_noun_status(
                "I stopped using the Nikon D750 because it's too heavy.",
                "Nikon D750",
            ),
            "abandoned",
        )

    def test_abandoned_chinese(self):
        self.assertEqual(
            _detect_noun_status("我已經不用 Rolleiflex 了，太重了。", "Rolleiflex"),
            "abandoned",
        )

    def test_mentioned_default(self):
        self.assertEqual(
            _detect_noun_status("Rolleiflex is a classic camera.", "Rolleiflex"),
            "mentioned",
        )

    def test_nearest_verb_wins_in_multi_noun_sentence(self):
        # 「bought Rolleiflex but stopped using Leica」
        # Rolleiflex 最近的動詞是 bought → acquired
        # Leica M6 最近的動詞是 stopped → abandoned
        # 這是多名詞句子中 nearest-verb 演算法的關鍵保證。
        text = "I bought a Rolleiflex but I stopped using my Leica M6."
        self.assertEqual(_detect_noun_status(text, "Rolleiflex"), "acquired")
        self.assertEqual(_detect_noun_status(text, "Leica M6"), "abandoned")

    def test_chinese_comma_stops_cross_clause_leak(self):
        # 中文逗號 + 「但」應該把子句切開，避免「不用」污染前半段的名詞。
        text = "我買了 Nikon D750 配 24-70mm 鏡頭，但已經不用 Canon 5D 了。"
        self.assertEqual(_detect_noun_status(text, "Nikon D750"), "acquired")
        self.assertEqual(_detect_noun_status(text, "24-70mm"), "acquired")
        self.assertEqual(_detect_noun_status(text, "Canon 5D"), "abandoned")

    def test_english_but_stops_cross_clause_leak(self):
        # 英文 `but` 作為對比連接詞也應該截斷語意範圍。
        text = "I still use my Rolleiflex but I sold my Leica M6."
        self.assertEqual(_detect_noun_status(text, "Rolleiflex"), "active")
        self.assertEqual(_detect_noun_status(text, "Leica M6"), "divested")

    def test_noun_not_in_text_returns_mentioned(self):
        self.assertEqual(
            _detect_noun_status("some random sentence", "Rolleiflex"),
            "mentioned",
        )

    def test_window_limits_context_scope(self):
        # noun 和動詞之間距離超過預設視窗 → 不算命中
        very_far = "I bought a car " + ("blah " * 50) + "Nikon D750"
        self.assertEqual(
            _detect_noun_status(very_far, "Nikon D750", window=10),
            "mentioned",
        )


class ExtractKeyNounsWithStatusTests(unittest.TestCase):
    def test_returns_item_and_status_dicts(self):
        text = "I bought a Nikon D750 and I still use my Sony A7R III."
        entries = extract_key_nouns_with_status(text)
        self.assertGreaterEqual(len(entries), 2)
        for e in entries:
            self.assertIn("item", e)
            self.assertIn("status", e)

    def test_pure_extract_key_nouns_still_returns_strings(self):
        """B 必須保持 retrieve_top_k 用的字串版相容性。"""
        nouns = extract_key_nouns("I bought a Nikon D750.")
        self.assertTrue(all(isinstance(n, str) for n in nouns))
        self.assertIn("Nikon D750", nouns)

    def test_mixed_statuses_in_one_text(self):
        text = "I bought a Rolleiflex but I stopped using my Leica M6."
        entries = extract_key_nouns_with_status(text)
        status_map = {e["item"]: e["status"] for e in entries}
        self.assertEqual(status_map.get("Rolleiflex"), "acquired")
        self.assertEqual(status_map.get("Leica M6"), "abandoned")


class EncodeEndpointKeyStatusTests(unittest.TestCase):
    def test_key_line_includes_status_markers(self):
        block = encode_endpoint(
            user_input="I bought a Nikon D750 last week.",
            result="Here's the Lightroom Classic 12.5 workflow.",
            topic="photography",
            zone="A",
            session_date="2026-04-16",
            preferences=[],
            endpoint_index=1,
            session_ref="evt1",
            turn_ref="1",
        )
        self.assertIn("KEY:", block)
        self.assertIn("Nikon D750 [acquired]", block)

    def test_key_line_omits_status_for_mentioned(self):
        block = encode_endpoint(
            user_input="Rolleiflex is a classic camera.",
            result="Yes, it's well known.",
            topic="photography",
            zone="B",
            session_date="2026-04-16",
            preferences=[],
            endpoint_index=1,
        )
        # 預設狀態 mentioned 不印標記，避免噪音
        self.assertIn("Rolleiflex", block)
        self.assertNotIn("[mentioned]", block)

    def test_abandoned_noun_renders_correctly(self):
        block = encode_endpoint(
            user_input="I stopped using my Leica M6 because it's too heavy.",
            result="Consider lighter options.",
            topic="photography",
            zone="B",
            session_date="2026-04-16",
            preferences=[],
            endpoint_index=1,
        )
        self.assertIn("Leica M6 [abandoned]", block)


# ----------------------------------------------------------------------
# C：pending_confirmation 自然融入語境（prompt 措辭）
# ----------------------------------------------------------------------
class PendingConfirmationPromptToneTests(unittest.TestCase):
    """檢查 execute_task / generate_restate 的 system prompt 包含
    「pending_confirmation 自然融入、不要客服句型」這條指令。

    Prompt 文字層的測試走 `unittest.mock.patch` 攔截 `_call`，
    不真的連線到 API。
    """

    def _make_client(self) -> LLMClient:
        import os
        os.environ.pop("CLAUDE_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        client = LLMClient()
        # 強制脫離 fallback，好讓 prompt 實際被組出來。
        client.fallback_mode = False
        client._client = object()  # 任意非 None，後面 _call 會被 mock
        return client

    def test_execute_task_injects_natural_confirmation_line(self):
        from unittest.mock import patch
        client = self._make_client()
        captured: dict[str, str] = {}

        def _fake_call(system: str, user: str, max_tokens: int = 1024) -> str:
            captured["system"] = system
            captured["user"] = user
            return "OK"

        with patch.object(client, "_call", side_effect=_fake_call):
            client.execute_task(
                goal="學攝影",
                motivation="",
                constraints=[],
                strategy_brief="此使用者盲區：時間（自帶率 4%）。",
                encoded_top_k=(
                    "[E001] 2026-01-01 | photography | zone:A\n"
                    "GOAL: 學底片攝影\nRESULT: 推薦 Rolleiflex\n"
                    "KEY: Rolleiflex [active]\n"
                    "STATUS: pending_confirmation"
                ),
            )

        sys_prompt = captured["system"]
        # 指令的關鍵字詞
        self.assertIn("pending_confirmation", sys_prompt)
        self.assertIn("自然融入", sys_prompt)
        self.assertIn("不要用獨立的客服句型", sys_prompt)
        # 也要提到 KEY 狀態標記的處理
        self.assertIn("[abandoned]", sys_prompt)

    def test_generate_restate_injects_natural_confirmation_line(self):
        from unittest.mock import patch
        client = self._make_client()
        captured: dict[str, str] = {}

        def _fake_call(system: str, user: str, max_tokens: int = 1024) -> str:
            captured["system"] = system
            captured["user"] = user
            return '{"restate_text": "", "gap_questions": [], "gap_categories": []}'

        with patch.object(client, "_call", side_effect=_fake_call):
            client.generate_restate(
                user_input="想升級相機",
                baseline_summary={},
                allowed_categories=["budget"],
                strategy_brief="此使用者盲區：預算。",
                encoded_top_k=(
                    "[E001] 2026-01-01 | photography | zone:A\n"
                    "GOAL: 買相機\nRESULT: ...\n"
                    "KEY: Nikon D750 [abandoned]\n"
                    "STATUS: pending_confirmation"
                ),
            )

        sys_prompt = captured["system"]
        self.assertIn("pending_confirmation", sys_prompt)
        self.assertIn("自然融入", sys_prompt)
        self.assertIn("客服句型", sys_prompt)

    def test_execute_task_skips_tone_block_when_no_top_k(self):
        """沒密碼表就不該注入這段 prompt，避免冷啟動時浪費 tokens。"""
        from unittest.mock import patch
        client = self._make_client()
        captured: dict[str, str] = {}

        def _fake_call(system: str, user: str, max_tokens: int = 1024) -> str:
            captured["system"] = system
            return "OK"

        with patch.object(client, "_call", side_effect=_fake_call):
            client.execute_task(
                goal="學攝影",
                motivation="",
                constraints=[],
                strategy_brief="此使用者尚無基線：目前是冷啟動。",
                encoded_top_k=None,
            )

        self.assertNotIn("pending_confirmation", captured["system"])

    def test_execute_task_prompts_against_internal_terminology_leak(self):
        """BUG 2：system prompt 必須明確禁止 LLM 在回應中提到「密碼表」
        「CODEBOOK」「端點」等內部術語，並提供「根據你之前提過」「根據我
        對你的認識」這類合規改寫範例。"""
        from unittest.mock import patch
        client = self._make_client()
        captured: dict[str, str] = {}

        def _fake_call(system: str, user: str, max_tokens: int = 1024) -> str:
            captured["system"] = system
            return "OK"

        with patch.object(client, "_call", side_effect=_fake_call):
            client.execute_task(goal="學攝影", motivation="", constraints=[])

        sys_prompt = captured["system"]
        # 必須明確禁止的內部術語
        self.assertIn("密碼表", sys_prompt)
        self.assertIn("CODEBOOK", sys_prompt)
        self.assertIn("端點", sys_prompt)
        # 必須提供合規改寫方式
        self.assertIn("根據你之前提過", sys_prompt)
        self.assertIn("根據我對你的認識", sys_prompt)
        # 必須有「不要告訴使用者系統底層」這類指引
        self.assertIn("不合規", sys_prompt)
        self.assertIn("合規", sys_prompt)


if __name__ == "__main__":
    unittest.main()
