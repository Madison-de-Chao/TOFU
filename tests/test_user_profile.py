"""user_profile 模組的單元測試。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta, timezone

from src.middleware.user_profile import (
    ProfileStore,
    _default_profile,
    _now_iso,
    check_pillar_observation_conflict,
    classify_preference_expression,
    classify_receiving_preference,
    classify_pace,
    classify_info_appetite,
    classify_context_need,
    extract_domains,
    extract_preferences,
    infer_satisfaction,
    render_profile_summary,
)


class DefaultProfileTests(unittest.TestCase):
    def test_default_has_implicit_preference(self):
        profile = _default_profile()
        self.assertEqual(
            profile["communication_style"]["preference_expression"], "implicit"
        )

    def test_default_data_source_is_default(self):
        profile = _default_profile()
        self.assertEqual(
            profile["communication_style"]["data_source"], "default"
        )
        self.assertEqual(
            profile["decision_style"]["data_source"], "default"
        )


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test_profile.json")
        self.store = ProfileStore(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_creates_default(self):
        profile = self.store.load()
        self.assertEqual(profile["version"], "standard")
        self.assertEqual(profile["meta"]["maturity"], "cold_start")

    def test_save_and_load_roundtrip(self):
        profile = self.store.load()
        profile["communication_style"]["receiving_preference"] = "minimal"
        self.store.save(profile)

        loaded = self.store.load()
        self.assertEqual(
            loaded["communication_style"]["receiving_preference"], "minimal"
        )

    def test_apply_human_calibration(self):
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "observed"
        self.store.save(profile)

        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        loaded = self.store.load()
        self.assertEqual(
            loaded["communication_style"]["receiving_preference"], "structured"
        )
        self.assertEqual(
            loaded["communication_style"]["data_source"], "human_calibrated"
        )

    def test_human_calibrated_not_overwritten(self):
        """human_calibrated 的維度不被 update_from_interaction 覆蓋。"""
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "human_calibrated"
        profile["communication_style"]["receiving_preference"] = "structured"
        self.store.save(profile)

        self.store.update_from_interaction(
            start_data={"user_input": "test"},
            end_data={"result": "ok", "user_satisfaction": "satisfied"},
            user_response="ok",
        )
        loaded = self.store.load()
        self.assertEqual(
            loaded["communication_style"]["data_source"], "human_calibrated"
        )

    def test_is_mature(self):
        self.assertFalse(self.store.is_mature())


class PreferenceExpressionTests(unittest.TestCase):
    def test_explicit(self):
        result = classify_preference_expression([
            {"user_input": "我喜歡戶外活動"},
            {"user_input": "我想要有景觀的地方"},
            {"user_input": "我偏好安靜的環境"},
        ])
        self.assertEqual(result, "explicit")

    def test_implicit(self):
        result = classify_preference_expression([
            {"user_input": "上次那個場地不錯"},
            {"user_input": "之前買的露營裝備很好用"},
        ])
        self.assertEqual(result, "implicit")

    def test_default_implicit(self):
        result = classify_preference_expression([
            {"user_input": "幫我辦活動"},
        ])
        self.assertEqual(result, "implicit")

    def test_exclusion(self):
        result = classify_preference_expression([
            {"user_input": "不要太商業化的"},
            {"user_input": "別給我重複的"},
            {"user_input": "避免高價位"},
        ])
        self.assertEqual(result, "exclusion")

    def test_explicit_english(self):
        result = classify_preference_expression([
            {"user_input": "I love outdoor activities"},
            {"user_input": "I prefer quiet places"},
            {"user_input": "my favorite cuisine is Japanese"},
        ])
        self.assertEqual(result, "explicit")

    def test_implicit_english(self):
        result = classify_preference_expression([
            {"user_input": "I bought a new tent"},
            {"user_input": "last time that was great"},
        ])
        self.assertEqual(result, "implicit")

    def test_exclusion_english(self):
        result = classify_preference_expression([
            {"user_input": "I don't like spicy food"},
            {"user_input": "horror movies are not my thing"},
            {"user_input": "I hate long meetings"},
        ])
        self.assertEqual(result, "exclusion")

    def test_english_case_insensitive(self):
        # 小寫、大寫、句首大寫變體都應匹配
        result = classify_preference_expression([
            {"user_input": "i love pizza"},
            {"user_input": "MY FAVORITE color is blue"},
            {"user_input": "I Like jazz music"},
        ])
        self.assertEqual(result, "explicit")

    def test_not_my_thing_only_exclusion(self):
        # 回歸測試：「not my thing」應僅計入 exclusion，不應同時觸發 implicit
        result = classify_preference_expression([
            {"user_input": "this is not my thing"},
        ])
        self.assertEqual(result, "exclusion")


class CommunicationStyleTests(unittest.TestCase):
    def test_minimal_for_short_input(self):
        result = classify_receiving_preference([
            {"user_input": "好"},
            {"user_input": "嗯"},
            {"user_input": "可以"},
        ])
        self.assertEqual(result, "minimal")

    def test_conversational_default(self):
        result = classify_receiving_preference([])
        self.assertEqual(result, "conversational")


class PaceTests(unittest.TestCase):
    def test_fast_pace(self):
        result = classify_pace([
            {"user_confirmation": "confirmed"},
            {"user_confirmation": "confirmed"},
            {"user_confirmation": "confirmed"},
            {"user_confirmation": "modified"},
        ])
        self.assertEqual(result, "fast")

    def test_deliberate_default(self):
        result = classify_pace([])
        self.assertEqual(result, "deliberate")


class ContextNeedTests(unittest.TestCase):
    def test_why_first(self):
        result = classify_context_need([
            {"user_input": "為什麼要先找場地"},
            {"user_input": "原因是什麼"},
        ])
        self.assertEqual(result, "why_first")

    def test_what_first_default(self):
        result = classify_context_need([])
        self.assertEqual(result, "what_first")


class PreferenceExtractionTests(unittest.TestCase):
    def test_explicit_preference(self):
        prefs = extract_preferences([
            {"user_input": "我喜歡日式料理"},
        ])
        self.assertTrue(any(p["type"] == "explicit" for p in prefs))

    def test_exclusion_preference(self):
        prefs = extract_preferences([
            {"user_input": "不要太辣的東西"},
        ])
        self.assertTrue(any(p["type"] == "exclusion" for p in prefs))

    def test_implicit_preference(self):
        prefs = extract_preferences([
            {"user_input": "我買了行動電源"},
        ])
        self.assertTrue(any(p["type"] == "implicit" for p in prefs))

    def test_explicit_english_preference(self):
        prefs = extract_preferences([
            {"user_input": "I love hiking in the mountains"},
        ])
        self.assertTrue(any(p["type"] == "explicit" for p in prefs))

    def test_implicit_english_preference(self):
        prefs = extract_preferences([
            {"user_input": "I bought a new camping stove"},
        ])
        self.assertTrue(any(p["type"] == "implicit" for p in prefs))

    def test_exclusion_english_preference(self):
        prefs = extract_preferences([
            {"user_input": "I don't like crowded places"},
        ])
        self.assertTrue(any(p["type"] == "exclusion" for p in prefs))

    def test_english_preference_case_insensitive(self):
        prefs = extract_preferences([
            {"user_input": "i love sushi restaurants"},
        ])
        self.assertTrue(any(p["type"] == "explicit" for p in prefs))

    def test_slice_indices_correct_with_unicode_case_change(self):
        # 回歸測試:Turkish 大寫 İ (U+0130) 的 .lower() 會變成 2 個
        # codepoint (i + U+0307 combining dot),會移動後續字元的索引。
        # 若用 text.lower() 取索引再切原始 text,context/item_text 會位移;
        # 改用 regex match span 從原始 text 取,才能保證索引正確。
        prefs = extract_preferences([
            {"user_input": "İstanbul I love pizza"},
        ])
        explicit = [p for p in prefs if p["type"] == "explicit"]
        self.assertTrue(explicit)
        # 關鍵:"I love" 後應該是 " pizza"(含前導空格),若索引偏移會變 "pizza"
        self.assertTrue(explicit[0]["item"].startswith("pizza"))
        self.assertIn("I love", explicit[0]["context"])


class SatisfactionTests(unittest.TestCase):
    def test_unsatisfied_signal(self):
        result = infer_satisfaction(
            {"result": "some result"},
            {"user_input": "算了不用了"},
        )
        self.assertEqual(result, "implicit_unsatisfied")

    def test_unknown_without_next(self):
        result = infer_satisfaction({"result": "some result"}, None)
        self.assertEqual(result, "unknown")


class ProfileSummaryTests(unittest.TestCase):
    def test_render_returns_string(self):
        profile = _default_profile()
        summary = render_profile_summary(profile)
        self.assertIn("逗福Tofu", summary)
        self.assertIn("溝通風格", summary)


class StateMachineTransitionTests(unittest.TestCase):
    """狀態轉移規則的完整測試。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test_profile.json")
        self.store = ProfileStore(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _interact(self, user_input="test", satisfaction="no_feedback"):
        return self.store.update_from_interaction(
            start_data={"user_input": user_input},
            end_data={"result": "ok", "user_satisfaction": satisfaction},
            user_response="ok",
        )

    # ---- 規則 1：default → observed ----
    def test_rule1_default_to_observed(self):
        profile = self._interact()
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        self.assertEqual(profile["communication_style"]["zone"], "A")
        self.assertEqual(
            profile["decision_style"]["data_source"], "observed"
        )

    # ---- 規則 2：default → pillars_mapped ----
    def test_rule2_default_to_pillars_mapped(self):
        pillars = {"day": {"stem": "甲", "branch": None}}
        profile, _msg = self.store.apply_pillars_mapping(pillars)
        self.assertEqual(
            profile["communication_style"]["data_source"], "pillars_mapped"
        )
        self.assertEqual(profile["communication_style"]["zone"], "B")
        self.assertEqual(
            profile["decision_style"]["data_source"], "pillars_mapped"
        )

    # ---- P0：observed 不被 pillars_mapped 覆蓋 ----
    def test_p0_observed_not_overwritten_by_pillars(self):
        """規則 2 注意：已有觀測值時，映射不覆蓋。"""
        # 先產生觀測值
        self._interact()
        profile = self.store.load()
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        old_pref = profile["communication_style"]["preference_expression"]

        # 嘗試套用四柱映射
        pillars = {"day": {"stem": "甲", "branch": None}}
        profile, _msg = self.store.apply_pillars_mapping(pillars)

        # data_source 應該還是 observed，不被覆蓋
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        self.assertEqual(
            profile["communication_style"]["preference_expression"], old_pref
        )

        # 但 rbh_log 應該有記錄
        rbh = profile.get("rbh_log", [])
        self.assertTrue(len(rbh) > 0)
        self.assertTrue(
            any(
                e.get("possible_reason", "").startswith("pillars_mapped 未覆蓋")
                for e in rbh
            )
        )

    def test_pillars_not_overwrite_human_calibrated(self):
        """human_calibrated 也不被 pillars_mapped 覆蓋。"""
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "minimal"
        )
        pillars = {"day": {"stem": "甲", "branch": None}}
        profile, _msg = self.store.apply_pillars_mapping(pillars)
        self.assertEqual(
            profile["communication_style"]["data_source"], "human_calibrated"
        )
        self.assertEqual(
            profile["communication_style"]["receiving_preference"], "minimal"
        )

    # ---- 規則 3：pillars_mapped → pillars_validated ----
    def test_rule3_pillars_mapped_to_validated(self):
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "pillars_mapped"
        profile["communication_style"]["preference_expression"] = "explicit"
        self.store.save(profile)

        entry = check_pillar_observation_conflict(
            profile,
            "communication_style.preference_expression",
            "explicit",  # 觀測一致
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry["action"], "pillars_validated")
        self.assertEqual(
            profile["communication_style"]["data_source"], "pillars_validated"
        )

    # ---- 規則 4：pillars_mapped → observed（衝突 >= 3）----
    def test_rule4_pillars_mapped_to_observed_on_conflict(self):
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "pillars_mapped"
        profile["communication_style"]["preference_expression"] = "explicit"
        # 先寫入 2 筆衝突記錄
        profile["rbh_log"] = [
            {"dimension": "communication_style.preference_expression",
             "match": False, "conflict_count": 1},
            {"dimension": "communication_style.preference_expression",
             "match": False, "conflict_count": 2},
        ]
        self.store.save(profile)

        entry = check_pillar_observation_conflict(
            profile,
            "communication_style.preference_expression",
            "implicit",  # 第 3 次衝突
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry["action"], "observed_override")
        self.assertEqual(entry["conflict_count"], 3)
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        self.assertEqual(
            profile["communication_style"]["preference_expression"], "implicit"
        )

    # ---- 規則 5：任何 → human_calibrated ----
    def test_rule5_any_to_human_calibrated(self):
        self._interact()  # observed
        profile = self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        self.assertEqual(
            profile["communication_style"]["data_source"], "human_calibrated"
        )
        self.assertIsNotNone(
            profile["communication_style"]["calibration_timestamp"]
        )
        self.assertEqual(
            profile["communication_style"]["calibration_value"], "structured"
        )
        self.assertEqual(
            profile["communication_style"]["interactions_since_calibration"], 0
        )

    def test_rule5_human_calibration_records_rbh_log(self):
        profile = self.store.apply_human_calibration(
            "decision_style.pace", "fast"
        )
        rbh = profile.get("rbh_log", [])
        human_entries = [e for e in rbh if e["action"] == "human_override"]
        self.assertEqual(len(human_entries), 1)
        self.assertEqual(human_entries[0]["previous_data_source"], "default")
        self.assertEqual(human_entries[0]["new_data_source"], "human_calibrated")

    # ---- 規則 5：human_calibrated 不被 update_from_interaction 覆蓋 ----
    def test_rule5_human_calibrated_blocks_observation(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        profile = self._interact(user_input="好")
        self.assertEqual(
            profile["communication_style"]["data_source"], "human_calibrated"
        )
        self.assertEqual(
            profile["communication_style"]["receiving_preference"], "structured"
        )

    # ---- 規則 5：human_calibrated 不被 check_pillar_observation_conflict 覆蓋 ----
    def test_rule5_human_calibrated_blocks_pillar_conflict(self):
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "human_calibrated"
        profile["communication_style"]["preference_expression"] = "explicit"
        self.store.save(profile)

        entry = check_pillar_observation_conflict(
            profile,
            "communication_style.preference_expression",
            "implicit",
        )
        self.assertIsNone(entry)

    # ---- 規則 6：human_calibrated → human_calibrated_stale（90 天）----
    def test_rule6_stale_after_90_days(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        # 手動設定時間為 91 天前
        profile = self.store.load()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat(
            timespec="seconds"
        )
        profile["communication_style"]["calibration_timestamp"] = old_ts
        self.store.save(profile)

        profile = self._interact()
        self.assertEqual(
            profile["communication_style"]["data_source"],
            "human_calibrated_stale",
        )

    # ---- 規則 6：human_calibrated → human_calibrated_stale（50 筆互動）----
    def test_rule6_stale_after_50_interactions(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        # 手動設定已累積 51 筆互動
        profile = self.store.load()
        profile["communication_style"]["interactions_since_calibration"] = 51
        self.store.save(profile)

        profile = self._interact()
        self.assertEqual(
            profile["communication_style"]["data_source"],
            "human_calibrated_stale",
        )

    # ---- 規則 6：尚未過期不降級 ----
    def test_rule6_not_stale_within_threshold(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        # 在安全範圍內互動
        profile = self._interact()
        self.assertEqual(
            profile["communication_style"]["data_source"], "human_calibrated"
        )

    # ---- 規則 7：human_calibrated_stale → observed（衝突 >= 5）----
    def test_rule7_stale_to_observed_on_conflict_5(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        profile = self.store.load()
        # 設定為已過期
        profile["communication_style"]["data_source"] = "human_calibrated_stale"
        profile["communication_style"]["calibration_value"] = "structured"
        # 預先記入 4 筆 stale 衝突
        for i in range(4):
            profile.setdefault("rbh_log", []).append({
                "dimension": "communication_style.receiving_preference",
                "prediction": "structured",
                "observed": "minimal",
                "match": False,
                "conflict_count": i + 1,
                "action": "conflict_recorded",
                "previous_data_source": "human_calibrated_stale",
                "new_data_source": "human_calibrated_stale",
                "possible_reason": "",
                "timestamp": _now_iso(),
            })
        self.store.save(profile)

        # 用極短輸入觸發 → 觀測值 = "minimal"，和 "structured" 衝突
        profile = self._interact(user_input="好")
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        # 應有通知
        notifications = self.store.get_pending_notifications()
        self.assertTrue(len(notifications) > 0)
        self.assertIn("structured", notifications[0])

    # ---- 規則 7：衝突 < 5 不覆蓋 ----
    def test_rule7_stale_conflict_below_threshold(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "human_calibrated_stale"
        profile["communication_style"]["calibration_value"] = "structured"
        self.store.save(profile)

        # 只一次衝突，不應覆蓋
        profile = self._interact(user_input="好")
        self.assertEqual(
            profile["communication_style"]["data_source"],
            "human_calibrated_stale",
        )

    # ---- 規則 8：human_calibrated_stale → human_calibrated（再次校準）----
    def test_rule8_stale_to_human_calibrated_on_recalibration(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "human_calibrated_stale"
        self.store.save(profile)

        # 使用者再次確認
        profile = self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        self.assertEqual(
            profile["communication_style"]["data_source"], "human_calibrated"
        )
        self.assertEqual(
            profile["communication_style"]["interactions_since_calibration"], 0
        )

    def test_rule8_stale_renewed_rbh_log(self):
        """stale → human_calibrated（同值）記 stale_renewed action。"""
        self.store.apply_human_calibration(
            "decision_style.pace", "fast"
        )
        profile = self.store.load()
        profile["decision_style"]["data_source"] = "human_calibrated_stale"
        profile["decision_style"]["calibration_value"] = "fast"
        self.store.save(profile)

        profile = self.store.apply_human_calibration(
            "decision_style.pace", "fast"
        )
        rbh = profile.get("rbh_log", [])
        renewed = [e for e in rbh if e["action"] == "stale_renewed"]
        self.assertEqual(len(renewed), 1)
        self.assertEqual(
            renewed[0]["previous_data_source"], "human_calibrated_stale"
        )

    # ---- 規則 9：observed → observed ----
    def test_rule9_observed_updates(self):
        self._interact()
        profile = self._interact(user_input="test again")
        self.assertEqual(
            profile["communication_style"]["data_source"], "observed"
        )
        self.assertGreaterEqual(
            profile["communication_style"]["sample_count"], 2
        )

    # ---- interactions_since_calibration 遞增 ----
    def test_interactions_since_calibration_increments(self):
        self.store.apply_human_calibration(
            "communication_style.receiving_preference", "structured"
        )
        self._interact()
        self._interact()
        profile = self.store.load()
        self.assertEqual(
            profile["communication_style"]["interactions_since_calibration"], 2
        )

    # ---- check_pillar_observation_conflict 不覆蓋 stale ----
    def test_pillar_conflict_skips_stale(self):
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "human_calibrated_stale"
        profile["communication_style"]["preference_expression"] = "explicit"
        self.store.save(profile)

        entry = check_pillar_observation_conflict(
            profile,
            "communication_style.preference_expression",
            "implicit",
        )
        self.assertIsNone(entry)

    # ---- rbh_log 格式驗證 ----
    def test_rbh_log_has_previous_and_new_data_source(self):
        """rbh_log entries should contain previous_data_source and new_data_source."""
        profile = self.store.load()
        profile["communication_style"]["data_source"] = "pillars_mapped"
        profile["communication_style"]["preference_expression"] = "explicit"
        self.store.save(profile)

        entry = check_pillar_observation_conflict(
            profile,
            "communication_style.preference_expression",
            "explicit",
        )
        self.assertIn("previous_data_source", entry)
        self.assertIn("new_data_source", entry)
        self.assertEqual(entry["previous_data_source"], "pillars_mapped")
        self.assertEqual(entry["new_data_source"], "pillars_validated")


class DefaultProfileFieldTests(unittest.TestCase):
    """確認新欄位存在於預設畫像。"""

    def test_calibration_fields_exist(self):
        profile = _default_profile()
        for section in ("communication_style", "decision_style"):
            self.assertIn("calibration_timestamp", profile[section])
            self.assertIsNone(profile[section]["calibration_timestamp"])
            self.assertIn("calibration_value", profile[section])
            self.assertIsNone(profile[section]["calibration_value"])
            self.assertIn("interactions_since_calibration", profile[section])
            self.assertEqual(
                profile[section]["interactions_since_calibration"], 0
            )


class RenderProfileSummaryStaleTests(unittest.TestCase):
    """render_profile_summary 應該能顯示 human_calibrated_stale。"""

    def test_stale_source_label(self):
        profile = _default_profile()
        profile["communication_style"]["data_source"] = "human_calibrated_stale"
        summary = render_profile_summary(profile)
        self.assertIn("真人校準（已過期）", summary)


if __name__ == "__main__":
    unittest.main()
