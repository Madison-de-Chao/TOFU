"""v0.6 PR-D：/check 模式（ISF 資訊完整性檢驗器）三階段測試。

對應 `docs/specs/逗福_v0_6_開發_Spec_v2.md` §1.6。

範圍：
- `src/modes/check_prompt.py` 的 ISF v3.2 常數與 user message 組裝
- `src/middleware/check.py` 的 session state 管理（建、讀、刪、sweep、fallback）
- `LLMClient.run_check_stage` fallback 路徑與三個階段的模板
- `run_check_mode` 第一階段端到端：建 session → LLM 呼叫 → 寫回 stage1_output
- `run_check_followup` 1/2/3 接續：找 session、跑對應階段、寫端點、清 state
- dispatcher 在有 active session 時把 1/2/3 視為接續訊號
- 與 baseline 的隔離：/check 端點的 mode='check' 被 baseline 自動排除
"""

from __future__ import annotations

import builtins
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from src.api.base_client import (
    _fallback_check_full,
    _fallback_check_summary,
    _fallback_check_warning,
)
from src.api.claude_client import LLMClient
from src.middleware import baseline as baseline_mod
from src.middleware import check as check_mod
from src.middleware.endpoint import EndpointStore
from src.middleware.guardrails import GuardrailTracker
from src.modes.check_prompt import (
    ISF_IMAGE_FALLBACK_INSTRUCTIONS,
    ISF_V32_PROMPT,
    STAGE_FULL,
    STAGE_SUMMARY,
    STAGE_WARNING,
    build_check_system_prompt,
    build_check_user_message,
)


# ---------------------------------------------------------------------------
# 1. ISF prompt 常數 + 組裝
# ---------------------------------------------------------------------------
class CheckPromptConstantTests(unittest.TestCase):
    def test_isf_prompt_verbatim_preserved(self):
        """MOMO 原文的關鍵詞必須 verbatim 保留。"""
        self.assertIn("資訊完整性檢驗器", ISF_V32_PROMPT)
        self.assertIn("ISF 公約", ISF_V32_PROMPT)
        self.assertIn("虹靈御所", ISF_V32_PROMPT)
        # ISF 公式
        self.assertIn("V × 0.30", ISF_V32_PROMPT)
        self.assertIn("M × 0.25", ISF_V32_PROMPT)
        # 三階段架構
        self.assertIn("第一階段", ISF_V32_PROMPT)
        self.assertIn("第二階段", ISF_V32_PROMPT)
        self.assertIn("第三階段", ISF_V32_PROMPT)
        # 互動選項
        self.assertIn("1️⃣", ISF_V32_PROMPT)
        self.assertIn("2️⃣", ISF_V32_PROMPT)
        self.assertIn("3️⃣", ISF_V32_PROMPT)

    def test_image_fallback_instructions_present(self):
        """逗福版疊加指令必須涵蓋四個重點。"""
        s = ISF_IMAGE_FALLBACK_INSTRUCTIONS
        # A. Zone 定義獨立
        self.assertIn("Zone", s)
        self.assertIn("獨立", s)
        # B. 圖片降級為文字警示稿
        self.assertIn("[WARNING_META]", s)
        self.assertIn("不要嘗試產出圖片", s)
        # C. 階段切換由程式碼控制
        self.assertIn("階段由逗福外層", s)
        # D. 不碰逗福身份（/check 下的身份交接）
        self.assertIn("ISF", s)

    def test_build_system_prompt_concatenates_both(self):
        sp = build_check_system_prompt()
        self.assertIn("資訊完整性檢驗器", sp)
        self.assertIn("[WARNING_META]", sp)

    def test_build_user_message_summary(self):
        msg = build_check_user_message(
            stage=STAGE_SUMMARY,
            user_content="投資保證獲利的 LINE 訊息",
        )
        self.assertIn("第一階段", msg)
        self.assertIn("投資保證獲利", msg)
        self.assertIn("1️⃣", msg)

    def test_build_user_message_full_includes_stage1(self):
        msg = build_check_user_message(
            stage=STAGE_FULL,
            user_content="原始訊息",
            stage1_output="第一階段摘要內容 XYZ",
        )
        self.assertIn("第二階段", msg)
        self.assertIn("使用者已選「1」", msg)
        self.assertIn("第一階段摘要內容 XYZ", msg)
        self.assertIn("原始訊息", msg)

    def test_build_user_message_warning_includes_warning_instruction(self):
        msg = build_check_user_message(
            stage=STAGE_WARNING,
            user_content="原始訊息",
            stage1_output="第一階段摘要",
        )
        self.assertIn("警示文字稿", msg)
        self.assertIn("使用者已選「2」", msg)
        self.assertIn("[WARNING_META]", msg)
        self.assertIn("不要", msg)
        self.assertIn("圖片", msg)

    def test_build_user_message_rejects_unknown_stage(self):
        with self.assertRaises(ValueError):
            build_check_user_message(stage="bogus", user_content="x")


# ---------------------------------------------------------------------------
# 2. fallback 模板
# ---------------------------------------------------------------------------
class CheckFallbackTemplateTests(unittest.TestCase):
    def test_summary_fallback_is_honest_about_offline(self):
        out = _fallback_check_summary("投資 LINE 訊息")
        self.assertIn("資訊完整性快速摘要", out)
        # 必須明確標註離線，不能裝出真的評估過
        self.assertIn("離線", out)
        # 仍然印選單
        self.assertIn("1️⃣", out)
        self.assertIn("3️⃣", out)

    def test_summary_fallback_includes_content_preview(self):
        out = _fallback_check_summary("這是一則長長長長長長長長長長長長長長內容")
        # 預覽不宜過長（實作截 30 字）
        self.assertIn("這是一則", out)

    def test_full_fallback_references_165(self):
        out = _fallback_check_full("x", stage1_output=None)
        self.assertIn("165", out)

    def test_warning_fallback_has_warning_meta_block(self):
        out = _fallback_check_warning("x", stage1_output=None)
        self.assertIn("[WARNING_META]", out)
        self.assertIn("risk_level", out)
        self.assertIn("color_scheme", out)


# ---------------------------------------------------------------------------
# 3. LLMClient.run_check_stage 走 fallback
# ---------------------------------------------------------------------------
class LLMClientCheckStageFallbackTests(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient()  # no API key → fallback
        self.assertTrue(self.client.fallback_mode)

    def test_summary_stage(self):
        out = self.client.run_check_stage(
            stage=STAGE_SUMMARY, user_content="測試"
        )
        self.assertIn("快速摘要", out)

    def test_full_stage(self):
        out = self.client.run_check_stage(
            stage=STAGE_FULL, user_content="x", stage1_output="s1"
        )
        self.assertIn("完整建議", out)

    def test_warning_stage(self):
        out = self.client.run_check_stage(
            stage=STAGE_WARNING, user_content="x", stage1_output="s1"
        )
        self.assertIn("[WARNING_META]", out)

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            self.client.run_check_stage(stage="bogus", user_content="x")


# ---------------------------------------------------------------------------
# 4. check.py session state 管理
# ---------------------------------------------------------------------------
class CheckSessionStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "sessions")

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_then_load_roundtrip(self):
        session = check_mod.create_session(self.dir, user_content="hello")
        sid = session["session_id"]
        self.assertTrue(os.path.isdir(self.dir))
        loaded = check_mod.load_session(self.dir, sid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["user_content"], "hello")
        self.assertIsNone(loaded["stage1_output"])
        self.assertEqual(loaded["mode"], "check")

    def test_save_stage1_output_updates_session(self):
        sid = check_mod.create_session(
            self.dir, user_content="x"
        )["session_id"]
        check_mod.save_stage1_output(self.dir, sid, "stage1 result text")
        loaded = check_mod.load_session(self.dir, sid)
        self.assertEqual(loaded["stage1_output"], "stage1 result text")

    def test_load_session_missing_returns_none(self):
        self.assertIsNone(
            check_mod.load_session(self.dir, "deadbeef-dead-beef-dead")
        )

    def test_load_session_rejects_illegal_id(self):
        # `../` 路徑注入嘗試必須被拒
        self.assertIsNone(
            check_mod.load_session(self.dir, "../../etc/passwd")
        )

    def test_load_session_corrupted_file_returns_none(self):
        sid = check_mod.create_session(self.dir, user_content="x")["session_id"]
        path = os.path.join(
            self.dir, f"check_{sid}.json"
        )
        with open(path, "w") as f:
            f.write("{ this is not valid json")
        self.assertIsNone(check_mod.load_session(self.dir, sid))

    def test_find_latest_session_by_mtime(self):
        s1 = check_mod.create_session(self.dir, user_content="first")
        time.sleep(0.02)  # 確保 mtime 不同
        s2 = check_mod.create_session(self.dir, user_content="second")
        latest = check_mod.find_latest_session(self.dir)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["session_id"], s2["session_id"])

    def test_find_latest_session_empty_dir(self):
        os.makedirs(self.dir, exist_ok=True)
        self.assertIsNone(check_mod.find_latest_session(self.dir))

    def test_find_latest_session_nonexistent_dir(self):
        self.assertIsNone(
            check_mod.find_latest_session(
                os.path.join(self.tmp.name, "nope")
            )
        )

    def test_find_latest_skips_corrupted_files(self):
        s1 = check_mod.create_session(self.dir, user_content="good")
        # 放一個壞檔——mtime 較新，但內容無效
        bad_path = os.path.join(self.dir, "check_abcdef12-aaaa-bbbb-cccc.json")
        with open(bad_path, "w") as f:
            f.write("garbage")
        time.sleep(0.02)
        os.utime(bad_path, None)  # 觸發 mtime 更新
        latest = check_mod.find_latest_session(self.dir)
        # 應該跳過壞檔，回傳好檔
        self.assertIsNotNone(latest)
        self.assertEqual(latest["session_id"], s1["session_id"])

    def test_discard_session(self):
        sid = check_mod.create_session(self.dir, user_content="x")["session_id"]
        self.assertTrue(check_mod.discard_session(self.dir, sid))
        self.assertIsNone(check_mod.load_session(self.dir, sid))

    def test_discard_session_missing_returns_false(self):
        self.assertFalse(
            check_mod.discard_session(self.dir, "deadbeef-dead-beef-dead")
        )

    def test_sweep_removes_old_sessions(self):
        # 建一個 session，然後把它的 created_at 手動改成 25 小時前
        session = check_mod.create_session(self.dir, user_content="old")
        sid = session["session_id"]
        path = os.path.join(self.dir, f"check_{sid}.json")
        session["created_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=25)
        ).isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False)

        # 新的 session
        new_s = check_mod.create_session(self.dir, user_content="fresh")

        removed = check_mod.sweep_old_sessions(self.dir, ttl_hours=24)
        self.assertEqual(removed, 1)
        # 新的還在
        self.assertIsNotNone(check_mod.load_session(self.dir, new_s["session_id"]))
        # 舊的被刪
        self.assertIsNone(check_mod.load_session(self.dir, sid))

    def test_sweep_tolerates_missing_directory(self):
        self.assertEqual(
            check_mod.sweep_old_sessions(
                os.path.join(self.tmp.name, "nothere")
            ),
            0,
        )


# ---------------------------------------------------------------------------
# 5. run_check_mode + run_check_followup 端到端
# ---------------------------------------------------------------------------
class RunCheckModeFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )
        self.session_dir = os.path.join(self.tmp.name, "sessions")
        self.client = LLMClient()  # fallback mode

    def tearDown(self):
        self.tmp.cleanup()

    def _run_mode(self, user_input: str, **kwargs):
        from src.main import run_check_mode
        with redirect_stdout(io.StringIO()) as out:
            ok = run_check_mode(
                user_input, self.store, self.client,
                session_dir=self.session_dir,
                **kwargs,
            )
        return ok, out.getvalue()

    def _run_followup(self, choice: str, **kwargs):
        from src.main import run_check_followup
        with redirect_stdout(io.StringIO()) as out:
            ok = run_check_followup(
                choice, self.store, self.client,
                session_dir=self.session_dir,
                **kwargs,
            )
        return ok, out.getvalue()

    def test_stage1_creates_session_with_stage1_output_saved(self):
        ok, stdout = self._run_mode("投資保證獲利 LINE")
        self.assertTrue(ok)
        # 沒寫入 endpoints（PR-D 策略：第一階段不寫端點）
        self.assertEqual(len(self.store.all()), 0)
        # 有寫入 session
        session = check_mod.find_latest_session(self.session_dir)
        self.assertIsNotNone(session)
        self.assertEqual(session["user_content"], "投資保證獲利 LINE")
        self.assertIsNotNone(session["stage1_output"])
        # stdout 有印摘要與選單提示
        self.assertIn("第一階段", stdout)
        self.assertIn("1", stdout)

    def test_meta_query_does_not_enter_check(self):
        ok, stdout = self._run_mode("你是什麼")
        self.assertFalse(ok)
        # 沒建 session、沒寫端點
        self.assertIsNone(check_mod.find_latest_session(self.session_dir))
        self.assertEqual(len(self.store.all()), 0)
        self.assertIn("逗福Tofu", stdout)

    def test_followup_choice_1_writes_endpoints_with_mode_check(self):
        self._run_mode("一則可疑訊息")
        ok, stdout = self._run_followup("1")
        self.assertTrue(ok)

        rows = self.store.all()
        starts = [r for r in rows if r.get("type") == "start"]
        ends = [r for r in rows if r.get("type") == "end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(starts[0]["start_data"]["mode"], "check")
        self.assertEqual(ends[0]["end_data"]["check_final_stage"], STAGE_FULL)
        # Session 已清除
        self.assertIsNone(
            check_mod.find_latest_session(self.session_dir)
        )

    def test_followup_choice_2_saves_warning_stage(self):
        self._run_mode("可疑訊息")
        ok, stdout = self._run_followup("2")
        self.assertTrue(ok)
        self.assertIn("警示", stdout)
        ed = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]["end_data"]
        self.assertEqual(ed["check_final_stage"], STAGE_WARNING)

    def test_followup_choice_3_ends_without_llm_call(self):
        self._run_mode("訊息")
        # 用 mock 驗證 run_check_stage 在第三選項時不會被呼叫
        with patch.object(
            self.client, "run_check_stage", side_effect=AssertionError(
                "選 3 不該再呼叫 LLM"
            )
        ):
            with redirect_stdout(io.StringIO()) as out:
                from src.main import run_check_followup
                ok = run_check_followup(
                    "3", self.store, self.client,
                    session_dir=self.session_dir,
                )
        self.assertTrue(ok)
        self.assertIn("結束", out.getvalue())
        # 端點仍然寫入（視為一次完整 session）
        ed = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]["end_data"]
        # stage=end 的結束句
        self.assertIn("結束本次分析", ed["result"])

    def test_followup_without_active_session_prints_hint(self):
        # 沒跑過 /check 就直接選 1
        ok, stdout = self._run_followup("1")
        self.assertFalse(ok)
        self.assertIn("找不到", stdout)
        self.assertEqual(len(self.store.all()), 0)


# ---------------------------------------------------------------------------
# 6. baseline 與 /check 的隔離
# ---------------------------------------------------------------------------
class CheckEndpointsExcludedFromBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )
        self.session_dir = os.path.join(self.tmp.name, "sessions")
        self.client = LLMClient()

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_mode_endpoints_not_in_baseline_statistics(self):
        """跑兩次 /check 後 baseline 應該看不到這些端點。"""
        from src.main import run_check_mode, run_check_followup
        for msg in ["詐騙 LINE", "可疑的投資簡訊"]:
            with redirect_stdout(io.StringIO()):
                run_check_mode(
                    msg, self.store, self.client,
                    session_dir=self.session_dir,
                )
                run_check_followup(
                    "1", self.store, self.client,
                    session_dir=self.session_dir,
                )
        baseline = baseline_mod.compute_baseline(self.store.all())
        # 兩筆都是 check mode，不應計入 total_events
        self.assertEqual(baseline["total_events"], 0)


# ---------------------------------------------------------------------------
# 7. dispatcher 路由測試
# ---------------------------------------------------------------------------
class CheckDispatcherTests(unittest.TestCase):
    def test_help_lists_check_without_coming_soon(self):
        from src.main import HELP_ITEMS
        check_line = next(
            (desc for cmd, desc in HELP_ITEMS if cmd.startswith("/check")),
            None,
        )
        self.assertIsNotNone(check_line)
        self.assertNotIn("即將上線", check_line)
        self.assertIn("ISF", check_line)

    def test_mode_commands_still_include_check(self):
        from src.main import MODE_COMMANDS
        self.assertIn("check", MODE_COMMANDS)


# ---------------------------------------------------------------------------
# 8. Boot-time sweep：main 啟動時清舊 session
# ---------------------------------------------------------------------------
class BootTimeSweepTests(unittest.TestCase):
    def test_main_startup_sweeps_old_sessions(self):
        """MOMO review 小建議：main 啟動時應該呼叫一次 sweep_old_sessions。"""
        import src.main as main_mod
        # monkey-patch sweep 記錄被叫幾次；patch main loop 讓它立刻 return
        calls = []

        def fake_sweep(session_dir, **kwargs):
            calls.append(session_dir)
            return 0

        with patch.object(
            check_mod, "sweep_old_sessions", side_effect=fake_sweep
        ):
            # 讓 main loop 立刻以 /exit 離開
            with patch("builtins.input", side_effect=["/exit"]):
                with redirect_stdout(io.StringIO()):
                    main_mod.main()
        # sweep 應該被叫至少一次，且目標是 SESSIONS_DIR
        self.assertTrue(len(calls) >= 1)
        self.assertEqual(calls[0], main_mod.SESSIONS_DIR)


if __name__ == "__main__":
    unittest.main()
