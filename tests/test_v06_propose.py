"""v0.6 PR-C：/propose 5 輪自問自答 + 交卷 + meta 解析測試。

對應 `docs/specs/逗福_v0_6_開發_Spec_v2.md` §1.5。

範圍：
- parse_propose_meta：JSON 區塊解析、code fence、型別驗證、巢狀括號
- format_propose_history：跨輪紀錄格式化
- run_propose_mode：5 輪 loop、submit_now=true 提前交卷、第 6 輪強制交卷
- 端點寫入：start + end + 每輪 black_box，共用 event_id
- guardrail 上限中斷：保留已完成輪次並強制進交卷
- dispatcher：/propose <內容> 路由到 run_propose_mode；空內容印 usage 提示
"""

from __future__ import annotations

import builtins
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import patch

from src.api.claude_client import LLMClient, _fallback_execute_task
from src.middleware.endpoint import EndpointStore
from src.middleware.guardrails import GuardrailTracker, GuardrailTripped
from src.middleware.propose import (
    format_propose_history,
    parse_propose_meta,
)


# ---------------------------------------------------------------------------
# 1. parse_propose_meta
# ---------------------------------------------------------------------------
class ProposeMetaParserTests(unittest.TestCase):
    def test_parses_basic_submit_true(self):
        text = """本輪焦點：場地人數確認
[PROPOSE_META]
{"submit_now": true, "round": 3, "reason": "已收齊關鍵變數"}
"""
        meta = parse_propose_meta(text)
        self.assertEqual(
            meta,
            {"submit_now": True, "round": 3, "reason": "已收齊關鍵變數"},
        )

    def test_parses_basic_submit_false(self):
        text = """[PROPOSE_META]
{"submit_now": false, "round": 2, "reason": "still need budget"}"""
        meta = parse_propose_meta(text)
        self.assertIsNotNone(meta)
        self.assertFalse(meta["submit_now"])
        self.assertEqual(meta["round"], 2)

    def test_parses_json_inside_markdown_code_fence(self):
        text = """[PROPOSE_META]
```json
{"submit_now": true, "round": 5, "reason": "max rounds reached"}
```
"""
        meta = parse_propose_meta(text)
        self.assertIsNotNone(meta)
        self.assertTrue(meta["submit_now"])
        self.assertEqual(meta["round"], 5)

    def test_returns_none_when_marker_absent(self):
        self.assertIsNone(parse_propose_meta("just plain text"))
        self.assertIsNone(parse_propose_meta(""))

    def test_returns_none_for_malformed_json(self):
        text = '[PROPOSE_META]\n{"submit_now": true, "round":, "reason": "x"}'
        self.assertIsNone(parse_propose_meta(text))

    def test_returns_none_when_marker_present_but_no_braces(self):
        text = "[PROPOSE_META]\nsubmit_now = true"
        self.assertIsNone(parse_propose_meta(text))

    def test_rejects_non_bool_submit_now(self):
        text = '[PROPOSE_META]\n{"submit_now": "yes", "round": 1, "reason": "x"}'
        self.assertIsNone(parse_propose_meta(text))

    def test_rejects_bool_as_round(self):
        # isinstance(True, int) 是 True，parser 要明確排除這種狀況
        text = '[PROPOSE_META]\n{"submit_now": true, "round": true, "reason": "x"}'
        self.assertIsNone(parse_propose_meta(text))

    def test_rejects_non_int_round(self):
        text = '[PROPOSE_META]\n{"submit_now": true, "round": "3", "reason": "x"}'
        self.assertIsNone(parse_propose_meta(text))

    def test_rejects_non_string_reason(self):
        text = '[PROPOSE_META]\n{"submit_now": true, "round": 1, "reason": 42}'
        self.assertIsNone(parse_propose_meta(text))

    def test_rejects_missing_field(self):
        text = '[PROPOSE_META]\n{"submit_now": true, "round": 1}'
        self.assertIsNone(parse_propose_meta(text))

    def test_tolerates_braces_in_reason_string(self):
        # 字串內的 { } 不應該弄亂括號計數
        text = (
            '[PROPOSE_META]\n'
            '{"submit_now": true, "round": 4, "reason": "payload {a:1, b:2}"}'
        )
        meta = parse_propose_meta(text)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["reason"], "payload {a:1, b:2}")

    def test_tolerates_escaped_quotes_in_reason(self):
        text = (
            '[PROPOSE_META]\n'
            '{"submit_now": false, "round": 2, "reason": "they said \\"ok\\""}'
        )
        meta = parse_propose_meta(text)
        self.assertIsNotNone(meta)
        self.assertIn("ok", meta["reason"])


# ---------------------------------------------------------------------------
# 2. format_propose_history
# ---------------------------------------------------------------------------
class FormatProposeHistoryTests(unittest.TestCase):
    def test_empty_list_returns_empty_string(self):
        self.assertEqual(format_propose_history([]), "")

    def test_single_round(self):
        rounds = [{"round": 1, "text": "第一輪內容", "meta": {}}]
        out = format_propose_history(rounds)
        self.assertIn("第 1 輪", out)
        self.assertIn("第一輪內容", out)

    def test_multiple_rounds_preserve_order(self):
        rounds = [
            {"round": 1, "text": "A", "meta": {}},
            {"round": 2, "text": "B", "meta": {}},
            {"round": 3, "text": "C", "meta": {}},
        ]
        out = format_propose_history(rounds)
        pos_a = out.find("A")
        pos_b = out.find("B")
        pos_c = out.find("C")
        self.assertLess(pos_a, pos_b)
        self.assertLess(pos_b, pos_c)

    def test_skips_rounds_with_empty_text(self):
        rounds = [
            {"round": 1, "text": "actual content", "meta": {}},
            {"round": 2, "text": "", "meta": {}},
            {"round": 3, "text": "   ", "meta": {}},
        ]
        out = format_propose_history(rounds)
        self.assertIn("actual content", out)
        # 只有一輪有內容
        self.assertEqual(out.count("### 第"), 1)

    # -- v0.6 PR-G 壓縮版測試（MOMO 2026-04-16 bug report 根治修法 Step 1）---
    def test_compresses_to_focus_and_assumptions_when_structured(self):
        """LLM 輸出有標準「本輪焦點 / 暫定假設」標籤時，壓縮到只保留這兩段。

        效果：5 輪 × 全文 → 5 輪 × 一句焦點 + bullet 假設，大幅減少 Haiku
        注意力稀釋的機會（MOMO Bug 1 根因）。
        """
        body = (
            "**本輪焦點**：預算維度\n\n"
            "**缺的關鍵問題**：\n"
            "- 問題：上限多少？\n"
            "- 重要性：改變住宿層級\n"
            "- 暫定假設：NT$10 萬\n\n"
            "- 問題：分攤方式？\n"
            "- 重要性：影響結帳流程\n"
            "- 暫定假設：個人全額\n\n"
            "這裡有一大段不重要的廢話，像是對前輪的總結、換句話說、"
            "以及其他裝飾性文字，壓縮後不該出現。\n\n"
            "[PROPOSE_META]\n"
            '{"submit_now": false, "round": 1, "reason": "x"}'
        )
        out = format_propose_history([{"round": 1, "text": body}])
        # 焦點跟假設必須在
        self.assertIn("預算維度", out)
        self.assertIn("NT$10 萬", out)
        self.assertIn("個人全額", out)
        # 其他廢話 / 原問題細節 / meta 區塊都不該在
        self.assertNotIn("裝飾性文字", out)
        self.assertNotIn("換句話說", out)
        self.assertNotIn("上限多少", out)
        self.assertNotIn("PROPOSE_META", out)

    def test_falls_back_to_excerpt_when_structure_missing(self):
        """若 LLM 輸出沒有「本輪焦點 / 暫定假設」標籤，退回前 200 字摘要
        而非完全丟棄——避免下一輪失去所有上下文。
        """
        body = "這是沒結構的輸出" * 5
        out = format_propose_history([{"round": 1, "text": body}])
        self.assertIn("第 1 輪", out)
        self.assertIn("未解析到結構", out)
        # 原文片段有進來
        self.assertIn("這是沒結構的輸出", out)

    def test_strips_meta_block_even_in_fallback(self):
        """Fallback 摘要也應去掉 `[PROPOSE_META]` 區塊，避免 JSON 擾亂下輪 LLM。"""
        body = (
            "零結構文字\n\n"
            "[PROPOSE_META]\n"
            '{"submit_now": false, "round": 1, "reason": "foo"}'
        )
        out = format_propose_history([{"round": 1, "text": body}])
        self.assertNotIn("PROPOSE_META", out)
        self.assertNotIn("submit_now", out)
        self.assertIn("零結構文字", out)

    def test_extracts_multiple_assumptions(self):
        """有多組「暫定假設」時全部抽出，dedup 並保持順序。"""
        body = (
            "本輪焦點：XXX\n"
            "- 暫定假設：A\n"
            "- 暫定假設：B\n"
            "- 暫定假設：A\n"  # 重複
            "- 暫定假設：C\n"
        )
        out = format_propose_history([{"round": 1, "text": body}])
        self.assertIn("A", out)
        self.assertIn("B", out)
        self.assertIn("C", out)
        # dedup：A 只出現一次在 bullet 清單裡
        a_lines = [
            ln for ln in out.split("\n")
            if ln.strip().startswith("- A") or ln.strip() == "- A"
        ]
        self.assertEqual(len(a_lines), 1)


# ---------------------------------------------------------------------------
# 3. run_propose_mode 整合測試（fallback client，無 API key）
# ---------------------------------------------------------------------------
class RunProposeModeFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )
        self.client = LLMClient()  # fallback mode（無 API key）
        self.assertTrue(self.client.fallback_mode)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_no_input(self, user_input: str, **kwargs):
        """跑 run_propose_mode 並確保不會要求 stdin。"""
        from src.main import run_propose_mode
        with patch.object(builtins, "input", side_effect=AssertionError(
            "propose 模式不該呼叫 input()"
        )):
            with redirect_stdout(io.StringIO()) as out:
                ok = run_propose_mode(
                    user_input, self.store, self.client, **kwargs
                )
        return ok, out.getvalue()

    def test_propose_runs_all_5_rounds_with_fallback(self):
        # fallback 模式的 propose 輸出固定 submit_now=false，應該跑滿 5 輪
        ok, _ = self._run_no_input("規劃一場 50 人的發表會")
        self.assertTrue(ok)

        rows = self.store.all()
        starts = [r for r in rows if r.get("type") == "start"]
        ends = [r for r in rows if r.get("type") == "end"]
        blackboxes = self.store.black_boxes()

        # 整個 session 只有一對 start/end
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        # 共用 event_id
        self.assertEqual(starts[0]["event_id"], ends[0]["event_id"])

        sd = starts[0]["start_data"]
        self.assertEqual(sd["mode"], "propose")
        self.assertEqual(sd["goal"], "規劃一場 50 人的發表會")
        self.assertEqual(sd["user_confirmation"], "auto_propose")

        ed = ends[0]["end_data"]
        self.assertIn("result", ed)
        self.assertEqual(ed["propose_rounds"], 5)  # fallback 跑滿 5 輪
        self.assertFalse(ed["propose_submitted_early"])
        self.assertFalse(ed["propose_interrupted_by_guardrail"])
        # 跑滿 5 輪 → 第 6 輪交卷
        self.assertEqual(ed["propose_final_round_number"], 6)

        # 每輪都有 black_box
        my_bbs = [
            b for b in blackboxes
            if b.get("event_id") == starts[0]["event_id"]
        ]
        self.assertEqual(len(my_bbs), 5)
        # 檢查輪次連續
        rounds_seen = sorted(
            (b.get("black_box") or {}).get("round") for b in my_bbs
        )
        self.assertEqual(rounds_seen, [1, 2, 3, 4, 5])

    def test_propose_final_text_contains_four_sections(self):
        ok, stdout_text = self._run_no_input("辦一場 30 人生日派對")
        self.assertTrue(ok)
        ed = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]["end_data"]
        final = ed["result"]
        # fallback 的 propose_final 模板含四段標題
        self.assertIn("最終提案", final)
        self.assertIn("已知資訊", final)
        self.assertIn("自行假設", final)
        self.assertIn("風險點", final)

    def test_propose_stdout_shows_round_progress(self):
        _, stdout_text = self._run_no_input("辦派對")
        # CLI 應該印出每輪標題
        self.assertIn("round 1/5", stdout_text)
        self.assertIn("round 5/5", stdout_text)
        self.assertIn("交卷", stdout_text)

    def test_meta_query_does_not_enter_propose(self):
        from src.main import run_propose_mode
        with patch.object(builtins, "input", side_effect=AssertionError(
            "meta 攔截時不該走 propose 流程"
        )):
            with redirect_stdout(io.StringIO()) as out:
                ok = run_propose_mode(
                    "你是什麼", self.store, self.client,
                )
        self.assertFalse(ok)
        # 沒寫入任何端點
        self.assertEqual(len(self.store.all()), 0)
        self.assertIn("逗福Tofu", out.getvalue())


# ---------------------------------------------------------------------------
# 4. submit_now=true 提前交卷（需要能控制 execute_task 輸出的 mock client）
# ---------------------------------------------------------------------------
class _ScriptedClient:
    """測試用的假 client，依輪數回傳不同的輸出。

    v4.0 P0-1：預設 final_response 更新為 ATL-3 gate 相容版本（含產出物、
    時間窗、驗收條件），避免每個 propose_final mock 都要重複寫這些欄位。
    """

    fallback_mode = False

    def __init__(
        self,
        responses: list[str],
        final_response: str = (
            "### 1. 最終提案\n"
            "48 小時內產出 1 份完整方案，至少包含 3 個可驗證步驟。\n\n"
            "### 2. 已知資訊\n無\n\n"
            "### 3. 自行假設\n無\n\n"
            "### 4. 風險點\n無"
        ),
    ):
        self._responses = list(responses)
        self._final = final_response
        self.calls: list[dict] = []

    def execute_task(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        mode = kwargs.get("output_mode")
        if mode == "propose_final":
            return self._final
        if self._responses:
            return self._responses.pop(0)
        return "（no more scripted responses）"

    def generate_restate(self, **kwargs):  # pragma: no cover - 不會被呼叫
        return {}

    def merge_confirmation(self, **kwargs):  # pragma: no cover
        return {}

    def analyze_deviation(self, **kwargs):  # pragma: no cover
        return ""

    def answer_with_profile(self, **kwargs):  # pragma: no cover
        return ""


class _MultiFinalScriptedClient(_ScriptedClient):
    """延伸版：propose_final 可以回傳一系列不同輸出（用於測 evasion retry）。

    第 N 次收到 output_mode='propose_final' 時，回第 N 個 final_responses；
    用完後回最後一個。用來模擬「第一次迴避、第二次重寫後乾淨」的情境。
    """

    def __init__(
        self,
        responses: list[str],
        final_responses: list[str],
    ):
        super().__init__(responses=responses, final_response="")
        self._final_queue = list(final_responses)
        self._final_default = (
            final_responses[-1] if final_responses else "[最終提案]"
        )

    def execute_task(self, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        mode = kwargs.get("output_mode")
        if mode == "propose_final":
            if self._final_queue:
                return self._final_queue.pop(0)
            return self._final_default
        if self._responses:
            return self._responses.pop(0)
        return "（no more scripted responses）"


class ProposeEarlySubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_submit_now_true_stops_loop_early(self):
        from src.main import run_propose_mode
        responses = [
            # Round 1: submit_now=false
            '本輪焦點：A\n[PROPOSE_META]\n'
            '{"submit_now": false, "round": 1, "reason": "need more"}',
            # Round 2: submit_now=true → 提前交卷
            '本輪焦點：B\n[PROPOSE_META]\n'
            '{"submit_now": true, "round": 2, "reason": "enough"}',
            # Round 3+ 不該被呼叫
            "should not be called",
            "should not be called",
            "should not be called",
        ]
        client = _ScriptedClient(
            responses=responses,
            final_response=(
                "### 1. 最終提案\n"
                "48 小時內產出 1 份方案，至少含 3 個可驗證步驟。\n\n"
                "### 2. 已知資訊\n無\n\n"
                "### 3. 自行假設\n無\n\n"
                "### 4. 風險點\n無"
            ),
        )
        with patch.object(builtins, "input", side_effect=AssertionError(
            "不該問 input"
        )):
            with redirect_stdout(io.StringIO()):
                ok = run_propose_mode("辦派對", self.store, client)
        self.assertTrue(ok)

        # execute_task 應該被呼叫 3 次：2 輪 round + 1 次 propose_final
        modes_called = [c.get("output_mode") for c in client.calls]
        self.assertEqual(modes_called, ["propose", "propose", "propose_final"])

        ed = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]["end_data"]
        self.assertEqual(ed["propose_rounds"], 2)
        self.assertTrue(ed["propose_submitted_early"])
        # 第 2 輪提前交卷 → 第 3 輪是交卷輪
        self.assertEqual(ed["propose_final_round_number"], 3)

    def test_extra_context_accumulates_across_rounds(self):
        from src.main import run_propose_mode
        responses = [
            '輪 1 內容\n[PROPOSE_META]\n'
            '{"submit_now": false, "round": 1, "reason": "r1"}',
            '輪 2 內容\n[PROPOSE_META]\n'
            '{"submit_now": true, "round": 2, "reason": "r2"}',
        ]
        client = _ScriptedClient(responses=responses)
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("任務", self.store, client)

        # 第 1 輪 extra_context 應該是 None（沒前輪）
        self.assertIsNone(client.calls[0].get("extra_context"))
        # 第 2 輪 extra_context 應包含第 1 輪的內容
        ec_round2 = client.calls[1].get("extra_context") or ""
        self.assertIn("第 1 輪", ec_round2)
        self.assertIn("輪 1 內容", ec_round2)
        # 交卷輪應包含第 1 輪與第 2 輪
        ec_final = client.calls[2].get("extra_context") or ""
        self.assertIn("輪 1 內容", ec_final)
        self.assertIn("輪 2 內容", ec_final)


# ---------------------------------------------------------------------------
# 4b. MOMO 2026-04-16 bug report 整合回歸測試
# ---------------------------------------------------------------------------
# 這組測試用 scripted client 模擬 Haiku 輸出，端到端驗證：
# - Bug 1：run_propose_mode 的 context 累積機制（prompt 層已修，這裡驗 plumbing）
# - Bug 2：final 輪的輸出**原文**被寫進 end_data.result，不被截斷或轉寫
# - Bug 3：dispatcher 層 /propose 空參數走 usage hint，不落到「未知指令」
class ProposeBugReportRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_bug1_all_five_rounds_history_reaches_final_submit(self):
        """Bug 1 回歸：跑滿 5 輪時，交卷輪的 extra_context 要含**全部** 5 輪焦點。

        MOMO 實測第 2、3 輪幾乎一模一樣 / 第 4、5 輪幾乎一模一樣，
        表示 Haiku 沒看到完整前面輪次——此測試鎖死「5 輪都有傳下去」的 plumbing。

        v0.6 PR-G：format_propose_history 改壓縮版後，extra_context 只含
        焦點 + 假設 bullet，不含原文。此測試驗「焦點」字串有到。
        """
        from src.main import run_propose_mode
        # 5 輪都 submit_now=false → 跑滿 5 輪 + 第 6 輪交卷
        # 使用標準「本輪焦點」標籤，驗證壓縮路徑
        responses = [
            f'**本輪焦點**：維度_{i}\n'
            f'- 暫定假設：假設_{i}\n\n'
            f'[PROPOSE_META]\n'
            f'{{"submit_now": false, "round": {i}, "reason": "r{i}"}}'
            for i in range(1, 6)
        ]
        client = _ScriptedClient(responses=responses)
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃旅行", self.store, client)

        # 應該 5 輪 propose + 1 輪 propose_final，共 6 次呼叫
        modes = [c.get("output_mode") for c in client.calls]
        self.assertEqual(
            modes, ["propose"] * 5 + ["propose_final"]
        )
        # 交卷輪（第 6 次呼叫）的 extra_context 必須含全部 5 輪焦點與假設
        ec_final = client.calls[5].get("extra_context") or ""
        for i in range(1, 6):
            self.assertIn(f"維度_{i}", ec_final, msg=f"交卷輪缺第 {i} 輪焦點")
            self.assertIn(f"假設_{i}", ec_final, msg=f"交卷輪缺第 {i} 輪假設")
            self.assertIn(f"第 {i} 輪", ec_final, msg=f"交卷輪缺第 {i} 輪 header")

    def test_bug1_each_round_sees_all_previous_rounds(self):
        """Bug 1 回歸：每一輪的 extra_context 要含**到上一輪為止**的所有歷史。

        這是防止「第 3 輪只看到第 2 輪，忘了第 1 輪」的回歸。
        v0.6 PR-G：用標準「本輪焦點」標籤驗壓縮路徑的歷史累積。
        """
        from src.main import run_propose_mode
        responses = [
            f'**本輪焦點**：維度_{i}\n'
            f'- 暫定假設：假設_{i}\n\n'
            f'[PROPOSE_META]\n'
            f'{{"submit_now": false, "round": {i}, "reason": "x"}}'
            for i in range(1, 6)
        ]
        client = _ScriptedClient(responses=responses)
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("任務", self.store, client)

        # 第 N 輪（index N-1）的 extra_context 應含 維度_1..維度_(N-1)
        for n in range(1, 6):
            ec = client.calls[n - 1].get("extra_context")
            if n == 1:
                self.assertIsNone(ec, "第 1 輪不該有前輪 context")
                continue
            ec_text = ec or ""
            for prev in range(1, n):
                self.assertIn(
                    f"維度_{prev}", ec_text,
                    msg=f"第 {n} 輪 extra_context 缺維度_{prev}",
                )
            # 不該看到尚未發生的後續輪
            for future in range(n, 6):
                self.assertNotIn(
                    f"維度_{future}", ec_text,
                    msg=f"第 {n} 輪 extra_context 不該含未來的維度_{future}",
                )

    def test_bug1_compressed_history_excludes_raw_body(self):
        """Bug 1 根治驗證（PR-G）：壓縮後的 extra_context **不該含原文廢話**。

        原文（壓縮前會進 history 的全文）裡的裝飾性文字、換句話說、重複敘述
        不應進入下一輪 Haiku 的上下文。只保留焦點 + 假設 bullet。
        """
        from src.main import run_propose_mode
        responses = [
            '**本輪焦點**：預算\n'
            '這段是裝飾性的廢話文字，不該進入壓縮後的 history。\n'
            '甚至可能有換句話說、對前輪的無意義總結...\n'
            '- 暫定假設：NT$10 萬\n\n'
            '[PROPOSE_META]\n'
            '{"submit_now": false, "round": 1, "reason": "x"}',
            '**本輪焦點**：地點\n'
            '- 暫定假設：東京\n\n'
            '[PROPOSE_META]\n'
            '{"submit_now": true, "round": 2, "reason": "enough"}',
        ]
        client = _ScriptedClient(responses=responses)
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃旅行", self.store, client)

        # 第 2 輪 extra_context 應該**不含**第 1 輪的裝飾性原文
        ec_round2 = client.calls[1].get("extra_context") or ""
        self.assertIn("預算", ec_round2)
        self.assertIn("NT$10 萬", ec_round2)
        self.assertNotIn("裝飾性的廢話", ec_round2)
        self.assertNotIn("換句話說", ec_round2)
        self.assertNotIn("PROPOSE_META", ec_round2)

    def test_bug2_final_response_is_written_to_end_data_verbatim(self):
        """Bug 2 回歸：scripted client 交卷輪輸出的原文要**完整**寫進 end_data.result。

        MOMO 回饋「最終提案欄位內容：我建議你先從東北亞深度文化體驗類型中選出至少三種」
        這種偽裝成提案的反問（prompt 層已禁）。此測試鎖死：若 Haiku 交的是一份
        真正的具體提案，它會**原樣**進端點紀錄，不被程式碼截斷或重寫。
        """
        from src.main import run_propose_mode
        final_text = (
            "### 1. 最終提案\n"
            "建議日本東北 6 天 + 首爾 4 天，預算 NT$9 萬，4/25-5/4 出發。"
            "Day 1-3 仙台 / Day 4-6 青森 / Day 7-10 首爾。先訂 4/25 台北→仙台機票。\n\n"
            "### 2. 已知資訊清單\n- 10 天 / 東北亞偏好 / 預算不緊\n\n"
            "### 3. 自行假設資訊清單\n- 假設預算 NT$9 萬\n\n"
            "### 4. 風險點清單\n- 櫻花季機票漲價"
        )
        responses = [
            '早交\n[PROPOSE_META]\n'
            '{"submit_now": true, "round": 1, "reason": "enough"}',
        ]
        client = _ScriptedClient(
            responses=responses, final_response=final_text
        )
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃 10 天東北亞旅行", self.store, client)

        end_row = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]
        # 交卷原文 must 100% 落在 result 裡（含具體地名、預算、Day 安排）
        result = end_row["end_data"]["result"]
        self.assertEqual(result, final_text)
        self.assertIn("仙台", result)
        self.assertIn("NT$9 萬", result)
        self.assertIn("Day 1-3", result)

    def test_propose_final_evasive_triggers_retry(self):
        """PR-G step 2：交卷若含迴避句型 → 自動重寫，最終結果不含迴避。"""
        from src.main import run_propose_mode
        responses = [
            '**本輪焦點**：需求\n'
            '- 暫定假設：A\n\n'
            '[PROPOSE_META]\n'
            '{"submit_now": true, "round": 1, "reason": "enough"}',
        ]
        # 第一次交卷含「我建議你先從」（偽裝成提案的反問）
        evasive_final = (
            "### 1. 最終提案\n"
            "我建議你先從三個方向中選出一個感興趣的。\n\n"
            "### 2. 已知資訊\n無\n\n"
            "### 3. 自行假設\n無\n\n"
            "### 4. 風險點\n無"
        )
        clean_final = (
            "### 1. 最終提案\n"
            "48 小時內產出 1 份完整行程，至少含 3 個可驗證節點：\n"
            "建議日本東北 6 天：Day 1-3 仙台、Day 4-6 青森，預算 NT$9 萬。\n\n"
            "### 2. 已知資訊\n- 10 天\n\n"
            "### 3. 自行假設\n- NT$9 萬\n\n"
            "### 4. 風險點\n- 櫻花季漲價"
        )
        client = _MultiFinalScriptedClient(
            responses=responses,
            final_responses=[evasive_final, clean_final],
        )
        # validator 只在非 fallback 時觸發；確保 client.fallback_mode = False
        self.assertFalse(client.fallback_mode)
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃旅行", self.store, client)

        # 應該有 2 次 propose_final 呼叫（第 1 次偵測到迴避、第 2 次重寫）
        final_calls = [
            c for c in client.calls if c.get("output_mode") == "propose_final"
        ]
        self.assertEqual(len(final_calls), 2)
        # 第 2 次呼叫的 extra_context 應含重寫 hint（「上一次輸出」）
        self.assertIn("上一次輸出", final_calls[1]["extra_context"])

        end_row = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]
        # 最終 result 應該是乾淨的那份（不含「我建議你先從」）
        self.assertIn("仙台", end_row["end_data"]["result"])
        self.assertNotIn("我建議你先從", end_row["end_data"]["result"])
        # 未降級（第二次成功）
        self.assertFalse(end_row["end_data"]["evasion_retry_degraded"])

    def test_propose_final_double_evasive_marks_degraded(self):
        """PR-G step 2：連續兩次都迴避 → 結果前綴 [逗福警告]，flag 設 True。"""
        from src.main import run_propose_mode
        from src.middleware.response_validator import DEGRADED_WARNING_PREFIX
        responses = [
            '**本輪焦點**：需求\n'
            '- 暫定假設：A\n\n'
            '[PROPOSE_META]\n'
            '{"submit_now": true, "round": 1, "reason": "enough"}',
        ]
        evasive_a = "### 1. 最終提案\n你可以思考一下你想去哪。\n\n### 2.\n### 3.\n### 4."
        evasive_b = "### 1. 最終提案\n建議你先確認預算上限。\n\n### 2.\n### 3.\n### 4."
        client = _MultiFinalScriptedClient(
            responses=responses,
            final_responses=[evasive_a, evasive_b],
        )
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃旅行", self.store, client)

        end_row = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]
        result = end_row["end_data"]["result"]
        self.assertTrue(
            result.startswith(DEGRADED_WARNING_PREFIX),
            f"降級 prefix 未加在最前面：{result[:50]!r}",
        )
        self.assertTrue(end_row["end_data"]["evasion_retry_degraded"])

    def test_propose_final_clean_no_retry(self):
        """PR-G step 2：交卷輸出乾淨 → 不觸發重試，只一次 propose_final。"""
        from src.main import run_propose_mode
        responses = [
            '**本輪焦點**：需求\n'
            '- 暫定假設：A\n\n'
            '[PROPOSE_META]\n'
            '{"submit_now": true, "round": 1, "reason": "enough"}',
        ]
        clean_final = (
            "### 1. 最終提案\n"
            "48 小時內產出 1 份方案，至少含 3 個可驗證步驟：\n"
            "建議日本東北 6 天，預算 NT$9 萬。\n\n"
            "### 2. 已知資訊\n無\n\n"
            "### 3. 自行假設\n無\n\n"
            "### 4. 風險點\n無"
        )
        client = _ScriptedClient(
            responses=responses, final_response=clean_final
        )
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_propose_mode("規劃旅行", self.store, client)

        final_calls = [
            c for c in client.calls if c.get("output_mode") == "propose_final"
        ]
        # 只有一次 propose_final
        self.assertEqual(len(final_calls), 1)
        end_row = [
            r for r in self.store.all() if r.get("type") == "end"
        ][0]
        self.assertFalse(end_row["end_data"]["evasion_retry_degraded"])

    def test_bug3_dispatcher_empty_propose_prints_usage_hint(self):
        """Bug 3 回歸：dispatcher 層 /propose 空參數 → 印 usage hint。

        MOMO 回饋「第一次輸入『/propose』…實際回應『未知指令』」。
        驗 main() loop 看到 `/propose` 後不走「未知指令」分支。
        """
        from src import main as main_mod

        # 準備：讓 main() 走「提示 usage hint → 下一輪看到 /exit → 返回 0」
        inputs = iter(["/propose", "/exit"])

        def fake_input(prompt=""):
            try:
                return next(inputs)
            except StopIteration:
                return "/exit"

        buf = io.StringIO()
        with patch.object(builtins, "input", fake_input):
            with patch("src.main.EndpointStore") as MockStore:
                # 用真的 store，但指向 tmp
                MockStore.side_effect = lambda *a, **kw: EndpointStore(
                    os.path.join(self.tmp.name, "e.json")
                )
                with redirect_stdout(buf):
                    try:
                        main_mod.main()
                    except SystemExit:
                        pass
        output = buf.getvalue()
        # 必須含 /propose 專用的 usage hint（而非「未知指令」）
        self.assertIn("/propose 需要在指令後面接提案主題", output)
        self.assertNotIn("未知指令", output)


# ---------------------------------------------------------------------------
# 5. guardrail 中斷
# ---------------------------------------------------------------------------
class ProposeGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )
        self.client = LLMClient()  # fallback mode

    def tearDown(self):
        self.tmp.cleanup()

    def test_guardrail_pretrip_blocks_entire_session(self):
        """session 開始前就觸發護欄 → 不該寫任何端點。"""
        from src.main import run_propose_mode
        guardrails = GuardrailTracker()
        # 把 cost 推到上限，使 check_before_call 直接拋
        guardrails.cumulative_cost_usd = 999.0

        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                ok = run_propose_mode(
                    "任務", self.store, self.client,
                    guardrails=guardrails,
                )
        self.assertFalse(ok)
        self.assertEqual(len(self.store.all()), 0)


# ---------------------------------------------------------------------------
# 6. dispatcher 路由測試
# ---------------------------------------------------------------------------
class ProposeDispatcherTests(unittest.TestCase):
    def test_help_lists_propose_without_coming_soon(self):
        from src.main import HELP_ITEMS
        propose_line = next(
            (desc for cmd, desc in HELP_ITEMS if cmd.startswith("/propose")),
            None,
        )
        self.assertIsNotNone(propose_line)
        # 之前的「即將上線」標註應該移除
        self.assertNotIn("即將上線", propose_line)
        # 新版說明應該提到「5 輪」或「交卷」
        self.assertTrue("5 輪" in propose_line or "交卷" in propose_line)

    def test_dispatcher_constants_still_include_propose(self):
        # dispatcher 的解析集合還是要有 propose
        from src.main import MODE_COMMANDS
        self.assertIn("propose", MODE_COMMANDS)

    def test_parse_propose_empty_args_returns_tuple_with_empty_content(self):
        # MOMO 2026-04-16 bug report Bug 3：/propose（只打指令沒內容）
        # 應該給 usage hint 而非『未知指令』。此測試 lock-in
        # _parse_mode_command 對空內容回傳 ("propose", "")，讓上層
        # dispatcher 走 usage hint 分支，而非落到最後的「未知指令」。
        from src.main import _parse_mode_command
        self.assertEqual(_parse_mode_command("/propose"), ("propose", ""))
        # 多餘空格也不會讓它掉出 MODE 分支（變成未知指令）
        self.assertEqual(_parse_mode_command("/propose "), ("propose", ""))
        self.assertEqual(_parse_mode_command("/PROPOSE"), ("propose", ""))


# ---------------------------------------------------------------------------
# 8. Output prompt 文字規格（MOMO 2026-04-16 bug report 回歸測試）
# ---------------------------------------------------------------------------
# 這組測試不呼叫 LLM，只驗證 prompt 常數有包含「禁止迴避句型 / 要求具體內容」
# 這類關鍵指示——避免未來 refactor 不小心刪掉，讓 Haiku 又退回到
# 「我教你思考」的方法論模式。
class OutputPromptContentTests(unittest.TestCase):
    def test_default_prompt_forbids_methodology_avoidance(self):
        from src.api.claude_client import OUTPUT_PROMPT_DEFAULT
        # 必須含「直接給答案」類要求
        self.assertIn("幫我做", OUTPUT_PROMPT_DEFAULT)
        self.assertIn("具體", OUTPUT_PROMPT_DEFAULT)
        # 必須禁止「你可以思考 / 搜尋」類推諉句型
        self.assertIn("你可以思考", OUTPUT_PROMPT_DEFAULT)
        self.assertIn("搜尋引擎", OUTPUT_PROMPT_DEFAULT)

    def test_free_prompt_forbids_methodology_avoidance(self):
        from src.api.claude_client import OUTPUT_PROMPT_FREE
        self.assertIn("幫我做", OUTPUT_PROMPT_FREE)
        # /free 特別強調：禁止「你可以思考」與「搜尋引擎」推諉
        self.assertIn("你可以思考", OUTPUT_PROMPT_FREE)
        self.assertIn("搜尋引擎", OUTPUT_PROMPT_FREE)
        # 也要明確點出「確實不知道時直接說不確定、不要包裝成方法論」
        self.assertIn("不確定", OUTPUT_PROMPT_FREE)

    def test_propose_prompt_forbids_round_repetition(self):
        from src.api.claude_client import OUTPUT_PROMPT_PROPOSE
        # MOMO 2026-04-16 bug report Bug 1：輪次重複
        # prompt 必須明確要求讀前幾輪紀錄 + 列出已涵蓋維度 + 本輪挑未涵蓋維度
        self.assertIn("前面已經涵蓋", OUTPUT_PROMPT_PROPOSE)
        self.assertIn("未涵蓋", OUTPUT_PROMPT_PROPOSE)
        self.assertIn("嚴禁重複", OUTPUT_PROMPT_PROPOSE)
        # 也要給模型一個「主要維度都碰過了就直接交卷」的出口
        self.assertIn("submit_now:true", OUTPUT_PROMPT_PROPOSE)

    def test_propose_final_prompt_requires_concrete_plan(self):
        from src.api.claude_client import OUTPUT_PROMPT_PROPOSE_FINAL
        # MOMO 2026-04-16 bug report Bug 2：最終提案沒交出具體方案
        # prompt 必須明確要求「具體地名/店家/預算/時段」
        self.assertIn("具體地名", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("預算區間", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("時段安排", OUTPUT_PROMPT_PROPOSE_FINAL)
        # 必須禁止「我建議你先從 X 選出...」這種偽裝成提案的反問
        self.assertIn("我建議你先從", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("推延", OUTPUT_PROMPT_PROPOSE_FINAL)


if __name__ == "__main__":
    unittest.main()
