"""v0.6 PR-B：五模式開關 + mode 欄位 + dispatcher + 4 份 output prompt 測試。

對應 `docs/specs/逗福_v0_6_開發_Spec_v2.md` §1.2-1.7。

這些測試只驗證 PR-B scope 內的行為：
- OUTPUT_PROMPT_* 常數存在且語氣正確
- execute_task 接 output_mode 並路由到對應 prompt
- _fallback_execute_task 依 output_mode 切換輸出結構
- run_one_interaction 以 mode='free'/'risk' 跳過 restate/confirm
- dispatcher 解析 /free /risk /propose /check
- /propose /check 印「即將上線」而不是「未知指令」
- start_data 帶 `mode` 欄位
- baseline 排除 mode == 'check' 的端點
- 內部術語禁止清單補上 retrieved / top_k

`/propose` 的多輪 loop 與 `/check` 的 ISF 三階段在 PR-C / PR-D 才會實作，
本檔不涵蓋。
"""

from __future__ import annotations

import builtins
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.api.claude_client import (
    OUTPUT_PROMPT_DEFAULT,
    OUTPUT_PROMPT_FREE,
    OUTPUT_PROMPT_PROPOSE,
    OUTPUT_PROMPT_PROPOSE_FINAL,
    OUTPUT_PROMPT_RISK,
    OUTPUT_PROMPTS,
    VALID_OUTPUT_MODES,
    LLMClient,
    _fallback_execute_task,
)
from src.middleware import baseline as baseline_mod
from src.middleware.endpoint import EndpointStore, new_event_id


# ---------------------------------------------------------------------------
# 1. OUTPUT_PROMPT_* 常數與模式表
# ---------------------------------------------------------------------------
class OutputPromptConstantTests(unittest.TestCase):
    def test_four_output_prompts_exist(self):
        # PR-B scope: default / free / risk / propose + PROPOSE_FINAL 支援 PR-C
        self.assertIn("default", OUTPUT_PROMPTS)
        self.assertIn("free", OUTPUT_PROMPTS)
        self.assertIn("risk", OUTPUT_PROMPTS)
        self.assertIn("propose", OUTPUT_PROMPTS)
        self.assertIn("propose_final", OUTPUT_PROMPTS)

    def test_valid_output_modes_includes_all_prompts(self):
        # /check 不在這裡——check 模式繞過 execute_task
        self.assertNotIn("check", VALID_OUTPUT_MODES)
        for key in ("default", "free", "risk", "propose", "propose_final"):
            self.assertIn(key, VALID_OUTPUT_MODES)

    def test_default_prompt_still_has_original_rules(self):
        # 保留現有的「4-8 句」「48 小時內可做的事」等關鍵規則，
        # 避免 PR-B 的重構退化 default 模式。
        self.assertIn("4～8 句", OUTPUT_PROMPT_DEFAULT)
        self.assertIn("48 小時", OUTPUT_PROMPT_DEFAULT)
        self.assertIn("你在跟人類說話", OUTPUT_PROMPT_DEFAULT)

    def test_free_prompt_keeps_fallback_question_escape_hatch(self):
        # MOMO 明確要求：OUTPUT_PROMPT_FREE 不能變成「什麼都硬答」，
        # 要保留「遇到完全無法推論的才問 1 題」的彈性。
        self.assertIn("直接給建議", OUTPUT_PROMPT_FREE)
        self.assertIn("不要反問", OUTPUT_PROMPT_FREE)
        # 彈性例外條款必須存在
        self.assertIn("彈性例外", OUTPUT_PROMPT_FREE)
        # 對「完全沒主題」的情況仍允許問
        self.assertTrue(
            "連基本前提都缺" in OUTPUT_PROMPT_FREE
            or "完全寫不出" in OUTPUT_PROMPT_FREE,
            "OUTPUT_PROMPT_FREE 必須保留「資訊極度不足才問」的退路",
        )

    def test_free_prompt_requires_personalization_fallback(self):
        """MOMO review：/free 對新使用者不能變成『隨便找幾家』。

        畫像為空時要明確退回「常見選項」並標註「非個人化」，
        避免 /free 把逗福的核心價值（根據畫像與端點記憶推薦）抽掉。
        """
        self.assertIn("個人化原則", OUTPUT_PROMPT_FREE)
        # 必須同時涵蓋「有畫像優先用畫像」與「沒畫像退回常見選項」兩邊
        self.assertIn("優先使用", OUTPUT_PROMPT_FREE)
        self.assertTrue(
            "畫像為空" in OUTPUT_PROMPT_FREE
            or "記憶不存在" in OUTPUT_PROMPT_FREE
            or "新使用者" in OUTPUT_PROMPT_FREE,
            "必須明確處理『畫像為空』的情況",
        )
        self.assertIn("非個人化", OUTPUT_PROMPT_FREE)

    def test_risk_prompt_requires_falsification_format(self):
        self.assertIn("風險清單", OUTPUT_PROMPT_RISK)
        self.assertIn("觸發條件", OUTPUT_PROMPT_RISK)
        self.assertIn("falsification", OUTPUT_PROMPT_RISK)
        # 要求具體 X，禁止空泛
        self.assertIn("可觀察", OUTPUT_PROMPT_RISK)

    def test_risk_prompt_word_budget_is_per_risk_not_total(self):
        """MOMO review：250 字整份上限會擠得太緊，改成每項 80-150 字。

        整份不設硬上限但要求精簡；寧可少一項紮實，也不為湊 5 項稀釋品質。
        """
        # 不再有「全文不超過 250 字」這種整份硬上限
        self.assertNotIn("全文不超過 250 字", OUTPUT_PROMPT_RISK)
        self.assertNotIn("全文不超過 250", OUTPUT_PROMPT_RISK)
        # 每項 80-150 字的篇幅規則必須存在
        self.assertIn("每項風險", OUTPUT_PROMPT_RISK)
        self.assertIn("80-150", OUTPUT_PROMPT_RISK)
        # 整份不設硬上限、要求精簡
        self.assertTrue(
            "不設硬上限" in OUTPUT_PROMPT_RISK or "整份" in OUTPUT_PROMPT_RISK,
            "需要說明整份篇幅的處理方式",
        )

    def test_propose_prompt_has_submit_now_meta(self):
        self.assertIn("[PROPOSE_META]", OUTPUT_PROMPT_PROPOSE)
        self.assertIn("submit_now", OUTPUT_PROMPT_PROPOSE)
        self.assertIn("缺的關鍵問題", OUTPUT_PROMPT_PROPOSE)

    def test_propose_prompt_does_not_ask_for_tentative_proposal_per_round(self):
        """MOMO review：前 1-5 輪不該有「暫定提案」段，留給 PROPOSE_FINAL。

        前幾輪的結構應該是「本輪焦點 → 缺的關鍵問題 → META」，四段式提案
        （最終提案 / 已知 / 假設 / 風險）只在交卷輪出現。
        """
        # 前幾輪的 prompt 不該要求輸出「本輪暫定提案」
        self.assertNotIn("本輪暫定提案", OUTPUT_PROMPT_PROPOSE)
        # 必須明確指示「不要輸出完整提案」
        self.assertTrue(
            "不要" in OUTPUT_PROMPT_PROPOSE and "完整提案" in OUTPUT_PROMPT_PROPOSE,
            "必須明確指示收資訊輪不要輸出完整提案",
        )
        # 要交代職責分工：完整提案由 PROPOSE_FINAL / 交卷輪負責
        self.assertTrue(
            "PROPOSE_FINAL" in OUTPUT_PROMPT_PROPOSE
            or "交卷輪" in OUTPUT_PROMPT_PROPOSE,
            "必須說明完整提案留給交卷輪",
        )

    def test_propose_final_has_four_sections(self):
        self.assertIn("最終提案", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("已知資訊", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("自行假設", OUTPUT_PROMPT_PROPOSE_FINAL)
        self.assertIn("風險點", OUTPUT_PROMPT_PROPOSE_FINAL)


# ---------------------------------------------------------------------------
# 2. execute_task 接 output_mode
# ---------------------------------------------------------------------------
class ExecuteTaskOutputModeTests(unittest.TestCase):
    def test_execute_task_rejects_unknown_output_mode(self):
        client = LLMClient()
        with self.assertRaises(ValueError) as ctx:
            client.execute_task(
                goal="辦派對",
                motivation="",
                constraints=[],
                output_mode="nonsense",
            )
        self.assertIn("Unknown output_mode", str(ctx.exception))

    def test_fallback_default_mode_keeps_existing_structure(self):
        out = _fallback_execute_task(
            goal="辦派對",
            motivation="慶生",
            constraints=["30 人以下"],
            output_mode="default",
        )
        # 默認模式維持原 4 步結構
        self.assertIn("建議可以這樣進行", out)
        self.assertIn("反駁條件", out)

    def test_fallback_free_mode_produces_direct_advice(self):
        out = _fallback_execute_task(
            goal="找週末拉麵店",
            motivation="",
            constraints=[],
            output_mode="free",
        )
        self.assertIn("/free", out)
        # 明確假設標註
        self.assertIn("假設", out)
        # 不該含「補位問題」等默認模式用詞
        self.assertNotIn("建議可以這樣進行", out)

    def test_fallback_risk_mode_produces_risk_list(self):
        out = _fallback_execute_task(
            goal="投資某新創",
            motivation="",
            constraints=[],
            output_mode="risk",
        )
        self.assertIn("風險", out)
        self.assertIn("觸發條件", out)
        self.assertIn("整體評估", out)

    def test_fallback_propose_includes_meta_block(self):
        out = _fallback_execute_task(
            goal="辦一場研討會",
            motivation="",
            constraints=[],
            output_mode="propose",
        )
        self.assertIn("[PROPOSE_META]", out)
        self.assertIn('"submit_now"', out)

    def test_fallback_propose_final_has_four_sections(self):
        out = _fallback_execute_task(
            goal="辦一場研討會",
            motivation="",
            constraints=[],
            output_mode="propose_final",
        )
        self.assertIn("最終提案", out)
        self.assertIn("已知資訊", out)
        self.assertIn("自行假設", out)
        self.assertIn("風險點", out)


# ---------------------------------------------------------------------------
# 3. execute_task system prompt 有正確注入 output prompt
# ---------------------------------------------------------------------------
class ExecuteTaskSystemPromptCompositionTests(unittest.TestCase):
    def _capture_system(self, client, **kwargs):
        """利用 monkey-patch `_call` 捕捉最終組裝出的 system prompt。"""
        captured: dict[str, str] = {}

        def fake_call(system, user, max_tokens=1024):
            captured["system"] = system
            captured["user"] = user
            return "[mock response]"

        client.fallback_mode = False
        client._client = object()  # 任意 truthy，跳過 fallback
        with patch.object(client, "_call", side_effect=fake_call):
            client.execute_task(
                goal=kwargs.get("goal", "辦派對"),
                motivation=kwargs.get("motivation", ""),
                constraints=kwargs.get("constraints", []),
                strategy_brief=kwargs.get("strategy_brief"),
                encoded_top_k=kwargs.get("encoded_top_k"),
                output_mode=kwargs.get("output_mode", "default"),
            )
        return captured

    def test_default_mode_uses_default_prompt(self):
        client = LLMClient()
        cap = self._capture_system(client, output_mode="default")
        self.assertIn("default 補位模式", cap["system"])
        self.assertIn("4～8 句", cap["system"])

    def test_free_mode_uses_free_prompt(self):
        client = LLMClient()
        cap = self._capture_system(client, output_mode="free")
        self.assertIn("/free 模式", cap["system"])
        self.assertIn("直接給建議", cap["system"])

    def test_risk_mode_uses_risk_prompt(self):
        client = LLMClient()
        cap = self._capture_system(client, output_mode="risk")
        self.assertIn("/risk 模式", cap["system"])
        self.assertIn("風險", cap["system"])
        self.assertIn("falsification", cap["system"])

    def test_propose_mode_uses_propose_prompt(self):
        client = LLMClient()
        cap = self._capture_system(client, output_mode="propose")
        self.assertIn("/propose 模式", cap["system"])
        self.assertIn("[PROPOSE_META]", cap["system"])

    def test_forbidden_list_includes_retrieved_and_top_k(self):
        """Spec v2 §2.2：forbidden list 補 retrieved、top_k。"""
        client = LLMClient()
        cap = self._capture_system(client, output_mode="default")
        self.assertIn("retrieved", cap["system"])
        self.assertIn("top_k", cap["system"])
        # 既有的也還在
        self.assertIn("密碼表", cap["system"])
        self.assertIn("CODEBOOK", cap["system"])

    def test_forbidden_list_applies_to_all_modes(self):
        client = LLMClient()
        for mode in ("default", "free", "risk", "propose"):
            cap = self._capture_system(client, output_mode=mode)
            self.assertIn("retrieved", cap["system"], f"mode={mode} 缺 retrieved")
            self.assertIn("top_k", cap["system"], f"mode={mode} 缺 top_k")
            self.assertIn("內部術語隔離", cap["system"], f"mode={mode} 缺隔離區塊")


# ---------------------------------------------------------------------------
# 4. generate_restate 的 forbidden list 也要補上（Spec v2 §2.2）
# ---------------------------------------------------------------------------
class RestateForbiddenListTests(unittest.TestCase):
    def test_generate_restate_forbids_internal_terms(self):
        captured: dict[str, str] = {}

        def fake_call(system, user, max_tokens=1024):
            captured["system"] = system
            return '{"restate_text": "", "gap_questions": [], "gap_categories": []}'

        client = LLMClient()
        client.fallback_mode = False
        client._client = object()
        with patch.object(client, "_call", side_effect=fake_call):
            with patch.object(client, "_extract_json", return_value={
                "restate_text": "", "gap_questions": [], "gap_categories": []
            }):
                client.generate_restate(
                    user_input="辦派對",
                    baseline_summary={},
                    allowed_categories=["budget"],
                )
        system = captured["system"]
        self.assertIn("內部術語隔離", system)
        self.assertIn("retrieved", system)
        self.assertIn("top_k", system)
        self.assertIn("密碼表", system)


# ---------------------------------------------------------------------------
# 5. _parse_mode_command dispatcher 解析
# ---------------------------------------------------------------------------
class ParseModeCommandTests(unittest.TestCase):
    def test_parse_free_with_content(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/free 幫我找拉麵店"),
            ("free", "幫我找拉麵店"),
        )

    def test_parse_free_with_no_content(self):
        from src.main import _parse_mode_command
        self.assertEqual(_parse_mode_command("/free"), ("free", ""))

    def test_parse_risk_with_content(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/risk 投資某新創"),
            ("risk", "投資某新創"),
        )

    def test_parse_propose_with_content(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/propose 辦研討會"),
            ("propose", "辦研討會"),
        )

    def test_parse_check_with_content(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/check 請匯款 NT$50000"),
            ("check", "請匯款 NT$50000"),
        )

    def test_parse_returns_none_for_non_mode_commands(self):
        from src.main import _parse_mode_command
        self.assertIsNone(_parse_mode_command("/help"))
        self.assertIsNone(_parse_mode_command("/baseline"))
        self.assertIsNone(_parse_mode_command("辦派對"))
        self.assertIsNone(_parse_mode_command(""))
        self.assertIsNone(_parse_mode_command("/"))
        self.assertIsNone(_parse_mode_command("/bogus abc"))

    def test_parse_is_case_insensitive_on_command_name(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/FREE 幫我規劃"),
            ("free", "幫我規劃"),
        )

    def test_parse_preserves_content_with_slashes_and_colons(self):
        from src.main import _parse_mode_command
        self.assertEqual(
            _parse_mode_command("/check https://evil.com/抽獎"),
            ("check", "https://evil.com/抽獎"),
        )


# ---------------------------------------------------------------------------
# 6. mode 欄位寫入 start_data + run_one_interaction 的 mode 分流
# ---------------------------------------------------------------------------
class RunOneInteractionModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "endpoints.json")
        self.store = EndpointStore(self.path)
        self.client = LLMClient()  # fallback mode（無 API key）

    def tearDown(self):
        self.tmp.cleanup()

    def _consume_input(self, responses):
        it = iter(responses)
        return patch.object(builtins, "input", lambda prompt="": next(it))

    def test_default_mode_writes_mode_field(self):
        from src.main import run_one_interaction
        with self._consume_input(["預算三萬"]):
            run_one_interaction("辦派對", self.store, self.client)
        starts = [r for r in self.store.all() if r["type"] == "start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["start_data"]["mode"], "default")

    def test_free_mode_skips_restate_and_user_input(self):
        from src.main import run_one_interaction
        # free 模式不應觸發 input() ——若觸發，這裡會拋 StopIteration
        stdout = io.StringIO()
        with patch.object(builtins, "input", side_effect=AssertionError(
            "free 模式不該呼叫 input()"
        )):
            with redirect_stdout(stdout):
                ok = run_one_interaction(
                    "找週末拉麵店",
                    self.store,
                    self.client,
                    mode="free",
                )
        self.assertTrue(ok)
        starts = [r for r in self.store.all() if r["type"] == "start"]
        self.assertEqual(len(starts), 1)
        sd = starts[0]["start_data"]
        self.assertEqual(sd["mode"], "free")
        self.assertEqual(sd["goal"], "找週末拉麵店")
        self.assertEqual(sd["user_confirmation"], "auto_free")
        # 沒有 gap_questions（不反問）
        self.assertEqual(sd.get("gap_questions"), [])
        # CLI 有印 [/free] 提示
        self.assertIn("/free", stdout.getvalue())

    def test_risk_mode_skips_restate_and_user_input(self):
        from src.main import run_one_interaction
        with patch.object(builtins, "input", side_effect=AssertionError(
            "risk 模式不該呼叫 input()"
        )):
            with redirect_stdout(io.StringIO()):
                ok = run_one_interaction(
                    "投資某新創",
                    self.store,
                    self.client,
                    mode="risk",
                )
        self.assertTrue(ok)
        starts = [r for r in self.store.all() if r["type"] == "start"]
        self.assertEqual(len(starts), 1)
        sd = starts[0]["start_data"]
        self.assertEqual(sd["mode"], "risk")
        self.assertEqual(sd["user_confirmation"], "auto_risk")

    def test_unsupported_mode_raises(self):
        from src.main import run_one_interaction
        with self.assertRaises(ValueError):
            run_one_interaction(
                "x", self.store, self.client, mode="propose",
            )
        with self.assertRaises(ValueError):
            run_one_interaction(
                "x", self.store, self.client, mode="check",
            )
        with self.assertRaises(ValueError):
            run_one_interaction(
                "x", self.store, self.client, mode="bogus",
            )


# ---------------------------------------------------------------------------
# 7. baseline 排除 mode == 'check' 的端點
# ---------------------------------------------------------------------------
class BaselineExcludesCheckModeTests(unittest.TestCase):
    def _make_start(self, goal, mode):
        return {
            "type": "start",
            "endpoint_id": f"E-{goal}",
            "event_id": new_event_id(),
            "timestamp": "2026-04-16T00:00:00+00:00",
            "start_data": {
                "user_input": goal,
                "goal": goal,
                "gap_categories": ["budget"],
                "constraints": [],
                "user_confirmation": "confirmed",
                "mode": mode,
            },
        }

    def test_check_mode_endpoints_excluded_from_gap_counter(self):
        rows = [
            self._make_start("辦派對", "default"),
            self._make_start("買禮物", "default"),
            self._make_start("這封信是詐騙嗎", "check"),  # 應被排除
            self._make_start("這網站安全嗎", "check"),    # 應被排除
        ]
        baseline = baseline_mod.compute_baseline(rows)
        # 兩個 default 端點納入，兩個 check 端點被排除
        self.assertEqual(baseline["total_events"], 2)
        self.assertIn("budget", baseline["top_gap_categories"])
        # _raw 內含實際 counter，驗證 budget 被計次兩次而不是四次
        raw_gap_counter = baseline.get("_raw", {}).get("gap_cat") or {}
        self.assertEqual(raw_gap_counter.get("budget"), 2)

    def test_endpoints_without_mode_still_counted(self):
        # 舊端點沒有 mode 欄位，預設視為 default，要繼續納入統計
        rows = [self._make_start("辦派對", "default")]
        rows[0]["start_data"].pop("mode")  # 模擬舊端點
        baseline = baseline_mod.compute_baseline(rows)
        self.assertEqual(baseline["total_events"], 1)


# ---------------------------------------------------------------------------
# 8. PR-G step 2：default / free 的迴避句型重試整合測試
# ---------------------------------------------------------------------------
class _RetryingScriptedClient:
    """測試用 client，第一次 execute_task 回迴避、第二次回乾淨（或再迴避）。

    實作 run_one_interaction 需要的完整介面。
    """

    fallback_mode = False

    def __init__(self, execute_responses: list[str]):
        self._execute_responses = list(execute_responses)
        self.execute_calls: list[dict] = []

    def execute_task(self, **kwargs) -> str:
        self.execute_calls.append(dict(kwargs))
        if self._execute_responses:
            return self._execute_responses.pop(0)
        return "（responses exhausted）"

    def generate_restate(self, user_input: str, **kwargs):
        # /free 模式不走 restate，此 method 不會被呼叫，但留著作為介面完整性
        return {"restate_text": user_input, "gap_questions": [], "gap_categories": []}

    def merge_confirmation(self, **kwargs):
        return {}

    def analyze_deviation(self, **kwargs):
        return ""

    def answer_with_profile(self, **kwargs):
        return ""


class FreeModeEvasionRetryTests(unittest.TestCase):
    """/free 模式：第一次回應含迴避句型 → 自動重寫 → 結果採用乾淨版。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EndpointStore(
            os.path.join(self.tmp.name, "endpoints.json")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_free_mode_evasive_triggers_retry(self):
        from src.main import run_one_interaction
        client = _RetryingScriptedClient(execute_responses=[
            "你可以思考一下你想去哪裡。",  # 迴避
            "推薦日本東北十和田湖，NT$9 萬 10 天。",  # 乾淨
        ])
        with patch.object(builtins, "input", side_effect=AssertionError(
            "free 不該呼叫 input"
        )):
            with redirect_stdout(io.StringIO()):
                run_one_interaction(
                    "規劃旅行", self.store, client, mode="free",
                )
        # 兩次 execute_task：第一次偵測到迴避、第二次重寫
        self.assertEqual(len(client.execute_calls), 2)
        # 第二次應帶 retry hint
        self.assertIn("上一次輸出", client.execute_calls[1]["extra_context"])
        # 最終端點 result 是乾淨版
        end = [r for r in self.store.all() if r["type"] == "end"][0]
        self.assertIn("十和田湖", end["end_data"]["result"])
        self.assertNotIn("你可以思考", end["end_data"]["result"])
        self.assertFalse(end["end_data"]["evasion_retry_degraded"])

    def test_free_mode_double_evasive_marks_degraded(self):
        from src.main import run_one_interaction
        from src.middleware.response_validator import DEGRADED_WARNING_PREFIX
        client = _RetryingScriptedClient(execute_responses=[
            "你可以思考一下你想去哪裡。",
            "建議你先確認預算再決定。",  # 第二次還是迴避
        ])
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_one_interaction(
                    "規劃旅行", self.store, client, mode="free",
                )
        end = [r for r in self.store.all() if r["type"] == "end"][0]
        result = end["end_data"]["result"]
        self.assertTrue(result.startswith(DEGRADED_WARNING_PREFIX))
        self.assertTrue(end["end_data"]["evasion_retry_degraded"])

    def test_free_mode_clean_no_retry(self):
        from src.main import run_one_interaction
        client = _RetryingScriptedClient(execute_responses=[
            "推薦日本東北十和田湖，NT$9 萬 10 天。",
        ])
        with patch.object(builtins, "input", side_effect=AssertionError):
            with redirect_stdout(io.StringIO()):
                run_one_interaction(
                    "規劃旅行", self.store, client, mode="free",
                )
        # 只一次 execute_task
        self.assertEqual(len(client.execute_calls), 1)
        end = [r for r in self.store.all() if r["type"] == "end"][0]
        self.assertFalse(end["end_data"]["evasion_retry_degraded"])


if __name__ == "__main__":
    unittest.main()
