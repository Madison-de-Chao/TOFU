"""claude_client 的 mock 測試。

覆蓋：
- JSON 解析（正常、code fence、非法格式）
- Fallback 模式（無 API key 時的確定性回應）
- Retry 機制（失敗重試直到成功或達上限）
- Rate limit 處理（429 重試）
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.claude_client import (
    BASE_BACKOFF_SECONDS,
    LLMClient,
    LLMClientError,
    MAX_RETRIES,
    _fallback_execute_task,
    _fallback_merge_confirmation,
    _is_rate_limit_error,
    _retry_after_seconds,
)


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class JSONParsingTests(unittest.TestCase):
    def test_parse_plain_json(self):
        data = LLMClient._extract_json('{"a": 1, "b": "hi"}')
        self.assertEqual(data, {"a": 1, "b": "hi"})

    def test_parse_json_with_prefix_text(self):
        raw = "以下是 JSON：\n{\"a\": 1}\n謝謝"
        data = LLMClient._extract_json(raw)
        self.assertEqual(data, {"a": 1})

    def test_parse_code_fence(self):
        raw = "```json\n{\"x\": [1,2,3]}\n```"
        data = LLMClient._extract_json(raw)
        self.assertEqual(data, {"x": [1, 2, 3]})

    def test_parse_code_fence_without_lang(self):
        raw = "```\n{\"y\": true}\n```"
        data = LLMClient._extract_json(raw)
        self.assertEqual(data, {"y": True})

    def test_parse_empty_raises(self):
        with self.assertRaises(LLMClientError):
            LLMClient._extract_json("")

    def test_parse_no_braces_raises(self):
        with self.assertRaises(LLMClientError):
            LLMClient._extract_json("no json here")

    def test_parse_malformed_raises(self):
        with self.assertRaises(LLMClientError):
            LLMClient._extract_json("{\"a\": }")


class FallbackModeTests(unittest.TestCase):
    def setUp(self):
        # 確保環境沒有 API key
        self._saved = dict()
        for k in ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY"):
            if k in __import__("os").environ:
                self._saved[k] = __import__("os").environ.pop(k)

    def tearDown(self):
        import os
        for k, v in self._saved.items():
            os.environ[k] = v

    def test_no_api_key_enters_fallback(self):
        client = LLMClient(api_key=None)
        self.assertTrue(client.fallback_mode)

    def test_fallback_generate_restate_cold_start(self):
        client = LLMClient(api_key=None)
        result = client.generate_restate(
            user_input="辦一場生日派對",
            baseline_summary={},
            allowed_categories=[
                "budget", "time", "venue", "headcount", "deadline"
            ],
        )
        self.assertIn("restate_text", result)
        self.assertIn("生日派對", result["restate_text"])
        self.assertTrue(len(result["gap_questions"]) > 0)
        self.assertTrue(len(result["gap_categories"]) > 0)
        # cold start 應從通用清單選類別
        for cat in result["gap_categories"]:
            self.assertIn(
                cat, ["budget", "time", "venue", "headcount", "deadline"]
            )

    def test_fallback_restate_varies_with_baseline(self):
        """不同 baseline 應產生不同補位提問。"""
        client = LLMClient(api_key=None)
        categories_allowed = [
            "budget", "time", "venue", "audience", "deliverable", "stakeholder"
        ]
        r_cold = client.generate_restate(
            user_input="做一個展覽",
            baseline_summary={},
            allowed_categories=categories_allowed,
        )
        r_mature = client.generate_restate(
            user_input="做一個展覽",
            baseline_summary={
                "top_gap_categories": ["audience", "deliverable", "stakeholder"]
            },
            allowed_categories=categories_allowed,
        )
        self.assertNotEqual(
            sorted(r_cold["gap_categories"]),
            sorted(r_mature["gap_categories"]),
        )
        self.assertIn("audience", r_mature["gap_categories"])

    def test_fallback_merge_confirmation_extracts_constraints(self):
        client = LLMClient(api_key=None)
        merged = client.merge_confirmation(
            original_input="辦派對",
            restate_text="你想辦派對",
            user_confirmation_text="預算三萬，10 人，在家裡",
        )
        self.assertEqual(merged["goal"], "辦派對")
        self.assertEqual(len(merged["constraints"]), 3)
        # 應該辨識出 budget / headcount / venue
        self.assertIn("budget", merged["answered_categories"])

    def test_fallback_execute_task_returns_text(self):
        client = LLMClient(api_key=None)
        out = client.execute_task("辦派對", "慶生", ["預算三萬"])
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)
        self.assertIn("預算三萬", out)


class RetryMechanismTests(unittest.TestCase):
    """Mock API 呼叫，驗證 retry 行為。"""

    def _mk_client(self, mock_messages_create):
        """建立一個跳過 fallback 的 client，注入 mock messages.create。"""
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = mock_messages_create
        return client

    def test_successful_call_no_retry(self):
        mock_create = MagicMock(return_value=_FakeResponse('{"restate_text":"ok"}'))
        client = self._mk_client(mock_create)
        with patch("src.api.claude_client.time.sleep"):
            text = client._call("sys", "user")
        self.assertEqual(mock_create.call_count, 1)
        self.assertIn("restate_text", text)

    def test_retry_then_success(self):
        side_effects = [
            RuntimeError("network hiccup"),
            _FakeResponse('{"ok": true}'),
        ]
        mock_create = MagicMock(side_effect=side_effects)
        client = self._mk_client(mock_create)
        with patch("src.api.claude_client.time.sleep") as mock_sleep:
            text = client._call("sys", "user")
        self.assertEqual(mock_create.call_count, 2)
        # 第一次失敗後應該睡過一次
        self.assertEqual(mock_sleep.call_count, 1)
        self.assertIn("ok", text)

    def test_all_retries_fail_raises(self):
        mock_create = MagicMock(side_effect=RuntimeError("down"))
        client = self._mk_client(mock_create)
        with patch("src.api.claude_client.time.sleep"):
            with self.assertRaises(LLMClientError):
                client._call("sys", "user")
        self.assertEqual(mock_create.call_count, MAX_RETRIES)


class RateLimitTests(unittest.TestCase):
    def _mk_client(self, mock_create):
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = mock_create
        return client

    def test_429_triggers_retry(self):
        class RateLimitError(Exception):
            status_code = 429

        side_effects = [RateLimitError("429 too many requests"), _FakeResponse('{"ok":1}')]
        mock_create = MagicMock(side_effect=side_effects)
        client = self._mk_client(mock_create)
        with patch("src.api.claude_client.time.sleep") as mock_sleep:
            client._call("sys", "user")
        self.assertEqual(mock_create.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_message_string_429_detected(self):
        """只從訊息字串判斷 429 也要能重試。"""
        side_effects = [RuntimeError("HTTP 429 rate limit exceeded"), _FakeResponse('{}')]
        mock_create = MagicMock(side_effect=side_effects)
        client = self._mk_client(mock_create)
        with patch("src.api.claude_client.time.sleep"):
            client._call("sys", "user")
        self.assertEqual(mock_create.call_count, 2)


class GenerateRestateIntegrationTests(unittest.TestCase):
    """整個 generate_restate 流程（注入 mock API response）。"""

    @staticmethod
    def _make_api_client(api_text: str) -> LLMClient:
        mock_create = MagicMock(return_value=_FakeResponse(api_text))
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = mock_create
        return client

    def test_generate_restate_parses_api_response(self):
        fake_json = (
            '{"restate_text": "你想辦派對。預算、場地有想法嗎？",'
            ' "gap_questions": ["預算多少？","場地在哪？"],'
            ' "gap_categories": ["budget","venue"],'
            ' "inferred_goal": "辦派對",'
            ' "inferred_motivation": "慶生",'
            ' "inferred_constraints": ["10 人"]}'
        )
        client = self._make_api_client(fake_json)

        with patch("src.api.claude_client.time.sleep"):
            out = client.generate_restate(
                "辦派對",
                baseline_summary={},
                allowed_categories=["budget", "venue"],
            )
        self.assertEqual(out["gap_categories"], ["budget", "venue"])
        self.assertEqual(out["inferred_goal"], "辦派對")
        self.assertEqual(out["inferred_constraints"], ["10 人"])

    def test_generate_restate_malformed_json_raises_llm_error(self):
        """P0-2A：API 回傳非 JSON 純文字 → generate_restate 應拋 LLMClientError。"""
        client = self._make_api_client("I cannot help with that")

        with patch("src.api.claude_client.time.sleep"):
            with self.assertRaises(LLMClientError):
                client.generate_restate(
                    "辦派對",
                    baseline_summary={},
                    allowed_categories=["budget", "venue"],
                )

    def test_merge_confirmation_malformed_json_raises_llm_error(self):
        """P0-2B：API 回傳壞 JSON → merge_confirmation 應拋 LLMClientError。"""
        # 壞 JSON 但有 { }，會觸發 json.JSONDecodeError → 應被封裝成 LLMClientError
        client = self._make_api_client('{"goal": "辦派對", "constraints": [}')

        with patch("src.api.claude_client.time.sleep"):
            with self.assertRaises(LLMClientError) as ctx:
                client.merge_confirmation(
                    original_input="辦派對",
                    restate_text="你想辦派對",
                    user_confirmation_text="預算三萬",
                )
        # 確認沒有原始的 json.JSONDecodeError 外洩
        import json
        self.assertNotIsInstance(ctx.exception, json.JSONDecodeError)


class TwoStageInferenceTests(unittest.TestCase):
    """v3.0 修訂版（2026-04-12）：驗證 execute_task 注入畫像時採用修訂後的
    system prompt 指令，以及 construct_profile 的 prompt 涵蓋 LongMemEval
    兩階段接入規格要求的五個維度。"""

    @staticmethod
    def _make_api_client(api_text: str) -> LLMClient:
        mock_create = MagicMock(return_value=_FakeResponse(api_text))
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = mock_create
        return client

    def _captured_system(self, client: LLMClient) -> str:
        mock_create = client._client.messages.create
        mock_create.assert_called()
        kwargs = mock_create.call_args.kwargs
        return kwargs.get("system", "")

    def test_execute_task_profile_block_uses_attention_guidance_wording(self):
        """修訂版規格：execute_task 注入畫像時必須含「注意力引導」「從完整對話歷史
        中查找細節」「排除型偏好」三段新文字。"""
        client = self._make_api_client("建議內容")
        profile = {
            "communication_style": {
                "preference_expression": "implicit",
                "receiving_preference": "conversational",
            },
            "decision_style": {"context_need": "why_first", "pace": "deliberate"},
            "interest_map": {
                "preferences": [
                    {"item": "脫口秀", "type": "implicit"},
                    {"item": "商業化景點", "type": "exclusion"},
                ],
            },
        }
        with patch("src.api.claude_client.time.sleep"):
            client.execute_task(
                goal="推薦影劇", motivation="", constraints=[], user_profile=profile,
            )
        system = self._captured_system(client)
        # 注意力引導語
        self.assertIn("注意力引導", system)
        self.assertIn("從完整對話歷史中查找細節", system)
        # 新用語：排除型偏好（取代舊「負面偏好」）
        self.assertIn("排除型偏好", system)
        self.assertNotIn("負面偏好（exclusion 類）必須", system)
        # 既有的「基於偏好做新推薦」仍保留
        self.assertIn("基於偏好做新推薦", system)

    def test_execute_task_profile_block_skipped_without_profile(self):
        """未傳入 user_profile 時，system prompt 不應含畫像區塊。"""
        client = self._make_api_client("建議內容")
        with patch("src.api.claude_client.time.sleep"):
            client.execute_task(goal="辦派對", motivation="", constraints=[])
        system = self._captured_system(client)
        self.assertNotIn("使用者畫像", system)
        self.assertNotIn("注意力引導", system)

    def test_construct_profile_prompt_covers_five_dimensions(self):
        """LongMemEval 兩階段接入規格（v1.0）與畫像引擎（v0.2）要求 construct_profile
        prompt 需涵蓋溝通風格 / 決策模式 / 興趣領域 / 偏好清單 / 時間線索引。"""
        client = self._make_api_client('{"communication_style": {}}')
        with patch("src.api.claude_client.time.sleep"):
            client.construct_profile("一段完整 haystack")
        system = self._captured_system(client)
        self.assertIn("溝通風格", system)
        self.assertIn("決策模式", system)
        self.assertIn("興趣領域", system)
        self.assertIn("完整偏好清單", system)
        self.assertIn("時間線索引", system)
        # 排除偏好用詞（LongMemEval 規格用語）
        self.assertIn("排除偏好", system)
        # 明確要求不回答問題（v1.1 開頭句：「你的任務不是回答問題」）
        self.assertIn("不是回答問題", system)

    def test_construct_profile_uses_3000_max_tokens(self):
        """LongMemEval 接入規格明定第一階段 max_tokens=3000
        （v13 實測報告確認 3000 已足以涵蓋 preference / temporal-reasoning
        兩類 haystack，v1.1 維持不變）。"""
        client = self._make_api_client('{}')
        with patch("src.api.claude_client.time.sleep"):
            client.construct_profile("一段完整 haystack")
        kwargs = client._client.messages.create.call_args.kwargs
        self.assertEqual(kwargs.get("max_tokens"), 3000)

    def test_construct_profile_fallback_returns_empty(self):
        """fallback 模式下 construct_profile 不呼叫 API，回傳空 dict。"""
        client = LLMClient(api_key=None)
        self.assertTrue(client.fallback_mode)
        out = client.construct_profile("任何 haystack")
        self.assertEqual(out, {})

    # ------------------------------------------------------------------
    # v1.1 LongMemEval Adapter 規格新增測試
    # ------------------------------------------------------------------

    def test_answer_with_profile_fallback_returns_placeholder(self):
        """v1.1：fallback 模式下 answer_with_profile 回傳
        ``[fallback] 無法回答：...``，方便 evaluator 一眼辨識未跑到 LLM。"""
        client = LLMClient(api_key=None)
        self.assertTrue(client.fallback_mode)
        result = client.answer_with_profile(
            haystack="session 1 內容",
            question="使用者提過什麼偏好？",
            profile=None,
        )
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("[fallback] 無法回答："))
        self.assertIn("使用者提過什麼偏好？", result)

    def test_answer_with_profile_attaches_haystack_to_user_content(self):
        """v1.1：answer_with_profile 把 haystack 附在 user content 後面，
        system prompt 含 TOFU 身份、四條重要指引、輪廓區塊與時間線索引。"""
        client = self._make_api_client("使用者喜歡脫口秀")
        profile = {
            "communication_style": {"preference_expression": "implicit"},
            "interest_map": {
                "preferences": [
                    {"type": "implicit", "item": "脫口秀", "context": "session 3 反覆提到"},
                    {"type": "exclusion", "item": "商業化景點"},
                ],
            },
            "timeline": [
                "2024-03-15 使用者第一次提到脫口秀",
                "2024-04-02 使用者買了脫口秀門票",
            ],
        }
        with patch("src.api.claude_client.time.sleep"):
            client.answer_with_profile(
                haystack="Session 1：使用者說喜歡脫口秀。",
                question="使用者提過什麼娛樂偏好？",
                profile=profile,
            )
        mock_create = client._client.messages.create
        kwargs = mock_create.call_args.kwargs
        system = kwargs.get("system", "")
        messages = kwargs.get("messages", [])
        user_content = messages[0]["content"] if messages else ""

        # user content：問題 + 完整 haystack
        self.assertIn("問題：使用者提過什麼娛樂偏好？", user_content)
        self.assertIn("以下是完整的對話歷史", user_content)
        self.assertIn("Session 1", user_content)

        # system prompt：TOFU 身份 + 任務說明
        self.assertIn("逗福Tofu", system)
        self.assertIn("根據對話歷史回答使用者的問題", system)
        self.assertIn("不要編造", system)

        # 輪廓區塊
        self.assertIn("使用者輪廓（注意力引導）", system)
        self.assertIn("偏好表達方式：implicit", system)
        self.assertIn("[implicit] 脫口秀", system)
        self.assertIn("（session 3 反覆提到）", system)
        self.assertIn("[exclusion] 商業化景點", system)

        # 時間線索引
        self.assertIn("時間線索引", system)
        self.assertIn("2024-03-15 使用者第一次提到脫口秀", system)

        # 四條重要指引
        self.assertIn("完整對話歷史在下方提供", system)
        self.assertIn("從對話歷史中查找細節", system)
        self.assertIn("這些不一定在輪廓裡，但一定在對話歷史裡", system)
        self.assertIn("排除型偏好（exclusion 類）必須在回應中體現", system)

    def test_answer_with_profile_without_profile(self):
        """v1.1：未傳 profile 時仍應回答（只用 haystack），
        system prompt 不含「使用者輪廓」區塊。"""
        client = self._make_api_client("答案")
        with patch("src.api.claude_client.time.sleep"):
            client.answer_with_profile(
                haystack="Session 1：2024-03-15 的事。",
                question="什麼時候發生的？",
                profile=None,
            )
        kwargs = client._client.messages.create.call_args.kwargs
        system = kwargs.get("system", "")
        messages = kwargs.get("messages", [])
        user_content = messages[0]["content"] if messages else ""
        # 無 profile 時不含輪廓區塊
        self.assertNotIn("使用者輪廓", system)
        self.assertNotIn("時間線索引", system)
        # 但仍保留 TOFU 身份與 QA 任務規則
        self.assertIn("逗福Tofu", system)
        self.assertIn("根據對話歷史回答使用者的問題", system)
        # haystack 仍然附在 user content 後面
        self.assertIn("問題：什麼時候發生的？", user_content)
        self.assertIn("Session 1", user_content)

    def test_answer_with_profile_uses_500_max_tokens(self):
        """v1.1：answer_with_profile 的 max_tokens 為 500（規格明定）。"""
        client = self._make_api_client("答案")
        with patch("src.api.claude_client.time.sleep"):
            client.answer_with_profile(
                haystack="Session 1",
                question="Q?",
                profile=None,
            )
        kwargs = client._client.messages.create.call_args.kwargs
        self.assertEqual(kwargs.get("max_tokens"), 500)

    def test_try_repair_json_truncated(self):
        """v1.1：截斷的 JSON 可以被修復。"""
        client = LLMClient(api_key=None)
        truncated = '{"a": 1, "b": {"c": [1, 2, 3'
        result = client._try_repair_json(truncated)
        self.assertIsNotNone(result)
        self.assertEqual(result["a"], 1)

    def test_try_repair_json_valid(self):
        """v1.1：完整的 JSON 不需要修復，回傳 None。"""
        client = LLMClient(api_key=None)
        valid = '{"a": 1}'
        result = client._try_repair_json(valid)
        # 不是截斷問題，回傳 None
        self.assertIsNone(result)


class FallbackGapTests(unittest.TestCase):
    """P1-4：補 fallback 模式的邊界分支測試。"""

    def test_execute_task_no_constraints_branch(self):
        """無 constraints 時應輸出提醒使用者確認硬限制的文案。"""
        out = _fallback_execute_task(
            goal="測試目標",
            motivation="",
            constraints=[],
        )
        self.assertIn("確認是否還有沒列出來的硬限制", out)

    def test_execute_task_with_constraints_mentions_them(self):
        out = _fallback_execute_task(
            goal="辦派對",
            motivation="",
            constraints=["預算三萬", "10 人"],
        )
        self.assertIn("預算三萬", out)

    def test_merge_confirmation_drops_pure_negation_phrases(self):
        """P1-4：純否定短語不應進入 constraints。"""
        for neg in ["不對", "不是", "錯了", "wrong", "no"]:
            merged = _fallback_merge_confirmation(
                original_input="辦派對",
                restate_text="你想辦派對",
                user_confirmation_text=neg,
            )
            self.assertNotIn(
                neg,
                merged["constraints"],
                f"'{neg}' should not be in constraints",
            )


class RateLimitHelperTests(unittest.TestCase):
    """P0-3：直接測試 _is_rate_limit_error 與 _retry_after_seconds 的所有分支。"""

    def test_class_name_rate_limit_error_detected(self):
        """P0-3A：類別名為 RateLimitError 的例外應被識別為 rate limit。"""

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("too many")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_status_code_429_detected(self):
        class SomeApiError(Exception):
            status_code = 429

        self.assertTrue(_is_rate_limit_error(SomeApiError("x")))

    def test_not_rate_limit(self):
        self.assertFalse(_is_rate_limit_error(RuntimeError("network down")))

    def test_retry_after_header_present(self):
        """P0-3B：Retry-After header 存在 → 直接使用其秒數。"""

        class _FakeHeaders:
            def __init__(self, d):
                self._d = d

            def get(self, key, default=None):
                return self._d.get(key, default)

        class _FakeResp:
            def __init__(self, headers):
                self.headers = headers

        class FakeExc(Exception):
            pass

        exc = FakeExc("rate limited")
        exc.response = _FakeResp(_FakeHeaders({"retry-after": "3.5"}))
        self.assertAlmostEqual(_retry_after_seconds(exc, attempt=0), 3.5)

    def test_retry_after_header_missing(self):
        """P0-3C：沒有 response 屬性 → fallback 指數退避。"""
        exc = RuntimeError("rate limited")
        expected = BASE_BACKOFF_SECONDS * (2 ** 1)
        self.assertAlmostEqual(_retry_after_seconds(exc, attempt=1), expected)

    def test_retry_after_header_invalid_value(self):
        """P0-3D：Retry-After header 為非法值 → fallback 指數退避。"""

        class _FakeHeaders:
            def __init__(self, d):
                self._d = d

            def get(self, key, default=None):
                return self._d.get(key, default)

        class _FakeResp:
            def __init__(self, headers):
                self.headers = headers

        class FakeExc(Exception):
            pass

        exc = FakeExc("rate limited")
        exc.response = _FakeResp(_FakeHeaders({"retry-after": "not_a_number"}))
        expected = BASE_BACKOFF_SECONDS * (2 ** 2)
        self.assertAlmostEqual(_retry_after_seconds(exc, attempt=2), expected)


class RetrySleepSecondsTests(unittest.TestCase):
    """P0-4：驗證 retry 間隔的實際秒數依指數退避公式。"""

    def test_exponential_backoff_sequence(self):
        """連續失敗兩次後成功，第 1 次 sleep=1s、第 2 次 sleep=2s。"""
        side_effects = [
            RuntimeError("down once"),
            RuntimeError("down twice"),
            _FakeResponse('{"ok": true}'),
        ]
        mock_create = MagicMock(side_effect=side_effects)
        client = LLMClient.__new__(LLMClient)
        client._model = "test-model"
        client._notifier = None
        client.fallback_mode = False
        client._client = MagicMock()
        client._client.messages.create = mock_create

        with patch("src.api.claude_client.time.sleep") as mock_sleep:
            client._call("sys", "user")

        self.assertEqual(mock_create.call_count, 3)
        actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertEqual(
            actual_sleeps,
            [BASE_BACKOFF_SECONDS * (2 ** 0), BASE_BACKOFF_SECONDS * (2 ** 1)],
        )


if __name__ == "__main__":
    unittest.main()
