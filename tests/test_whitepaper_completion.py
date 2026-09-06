"""Whitepaper regression contracts; synthetic inputs and no paid API calls."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from src.api.factory import create_client
from src.api.openai_client import OpenAIClient
from src.api.claude_client import LLMClientError
from src.middleware.delivery import audit_delivery, render_delivery
from src.middleware.endpoint import EndpointStore, retrieve_top_k_endpoints
from src.middleware.lookup import pack_lookup
from src.middleware.preferences import merge_preferences
from src.middleware import check as check_mod
from src.middleware.confirmation import check_trajectory_convergence
from src import main


def response(content, **extra):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}], **extra}).encode())


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TOFU_MODEL": "", "OPENAI_MODEL": "",
            "OPENAI_API_KEY": "", "TOFU_PROVIDER": "offline"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def client(self, **kwargs):
        client = OpenAIClient(api_key="test-secret", model="fixture-model", **kwargs)
        client._opener = Mock()
        return client

    def test_serialized_chat_contract_and_usage(self):
        c = self.client()
        c._opener.open.return_value = response("完成", usage={"prompt_tokens": 12})
        self.assertEqual(c._call("system", "原話", 123), "完成")
        args, kwargs = c._opener.open.call_args
        self.assertEqual(kwargs["timeout"], 30)
        payload = json.loads(args[0].data)
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "原話"})
        self.assertEqual(payload["max_completion_tokens"], 123)
        self.assertEqual(c.last_usage["prompt_tokens"], 12)

    def test_legacy_token_parameter_and_local_server(self):
        c = self.client(base_url="http://127.0.0.1:8999/v1", token_parameter="max_tokens")
        c._opener.open.return_value = response("ok")
        c._call("s", "u", 15)
        req = c._opener.open.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8999/v1/chat/completions")
        self.assertEqual(json.loads(req.data)["max_tokens"], 15)

    def test_restate_uses_shared_json_parser(self):
        c = self.client()
        c._opener.open.return_value = response(json.dumps({
            "restate_text": "你要辦活動", "gap_questions": [], "gap_categories": [],
            "inferred_goal": "辦活動", "inferred_motivation": "", "inferred_constraints": [],
        }))
        result = c.generate_restate("辦活動", {}, [])
        self.assertEqual(result["restate_text"], "你要辦活動")

    def test_auth_failure_not_retried_and_secrets_not_echoed(self):
        c = self.client()
        c._opener.open.side_effect = HTTPError(c._url, 401, "test-secret", {}, None)
        with self.assertRaises(LLMClientError) as err:
            c._call("s", "private-input")
        self.assertNotIn("test-secret", str(err.exception))
        self.assertEqual(c._opener.open.call_count, 1)

    @patch("src.api.openai_client.time.sleep")
    def test_rate_limit_retried_then_success(self, sleep):
        c = self.client()
        c._opener.open.side_effect = [HTTPError(c._url, 429, "limit", {"Retry-After": "1"}, None), response("ok")]
        self.assertEqual(c._call("s", "u"), "ok")
        sleep.assert_called_once_with(1)

    @patch("src.api.openai_client.time.sleep")
    def test_network_failure_has_bounded_attempts(self, sleep):
        c = self.client()
        c._opener.open.side_effect = URLError("secret")
        with self.assertRaises(LLMClientError):
            c._call("s", "u")
        self.assertEqual(c._opener.open.call_count, 3)

    def test_malformed_and_empty_responses_fail_explicitly(self):
        for raw in [b"no json", b"{}", b'{"choices":[]}', b'{"choices":[{"message":{"content":null}}]}']:
            with self.subTest(raw=raw):
                c = self.client()
                c._opener.open.return_value = io.BytesIO(raw)
                with self.assertRaises(LLMClientError):
                    c._call("s", "u")

    def test_invalid_remote_plaintext_or_credentials_rejected(self):
        for url in ["http://example.com/v1", "https://user:secret@example.com/v1", "https://example.com/v1?k=secret"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.client(base_url=url)

    def test_model_is_explicit_and_provider_is_validated(self):
        with self.assertRaises(ValueError):
            OpenAIClient(api_key="test")
        with self.assertRaises(ValueError):
            create_client("typo")

    def test_offline_does_not_use_ambient_claude_key(self):
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "must-not-be-used"}), patch("src.api.claude_client.Anthropic") as sdk:
            c = create_client("offline")
            self.assertTrue(c.fallback_mode)
            sdk.assert_not_called()

    def test_openai_batch_fails_honestly(self):
        c = self.client()
        with self.assertRaises(LLMClientError):
            c.submit_batch([])


class GovernanceTests(unittest.TestCase):
    def test_code_fences_remain_copyable(self):
        raw = '```python\ntext = """hello\n\nworld"""\nprint(text)\n```'
        self.assertIn(raw, render_delivery(audit_delivery(raw)))

    def test_band_name_is_not_a_violent_trajectory(self):
        self.assertFalse(check_trajectory_convergence(["I like The Killers", "The Killers concert", "The Killers album"]))
        self.assertTrue(check_trajectory_convergence(["kill", "killed", "killing"]))
        with self.assertRaises(ValueError):
            check_trajectory_convergence([], 0)

    def test_date_is_not_a_source(self):
        audit = audit_delivery("Zone A：根據 2023 年的研究，成功率 98%。")
        self.assertEqual(audit["segments"][0]["zone"], "B")
        self.assertTrue(audit["segments"][0]["downgraded"])

    def test_source_in_one_line_does_not_upgrade_another(self):
        audit = audit_delivery("Zone A：資料 https://example.com/report\nZone A：成功率 98%。\nZone C：我認為值得嘗試。")
        self.assertEqual([s["zone"] for s in audit["segments"]], ["A", "B", "C"])
        self.assertEqual(audit["external_verification"], "not_performed")
        self.assertTrue(audit["mixed_zones"])

    def test_missing_counterevidence_is_not_fabricated(self):
        audit = audit_delivery("可能會成功。")
        self.assertIsNone(audit["segments"][0]["falsification_condition"])
        self.assertTrue(audit["degraded"])
        self.assertIn("未提供", render_delivery(audit))

    def test_stance_is_preserved_during_degradation(self):
        self.assertEqual(audit_delivery("Zone C：我認為保留現狀。", degraded=True)["segments"][0]["zone"], "C")

    def test_lookup_preserves_raw_storage_and_bounds_transmission(self):
        rows = [{"endpoint_id": str(i), "start_data": {"user_input": "預算 " + str(i),
                 "gap_categories": ["budget"], "answered_categories": ["scope"]}, "timestamp": "2026-09-06"} for i in range(40)]
        before = json.dumps(rows)
        selected, encoded, audit = pack_lookup(rows, categories=["budget", "scope"], max_chars=500, top_k=3)
        self.assertLessEqual(len(encoded), 500)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(audit["unknown_categories"], ["budget"])
        self.assertEqual(audit["encoded_chars"], len(encoded))
        self.assertEqual(json.dumps(rows), before)
        self.assertIn("[E002]", encoded)
        self.assertTrue(audit["omitted_endpoint_ids"])

    def test_empty_lookup_is_still_recorded(self):
        _, encoded, audit = pack_lookup([], categories=["budget"])
        self.assertEqual(encoded, "")
        self.assertTrue(audit["performed"])
        self.assertEqual(audit["unknown_categories"], ["budget"])

    def test_preference_normalization_keeps_evidence_and_polarity(self):
        prefs = [{"item": "Indie  Rock", "type": "explicit", "context": "one"},
                 {"item": "ｉｎｄｉｅ rock", "type": "explicit", "context": "two"},
                 {"item": "indie rock", "type": "exclusion"}]
        merged = merge_preferences(prefs, [])
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(merged[0]["evidence"]), 2)
        self.assertNotIn("evidence", prefs[0])
        self.assertEqual(merge_preferences(merged, []), merged)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EndpointStore(str(Path(self.tmp.name) / "endpoints.jsonl"))

    def test_retrieval_is_not_user_confirmation(self):
        row = self.store.append_start(event_id="test", start_data={"goal": "音樂", "user_input": "音樂"})
        self.store.append_end(event_id="test", end_data={"result": "test"})
        self.store.mark_hit([row["endpoint_id"]], current_round=0)
        self.store.apply_cooldown(current_round=100)
        self.store.mark_hit([row["endpoint_id"]], current_round=100, reactivate_pending=False)
        self.assertEqual(self.store.starts()[0]["status"], "pending_confirmation")
        self.assertTrue(self.store.resolve_memory(row["endpoint_id"], "superseded"))
        self.assertEqual(retrieve_top_k_endpoints(self.store.all(), "音樂")[0], [])
        self.assertEqual(len(self.store.all()), 2)
        self.assertTrue(self.store.resolve_memory(row["endpoint_id"], "active"))
        self.assertEqual(len(self.store.starts()[0]["status_history"]), 2)

    def test_failed_memory_resolution_is_non_destructive(self):
        self.assertFalse(self.store.resolve_memory("missing", "active"))
        with self.assertRaises(ValueError):
            self.store.resolve_memory("missing", "delete")

    def test_free_audits_before_print_and_persists_same_audit(self):
        c = Mock(fallback_mode=True)
        c.execute_task.return_value = "Zone A：2026 年會有 98% 成功率。"
        c.analyze_deviation.return_value = "未驗證"
        observed = []
        def capture(text):
            if text.startswith("[逗福Tofu 執行]"):
                self.assertIn("Zone B", text)
                self.assertIn("待驗證", text)
                observed.append(text)
        with patch.object(main, "_print", side_effect=capture), patch.object(main, "_try_tag_endpoint"):
            self.assertTrue(main.run_one_interaction("評估活動預算", self.store, c, mode="free"))
        self.assertEqual(len(observed), 1)
        end = self.store.ends()[0]["end_data"]
        self.assertEqual(end["delivery_audit"]["zone"], "B")
        self.assertTrue(self.store.starts()[0]["start_data"]["lookup_audit"]["performed"])

    def test_check_lookup_is_recorded_without_private_history_injection(self):
        c = Mock(fallback_mode=True)
        c.run_check_stage.return_value = "資料未經查證"
        with patch.object(main, "_print"), patch.object(main, "_try_tag_endpoint"):
            self.assertTrue(main.run_check_mode("這個活動消息是真的嗎", self.store, c, session_dir=self.tmp.name))
            session = check_mod.find_latest_session(self.tmp.name)
            self.assertTrue(session["lookup_audit"]["performed"])
            self.assertFalse(session["lookup_audit"]["injected_into_model"])
            self.assertTrue(main.run_check_followup("1", self.store, c, session_dir=self.tmp.name))
        self.assertTrue(self.store.ends()[0]["end_data"]["delivery_audit"])
        self.assertEqual(set(c.run_check_stage.call_args.kwargs), {"stage", "user_content", "stage1_output"})

    def test_openai_provider_does_not_call_background_claude_tagger(self):
        with patch.dict(os.environ, {"TOFU_PROVIDER": "openai"}), patch.object(main, "_get_word_pipeline") as pipeline:
            main._try_tag_endpoint("test", "input", "result")
            pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
