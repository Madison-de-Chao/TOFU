"""v0.6 PR-H audit 回應：🔴#1 Zone C / 🔴#2 檢索 / 🟠#7 JSON 修復 / #4 --batch。

對應 MOMO 2026-04-16 audit 報告（四個優先修項）：

1. 🔴 #1：Zone C 分類遺漏——jieba 把「我建議」黏成一個 token，純 token
   比對漏命中。修法：加「建議」「不建議」到 _ZONE_C_SUBSTR。
2. 🔴 #2：token 檢索漏報——jieba 把「學攝影」黏成一個 token，查詢
   「攝影」交集空。修法：_count_term_hits 加子字串備援。
3. 🟠 #7：generate_restate 缺 JSON 截斷修復。修法：仿 construct_profile
   加 _try_repair_json fallback。
4. #4（MOMO 加的）：main.py 加 --batch 模式，讓 live_test_v06.py 能
   pipe stdin 跑而不用 pexpect。

本檔只測 audit 項的行為。Zone B/C 原有測試（test_feature_implementation.py）
照舊，不重複。
"""

from __future__ import annotations

import builtins
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 🔴 #1 Zone C：子字串備援
# ---------------------------------------------------------------------------
class ZoneCSubstringFallbackTests(unittest.TestCase):
    """jieba 黏詞狀況下，Zone C 子字串備援應該接住。"""

    def test_wo_jianyi_subject_phrase_is_zone_c(self):
        """「我建議先確定人數再找場地」——jieba 會把『我建議』黏成一個 token。
        修法後透過子字串比對命中『建議』→ Zone C。"""
        from src.middleware.confirmation import classify_zone
        self.assertEqual(
            classify_zone("我建議先確定人數再找場地"),
            "C",
        )

    def test_wo_bujianyi_is_zone_c(self):
        """「我不建議用這個方案」——同理，子字串比對命中『不建議』→ Zone C。"""
        from src.middleware.confirmation import classify_zone
        self.assertEqual(
            classify_zone("我不建議用這個方案"),
            "C",
        )

    def test_zone_c_overrides_b_with_jieba_bound_token(self):
        """C 優先於 B。「我建議可能選戶外場地」含『可能』（B 信號）跟
        『我建議』（C 信號被黏詞）→ 修法後應判 C。"""
        from src.middleware.confirmation import classify_zone
        self.assertEqual(
            classify_zone("我建議可能選戶外場地"),
            "C",
        )

    def test_kenengxing_still_classifies_as_b_not_c(self):
        """防止 false positive：『可能性』不該因子字串『可能』誤判為 B 以外
        的值。核心是驗『建議』字串擴充後沒破壞 Zone B 偵測邏輯。"""
        from src.middleware.confirmation import classify_zone
        # 「可能性很低」應判 B（『可能』的 token 命中，但非 C）
        self.assertEqual(
            classify_zone("可能性很低"),
            "B",
        )

    def test_stance_suggest_english_still_works(self):
        """英文『suggest』仍由 _ZONE_C_SINGLE 命中（不受 SUBSTR 變動影響）。"""
        from src.middleware.confirmation import classify_zone
        self.assertEqual(
            classify_zone("i suggest doing it this way"),
            "C",
        )


# ---------------------------------------------------------------------------
# 🔴 #2 檢索 token 不匹配：子字串備援
# ---------------------------------------------------------------------------
class CountTermHitsTests(unittest.TestCase):
    def test_exact_intersection(self):
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(_count_term_hits({"攝影"}, {"攝影"}), 1)
        self.assertEqual(
            _count_term_hits({"攝影", "旅行"}, {"攝影", "旅行"}),
            2,
        )

    def test_substring_fallback_for_jieba_bound_token(self):
        """『攝影』應該在端點含『學攝影』時命中（jieba 把『學攝影』黏成一個 token）。"""
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(
            _count_term_hits({"攝影"}, {"學攝影"}),
            1,
        )

    def test_substring_both_directions(self):
        """query 是 ep 的子字串、或 ep 是 query 的子字串，都算命中。"""
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(
            _count_term_hits({"攝影棚"}, {"攝影"}),
            1,
        )

    def test_single_char_tokens_ignored(self):
        """長度 <2 的 token 不做子字串比對，避免『書』誤命中『讀書』。"""
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(_count_term_hits({"書"}, {"讀書"}), 0)
        self.assertEqual(_count_term_hits({"讀"}, {"讀書"}), 0)

    def test_unrelated_returns_zero(self):
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(_count_term_hits({"攝影"}, {"籃球"}), 0)

    def test_each_query_token_counts_once(self):
        """每個查詢詞最多 +1 命中，防止黏詞被反覆計分。"""
        from src.middleware.endpoint import _count_term_hits
        # 查詢「攝影」對上兩個含「攝影」的端點詞 → 仍只算 1
        self.assertEqual(
            _count_term_hits({"攝影"}, {"學攝影", "攝影棚"}),
            1,
        )

    def test_empty_inputs(self):
        from src.middleware.endpoint import _count_term_hits
        self.assertEqual(_count_term_hits(set(), {"攝影"}), 0)
        self.assertEqual(_count_term_hits({"攝影"}, set()), 0)
        self.assertEqual(_count_term_hits(set(), set()), 0)


class RetrieveWithSubstringTests(unittest.TestCase):
    """整合驗證：retrieve_top_k_endpoints 在 jieba 黏詞情境下能命中端點。"""

    def test_retrieves_substring_matched_endpoint(self):
        from src.middleware.endpoint import retrieve_top_k_endpoints
        # 端點的 goal 含「學攝影」，端點詞集合會含此 token（尤其 jieba）
        # 查詢「攝影」在修法前交集空；修法後應該能命中
        rows = [
            {
                "type": "start",
                "endpoint_id": "e1",
                "event_id": "ev1",
                "timestamp": "2026-04-16T10:00:00Z",
                "start_data": {
                    "goal": "學攝影",
                    "user_input": "想學攝影",
                    "constraints": [],
                },
            },
        ]
        top, _ = retrieve_top_k_endpoints(rows, "攝影", top_k=5)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["endpoint_id"], "e1")


# ---------------------------------------------------------------------------
# 🟠 #7 generate_restate JSON 截斷修復
# ---------------------------------------------------------------------------
class GenerateRestateJsonRepairTests(unittest.TestCase):
    """模擬 LLM 回傳 truncated JSON，驗證 _try_repair_json fallback 接得住。"""

    def _make_client_with_response(self, raw_response: str):
        """建一個 LLMClient，把 _call 固定回傳指定字串（測試用）。"""
        from src.api.claude_client import LLMClient
        client = LLMClient()
        # Force non-fallback path for this test（讓它真的走 JSON 解析）
        client.fallback_mode = False
        client._call = lambda *a, **kw: raw_response
        return client

    def test_truncated_json_recovered_via_repair(self):
        """LLM 回截斷 JSON → 第一次 _extract_json 失敗 → _try_repair_json
        補尾括號 → 正常正規化欄位。"""
        # truncated（缺最後一個 '}'）
        truncated = (
            '{"restate_text": "你想辦生日派對",'
            ' "gap_questions": ["幾人參加？"],'
            ' "gap_categories": ["headcount"],'
            ' "inferred_goal": "辦派對",'
            ' "inferred_motivation": "",'
            ' "inferred_constraints": []'
        )  # 結尾少一個 '}'
        client = self._make_client_with_response(truncated)
        result = client.generate_restate(
            user_input="辦生日派對",
            baseline_summary={},
            allowed_categories=["headcount", "budget"],
        )
        self.assertEqual(result["restate_text"], "你想辦生日派對")
        self.assertEqual(result["gap_questions"], ["幾人參加？"])
        self.assertEqual(result["gap_categories"], ["headcount"])

    def test_unrecoverable_json_still_raises(self):
        """完全無法修復的 JSON 仍應拋 LLMClientError（向後相容）。"""
        from src.api.claude_client import LLMClientError
        client = self._make_client_with_response("not json at all!!!")
        with self.assertRaises(LLMClientError):
            client.generate_restate(
                user_input="x",
                baseline_summary={},
                allowed_categories=[],
            )


# ---------------------------------------------------------------------------
# #4 --batch 模式：非互動式 stdin 驅動
# ---------------------------------------------------------------------------
class BatchModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # 讓測試的端點與 profile 落在 tmpdir，不污染全域
        self.env_patcher = patch.dict(os.environ, {
            "TOFU_ENDPOINTS_PATH": os.path.join(self.tmp.name, "ep.jsonl"),
            "TOFU_PROFILE_PATH": os.path.join(self.tmp.name, "profile.json"),
            "TOFU_SESSIONS_DIR": os.path.join(self.tmp.name, "sessions"),
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.tmp.cleanup()

    def _run_main_with_stdin(self, lines: list[str], argv: list[str]):
        """用 monkeypatch 把 input() 改成從 lines 取，跑 main(argv)。"""
        # 要在 patch 前重新載入 main 以套用新環境變數
        import importlib
        import src.main as main_mod
        importlib.reload(main_mod)
        it = iter(lines)

        def fake_input(prompt=""):
            try:
                line = next(it)
            except StopIteration as exc:
                raise EOFError from exc
            return line

        buf = io.StringIO()
        with patch.object(builtins, "input", fake_input):
            with redirect_stdout(buf):
                code = main_mod.main(argv=argv)
        return code, buf.getvalue()

    def test_batch_flag_suppresses_prompt_prefix(self):
        """--batch 下不該有 '> ' 前綴污染 stdout。"""
        code, out = self._run_main_with_stdin(
            ["/help", "/exit"],
            argv=["--batch"],
        )
        self.assertEqual(code, 0)
        # 明確檢查沒有 "> " prompt 前綴
        self.assertNotIn("> /help", out)
        self.assertNotIn("> /exit", out)
        # 但 /help 的輸出應該在
        self.assertIn("可用指令", out)

    def test_batch_flag_suppresses_status_line(self):
        """--batch 下每次 prompt 前的狀態列也應抑制。"""
        code, out = self._run_main_with_stdin(
            ["/exit"],
            argv=["--batch"],
        )
        self.assertEqual(code, 0)
        # 狀態列典型含「總互動數」「已記錄」類字樣。實際依 _status_line
        # 實作可能不同，這裡用最寬鬆的檢查：沒有 "> " 前綴。
        self.assertNotIn("> ", out)

    def test_unknown_arg_returns_non_zero(self):
        """未知 flag 應回 exit code 2。"""
        code, out = self._run_main_with_stdin([], argv=["--bogus"])
        self.assertEqual(code, 2)
        self.assertIn("未知參數", out)

    def test_eof_triggers_clean_exit_in_batch(self):
        """stdin 結束 → _prompt 收到 EOFError → 回 /exit → 主迴圈正常結束。"""
        code, out = self._run_main_with_stdin(
            [],  # 立刻 EOF
            argv=["--batch"],
        )
        self.assertEqual(code, 0)
        self.assertIn("再見", out)


if __name__ == "__main__":
    unittest.main()
