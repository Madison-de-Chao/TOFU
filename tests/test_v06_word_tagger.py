"""v0.6 PR-E：word_index + word_tagger 單元測試。

覆蓋：
- WordIndexStore：檔案命名、upsert（新增/累積）、read_category、find_endpoint_ids
- WordTaggerGuardrail：$1/day cap、daily reset
- HaikuWordTagger：離線 fallback（無 API key）
- _parse_tagger_output：各種格式變體
- tag_and_index_endpoint：整合（mock tagger）
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.word_index import (
    KIND_ADJ,
    KIND_NOUN,
    KIND_TIME,
    KIND_VERB,
    WordIndexStore,
)
from src.middleware.word_tagger import (
    HaikuWordTagger,
    WordTaggerGuardrail,
    _parse_tagger_output,
    tag_and_index_endpoint,
)


# ---------------------------------------------------------------------------
# WordIndexStore 測試
# ---------------------------------------------------------------------------


class WordIndexFileNameTests(unittest.TestCase):
    """_filename_for 的路由邏輯。"""

    from src.middleware.word_index import _filename_for

    def test_noun_food(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("noun", "食"), "noun_食.jsonl")

    def test_noun_work(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("noun", "工作"), "noun_工作.jsonl")

    def test_noun_unknown_category_falls_back_to_other(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("noun", "太空"), "noun_其他.jsonl")

    def test_noun_no_category_falls_back_to_other(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("noun", None), "noun_其他.jsonl")

    def test_adj_color(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("adj", "顏色"), "adj_顏色.jsonl")

    def test_adj_unknown_falls_back_to_evaluation(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("adj", "怪"), "adj_評價.jsonl")

    def test_verb_unified_file(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("verb", "食"), "verbs.jsonl")
        self.assertEqual(_filename_for("verb", None), "verbs.jsonl")

    def test_time_unified_file(self):
        from src.middleware.word_index import _filename_for
        self.assertEqual(_filename_for("time", None), "time.jsonl")
        self.assertEqual(_filename_for("time", "時間"), "time.jsonl")

    def test_invalid_kind_raises(self):
        from src.middleware.word_index import _filename_for
        with self.assertRaises(ValueError):
            _filename_for("unknown", None)


class WordIndexUpsertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WordIndexStore(self.tmp.name)

    def _read_jsonl(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows

    def test_upsert_creates_new_noun_entry(self):
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        rows = self.store.read_category(KIND_NOUN, "食")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["word"], "拉麵")
        self.assertEqual(rows[0]["categories"], ["食"])
        self.assertEqual(rows[0]["endpoint_ids"], ["E001"])
        self.assertEqual(rows[0]["count"], 1)

    def test_upsert_cumulates_endpoint_ids(self):
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E002")
        rows = self.store.read_category(KIND_NOUN, "食")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["endpoint_ids"], ["E001", "E002"])
        self.assertEqual(rows[0]["count"], 2)

    def test_upsert_deduplicates_endpoint_ids(self):
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")  # 重複
        rows = self.store.read_category(KIND_NOUN, "食")
        self.assertEqual(rows[0]["endpoint_ids"], ["E001"])
        self.assertEqual(rows[0]["count"], 2)  # count 仍累積

    def test_verb_cross_category_accumulation(self):
        """動詞「看」可以跨類：先是[樂]，後是[育]，categories 應累積。"""
        self.store.upsert_word("看", KIND_VERB, "樂", "E001")
        self.store.upsert_word("看", KIND_VERB, "育", "E002")

        verbs = self.store.read_category(KIND_VERB, None)
        see = next(v for v in verbs if v["word"] == "看")
        self.assertIn("樂", see["categories"])
        self.assertIn("育", see["categories"])
        self.assertEqual(see["endpoint_ids"], ["E001", "E002"])
        self.assertEqual(see["count"], 2)

    def test_noun_different_categories_have_separate_files(self):
        """「台北」在[行]檔；「電影」在[樂]檔。各自獨立，不混入。"""
        self.store.upsert_word("台北", KIND_NOUN, "行", "E001")
        self.store.upsert_word("電影", KIND_NOUN, "樂", "E002")

        travel = self.store.read_category(KIND_NOUN, "行")
        entertain = self.store.read_category(KIND_NOUN, "樂")
        self.assertEqual(len(travel), 1)
        self.assertEqual(travel[0]["word"], "台北")
        self.assertEqual(len(entertain), 1)
        self.assertEqual(entertain[0]["word"], "電影")

    def test_time_word_stored_in_time_file(self):
        self.store.upsert_word("昨天", KIND_TIME, None, "E001")
        times = self.store.read_category(KIND_TIME, None)
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0]["word"], "昨天")

    def test_upsert_invalid_word_raises(self):
        with self.assertRaises(ValueError):
            self.store.upsert_word("", KIND_NOUN, "食", "E001")

    def test_upsert_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            self.store.upsert_word("拉麵", "unknown", "食", "E001")

    def test_upsert_no_endpoint_id_raises(self):
        with self.assertRaises(ValueError):
            self.store.upsert_word("拉麵", KIND_NOUN, "食", "")


class WordIndexReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WordIndexStore(self.tmp.name)

    def test_read_empty_category_returns_empty(self):
        rows = self.store.read_category(KIND_NOUN, "食")
        self.assertEqual(rows, [])

    def test_find_endpoint_ids_single_category(self):
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        self.store.upsert_word("烏龍麵", KIND_NOUN, "食", "E001")
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E002")

        eids = self.store.find_endpoint_ids(KIND_NOUN, "食", ["拉麵"])
        self.assertIn("E001", eids)
        self.assertIn("E002", eids)
        self.assertNotIn("E003", eids)

    def test_find_endpoint_ids_multiple_words_dedup(self):
        """兩個詞命中同一端點，只回傳一次。"""
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        self.store.upsert_word("烏龍麵", KIND_NOUN, "食", "E001")

        eids = self.store.find_endpoint_ids(KIND_NOUN, "食", ["拉麵", "烏龍麵"])
        self.assertEqual(eids.count("E001"), 1)

    def test_find_endpoint_ids_no_match_returns_empty(self):
        self.store.upsert_word("拉麵", KIND_NOUN, "食", "E001")
        eids = self.store.find_endpoint_ids(KIND_NOUN, "食", ["壽司"])
        self.assertEqual(eids, [])


class WordIndexConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WordIndexStore(self.tmp.name)

    def test_concurrent_upsert_does_not_lose_entries(self):
        n_threads = 6
        per_thread = 10

        def worker(tid: int):
            for i in range(per_thread):
                self.store.upsert_word(
                    f"詞{tid}-{i}", KIND_NOUN, "食", f"E{tid}-{i}"
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = self.store.read_category(KIND_NOUN, "食")
        words = {r["word"] for r in rows}
        expected_total = n_threads * per_thread
        self.assertEqual(len(words), expected_total)


# ---------------------------------------------------------------------------
# WordTaggerGuardrail 測試
# ---------------------------------------------------------------------------


class WordTaggerGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_file = os.path.join(self.tmp.name, "_guardrail.json")

    def _make_guardrail(self, max_cost: float = 1.0) -> WordTaggerGuardrail:
        return WordTaggerGuardrail(self.state_file, max_cost_per_day=max_cost)

    def test_check_returns_true_when_no_cost_recorded(self):
        g = self._make_guardrail()
        self.assertTrue(g.check_before_tag())

    def test_check_returns_false_when_cost_exceeded(self):
        g = self._make_guardrail(max_cost=0.001)
        g.record_cost(0.002)
        self.assertFalse(g.check_before_tag())

    def test_cost_persists_across_instances(self):
        g1 = self._make_guardrail()
        g1.record_cost(0.5)

        g2 = self._make_guardrail()
        self.assertAlmostEqual(g2.today_cost_usd, 0.5, places=4)

    def test_daily_reset_on_new_date(self):
        """模擬日期切換：舊日期的累計被清零。"""
        g = self._make_guardrail(max_cost=0.001)
        g.record_cost(0.002)
        self.assertFalse(g.check_before_tag())

        # 把 state 裡的 date 改成昨天
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        state["date"] = "2000-01-01"
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f)

        # 重新讀，今天的累計應該是 0
        g2 = self._make_guardrail(max_cost=0.001)
        self.assertTrue(g2.check_before_tag())
        self.assertAlmostEqual(g2.today_cost_usd, 0.0, places=5)

    def test_record_negative_cost_does_not_subtract(self):
        g = self._make_guardrail()
        g.record_cost(-100.0)
        self.assertAlmostEqual(g.today_cost_usd, 0.0, places=5)


# ---------------------------------------------------------------------------
# _parse_tagger_output 測試
# ---------------------------------------------------------------------------


class ParseTaggerOutputTests(unittest.TestCase):
    def test_parses_standard_output(self):
        raw = "我 / 昨天[時間] / 吃[食] 完 / 拉麵[食] / 覺得 / 好吃[評價]"
        result = _parse_tagger_output(raw)
        words = {e["word"]: e for e in result}

        self.assertIn("昨天", words)
        self.assertEqual(words["昨天"]["kind"], KIND_TIME)

        self.assertIn("吃", words)
        self.assertEqual(words["吃"]["kind"], KIND_VERB)
        self.assertEqual(words["吃"]["category"], "食")

        self.assertIn("拉麵", words)
        self.assertEqual(words["拉麵"]["kind"], KIND_NOUN)
        self.assertEqual(words["拉麵"]["category"], "食")

        self.assertIn("好吃", words)
        self.assertEqual(words["好吃"]["kind"], KIND_ADJ)
        self.assertEqual(words["好吃"]["category"], "評價")

        # 代詞「我」、連接詞「覺得」不應出現
        self.assertNotIn("我", words)
        self.assertNotIn("覺得", words)
        self.assertNotIn("完", words)

    def test_ignores_unknown_category(self):
        raw = "拉麵[宇宙] / 電影[樂]"
        result = _parse_tagger_output(raw)
        words = {e["word"] for e in result}
        self.assertNotIn("拉麵", words)  # 宇宙 is not a valid category
        self.assertIn("電影", words)

    def test_deduplicates_same_word(self):
        raw = "拉麵[食] / 吃[食] / 拉麵[食]"
        result = _parse_tagger_output(raw)
        words = [e["word"] for e in result]
        self.assertEqual(words.count("拉麵"), 1)

    def test_empty_output_returns_empty_list(self):
        self.assertEqual(_parse_tagger_output(""), [])
        self.assertEqual(_parse_tagger_output("我 / 你 / 他"), [])

    def test_adj_categories(self):
        raw = "紅[顏色] / 熱[溫度] / 晴天[天氣] / 棒[評價] / 開心[情緒]"
        result = _parse_tagger_output(raw)
        cats = {e["word"]: e["category"] for e in result}
        self.assertEqual(cats.get("紅"), "顏色")
        self.assertEqual(cats.get("熱"), "溫度")
        self.assertEqual(cats.get("晴天"), "天氣")
        self.assertEqual(cats.get("棒"), "評價")
        self.assertEqual(cats.get("開心"), "情緒")

    def test_noun_multi_char_treated_as_noun_not_verb(self):
        raw = "電影[樂] / 拉麵[食]"
        result = _parse_tagger_output(raw)
        kinds = {e["word"]: e["kind"] for e in result}
        self.assertEqual(kinds.get("電影"), KIND_NOUN)
        self.assertEqual(kinds.get("拉麵"), KIND_NOUN)

    def test_single_char_verb_marker(self):
        raw = "看[樂] / 吃[食]"
        result = _parse_tagger_output(raw)
        kinds = {e["word"]: e["kind"] for e in result}
        self.assertEqual(kinds.get("看"), KIND_VERB)
        self.assertEqual(kinds.get("吃"), KIND_VERB)


# ---------------------------------------------------------------------------
# HaikuWordTagger（offline 模式）
# ---------------------------------------------------------------------------


class HaikuWordTaggerOfflineTests(unittest.TestCase):
    """無 API key → 離線模式，tag() 回 None。"""

    def test_offline_mode_without_key(self):
        tagger = HaikuWordTagger(api_key=None)
        self.assertTrue(tagger.offline)
        result = tagger.tag("我昨天吃拉麵")
        self.assertIsNone(result)

    def test_empty_text_returns_empty_list_even_offline(self):
        """空文字在離線模式下也回 []——空字串不需要 API 就能確定「沒有詞」。"""
        tagger = HaikuWordTagger(api_key=None)
        # tag() 先查 text 是否為空（回 []），再查 offline（回 None）。
        # 空文字路徑先行，所以即使離線也是 []。
        result = tagger.tag("")
        self.assertEqual(result, [])

    def test_estimate_cost_is_positive(self):
        tagger = HaikuWordTagger(api_key=None)
        cost = tagger.estimate_cost_for_text("我昨天吃拉麵去看電影")
        self.assertGreater(cost, 0.0)


# ---------------------------------------------------------------------------
# tag_and_index_endpoint 整合測試（mock tagger）
# ---------------------------------------------------------------------------


class TagAndIndexEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.word_store = WordIndexStore(os.path.join(self.tmp.name, "words"))
        self.guardrail = WordTaggerGuardrail(
            os.path.join(self.tmp.name, "_guardrail.json"),
            max_cost_per_day=10.0,
        )

    def _make_mock_tagger(self, tag_result):
        tagger = MagicMock(spec=HaikuWordTagger)
        tagger.offline = False
        tagger.tag.return_value = tag_result
        tagger.estimate_cost_for_text.return_value = 0.001
        return tagger

    def test_successful_tagging_writes_to_word_index(self):
        tagger = self._make_mock_tagger([
            {"word": "拉麵", "kind": "noun", "category": "食"},
            {"word": "吃", "kind": "verb", "category": "食"},
        ])

        result = tag_and_index_endpoint(
            endpoint_id="E001",
            text="我吃拉麵",
            tagger=tagger,
            guardrail=self.guardrail,
            word_store=self.word_store,
        )

        self.assertTrue(result["tagged"])
        self.assertEqual(result["count"], 2)

        food_nouns = self.word_store.read_category(KIND_NOUN, "食")
        self.assertTrue(any(r["word"] == "拉麵" for r in food_nouns))

        verbs = self.word_store.read_category(KIND_VERB, None)
        self.assertTrue(any(r["word"] == "吃" for r in verbs))

    def test_tagger_failure_returns_tagged_false(self):
        tagger = self._make_mock_tagger(None)

        result = tag_and_index_endpoint(
            endpoint_id="E001",
            text="我吃拉麵",
            tagger=tagger,
            guardrail=self.guardrail,
            word_store=self.word_store,
        )

        self.assertFalse(result["tagged"])
        self.assertEqual(result["reason"], "tagger_failed")

        # 詞索引應是空的（沒有寫入）
        self.assertEqual(self.word_store.read_category(KIND_NOUN, "食"), [])

    def test_cost_limit_blocks_tagging(self):
        guardrail = WordTaggerGuardrail(
            os.path.join(self.tmp.name, "_guardrail_tight.json"),
            max_cost_per_day=0.0,  # 0 → 立刻超限
        )
        tagger = self._make_mock_tagger([
            {"word": "拉麵", "kind": "noun", "category": "食"},
        ])

        result = tag_and_index_endpoint(
            endpoint_id="E001",
            text="我吃拉麵",
            tagger=tagger,
            guardrail=guardrail,
            word_store=self.word_store,
        )

        self.assertFalse(result["tagged"])
        self.assertEqual(result["reason"], "cost_limit_reached")
        tagger.tag.assert_not_called()

    def test_empty_text_returns_tagged_false(self):
        tagger = self._make_mock_tagger([])

        result = tag_and_index_endpoint(
            endpoint_id="E001",
            text="",
            tagger=tagger,
            guardrail=self.guardrail,
            word_store=self.word_store,
        )

        self.assertFalse(result["tagged"])
        self.assertEqual(result["reason"], "empty_text")

    def test_cost_is_recorded_after_success(self):
        tagger = self._make_mock_tagger([
            {"word": "拉麵", "kind": "noun", "category": "食"},
        ])
        tag_and_index_endpoint(
            endpoint_id="E001",
            text="我吃拉麵",
            tagger=tagger,
            guardrail=self.guardrail,
            word_store=self.word_store,
        )
        self.assertGreater(self.guardrail.today_cost_usd, 0.0)

    def test_empty_tag_result_tagged_true_with_count_zero(self):
        """tagger 回 []（純虛詞）→ 不視為失敗，count=0。"""
        tagger = self._make_mock_tagger([])

        result = tag_and_index_endpoint(
            endpoint_id="E001",
            text="我我我",  # 非空文字
            tagger=tagger,
            guardrail=self.guardrail,
            word_store=self.word_store,
        )

        # Note: 空文字 check 先通了（text 非空），tagger 回 [] → tagged True, count 0
        self.assertTrue(result.get("tagged"))
        self.assertEqual(result.get("count"), 0)


if __name__ == "__main__":
    unittest.main()
