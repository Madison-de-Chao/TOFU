# -*- coding: utf-8 -*-
"""v0.8 繁中大詞典斷詞品質回歸測試。

jieba 內建 dict.txt 以簡體語料為主，直接切繁體會產生跨詞界碎片
（「這禮/拜」「電費帳/單寄來」），污染興趣領域統計、拉低 top-k 檢索
的詞彙重疊率。repo 內建 resources/jieba/dict.txt.big 後應切出正確詞界。

jieba 未安裝時整組跳過（tokenize 會降級 2-gram，該路徑不在本測試範圍）。
"""
import unittest

from src.utils import tokenizer as tk


@unittest.skipIf(tk._jieba is None, "jieba 未安裝，斷詞走 2-gram 降級模式")
class BigDictSegmentationTests(unittest.TestCase):
    def test_traditional_compound_words_not_fragmented(self):
        cases = {
            "這禮拜的報表": {"禮拜", "報表"},
            "電費帳單寄來": {"電費", "帳單"},
            "預算大概多少錢": {"預算"},
        }
        for text, expected in cases.items():
            tokens = set(tk.tokenize(text))
            self.assertTrue(
                expected.issubset(tokens),
                f"{text!r} 斷詞缺少 {expected - tokens}，實際：{tokens}",
            )

    def test_no_cross_boundary_fragments(self):
        # 修復前的代表性碎片。「這禮」是「這禮拜」被切斷的殘影。
        tokens = set(tk.tokenize("這禮拜的報表"))
        self.assertNotIn("這禮", tokens)
        tokens = set(tk.tokenize("電費帳單寄來"))
        self.assertNotIn("電費帳", tokens)
        self.assertNotIn("單寄來", tokens)

    def test_tokenize_nouns_keeps_nouns_drops_function_words(self):
        tokens = set(tk.tokenize_nouns("突然覺得這樣比較好，還是先把報表做完"))
        self.assertIn("報表", tokens)
        for junk in ("突然", "這樣", "比較", "還是"):
            self.assertNotIn(junk, tokens)


if __name__ == "__main__":
    unittest.main()
