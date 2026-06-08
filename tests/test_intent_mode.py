"""意圖分流（detect_intent_mode）硬判準測試。

依規格 06209671「逗福Tofu 修復規格：意圖分流與規劃模式 v1」§3.1 測試案例表。

從嚴判準：唯有「命令句 + 執行目的」才判 execute；疑問句、陳述句、無法
判定一律 plan。判斷材料只有使用者最近一輪原始輸入的句式。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.middleware.confirmation import detect_intent_mode  # noqa: E402


class DetectIntentModeTests(unittest.TestCase):
    """§3.1 測試案例表逐列驗證。"""

    def test_statement_travel_preference(self):
        # 陳述句
        self.assertEqual(
            detect_intent_mode("我想去義大利玩，輕鬆度假"), "plan"
        )

    def test_statement_wedding_original_case(self):
        # 陳述句（婚禮原案）
        self.assertEqual(
            detect_intent_mode("輕鬆度假，坐長榮，米蘭進羅馬出"), "plan"
        )

    def test_question_two_weeks(self):
        # 疑問句
        self.assertEqual(
            detect_intent_mode("義大利兩週怎麼安排比較好？"), "plan"
        )

    def test_question_soft_nudge(self):
        # 疑問句（軟催促，從嚴歸規劃）
        self.assertEqual(detect_intent_mode("是不是該訂機票了？"), "plan")

    def test_imperative_but_think(self):
        # 命令句但目的=想
        self.assertEqual(detect_intent_mode("幫我想想這趟怎麼排"), "plan")

    def test_imperative_but_plan(self):
        # 命令句但目的=規劃
        self.assertEqual(detect_intent_mode("幫我規劃義大利行程"), "plan")

    def test_imperative_but_list_options(self):
        # 命令句但目的=列選項
        self.assertEqual(
            detect_intent_mode("幫我列出可以考慮的城市"), "plan"
        )

    def test_imperative_execute_book_flight(self):
        # 命令句+執行目的(訂)
        self.assertEqual(
            detect_intent_mode("幫我訂長榮米蘭進羅馬出的機票"), "execute"
        )

    def test_imperative_execute_book_hotel(self):
        # 命令句+執行目的(訂)
        self.assertEqual(
            detect_intent_mode("幫我把這三晚住宿訂下來"), "execute"
        )

    def test_imperative_list_todos(self):
        # 命令句+執行目的(列待辦)；清單仍走 execute（§3.2 第 4 點負責驗 deadline）
        self.assertEqual(
            detect_intent_mode("列出我接下來要做的事"), "execute"
        )

    def test_rhetorical_nudge_known_limitation(self):
        # §1/§3.1 標明的「已知漏洞」：反問式催促（疑問句但實為命令）判成 plan。
        # 這是規格刻意取捨——不開特例兜，靠後續對話補。語意上使用者其實要執行，
        # 但本函式回 plan 即「接受的誤判」，不視為 bug。
        self.assertEqual(
            detect_intent_mode("你可以現在幫我把票訂了嗎？"), "plan"
        )


class DetectIntentModeEdgeTests(unittest.TestCase):
    """邊界與防呆。"""

    def test_empty_input_defaults_to_plan(self):
        self.assertEqual(detect_intent_mode(""), "plan")
        self.assertEqual(detect_intent_mode("   "), "plan")

    def test_question_priority_over_execute_verb(self):
        # 疑問特徵優先於執行目的詞：含「訂」但是疑問句 → plan
        self.assertEqual(detect_intent_mode("要不要先訂機票"), "plan")

    def test_plain_execute_command_no_subject(self):
        # 動詞起首、無主詞、執行目的 → execute
        self.assertEqual(detect_intent_mode("訂明天台北到高雄的高鐵"), "execute")


class DetectIntentModeReviewRegressionTests(unittest.TestCase):
    """PR #34 review（Codex / Copilot）標出的 substring 誤判，鎖回歸。"""

    def test_noun_ending_in_question_particle_is_execute(self):
        # 「吧」是「酒吧」「吧台椅」的尾字，不應被當疑問語助詞攔成 plan。
        self.assertEqual(detect_intent_mode("幫我預約這間酒吧"), "execute")
        self.assertEqual(detect_intent_mode("幫我買吧台椅"), "execute")

    def test_preference_clause_does_not_override_execute(self):
        # 受詞含「想要 / 想去」不應因單字「想」被攔成 plan。
        self.assertEqual(detect_intent_mode("幫我訂我想要的餐廳"), "execute")
        self.assertEqual(
            detect_intent_mode("幫我預約我想去的餐廳"), "execute"
        )

    def test_arbitrary_clause_does_not_override_execute(self):
        # 「哪天都行 / 什麼都可以」是任意性表達，不應被當疑問詞攔成 plan。
        self.assertEqual(
            detect_intent_mode("幫我訂哪天都行的早班機"), "execute"
        )
        self.assertEqual(
            detect_intent_mode("幫我訂什麼都可以的便宜機票"), "execute"
        )

    def test_genuine_weak_question_still_plan(self):
        # 弱疑問特徵在「非執行命令」句仍應判 plan。
        self.assertEqual(detect_intent_mode("這是什麼"), "plan")
        self.assertEqual(detect_intent_mode("去哪玩比較好"), "plan")
        self.assertEqual(detect_intent_mode("這樣好吃嗎"), "plan")

    def test_agreement_plus_command_accepted_misjudgment(self):
        # 已知接受的誤判：祈使詞不在句首 → plan（從嚴，不擴張命令偵測）。
        self.assertEqual(detect_intent_mode("好吧，幫我訂機票"), "plan")


if __name__ == "__main__":
    unittest.main()
