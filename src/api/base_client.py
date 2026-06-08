"""LLM 後端抽象介面（P1-2）。

依 TOFU_功能實作開發規格書 P1-2：把 LLMClient 重構成抽象介面 + 多個後端。
任何新的後端（Claude / OpenAI / DeepSeek / ...）都必須實作這個介面。

架構決定：
- 所有後端共用同一組對外方法（generate_restate / merge_confirmation /
  execute_task / analyze_deviation）。
- 決策邏輯（端點比對、偏差偵測、頻率統計）不交給 LLM，仍在其他模組完成。
- 後端只負責自然語言生成與結構化抽取（NLG + structured output）。
- fallback_mode 屬性讓呼叫端知道當前客戶端是否為離線純程式碼模式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """所有 LLM 後端共用的抽象介面。

    子類別必須實作所有 abstractmethod。對外方法的語意由 claude_client.py
    中現有的 LLMClient 定義，任何新後端都要與之保持一致。
    """

    fallback_mode: bool = False

    @abstractmethod
    def generate_restate(
        self,
        user_input: str,
        baseline_summary: dict[str, Any],
        allowed_categories: list[str],
    ) -> dict[str, Any]:
        """產生復述 + 補位提問。"""

    @abstractmethod
    def merge_confirmation(
        self,
        original_input: str,
        restate_text: str,
        user_confirmation_text: str,
    ) -> dict[str, Any]:
        """把使用者對補位提問的回覆合併進結構化欄位。"""

    @abstractmethod
    def execute_task(
        self,
        goal: str,
        motivation: str,
        constraints: list[str],
        user_profile: dict | None = None,
        *,
        strategy_brief: str | None = None,
        encoded_top_k: str | None = None,
        output_mode: str = "default",
    ) -> str:
        """根據確認後的目標產出一段建議或執行步驟。

        v0.6（PR-B）：加 `output_mode` 參數，用來在 default / free / risk /
        propose / propose_final 五種輸出風格之間切換。思考引擎（畫像、top-k、
        策略摘要）照跑，差別只在最後用哪份 OUTPUT_PROMPT。`/check` 模式獨立
        於此方法，不走 execute_task。
        """

    @abstractmethod
    def analyze_deviation(
        self,
        goal: str,
        result: str,
        deviation: str,
    ) -> str:
        """出口檢查——分析結果與目標的偏差原因。"""

    @abstractmethod
    def answer_with_profile(
        self,
        haystack: str,
        question: str,
        profile: dict[str, Any] | None = None,
    ) -> str:
        """用輪廓 + 完整對話歷史回答問題。

        專為 LongMemEval 等評測場景設計。與 :meth:`execute_task` 的職責
        分離：execute_task 處理一般對話的任務執行（給建議、列步驟），
        ``answer_with_profile`` 處理根據完整對話歷史精確回答問題的場景。

        設計原則：輪廓是注意力引導，不是唯一資訊來源；完整 haystack 仍然
        送入 user content。
        """

    def run_check_stage(
        self,
        *,
        stage: str,
        user_content: str,
        stage1_output: str | None = None,
    ) -> str:
        """v0.6 PR-D：/check 模式（ISF 資訊完整性檢驗器）專用 LLM 呼叫。

        此方法**繞過** :meth:`execute_task`——不查 top-k、不算策略摘要、
        不碰使用者畫像。system prompt 固定為 ISF v3.2 + 逗福補充指令
        （見 :mod:`src.modes.check_prompt`），user message 由
        :func:`src.modes.check_prompt.build_check_user_message` 依當前
        階段組裝。

        stage 合法值：
        - ``"summary"``：第一階段精緻快速摘要
        - ``"full"``：第二階段完整報告（使用者選 1）
        - ``"warning"``：警示文字稿（使用者選 2，圖片降級）

        子類別應覆寫此方法改為真實 LLM API 呼叫。預設實作走 fallback
        模板（離線確定性回應），讓測試與無 API key 的環境仍可走完整流程。
        """
        # 晚期 import 避免循環：check_prompt 不依賴 base_client。
        from src.modes.check_prompt import (
            STAGE_SUMMARY,
            STAGE_FULL,
            STAGE_WARNING,
        )
        if stage == STAGE_SUMMARY:
            return _fallback_check_summary(user_content)
        if stage == STAGE_FULL:
            return _fallback_check_full(user_content, stage1_output)
        if stage == STAGE_WARNING:
            return _fallback_check_warning(user_content, stage1_output)
        raise ValueError(f"Unknown /check stage: {stage!r}")


# ---------------------------------------------------------------------------
# /check fallback 模板（v0.6 PR-D）
# ---------------------------------------------------------------------------
# 當 client 沒實作 run_check_stage 真實 API 呼叫時（離線模式 / OpenAI 骨架
# 後端 / 測試 mock），用這組模板回傳。模板刻意不裝「分析出結果」——而是
# 明確告訴使用者「目前為離線模式，ISF 分析需要 API 驅動」，避免騙人。

def _fallback_check_summary(user_content: str) -> str:
    """第一階段的 fallback 輸出：結構化版面但明確標註離線。"""
    preview = (user_content or "").strip().replace("\n", " ")
    if len(preview) > 30:
        preview = preview[:30] + "…"
    return (
        "📊 資訊完整性快速摘要\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📂 類別　｜🔍 未分類（離線 fallback 模式）\n"
        "\n"
        "🎯 可信度　｜⚠️ 未評估　《需 API 驅動》\n"
        "🧩 操控手法｜（離線模式無法逐項分析）\n"
        "📉 缺漏資訊｜（離線模式無法逐項分析）\n"
        "📘 可驗事實｜（離線模式無法逐項分析）\n"
        "\n"
        f"⚠️ 警語　　｜「收到內容：{preview}——目前離線，未做 ISF 判讀。」\n"
        "💡 建議行動｜設定 CLAUDE_API_KEY 後再貼一次；"
        "若懷疑是詐騙請直接撥 165 查證。\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "📍 請選擇下一步行動：\n"
        "1️⃣ 查看完整分析報告（第二階段）\n"
        "2️⃣ 生成警示文字稿\n"
        "3️⃣ 結束分析"
    )


def _fallback_check_full(
    user_content: str,
    stage1_output: str | None,
) -> str:
    """第二階段的 fallback 輸出。"""
    return (
        "🔍 完整建議與判斷報告（離線 fallback）\n\n"
        "1. 摘要：目前為離線模式，無法逐項做 ISF 評分與因果分析。\n"
        "2. Zone A（可驗證事實）：（離線模式無法分析）\n"
        "3. Zone B（缺漏分析）：（離線模式無法分析）\n"
        "4. 操控手法詳述：（離線模式無法分析）\n"
        "5. 查證方向：設定 CLAUDE_API_KEY 後重新貼入內容；"
        "若是緊急疑似詐騙，撥 165 或查官方公告。\n"
        "6. 風險結論：未評估。\n"
        "7. 建議行動：暫不採信任何「快速判讀」，用官方來源查證。"
    )


def _fallback_check_warning(
    user_content: str,
    stage1_output: str | None,
) -> str:
    """警示文字稿 fallback（圖片降級）。"""
    return (
        "⚠️ 小心詐騙警示 ⚠️\n"
        "━━━━━━━━━━━━━━━\n"
        "🚨 主標：不確定？先撥 165！\n"
        "💰 副標：目前離線無法判讀，建議直接查證\n"
        "🎨 設計建議：黃色底 + 黑色字（提醒級）\n"
        "━━━━━━━━━━━━━━━\n"
        "\n"
        "[WARNING_META]\n"
        "risk_level: unknown\n"
        "color_scheme: yellow_gray\n"
        "hotline: 165\n"
        "source: fallback_template"
    )
