"""OpenAI 相容後端（P1-2）。

DeepSeek V4 的 API 跟 OpenAI 格式相容，所以 OpenAIClient 可以同時支援
GPT 與 DeepSeek，只要改 base_url 與 model 參數。

最小可行版本：
- generate_restate 完整實作（能通介面）
- merge_confirmation / execute_task / analyze_deviation 骨架：
  沒有 API key 時 fallback 到 claude_client 的純程式碼 fallback 實作。
- 有 API key 時透過 OpenAI 相容的 chat completions 介面呼叫。

依規格書 P1-2「不要讓多模型重構破壞 fallback 模式」。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from src.api.base_client import BaseLLMClient
from src.api.claude_client import (
    TOFU_IDENTITY_PROMPT,
    LLMClientError,
    _fallback_analyze_deviation,
    _fallback_execute_task,
    _fallback_generate_restate,
    _fallback_merge_confirmation,
)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _extract_json(raw: str) -> dict[str, Any]:
    """與 claude_client._extract_json 對齊的 JSON 擷取工具。"""
    if not raw:
        raise LLMClientError("LLM 回傳空字串。")
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMClientError(f"LLM 回傳中找不到 JSON：{raw[:200]}")
        candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMClientError(
            f"LLM JSON 解析失敗：{exc}；原文片段：{candidate[:200]}"
        ) from exc


class OpenAIClient(BaseLLMClient):
    """OpenAI / DeepSeek 相容後端。

    最小可行版本：
    - 沒有 OPENAI_API_KEY 時進入 fallback 模式（完全對齊 ClaudeClient 行為）。
    - 有 key 時以 OpenAI 相容的 chat completions 介面呼叫。
    - base_url 可切到 DeepSeek（https://api.deepseek.com/v1）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        notifier: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self._notifier = notifier
        self._client = None
        self.fallback_mode = False

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            self.fallback_mode = True
            return

        # 延遲 import 讓 openai SDK 變成可選依賴。
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.fallback_mode = True
            self._notify(
                "[逗福Tofu] 找不到 openai 套件，暫時進入 fallback 模式。"
                "請 `pip install openai` 後再試。"
            )
            return

        self._client = OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
        )

    # ------------------------------------------------------------------
    def _notify(self, msg: str) -> None:
        if self._notifier is not None:
            try:
                self._notifier(msg)
            except Exception:
                pass

    def _chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """最小的 chat completions 呼叫——無 retry，骨架版。"""
        if self._client is None:
            raise LLMClientError("fallback mode should not call _chat")
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # pragma: no cover - network path
            raise LLMClientError(f"OpenAI API 呼叫失敗：{exc}") from exc

        try:
            return (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise LLMClientError(f"OpenAI 回傳結構異常：{exc}") from exc

    # ------------------------------------------------------------------
    def generate_restate(
        self,
        user_input: str,
        baseline_summary: dict[str, Any],
        allowed_categories: list[str],
    ) -> dict[str, Any]:
        if self.fallback_mode:
            return _fallback_generate_restate(
                user_input, baseline_summary, allowed_categories
            )

        cats_hint = "、".join(allowed_categories) if allowed_categories else "（無）"
        freq_cats = baseline_summary.get("top_gap_categories", [])
        system_prompt = (
            TOFU_IDENTITY_PROMPT
            + "\n\n## 你現在的任務：復述確認\n\n"
            "用你自己的話復述使用者的輸入，並把常被漏掉的面向自然帶出來。\n"
            "嚴格以 JSON 格式回覆。"
        )
        user_prompt = (
            f"使用者輸入：{user_input}\n"
            f"常被 TOFU 補位的面向：{freq_cats or '（無）'}\n"
            f"允許的 gap_categories：{cats_hint}\n"
            f"\n"
            f"請回覆 JSON：\n"
            f'{{"restate_text": "...", '
            f'"gap_questions": ["..."], '
            f'"gap_categories": ["..."], '
            f'"inferred_goal": "...", '
            f'"inferred_motivation": "...", '
            f'"inferred_constraints": ["..."]}}'
        )
        raw = self._chat(system_prompt, user_prompt)
        data = _extract_json(raw)
        return {
            "restate_text": str(data.get("restate_text", "")).strip(),
            "gap_questions": [str(q) for q in data.get("gap_questions", []) if q],
            "gap_categories": [
                str(c).strip().lower() for c in data.get("gap_categories", []) if c
            ],
            "inferred_goal": str(data.get("inferred_goal", "")).strip(),
            "inferred_motivation": str(
                data.get("inferred_motivation", "")
            ).strip(),
            "inferred_constraints": [
                str(c).strip() for c in data.get("inferred_constraints", []) if c
            ],
        }

    def merge_confirmation(
        self,
        original_input: str,
        restate_text: str,
        user_confirmation_text: str,
    ) -> dict[str, Any]:
        # 骨架版：無 API key 時 fallback；有 API key 時直接用 fallback 的結構，
        # 避免 placeholder LLM 呼叫出問題。後續補齊。
        if self.fallback_mode:
            return _fallback_merge_confirmation(
                original_input, restate_text, user_confirmation_text
            )
        # 暫時用 fallback 邏輯（確保介面可通）
        return _fallback_merge_confirmation(
            original_input, restate_text, user_confirmation_text
        )

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
        extra_context: str | None = None,
    ) -> str:
        # OpenAI 後端目前仍為骨架版，execute_task 全走 fallback 模板；
        # 五模式開關（v0.6 PR-B）透過 fallback 的 output_mode 分支生效。
        # extra_context（v0.6 PR-C）接受但 fallback 不用，保持簽名相容。
        return _fallback_execute_task(
            goal, motivation, constraints,
            strategy_brief=strategy_brief,
            output_mode=output_mode,
            extra_context=extra_context,
        )

    def analyze_deviation(
        self,
        goal: str,
        result: str,
        deviation: str,
    ) -> str:
        if self.fallback_mode:
            return _fallback_analyze_deviation(goal, result, deviation)
        return _fallback_analyze_deviation(goal, result, deviation)

    def answer_with_profile(
        self,
        haystack: str,
        question: str,
        profile: dict[str, Any] | None = None,
    ) -> str:
        """LongMemEval 兩階段推論第二階段——OpenAI 後端的骨架版實作。

        對齊 ClaudeClient 的簽名。fallback 模式下回傳固定字串，確保
        介面可通；非 fallback 模式則明確拋錯，避免誤把 placeholder
        當成真實 OpenAI 回覆。
        """
        if self.fallback_mode:
            return f"[fallback] 無法回答：{question}"
        raise NotImplementedError(
            "OpenAIClient.answer_with_profile 尚未實作非 fallback 模式；"
            "請改用 fallback 模式或將此方法接到實際的 OpenAI `_chat` 流程。"
        )
