"""批次評測（Batch Mode）——省 50% 成本的題庫跑題器。

用途
----
在不跑完整 `run_one_interaction` 思考引擎的前提下，把 500 題直接餵給 LLM
的五個模式（default / free / risk / propose / check），透過 Anthropic
Message Batches API 取得 50% 折扣。典型使用者：測題庫的人想看各模式在
大量輸入下的輸出風格，不在乎補位/畫像/端點記憶。

重點
----
- 單輪模式（default / free / risk / check）：所有題目組成**單次** batch
  送出，等回來後寫入結果。
- /propose：需要 5 輪 loop（每輪依賴前一輪輸出）。用**多波 batch**：
  第 1 波所有題目的第 1 輪一起送，收回後塞進第 2 波的 `extra_context`，
  依此類推。最後第 6 波用 `propose_final` 交卷。

成本估算
--------
Haiku 4.5 batch 單價約 $0.4/$2 per MT（輸入/輸出），單題約 $0.001；
/propose 單題 5-6 次約 $0.005。500 題平均 $0.82 量級。

CLI
---
透過 ``eval/batch_mode_test.py`` 執行。

設計決策
--------
- 這個模組**不**碰 EndpointStore / ProfileStore——純無狀態、不寫畫像。
  Batch 模式的目的是測「給定題目 + 模式時，LLM 會怎麼答」，而不是測
  完整的認知中間層行為。
- 故意不使用 ``execute_task``：``execute_task`` 會拼接畫像、密碼表、
  extra_context 等額外欄位，這些在純題庫測試中不需要，反而會讓輸出
  帶上不相關的系統腔。真的要測完整流程請用 ``longmemeval_real_test``。
- /check 沿用 :mod:`src.modes.check_prompt` 的 ISF v3.2 原文，確保和
  互動模式下的 `/check` 輸出一致。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from src.api.claude_client import (
    DEFAULT_MAX_TOKENS,
    LLMClient,
    OUTPUT_PROMPTS,
    TOFU_IDENTITY_PROMPT,
)

logger = logging.getLogger("tofu.batch_runner")


# ---------------------------------------------------------------------------
# Haiku 4.5 batch 單價（USD / 百萬 token）：0.8 / 4 的 50% = 0.4 / 2
# 這是粗估，實際計費以 Anthropic 帳單為準。
# ---------------------------------------------------------------------------
BATCH_PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 0.40, "output": 2.00},
    "claude-opus-4-6": {"input": 7.50, "output": 37.50},
    "claude-sonnet-4-6": {"input": 1.50, "output": 7.50},
}

PROPOSE_MAX_ROUNDS = 5  # 與 main.MAX_PROPOSE_ROUNDS 對齊
BATCH_MODES_SINGLE_CALL = frozenset({"default", "free", "risk", "check"})
BATCH_MODES_MULTI_CALL = frozenset({"propose"})
VALID_BATCH_MODES = BATCH_MODES_SINGLE_CALL | BATCH_MODES_MULTI_CALL


# ---------------------------------------------------------------------------
# Prompt 組裝
# ---------------------------------------------------------------------------
def _build_mode_system_prompt(mode: str) -> str:
    """產生指定模式的 system prompt（身份 + 任務）。"""
    if mode == "check":
        from src.modes.check_prompt import build_check_system_prompt
        return build_check_system_prompt()

    if mode == "propose_final":
        task_prompt = OUTPUT_PROMPTS["propose_final"]
    elif mode == "propose":
        task_prompt = OUTPUT_PROMPTS["propose"]
    elif mode in OUTPUT_PROMPTS:
        task_prompt = OUTPUT_PROMPTS[mode]
    else:
        raise ValueError(f"Unknown batch mode: {mode!r}")

    return TOFU_IDENTITY_PROMPT + "\n\n" + task_prompt


def _build_mode_user_prompt(
    mode: str,
    question: str,
    *,
    extra_context: str | None = None,
) -> str:
    """產生指定模式的 user message。"""
    if mode == "check":
        from src.modes.check_prompt import (
            build_check_user_message,
            STAGE_SUMMARY,
        )
        return build_check_user_message(
            stage=STAGE_SUMMARY,
            user_content=question,
        )

    # default / free / risk / propose / propose_final 都走 execute_task 風格
    extra_block = ""
    if extra_context:
        extra_block = (
            "## 前幾輪紀錄（/propose 接續上下文）\n\n"
            + extra_context.strip()
            + "\n\n---\n\n"
        )

    return (
        f"{extra_block}"
        f"目標：{question}\n"
        f"動機：（未提供）\n"
        f"限制條件：\n（無）\n"
        f"\n"
        f"請給出建議或步驟。"
    )


def _max_tokens_for_mode(mode: str) -> int:
    if mode == "check":
        return 800
    if mode in ("propose", "propose_final"):
        return 1200
    return DEFAULT_MAX_TOKENS


def build_request(
    custom_id: str,
    mode: str,
    question: str,
    *,
    model: str,
    extra_context: str | None = None,
) -> dict[str, Any]:
    """組一筆 batch request。"""
    system = _build_mode_system_prompt(mode)
    user = _build_mode_user_prompt(mode, question, extra_context=extra_context)
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": _max_tokens_for_mode(mode),
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
    }


# ---------------------------------------------------------------------------
# 單輪 batch 跑法（default / free / risk / check）
# ---------------------------------------------------------------------------
def run_single_round_batch(
    client: LLMClient,
    items: Iterable[dict[str, Any]],
    *,
    mode: str,
    model: str,
    poll_interval: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """把 items 組成一批送出去，等回來後回傳 {id: result}。

    items: 每筆 {"id": "q001", "question": "..."}
    mode: default / free / risk / check
    """
    if mode not in BATCH_MODES_SINGLE_CALL:
        raise ValueError(
            f"run_single_round_batch 只支援 {sorted(BATCH_MODES_SINGLE_CALL)}；"
            f"/propose 請用 run_propose_batch。"
        )

    requests = [
        build_request(it["id"], mode, it["question"], model=model)
        for it in items
    ]
    if not requests:
        return {}

    logger.info("[%s] 送出 %d 筆 batch 請求...", mode, len(requests))
    batch_id = client.submit_batch(requests)
    client.wait_for_batch(batch_id, poll_interval=poll_interval)
    return client.get_batch_results(batch_id)


# ---------------------------------------------------------------------------
# /propose 多波 batch（5 輪 loop + 1 輪交卷）
# ---------------------------------------------------------------------------
def _format_round_history(history: list[dict[str, str]]) -> str:
    """把前幾輪輸出壓成一個 extra_context 字串。"""
    lines: list[str] = []
    for row in history:
        lines.append(f"### 第 {row['round']} 輪\n{row['text']}")
    return "\n\n".join(lines)


def run_propose_batch(
    client: LLMClient,
    items: Iterable[dict[str, Any]],
    *,
    model: str,
    poll_interval: float = 30.0,
    max_rounds: int = PROPOSE_MAX_ROUNDS,
) -> dict[str, dict[str, Any]]:
    """/propose 多波 batch：每一輪用前一輪結果塞 extra_context。

    回傳 {id: {"rounds": [{round, text}, ...], "final": "..."}}

    注意：這裡不做 submit_now 提前交卷；所有題目都跑滿 5 輪才進交卷輪。
    原因是 batch 模式天生適合同步推進——不同題目 submit_now 時間不同
    會打斷 batching 的一致性。真的要用 submit_now 提前交卷請走互動模式。
    """
    items_list = list(items)
    if not items_list:
        return {}

    # 依 id 收集每題歷史
    histories: dict[str, list[dict[str, str]]] = {it["id"]: [] for it in items_list}
    questions: dict[str, str] = {it["id"]: it["question"] for it in items_list}

    logger.info(
        "[propose] 開始 %d 題 × %d 輪 loop（共 %d 波 batch）",
        len(items_list), max_rounds, max_rounds + 1,
    )

    # 1-N 輪：收資訊輪
    for round_num in range(1, max_rounds + 1):
        requests: list[dict[str, Any]] = []
        for qid, question in questions.items():
            extra = _format_round_history(histories[qid]) or None
            custom_id = f"{qid}::r{round_num}"
            requests.append(
                build_request(
                    custom_id, "propose", question,
                    model=model, extra_context=extra,
                )
            )

        logger.info("[propose round %d/%d] 送 %d 筆...",
                    round_num, max_rounds, len(requests))
        batch_id = client.submit_batch(requests)
        client.wait_for_batch(batch_id, poll_interval=poll_interval)
        round_results = client.get_batch_results(batch_id)

        # 把結果塞回 histories
        for qid in list(questions.keys()):
            custom_id = f"{qid}::r{round_num}"
            res = round_results.get(custom_id, {})
            text = res.get("text") or f"[ERROR round {round_num}] {res.get('error', 'missing')}"
            histories[qid].append({"round": round_num, "text": text})

    # 最後一波：交卷輪 propose_final
    final_requests: list[dict[str, Any]] = []
    for qid, question in questions.items():
        extra = _format_round_history(histories[qid]) or None
        final_requests.append(
            build_request(
                f"{qid}::final", "propose_final", question,
                model=model, extra_context=extra,
            )
        )
    logger.info("[propose final] 送 %d 筆交卷請求...", len(final_requests))
    batch_id = client.submit_batch(final_requests)
    client.wait_for_batch(batch_id, poll_interval=poll_interval)
    final_results = client.get_batch_results(batch_id)

    # 組裝最終輸出
    out: dict[str, dict[str, Any]] = {}
    for qid in questions:
        fres = final_results.get(f"{qid}::final", {})
        out[qid] = {
            "rounds": histories[qid],
            "final": fres.get("text") or f"[ERROR final] {fres.get('error', 'missing')}",
            "final_usage": fres.get("usage", {}),
        }
    return out


# ---------------------------------------------------------------------------
# 成本估算
# ---------------------------------------------------------------------------
def estimate_batch_cost(
    results: dict[str, dict[str, Any]],
    model: str,
) -> dict[str, float]:
    """根據 batch 結果裡的 usage 算總成本（batch 價，已打 5 折）。"""
    price = BATCH_PRICE_PER_MTOK.get(model)
    if price is None:
        return {"input_tokens": 0, "output_tokens": 0, "usd": 0.0, "note": f"未知 model: {model}"}

    in_tok = 0
    out_tok = 0
    for res in results.values():
        usage = res.get("usage") if isinstance(res, dict) else None
        if not usage:
            continue
        in_tok += int(usage.get("input_tokens", 0) or 0)
        out_tok += int(usage.get("output_tokens", 0) or 0)

    usd = (in_tok / 1_000_000) * price["input"] + (out_tok / 1_000_000) * price["output"]
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd": round(usd, 4),
    }
