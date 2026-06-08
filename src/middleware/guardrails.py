"""v0.5 腳本護欄。

依 20260416_逗福Tofu_端點檢索與元動機偵測_邏輯鏈_v0_5.md 定義：

```
MAX_COST_USD = 5.0
MAX_SESSIONS = 500
MAX_RUNTIME_MINUTES = 120

# 任一觸發 → 中斷 → 輸出已完成結果
# 每題跑之前先算預估成本，超過 MAX_COST_USD 不開始
```

這個模組是純程式碼，不依賴 LLM。所有累積狀態存在 `GuardrailTracker`
實例裡，由 caller 負責生命週期。CLI 在每次 API call 前呼叫
`check_before_call()`，任何一條護欄觸發即 raise `GuardrailTripped`。

設計要點：
- 成本估算走固定 per-call 估值，來自 v0.5 成本表（一次互動 ≈ $0.012）。
- session 計數以「一次完整 run_one_interaction 呼叫一次 record_session()」
  為單位；guardrails 不動態地分辨 restate vs execute 各算不算半次。
- runtime 以 `__init__` 當下的 wall time 為 0 點。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

# v0.5 成本表（見 17.5 節 token cost lockdown）
# input 合計 ≈ 3,650 tokens、output ≈ 800 tokens，一次 call ≈ $0.006，
# 一次互動（restate + execute = 2 calls）≈ $0.012。
DEFAULT_COST_PER_CALL_USD = 0.006
DEFAULT_CALLS_PER_SESSION = 2

# v0.5 護欄常數（17.7 節）
MAX_COST_USD = 5.0
MAX_SESSIONS = 500
MAX_RUNTIME_MINUTES = 120


class GuardrailTripped(RuntimeError):
    """任一護欄觸發時拋出。

    caller 應截斷互動流程，輸出已完成結果並退出。這不是要讓使用者「繞過」
    護欄，而是明確告知「v0.5 的這三條硬線之一被觸發了」。
    """

    def __init__(self, reason: str, field_name: str, current: float, limit: float) -> None:
        super().__init__(reason)
        self.field_name = field_name
        self.current = current
        self.limit = limit


@dataclass
class GuardrailTracker:
    """累積追蹤成本、session 數、運行時間。

    使用方式（CLI 單次啟動內共用一個實例）：

        tracker = GuardrailTracker()
        for turn in session:
            tracker.check_before_call()
            ...  # 呼叫 LLM
            tracker.record_call()
            ...  # 一次互動結束
            tracker.record_session()
    """

    max_cost_usd: float = MAX_COST_USD
    max_sessions: int = MAX_SESSIONS
    max_runtime_minutes: int = MAX_RUNTIME_MINUTES
    cost_per_call_usd: float = DEFAULT_COST_PER_CALL_USD
    calls_per_session: int = DEFAULT_CALLS_PER_SESSION

    cumulative_cost_usd: float = 0.0
    sessions_completed: int = 0
    # 注意：started_at 在 __post_init__ 中被覆蓋為 time_provider()，
    # 這樣測試時注入 fake time_provider 後，起點時間也走 fake。
    started_at: float = 0.0

    # 測試用注入點：預設 time.monotonic，可被單元測試替換
    time_provider: Callable[[], float] = field(default=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        # 如果呼叫端沒明確指定 started_at，就用 time_provider() 當作起點
        if self.started_at == 0.0:
            self.started_at = self.time_provider()

    def elapsed_seconds(self) -> float:
        return max(0.0, self.time_provider() - self.started_at)

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0

    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------
    def would_exceed_cost(self, extra_calls: int = 1) -> bool:
        """預估再跑 extra_calls 次會不會超 MAX_COST_USD。"""
        projected = self.cumulative_cost_usd + extra_calls * self.cost_per_call_usd
        return projected > self.max_cost_usd

    def remaining_budget_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.cumulative_cost_usd)

    def status_line(self) -> str:
        return (
            f"[guardrails] cost=${self.cumulative_cost_usd:.3f}/"
            f"${self.max_cost_usd:.2f}"
            f" | sessions={self.sessions_completed}/{self.max_sessions}"
            f" | runtime={self.elapsed_minutes():.1f}/{self.max_runtime_minutes} min"
        )

    # ------------------------------------------------------------------
    # 檢查
    # ------------------------------------------------------------------
    def check_before_call(
        self,
        extra_calls: int | None = None,
    ) -> None:
        """在呼叫 LLM 之前呼叫。

        任一條件觸發 → raise GuardrailTripped：
        1. 預估下一次呼叫會讓累積成本超過 max_cost_usd
        2. sessions_completed >= max_sessions
        3. elapsed_minutes >= max_runtime_minutes

        extra_calls 預設為 calls_per_session（= 2，代表一次互動 restate +
        execute 兩次 API call）；可由 caller 覆寫。
        """
        if extra_calls is None:
            extra_calls = self.calls_per_session

        if self.would_exceed_cost(extra_calls=extra_calls):
            projected = (
                self.cumulative_cost_usd + extra_calls * self.cost_per_call_usd
            )
            raise GuardrailTripped(
                reason=(
                    f"成本護欄觸發：預估下一次互動會讓累積成本達"
                    f" ${projected:.3f}，超過上限 ${self.max_cost_usd:.2f}"
                ),
                field_name="cost",
                current=projected,
                limit=self.max_cost_usd,
            )

        if self.sessions_completed >= self.max_sessions:
            raise GuardrailTripped(
                reason=(
                    f"Session 護欄觸發：已完成 {self.sessions_completed} 次互動，"
                    f"達上限 {self.max_sessions}"
                ),
                field_name="sessions",
                current=self.sessions_completed,
                limit=self.max_sessions,
            )

        elapsed_min = self.elapsed_minutes()
        if elapsed_min >= self.max_runtime_minutes:
            raise GuardrailTripped(
                reason=(
                    f"運行時間護欄觸發：已跑 {elapsed_min:.1f} 分鐘，"
                    f"達上限 {self.max_runtime_minutes}"
                ),
                field_name="runtime",
                current=elapsed_min,
                limit=float(self.max_runtime_minutes),
            )

    # ------------------------------------------------------------------
    # 累積
    # ------------------------------------------------------------------
    def record_call(self, cost_usd: float | None = None) -> None:
        """成功呼叫一次 LLM 之後呼叫，累積成本。"""
        c = self.cost_per_call_usd if cost_usd is None else float(cost_usd)
        self.cumulative_cost_usd += c

    def record_session(self) -> None:
        """一次完整的 run_one_interaction 結束後呼叫。"""
        self.sessions_completed += 1

    def reset(self) -> None:
        self.cumulative_cost_usd = 0.0
        self.sessions_completed = 0
        self.started_at = self.time_provider()
