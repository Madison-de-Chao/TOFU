"""逗福Tofu 公測版 CLI 入口。

執行方式：
    python src/main.py

第一次執行前建議（可選）：
    cp .env.example .env
    # 編輯 .env 填入 CLAUDE_API_KEY（可留空，會進入 fallback 模式）
    pip install -r requirements.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# 讓 `python src/main.py` 和 `python -m src.main` 都能 work
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.api.claude_client import LLMClient, LLMClientError, build_atl3_retry_hint
from src.middleware import baseline as baseline_mod
from src.middleware import confirmation as confirm_mod
from src.middleware import intake_form as form_mod
from src.middleware.endpoint import (
    DEFAULT_TOP_K,
    RECENT_FLOOR_N,
    EndpointStore,
    _now_iso,
    encode_retrieved_endpoints,
    make_end_data,
    new_event_id,
    retrieve_top_k_endpoints,
)
from src.middleware.guardrails import (
    GuardrailTracker,
    GuardrailTripped,
)
from src.middleware.propose import (
    format_propose_history,
    parse_propose_meta,
)
from src.middleware.response_validator import (
    build_retry_hint,
    detect_evasion_phrases,
    is_evasive_response,
    mark_degraded,
)
from src.middleware import check as check_mod
from src.modes.check_prompt import (
    STAGE_SUMMARY,
    STAGE_FULL,
    STAGE_WARNING,
    STAGE_END,
)
from src.middleware.user_profile import (
    ProfileStore,
    render_profile_summary,
)
from src.middleware.emotion_detector import detect_emotion_rule_based
from src.middleware.cooling_mode import (
    COOLING_MODE_PROMPT_ADDENDUM,
    should_enter_cooling_mode,
)
from src.middleware.zone_history import (
    ZoneHistoryStore,
    atl4_consistency_check,
    compute_question_signature,
    extract_question_features,
)
from src.utils.meta import META_KEYWORDS, is_meta_query
from src.utils.pillars_map import VALID_STEMS, VALID_BRANCHES
from src.middleware.word_tagger import (
    HaikuWordTagger,
    WordTaggerGuardrail,
    tag_and_index_endpoint,
)
from src.middleware.word_index import WordIndexStore


# v0.6 PR-E：production 端點檔升級為 JSONL（append-only + filelock）。
# legacy `data/endpoints.json` 存在時 EndpointStore 會自動 migrate。
DATA_PATH = os.environ.get(
    "TOFU_ENDPOINTS_PATH",
    str(_PROJECT_ROOT / "data" / "endpoints.jsonl"),
)

PROFILE_PATH = os.environ.get(
    "TOFU_PROFILE_PATH",
    str(_PROJECT_ROOT / "data" / "user_profile.json"),
)

# v0.6 PR-D：/check 模式的 session state 目錄
SESSIONS_DIR = os.environ.get(
    "TOFU_SESSIONS_DIR",
    str(_PROJECT_ROOT / "data" / "sessions"),
)

# v4.0 P0-2：ATL-4 跨輪一致性歷史檔（per-user JSONL）
ZONE_HISTORY_PATH = os.environ.get(
    "TOFU_ZONE_HISTORY_PATH",
    str(_PROJECT_ROOT / "data" / "zone_history.jsonl"),
)

# v0.6 PR-E：詞性分類檔目錄（data/words/）
WORDS_DIR = os.environ.get(
    "TOFU_WORDS_DIR",
    str(_PROJECT_ROOT / "data" / "words"),
)

# v0.6 PR-E：模組層級的 tagger/guardrail/word_store 懶初始化（每 process 一份）。
# 由 _try_tag_endpoint() 按需初始化；main() 不必手動建立。
_word_tagger: HaikuWordTagger | None = None
_word_guardrail: WordTaggerGuardrail | None = None
_word_store: WordIndexStore | None = None


def _get_word_pipeline() -> tuple[HaikuWordTagger, WordTaggerGuardrail, WordIndexStore]:
    """懶初始化詞標記 pipeline。thread-safe：CPython GIL 保護簡單賦值。"""
    global _word_tagger, _word_guardrail, _word_store
    if _word_tagger is None:
        _word_tagger = HaikuWordTagger()
    if _word_guardrail is None:
        guardrail_file = os.path.join(WORDS_DIR, "_guardrail.json")
        os.makedirs(WORDS_DIR, exist_ok=True)
        _word_guardrail = WordTaggerGuardrail(guardrail_file)
    if _word_store is None:
        _word_store = WordIndexStore(WORDS_DIR)
    return _word_tagger, _word_guardrail, _word_store


def _try_tag_endpoint(
    endpoint_id: str,
    user_input: str,
    result: str,
) -> None:
    """互動結束後在背景觸發詞性標記，失敗靜默（不阻塞主流程）。

    依 Spec v2 §3.5 + MOMO 三條 fallback 規則：
    - Rule 1：API 失敗 → tagged_at:null，不中斷。
    - Rule 2：格式錯誤 → 在 HaikuWordTagger 內部重試一次，仍失敗 → null。
    - Rule 3：背景補標 job 為可選優化，不進 PR-E。
    """
    try:
        tagger, guardrail, word_store = _get_word_pipeline()
        text = f"{user_input}\n{result}".strip()
        tag_and_index_endpoint(
            endpoint_id=endpoint_id,
            text=text,
            tagger=tagger,
            guardrail=guardrail,
            word_store=word_store,
        )
    except Exception as exc:  # pragma: no cover - 防禦
        print(
            f"[逗福Tofu 警告] 詞性標記例外（endpoint_id={endpoint_id}）：{exc}",
            file=sys.stderr,
            flush=True,
        )


# 滿意度問答節奏：每 N 次互動問一次，減少煩躁感
SATISFACTION_INTERVAL = 3


# ----------------------------------------------------------------------
# 小工具：CLI I/O
# ----------------------------------------------------------------------
# v0.6 PR-H audit #4：`--batch` 模式支援（非互動式、從 stdin 餵輸入）。
# 目的：讓 live_test_v06.py 類型的自動化腳本可以 pipe 多行輸入進來跑，
# 不用 pexpect / spawn。為 True 時：
# - `_prompt()` 不印 "> " 前綴（避免污染 stdout 被程式讀取）
# - 主迴圈不印狀態列（每次 prompt 前的一行統計）
# 其他行為（EOF → /exit、滿意度提問、/free 不反問）皆同互動式。
_BATCH_MODE = False


def _prompt(text: str) -> str:
    try:
        return input("" if _BATCH_MODE else text)
    except EOFError:
        return "/exit"


def _print(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------
# 指令說明表
# ----------------------------------------------------------------------
HELP_ITEMS: list[tuple[str, str]] = [
    ("/help", "顯示所有可用指令"),
    ("/about", "了解逗福Tofu 是什麼"),
    ("/free <內容>", "直接給建議，不反問（假設補位 + 明確標註）"),
    ("/risk <內容>", "輸出風險清單（ATL-1 falsification 格式）"),
    ("/propose <內容>", "5 輪自問自答提案（補夠資訊提前交卷，最多 6 次 LLM 呼叫）"),
    ("/check <內容>", "ISF 資訊完整性檢驗（貼疑似詐騙或可疑訊息；輸入 1/2/3 接續）"),
    ("/form", "顯示個人化提問單（累積 5 筆互動後可用）"),
    ("/baseline", "顯示目前的邏輯基線摘要"),
    ("/history", "顯示最近 5 筆互動的摘要"),
    ("/stats", "顯示使用統計（修正率、滿意度、常被補位面向、Zone 分布）"),
    ("/blackbox", "顯示最近的黑盒子回溯紀錄（偏差觸發時自動記錄）"),
    ("/profile", "顯示使用者畫像（溝通風格、偏好、滿意度）"),
    ("/pillars", "輸入四柱天干地支（有多少記多少）"),
    ("/reset", "清除所有端點紀錄（需二次確認）"),
    ("/export", "把端點紀錄匯出為 Markdown 檔案"),
    ("/exit", "離開 逗福Tofu"),
]


# v0.6 PR-B：五模式 slash 指令名稱集合
MODE_COMMANDS: frozenset[str] = frozenset({"free", "risk", "propose", "check"})


def _parse_mode_command(line: str) -> tuple[str, str] | None:
    """解析 `/mode <content>` 型指令。

    回傳 ``(mode, content)``，若不是模式指令則回傳 ``None``。
    - ``/free 幫我看看`` → ``("free", "幫我看看")``
    - ``/risk`` → ``("risk", "")``（content 為空，由上層決定怎麼提示）
    - ``/propose`` / ``/check`` 同樣被識別，由上層印「即將上線」提示。
    - ``/unknown xxx`` → ``None``（維持原本「未知指令」提示）
    """
    if not line.startswith("/"):
        return None
    stripped = line[1:].strip()
    if not stripped:
        return None
    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()
    if cmd not in MODE_COMMANDS:
        return None
    content = parts[1] if len(parts) > 1 else ""
    return cmd, content


def _render_help() -> str:
    lines = ["[逗福Tofu] 可用指令："]
    for cmd, desc in HELP_ITEMS:
        lines.append(f"  {cmd:<10} — {desc}")
    lines.append("")
    lines.append("了解工具來源：yyuniverse.com")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Meta 問題攔截與 /about 自我介紹
# ----------------------------------------------------------------------
# 使用者用自然語言問「你是什麼」「你能做什麼」這類問題時，
# 不應該進入復述補位流程（會問他預算和時間）；直接印出 /about 文字即可。
# `META_KEYWORDS` 與 `is_meta_query` 由 src.utils.meta 集中提供，
# main.py 與 claude_client.py 共用同一份來源（見 DESIGN.md §16）。

_ABOUT_TEXT = (
    "[逗福Tofu] 我是逗福Tofu，一個認知中間層。\n"
    "\n"
    "我跑在你和 AI 之間，做三件事：\n"
    "1. 復述確認：你說完需求後，我先復述一遍確認理解正確，同時帶出你可能漏掉的面向。\n"
    "2. 補位：根據你過去的互動模式，主動提醒你這次沒提到但通常會考慮的事。\n"
    "3. 端點記錄：只記你的目標和結果，中間過程不記。\n"
    "\n"
    "跟其他 AI 最大的不同：\n"
    "其他 AI 被訓練成說你想聽的話。我被設計成說你需要聽的話。\n"
    "我先確認再記，不是直接記。記憶存在程式碼層，換 AI 不丟。\n"
    "\n"
    "三條底線：說真話、說人話、守住邊界。\n"
    "\n"
    "試試看：直接輸入你想做的事。\n"
    "輸入 /help 查看所有指令。\n"
    "了解更多：yyuniverse.com"
)


# ----------------------------------------------------------------------
# 歡迎訊息 / 互動狀態列
# ----------------------------------------------------------------------
WELCOME_TEXT = (
    "歡迎使用 逗福Tofu — 認知中間層。\n"
    "\n"
    "逗福Tofu 會先復述你的需求並確認理解正確，\n"
    "同時帶出你可能漏掉的面向。\n"
    "\n"
    "試試看：直接輸入一個你想做的事。\n"
    "輸入 /help 查看所有指令。\n"
    "\n"
    "內建思維工具來自元壹宇宙（YuanYi Universe）。\n"
    "了解更多：yyuniverse.com"
)


def _status_line(store: EndpointStore) -> str:
    completed = store.completed_count()
    baseline = baseline_mod.compute_baseline(store.all())
    mature = baseline_mod.is_mature(baseline)
    baseline_part = "基線已建立" if mature else "基線建立中"
    form_part = "提問單可用" if form_mod.should_generate(completed) else "提問單未就緒"
    return f"[互動 #{completed} | {baseline_part} | {form_part}]"


# ----------------------------------------------------------------------
# 滿意度收集
# ----------------------------------------------------------------------
def _should_ask_satisfaction(interaction_index: int) -> bool:
    """每 SATISFACTION_INTERVAL 次問一次。index 從 1 開始。"""
    if interaction_index <= 0:
        return False
    return interaction_index % SATISFACTION_INTERVAL == 0


def _collect_satisfaction() -> tuple[str, str]:
    """回傳 (user_satisfaction, reason)。"""
    raw = _prompt("[逗福Tofu] 這個結果符合你的預期嗎？(是/否/跳過) > ").strip()
    if not raw:
        return "no_feedback", ""
    lowered = raw.lower()
    if raw in {"是", "對", "yes", "y"} or lowered in {"yes", "y", "ok", "好"}:
        return "satisfied", ""
    if raw in {"否", "不是", "no", "n"} or lowered in {"no", "n"}:
        reason = _prompt("  不符合的原因是？（可空） > ").strip()
        return "unsatisfied", reason
    if raw in {"跳過", "skip"} or lowered in {"skip", "s"}:
        return "no_feedback", ""
    return "no_feedback", ""


def _latest_completed_endpoint(
    store: EndpointStore,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """取最近一筆 start+end 配對的 (start_data, end_data)（依 timestamp 最新）。

    用於修法 B 的跨題延續框定。store.completed_events() 回傳所有已湊齊
    start+end 的事件「列」對；這裡挑 timestamp 最新的一對，回傳其
    start_data／end_data 內層 dict（caller 直接讀 goal／result）。
    """
    pairs = store.completed_events()
    if not pairs:
        return None

    def _ts(pair: tuple[dict[str, Any], dict[str, Any]]) -> str:
        start_row, end_row = pair
        return str(end_row.get("timestamp") or start_row.get("timestamp") or "")

    start_row, end_row = max(pairs, key=_ts)
    return (start_row.get("start_data") or {}, end_row.get("end_data") or {})


# ----------------------------------------------------------------------
# v4.2 復述累積與補位去重輔助函式
# ----------------------------------------------------------------------
def _make_confirmed_facts_state() -> dict[str, Any]:
    """空白的已確認事實狀態。

    goal / motivation 為「最新值覆蓋」槽位；constraints / extras 為去重 append
    的清單。這樣每維度只會在 prior_context 出現一次，避免 LLM 跨輪 refine
    措辭時累積多筆「目標：…」/「動機：…」污染復述提示。
    """
    return {
        "goal": "",
        "motivation": "",
        "constraints": [],
        "extras": [],
    }


def _update_confirmed_facts(
    state: dict[str, Any],
    round_data: dict[str, Any],
    user_response: str,
) -> None:
    """以本輪 round_data 更新已確認事實狀態。

    goal / motivation 採「最新值覆蓋」：LLM 每輪會 refine 同維度，只保留
    最新版本。constraints 採「去重 append」。
    extras（使用者原始補充）只在本輪 goal / motivation / constraints 都
    沒實質更新時才補上——避免重複注入已經被結構化欄位涵蓋的資訊。
    """
    goal = (round_data.get("goal", "") or "").strip()
    user_input = (round_data.get("user_input", "") or "").strip()
    motivation = (round_data.get("motivation", "") or "").strip()
    new_constraints = [
        c for c in (round_data.get("constraints", []) or []) if c
    ]
    response_trimmed = (user_response or "").strip()

    goal_changed = (
        bool(goal) and goal != user_input and goal != state.get("goal", "")
    )
    motivation_changed = (
        bool(motivation) and motivation != state.get("motivation", "")
    )
    constraints_added = False

    if goal_changed:
        state["goal"] = goal
    if motivation_changed:
        state["motivation"] = motivation
    for c in new_constraints:
        if c not in state["constraints"]:
            state["constraints"].append(c)
            constraints_added = True

    if (
        len(response_trimmed) > 5
        and not (goal_changed or motivation_changed or constraints_added)
        and response_trimmed not in state["extras"]
    ):
        state["extras"].append(response_trimmed)


def _render_confirmed_facts(state: dict[str, Any]) -> list[str]:
    """把已確認事實狀態渲染為 bullet 文字清單，供 prior_context 注入。"""
    facts: list[str] = []
    if state.get("goal"):
        facts.append(f"目標：{state['goal']}")
    if state.get("motivation"):
        facts.append(f"動機：{state['motivation']}")
    for c in state.get("constraints", []):
        facts.append(f"條件：{c}")
    for e in state.get("extras", []):
        facts.append(f"使用者補充：{e}")
    return facts


def _deduplicate_gap_questions(
    current_questions: list[str],
    prior_lines: list[str],
) -> list[int]:
    """程式碼層的 gap_questions 去重，回傳「保留下來的索引」清單。

    原理：用 tokenizer 分詞後取交集。如果本輪問題的關鍵詞與前幾輪
    **某個**問題的關鍵詞重疊率 > 60%，判定為重複、移除。

    這是 LLM 語意去重的保險層——LLM 會換措辭問同一件事，程式碼層
    靠關鍵詞重疊抓住這種情況。

    閾值 60% 偏寬鬆：寧可多問一題不同的，也不要錯殺一題真正新的。

    每輪的 Q&A 行裡多題以「、」串接，這裡會把單一行拆成多題分別比對；
    若整輪當成一個 token 集，短中文問題會跨題湊出虛假重疊（false positive）。

    回傳索引而非過濾後的問題，是為了讓呼叫端用同一組索引同步過濾
    parallel 的 `gap_categories`，避免「問題被移除、類別還留著」造成
    baseline 與未解缺口統計被污染。
    """
    from src.utils.tokenizer import tokenize

    prior_question_tokens: list[set[str]] = []
    for line in prior_lines:
        if "提問：" in line and "；" in line:
            q_part = line.split("提問：", 1)[1].split("；", 1)[0]
            for single_q in q_part.split("、"):
                tokens = set(tokenize(single_q, min_token_len=2))
                if tokens:
                    prior_question_tokens.append(tokens)

    if not prior_question_tokens:
        return list(range(len(current_questions)))

    kept_indexes: list[int] = []
    for i, q in enumerate(current_questions):
        q_tokens = set(tokenize(q, min_token_len=2))
        if not q_tokens:
            kept_indexes.append(i)
            continue

        is_duplicate = False
        for prior_tokens in prior_question_tokens:
            if not prior_tokens:
                continue
            overlap = q_tokens & prior_tokens
            shorter = min(len(q_tokens), len(prior_tokens))
            if shorter > 0 and len(overlap) / shorter > 0.6:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_indexes.append(i)

    return kept_indexes


# ----------------------------------------------------------------------
# 核心互動流程
# ----------------------------------------------------------------------
def run_one_interaction(
    user_input: str,
    store: EndpointStore,
    client: LLMClient,
    ask_satisfaction: bool = False,
    profile_store: ProfileStore | None = None,
    auto_confirm: bool = False,
    guardrails: GuardrailTracker | None = None,
    *,
    mode: str = "default",
    zone_history_store: ZoneHistoryStore | None = None,
) -> bool:
    """完整走一輪：載入基線 → 產生復述 → 等確認 → 合併 → 寫 start → 執行 → 寫 end → 重算基線。

    回傳 True 表示這輪有順利完成（有寫入 end 端點）。

    auto_confirm: 為 True 時自動確認復述（不等待人類輸入）。
    適用於 API 模式、批次處理、自動化測試等不需要人類即時回應的場景。

    guardrails: v0.5 護欄。非 None 時會在呼叫 LLM 前做預估檢查，
    任一條件觸發則中斷此輪並印出護欄訊息（參見 src.middleware.guardrails）。

    mode: v0.6 PR-B 五模式開關。
    - ``default``：現有補位模式（restate → confirm → execute）
    - ``free``：思考引擎照跑，跳過 restate/confirm，execute_task 用 FREE prompt
    - ``risk``：思考引擎照跑，跳過 restate/confirm，execute_task 用 RISK prompt

    ``propose`` 與 ``check`` 由 dispatcher 另外導向專屬函式，本函式只處理
    前三者。思考引擎（冷卻、top_k 查表、策略摘要、首輪方向偵測、軌跡收斂檢查）
    對所有支援模式同樣地跑；差別只在 restate/confirm 是否執行、以及 execute_task
    最後用哪份 output prompt。
    """
    if mode not in ("default", "free", "risk"):
        raise ValueError(
            f"run_one_interaction 不支援 mode={mode!r}；"
            "合法值：default / free / risk（propose / check 由 dispatcher 另行處理）"
        )
    # 0. Meta 問題攔截：使用者問「你是什麼」這類關於逗福本身的問題，
    #    直接印自我介紹，不進入補位流程、不寫端點、不呼叫 API。
    if is_meta_query(user_input):
        _print(_ABOUT_TEXT)
        return False

    # v0.5 腳本護欄：在任何 LLM 呼叫之前先檢查預估成本/時間/session
    if guardrails is not None:
        try:
            guardrails.check_before_call()
        except GuardrailTripped as tripped:
            _print(f"[逗福Tofu 系統訊息] {tripped}")
            _print("[逗福Tofu] 已中斷這一輪互動，輸出至今為止的已完成結果。")
            return False

    # 1. 載入端點 + 計算基線（純程式碼）
    endpoints = store.all()

    # P1-1：CIP-X 軌跡收斂檢查——連續多筆 goal 都往危險方向收斂時，
    # 直接阻斷這一輪互動（不寫端點、不呼叫 API）。
    recent_goals: list[str] = []
    for row in endpoints:
        if row.get("type") == "start":
            sd = row.get("start_data") or {}
            g = sd.get("goal") or sd.get("user_input") or ""
            if g:
                recent_goals.append(g)
    # 把這次的 user_input 當成第 N 筆參與趨勢判斷
    recent_goals.append(user_input)
    if confirm_mod.check_trajectory_convergence(recent_goals):
        _print(
            "[逗福Tofu 系統訊息] 偵測到連續互動的意圖軌跡異常。\n"
            "逗福Tofu 無法繼續處理此方向的請求。\n"
            "如果你正在經歷困難，請聯繫專業協助。"
        )
        return False

    # P0-4：先粗略推斷本次輸入的 topic，決定要餵哪份基線。
    current_topic = confirm_mod.infer_topic(user_input, [])
    topic_baseline = baseline_mod.compute_baseline(
        endpoints, topic_filter=current_topic
    )
    general_baseline = baseline_mod.compute_baseline(endpoints)
    # 如果同 topic 基線夠（>=3 筆），用 topic 基線；否則用全部基線
    if topic_baseline.get("total_events", 0) >= 3:
        baseline = topic_baseline
    else:
        baseline = general_baseline

    # v0.5 第四層：強制查表 gate（不可跳過）。
    # 先套冷卻（90 天 / 50 輪未命中 → pending_confirmation，降權但不刪除），
    # 然後查 top-k 端點並編碼成密碼表。
    current_round = store.current_round()
    store.apply_cooldown(current_round=current_round)
    # 重新讀取一次，確保拿到 cooldown 過後的 status
    endpoints = store.all()
    top_rows, known_gap_cats = retrieve_top_k_endpoints(
        endpoints, user_input, top_k=DEFAULT_TOP_K,
        recent_floor_n=RECENT_FLOOR_N,
    )
    encoded_top_k = encode_retrieved_endpoints(top_rows) if top_rows else ""
    # 暫存命中 ids；mark_hit 延到互動成功（end 寫入後）才執行，
    # 避免 restate 失敗或使用者中止時產生「虛假命中」（#PR review）。
    pending_hit_ids = {
        row.get("endpoint_id") for row in top_rows if row.get("endpoint_id")
    }

    # v0.5 畫像線：把 baseline 全量統計翻成補位策略摘要（≈ 80-100 tokens）。
    user_profile_data = None
    if profile_store is not None:
        try:
            user_profile_data = profile_store.load()
        except Exception:  # pragma: no cover - defensive
            user_profile_data = None
    # 策略摘要用「全體基線」而非 topic 基線——畫像線管的是廣度
    strategy_brief = baseline_mod.compute_strategy_brief(
        general_baseline, user_profile=user_profile_data,
    )

    # v0.5 第五層：首輪方向偵測——通則尚未建立 + 只提到自己 → 要兩個反問
    first_round_direction_hint = confirm_mod.detect_first_round_direction(
        user_input=user_input,
        completed_rounds=store.completed_count(),
    )

    # 2. 呼叫 LLM（或 fallback）產生復述——僅 default 模式跑
    #    free / risk 模式跳過 restate：使用者主動選了「不要反問」，
    #    我們就尊重他的選擇，把 user_input 直接當 goal，由 execute_task
    #    的 OUTPUT_PROMPT_FREE / _RISK 負責用「明確假設」或「風險清單」格式回應。
    #
    # v4.1 收斂閘門：default 模式的補位改為多輪迴圈。兩個停損條件先到先停：
    #   - 連續 3 輪空輪（has_effective_info=False）→ convergence_reason="empty_streak"
    #   - 總輪數達 5 輪 → convergence_reason="max_rounds"
    # LLM 若在某輪主動把 gap_questions 設為空陣列（第一輪就判夠答、或跨輪後
    # 判斷已足），視為 info_sufficient 收斂。
    #
    # Copilot review #1：既有端點的 token 集合在進入迴圈前預先計算、快取，
    # 避免每一輪 has_effective_info 重複斷詞全量端點。
    # Copilot review #6：auto_confirm=True 表示 caller（批次/CI）不等待人類
    # 輸入；若放進多輪迴圈會把字面值 "confirmed" 重複視為有效資訊，一路跑到
    # max_rounds 浪費 5 次 LLM 呼叫。此情境直接走單輪流程。
    if mode == "default":
        convergence_state = confirm_mod.make_convergence_state()
        prior_rounds_lines: list[str] = []
        confirmed_facts_state: dict[str, Any] = _make_confirmed_facts_state()
        start_data: dict[str, Any] | None = None
        user_response: str = ""
        existing_tokens_cache = confirm_mod.extract_endpoint_tokens(endpoints)
        # B：跨題延續框定（在 while True 補位迴圈開始前算一次）。
        # 從嚴偵測命中時，注入一段純文字讓 LLM 知道本題在延續上一題；
        # 不增加 LLM 呼叫次數。
        continuation_frame = None
        if confirm_mod.detect_continuation(user_input, store.completed_count()):
            last = _latest_completed_endpoint(store)  # 最近一筆 start+end 配對
            if last is not None:
                sd, ed = last
                goal = sd.get("goal") or sd.get("user_input") or ""
                result_head = (ed.get("result") or "")[:60]
                continuation_frame = (
                    f"（本題延續上一題）上一題目標：{goal}；"
                    f"上一題結論摘要：{result_head}"
                )
        while True:
            round_num = int(convergence_state.get("total_rounds") or 0) + 1
            # 首輪方向偵測僅第一輪生效；第二輪（含）之後依靠 prior_rounds_context。
            round_direction_hint = (
                first_round_direction_hint if round_num == 1 else None
            )
            # v4.2：prior_context 改為「已確認事實 + Q&A 記錄」兩段結構。
            # 已確認事實供 LLM 寫入 restate_text（累積規則）；
            # Q&A 記錄供 LLM 判斷哪些維度已問過（不重複規則）。
            if prior_rounds_lines:
                facts_section = ""
                confirmed_facts = _render_confirmed_facts(confirmed_facts_state)
                if confirmed_facts:
                    facts_section = (
                        "【已確認事實（不得再問、必須寫入復述）】\n"
                        + "\n".join(f"- {f}" for f in confirmed_facts)
                        + "\n\n"
                    )
                qa_section = (
                    "【前幾輪 Q&A 記錄】\n"
                    + "\n".join(prior_rounds_lines)
                )
                prior_context = facts_section + qa_section
            else:
                prior_context = None
            # B（#33 修法B）：第一輪把跨題延續框定併入 prior_context
            # （不污染 prior_rounds_lines 的「補位輪紀錄」語意）。v4.2 兩段
            # 結構與延續框定互補：第一輪 prior_context 為 None 時改用框定，
            # 之後輪次則疊在事實/Q&A 之上。
            if continuation_frame and round_num == 1:
                prior_context = (
                    f"{continuation_frame}\n{prior_context}"
                    if prior_context
                    else continuation_frame
                )
            try:
                restate = confirm_mod.build_restate_payload(
                    user_input=user_input,
                    baseline=baseline,
                    allowed_categories=baseline_mod.ALLOWED_GAP_CATEGORIES,
                    client=client,
                    strategy_brief=strategy_brief,
                    encoded_top_k=encoded_top_k,
                    first_round_direction_hint=round_direction_hint,
                    prior_rounds_context=prior_context,
                )
                if guardrails is not None and not getattr(client, "fallback_mode", False):
                    guardrails.record_call()
            except LLMClientError as exc:
                _print(f"[錯誤] 產生復述失敗：{exc}")
                return False

            gap_questions = restate.get("gap_questions") or []

            # v4.2 程式碼層去重：比對本輪問題與前幾輪已問過的問題，靠關鍵詞
            # 重疊抓 LLM 換措辭問同一件事的情況。若去重後變空，下面 info_sufficient
            # 分支會接住——視為「已無可問新維度」收斂。
            # 同步以保留索引過濾 parallel 的 gap_categories，避免被移除的問題
            # 對應的類別仍被 finalize_start_data 持久化、污染基線統計。
            # gap_categories 只在與原始 gap_questions 等長時才視為 parallel；
            # LLM 偶爾會回比較短/長的 categories，那就保留原樣不動。
            if prior_rounds_lines and gap_questions:
                kept_indexes = _deduplicate_gap_questions(
                    gap_questions, prior_rounds_lines,
                )
                if len(kept_indexes) != len(gap_questions):
                    raw_categories = restate.get("gap_categories") or []
                    if len(raw_categories) == len(gap_questions):
                        restate["gap_categories"] = [
                            raw_categories[i] for i in kept_indexes
                        ]
                    gap_questions = [gap_questions[i] for i in kept_indexes]
                    restate["gap_questions"] = gap_questions

            # info_sufficient：LLM 主動回空 gap_questions 代表資訊已足或無可追問。
            if not gap_questions:
                convergence_state["converged"] = True
                convergence_state["convergence_reason"] = "info_sufficient"
                # Copilot review #4：把實際呼叫過 restate 的輪數記進 total_rounds，
                # 避免「第 2 輪才觸發 info_sufficient」時誤算成 1 輪。
                convergence_state["total_rounds"] = round_num
                if start_data is None:
                    # 第一輪即判夠答——直接用 LLM 的 inferred 值組 start_data，
                    # 不必再問使用者、不呼叫 merge_confirmation。
                    # Copilot review #5：這條分支不問使用者，把 user_response 固化為
                    # "auto_info_sufficient" 方便下游 black_box / profile 寫入。
                    user_response = "auto_info_sufficient"
                    start_data = {
                        "user_input": user_input,
                        "tofu_understanding": restate.get("restate_text", ""),
                        "gap_questions": [],
                        "user_confirmation": "auto_info_sufficient",
                        "user_modifications": "",
                        "goal": restate.get("inferred_goal") or user_input,
                        "motivation": restate.get("inferred_motivation") or "",
                        "constraints": list(restate.get("inferred_constraints") or []),
                        "gap_categories": [],
                        "_answered_categories": [],
                    }
                    if restate.get("reasoning_steps"):
                        start_data["reasoning_steps"] = restate["reasoning_steps"]
                    if restate.get("emotion_state"):
                        start_data["emotion_state"] = restate["emotion_state"]
                break

            # 印出本輪復述（含補位提問）
            restate_text = restate.get("restate_text") or "（未取得復述文字）"
            if round_num == 1:
                _print(f"[逗福Tofu 復述] {restate_text}")
            else:
                _print(f"[逗福Tofu 補位 第 {round_num} 輪] {restate_text}")

            # 3. 等使用者確認或補充
            if auto_confirm:
                user_response = "confirmed"
            else:
                user_response = _prompt("> ").strip()
                if not user_response:
                    _print("[逗福Tofu] 未收到回應，取消這次互動。")
                    return False
                if user_response.lower() in {"/exit", "/quit"}:
                    raise SystemExit(0)

            # 4. 合併補位回答
            try:
                round_data = confirm_mod.finalize_start_data(
                    user_input=user_input,
                    restate=restate,
                    user_response=user_response,
                    client=client,
                )
                if guardrails is not None and not getattr(client, "fallback_mode", False):
                    guardrails.record_call()
            except LLMClientError as exc:
                _print(f"[錯誤] 合併確認回答失敗：{exc}")
                return False

            if start_data is None:
                start_data = round_data
            else:
                # 累積合併：後輪優先覆蓋非空欄位；gap_questions / gap_categories
                # / _answered_categories / constraints 以 append 累積。
                if round_data.get("goal"):
                    start_data["goal"] = round_data["goal"]
                if round_data.get("motivation"):
                    start_data["motivation"] = round_data["motivation"]
                if round_data.get("tofu_understanding"):
                    start_data["tofu_understanding"] = round_data["tofu_understanding"]
                if round_data.get("user_confirmation"):
                    start_data["user_confirmation"] = round_data["user_confirmation"]
                if round_data.get("user_modifications"):
                    prior_mod = start_data.get("user_modifications") or ""
                    new_mod = round_data["user_modifications"]
                    start_data["user_modifications"] = (
                        f"{prior_mod}；{new_mod}" if prior_mod else new_mod
                    )
                for c in round_data.get("constraints", []) or []:
                    if c not in start_data.setdefault("constraints", []):
                        start_data["constraints"].append(c)
                for q in round_data.get("gap_questions", []) or []:
                    start_data.setdefault("gap_questions", []).append(q)
                for cat in round_data.get("gap_categories", []) or []:
                    if cat not in start_data.setdefault("gap_categories", []):
                        start_data["gap_categories"].append(cat)
                for cat in round_data.get("_answered_categories", []) or []:
                    if cat not in start_data.setdefault("_answered_categories", []):
                        start_data["_answered_categories"].append(cat)
                if round_data.get("emotion_state") and not start_data.get("emotion_state"):
                    start_data["emotion_state"] = round_data["emotion_state"]

            # 判定本輪是否有效資訊 + 是否繼續
            # Copilot #1：用預先計算的 existing_tokens_cache，避免每輪重複斷詞。
            effective = confirm_mod.has_effective_info(
                user_response, existing_tokens=existing_tokens_cache,
            )
            # v4.2：本輪的 round_data 經過 finalize_start_data 合併後，更新
            # 已確認事實狀態（goal / motivation latest-wins、constraints / 原始
            # 補充去重 append），供下一輪 prior_context 注入「必須寫入復述」區塊。
            _update_confirmed_facts(
                confirmed_facts_state, round_data, user_response,
            )
            questions_str = "、".join(str(q) for q in gap_questions)
            prior_rounds_lines.append(
                f"第 {round_num} 輪提問：{questions_str}；"
                f"使用者回：{user_response}"
            )
            # Copilot #6：auto_confirm 模式不進多輪——強制本輪後直接收斂。
            # 理由：caller 給的 user_response 是字面值 "confirmed"（不是真正的
            # 資訊回覆），繼續多輪只會浪費 LLM 呼叫。用 info_sufficient 收斂
            # 比 empty_streak 語意更準（caller 是主動聲明「這輪就夠」）。
            if auto_confirm:
                convergence_state["converged"] = True
                convergence_state["convergence_reason"] = "info_sufficient"
                convergence_state["total_rounds"] = round_num
                break
            keep_going = confirm_mod.should_continue_gap_filling(
                convergence_state, effective,
            )
            if not keep_going:
                break

        # 萬一從未進入迴圈體（理論上不會），保底組一份 start_data。
        if start_data is None:  # pragma: no cover - defensive
            start_data = {
                "user_input": user_input,
                "tofu_understanding": user_input,
                "gap_questions": [],
                "user_confirmation": "confirmed",
                "user_modifications": "",
                "goal": user_input,
                "motivation": "",
                "constraints": [],
                "gap_categories": [],
                "_answered_categories": [],
            }

        # 收斂閘門結果寫回 start_data（端點欄位於 append_start 前組好）。
        convergence_reason = convergence_state.get("convergence_reason")
        total_gap_rounds = int(convergence_state.get("total_rounds") or 0) or 1
        unresolved_gaps: list[str] = []
        if convergence_reason in ("empty_streak", "max_rounds"):
            # 收斂時仍未解的維度 = 問過的 gap_categories 扣掉已答的
            asked = {
                str(c).strip().lower() for c in start_data.get("gap_categories") or []
                if str(c).strip()
            }
            answered = {
                str(c).strip().lower() for c in start_data.get("_answered_categories") or []
                if str(c).strip()
            }
            unresolved_gaps = sorted(asked - answered)
        start_data["unresolved_gaps"] = unresolved_gaps
        start_data["convergence_reason"] = convergence_reason
        start_data["total_gap_rounds"] = total_gap_rounds

        if convergence_reason == "empty_streak":
            _print(
                f"[逗福Tofu 收斂] 連續 {confirm_mod.CONVERGENCE_EMPTY_STREAK_THRESHOLD} "
                f"輪空輪停損，以目前資訊進入執行。"
            )
        elif convergence_reason == "max_rounds":
            _print(
                f"[逗福Tofu 收斂] 已達補位總輪數上限 "
                f"{confirm_mod.CONVERGENCE_MAX_ROUNDS}，強制進入執行。"
            )
    else:
        # v0.6 PR-B：free / risk 模式——跳過 restate 與 user 確認
        restate = {
            "restate_text": "",
            "gap_questions": [],
            "gap_categories": [],
            "inferred_goal": user_input,
            "inferred_motivation": "",
            "inferred_constraints": [],
        }
        user_response = f"auto_{mode}"
        start_data = {
            "user_input": user_input,
            "tofu_understanding": user_input,  # 省略復述，保留原輸入方便追溯
            "gap_questions": [],
            "user_confirmation": f"auto_{mode}",
            "user_modifications": "",
            "goal": user_input,
            "motivation": "",
            "constraints": [],
            "gap_categories": [],
            "_answered_categories": [],
            # v4.1：free / risk 模式不跑補位，收斂閘門欄位填預設值即可。
            "unresolved_gaps": [],
            "convergence_reason": None,
            "total_gap_rounds": 0,
        }
        _print(f"[逗福Tofu /{mode}] 直接處理：{user_input}")

    # v4.0 P1：情緒三軸偵測（兩層）+ 冷卻模式判定
    # 優先用 restate 階段 LLM 搭車偵測的 emotion_state（精度較高、信心度較高）；
    # 若 LLM 沒回傳（free/risk 模式不走 restate / LLM 忘了帶），退回規則式偵測。
    # 冷卻模式觸發時本輪走 ATL-3 skip_gate 路徑（不強制交付），並在
    # execute_task 的 extra_context 前綴注入 COOLING_MODE_PROMPT_ADDENDUM。
    emotion_state = start_data.get("emotion_state") or detect_emotion_rule_based(user_input)
    start_data["emotion_state"] = emotion_state
    current_profile_for_cooling = user_profile_data
    if profile_store and current_profile_for_cooling is None:
        current_profile_for_cooling = profile_store.load()
    profile_baseline = (
        (current_profile_for_cooling or {}).get("emotion_baseline") or {}
    )
    cooling_mode = should_enter_cooling_mode(emotion_state, profile_baseline)

    # 5. 寫入 start 端點
    event_id = new_event_id()
    # _answered_categories 是內部輔助欄位，不寫入磁碟
    persisted_start = {k: v for k, v in start_data.items() if not k.startswith("_")}
    # P0-1：記錄 goal 的 Zone 分類，供 /stats 統計使用
    persisted_start["zone"] = confirm_mod.classify_zone(
        persisted_start.get("goal", "")
    )
    # P0-4：記錄 topic 分類——先以最終 goal + gap_categories 再推斷一次，
    # 避免受限於流程開頭只用 user_input 推斷的結果。
    persisted_start["topic"] = confirm_mod.infer_topic(
        persisted_start.get("goal", "") or user_input,
        persisted_start.get("gap_categories") or [],
    )
    # v0.6 PR-B：端點加 mode 欄位
    # 所有模式產出的端點都帶 mode，baseline 統計會排除 mode == 'check'（PR-D
    # 後才會出現的值），確保 /check 的事實查核內容不污染使用者畫像。
    persisted_start["mode"] = mode
    # v4.0 P1：冷卻模式旗標寫進 start_data，供事後分析
    persisted_start["cooling_mode"] = cooling_mode
    # 意圖分流（規格 06209671 §2.2）：只在 default 路徑生效。
    # free / risk / propose 等模式由使用者顯式指定，跳過分流。
    #   intent_mode == "execute" → 維持現有工單路徑（output_mode 不變）。
    #   intent_mode == "plan"    → output_mode = "consult"（規劃／諮詢 prompt）。
    # 判斷材料只有使用者本輪原始輸入的句式，不看 Tofu 推測的 goal/motivation。
    intent_mode = "execute"
    output_mode = mode
    if mode == "default":
        intent_mode = confirm_mod.detect_intent_mode(user_input)
        if intent_mode == "plan":
            output_mode = "consult"
    persisted_start["intent_mode"] = intent_mode
    persisted_start["output_mode"] = output_mode
    store.append_start(event_id=event_id, start_data=persisted_start)

    if cooling_mode:
        _print("[逗福Tofu] 偵測到情緒升溫訊號，本輪進入冷卻模式：只做事實釐清。")

    # 6. 載入使用者畫像（如有），傳給 execute_task
    user_profile = current_profile_for_cooling

    # v4.0 P0-1：execute_task 外圍包 ATL-3 前驗證閘門 + 重試
    # 依 spec §P0-1 目標為 propose_final_is_real，ATL-3 閘門只在 propose
    # 模式觸發（spec 驗收項目 `test_atl3_gate_not_applied_to_default_mode_unnecessarily`）。
    # default / free / risk 模式不跑閘門，保留原互動體驗。
    # 冷卻模式下一律 skip_gate（spec §P1：冷卻模式不強制交付條件）。
    cooling_extra = COOLING_MODE_PROMPT_ADDENDUM if cooling_mode else None
    skip_atl3_gate = cooling_mode or mode not in ("propose", "propose_final")
    # Gate 預設值只在「閘門已被 skip 且未被呼叫」的情境下使用；任一呼叫路徑
    # 回傳或拋例外，都必須覆寫這個字典，避免把 execute_task 失敗誤記為 ATL-3 通過
    # （Codex P1 review r3105635898）。
    atl3_gate_result: dict[str, Any] = {
        "passed": True, "retry_count": 0, "degraded": False,
        "failure_reasons": [], "failure_history": [], "skipped": True,
    }
    try:
        result, atl3_gate_result = _execute_task_with_atl3_gate(
            client=client,
            goal=start_data.get("goal", ""),
            motivation=start_data.get("motivation", ""),
            constraints=start_data.get("constraints", []),
            user_profile=user_profile,
            strategy_brief=strategy_brief,
            encoded_top_k=encoded_top_k,
            output_mode=output_mode,
            extra_context=cooling_extra,
            skip_gate=skip_atl3_gate,
            guardrails=guardrails,
        )
    except LLMClientError as exc:
        _print(f"[錯誤] 任務執行失敗：{exc}")
        result = "（任務執行失敗，未取得結果）"
        # execute_task 拋例外 → ATL-3 沒機會驗收 → 明確標記為失敗 + degraded，
        # 避免下游的 ATL-4 把這筆錯誤 result 當成正常樣本寫進 zone_history。
        atl3_gate_result = {
            "passed": False,
            "retry_count": 0,
            "degraded": True,
            "failure_reasons": [f"execute_task 拋例外：{exc}"],
            "failure_history": [],
            "execution_error": str(exc),
        }

    # v0.6 PR-G step 2：迴避句型偵測 + 重試一次（default / free 模式）
    # MOMO 2026-04-16 bug report：Haiku 會出現「你可以思考」「你可以搜尋」
    # 這類推諉句型。Prompt 禁令不完全可靠，這裡加 runtime 偵測 + 一次重試。
    # fallback 模式、呼叫失敗、guardrail 不允許重試時一律保留原回應。
    # 規格 06209671：以 output_mode 把關，consult（規劃）模式排除——迴避重試
    # 會把輸出推回「幫我做」的執行語氣，與規劃模式的「想清楚選項」哲學相衝突。
    evasion_retry_degraded = False
    hits = detect_evasion_phrases(result)
    if (
        output_mode in ("default", "free")
        and not getattr(client, "fallback_mode", False)
        and hits
    ):
        can_retry = True
        if guardrails is not None:
            try:
                guardrails.check_before_call()
            except GuardrailTripped:
                can_retry = False
        if can_retry:
            retry_hint = build_retry_hint(hits)
            # Copilot review r3105635894：冷卻模式下的重寫呼叫也必須保留
            # COOLING_MODE_PROMPT_ADDENDUM，否則重寫版本會違反「不給建議」的
            # 約束。把冷卻 addendum 前綴到 retry_hint，兩層訊號一起進去。
            retry_extra = (
                (cooling_extra + "\n\n" + retry_hint) if cooling_extra else retry_hint
            )
            try:
                retry_result = _call_execute_task(
                    client=client,
                    goal=start_data.get("goal", ""),
                    motivation=start_data.get("motivation", ""),
                    constraints=start_data.get("constraints", []),
                    user_profile=user_profile,
                    strategy_brief=strategy_brief,
                    encoded_top_k=encoded_top_k,
                    output_mode=output_mode,
                    extra_context=retry_extra,
                )
                if guardrails is not None:
                    guardrails.record_call()
                if is_evasive_response(retry_result, mode):
                    result = mark_degraded(retry_result)
                    evasion_retry_degraded = True
                    _print(
                        "[逗福Tofu] 連續兩次偵測到迴避句型，本回應已標記為降級。"
                    )
                else:
                    result = retry_result
                    _print(
                        "[逗福Tofu] 上次輸出含迴避句型，已自動重寫。"
                    )
            except LLMClientError as retry_exc:
                _print(
                    f"[逗福Tofu] 重寫失敗：{retry_exc}，保留原回應。"
                )

    _print(f"[逗福Tofu 執行] {result}")

    # 7. 滿意度收集（依間隔）
    satisfaction = "no_feedback"
    reason = ""
    if ask_satisfaction:
        satisfaction, reason = _collect_satisfaction()

    # 8. 程式碼層偏差偵測 + 寫入 end 端點
    deviation = confirm_mod.detect_deviation(start_data.get("goal", ""), result)
    end_data = make_end_data(
        result=result,
        user_satisfaction=satisfaction,
        deviation_from_goal=deviation,
    )
    # v0.6 PR-G step 2：記錄迴避句型重試是否降級（分析用）
    end_data["evasion_retry_degraded"] = evasion_retry_degraded
    # P0-1：記錄 result 的 Zone 分類
    end_data["zone"] = confirm_mod.classify_zone(result)
    if reason:
        end_data["unsatisfied_reason"] = reason

    # v2.0 P1-1：ATL-3 下一步具體性檢測
    # 規格 06209671 §2.5：具體性三要素（產出物/時間窗/驗收條件）檢查
    # **只在執行模式生效**。規劃（consult）模式跳過——規劃本就不該被強加
    # deadline/驗收條件，否則會把規劃輸出打回成工單。但空泛型黑名單（純廢話
    # 偵測）在 consult 模式仍保留：規劃可以不含 deadline，但不該是空話。
    atl_action = confirm_mod.check_action_specificity(result)
    end_data["atl_action_check"] = atl_action
    if output_mode == "consult":
        if atl_action.get("vague_phrases"):
            _print(
                "[逗福Tofu] 注意：這次的規劃出現空泛句型"
                "（例如「進一步研究」「再觀察」），建議講得更具體一點。"
            )
    elif not atl_action.get("is_specific", True):
        _print(
            "[逗福Tofu] 注意：這次的建議比較籠統，"
            "缺少具體的時間窗/產出物/驗收條件。"
        )

    # v2.0 P1-2：ATL-1 可反駁條件檢測
    atl_fals = confirm_mod.check_falsification(result)
    end_data["atl_falsification_check"] = atl_fals
    if atl_fals.get("is_generic", False):
        _print(
            "[逗福Tofu] 注意：反駁條件過於籠統"
            "（例如「若有新證據」），缺乏具體數字或時間。"
        )

    # v2.0 P1-3：ATL-2 來源可回溯性檢測
    atl_src = confirm_mod.check_source_traceability(result, end_data["zone"])
    end_data["atl_source_check"] = atl_src
    if atl_src.get("should_downgrade", False):
        end_data["zone_downgraded"] = True
        end_data["zone_downgrade_reason"] = (
            "Zone A 內容缺乏具體來源，降級為 Zone B"
        )
        end_data["zone"] = "B"
        _print(
            "[逗福Tofu] 資訊分層提示：這次回覆中有聲稱為事實但缺乏來源的"
            "內容，已標記為推測（Zone B）。"
        )

    # v4.0 P0-1：ATL-3 前驗證閘門結果寫進 end_data
    end_data["atl3_gate_result"] = atl3_gate_result
    if atl3_gate_result.get("degraded"):
        # 第 3 次仍不通過 → 強制 Zone 降級為 B（spec §P0-1）
        end_data["zone_downgraded"] = True
        prior_reason = end_data.get("zone_downgrade_reason", "")
        degraded_reason = "ATL-3 閘門三次重試仍未通過，強制降級為 Zone B"
        end_data["zone_downgrade_reason"] = (
            f"{prior_reason}；{degraded_reason}" if prior_reason else degraded_reason
        )
        end_data["zone"] = "B"
        _print(
            "[逗福Tofu] ATL-3 前驗證閘門：連續 3 次輸出不符具體性檢查，本回應已降級。"
        )

    # v4.0 P0-2：ATL-4 跨輪一致性檢查
    # 冷卻模式或 ATL-3 降級的端點不寫進 zone_history，避免污染歷史
    if zone_history_store is not None and not persisted_start.get("cooling_mode") \
            and not atl3_gate_result.get("degraded"):
        try:
            # Copilot review r3105635893：meta_motivation_direction 統一由
            # `detect_first_round_direction` 產生，避免「只有 /free 模式硬編 self」
            # 導致 signature 在不同模式下語意不一致。目前只能判定 `self_only`
            # （使用者只提到自己），其他情境仍為空字串，未來擴充 other/both 時
            # 再補。
            direction_signal = (
                "self" if first_round_direction_hint == "self_only" else ""
            )
            features = extract_question_features(
                user_input=user_input,
                meta_motivation_direction=direction_signal,
                topic=persisted_start.get("topic") or "",
                mode=mode,
            )
            sig = compute_question_signature(features)
            consistency = atl4_consistency_check(
                sig, end_data["zone"], zone_history_store,
            )
            end_data["atl4_consistency_check"] = consistency
            if not consistency["is_consistent"]:
                end_data["zone_drift_warning"] = consistency["warning"]
                _print(
                    f"[逗福Tofu] 跨輪一致性警告：{consistency['warning']}"
                )
            # 寫入本輪紀錄（即使警告也寫，讓歷史自然累積）
            zone_history_store.append({
                "signature": sig,
                "question_features": features,
                "zone_assignment": end_data["zone"],
                "endpoint_id": event_id,
                "timestamp": _now_iso(),
                "confidence": 0.85,
            })
        except Exception as exc:  # noqa: BLE001 — ATL-4 失敗不可擋主流程
            _print(f"[逗福Tofu] ATL-4 檢查失敗（靜默跳過）：{exc}")

    # P0-2 / P0-3：偏差偵測觸發 → 黑盒子回溯 + 出口檢查
    if deviation:
        analyzer = getattr(client, "analyze_deviation", None)
        if callable(analyzer):
            try:
                analysis = analyzer(
                    goal=start_data.get("goal", ""),
                    result=result,
                    deviation=deviation,
                )
            except LLMClientError as exc:
                analysis = f"出口檢查失敗：{exc}"
        else:
            # 向後相容：client 未實作 analyze_deviation 時，
            # 用 fallback 版本做純程式碼出口檢查。
            from src.api.claude_client import _fallback_analyze_deviation
            analysis = _fallback_analyze_deviation(
                goal=start_data.get("goal", ""),
                result=result,
                deviation=deviation,
            )
        black_box_data = {
            "original_input": user_input,
            "restate_text": restate.get("restate_text", ""),
            "user_confirmation": user_response,
            "start_data": persisted_start,
            "result": result,
            "deviation": deviation,
            "analysis": analysis,
            "timestamp": _now_iso(),
        }
        store.append_black_box(event_id=event_id, black_box_data=black_box_data)
        end_data["black_box_recorded"] = True
        _print(
            "[逗福Tofu] 結果與目標有落差，已記錄回溯資料。輸入 /blackbox 可查看。"
        )
        _print(f"[逗福Tofu 出口檢查] {analysis}")

    end_row = store.append_end(event_id=event_id, end_data=end_data)
    _print("[端點已記錄]")

    # v0.6 PR-E：端點寫入後觸發詞性標記（失敗靜默）
    _try_tag_endpoint(
        endpoint_id=end_row["endpoint_id"],
        user_input=user_input,
        result=result,
    )

    # v0.5：互動成功寫入 end 後才 mark_hit，避免虛假命中
    if pending_hit_ids:
        store.mark_hit(pending_hit_ids, current_round=current_round)

    # v0.5 護欄：一次完整互動結束，累積一次 session
    if guardrails is not None:
        guardrails.record_session()

    # P1-1：CIP-X 軌跡收斂檢查發生在 start 寫入前，見流程開頭。

    # v3.0：更新使用者畫像
    if profile_store:
        profile_store.update_from_interaction(
            start_data=start_data,
            end_data=end_data,
            user_response=user_response,
        )

    # 9. 自動檢查提問單門檻（僅提示，不自動彈出）
    completed = store.completed_count()
    if form_mod.should_generate(completed) and completed in {5, 10, 15, 20, 30, 50}:
        _print(
            f"[逗福Tofu] 已累積 {completed} 筆互動紀錄，輸入 /form 可檢視目前的個人化提問單。"
        )

    return True


# ----------------------------------------------------------------------
# v0.6 PR-C：/propose 模式 5 輪 loop + 第 6 輪交卷
# ----------------------------------------------------------------------
# 依 Spec v2 §1.5。propose 模式不是一次 call 完成：
#   輪次 1-5：execute_task(output_mode='propose', extra_context=前輪紀錄)
#     → 解析 `[PROPOSE_META]` JSON；submit_now=true 跳到交卷輪
#   輪次 6（強制交卷）：execute_task(output_mode='propose_final')
#     → 四段式（最終提案 / 已知 / 假設 / 風險）
#
# 每一輪的 LLM 輸出都透過 black_box 留痕（同一個 event_id）；
# 整個 propose session 寫一對 start/end（start 在第 1 輪開頭、end 在交卷輪
# 結束），goal 為原始 user_input，end_data.result 為交卷內容。
# 這樣 baseline 統計不會被 /propose 的多輪特性打爆（每個 session 只算一筆）。

MAX_PROPOSE_ROUNDS = 5


def run_propose_mode(
    user_input: str,
    store: EndpointStore,
    client: LLMClient,
    ask_satisfaction: bool = False,
    profile_store: ProfileStore | None = None,
    guardrails: GuardrailTracker | None = None,
) -> bool:
    """/propose 模式：5 輪自問自答 + 第 6 輪強制交卷。

    回傳 True 表示這個 session 有走到交卷輪（不論中途是否因 guardrail 中斷）。
    回傳 False 表示連 start 都沒寫入（meta 攔截、guardrail 預判、軌跡收斂）。
    """
    # 0. Meta 攔截：關於逗福自己的問題不進入 propose 流程
    if is_meta_query(user_input):
        _print(_ABOUT_TEXT)
        return False

    # 1. 護欄預判（每輪都會再檢查一次，這裡先擋顯然超標的情況）
    if guardrails is not None:
        try:
            guardrails.check_before_call(extra_calls=MAX_PROPOSE_ROUNDS + 1)
        except GuardrailTripped as tripped:
            _print(f"[逗福Tofu 系統訊息] {tripped}")
            _print(
                "[逗福Tofu] /propose 模式會呼叫 LLM 最多 6 次，目前餘額不足。"
                "已中斷。"
            )
            return False

    endpoints = store.all()

    # 2. CIP-X 軌跡收斂檢查
    recent_goals: list[str] = []
    for row in endpoints:
        if row.get("type") == "start":
            sd = row.get("start_data") or {}
            g = sd.get("goal") or sd.get("user_input") or ""
            if g:
                recent_goals.append(g)
    recent_goals.append(user_input)
    if confirm_mod.check_trajectory_convergence(recent_goals):
        _print(
            "[逗福Tofu 系統訊息] 偵測到連續互動的意圖軌跡異常。\n"
            "逗福Tofu 無法繼續處理此方向的請求。\n"
            "如果你正在經歷困難，請聯繫專業協助。"
        )
        return False

    # 3. 思考引擎（與 default / free / risk 共用）
    current_topic = confirm_mod.infer_topic(user_input, [])
    topic_baseline = baseline_mod.compute_baseline(
        endpoints, topic_filter=current_topic
    )
    general_baseline = baseline_mod.compute_baseline(endpoints)
    baseline = (
        topic_baseline
        if topic_baseline.get("total_events", 0) >= 3
        else general_baseline
    )

    current_round = store.current_round()
    store.apply_cooldown(current_round=current_round)
    endpoints = store.all()
    top_rows, _known_gap_cats = retrieve_top_k_endpoints(
        endpoints, user_input, top_k=DEFAULT_TOP_K,
        recent_floor_n=RECENT_FLOOR_N,
    )
    encoded_top_k = encode_retrieved_endpoints(top_rows) if top_rows else ""
    pending_hit_ids = {
        row.get("endpoint_id") for row in top_rows if row.get("endpoint_id")
    }

    user_profile_data = None
    if profile_store is not None:
        try:
            user_profile_data = profile_store.load()
        except Exception:  # pragma: no cover - defensive
            user_profile_data = None
    strategy_brief = baseline_mod.compute_strategy_brief(
        general_baseline, user_profile=user_profile_data,
    )

    # 4. 寫 start 端點（整個 propose session 只寫一次）
    event_id = new_event_id()
    start_data: dict[str, Any] = {
        "user_input": user_input,
        "tofu_understanding": user_input,
        "gap_questions": [],
        "user_confirmation": "auto_propose",
        "user_modifications": "",
        "goal": user_input,
        "motivation": "",
        "constraints": [],
        "gap_categories": [],
        "mode": "propose",
        "zone": confirm_mod.classify_zone(user_input),
        "topic": confirm_mod.infer_topic(user_input, []),
        # v4.1：/propose 的自問自答迴圈不走補位收斂閘門，三欄位填預設值保一致。
        "unresolved_gaps": [],
        "convergence_reason": None,
        "total_gap_rounds": 0,
    }
    store.append_start(event_id=event_id, start_data=start_data)

    _print(
        f"[逗福Tofu /propose] 開始 5 輪自問自答（最多 6 次 LLM 呼叫）。"
        f"若中途資訊已夠會自動提前交卷。"
    )

    # 5. Loop 1-5 輪
    rounds_history: list[dict[str, Any]] = []
    submitted_early = False
    interrupted_by_guardrail = False

    for round_num in range(1, MAX_PROPOSE_ROUNDS + 1):
        # 每輪呼叫前檢查護欄
        if guardrails is not None:
            try:
                guardrails.check_before_call()
            except GuardrailTripped as tripped:
                _print(f"[逗福Tofu 系統訊息] {tripped}")
                _print(
                    f"[逗福Tofu /propose] 第 {round_num} 輪前觸發護欄，"
                    f"保留前 {len(rounds_history)} 輪結果並強制進交卷輪。"
                )
                interrupted_by_guardrail = True
                break

        extra_context = format_propose_history(rounds_history) or None
        try:
            text = _call_execute_task(
                client=client,
                goal=user_input,
                motivation="",
                constraints=[],
                user_profile=user_profile_data,
                strategy_brief=strategy_brief,
                encoded_top_k=encoded_top_k,
                output_mode="propose",
                extra_context=extra_context,
            )
            if guardrails is not None and not getattr(
                client, "fallback_mode", False
            ):
                guardrails.record_call()
        except LLMClientError as exc:
            _print(
                f"[錯誤] /propose 第 {round_num} 輪呼叫失敗：{exc}。"
                f"保留前 {len(rounds_history)} 輪結果並進交卷輪。"
            )
            break

        _print(f"[逗福Tofu /propose round {round_num}/{MAX_PROPOSE_ROUNDS}]")
        _print(text)

        meta = parse_propose_meta(text) or {}
        rounds_history.append(
            {"round": round_num, "text": text, "meta": meta}
        )

        # 每輪留痕到 black_box（共用 event_id）
        store.append_black_box(
            event_id=event_id,
            black_box_data={
                "mode": "propose",
                "round": round_num,
                "text": text,
                "meta": meta,
                "timestamp": _now_iso(),
            },
        )

        if meta.get("submit_now") is True:
            _print(
                f"[逗福Tofu /propose] 第 {round_num} 輪資訊已夠，"
                f"提前進交卷輪。"
            )
            submitted_early = True
            break

    # 6. 第 6 輪：強制交卷（即使前面 rounds_history 為空也要交）
    final_text = ""
    if guardrails is not None and not interrupted_by_guardrail:
        try:
            guardrails.check_before_call()
        except GuardrailTripped as tripped:
            _print(f"[逗福Tofu 系統訊息] {tripped}")
            interrupted_by_guardrail = True

    final_evasion_retry_degraded = False
    # v4.0 P0-1：propose_final 階段啟用 ATL-3 前驗證閘門（spec 主要目標）
    # 與既有 evasion retry 一致：fallback（離線 mock）模式下 skip，避免假資料
    # 把假回應打成 degraded；真實 LLM 跑時才嚴格檢查。
    # 預設值只在「完全沒跑到閘門」（fallback / 中斷）的情境用；任一呼叫路徑
    # 回傳或拋例外，都必須覆寫此字典，避免把失敗/中斷誤記為 ATL-3 通過
    # （Codex P2 review r3105635899）。
    propose_atl3_gate_result: dict[str, Any] = {
        "passed": True, "retry_count": 0, "degraded": False,
        "failure_reasons": [], "failure_history": [], "skipped": True,
    }
    propose_skip_gate = getattr(client, "fallback_mode", False)
    if interrupted_by_guardrail:
        final_text = (
            "（/propose 交卷階段因護欄中斷，保留前面輪次紀錄。）\n\n"
            + format_propose_history(rounds_history)
        )
        # 護欄中斷 → 閘門沒機會跑，明確標記為未驗證 + degraded，避免後續
        # 下游把中斷時的提案內容當成 ATL-3 通過。
        propose_atl3_gate_result = {
            "passed": False,
            "retry_count": 0,
            "degraded": True,
            "failure_reasons": ["交卷階段被護欄中斷，ATL-3 未執行"],
            "failure_history": [],
            "interrupted_by_guardrail": True,
        }
    else:
        try:
            final_text, propose_atl3_gate_result = _execute_task_with_atl3_gate(
                client=client,
                goal=user_input,
                motivation="",
                constraints=[],
                user_profile=user_profile_data,
                strategy_brief=strategy_brief,
                encoded_top_k=encoded_top_k,
                output_mode="propose_final",
                extra_context=format_propose_history(rounds_history) or None,
                guardrails=guardrails,
                skip_gate=propose_skip_gate,
            )
        except LLMClientError as exc:
            final_text = (
                f"（/propose 交卷失敗：{exc}）\n\n"
                + format_propose_history(rounds_history)
            )
            # 同 run_one_interaction：交卷拋例外 → 明確標記 ATL-3 失敗 + degraded
            propose_atl3_gate_result = {
                "passed": False,
                "retry_count": 0,
                "degraded": True,
                "failure_reasons": [f"交卷 execute_task 拋例外：{exc}"],
                "failure_history": [],
                "execution_error": str(exc),
            }

        # v0.6 PR-G step 2：PROPOSE_FINAL 迴避句型偵測 + 重試一次
        # MOMO 2026-04-16 bug report Bug 2：最終提案交出的是「我建議你先從 X
        # 選出」這種偽裝成提案的反問。Prompt 禁令不完全可靠 → runtime 偵測。
        if (
            not getattr(client, "fallback_mode", False)
            and is_evasive_response(final_text, "propose_final")
        ):
            hits = detect_evasion_phrases(final_text)
            can_retry = True
            if guardrails is not None:
                try:
                    guardrails.check_before_call()
                except GuardrailTripped:
                    can_retry = False
            if can_retry:
                retry_hint = build_retry_hint(hits)
                combined_context = retry_hint + (
                    format_propose_history(rounds_history) or ""
                )
                try:
                    retry_final = _call_execute_task(
                        client=client,
                        goal=user_input,
                        motivation="",
                        constraints=[],
                        user_profile=user_profile_data,
                        strategy_brief=strategy_brief,
                        encoded_top_k=encoded_top_k,
                        output_mode="propose_final",
                        extra_context=combined_context,
                    )
                    if guardrails is not None:
                        guardrails.record_call()
                    if is_evasive_response(retry_final, "propose_final"):
                        final_text = mark_degraded(retry_final)
                        final_evasion_retry_degraded = True
                        _print(
                            "[逗福Tofu /propose] 交卷連續兩次偵測到迴避句型，"
                            "本提案已標記為降級。"
                        )
                    else:
                        final_text = retry_final
                        _print(
                            "[逗福Tofu /propose] 交卷首次含迴避句型，已自動重寫。"
                        )
                except LLMClientError as retry_exc:
                    _print(
                        f"[逗福Tofu /propose] 交卷重寫失敗：{retry_exc}，保留原提案。"
                    )

    _print("[逗福Tofu /propose 交卷] 最終提案：")
    _print(final_text)

    # 7. 寫 end 端點
    end_data = make_end_data(
        result=final_text,
        user_satisfaction="no_feedback",
        deviation_from_goal="",
    )
    end_data["zone"] = confirm_mod.classify_zone(final_text)
    end_data["propose_rounds"] = len(rounds_history)
    end_data["propose_submitted_early"] = submitted_early
    end_data["propose_interrupted_by_guardrail"] = interrupted_by_guardrail
    # v0.6 PR-G step 2：PROPOSE_FINAL 是否降級
    end_data["evasion_retry_degraded"] = final_evasion_retry_degraded
    # v4.0 P0-1：記錄 ATL-3 前驗證閘門結果
    end_data["atl3_gate_result"] = propose_atl3_gate_result
    if propose_atl3_gate_result.get("degraded"):
        end_data["zone_downgraded"] = True
        end_data["zone_downgrade_reason"] = (
            "ATL-3 閘門三次重試仍未通過，強制降級為 Zone B"
        )
        end_data["zone"] = "B"
        _print(
            "[逗福Tofu /propose] ATL-3 前驗證閘門：連續 3 次交卷不符具體性，本提案已降級。"
        )
    # MOMO review：補記「到底是第幾輪真的交卷」，用於後續分析
    # 「/propose 通常第幾輪收斂」。
    # - 正常跑滿 5 輪：final_round_number = 6（第 6 輪強制交卷）
    # - 提前交卷：final_round_number = rounds_history 最後一筆 +1
    #   （例：第 3 輪 submit_now=true → 第 4 輪是交卷輪 → 4）
    # - 護欄中斷：final_round_number = len(rounds_history) +1
    #   （交卷輪即使被護欄中斷也視為「發生過」）
    end_data["propose_final_round_number"] = len(rounds_history) + 1
    propose_end_row = store.append_end(event_id=event_id, end_data=end_data)
    _print("[端點已記錄]")

    # v0.6 PR-E：端點寫入後觸發詞性標記（失敗靜默）
    _try_tag_endpoint(
        endpoint_id=propose_end_row["endpoint_id"],
        user_input=start_data.get("user_input", ""),
        result=end_data.get("result", ""),
    )

    # 8. mark_hit + 護欄 + 畫像更新（同 default 模式）
    if pending_hit_ids:
        store.mark_hit(pending_hit_ids, current_round=current_round)
    if guardrails is not None:
        guardrails.record_session()
    if profile_store:
        profile_store.update_from_interaction(
            start_data=start_data,
            end_data=end_data,
            user_response="auto_propose",
        )

    return True


def _execute_task_with_atl3_gate(
    client: LLMClient,
    *,
    goal: str,
    motivation: str,
    constraints: list[str],
    user_profile: dict[str, Any] | None,
    strategy_brief: str | None,
    encoded_top_k: str | None,
    output_mode: str,
    extra_context: str | None = None,
    max_retries: int = 2,
    skip_gate: bool = False,
    guardrails: GuardrailTracker | None = None,
) -> tuple[str, dict[str, Any]]:
    """v4.0 P0-1：execute_task 包 ATL-3 前驗證閘門 + 重試。

    流程：
    1. 呼叫 execute_task
    2. 跑 confirm_mod.atl3_gate
    3. 不通過 → 把 failure_reasons 組成 ATL3_RETRY_PROMPT，嵌進 extra_context
       前綴，重新呼叫 execute_task（最多 max_retries 次）
    4. max_retries 後仍不通過 → degraded=True（Zone 降級交由 caller 處理）

    回傳 (result, atl3_gate_result)，其中 atl3_gate_result 結構：
    {
        "passed": bool,
        "retry_count": int,
        "degraded": bool,
        "failure_reasons": list[str],
        "failure_history": list[list[str]],
    }

    skip_gate=True 時只呼叫一次 execute_task、不跑閘門，用於冷卻模式
    （見 v4.0 P1：冷卻模式下不強制交付條件）。
    """
    retry_count = 0
    failure_history: list[list[str]] = []
    base_extra = extra_context or ""
    result = ""

    while retry_count <= max_retries:
        # 第 0 次用原 extra_context，之後把失敗原因拼在前面
        if retry_count == 0:
            current_extra: str | None = base_extra or None
        else:
            hint = build_atl3_retry_hint(failure_history[-1])
            current_extra = hint + ("\n\n" + base_extra if base_extra else "")

        # guardrails：檢查是否還能再打 API
        if retry_count > 0 and guardrails is not None:
            try:
                guardrails.check_before_call()
            except GuardrailTripped:
                # 配額耗盡——放棄重試，回傳目前的 result 並標記 degraded
                break

        result = _call_execute_task(
            client=client,
            goal=goal,
            motivation=motivation,
            constraints=constraints,
            user_profile=user_profile,
            strategy_brief=strategy_brief,
            encoded_top_k=encoded_top_k,
            output_mode=output_mode,
            extra_context=current_extra,
        )
        if guardrails is not None and not getattr(client, "fallback_mode", False):
            guardrails.record_call()

        if skip_gate:
            return result, {
                "passed": True,
                "retry_count": 0,
                "degraded": False,
                "failure_reasons": [],
                "failure_history": [],
                "skipped": True,
            }

        gate = confirm_mod.atl3_gate(result, output_mode)
        if gate["passed"]:
            return result, {
                "passed": True,
                "retry_count": retry_count,
                "degraded": False,
                "failure_reasons": [],
                "failure_history": failure_history,
            }

        failure_history.append(list(gate["failure_reasons"]))
        retry_count += 1

    # 第 max_retries+1 次仍沒通過 → degraded
    return result, {
        "passed": False,
        "retry_count": retry_count,
        "degraded": True,
        "failure_reasons": failure_history[-1] if failure_history else [],
        "failure_history": failure_history,
    }


def _call_execute_task(
    client: LLMClient,
    *,
    goal: str,
    motivation: str,
    constraints: list[str],
    user_profile: dict[str, Any] | None,
    strategy_brief: str | None,
    encoded_top_k: str | None,
    output_mode: str,
    extra_context: str | None,
) -> str:
    """呼叫 client.execute_task，並做舊簽名的 TypeError 三層退化。

    層級：
    1. 完整 v0.6 PR-C 簽名（含 extra_context）
    2. 退回 v0.6 PR-B（沒 extra_context，保留 output_mode）
    3. 退回 v0.5（沒 output_mode，保留 strategy_brief / encoded_top_k）
    4. 退回 v0.3（只剩 goal / motivation / constraints / user_profile）

    舊 mock client 與骨架後端都能通過而不引發 TypeError。
    """
    try:
        return client.execute_task(
            goal=goal,
            motivation=motivation,
            constraints=constraints,
            user_profile=user_profile,
            strategy_brief=strategy_brief,
            encoded_top_k=encoded_top_k,
            output_mode=output_mode,
            extra_context=extra_context,
        )
    except TypeError as exc:
        if (
            "unexpected keyword argument" not in str(exc)
            and "positional argument" not in str(exc)
        ):
            raise
    try:
        return client.execute_task(
            goal=goal,
            motivation=motivation,
            constraints=constraints,
            user_profile=user_profile,
            strategy_brief=strategy_brief,
            encoded_top_k=encoded_top_k,
            output_mode=output_mode,
        )
    except TypeError as exc:
        if (
            "unexpected keyword argument" not in str(exc)
            and "positional argument" not in str(exc)
        ):
            raise
    try:
        return client.execute_task(
            goal=goal,
            motivation=motivation,
            constraints=constraints,
            user_profile=user_profile,
            strategy_brief=strategy_brief,
            encoded_top_k=encoded_top_k,
        )
    except TypeError as exc:
        if (
            "unexpected keyword argument" not in str(exc)
            and "positional argument" not in str(exc)
        ):
            raise
    return client.execute_task(
        goal=goal,
        motivation=motivation,
        constraints=constraints,
        user_profile=user_profile,
    )


# ----------------------------------------------------------------------
# v0.6 PR-D：/check 模式（ISF 資訊完整性檢驗器）三階段
# ----------------------------------------------------------------------
# 依 Spec v2 §1.6。/check 完全繞過主思考引擎——不查 top-k、不算策略、
# 不碰畫像。每個 /check session 有三個可能的階段：
#
# 1. summary（第一階段，使用者打 /check <內容> 時觸發）
# 2. full（第二階段完整報告，使用者選 1 時觸發）
# 3. warning（警示文字稿，使用者選 2 時觸發）
#
# Session state 寫在 data/sessions/check_<uuid>.json：
# - stage1_output 的存在代表「可以接續 2/3 階段」
# - 每次啟動 /check 前先掃 24 小時以上的舊 session
# - 使用者輸入 1/2/3 時找最近 session 接續；失敗就 fallback 提示重打
#
# ISF 輸出**不進 baseline**（mode='check'，PR-B 已落地排除邏輯）——
# 事實查核的內容不該污染使用者畫像。
#
# 端點策略：每次 /check session 寫一對 start/end（mode='check'），
# goal = 使用者貼的前 60 字摘要、end_data.result = 最終階段輸出。
# 中間的第一階段輸出存 session state（不進 endpoints.json）；
# 使用者選 3「結束」的情形也視為完整 session 並寫入 end。


def run_check_mode(
    user_input: str,
    store: EndpointStore,
    client: LLMClient,
    guardrails: GuardrailTracker | None = None,
    session_dir: str = SESSIONS_DIR,
) -> bool:
    """/check 模式：建立 session、跑第一階段、寫入 session state、印選單。

    回傳 True 代表第一階段順利完成（已寫 session + 已印選單）；
    回傳 False 代表連 session 都沒建（meta 攔截 / 護欄預判觸發）。
    """
    # 0. Meta 攔截
    if is_meta_query(user_input):
        _print(_ABOUT_TEXT)
        return False

    # 1. 啟動前先 sweep 舊 session
    check_mod.sweep_old_sessions(session_dir)

    # 2. 護欄預判（僅計 1 次呼叫——第一階段）
    if guardrails is not None:
        try:
            guardrails.check_before_call()
        except GuardrailTripped as tripped:
            _print(f"[逗福Tofu 系統訊息] {tripped}")
            _print(
                "[逗福Tofu] /check 模式需要 1 次 LLM 呼叫（以及後續可能"
                "再 1 次的第二階段），目前餘額不足，已中斷。"
            )
            return False

    # 3. 建立 session state（先寫檔，再呼叫 LLM，失敗時 state 仍可追溯）
    session = check_mod.create_session(session_dir, user_content=user_input)
    session_id = session["session_id"]

    # 4. 呼叫 LLM 跑第一階段
    try:
        stage1_output = client.run_check_stage(
            stage=STAGE_SUMMARY,
            user_content=user_input,
        )
        if guardrails is not None and not getattr(
            client, "fallback_mode", False
        ):
            guardrails.record_call()
    except LLMClientError as exc:
        _print(f"[錯誤] /check 第一階段呼叫失敗：{exc}")
        check_mod.discard_session(session_dir, session_id)
        return False

    # 5. 寫回 session state
    check_mod.save_stage1_output(session_dir, session_id, stage1_output)

    # 6. 印出第一階段輸出（含選單；選單在 ISF prompt 裡就會帶出來）
    _print("[逗福Tofu /check] 第一階段（精緻快速摘要）：")
    _print(stage1_output)
    _print(
        "[逗福Tofu] 輸入數字接續："
        "1 = 完整報告；2 = 警示文字稿；3 = 結束。"
    )
    return True


def run_check_followup(
    choice: str,
    store: EndpointStore,
    client: LLMClient,
    guardrails: GuardrailTracker | None = None,
    session_dir: str = SESSIONS_DIR,
) -> bool:
    """使用者輸入 ``1`` / ``2`` / ``3`` 時的接續處理。

    找最近的 /check session；讀取失敗或沒有 session → fallback 提示使用者
    重新打 ``/check``（不報錯、不拋例外）。成功接續後：
    - choice=1 → 執行第二階段完整報告
    - choice=2 → 執行警示文字稿
    - choice=3 → 結束，不呼叫 LLM
    三者都會把該 session 的 start/end 端點寫進 endpoints.json（mode='check'），
    然後清除 session state 檔。
    """
    session = check_mod.find_latest_session(session_dir)
    if session is None:
        _print(
            "[逗福Tofu] 找不到最近的 /check 分析紀錄。"
            "請重新輸入 /check <內容> 開始一輪新分析。"
        )
        return False

    session_id = session["session_id"]
    user_content = session.get("user_content", "")
    stage1_output = session.get("stage1_output")

    stage_map = {"1": STAGE_FULL, "2": STAGE_WARNING, "3": STAGE_END}
    stage = stage_map.get(choice)
    if stage is None:  # 理論上 dispatcher 已過濾，這裡是保險
        _print(f"[逗福Tofu] 不認得的選項：{choice}。請輸入 1 / 2 / 3。")
        return False

    # 針對 1/2 的 LLM 呼叫要先過護欄；3 不呼叫 LLM
    final_output: str
    if stage == STAGE_END:
        final_output = (
            "[逗福Tofu /check] 已結束本次分析。"
            "若之後還有疑問可隨時再用 /check 或撥 165 查證。"
        )
        _print(final_output)
    else:
        if guardrails is not None:
            try:
                guardrails.check_before_call()
            except GuardrailTripped as tripped:
                _print(f"[逗福Tofu 系統訊息] {tripped}")
                _print(
                    "[逗福Tofu] /check 後續階段需要額外 LLM 呼叫，"
                    "目前餘額不足，保留 session 供下次接續。"
                )
                return False

        try:
            final_output = client.run_check_stage(
                stage=stage,
                user_content=user_content,
                stage1_output=stage1_output,
            )
            if guardrails is not None and not getattr(
                client, "fallback_mode", False
            ):
                guardrails.record_call()
        except LLMClientError as exc:
            _print(f"[錯誤] /check 階段 {stage} 呼叫失敗：{exc}")
            return False

        label = {
            STAGE_FULL: "第二階段（完整報告）",
            STAGE_WARNING: "警示文字稿",
        }[stage]
        _print(f"[逗福Tofu /check] {label}：")
        _print(final_output)

    # 寫入端點：start + end，mode='check'，不進 baseline（baseline 已排除）
    event_id = new_event_id()
    goal_summary = user_content.strip().replace("\n", " ")
    if len(goal_summary) > 60:
        goal_summary = goal_summary[:60] + "…"
    start_data = {
        "user_input": user_content,
        "tofu_understanding": user_content,
        "gap_questions": [],
        "user_confirmation": "auto_check",
        "user_modifications": "",
        "goal": goal_summary,
        "motivation": "",
        "constraints": [],
        "gap_categories": [],
        "mode": "check",
        "zone": confirm_mod.classify_zone(user_content),
        "topic": confirm_mod.infer_topic(user_content, []),
        "check_session_id": session_id,
        "check_stage1_output": stage1_output,
        # v4.1：/check 不走補位流程。
        "unresolved_gaps": [],
        "convergence_reason": None,
        "total_gap_rounds": 0,
    }
    store.append_start(event_id=event_id, start_data=start_data)

    end_data = make_end_data(
        result=final_output,
        user_satisfaction="no_feedback",
        deviation_from_goal="",
    )
    end_data["zone"] = confirm_mod.classify_zone(final_output)
    end_data["check_final_stage"] = stage
    check_end_row = store.append_end(event_id=event_id, end_data=end_data)
    _print("[端點已記錄]")

    # v0.6 PR-E：端點寫入後觸發詞性標記（失敗靜默）
    _try_tag_endpoint(
        endpoint_id=check_end_row["endpoint_id"],
        user_input=user_content,
        result=final_output,
    )

    # 清除 session state
    check_mod.discard_session(session_dir, session_id)

    # session 是一次完整互動
    if guardrails is not None:
        guardrails.record_session()
    return True


def run_form_command(
    store: EndpointStore,
    client: LLMClient,
    ask_satisfaction: bool = False,
    profile_store: ProfileStore | None = None,
    guardrails: GuardrailTracker | None = None,
) -> bool:
    completed = store.completed_count()
    if not form_mod.should_generate(completed):
        _print(
            f"[逗福Tofu] 目前累積 {completed} 筆互動，提問單需累積至 5 筆後才會生成。"
        )
        return False

    baseline = baseline_mod.compute_baseline(store.all())
    fields = form_mod.generate_form(baseline)
    _print(form_mod.render_form(fields))

    answers: dict[str, str] = {}
    for f in fields:
        ans = _prompt(f"  {f['label']}：").strip()
        if ans.lower() == "/skip":
            _print("[逗福Tofu] 已略過提問單。請直接輸入你的需求。")
            return False
        answers[f["key"]] = ans

    natural = form_mod.compose_natural_input(answers, fields)
    _print(f"[逗福Tofu] 已把你的填答整理成：「{natural}」")
    return run_one_interaction(
        natural, store, client,
        ask_satisfaction=ask_satisfaction,
        profile_store=profile_store,
        guardrails=guardrails,
    )


# ----------------------------------------------------------------------
# 新增 CLI 指令：/baseline /history /reset /export
# ----------------------------------------------------------------------
def run_baseline_command(store: EndpointStore) -> None:
    baseline = baseline_mod.compute_baseline(store.all())
    _print(baseline_mod.summarize_baseline(baseline))


def _iter_recent_events(store: EndpointStore, n: int = 5) -> list[tuple[dict, dict]]:
    """取最近 n 筆已完成的互動（依 start timestamp 排序）。"""
    pairs = store.completed_events()
    pairs_sorted = sorted(
        pairs,
        key=lambda p: p[0].get("timestamp", ""),
        reverse=True,
    )
    return pairs_sorted[:n]


def run_history_command(store: EndpointStore) -> None:
    recent = _iter_recent_events(store, n=5)
    if not recent:
        _print("[逗福Tofu] 尚無任何互動紀錄。")
        return
    _print("[逗福Tofu] 最近 5 筆互動：")
    for idx, (start_row, end_row) in enumerate(recent, start=1):
        sd = start_row.get("start_data") or {}
        ed = end_row.get("end_data") or {}
        goal = sd.get("goal") or sd.get("user_input") or "（未填）"
        result = ed.get("result") or "（未取得結果）"
        # 結果太長時截斷
        result_short = result.splitlines()[0][:60]
        satisfaction = ed.get("user_satisfaction", "no_feedback")
        satisfaction_label = {
            "satisfied": "滿意",
            "unsatisfied": "不滿意",
            "no_feedback": "未回饋",
        }.get(satisfaction, satisfaction)
        _print(f"  {idx}. 目標：{goal}")
        _print(f"     結果：{result_short}{'…' if len(result) > len(result_short) else ''}")
        _print(f"     滿意度：{satisfaction_label}")


def run_stats_command(store: EndpointStore) -> None:
    """P1-7：飛輪指標最小版。

    全部由程式碼計算，不呼叫 LLM。
    """
    all_rows = store.all()
    starts = [r for r in all_rows if r.get("type") == "start"]
    ends = [r for r in all_rows if r.get("type") == "end"]

    completed = store.completed_count()
    if completed == 0 and not starts:
        _print("[逗福Tofu] 尚無任何互動紀錄，/stats 暫時沒東西可顯示。")
        return

    # 復述修正率：有 user_modifications 且非空的 start 端點 / 全部 start 端點
    modified_starts = sum(
        1
        for r in starts
        if (r.get("start_data") or {}).get("user_modifications")
    )
    total_starts = len(starts)
    modification_ratio = (
        modified_starts / total_starts if total_starts else 0.0
    )

    # 滿意度分布
    sat_counter = {"satisfied": 0, "unsatisfied": 0, "no_feedback": 0}
    for r in ends:
        key = (r.get("end_data") or {}).get("user_satisfaction", "no_feedback")
        if key not in sat_counter:
            sat_counter[key] = 0
        sat_counter[key] += 1

    # 最常被補位面向：所有 start 端點的 gap_categories 累加
    from collections import Counter

    gap_counter: Counter[str] = Counter()
    for r in starts:
        for cat in (r.get("start_data") or {}).get("gap_categories") or []:
            if cat:
                gap_counter[cat] += 1
    top_gaps = gap_counter.most_common(3)

    form_ready = form_mod.should_generate(completed)

    # P0-1：Zone 分布（看 end_data 中的 zone 分類）
    zone_counter: Counter[str] = Counter()
    for r in ends:
        z = (r.get("end_data") or {}).get("zone") or "unknown"
        zone_counter[z] += 1

    # P0-2：黑盒子計數
    bb_total = len(store.black_boxes())

    # P0-4：topic 分布
    topic_counter: Counter[str] = Counter()
    for r in starts:
        t = (r.get("start_data") or {}).get("topic") or "unknown"
        topic_counter[t] += 1

    lines = [
        "[逗福Tofu] 使用統計：",
        f"  總互動數：{completed}",
        f"  復述修正率：{modification_ratio * 100:.1f}%"
        f"（{modified_starts}/{total_starts} 次復述被使用者修正）",
        f"  滿意度：satisfied {sat_counter['satisfied']} "
        f"/ unsatisfied {sat_counter['unsatisfied']} "
        f"/ no_feedback {sat_counter['no_feedback']}",
    ]
    if top_gaps:
        gap_text = "、".join(f"{cat} ({n}次)" for cat, n in top_gaps)
        lines.append(f"  最常被補位的面向：{gap_text}")
    else:
        lines.append("  最常被補位的面向：（尚無資料）")

    if sum(zone_counter.values()):
        zone_text = (
            f"A {zone_counter.get('A', 0)}"
            f" / B {zone_counter.get('B', 0)}"
            f" / C {zone_counter.get('C', 0)}"
            f" / unknown {zone_counter.get('unknown', 0)}"
        )
        lines.append(f"  結果 Zone 分布：{zone_text}")

    if sum(topic_counter.values()):
        topic_text = "、".join(
            f"{t} ({n}次)" for t, n in topic_counter.most_common()
        )
        lines.append(f"  事件主題分布：{topic_text}")

    lines.append(f"  黑盒子回溯次數：{bb_total}")

    # v2.0：ATL 通過率
    atl_action_total = 0
    atl_action_pass = 0
    atl_fals_total = 0
    atl_fals_generic = 0
    for r in ends:
        ed = r.get("end_data") or {}
        ac = ed.get("atl_action_check")
        if ac:
            atl_action_total += 1
            if ac.get("is_specific"):
                atl_action_pass += 1
        fc = ed.get("atl_falsification_check")
        if fc:
            atl_fals_total += 1
            if fc.get("is_generic"):
                atl_fals_generic += 1

    if atl_action_total:
        atl3_rate = atl_action_pass / atl_action_total * 100
        lines.append(f"  ATL-3 具體性通過率：{atl3_rate:.0f}%（{atl_action_pass}/{atl_action_total}）")
    if atl_fals_total:
        atl1_generic = atl_fals_generic / atl_fals_total * 100
        lines.append(f"  ATL-1 萬用句型比率：{atl1_generic:.0f}%（{atl_fals_generic}/{atl_fals_total}）")

    form_state = "已生成" if form_ready else "未就緒"
    lines.append(f"  提問單狀態：{form_state}（{completed} 筆基礎）")
    _print("\n".join(lines))


def run_blackbox_command(store: EndpointStore, n: int = 5) -> None:
    """P0-2：顯示最近 n 筆黑盒子回溯紀錄。"""
    boxes = store.black_boxes()
    if not boxes:
        _print("[逗福Tofu] 尚無黑盒子回溯紀錄——代表目前所有互動的結果都沒偏離目標。")
        return
    recent = boxes[-n:][::-1]
    _print(f"[逗福Tofu] 最近 {len(recent)} 筆黑盒子回溯（新→舊）：")
    for idx, row in enumerate(recent, start=1):
        bb = row.get("black_box") or {}
        sd = bb.get("start_data") or {}
        goal = sd.get("goal") or sd.get("user_input") or "（未填）"
        result = bb.get("result") or "（未取得結果）"
        deviation = bb.get("deviation") or "（未記錄）"
        analysis = bb.get("analysis") or "（未做出口檢查）"
        result_short = result.splitlines()[0][:60]
        _print(f"  {idx}. 事件：{row.get('event_id', '')}")
        _print(f"     目標：{goal}")
        _print(f"     結果：{result_short}{'…' if len(result) > len(result_short) else ''}")
        _print(f"     偏差：{deviation}")
        _print(f"     出口檢查：{analysis}")
        _print(f"     時間：{bb.get('timestamp', row.get('timestamp', ''))}")


def run_profile_command(profile_store: ProfileStore) -> None:
    """v3.0 P0-7：顯示使用者畫像 + 嵌入確認項。"""
    profile = profile_store.load()
    _print(render_profile_summary(profile))

    # 嵌入確認項（在欄位資料來源不是 default/None 時詢問，
    # 涵蓋 observed、pillars_mapped、pillars_validated 等）
    cs = profile.get("communication_style", {})
    if cs.get("data_source") not in ("default", None):
        _print("")
        answer = _prompt("  溝通風格描述對嗎？(對/不對/跳過) > ").strip()
        if answer in {"不對", "no", "n"}:
            new_val = _prompt("  你覺得接收偏好應該是？(structured/conversational/minimal) > ").strip()
            if new_val in ("structured", "conversational", "minimal"):
                profile_store.apply_human_calibration(
                    "communication_style.receiving_preference", new_val
                )
                _print("[逗福Tofu] 已更新，感謝校準。")

    ds = profile.get("decision_style", {})
    if ds.get("data_source") not in ("default", None):
        answer = _prompt("  決策方式描述對嗎？(對/不對/跳過) > ").strip()
        if answer in {"不對", "no", "n"}:
            new_val = _prompt("  你覺得決策速度應該是？(fast/deliberate/avoidant) > ").strip()
            if new_val in ("fast", "deliberate", "avoidant"):
                profile_store.apply_human_calibration(
                    "decision_style.pace", new_val
                )
                _print("[逗福Tofu] 已更新，感謝校準。")


def run_pillars_command(profile_store: ProfileStore) -> None:
    """v3.0 P1-2：四柱輸入介面。有多少記多少。"""
    _print("[逗福Tofu] 輸入你已知的四柱天干地支（不知道的直接按 Enter 跳過）：")
    profile = profile_store.load()
    raw_pillars = profile.get("pillars")
    pillars = raw_pillars if isinstance(raw_pillars, dict) else {}
    profile["pillars"] = pillars

    for pillar_name, label in [("year", "年柱"), ("month", "月柱"), ("day", "日柱"), ("hour", "時柱")]:
        stem_input = _prompt(f"  {label}天干（甲乙丙丁戊己庚辛壬癸）> ").strip()
        branch_input = _prompt(f"  {label}地支（子丑寅卯辰巳午未申酉戌亥）> ").strip()

        if stem_input and stem_input in VALID_STEMS:
            pillars.setdefault(pillar_name, {})["stem"] = stem_input
        if branch_input and branch_input in VALID_BRANCHES:
            pillars.setdefault(pillar_name, {})["branch"] = branch_input

    pillars["source"] = "user_input"
    # 計算完整度
    has_data = any(
        pillars.get(p, {}).get("stem") or pillars.get(p, {}).get("branch")
        for p in ("year", "month", "day", "hour")
    )
    all_stems = all(
        pillars.get(p, {}).get("stem") for p in ("year", "month", "day", "hour")
    )
    if all_stems:
        pillars["completeness"] = "full"
    elif has_data:
        pillars["completeness"] = "partial"
    else:
        pillars["completeness"] = "none"

    profile["pillars"] = pillars
    profile_store.save(profile)

    # 自動查映射表（透過 ProfileStore 方法，尊重 observed > pillars_mapped 優先權）
    _profile, msg = profile_store.apply_pillars_mapping(pillars)
    _print(f"[逗福Tofu] {msg}")


def run_reset_command(store: EndpointStore) -> None:
    total = len(store.all())
    if total == 0:
        _print("[逗福Tofu] 目前沒有任何端點紀錄，無需清除。")
        return
    _print(f"[逗福Tofu] 即將清除 {total} 筆端點紀錄，此動作無法復原。")
    confirm = _prompt("  確認清除？(輸入 YES 確認) > ").strip()
    if confirm != "YES":
        _print("[逗福Tofu] 已取消清除。")
        return
    store.clear()
    _print("[逗福Tofu] 已清除所有端點紀錄。")


def run_export_command(store: EndpointStore) -> None:
    pairs = store.completed_events()
    if not pairs:
        _print("[逗福Tofu] 尚無已完成的互動可匯出。")
        return

    # 依時間排序
    pairs_sorted = sorted(pairs, key=lambda p: p[0].get("timestamp", ""))
    # 匯出位置跟隨 store 自身的端點路徑，而非模組層級的 DATA_PATH。
    # 這樣當使用者用 TOFU_ENDPOINTS_PATH 指定自訂路徑時，匯出檔會落到正確的目錄。
    export_dir = Path(store.path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "endpoints_export.md"

    baseline = baseline_mod.compute_baseline(store.all())
    lines: list[str] = [
        "# 逗福Tofu 端點紀錄匯出",
        "",
        f"- 總互動數：{len(pairs_sorted)}",
        f"- 修正率：{baseline.get('modified_ratio', 0.0) * 100:.1f}%",
        "",
        "---",
        "",
    ]
    for idx, (start_row, end_row) in enumerate(pairs_sorted, start=1):
        sd = start_row.get("start_data") or {}
        ed = end_row.get("end_data") or {}
        lines.append(f"## {idx}. {sd.get('goal') or sd.get('user_input') or '（未填）'}")
        lines.append("")
        lines.append(f"- 時間：{start_row.get('timestamp', '')}")
        lines.append(f"- 原始輸入：{sd.get('user_input', '')}")
        lines.append(f"- 逗福Tofu 復述：{sd.get('tofu_understanding', '')}")
        gqs = sd.get("gap_questions") or []
        if gqs:
            lines.append(f"- 補位提問：{'、'.join(gqs)}")
        lines.append(f"- 確認狀態：{sd.get('user_confirmation', '')}")
        if sd.get("user_modifications"):
            lines.append(f"- 使用者修正：{sd.get('user_modifications', '')}")
        lines.append(f"- 動機：{sd.get('motivation', '') or '（未填）'}")
        cs = sd.get("constraints") or []
        if cs:
            lines.append(f"- 限制條件：{'、'.join(cs)}")
        lines.append("")
        lines.append("**執行結果：**")
        lines.append("")
        result = ed.get("result", "")
        for rline in result.splitlines():
            lines.append(f"> {rline}")
        lines.append("")
        lines.append(f"- 滿意度：{ed.get('user_satisfaction', 'no_feedback')}")
        if ed.get("deviation_from_goal"):
            lines.append(f"- 偏差偵測：{ed.get('deviation_from_goal', '')}")
        if ed.get("unsatisfied_reason"):
            lines.append(f"- 不滿意原因：{ed.get('unsatisfied_reason', '')}")
        lines.append("")

    export_path.write_text("\n".join(lines), encoding="utf-8")
    _print(f"[逗福Tofu] 已匯出 {len(pairs_sorted)} 筆互動到 {export_path}")


# ----------------------------------------------------------------------
# 主迴圈
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    # v0.6 PR-H audit #4：解析 CLI 參數（目前只支援 `--batch`）。
    # 不用 argparse 避免引入相依；手動解析就好，未來有更多 flag 再補上。
    # 僅在 argv **顯式傳入**時嚴格檢查未知參數；未傳入時讀 sys.argv[1:]
    # 但不對未知參數抱怨——因為 unittest 等 runner 會把自身 argv 塞進來，
    # 不該讓 CLI main() 在這些情境下 return 非零。
    global _BATCH_MODE
    explicit_argv = argv is not None
    args = list(argv if explicit_argv else sys.argv[1:])
    if "--batch" in args:
        _BATCH_MODE = True
        args = [a for a in args if a != "--batch"]
    if explicit_argv and args:
        _print(f"[逗福Tofu] 未知參數：{args}（目前支援：--batch）")
        return 2

    store = EndpointStore(DATA_PATH)
    profile_store = ProfileStore(PROFILE_PATH)
    zone_history_store = ZoneHistoryStore(ZONE_HISTORY_PATH)
    guardrails = GuardrailTracker()

    # v0.6 PR-D 小修：啟動時也跑一次 /check session sweep。
    # 光在 run_check_mode 裡 sweep 不夠——如果使用者再也不打 /check，
    # 累積的 session 檔會永遠卡在磁碟上。啟動時掃一次，確保系統每次冷
    # 啟動都有清潔時刻。靜默執行，不印訊息避免干擾首次使用引導。
    try:
        check_mod.sweep_old_sessions(SESSIONS_DIR)
    except OSError:
        pass  # 磁碟權限問題不該擋下 CLI 啟動

    _print("逗福Tofu 公測版 — 認知中間層")
    _print("-" * 48)

    # 首次使用引導
    if len(store.all()) == 0:
        _print(WELCOME_TEXT)
        _print("-" * 48)
    else:
        _print("輸入 /help 查看所有指令；輸入 /exit 離開。")
        _print("-" * 48)

    # 公測版後端：Claude（推薦 Haiku）。沒有 API key 時自動進入離線 fallback。
    client = LLMClient(notifier=_print)

    if client.fallback_mode:
        _print("[逗福Tofu] 目前為離線模式，設定 CLAUDE_API_KEY 後可獲得 AI 驅動的回應。")

    # 滿意度問答節奏降低：v3.0 從每 3 次改為每 10 次
    global SATISFACTION_INTERVAL
    SATISFACTION_INTERVAL = 10

    while True:
        # 在每次 prompt 前顯示狀態列（batch 模式下抑制，避免污染 stdout）
        if not _BATCH_MODE:
            _print(_status_line(store))
        try:
            line = _prompt("> ").strip()
        except KeyboardInterrupt:
            _print("\n[逗福Tofu] 已中斷。")
            return 0

        if not line:
            continue
        lowered = line.lower()

        # v0.6 PR-D：/check 接續 —— 使用者輸入 1 / 2 / 3 且最近有 active check
        # session，視為第二階段選擇。若沒 active session，不攔截（可能是
        # 使用者自然語言就是「1」「2」「3」等無關輸入）。
        if line.strip() in {"1", "2", "3"}:
            active = check_mod.find_latest_session(SESSIONS_DIR)
            if active is not None:
                try:
                    run_check_followup(
                        choice=line.strip(),
                        store=store,
                        client=client,
                        guardrails=guardrails,
                    )
                except SystemExit:
                    _print("[逗福Tofu] 再見。")
                    return 0
                continue

        if lowered in {"/exit", "/quit"}:
            _print("[逗福Tofu] 再見。")
            return 0
        if lowered == "/help":
            _print(_render_help())
            continue
        if lowered == "/about":
            _print(_ABOUT_TEXT)
            continue
        if lowered == "/baseline":
            run_baseline_command(store)
            continue
        if lowered == "/history":
            run_history_command(store)
            continue
        if lowered == "/stats":
            run_stats_command(store)
            continue
        if lowered == "/blackbox":
            run_blackbox_command(store)
            continue
        if lowered == "/profile":
            run_profile_command(profile_store)
            continue
        if lowered == "/pillars":
            run_pillars_command(profile_store)
            continue
        if lowered == "/reset":
            run_reset_command(store)
            continue
        if lowered == "/export":
            run_export_command(store)
            continue
        if lowered == "/form":
            ask_sat = _should_ask_satisfaction(store.completed_count() + 1)
            run_form_command(
                store, client, ask_satisfaction=ask_sat,
                profile_store=profile_store, guardrails=guardrails,
            )
            continue

        # v0.6 PR-B：五模式開關 slash dispatcher
        # /free 和 /risk 走 run_one_interaction(mode=...)，共用思考引擎。
        # /propose 與 /check 的實作分別留給 PR-C 與 PR-D，這裡只印「即將上線」
        # 提示，避免使用者以為功能壞掉。
        parsed = _parse_mode_command(line)
        if parsed is not None:
            cmd_mode, content = parsed
            if cmd_mode in ("free", "risk"):
                if not content:
                    _print(
                        f"[逗福Tofu] /{cmd_mode} 需要在指令後面接內容。"
                        f"例如：/{cmd_mode} 幫我規劃週末行程。"
                    )
                    continue
                try:
                    ask_sat = _should_ask_satisfaction(
                        store.completed_count() + 1
                    )
                    run_one_interaction(
                        content, store, client,
                        ask_satisfaction=ask_sat,
                        profile_store=profile_store,
                        guardrails=guardrails,
                        mode=cmd_mode,
                        zone_history_store=zone_history_store,
                    )
                except SystemExit:
                    _print("[逗福Tofu] 再見。")
                    return 0
                continue
            if cmd_mode == "propose":
                if not content:
                    _print(
                        "[逗福Tofu] /propose 需要在指令後面接提案主題。"
                        "例如：/propose 規劃一場 50 人的線上發表會。"
                    )
                    continue
                try:
                    ask_sat = _should_ask_satisfaction(
                        store.completed_count() + 1
                    )
                    run_propose_mode(
                        content, store, client,
                        ask_satisfaction=ask_sat,
                        profile_store=profile_store,
                        guardrails=guardrails,
                    )
                except SystemExit:
                    _print("[逗福Tofu] 再見。")
                    return 0
                continue
            if cmd_mode == "check":
                if not content:
                    _print(
                        "[逗福Tofu] /check 需要在指令後面接要檢驗的內容。"
                        "例如：/check 我收到一封說我中獎的信。"
                    )
                    continue
                try:
                    run_check_mode(
                        content, store, client,
                        guardrails=guardrails,
                    )
                except SystemExit:
                    _print("[逗福Tofu] 再見。")
                    return 0
                continue

        if line.startswith("/"):
            _print(
                f"[逗福Tofu] 未知指令：{line}（可用：/free /risk /propose /check；"
                "輸入 /help 查看其他指令）"
            )
            continue

        try:
            ask_sat = _should_ask_satisfaction(store.completed_count() + 1)
            run_one_interaction(
                line, store, client, ask_satisfaction=ask_sat,
                profile_store=profile_store,
                guardrails=guardrails,
                zone_history_store=zone_history_store,
            )
        except SystemExit:
            _print("[逗福Tofu] 再見。")
            return 0


if __name__ == "__main__":
    sys.exit(main())
