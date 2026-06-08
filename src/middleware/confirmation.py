"""復述確認模組。

負責：
1. 偵測使用者輸入是『確認』還是『修正』（純程式碼字串判斷）。
2. 驅動一次完整的『產生復述 → 等使用者回應 → 合併補位回答』流程。
3. ATL（反演示層）品質驗證（v2.0 P1-1/P1-2/P1-3）。

本模組本身不呼叫 LLM，但會調用 claude_client 模組。
邏輯判斷（是否為修正、關鍵詞偵測）完全在程式碼層。

v2.0 變更：
- P0-1：斷詞改為呼叫 tokenizer.tokenize()
- P0-2：否定詞偵測升級為三層（肯定詞短路 + 固定片語 + 斷詞後模式比對）
- P1-1：ATL-3 下一步具體性檢測
- P1-2：ATL-1 可反駁條件可操作性檢測
- P1-3：ATL-2 來源可回溯性檢測
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.tokenizer import tokenize as _tokenize_text

# ----------------------------------------------------------------------
# v2.0 P0-2：否定詞偵測升級（三層偵測）
# ----------------------------------------------------------------------
# 第零層：肯定詞短路——整句就是肯定詞時，直接判定為非修正。
AFFIRMATION_PHRASES = [
    "沒錯", "不錯", "還不錯", "對", "是的", "好的", "可以",
    "就這樣", "沒問題", "ok", "okay", "好", "嗯", "是",
    "yes", "yeah", "yep", "sure", "right", "correct",
]

# 第一層：固定片語（保留原清單 + 擴充）
NEGATION_PHRASES = [
    # 原有清單
    "不對", "不是這樣", "不是", "錯了", "錯誤", "不正確",
    "弄錯", "搞錯", "誤解", "修正", "不該", "再想想",
    "wrong", "incorrect", "not right", "not correct",
    "that's not", "no,", "no ",
    # v2.0 擴充
    "不太對", "不太正確", "不太準",
    "有點偏", "偏了",
    "想調整", "想改", "要改", "改一下",
    "換個方向",
    "跟我想的不一樣", "不完全是", "不完全對",
    "少了", "漏了", "漏掉", "沒提到",
    "重新", "重來", "再來一次",
]

# 第二層：斷詞後模式比對的否定副詞 + 修正動詞
_NEGATION_ADVERBS = {"不", "不太", "沒", "沒有", "未", "別", "勿"}
_CORRECTION_VERBS = {
    "對", "正確", "準", "準確", "完整", "夠", "好",
    "是", "算", "符合", "一樣", "理解",
}


def is_modification(user_text: str) -> bool:
    """偵測使用者對復述的回應是否為『修正』。

    v2.0 三層偵測：
    第零層：肯定詞短路
    第一層：固定片語比對
    第二層：斷詞後否定副詞 + 修正動詞組合
    """
    if not user_text:
        return False

    stripped = user_text.strip()
    lowered = stripped.lower()

    # 第零層：肯定詞短路——整句就是肯定詞，直接回 False
    for phrase in AFFIRMATION_PHRASES:
        if lowered == phrase or lowered == phrase.lower():
            return False

    # 第一層：固定片語
    for phrase in NEGATION_PHRASES:
        if phrase in lowered:
            return True

    # 第二層：斷詞後模式比對（min_token_len=1 以保留否定副詞單字 token）
    tokens = _tokenize_text(stripped, min_token_len=1)
    for i, tok in enumerate(tokens):
        if tok in _NEGATION_ADVERBS:
            # 檢查下一個 token 是否為修正動詞
            if i + 1 < len(tokens) and tokens[i + 1] in _CORRECTION_VERBS:
                return True

    return False


def classify_confirmation(user_text: str) -> tuple[str, str]:
    """回傳 (user_confirmation, user_modifications)。

    - confirmed: 沒有偵測到否定，user_modifications 為空字串
    - modified: 偵測到否定，user_modifications 為整段使用者輸入（作為修正內容）
    """
    if is_modification(user_text):
        return "modified", user_text.strip()
    return "confirmed", ""


def build_restate_payload(
    user_input: str,
    baseline: dict[str, Any],
    allowed_categories: list[str],
    client,  # type: ignore[no-untyped-def]
    *,
    strategy_brief: str | None = None,
    encoded_top_k: str | None = None,
    first_round_direction_hint: str | None = None,
    prior_rounds_context: str | None = None,
) -> dict[str, Any]:
    """呼叫 LLM 產生復述。回傳已正規化的 dict。

    v0.5 新增三個 keyword-only 參數，來自：
    - baseline.compute_strategy_brief()（畫像線策略摘要）
    - endpoint.retrieve_top_k_endpoints → encode_retrieved_endpoints（密碼表）
    - confirmation.detect_first_round_direction()（首輪方向偵測）
    舊 caller 不帶新參數時，行為與 v0.3 相同。

    v4.1 收斂閘門新增一個 keyword-only 參數：
    - prior_rounds_context: 多輪補位中，把前幾輪問過的題與使用者回答串成一段
      純文字，讓 LLM 這一輪不要重複問同樣的維度。單輪互動（第一輪 / 未進多輪）
      傳 None 即可。

    兼容性：舊式 client（測試用的 mock / 第三方 backend）若不認得 v0.5 或 v4.1
    的 keyword-only 參數，我們會退回去用更舊的簽名呼叫。
    """
    kwargs = {
        "user_input": user_input,
        "baseline_summary": baseline,
        "allowed_categories": allowed_categories,
        "strategy_brief": strategy_brief,
        "encoded_top_k": encoded_top_k,
        "first_round_direction_hint": first_round_direction_hint,
    }
    if prior_rounds_context:
        kwargs["prior_rounds_context"] = prior_rounds_context
    try:
        return client.generate_restate(**kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" not in msg and "positional argument" not in msg:
            raise
        # 第一階回退：先拿掉 prior_rounds_context（v4.1 新參數）。
        if "prior_rounds_context" in kwargs:
            kwargs.pop("prior_rounds_context", None)
            try:
                return client.generate_restate(**kwargs)
            except TypeError as exc2:
                msg = str(exc2)
                if "unexpected keyword argument" not in msg and \
                        "positional argument" not in msg:
                    raise
        # 第二階回退：舊式 v0.3 簽名。
        return client.generate_restate(
            user_input=user_input,
            baseline_summary=baseline,
            allowed_categories=allowed_categories,
        )


def finalize_start_data(
    user_input: str,
    restate: dict[str, Any],
    user_response: str,
    client,  # type: ignore[no-untyped-def]
) -> dict[str, Any]:
    """把 restate 結果與使用者的補位回答合併成 start_data 所需欄位。

    MVP 策略：
    - 判定確認/修正由程式碼做。
    - 實際的『把補位答案合併進 goal/motivation/constraints』由 LLM 做，
      因為純字串拼接品質太差。
    - 合併後的結構由 LLM 以 JSON 回傳，程式碼接手後再寫入端點。
    """
    confirmation, modifications = classify_confirmation(user_response)

    # 無論是確認或修正，都把使用者這次的回應丟給 LLM 做結構化合併。
    # （修正情況下，修正內容會覆蓋掉 restate 中的錯誤理解。）
    merged = client.merge_confirmation(
        original_input=user_input,
        restate_text=restate.get("restate_text", ""),
        user_confirmation_text=user_response,
    )

    # 如果 LLM 合併後沒給出 goal，退回使用 restate 推斷的 goal。
    final_goal = merged.get("goal") or restate.get("inferred_goal") or user_input
    final_motivation = merged.get("motivation") or restate.get("inferred_motivation") or ""
    final_constraints = merged.get("constraints") or restate.get("inferred_constraints") or []

    out: dict[str, Any] = {
        "user_input": user_input,
        "tofu_understanding": restate.get("restate_text", ""),
        "gap_questions": restate.get("gap_questions", []),
        "user_confirmation": confirmation,
        "user_modifications": modifications,
        "goal": final_goal,
        "motivation": final_motivation,
        "constraints": final_constraints,
        "gap_categories": restate.get("gap_categories", []),
        "_answered_categories": merged.get("answered_categories", []),
    }
    # P1-3：若 restate 含六步 OS 的 reasoning_steps，一併帶回 start_data
    # （可選欄位，缺少時不影響流程）
    if restate.get("reasoning_steps"):
        out["reasoning_steps"] = restate["reasoning_steps"]
    # v4.0 P1：若 restate 含 LLM 搭車偵測的 emotion_state，一併帶回
    # （main.py 會優先採用這個，fallback 才用規則式偵測）
    if restate.get("emotion_state"):
        out["emotion_state"] = restate["emotion_state"]
    return out


def detect_deviation(goal: str, result: str) -> str:
    """程式碼層的偏差偵測。

    三層檢查（依 TOFU_功能實作開發規格 P0-2）：
    1. 關鍵詞重疊（v2.0：改用 tokenizer 斷詞比對）
    2. 同義詞映射（生日=慶生、預算=經費=費用）
    3. 長度異常（結果過短 + 無關鍵詞匹配 → 合併標記為偏離）

    回傳空字串表示無偏差；否則回傳簡短描述（含「偏離」字樣）。
    """
    if not goal or not result:
        return ""

    goal_lower = goal.lower()
    result_lower = result.lower()

    # v2.0：使用 tokenizer 斷詞取代 2-gram
    tokens = _tokenize_text(goal)
    if not tokens:
        return ""

    # 第 1 層：關鍵詞重疊
    hit = any(t in result_lower for t in tokens)
    if hit:
        return ""

    # 第 2 層：同義詞映射——key 或其替代說法在 goal 出現時，
    # 檢查 result 是否包含該同義詞族中的任一說法。
    for key, syns in _SYNONYMS.items():
        family = [key] + syns
        if any(term.lower() in goal_lower for term in family):
            if any(term.lower() in result_lower for term in family):
                return ""

    # 第 3 層：長度檢查——關鍵詞完全沒對上，且結果異常簡短（< 10 chars），
    # 明顯是敷衍（例如「已完成」「好」「done」）。這層只在關鍵詞未命中後生效。
    if len(result.strip()) < 10:
        return "結果異常簡短且未觸及目標中的關鍵詞，可能偏離主題。"

    return "結果未觸及目標中的關鍵詞，可能偏離主題。"


# ----------------------------------------------------------------------
# P0-2：同義詞映射（基礎版）
# ----------------------------------------------------------------------
# 用於 detect_deviation 的同義詞比對。key 是「主形」，value 是等價的替代說法。
# 規格書 P0-2：持續擴充。
_SYNONYMS: dict[str, list[str]] = {
    "生日": ["慶生", "壽宴", "壽星"],
    "預算": ["經費", "費用", "花費", "開銷"],
    "場地": ["地點", "場所", "venue"],
    "派對": ["聚會", "party", "宴會"],
    "活動": ["event", "聚會"],
    "會議": ["meeting", "會談"],
    "報告": ["presentation", "簡報"],
    "專案": ["project", "計畫"],
}


# ----------------------------------------------------------------------
# P0-1：Zone A/B/C 分類
# ----------------------------------------------------------------------
# 依 TOFU_功能實作開發規格書 P0-1：
#   Zone A：事實（含可查證資訊：數字、日期、具體名詞）
#   Zone B：推測（含「可能」「通常」「根據經驗」等不確定詞）
#   Zone C：立場（含「建議」「我認為」「你可以考慮」等價值判斷）
# 這個分類結果記錄在端點的 start_data 和 end_data 裡，供 /stats 統計使用。
_ZONE_A_PATTERNS = [
    r"\d+",                        # 任何數字（金額、人數、日期）
    r"\d{4}[-/年]",                # 年份
    r"\d+[月日時分秒]",            # 日期時間
    r"\d+[元萬塊%％]",             # 金額 / 百分比
]

_ZONE_B_KEYWORDS = [
    "可能", "或許", "也許", "大概", "應該", "通常", "一般",
    "根據經驗", "據說", "聽說", "推測", "推估", "估計", "猜",
    "maybe", "probably", "perhaps", "might", "could be", "usually",
]

_ZONE_C_KEYWORDS = [
    "建議", "我認為", "我覺得", "我想", "依我看", "值得",
    "你可以考慮", "不妨", "最好", "應該要", "不建議",
    "recommend", "suggest", "i think", "you should", "you can consider",
    "in my opinion",
]


# ----------------------------------------------------------------------
# P0-4：CBP 跨事件邊界——topic 推斷
# ----------------------------------------------------------------------
# 依 TOFU_功能實作開發規格書 P0-4：防止不同事件的端點記錄被混在一起推演。
# 從 goal 和 gap_categories 推斷主題分類。純程式碼，不呼叫 LLM。
# ----------------------------------------------------------------------
# P1-1：CIP-X 軌跡收斂（基礎版）
# ----------------------------------------------------------------------
# 規格書 P1-1：不做完整的軌跡收斂（那是完整版的事），先做最小的安全防線。
# 這不是關鍵字過濾——是趨勢偵測的輸入。單一筆出現危險詞不觸發，
# 連續多筆都出現才觸發。
DANGER_KEYWORDS: list[str] = [
    "傷害自己", "自殺", "自殘", "自我了結", "輕生",
    "殺害", "殺人", "殺了", "刺殺", "毒殺",
    "下毒", "炸彈", "炸藥", "爆破",
    "攻擊", "報復",
    "harm myself", "suicide", "kill", "attack", "weapon", "bomb",
]


# ----------------------------------------------------------------------
# v0.5 第五層：首輪方向偵測
# ----------------------------------------------------------------------
# 依 20260416_邏輯鏈_v0_5.md 第五層規則：
#   通則尚未建立時（completed_rounds < 3），使用者第一題只提到自己
#   → 兩個反問（一自己一別人）
# 後續依通則方向決定補位順序（這部分由畫像線和基線在後續輪次承接）。
# ----------------------------------------------------------------------
_SELF_PRONOUNS = [
    "我", "我的", "我想", "我要", "我該", "我在",
    "me", "my", "mine", "myself", "i ", "i'm", "i am", "i'd",
]
_OTHER_SIGNALS = [
    "她", "他", "他們", "她們", "家人", "爸", "媽", "父", "母",
    "朋友", "夥伴", "同事", "老闆", "客戶", "孩子", "小孩", "老婆", "老公",
    "伴侶", "隊友", "員工", "下屬", "同學", "鄰居", "團隊",
    "她的", "他的", "他們的",
    "she", "he", "they", "them", "friend", "family", "colleague",
    "boss", "client", "partner", "parents", "mom", "mum", "dad",
    "kids", "children", "wife", "husband", "teammate",
]

FIRST_ROUND_GENERAL_THRESHOLD = 3  # 已完成事件數 < 此門檻視為通則未建立


def detect_first_round_direction(
    user_input: str,
    completed_rounds: int,
    threshold: int = FIRST_ROUND_GENERAL_THRESHOLD,
) -> str | None:
    """偵測首輪方向。

    回傳：
    - "self_only": 通則未建立 + 輸入只提到自己 → 要兩個反問（一自己一別人）
    - None: 其他情況（通則已建立，或輸入已涉及他人）

    通則建立門檻沿用 `baseline.is_mature` 的 3 筆概念；completed_rounds
    為當前已完成事件數。caller 可自行覆寫 threshold。
    """
    if completed_rounds is None or completed_rounds >= threshold:
        return None
    if not user_input:
        return None
    text = user_input.lower()

    has_other = any(sig.lower() in text for sig in _OTHER_SIGNALS)
    if has_other:
        return None

    has_self = any(p.lower() in text for p in _SELF_PRONOUNS)
    if has_self:
        return "self_only"
    # 若輸入既沒自稱也沒他人（例：純事件描述「辦派對」），
    # 仍視為 self_only——因為使用者只講了自己要做的事。
    return "self_only"


# 修法 B：跨題延續偵測（從嚴判準）。
# 接 v0.3 主詞代名詞層級精神，只認明確指代詞；純省略主詞交給近因保底(A)兜。
_CONTINUATION_MARKERS = (
    "那", "這", "它", "牠", "他", "她", "其", "此", "該",
    "上述", "剛剛", "剛才", "前面",
)


def detect_continuation(user_input: str, completed_count: int) -> bool:
    """本題是否為對上一題的延續式追問（從嚴）。

    True 的條件（全部成立）：
      1. 已有至少一筆完成端點（completed_count >= 1）。
      2. user_input 含 _CONTINUATION_MARKERS 任一指代詞。
    純省略主詞（不含指代詞，如「線上的呢」）一律回 False，靠近因保底(A)兜。

    從嚴原則：寧可漏判靠 A，不可誤判亂接。指代詞誤觸的代價最差只是多一句
    框定，A 端點帶 topic／zone 標記、LLM 可自行判斷消化。
    """
    if completed_count < 1:
        return False
    if not user_input:
        return False
    return any(m in user_input for m in _CONTINUATION_MARKERS)


# ----------------------------------------------------------------------
# 意圖分流（detect_intent_mode）
# 依 06209671 修復規格：意圖分流與規劃模式 v1 §2.1。
# 依「使用者最近一輪原始輸入的句式」分流，**不**用 Tofu 推測的 goal/motivation。
# 從嚴：唯有「命令句 + 執行目的」才判 execute；其餘一律 plan。
# 與 detect_continuation 同一從嚴哲學——誤判成 plan 成本低（再說一句即跨入
# 執行），誤判成 execute 成本高（婚禮題災難）。
# ----------------------------------------------------------------------

# 疑問特徵分兩層（依 PR #34 review：substring 比對對單字疑問詞會誤命中）。
_INTENT_QUESTION_MARKS = ("？", "?")
# 強疑問詞：多字、疑問語氣明確，出現在句中任意位置都可靠判為疑問，誤判率低。
_INTENT_STRONG_QUESTION_WORDS = (
    "怎麼", "如何", "要不要", "該不該", "是不是",
    "好不好", "可不可以", "有沒有", "為什麼",
)
# 弱疑問詞：「哪／什麼」是單字疑問詞，但也常用於「任意性」表達
# （哪天都行、什麼都可以），出現在明確執行命令裡不該攔截——故排在
# execute 判斷之後（見 detect_intent_mode 第 5 步）。
_INTENT_WEAK_QUESTION_WORDS = ("哪", "什麼")
# 句末語氣助詞：嗎／呢／吧。既可當疑問助詞，也可能是名詞的一部分
# （酒吧、吧台）或祈使軟化詞（訂吧）。只在「去除句末標點後仍以其結尾」時
# 才視為疑問，且判斷順序排在 execute 之後，避免「幫我預約這間酒吧」「幫我買
# 吧台椅」被尾字 substring 誤導成規劃。
_INTENT_QUESTION_FINAL_PARTICLES = ("嗎", "呢", "吧")
_INTENT_TRAILING_PUNCT = "？?！!。.，,、 \t\r\n　"

# 祈使開頭（命令句式特徵之一）。
_INTENT_IMPERATIVE_LEADS = ("幫我", "幫忙", "請幫", "請", "麻煩", "給我", "替我", "幫")

# 主詞起首 → 視為陳述句，非「動詞起首且無主詞」的命令句。
_INTENT_SUBJECT_LEADS = (
    "我", "你", "他", "她", "它", "牠", "我們", "你們", "他們", "她們",
    "大家", "咱", "本人",
)

# 執行目的詞（動作指向把事做掉）。命令句 + 命中其一 → execute。
# 「列出」屬整理清單/待辦的執行動作；「列出選項/可以考慮」等規劃語境會先被
# _INTENT_PLAN_VERBS（考慮/列出選項…）攔成 plan，故這裡可安全保留「列出」。
_INTENT_EXECUTE_VERBS = (
    "訂", "預訂", "購買", "買", "寄", "送出", "發送", "提交",
    "安排好", "排好", "列待辦", "列出清單", "列出待辦", "列出", "設定提醒",
    "建立", "開立", "下單", "付款", "報名", "預約",
)

# 動詞起首偵測用：以這些動作詞開頭、且無主詞 → 命令句式成立。
# 「列出我接下來要做的事」即靠此命中（列出 = 整理待辦）。
_INTENT_VERB_STARTS = _INTENT_EXECUTE_VERBS + ("列",)

# 排除清單（動作指向思考/規劃，即使是祈使句也回 plan）。
# 命中其一即直接 plan，優先於執行目的詞。
# 注意：刻意**不收**單字「想」——它以 substring 比對會攔截任何含「想」的句子
# （如「幫我訂我想要的票」），把明確執行命令誤判成規劃。思考意圖改用更精確的
# 「想想／想一想」承接；純「幫我想辦法」這類無執行動詞的句子本就會落到預設
# plan，不需要靠單字「想」攔截（PR #34 review）。
_INTENT_PLAN_VERBS = (
    "想想", "想一想", "思考", "規劃", "計畫", "評估", "比較", "分析",
    "建議", "推薦", "列出選項", "列選項", "看一下", "看看",
    "研究", "整理想法", "考慮",
)


def detect_intent_mode(user_input: str) -> str:
    """依使用者最近一輪原始輸入的句式分流，回傳 "execute" 或 "plan"。

    從嚴判準（規格 §2.1，含 PR #34 review 修正）：
    1. 預設 "plan"。
    2. 可靠疑問特徵（句末問號 / 強疑問詞）→ "plan"，**優先於**命令判斷。
    3. 含排除詞（想想/規劃/評估/比較/建議/考慮…）→ "plan"，
       即使是祈使句（命令逗福思考 ≠ 命令逗福執行）。
    4. 同時滿足「命令句式」與「執行目的詞」→ "execute"。
       此步刻意排在「弱疑問詞 / 句末語氣助詞」之前，讓明確執行命令即使含
       任意性的「哪/什麼」（哪天都行）或受詞尾字是「吧/呢」（酒吧、吧台椅），
       仍走工單路徑。
    5. 弱疑問特徵（任意位置的「哪/什麼」，或句末語氣助詞嗎/呢/吧）→ "plan"。
    6. 其餘（含無法明確判定）→ "plan"。

    判斷材料只有 user_input 本身，不看 Tofu 推測的 goal/motivation。

    已知接受的誤判（從嚴取捨，非 bug）：
    - 反問式催促「你可以現在幫我把票訂了嗎？」→ plan（句末問號優先）。
    - 同意+命令「好吧，幫我訂機票」→ plan（祈使詞不在句首、未擴張命令偵測，
      以免把陳述句「訂了機票好開心」誤升級為 execute——誤判成 execute 的成本
      遠高於誤判成 plan）。
    """
    if not user_input:
        return "plan"

    text = user_input.strip()
    if not text:
        return "plan"

    # 2. 可靠疑問特徵（句末問號 / 強疑問詞）→ plan
    if text.rstrip().endswith(_INTENT_QUESTION_MARKS):
        return "plan"
    if any(w in text for w in _INTENT_STRONG_QUESTION_WORDS):
        return "plan"

    # 3. 排除詞（思考/規劃導向）→ plan，優先於執行目的詞
    if any(v in text for v in _INTENT_PLAN_VERBS):
        return "plan"

    # 4. 命令句式 + 執行目的詞 → execute（排在弱疑問特徵之前，見 docstring）
    if _is_imperative(text) and any(v in text for v in _INTENT_EXECUTE_VERBS):
        return "execute"

    # 5. 弱疑問特徵 → plan：任意位置的「哪/什麼」，或句末語氣助詞嗎/呢/吧
    if any(w in text for w in _INTENT_WEAK_QUESTION_WORDS):
        return "plan"
    if text.rstrip(_INTENT_TRAILING_PUNCT).endswith(
        _INTENT_QUESTION_FINAL_PARTICLES
    ):
        return "plan"

    # 6. 其餘一律 plan（從嚴）
    return "plan"


def _is_imperative(text: str) -> bool:
    """命令句式偵測：祈使開頭，或動詞起首且無主詞。"""
    if text.startswith(_INTENT_IMPERATIVE_LEADS):
        return True
    if text.startswith(_INTENT_SUBJECT_LEADS):
        # 主詞起首視為陳述句，非「動詞起首且無主詞」的命令句。
        return False
    if text.startswith(_INTENT_VERB_STARTS):
        return True
    return False


def check_trajectory_convergence(
    recent_goals: list[str],
    threshold: int = 3,
) -> bool:
    """檢查最近 N 筆目標是否往危險方向收斂。

    不是看單一關鍵字（那是關鍵字過濾）。
    是看連續多筆目標是否都指向同一個危險方向。

    - 單一筆出現危險詞不觸發（可能只是在討論新聞）
    - 連續 threshold 筆都出現才觸發（趨勢收斂）
    """
    if not recent_goals or len(recent_goals) < threshold:
        return False

    recent = recent_goals[-threshold:]
    danger_count = sum(
        1 for goal in recent
        if any(kw.lower() in (goal or "").lower() for kw in DANGER_KEYWORDS)
    )
    return danger_count >= threshold


# ----------------------------------------------------------------------
# v4.1 收斂閘門（Convergence Gate）
# 依 docs/philosophy/20260424_逗福Tofu_收斂閘門_實作規格_v1.md
# ----------------------------------------------------------------------
# 第五層差額補位的跨輪停損：單輪 Gate 判斷「要不要補位」；收斂閘門判斷
# 「補了幾輪該不該停」。兩個條件先到先停：
#   empty_streak：連續 3 輪 `has_effective_info == False` → 停損。
#   max_rounds ：補位總輪數達 5 輪 → 強制收斂。
# 任一觸發後，尚未解的維度寫入 start_data.unresolved_gaps 供下輪參考。
CONVERGENCE_MAX_ROUNDS = 5
CONVERGENCE_EMPTY_STREAK_THRESHOLD = 3
# 短回應長度門檻：≤ 此字元數的回應（例如「好」「對」「沒有」）直接視為無效資訊。
CONVERGENCE_SHORT_RESPONSE_LEN = 5


def make_convergence_state() -> dict[str, Any]:
    """初始化 convergence_state（interaction 進行中的狀態，不寫入端點）。"""
    return {
        "total_rounds": 0,
        "empty_round_count": 0,
        "converged": False,
        "convergence_reason": None,
    }


def should_continue_gap_filling(
    convergence_state: dict[str, Any],
    current_round_result: bool,
) -> bool:
    """每輪補位提問後呼叫。

    Args:
        convergence_state: 由 make_convergence_state 產生的 state dict；函式會
            直接就地更新 total_rounds / empty_round_count / converged /
            convergence_reason 四個欄位。
        current_round_result: 本輪使用者回應是否包含有效資訊。True 會把空輪
            計數歸零；False 則累加。

    Returns:
        True = 可以繼續下一輪補位；False = 要停止補問、進 execute。
    """
    state = convergence_state
    state["total_rounds"] = int(state.get("total_rounds") or 0) + 1

    if current_round_result:
        state["empty_round_count"] = 0
    else:
        state["empty_round_count"] = int(state.get("empty_round_count") or 0) + 1

    if state["empty_round_count"] >= CONVERGENCE_EMPTY_STREAK_THRESHOLD:
        state["converged"] = True
        state["convergence_reason"] = "empty_streak"
        return False

    if state["total_rounds"] >= CONVERGENCE_MAX_ROUNDS:
        state["converged"] = True
        state["convergence_reason"] = "max_rounds"
        return False

    return True


def has_effective_info(
    user_response: str,
    existing_endpoints: list[dict[str, Any]] | None = None,
    *,
    existing_tokens: set[str] | None = None,
) -> bool:
    """判定使用者本輪回應是否包含「有效新增資訊」。

    不呼叫 LLM；依序檢查兩條規則：
      一、長度門檻：stripped 後 ≤ CONVERGENCE_SHORT_RESPONSE_LEN 字元 → 無效。
      二、新增名詞：jieba 斷詞後的 token 集合，與既有端點的 token 集合做差
          集；差集非空 → 有效。

    規格 §「有效資訊」的判定要求「寧可判多不判少」——本實作故意偏寬鬆：
    只要有一個沒見過的 token 就算有效，把是否收斂的決定權交給輪數累積。

    Performance（Copilot review #1）：多輪迴圈在每一輪呼叫本函式時、重複
    對 `existing_endpoints` 全量斷詞成本會線性累積。caller 可預先呼叫
    `extract_endpoint_tokens()` 把既有端點的 token 集合算好，再透過
    `existing_tokens` kwarg 直接注入，本函式會優先採用該快取；未提供時
    仍沿用舊行為（現場從 `existing_endpoints` 斷詞計算）。
    """
    if not user_response:
        return False
    stripped = user_response.strip()
    if not stripped:
        return False
    if len(stripped) <= CONVERGENCE_SHORT_RESPONSE_LEN:
        return False

    tokens = {t for t in _tokenize_text(stripped, min_token_len=2) if t}
    if not tokens:
        return False

    if existing_tokens is not None:
        novel = tokens - existing_tokens
    else:
        novel = tokens - extract_endpoint_tokens(existing_endpoints)
    return len(novel) > 0


def extract_endpoint_tokens(
    endpoints: list[dict[str, Any]] | None,
) -> set[str]:
    """把既有端點（start_data）中的名詞 token 集中抽成一個 set。

    給 `has_effective_info()` 在多輪迴圈前預先呼叫一次，避免每輪重複斷詞。
    只看 start_data 的 goal / user_input / motivation / tofu_understanding
    / constraints 五個欄位——與 `has_effective_info` 的比對面一致。
    """
    tokens: set[str] = set()
    for row in endpoints or []:
        sd = row.get("start_data") if isinstance(row, dict) else None
        if not isinstance(sd, dict):
            continue
        for field in ("goal", "user_input", "motivation", "tofu_understanding"):
            text = sd.get(field) or ""
            if text:
                tokens.update(
                    t for t in _tokenize_text(str(text), min_token_len=2) if t
                )
        for c in sd.get("constraints") or []:
            tokens.update(
                t for t in _tokenize_text(str(c), min_token_len=2) if t
            )
    return tokens


TOPIC_CATEGORIES = [
    "event",     # 辦活動、派對、會議
    "work",      # 工作任務、專案、報告
    "personal",  # 個人決定、生活選擇
    "creative",  # 創作、設計、寫作
    "learning",  # 學習、研究、探索
    "other",
]

# 每個主題的關鍵詞（小寫）——中英混合。
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "event": [
        "活動", "派對", "慶生", "婚禮", "聚會", "會議", "尾牙", "春酒",
        "典禮", "開幕", "發表會",
        "party", "event", "meeting", "ceremony", "gathering",
    ],
    "work": [
        "工作", "報告", "專案", "任務", "提案", "簡報", "提交", "交付",
        "績效", "kpi", "會議紀錄",
        "work", "project", "task", "report", "presentation",
        "deadline",
    ],
    "personal": [
        "決定", "選擇", "要不要", "該不該", "感情", "關係", "搬家",
        "轉職", "買", "賣", "家人", "健康",
        "personal", "life", "should i", "decide",
    ],
    "creative": [
        "寫作", "創作", "設計", "畫", "作品", "小說", "詩", "攝影",
        "音樂", "作曲", "劇本",
        "writing", "design", "art", "creative", "music",
    ],
    "learning": [
        "學", "學習", "研究", "探索", "讀", "課程", "教", "練習",
        "learn", "study", "research", "explore", "course",
    ],
}


def infer_topic(goal: str, gap_categories: list[str] | None = None) -> str:
    """從 goal + gap_categories 推斷主題分類。純程式碼，不呼叫 LLM。

    依序掃 _TOPIC_KEYWORDS，回傳命中最多的主題；都沒命中時回 "other"。
    """
    if not goal:
        return "other"
    lower = goal.lower()

    scores: dict[str, int] = {}
    for topic, kws in _TOPIC_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in lower)
        if hit:
            scores[topic] = hit

    # gap_categories 也用來給 topic 加權——例如出現 venue/headcount
    # 比較可能是 event；出現 deliverable/stakeholder 比較可能是 work。
    gap_set = set((gap_categories or []))
    if gap_set & {"venue", "headcount"}:
        scores["event"] = scores.get("event", 0) + 1
    if gap_set & {"deliverable", "stakeholder"}:
        scores["work"] = scores.get("work", 0) + 1

    if not scores:
        return "other"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def classify_zone(text: str) -> str:
    """把一段文字粗略分類成 Zone A / B / C。

    - Zone A：事實陳述（有具體數字、日期、可查證資訊）
    - Zone B：推測（含不確定詞「可能」「通常」「根據經驗」）
    - Zone C：立場（含價值詞「建議」「我認為」「你可以考慮」）

    回傳："A" / "B" / "C" / "unknown"（無法判斷）

    優先順序：C > B > A。立場句常同時含建議與推測語氣，
    立場先蓋掉推測；沒有明顯立場與推測語氣的才判事實。

    v2.0 P0-1：Zone B/C 關鍵詞比對改用斷詞後 token 比對，
    避免「可能性」被子字串「可能」誤判為 Zone B。
    Zone A 正則不改（數字/日期偵測，正則比斷詞更準確）。
    """
    if not text:
        return "unknown"

    lowered = text.lower()

    # Zone C 和 Zone B 使用斷詞後 token 比對
    tokens = set(_tokenize_text(text))

    # 多詞關鍵詞（含空格或多字組合）仍用子字串比對
    _ZONE_C_MULTI = [
        "你可以考慮", "依我看", "應該要", "in my opinion",
        "you should", "you can consider", "i think",
    ]
    _ZONE_C_SINGLE = {"不妨", "最好", "值得", "recommend", "suggest"}
    # 需要子字串比對因為斷詞（尤其 jieba）會把它們黏進更大的 token：
    # - 「我認為」「我覺得」「我想」：被黏成一個 token
    # - 「建議」「不建議」：會被黏成「我建議」「不建議你」等 token（v0.6 PR-H
    #   audit #1 修正；純 token 比對會漏判「我建議先…」類句子）
    # 邊緣 false positive（「會議建議案」判 C）可接受——這些句子整體仍偏立場。
    _ZONE_C_SUBSTR = ["我認為", "我覺得", "我想", "建議", "不建議"]

    if any(kw in lowered for kw in _ZONE_C_MULTI):
        return "C"
    if any(kw in lowered for kw in _ZONE_C_SUBSTR):
        return "C"
    if tokens & _ZONE_C_SINGLE:
        return "C"

    _ZONE_B_MULTI = [
        "根據經驗", "could be",
    ]
    # 單字關鍵詞「猜」無法通過 len>=2 的 token 過濾，改用子字串比對。
    _ZONE_B_SUBSTR = ["猜"]
    _ZONE_B_SINGLE = {
        "可能", "可能性", "或許", "也許", "大概", "應該", "通常", "一般",
        "據說", "聽說", "推測", "推估", "估計",
        "maybe", "probably", "perhaps", "might", "usually",
    }

    if any(kw in lowered for kw in _ZONE_B_MULTI):
        return "B"
    if any(kw in text for kw in _ZONE_B_SUBSTR):
        return "B"
    if tokens & _ZONE_B_SINGLE:
        return "B"

    for pat in _ZONE_A_PATTERNS:
        if re.search(pat, text):
            return "A"

    return "unknown"


# ----------------------------------------------------------------------
# v2.0 P1-1：ATL-3 下一步具體性檢測
# ----------------------------------------------------------------------
_VAGUE_ACTION_BLACKLIST = [
    "進一步研究", "持續觀察", "可以考慮", "建議嘗試",
    "深入了解", "多方評估", "繼續關注", "適時調整",
]

# 時間窗指標：數字 + 時間單位
_TIME_WINDOW_RE = re.compile(
    r"\d+\s*(?:小時|天|日|週|周|月|年|分鐘|hours?|days?|weeks?|months?|minutes?)"
)
# 產出物指標：做出/產出/完成 + 具體名詞
_DELIVERABLE_PATTERNS = [
    r"產出.{1,10}(?:份|篇|個|頁|張|套|組)",
    r"完成.{1,10}(?:份|篇|個|頁|張|套|組)",
    r"做出.{1,10}",
    r"列出.{1,10}",
    r"寫.{1,10}(?:份|篇|頁)",
    r"準備.{1,10}(?:份|個|套)",
]
# 驗收條件指標：「包含」「至少」「可查證」等
_VERIFICATION_PATTERNS = [
    r"包含.{1,15}(?:來源|數據|資料|引用)",
    r"至少\s*\d+",
    r"可查證",
    r"可驗證",
]


def check_action_specificity(text: str) -> dict:
    """ATL-3：檢查 AI 輸出的下一步建議是否具體。

    三要素：產出物、時間窗、驗收條件。三個裡面至少兩個才算具體。
    同時偵測空泛型黑名單。
    """
    if not text:
        return {
            "is_specific": False,
            "has_deliverable": False,
            "has_time_window": False,
            "has_verification": False,
            "vague_phrases": [],
        }

    has_deliverable = any(
        re.search(pat, text) for pat in _DELIVERABLE_PATTERNS
    )
    has_time_window = bool(_TIME_WINDOW_RE.search(text))
    has_verification = any(
        re.search(pat, text) for pat in _VERIFICATION_PATTERNS
    )

    specificity_score = sum([has_deliverable, has_time_window, has_verification])

    vague_phrases = [p for p in _VAGUE_ACTION_BLACKLIST if p in text]

    is_specific = specificity_score >= 2 and not vague_phrases

    return {
        "is_specific": is_specific,
        "has_deliverable": has_deliverable,
        "has_time_window": has_time_window,
        "has_verification": has_verification,
        "vague_phrases": vague_phrases,
        "specificity_score": specificity_score,
    }


# ----------------------------------------------------------------------
# v2.0 P1-2：ATL-1 可反駁條件可操作性檢測
# ----------------------------------------------------------------------
_GENERIC_FALSIFICATION = [
    "若有新證據", "若事實不符", "若情況改變",
    "若後續研究", "若實際情況", "若條件不同",
]

_FALSIFICATION_LEAD_INS = [
    "如果", "若", "除非", "前提是", "不適用於",
    "if ", "unless ", "provided that",
]

_SPECIFIC_NUMBER_RE = re.compile(r"\d+")


def check_falsification(text: str) -> dict:
    """ATL-1：檢查 AI 輸出是否含有可操作的反駁條件。

    萬用句型（「若有新證據」）存在且沒有具體內容 = 不合規。
    """
    if not text:
        return {
            "has_falsification": False,
            "is_generic": False,
            "generic_phrases": [],
        }

    # 偵測萬用句型
    generic_phrases = [p for p in _GENERIC_FALSIFICATION if p in text]
    is_generic = len(generic_phrases) > 0

    # 偵測是否有任何反駁條件引導詞
    has_lead_in = any(lead in text.lower() for lead in _FALSIFICATION_LEAD_INS)

    # 如果有引導詞，檢查附近是否有具體數字或時間
    has_specific_content = False
    if has_lead_in:
        for lead in _FALSIFICATION_LEAD_INS:
            idx = text.lower().find(lead)
            if idx >= 0:
                # 檢查引導詞後 50 字內是否有數字
                window = text[idx: idx + 50]
                if _SPECIFIC_NUMBER_RE.search(window):
                    has_specific_content = True
                    break
                if _TIME_WINDOW_RE.search(window):
                    has_specific_content = True
                    break

    has_falsification = has_lead_in
    # 如果有萬用句型但沒有具體內容，標記為 generic
    if is_generic and not has_specific_content:
        is_generic = True
    elif is_generic and has_specific_content:
        is_generic = False

    return {
        "has_falsification": has_falsification,
        "is_generic": is_generic,
        "generic_phrases": generic_phrases,
        "has_specific_content": has_specific_content,
    }


# ----------------------------------------------------------------------
# v2.0 P1-3：ATL-2 來源可回溯性檢測
# ----------------------------------------------------------------------
_VAGUE_SOURCE_PHRASES = [
    "根據研究", "有數據", "有資料", "眾所周知",
    "一般認為", "普遍認為", "據說", "有報導",
]

_SPECIFIC_SOURCE_RE = re.compile(
    r"(?:"
    r"\d{4}\s*年"              # 年份
    r"|第\s*[\d一二三四五六七八九十]+\s*章"  # 章節引用
    r"|vol\.\s*\d+"            # 期刊卷號
    r"|https?://"              # URL
    r"|p\.\s*\d+"              # 頁碼
    r"|第\s*\d+\s*頁"          # 頁碼（中文）
    r")",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# v4.0 P0-1：ATL-3 前驗證閘門
# ----------------------------------------------------------------------
# 依 20260418_逗福Tofu_三機制落地_開發規格_v4_0.md。
# 把既有的 `check_action_specificity` 從「事後打分」升級為「事前閘門」。
# 本身只做判定；重試迴圈由 caller 負責（見 main.py 的
# `_execute_task_with_atl3_gate`）。
# ----------------------------------------------------------------------
def atl3_gate(result: str, mode: str = "default") -> dict:
    """ATL-3 前驗證閘門。

    失敗條件（任一命中即不通過）：
    - specificity_score < 2
    - has_deliverable == False
    - has_time_window == False
    - mode == 'propose' / 'propose_final' 時，has_verification == False
    - 含有迴避類空泛句型（vague_phrases）

    回傳：
    {
        "passed": bool,
        "failure_reasons": list[str],
        "action_check": dict,  # 底層 check_action_specificity 結果
    }
    """
    action_check = check_action_specificity(result)
    reasons: list[str] = []

    score = action_check.get("specificity_score", 0)
    if score < 2:
        reasons.append(f"具體性分數不足（{score}/3）")
    if not action_check.get("has_deliverable", False):
        reasons.append("缺產出物")
    if not action_check.get("has_time_window", False):
        reasons.append("缺時間窗")
    if mode in ("propose", "propose_final") and not action_check.get(
        "has_verification", False
    ):
        reasons.append("propose 模式缺驗收條件")
    vague = action_check.get("vague_phrases") or []
    if vague:
        # Copilot review r3105635897：把 list 轉成頓號分隔文字，避免 LLM
        # 看到 "['進一步研究']" 這種 Python list repr、訊號不乾淨。
        vague_text = "、".join(str(v) for v in vague)
        reasons.append(f"出現迴避句型：{vague_text}")

    return {
        "passed": len(reasons) == 0,
        "failure_reasons": reasons,
        "action_check": action_check,
    }


def check_source_traceability(text: str, zone: str) -> dict:
    """ATL-2：檢查 Zone A 內容是否有可回溯的來源。

    只對 Zone A 生效。Zone B/C 直接跳過。
    """
    if zone != "A":
        return {
            "needs_check": False,
            "should_downgrade": False,
            "vague_sources": [],
            "has_specific_source": False,
        }

    if not text:
        return {
            "needs_check": True,
            "should_downgrade": True,
            "vague_sources": [],
            "has_specific_source": False,
        }

    vague_sources = [p for p in _VAGUE_SOURCE_PHRASES if p in text]
    has_specific_source = bool(_SPECIFIC_SOURCE_RE.search(text))

    # Zone A 但沒有具體來源 → 建議降級為 Zone B
    should_downgrade = not has_specific_source

    return {
        "needs_check": True,
        "should_downgrade": should_downgrade,
        "vague_sources": vague_sources,
        "has_specific_source": has_specific_source,
    }
