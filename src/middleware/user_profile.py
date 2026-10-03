"""使用者畫像引擎。

v3.0 核心模組。畫像的作用是搜索策略觸發器——根據使用者的表達習慣，
決定逗福用什麼方式找偏好、給多少資訊、回覆多長。

三層資料來源（覆蓋優先級：真人校準 > 互動觀測 > 初始推定）：
1. 初始推定層：四柱映射（Zone B）
2. 互動觀測層：對話行為痕跡（Zone A）
3. 真人校準層：使用者直接確認或修正（最高優先級）

儲存於獨立的 user_profile.json，與 endpoints.json 分開。
"""

from __future__ import annotations

from src.middleware.preferences import merge_preferences

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from src.utils.tokenizer import tokenize as _tokenize
from src.utils.tokenizer import tokenize_nouns as _tokenize_nouns


# ----------------------------------------------------------------------
# P0-1：資料結構與預設值
# ----------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_profile() -> dict[str, Any]:
    """回傳一份空白的使用者畫像。"""
    return {
        "version": "standard",

        # 四柱資料（有多少記多少）
        "pillars": {
            "year":  {"stem": None, "branch": None},
            "month": {"stem": None, "branch": None},
            "day":   {"stem": None, "branch": None},
            "hour":  {"stem": None, "branch": None},
            "source": None,
            "completeness": "none",
        },

        # 維度 1：溝通風格
        "communication_style": {
            "receiving_preference": "conversational",
            "expression_density": "medium",
            "preference_expression": "implicit",
            "sample_count": 0,
            "confidence": "低",
            "data_source": "default",
            "zone": "B",
            "falsification_condition": None,
            "calibration_timestamp": None,
            "calibration_value": None,
            "interactions_since_calibration": 0,
        },

        # 維度 2：決策模式
        "decision_style": {
            "pace": "deliberate",
            "info_appetite": "moderate",
            "context_need": "what_first",
            "sample_count": 0,
            "confidence": "低",
            "data_source": "default",
            "zone": "B",
            "falsification_condition": None,
            "calibration_timestamp": None,
            "calibration_value": None,
            "interactions_since_calibration": 0,
        },

        # 維度 3：興趣領域與偏好清單
        "interest_map": {
            "domains": [],
            "preferences": [],
            "sample_count": 0,
            "confidence": "低",
        },

        # 維度 4：隱性滿意度
        "satisfaction_pattern": {
            "explicit_satisfied": 0,
            "explicit_unsatisfied": 0,
            "implicit_satisfied": 0,
            "implicit_unsatisfied": 0,
            "unknown": 0,
            "effective_satisfaction_rate": 0.0,
        },

        # 真人校準記錄
        "human_calibrations": [],

        # RBH 回收記錄
        "rbh_log": [],

        # v4.0 P1：情緒三軸基線（滾動 10 筆樣本）
        "emotion_baseline": {
            "typical_seven_emotion": None,
            "typical_mood": None,
            "last_emotion_sample": [],
            "stable_since": None,
        },

        "meta": {
            "total_interactions": 0,
            "profile_created": None,
            "last_updated": None,
            "maturity": "cold_start",
        },
    }


# ----------------------------------------------------------------------
# ProfileStore：讀寫、更新
# ----------------------------------------------------------------------
class ProfileStore:
    """使用者畫像的讀寫與更新。"""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return _default_profile()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return _default_profile()
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _default_profile()

    def save(self, profile: dict[str, Any]) -> None:
        if isinstance(profile.get("interest_map"), dict):
            im = profile["interest_map"]
            im["preferences"] = merge_preferences(im.get("preferences", []), [])
        profile["meta"]["last_updated"] = _now_iso()
        if not profile["meta"].get("profile_created"):
            profile["meta"]["profile_created"] = _now_iso()

        tmp_path = self.path + ".tmp"
        payload = json.dumps(profile, ensure_ascii=False, indent=2)
        json.loads(payload)  # 寫前驗證

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            try:
                f.flush()
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        os.replace(tmp_path, self.path)

    def update_from_interaction(
        self,
        start_data: dict[str, Any],
        end_data: dict[str, Any],
        user_response: str,
        next_interaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """互動結束後更新畫像。"""
        profile = self.load()
        meta = profile["meta"]
        meta["total_interactions"] = meta.get("total_interactions", 0) + 1
        meta["maturity"] = _compute_maturity(meta["total_interactions"])

        # 本次互動資料（用於分類函式）
        interaction_records = [{"user_input": start_data.get("user_input", ""),
                               "user_confirmation": start_data.get("user_confirmation", "confirmed"),
                               "gap_questions": start_data.get("gap_questions", []),
                               "_answered_categories": start_data.get("_answered_categories", [])}]

        # 通知列表（stale override 時填入）
        self._pending_notifications: list[str] = []

        # 更新溝通風格
        cs = profile["communication_style"]
        self._update_dimension(
            profile, cs, "communication_style", interaction_records,
            classifiers={
                "preference_expression": classify_preference_expression,
                "receiving_preference": classify_receiving_preference,
                "context_need": classify_context_need,
            },
        )

        # 更新決策模式
        ds = profile["decision_style"]
        self._update_dimension(
            profile, ds, "decision_style", interaction_records,
            classifiers={
                "pace": classify_pace,
                "info_appetite": classify_info_appetite,
            },
        )

        # 更新隱性滿意度
        sat_label = end_data.get("user_satisfaction", "no_feedback")
        sp = profile["satisfaction_pattern"]
        if sat_label == "satisfied":
            sp["explicit_satisfied"] = sp.get("explicit_satisfied", 0) + 1
        elif sat_label == "unsatisfied":
            sp["explicit_unsatisfied"] = sp.get("explicit_unsatisfied", 0) + 1
        else:
            # 嘗試推斷隱性滿意度
            implicit = infer_satisfaction(end_data, next_interaction)
            if implicit == "implicit_satisfied":
                sp["implicit_satisfied"] = sp.get("implicit_satisfied", 0) + 1
            elif implicit == "implicit_unsatisfied":
                sp["implicit_unsatisfied"] = sp.get("implicit_unsatisfied", 0) + 1
            else:
                sp["unknown"] = sp.get("unknown", 0) + 1

        # 更新有效滿意度
        total_sat = (
            sp.get("explicit_satisfied", 0) + sp.get("implicit_satisfied", 0)
        )
        total_unsat = (
            sp.get("explicit_unsatisfied", 0) + sp.get("implicit_unsatisfied", 0)
        )
        total_known = total_sat + total_unsat
        sp["effective_satisfaction_rate"] = (
            total_sat / total_known if total_known > 0 else 0.0
        )

        # 提取偏好
        user_input = start_data.get("user_input", "")
        if user_input:
            new_prefs = extract_preferences([{"user_input": user_input}])
            profile["interest_map"]["preferences"] = merge_preferences(
                profile["interest_map"].get("preferences", []), new_prefs,
            )

            # 提取興趣領域
            new_domains = extract_domains([{"user_input": user_input}])
            domain_map = {
                d["name"]: d
                for d in profile["interest_map"].get("domains", [])
            }
            for d in new_domains:
                if d["name"] in domain_map:
                    domain_map[d["name"]]["mention_count"] += d["mention_count"]
                else:
                    domain_map[d["name"]] = d
            profile["interest_map"]["domains"] = list(domain_map.values())
            profile["interest_map"]["sample_count"] = meta["total_interactions"]

        # v4.0 P1：更新情緒基線——從 start_data.emotion_state 滾動追加
        emotion_state = start_data.get("emotion_state")
        if isinstance(emotion_state, dict):
            from src.middleware.cooling_mode import append_emotion_sample
            eb = profile.setdefault("emotion_baseline", {
                "typical_seven_emotion": None,
                "typical_mood": None,
                "last_emotion_sample": [],
                "stable_since": None,
            })
            append_emotion_sample(eb, emotion_state, max_samples=10)
            # 從滾動樣本統計常態值
            samples = eb.get("last_emotion_sample", []) or []
            if samples:
                from collections import Counter
                seven_counter = Counter(
                    s.get("seven_emotion") for s in samples if isinstance(s, dict)
                )
                mood_counter = Counter(
                    s.get("mood_axis") for s in samples if isinstance(s, dict)
                )
                eb["typical_seven_emotion"] = seven_counter.most_common(1)[0][0]
                eb["typical_mood"] = mood_counter.most_common(1)[0][0]
                if not eb.get("stable_since"):
                    eb["stable_since"] = _now_iso()

        self.save(profile)
        return profile

    # ------------------------------------------------------------------
    # 內部：維度更新（處理 staleness 與 stale override）
    # ------------------------------------------------------------------
    def _check_and_apply_staleness(
        self, profile: dict[str, Any], dim: dict[str, Any], section_name: str,
    ) -> None:
        """規則 6：human_calibrated → human_calibrated_stale。

        過期條件（任一成立）：
        A. 距離 calibration_timestamp 超過 90 天。
        B. 距離校準後已累積超過 50 筆新互動。
        """
        if dim.get("data_source") != "human_calibrated":
            return

        ts_str = dim.get("calibration_timestamp")
        interactions_since = dim.get("interactions_since_calibration", 0)

        stale = False

        # 條件 A：90 天
        if ts_str:
            try:
                cal_time = datetime.fromisoformat(ts_str)
                now = datetime.now(timezone.utc)
                if (now - cal_time).days > 90:
                    stale = True
            except (ValueError, TypeError):
                pass

        # 條件 B：50 筆互動
        if interactions_since > 50:
            stale = True

        if stale:
            prev_source = dim["data_source"]
            dim["data_source"] = "human_calibrated_stale"
            profile.setdefault("rbh_log", []).append({
                "dimension": section_name,
                "prediction": dim.get("calibration_value", ""),
                "observed": "",
                "match": True,
                "conflict_count": 0,
                "action": "stale_expired",
                "previous_data_source": prev_source,
                "new_data_source": "human_calibrated_stale",
                "possible_reason": "calibration expired (90 days or 50 interactions)",
                "timestamp": _now_iso(),
            })

    def _update_dimension(
        self,
        profile: dict[str, Any],
        dim: dict[str, Any],
        section_name: str,
        interaction_records: list[dict],
        classifiers: dict[str, Any],
    ) -> None:
        """更新單一維度，處理所有 data_source 狀態轉移。"""
        source = dim.get("data_source", "default")

        # 追蹤 interactions_since_calibration
        if source in ("human_calibrated", "human_calibrated_stale"):
            dim["interactions_since_calibration"] = (
                dim.get("interactions_since_calibration", 0) + 1
            )

        # 規則 6：檢查 human_calibrated 是否過期
        self._check_and_apply_staleness(profile, dim, section_name)
        source = dim.get("data_source", "default")  # 可能已變為 stale

        if source == "human_calibrated":
            # 真人校準中，不更新
            return

        if source == "human_calibrated_stale":
            # 規則 7：過期後觀測重新生效，但需衝突 >= 5 才覆蓋
            dim["sample_count"] = dim.get("sample_count", 0) + 1
            dim["confidence"] = _compute_confidence(dim["sample_count"])

            calibration_value = dim.get("calibration_value")
            if calibration_value is None:
                return

            # 計算觀測值——取第一個 classifier 的主值欄位作比對
            for field, classifier_fn in classifiers.items():
                observed_val = classifier_fn(interaction_records)
                if observed_val != calibration_value:
                    # 找 rbh_log 中這個維度的 stale 衝突計數
                    rbh_log = profile.get("rbh_log", [])
                    stale_conflict_count = sum(
                        1 for e in rbh_log
                        if e.get("dimension") == f"{section_name}.{field}"
                        and not e.get("match", True)
                        and e.get("previous_data_source") == "human_calibrated_stale"
                    ) + 1

                    if stale_conflict_count >= 5:
                        # 規則 7：觀測覆蓋過期校準
                        old_val = dim.get(field)
                        dim[field] = observed_val
                        dim["data_source"] = "observed"
                        dim["zone"] = "A"
                        profile.setdefault("rbh_log", []).append({
                            "dimension": f"{section_name}.{field}",
                            "prediction": calibration_value,
                            "observed": observed_val,
                            "match": False,
                            "conflict_count": stale_conflict_count,
                            "action": "stale_override",
                            "previous_data_source": "human_calibrated_stale",
                            "new_data_source": "observed",
                            "possible_reason": "",
                            "timestamp": _now_iso(),
                        })
                        self._pending_notifications.append(
                            f"你之前說偏好 {calibration_value}，"
                            f"但最近的互動都偏向 {observed_val}。"
                            f"逗福已更新你的畫像。如果這不對，用 /profile 再修正。"
                        )
                        # 一旦覆蓋，跳出——其他欄位用正常 observed 邏輯
                        break
                    else:
                        # 記錄衝突但不覆蓋
                        profile.setdefault("rbh_log", []).append({
                            "dimension": f"{section_name}.{field}",
                            "prediction": calibration_value,
                            "observed": observed_val,
                            "match": False,
                            "conflict_count": stale_conflict_count,
                            "action": "conflict_recorded",
                            "previous_data_source": "human_calibrated_stale",
                            "new_data_source": "human_calibrated_stale",
                            "possible_reason": "",
                            "timestamp": _now_iso(),
                        })
            # 如果已經被覆蓋為 observed，後續交給正常 observed 邏輯
            if dim.get("data_source") == "human_calibrated_stale":
                return

        # 正常更新（default / observed / pillars_mapped / pillars_validated）
        dim["sample_count"] = dim.get("sample_count", 0) + 1
        dim["confidence"] = _compute_confidence(dim["sample_count"])
        if source in ("default", None):
            dim["data_source"] = "observed"
            dim["zone"] = "A"

        for field, classifier_fn in classifiers.items():
            dim[field] = classifier_fn(interaction_records)

    def get_pending_notifications(self) -> list[str]:
        """取得待通知訊息（stale override 時產生）。"""
        return getattr(self, "_pending_notifications", [])

    def apply_human_calibration(
        self, dimension: str, value: str,
    ) -> dict[str, Any]:
        """套用真人校準。dimension 格式："communication_style.receiving_preference"。

        規則 5：任何狀態 → human_calibrated。
        規則 8：human_calibrated_stale → human_calibrated（使用者再次校準）。
        """
        profile = self.load()

        parts = dimension.split(".")
        if len(parts) != 2:
            return profile

        section, field = parts
        if section not in profile or not isinstance(profile[section], dict):
            return profile

        old_value = profile[section].get(field)
        old_source = profile[section].get("data_source", "default")

        profile[section][field] = value
        profile[section]["data_source"] = "human_calibrated"
        profile[section]["zone"] = "A"
        profile[section]["calibration_timestamp"] = _now_iso()
        profile[section]["calibration_value"] = value
        profile[section]["interactions_since_calibration"] = 0

        profile["human_calibrations"].append({
            "dimension": dimension,
            "ai_judgment": old_value,
            "human_correction": value,
            "timestamp": _now_iso(),
        })

        # rbh_log：記錄狀態轉移
        action = "human_override"
        if old_source == "human_calibrated_stale":
            if old_value == value:
                action = "stale_renewed"
            else:
                action = "human_override"

        profile.setdefault("rbh_log", []).append({
            "dimension": dimension,
            "prediction": old_value,
            "observed": value,
            "match": old_value == value,
            "conflict_count": 0,
            "action": action,
            "previous_data_source": old_source,
            "new_data_source": "human_calibrated",
            "possible_reason": "",
            "timestamp": _now_iso(),
        })

        self.save(profile)
        return profile

    def apply_pillars_mapping(
        self, pillars: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """套用四柱映射。回傳 (profile, message)。

        規則 2（default → pillars_mapped）：
        - 如果已有觀測值（observed），映射不覆蓋——只寫入 rbh_log 供參考。
        - 如果已有真人校準（human_calibrated / human_calibrated_stale），也不覆蓋。
        - 只有 default 和 pillars_mapped 狀態下才寫入映射結果。
        """
        from src.utils.pillars_map import map_pillars_to_profile

        profile = self.load()
        mapped = map_pillars_to_profile(pillars)
        if not mapped:
            self.save(profile)
            return profile, "四柱資料已記錄。日主天干未提供，暫無法產出映射。"

        non_overwrite_sources = (
            "observed", "human_calibrated", "human_calibrated_stale",
            "pillars_validated", "imported",
        )

        for section, values in mapped.items():
            if section not in profile or not isinstance(profile[section], dict):
                continue
            current_source = profile[section].get("data_source", "default")

            if current_source in non_overwrite_sources:
                # 映射不覆蓋觀測 / 校準值，只記 rbh_log
                for field in ("preference_expression", "receiving_preference", "pace"):
                    if field in values and field in profile[section]:
                        rbh_entry = {
                            "dimension": f"{section}.{field}",
                            "prediction": values[field],
                            "observed": profile[section][field],
                            "match": values[field] == profile[section][field],
                            "conflict_count": 0,
                            "action": "consistent" if values[field] == profile[section][field] else "conflict_recorded",
                            "previous_data_source": current_source,
                            "new_data_source": current_source,
                            "possible_reason": f"pillars_mapped 未覆蓋 {current_source}",
                            "timestamp": _now_iso(),
                        }
                        profile.setdefault("rbh_log", []).append(rbh_entry)
            else:
                # default 或 pillars_mapped → 可以覆蓋
                profile[section].update(values)

        self.save(profile)
        return profile, "已根據四柱映射預測溝通風格和決策模式（Zone B，可反駁假設）。"

    def import_external_profile(
        self,
        parsed: dict[str, Any],
        data_source: str = "imported",
    ) -> dict[str, Any]:
        """從外部 AI 匯入的結構化畫像更新本地畫像。

        匯入的優先級：
        - 高於 default
        - 等同於 pillars_mapped（互不覆蓋）
        - 低於 observed / human_calibrated / human_calibrated_stale / pillars_validated

        parsed 預期結構：
            {
                "communication_style": {
                    "preference_expression": "explicit|implicit|exclusion",
                    "receiving_preference": "structured|conversational|minimal",
                    ...
                },
                "decision_style": {
                    "pace": "fast|deliberate|avoidant",
                    "info_appetite": "minimal|moderate|exhaustive",
                    ...
                },
                "interest_map": {
                    "preferences": [{"item": str, "type": str}, ...],
                    "domains": [{"name": str, ...}, ...],
                },
            }
        """
        profile = self.load()

        non_overwrite_sources = (
            "observed", "human_calibrated", "human_calibrated_stale",
            "pillars_validated",
        )

        # 更新 communication_style 與 decision_style
        for section in ("communication_style", "decision_style"):
            section_data = parsed.get(section)
            if not isinstance(section_data, dict):
                continue
            if section not in profile or not isinstance(profile[section], dict):
                continue

            current_source = profile[section].get("data_source", "default")
            if current_source in non_overwrite_sources:
                # 不覆蓋優先級較高或同級的觀測 / 校準值
                continue

            for field, value in section_data.items():
                if field in profile[section] and isinstance(value, str) and value:
                    profile[section][field] = value

            profile[section]["data_source"] = data_source
            profile[section]["zone"] = "B"

        # 合併 interest_map（去重）
        im_in = parsed.get("interest_map") or {}
        if isinstance(im_in, dict):
            existing_items = {
                p.get("item")
                for p in profile["interest_map"].get("preferences", [])
                if isinstance(p, dict)
            }
            for p in im_in.get("preferences", []) or []:
                if not isinstance(p, dict):
                    continue
                item = p.get("item")
                if not item or item in existing_items:
                    continue
                profile["interest_map"]["preferences"].append({
                    "item": item,
                    "type": p.get("type", "implicit"),
                    "context": p.get("context", ""),
                    "session_ref": p.get("session_ref", "imported"),
                })
                existing_items.add(item)

            domain_map = {
                d.get("name"): d
                for d in profile["interest_map"].get("domains", [])
                if isinstance(d, dict)
            }
            for d in im_in.get("domains", []) or []:
                if not isinstance(d, dict):
                    continue
                name = d.get("name")
                if not name:
                    continue
                if name in domain_map:
                    domain_map[name]["mention_count"] = (
                        domain_map[name].get("mention_count", 0)
                        + d.get("mention_count", 1)
                    )
                else:
                    domain_map[name] = {
                        "name": name,
                        "mention_count": d.get("mention_count", 1),
                        "depth": d.get("depth", "surface"),
                    }
            profile["interest_map"]["domains"] = list(domain_map.values())

        self.save(profile)
        return profile

    def is_mature(self) -> bool:
        profile = self.load()
        return profile["meta"].get("total_interactions", 0) > 20

    def export_rbh_log(self) -> list[dict]:
        profile = self.load()
        return profile.get("rbh_log", [])


# ----------------------------------------------------------------------
# P0-2：偏好表達方式分類（搜索策略的核心開關）
# ----------------------------------------------------------------------
_EXPLICIT_INDICATORS = [
    # 中文
    "我喜歡", "我想要", "我偏好", "我愛", "我欣賞",
    # 英文
    "I love", "I like", "I prefer", "I enjoy", "I really like",
    "my favorite", "my favourite",
]
_IMPLICIT_INDICATORS = [
    # 中文
    "上次那個", "之前用的", "我買了", "那個不錯",
    "我常去", "我習慣", "之前試過",
    # 英文
    "I bought", "I just got", "I just watched", "I just started",
    "I've been", "last time", "that was great", "that was nice",
    "really into",
]
_EXCLUSION_INDICATORS = [
    # 中文
    "不要", "不喜歡", "別給我", "避免", "討厭", "不想",
    # 英文
    "not my thing", "I don't like", "I hate", "I avoid",
    "don't want", "not a fan", "were not my thing",
]

# 中文指標詞的邊界防護。中文沒有空白詞界，純子字串比對會把
# 「媽問我『要不要』換車」「這件事『不要緊』」誤判成排除偏好——
# 而「要不要」幾乎都出現在別人問使用者的句子裡，切出來的是被詢問的
# 事項，不是使用者的偏好（v0.8 實測 50 輪互動的 4 條偏好中 2 條由此而來）。
# 用負向斷言排除這些常見的黏著情境；沒列在這裡的指標詞維持原樣。
_GUARDED_INDICATOR_PATTERNS: dict[str, str] = {
    "不要": r"(?<!要)不要(?!緊|臉)",   # 排除「要不要」「不要緊」「不要臉」
    "不想": r"(?<!想)不想",            # 排除「想不想」
    "不喜歡": r"(?<!喜)不喜歡",        # 排除「喜不喜歡」
}


def _compile_indicator_patterns(
    indicators: list[str],
) -> list[tuple[str, re.Pattern[str]]]:
    """預編譯 indicator 為大小寫不敏感的 regex pattern。

    使用 regex 而非 text.lower() + substring 比對,是為了:
    1. 避免每輪迴圈重複分配 .lower() 字串。
    2. 在 extract_preferences 中透過 match.start()/match.end() 直接拿到
       原始 text 的 match span,避免 Unicode case-folding 改變碼位長度
       (例:Turkish dotted 大寫 I)造成切片索引錯位。

    列在 _GUARDED_INDICATOR_PATTERNS 的指標詞改用帶邊界防護的 regex。
    """
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for kw in indicators:
        guarded = _GUARDED_INDICATOR_PATTERNS.get(kw)
        pattern = guarded if guarded is not None else re.escape(kw)
        compiled.append((kw, re.compile(pattern, re.IGNORECASE)))
    return compiled


_EXPLICIT_PATTERNS = _compile_indicator_patterns(_EXPLICIT_INDICATORS)
_IMPLICIT_PATTERNS = _compile_indicator_patterns(_IMPLICIT_INDICATORS)
_EXCLUSION_PATTERNS = _compile_indicator_patterns(_EXCLUSION_INDICATORS)


def classify_preference_expression(interactions: list[dict]) -> str:
    """分類使用者的偏好表達方式。

    掃描所有 user_input，統計三類關鍵詞出現頻率：
    取出現頻率最高者。都沒出現回 "implicit"（預設偏保守）。
    """
    explicit_count = 0
    implicit_count = 0
    exclusion_count = 0

    for item in interactions:
        text = item.get("user_input", "")
        if not text:
            continue
        # 使用預編譯的 regex pattern（case-insensitive）避免每輪重複分配。
        for _, pattern in _EXPLICIT_PATTERNS:
            if pattern.search(text):
                explicit_count += 1
        for _, pattern in _IMPLICIT_PATTERNS:
            if pattern.search(text):
                implicit_count += 1
        for _, pattern in _EXCLUSION_PATTERNS:
            if pattern.search(text):
                exclusion_count += 1

    if explicit_count == 0 and implicit_count == 0 and exclusion_count == 0:
        return "implicit"

    counts = {
        "explicit": explicit_count,
        "implicit": implicit_count,
        "exclusion": exclusion_count,
    }
    return max(counts, key=counts.get)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# P0-3：溝通接收偏好 + 決策模式分類
# ----------------------------------------------------------------------
def classify_receiving_preference(interactions: list[dict]) -> str:
    """structured / conversational / minimal"""
    if not interactions:
        return "conversational"

    total_len = 0
    count = 0
    for item in interactions:
        text = item.get("user_input", "")
        if text:
            total_len += len(text)
            count += 1

    if count == 0:
        return "conversational"

    avg_len = total_len / count
    if avg_len < 10:
        return "minimal"
    elif avg_len > 80:
        return "structured"
    return "conversational"


def classify_pace(interactions: list[dict]) -> str:
    """fast / deliberate / avoidant"""
    if not interactions:
        return "deliberate"

    confirmed = 0
    modified = 0
    total = 0

    for item in interactions:
        conf = item.get("user_confirmation", "")
        if not conf:
            continue
        total += 1
        if conf == "confirmed":
            confirmed += 1
        elif conf == "modified":
            modified += 1

    if total == 0:
        return "deliberate"

    confirmed_ratio = confirmed / total
    modified_ratio = modified / total

    if confirmed_ratio > 0.7:
        return "fast"
    if modified_ratio > 0.7:
        return "avoidant"
    return "deliberate"


def classify_info_appetite(interactions: list[dict]) -> str:
    """minimal / moderate / exhaustive"""
    if not interactions:
        return "moderate"

    total_gap_q = 0
    total_answered = 0

    for item in interactions:
        gaps = item.get("gap_questions", [])
        answered = item.get("_answered_categories", [])
        total_gap_q += len(gaps)
        total_answered += len(answered)

    if total_gap_q == 0:
        return "moderate"

    ratio = total_answered / total_gap_q
    if ratio < 0.3:
        return "minimal"
    if ratio > 0.7:
        return "exhaustive"
    return "moderate"


def classify_context_need(interactions: list[dict]) -> str:
    """why_first / what_first"""
    if not interactions:
        return "what_first"

    why_count = 0
    what_count = 0
    why_keywords = ["為什麼", "為何", "原因", "why", "reason"]
    what_keywords = ["怎麼做", "怎樣", "如何", "步驟", "how", "step"]

    for item in interactions:
        text = item.get("user_input", "")
        if not text:
            continue
        lower = text.lower()
        for kw in why_keywords:
            if kw in lower:
                why_count += 1
        for kw in what_keywords:
            if kw in lower:
                what_count += 1

    if why_count > what_count:
        return "why_first"
    return "what_first"


# ----------------------------------------------------------------------
# P0-4：興趣領域與偏好清單提取
# ----------------------------------------------------------------------
def extract_domains(interactions: list[dict]) -> list[dict]:
    """用 jieba 詞性標註提取興趣領域（只留名詞性詞），按 mention_count 排序。

    v0.8 修正：改用 tokenize_nouns。一般 tokenize 保留所有詞供檢索比對，
    但興趣領域不過濾詞性的話，「放在」「這樣」「突然」這類功能詞會佔滿
    榜單（實測 50 輪的前 25 名中虛詞佔八成），榜單就失去「這個人關心什麼」
    的意義。
    """
    from collections import Counter
    word_counter: Counter[str] = Counter()

    for item in interactions:
        text = item.get("user_input", "")
        if not text:
            continue
        tokens = _tokenize_nouns(text)
        word_counter.update(tokens)

    domains = []
    for word, count in word_counter.most_common(20):
        if count < 1:
            continue
        depth = "surface"
        if count >= 3:
            depth = "expert"
        elif count >= 2:
            depth = "engaged"
        domains.append({
            "name": word,
            "mention_count": count,
            "depth": depth,
        })
    return domains


def _extract_preference_item(text: str, max_len: int = 100) -> str:
    """從關鍵詞之後的文字提取偏好項目。

    不硬截字數。找句末標點或逗號作為切點，保留完整名詞。
    規則：
    1. 找第一個句末標點（. ! ? 。！？），如果在 max_len 內就用
    2. 找不到就找逗號或分號
    3. 都找不到就找最後一個空格（不切詞）
    4. 最後 fallback 才硬截
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text

    # 找句末標點
    for i, ch in enumerate(text):
        if ch in ".?!。？！" and i > 3:
            return text[:i].strip()

    # 找逗號/分號
    last_comma = -1
    for i, ch in enumerate(text):
        if ch in ",，;、" and i < max_len:
            last_comma = i
    if last_comma > 5:
        return text[:last_comma].strip()

    # 找空格（不切詞）
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.4:
        return cut[:last_space].strip()

    return cut.strip()


def extract_preferences(interactions: list[dict]) -> list[dict]:
    """三遍掃描：顯式、隱式、負面偏好。"""
    preferences: list[dict] = []

    for item in interactions:
        text = item.get("user_input", "")
        if not text:
            continue
        session_ref = item.get("session_ref", "")

        # 第一遍：顯式
        for _, pattern in _EXPLICIT_PATTERNS:
            match = pattern.search(text)
            if match:
                idx = match.start()
                kw_end = match.end()
                start = max(0, idx - 10)
                end = min(len(text), kw_end + 50)
                context = text[start:end]
                # 取關鍵詞後的內容作為偏好項目
                item_text = text[kw_end:].strip()
                if item_text:
                    item_text = _extract_preference_item(item_text)
                    preferences.append({
                        "item": item_text,
                        "type": "explicit",
                        "context": context,
                        "session_ref": session_ref,
                    })

        # 第二遍：隱式
        for _, pattern in _IMPLICIT_PATTERNS:
            match = pattern.search(text)
            if match:
                idx = match.start()
                kw_end = match.end()
                start = max(0, idx - 10)
                end = min(len(text), kw_end + 50)
                context = text[start:end]
                item_text = text[kw_end:].strip()
                if item_text:
                    item_text = _extract_preference_item(item_text)
                    preferences.append({
                        "item": item_text,
                        "type": "implicit",
                        "context": context,
                        "session_ref": session_ref,
                    })

        # 第三遍：負面（exclusion）
        for _, pattern in _EXCLUSION_PATTERNS:
            match = pattern.search(text)
            if match:
                idx = match.start()
                kw_end = match.end()
                start = max(0, idx - 10)
                end = min(len(text), kw_end + 50)
                context = text[start:end]
                # exclusion 的 item 從指標詞本身開始切（idx 而非 kw_end），
                # 讓否定詞留在 item 內：「避免踩到箱子」「不喜歡人多的場合」。
                # 只切右邊會存出「踩到箱子」這種語義相反的條目——下游雖有
                # [exclusion] 標記，但 item 文字必須自足，不能依賴每個
                # 消費端都記得看標記。
                item_text = text[idx:].strip()
                if len(item_text) > len(match.group(0)):
                    item_text = _extract_preference_item(item_text)
                    preferences.append({
                        "item": item_text,
                        "type": "exclusion",
                        "context": context,
                        "session_ref": session_ref,
                    })

    return preferences


# ----------------------------------------------------------------------
# P0-5：隱性滿意度
# ----------------------------------------------------------------------
def infer_satisfaction(
    current_end_data: dict,
    next_interaction: dict | None = None,
) -> str:
    """推斷隱性滿意度。

    implicit_satisfied:
        收到結果後直接問下一題（不同 topic）、結束對話、引用結果
    implicit_unsatisfied:
        同 topic 連續問同樣問題、「算了」「不用了」、放棄方向
    """
    if not current_end_data:
        return "unknown"

    result = current_end_data.get("result", "")

    # 如果有下一次互動，看是否為新 topic（滿意）或同 topic 重複（不滿意）
    if next_interaction:
        next_input = next_interaction.get("user_input", "")
        next_lower = next_input.lower() if next_input else ""

        # 不滿意訊號
        unsatisfied_signals = ["算了", "不用了", "換個方式", "重來", "再來"]
        if any(sig in next_lower for sig in unsatisfied_signals):
            return "implicit_unsatisfied"

        # 如果引用了結果
        if result and next_input:
            result_tokens = _tokenize(result)
            next_tokens = _tokenize(next_input)
            overlap = set(result_tokens) & set(next_tokens)
            if len(overlap) >= 2:
                return "implicit_satisfied"

    return "unknown"


# ----------------------------------------------------------------------
# P1-3：觀測校正與 RBH 回收
# ----------------------------------------------------------------------
def check_pillar_observation_conflict(
    profile: dict[str, Any],
    dimension: str,
    observed_value: str,
) -> dict[str, Any] | None:
    """比對觀測值與四柱映射預測值。

    一致 → pillars_validated
    衝突累計 >= 3 → 觀測覆蓋（但 human_calibrated / human_calibrated_stale 不參與自動覆蓋）
    """
    parts = dimension.split(".")
    if len(parts) != 2:
        return None

    section, field = parts
    if section not in profile or not isinstance(profile[section], dict):
        return None

    current_source = profile[section].get("data_source", "default")
    if current_source in ("human_calibrated", "human_calibrated_stale"):
        return None  # 真人校準（含過期）不被四柱衝突覆蓋

    predicted_value = profile[section].get(field)
    if predicted_value is None:
        return None

    match = predicted_value == observed_value

    # 找到這個維度在 rbh_log 中的衝突計數
    rbh_log = profile.get("rbh_log", [])
    conflict_count = sum(
        1 for entry in rbh_log
        if entry.get("dimension") == dimension and not entry.get("match", True)
    )

    if not match:
        conflict_count += 1

    action = ""
    new_source = current_source
    if match:
        if current_source == "pillars_mapped":
            profile[section]["data_source"] = "pillars_validated"
            new_source = "pillars_validated"
            action = "pillars_validated"
        else:
            action = "consistent"
    elif conflict_count >= 3:
        profile[section][field] = observed_value
        profile[section]["data_source"] = "observed"
        profile[section]["zone"] = "A"
        new_source = "observed"
        action = "observed_override"
    else:
        action = "conflict_recorded"

    rbh_entry = {
        "dimension": dimension,
        "prediction": predicted_value,
        "observed": observed_value,
        "match": match,
        "conflict_count": conflict_count,
        "action": action,
        "previous_data_source": current_source,
        "new_data_source": new_source,
        "possible_reason": "",
        "timestamp": _now_iso(),
    }

    return rbh_entry


# ----------------------------------------------------------------------
# 輔助函式
# ----------------------------------------------------------------------
def _compute_maturity(total: int) -> str:
    if total < 5:
        return "cold_start"
    if total <= 20:
        return "developing"
    return "stable"


def _compute_confidence(sample_count: int) -> str:
    if sample_count < 3:
        return "低"
    if sample_count < 8:
        return "中低"
    if sample_count < 15:
        return "中"
    if sample_count < 25:
        return "中高"
    return "高"


# ----------------------------------------------------------------------
# 外部畫像匯入解析（TOFU_使用者畫像匯入提示詞 v0.1）
# ----------------------------------------------------------------------
# 解析使用者把 ChatGPT/Claude/Gemini 等其他 AI 填好的 TOFU_PROFILE_v1
# 區塊內容貼回逗福時的純文字格式。對應 docs/specs/TOFU_使用者畫像匯入提示詞_v0_1.md
#
# 此 parser 只做格式解析，不做語意校驗。語意映射規則：
#
#   "直接說喜歡"   → preference_expression = "explicit"
#   "間接暗示"     → preference_expression = "implicit"
#   "主要說不要..."→ preference_expression = "exclusion"
#
#   "詳細分析"     → receiving_preference = "structured"
#   "簡短重點"     → receiving_preference = "minimal"
#   "自然對話"     → receiving_preference = "conversational"
#
#   "快速決定"     → pace = "fast"
#   "需要比較多資訊" → pace = "deliberate"
#   "反覆斟酌"     → pace = "avoidant"
#
#   "越多越好"     → info_appetite = "exhaustive"
#   "適量就好"     → info_appetite = "moderate"
#   "只要結論"     → info_appetite = "minimal"
# ----------------------------------------------------------------------

_IMPORT_PREF_EXPRESSION_MAP = {
    "直接說喜歡": "explicit",
    "explicit": "explicit",
    "間接暗示": "implicit",
    "implicit": "implicit",
    "主要說不要什麼": "exclusion",
    "排除": "exclusion",
    "exclusion": "exclusion",
}

_IMPORT_RECEIVING_MAP = {
    "詳細分析": "structured",
    "structured": "structured",
    "簡短重點": "minimal",
    "minimal": "minimal",
    "自然對話": "conversational",
    "conversational": "conversational",
}

_IMPORT_PACE_MAP = {
    "快速決定": "fast",
    "fast": "fast",
    "需要比較多資訊": "deliberate",
    "deliberate": "deliberate",
    "反覆斟酌": "avoidant",
    "avoidant": "avoidant",
}

_IMPORT_INFO_APPETITE_MAP = {
    "越多越好": "exhaustive",
    "exhaustive": "exhaustive",
    "適量就好": "moderate",
    "moderate": "moderate",
    "只要結論": "minimal",
    "minimal": "minimal",
}


def _match_keyword_value(text: str, mapping: dict[str, str]) -> str | None:
    """從文字中尋找 mapping 的鍵，回傳第一個命中對應的值。"""
    if not text:
        return None
    lowered = text.lower()
    for key, value in mapping.items():
        if key in text or key.lower() in lowered:
            return value
    return None


def _parse_numbered_list(block: str) -> list[str]:
    """從一個區塊裡抓出 1. / 2. / - / * 開頭的條目。"""
    items: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        # 跳過區塊標題行
        if line.startswith("###") or line.startswith("##"):
            continue
        # 數字編號 / 列點符號
        m = re.match(r"^\s*(?:\d+\.|[-*•])\s*(.+)$", line)
        if m:
            content = m.group(1).strip()
            # 移除尾端的「（如果超過...）」之類提示
            if content and not content.startswith("（") and not content.startswith("("):
                items.append(content)
    return items


def parse_imported_profile(raw_text: str) -> dict[str, Any]:
    """從使用者貼回的文字解析出結構化畫像。

    輸入是 docs/specs/TOFU_使用者畫像匯入提示詞_v0_1.md 規格的 TOFU_PROFILE_v1
    區塊內容（純文字、可帶 markdown 標題）。

    輸出格式可直接傳給 ProfileStore.import_external_profile。
    """
    profile: dict[str, Any] = {
        "communication_style": {},
        "decision_style": {},
        "interest_map": {"preferences": [], "domains": []},
    }

    if not raw_text:
        return profile

    # 切出 TOFU_PROFILE_v1 與 END_TOFU_PROFILE 之間的內容（若無標記則取整段）
    body = raw_text
    start_marker = re.search(r"TOFU_PROFILE_v1", raw_text)
    end_marker = re.search(r"END_TOFU_PROFILE", raw_text)
    if start_marker:
        start_idx = start_marker.end()
        end_idx = end_marker.start() if end_marker else len(raw_text)
        body = raw_text[start_idx:end_idx]

    # 用 ### 標題切區塊
    section_pattern = re.compile(r"^###\s*(.+?)\s*$", re.MULTILINE)
    matches = list(section_pattern.finditer(body))

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        seg_start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[title] = body[seg_start:seg_end]

    # ------ 溝通風格 ------
    comm_block = ""
    for title, content in sections.items():
        if "溝通風格" in title or "communication" in title.lower():
            comm_block = content
            break

    if comm_block:
        for raw in comm_block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if "：" not in line and ":" not in line:
                continue
            # 取冒號（中英文）前後拆成「欄位 / 回答」
            parts = re.split(r"[:：]", line, maxsplit=1)
            if len(parts) != 2:
                continue
            label, answer = parts[0].strip(), parts[1].strip()
            # 移除 markdown 列點符號
            label = re.sub(r"^[-*•]\s*", "", label)
            if not answer or answer.lower() == "unknown":
                continue

            if "表達偏好" in label or "preference" in label.lower():
                val = _match_keyword_value(answer, _IMPORT_PREF_EXPRESSION_MAP)
                if val:
                    profile["communication_style"]["preference_expression"] = val
            elif "回覆格式" in label or "格式" in label or "receiving" in label.lower():
                val = _match_keyword_value(answer, _IMPORT_RECEIVING_MAP)
                if val:
                    profile["communication_style"]["receiving_preference"] = val
            elif "決定的速度" in label or "速度" in label or "pace" in label.lower():
                val = _match_keyword_value(answer, _IMPORT_PACE_MAP)
                if val:
                    profile["decision_style"]["pace"] = val
            elif "資訊的需求量" in label or "需求量" in label or "appetite" in label.lower():
                val = _match_keyword_value(answer, _IMPORT_INFO_APPETITE_MAP)
                if val:
                    profile["decision_style"]["info_appetite"] = val

    # ------ 偏好清單 ------
    type_to_section = (
        ("explicit", ("明確偏好",)),
        ("implicit", ("隱式偏好", "隱性偏好")),
        ("exclusion", ("排除偏好",)),
    )
    for pref_type, keywords in type_to_section:
        for title, content in sections.items():
            if any(kw in title for kw in keywords):
                for item in _parse_numbered_list(content):
                    profile["interest_map"]["preferences"].append({
                        "item": item,
                        "type": pref_type,
                        "context": "",
                        "session_ref": "imported",
                    })
                break

    # ------ 興趣領域 ------
    for title, content in sections.items():
        if "興趣領域" in title:
            for name in _parse_numbered_list(content):
                profile["interest_map"]["domains"].append({
                    "name": name,
                    "mention_count": 1,
                    "depth": "surface",
                })
            break

    # ------ 關鍵名詞（合併進偏好的 context 不適合，當作 implicit 的補充偏好項目）----
    for title, content in sections.items():
        if "關鍵名詞" in title:
            for name in _parse_numbered_list(content):
                # 避免重複
                if any(p["item"] == name for p in profile["interest_map"]["preferences"]):
                    continue
                profile["interest_map"]["preferences"].append({
                    "item": name,
                    "type": "implicit",
                    "context": "key_noun",
                    "session_ref": "imported",
                })
            break

    return profile


def render_profile_summary(profile: dict[str, Any]) -> str:
    """把畫像渲染成人話摘要（用於 /profile 指令）。"""
    cs = profile.get("communication_style", {})
    ds = profile.get("decision_style", {})
    im = profile.get("interest_map", {})
    sp = profile.get("satisfaction_pattern", {})
    meta = profile.get("meta", {})

    # 溝通風格描述
    recv_map = {
        "structured": "你偏好有結構的回覆，逗福會給列點式建議。",
        "conversational": "你偏好自然對話，逗福用聊天方式回覆。",
        "minimal": "你偏好簡短回覆，逗福會給精簡版建議。",
    }
    recv_desc = recv_map.get(cs.get("receiving_preference", ""), "尚未確定")

    # 偏好表達描述
    pref_map = {
        "explicit": "你的偏好通常直接說出來，逗福用關鍵字比對找偏好。",
        "implicit": "你的偏好通常是間接帶過的，逗福會主動找隱藏線索。",
        "exclusion": "你習慣用排除法表達偏好，逗福會優先找排除約束。",
    }
    pref_desc = pref_map.get(cs.get("preference_expression", ""), "尚未確定")

    # 決策方式描述
    pace_map = {
        "fast": "你做決定很快，逗福直接給建議。",
        "deliberate": "你做決定前會想一下，逗福會給你 2-3 個選項。",
        "avoidant": "你傾向反覆斟酌，逗福會幫你收斂選項。",
    }
    pace_desc = pace_map.get(ds.get("pace", ""), "尚未確定")

    lines = [
        "[逗福Tofu] 目前對你的理解：",
        "",
        f"溝通風格：{recv_desc}",
        f"偏好表達：{pref_desc}",
        f"決策方式：{pace_desc}",
    ]

    # 興趣領域
    domains = im.get("domains", [])
    if domains:
        lines.append("")
        lines.append("[興趣領域]")
        depth_map = {"surface": "偶爾提到", "engaged": "常聊", "expert": "深度關注"}
        for i, d in enumerate(sorted(domains, key=lambda x: x.get("mention_count", 0), reverse=True)[:5], 1):
            name = d.get("name", "")
            if not name:
                continue
            depth_label = depth_map.get(d.get("depth", ""), "")
            lines.append(f"  {i}. {name}（{depth_label}）")

    # 已知偏好
    prefs = im.get("preferences", [])
    if prefs:
        lines.append("")
        lines.append("[已知偏好]")
        type_map = {"explicit": "顯式", "implicit": "隱式", "exclusion": "排除"}
        for p in prefs[:8]:
            item = p.get("item")
            if not isinstance(item, str) or not item:
                continue
            t = type_map.get(p.get("type", ""), "")
            lines.append(f"  - {item}（{t}）")

    # 滿意度
    total_sat = sp.get("explicit_satisfied", 0) + sp.get("implicit_satisfied", 0)
    total_unsat = sp.get("explicit_unsatisfied", 0) + sp.get("implicit_unsatisfied", 0)
    rate = sp.get("effective_satisfaction_rate", 0.0)
    lines.append("")
    lines.append(f"滿意度：{rate * 100:.0f}%（滿意 {total_sat} / 不滿意 {total_unsat}）")

    maturity = meta.get("maturity", "cold_start")
    maturity_map = {
        "cold_start": "冷啟動",
        "developing": "建立中",
        "stable": "穩定",
    }
    source = cs.get("data_source", "default")
    source_map = {
        "default": "預設",
        "imported": "外部匯入",
        "observed": "觀測",
        "pillars_mapped": "四柱映射",
        "pillars_validated": "四柱已驗證",
        "human_calibrated": "真人校準",
        "human_calibrated_stale": "真人校準（已過期）",
    }
    lines.append(
        f"畫像成熟度：{maturity_map.get(maturity, maturity)}"
        f"（{meta.get('total_interactions', 0)} 筆互動）"
    )
    lines.append(f"資料來源：{source_map.get(source, source)}")

    return "\n".join(lines)
