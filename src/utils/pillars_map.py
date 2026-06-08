"""四柱簡化映射表。

v3.0 P1-1：十天干 → 溝通風格 + 決策模式的映射。
來源：TOFU_四柱簡化映射表_v0_2

所有映射結果 zone: "B"（可反駁行為假設），附 falsification_condition。
有多少四柱映射多少，缺的維度不填。不追問使用者補資料。
"""

from __future__ import annotations

from typing import Any

DAY_MASTER_MAP: dict[str, dict[str, Any]] = {
    "甲": {
        "preference_expression": "explicit",
        "receiving_preference": "structured",
        "pace": "fast",
        "falsification_condition": {
            "preference_expression": "若使用者連續 5 次偏好表達皆為間接帶過，則需修正",
            "receiving_preference": "若使用者連續 3 次要求簡短回覆，則需修正",
            "pace": "若使用者連續 3 次修正復述後才確認，則需修正",
        },
    },
    "乙": {
        "preference_expression": "implicit",
        "receiving_preference": "conversational",
        "pace": "deliberate",
        "falsification_condition": {
            "preference_expression": "若使用者連續 5 次直接說出偏好，則需修正",
            "receiving_preference": "若使用者主動要求列點或結構化格式，則需修正",
            "pace": "若使用者連續 5 次第一輪就確認，則需修正",
        },
    },
    "丙": {
        "preference_expression": "explicit",
        "receiving_preference": "conversational",
        "pace": "fast",
        "falsification_condition": {
            "preference_expression": "若使用者偏好多透過行為暗示而非口頭表達，則需修正",
            "receiving_preference": "若使用者要求詳細分析報告格式，則需修正",
            "pace": "若使用者反覆修改方向超過 3 次，則需修正",
        },
    },
    "丁": {
        "preference_expression": "implicit",
        "receiving_preference": "structured",
        "pace": "deliberate",
        "falsification_condition": {
            "preference_expression": "若使用者頻繁直接表達偏好，則需修正",
            "receiving_preference": "若使用者偏好簡短回覆不要細節，則需修正",
            "pace": "若使用者多數時候第一輪就確認，則需修正",
        },
    },
    "戊": {
        "preference_expression": "explicit",
        "receiving_preference": "structured",
        "pace": "deliberate",
        "falsification_condition": {
            "preference_expression": "若使用者偏好透過行為而非語言表達喜好，則需修正",
            "receiving_preference": "若使用者多次跳過詳細分析直接問結論，則需修正",
            "pace": "若使用者連續快速確認少修正，則需修正",
        },
    },
    "己": {
        "preference_expression": "implicit",
        "receiving_preference": "conversational",
        "pace": "deliberate",
        "context_note": (
            "己土可能在不同情境展現不同接收偏好。"
            "工作/任務情境中可能偏 structured，"
            "個人/生活情境中偏 conversational。"
            "情境切換不算觀測衝突，不觸發覆蓋。"
        ),
        "falsification_condition": {
            "preference_expression": "若使用者頻繁直接說「我喜歡」「我要」，則需修正",
            "receiving_preference": (
                "若使用者在所有情境（含個人）中都主動要求結構化格式，"
                "則整體偏好應修正為 structured。"
                "注意：僅在工作情境要求結構化不構成修正理由"
            ),
            "pace": "若使用者多數時候秒回確認，則需修正",
        },
    },
    "庚": {
        "preference_expression": "explicit",
        "receiving_preference": "minimal",
        "pace": "fast",
        "falsification_condition": {
            "preference_expression": "若使用者偏好間接暗示而非直說，則需修正",
            "receiving_preference": "若使用者追問細節和背景資訊，則需修正",
            "pace": "若使用者需要多輪修正才確認，則需修正",
        },
    },
    "辛": {
        "preference_expression": "exclusion",
        "receiving_preference": "structured",
        "pace": "deliberate",
        "falsification_condition": {
            "preference_expression": (
                "若使用者多直接說「我喜歡X」而非「不要Y」，"
                "則 exclusion 假設需修正為 explicit"
            ),
            "receiving_preference": "若使用者偏好簡短對話式回覆，則需修正",
            "pace": "若使用者決策迅速少修正，則需修正",
        },
    },
    "壬": {
        "preference_expression": "implicit",
        "receiving_preference": "conversational",
        "pace": "fast",
        "falsification_condition": {
            "preference_expression": "若使用者頻繁明確表達偏好，則需修正",
            "receiving_preference": "若使用者要求結構化的分析報告，則需修正",
            "pace": "若使用者需要反覆確認才做決定，則需修正",
        },
    },
    "癸": {
        "preference_expression": "implicit",
        "receiving_preference": "minimal",
        "pace": "deliberate",
        "falsification_condition": {
            "preference_expression": "若使用者習慣直接表達需求，則需修正",
            "receiving_preference": "若使用者主動要求詳細分析，則需修正",
            "pace": "若使用者多數時候快速確認，則需修正",
        },
    },
}

# 天干列表（用於驗證）
VALID_STEMS = set(DAY_MASTER_MAP.keys())
VALID_BRANCHES = {
    "子", "丑", "寅", "卯", "辰", "巳",
    "午", "未", "申", "酉", "戌", "亥",
}


def map_pillars_to_profile(pillars: dict) -> dict | None:
    """從四柱產出搜索策略預測。目前只使用日主天干。

    有多少資料映射多少，缺的維度不填。不追問使用者補資料。
    """
    day = pillars.get("day", {})
    stem = day.get("stem")
    if not stem or stem not in DAY_MASTER_MAP:
        return None

    mapping = DAY_MASTER_MAP[stem]
    result: dict[str, Any] = {
        "communication_style": {
            "preference_expression": mapping["preference_expression"],
            "receiving_preference": mapping["receiving_preference"],
            "data_source": "pillars_mapped",
            "zone": "B",
            "falsification_condition": mapping["falsification_condition"],
        },
        "decision_style": {
            "pace": mapping["pace"],
            "data_source": "pillars_mapped",
            "zone": "B",
            "falsification_condition": mapping["falsification_condition"],
        },
    }

    # 己土情境備註
    if stem == "己" and mapping.get("context_note"):
        result["communication_style"]["context_note"] = mapping["context_note"]

    return result
