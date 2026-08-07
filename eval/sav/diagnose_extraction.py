#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""偏好提取診斷 — 掃一份 endpoints.jsonl，回報畫像提取的品質。純程式碼，不呼叫 API。

用途：
  1. 跑完 dump_profile 後驗屍：提取了幾條、幾條是誤匹配、domains 乾不乾淨。
  2. 回掃歷史資料（例如白皮書 6.1 的 244 筆）：統計「要不要」誤匹配
     在舊提取結果中的佔比。

用法:
  python3 eval/sav/diagnose_extraction.py _profile_run/data/endpoints.jsonl
"""
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.middleware.user_profile import (  # noqa: E402
    extract_domains,
    extract_preferences,
)

# 修復前的無防護指標（重現舊行為用）
_LEGACY_EXCLUSION = ["不要", "不喜歡", "別給我", "避免", "討厭", "不想"]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("用法：python3 diagnose_extraction.py <endpoints.jsonl>")
    path = sys.argv[1]

    inputs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != "start":
                continue
            text = (row.get("start_data") or {}).get("user_input", "")
            if text:
                inputs.append(text)
    print(f"start 端點 {len(inputs)} 筆")

    inter = [{"user_input": t} for t in inputs]

    # 1. 現行提取
    prefs = extract_preferences(inter)
    print(f"\n== 偏好提取（現行版）：{len(prefs)} 條 ==")
    for p in prefs:
        print(f"  [{p['type']}] {p['item']}")

    # 2. 誤匹配統計：舊版子字串比對會多抓幾條
    yaobuyao = [t for t in inputs if "要不要" in t]
    legacy_hits = [
        t for t in inputs
        if any(kw in t for kw in _LEGACY_EXCLUSION)
    ]
    guarded_hits = [
        t for t in inputs
        if re.search(r"(?<!要)不要(?!緊|臉)|(?<!喜)不喜歡|別給我|避免|討厭|(?<!想)不想", t)
    ]
    print(f"\n== 誤匹配統計 ==")
    print(f"  含「要不要」（舊版必誤抓）  {len(yaobuyao)} 筆")
    print(f"  舊版子字串命中               {len(legacy_hits)} 筆")
    print(f"  防護版命中                   {len(guarded_hits)} 筆")
    print(f"  → 差額 {len(legacy_hits) - len(guarded_hits)} 筆是修復所擋下的誤匹配")

    # 3. domains 品質
    domains = extract_domains(inter)
    print(f"\n== domains 前 20（名詞性過濾後） ==")
    for d in domains:
        print(f"  {d['name']:<10}{d['mention_count']:>3}  {d['depth']}")

    # 4. 判讀指引
    print(
        "\n判讀：\n"
        "  偏好 0 條且輸入是敘事式語料 → 正常，關鍵詞掃描的天花板本來就低，\n"
        "    參見 v0.8 實測（50 輪敘事語料指標詞覆蓋 2/50）。\n"
        "  「要不要」誤抓差額 > 0 → 舊資料建議重新提取。\n"
        "  domains 出現「這樣」「比較」等功能詞 → 斷詞未載入繁中大詞典，\n"
        "    確認 resources/jieba/dict.txt.big 存在。"
    )


if __name__ == "__main__":
    main()
