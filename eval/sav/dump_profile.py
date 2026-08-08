#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
畫像檢視 — 跑完互動後直接看逗福累積了什麼，不呼叫 API、不花錢

用法:
  # 只跑互動階段（不考試），把畫像留在指定目錄
  python3 eval/sav/dump_profile.py --card eval/sav/personas/P01_grok.json \
      --model claude-haiku-4-5-20251001 --keep _profile_run

  # 看已經跑完的目錄
  python3 eval/sav/dump_profile.py --inspect _profile_run/data
"""
import json, os, sys, argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import _sandbox  # noqa: F401, E402  詞表隔離——必須在 src.main 之前
from src.main import run_one_interaction  # noqa: E402
from src.middleware.endpoint import EndpointStore  # noqa: E402
from src.middleware.user_profile import ProfileStore  # noqa: E402


def inspect(data_dir):
    pf = os.path.join(data_dir, "user_profile.json")
    ep = os.path.join(data_dir, "endpoints.jsonl")

    print("=" * 66)
    if os.path.exists(pf):
        p = json.load(open(pf, encoding="utf-8"))
        cs = p.get("communication_style", {})
        ds = p.get("decision_style", {})
        im = p.get("interest_map", {})
        prefs = im.get("preferences", [])
        meta = p.get("meta", {}) or {}
        print(f"畫像成熟度      {meta.get('maturity', p.get('maturity', '?'))}")
        print(f"互動次數        {meta.get('total_interactions', p.get('total_interactions', '?'))}")
        print(f"偏好條目        {len(prefs)}")
        # domains 是 list[{name, mention_count, depth}]，不是 dict
        domains = im.get("domains") or []
        names = [d.get("name") for d in domains if isinstance(d, dict)][:8]
        print(f"領域            {names}（共 {len(domains)} 個）")
        print(f"表達方式        {cs.get('preference_expression')}")
        print(f"接收偏好        {cs.get('receiving_preference')}")
        print(f"決策風格        {ds}")
        print()
        print("偏好清單（送進 system prompt 的前 10 筆）：")
        for i, x in enumerate(prefs[:10], 1):
            print(f"  {i:>2}. {x}")
        if len(prefs) > 10:
            print(f"  （另有 {len(prefs)-10} 筆未進 prompt）")
    else:
        print(f"找不到畫像檔：{pf}")

    print()
    if os.path.exists(ep):
        st = EndpointStore(ep)
        print(f"端點事件        {st.completed_count()} 對 start/end")
    print("=" * 66)
    print()
    print("判讀重點：")
    print("  偏好條目為 0 或個位數 → 記憶沒累積起來，作答等同裸模型")
    print("  偏好條目充足但答題不準 → 記憶有累積，是題目測不到或檢索沒用上")
    print("  這兩種情況的處理方式完全不同，先看這個再決定改哪裡")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card")
    ap.add_argument("--model")
    ap.add_argument("--keep", default="_profile_run")
    ap.add_argument("--inspect")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()

    if a.inspect:
        inspect(a.inspect)
        return

    if not a.card or not a.model:
        sys.exit("需要 --card 與 --model，或用 --inspect 看既有目錄")

    from src.api.claude_client import LLMClient
    card = json.load(open(a.card, encoding="utf-8"))
    rev = sorted(card.get("revealed", []), key=lambda r: r.get("session_order", 0))[:a.limit]

    data_dir = os.path.join(a.keep, "data")
    os.makedirs(data_dir, exist_ok=True)
    store = EndpointStore(os.path.join(data_dir, "endpoints.jsonl"))
    prof = ProfileStore(os.path.join(data_dir, "user_profile.json"))
    client = LLMClient(model=a.model)

    print(f"跑 {len(rev)} 輪互動，模型 {a.model}")
    for i, r in enumerate(rev, 1):
        try:
            run_one_interaction(str(r.get("content", "")), store, client,
                                profile_store=prof, auto_confirm=True)
        except Exception as e:
            print(f"  第 {i} 輪失敗：{e}")
        if i % 10 == 0:
            print(f"  ...{i}")

    print()
    inspect(data_dir)


if __name__ == "__main__":
    main()
