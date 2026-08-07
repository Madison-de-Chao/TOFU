#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAV 判分彙總 — 純程式碼，不呼叫 LLM

用法:
  python3 scorer.py "results/*_raw.jsonl"
  python3 scorer.py "results/*_raw.jsonl" --threshold 15 --by-author
"""
import json, glob, argparse, os
from collections import defaultdict

CK = [10, 20, 30, 40, 50]


def slope(xs, ys):
    """最小平方法斜率，回傳每 10 輪的百分點變化"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b * 10 * 100  # 每 10 輪的百分點


def rate(d):
    n = d["hit"] + d["miss"]
    return d["hit"] / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="inferable 與 non_inferable 的差距門檻（百分點）")
    ap.add_argument("--by-author", action="store_true", help="依出題方分組")
    ap.add_argument("--condition", choices=["all", "plain", "anchored"],
                    default="all",
                    help="依題幹條件過濾。plain=無錨定句、anchored=有錨定句。"
                         "glob 常會同時吃到兩種檔案，混在一起判分會把 A/B "
                         "兩條曲線攪成一條，跑過 --anchored 的話務必分開判。")
    ap.add_argument("--out", default="summary.json")
    a = ap.parse_args()

    rows = []
    for f in glob.glob(a.pattern):
        src = os.path.basename(f).replace("_raw.jsonl", "")
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            anch = bool(r.get("anchored", False))
            if a.condition == "plain" and anch:
                continue
            if a.condition == "anchored" and not anch:
                continue
            r["_src"] = src
            r["_author"] = ("gpt" if "gpt" in src.lower() else
                            "claude" if "claude" in src.lower() else
                            "gemini" if "gemini" in src.lower() else
                            "grok" if "grok" in src.lower() else "other")
            rows.append(r)
    if not rows:
        raise SystemExit(f"找不到資料：{a.pattern}")

    if a.condition == "all":
        kinds = {bool(r.get("anchored", False)) for r in rows}
        if len(kinds) > 1:
            print("⚠ 警告：資料同時含 plain 與 anchored 兩種條件，曲線已被混合。\n"
                  "  請分開判分：--condition plain 與 --condition anchored 各跑一次。")

    def tally(sel):
        t = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "miss": 0, "unparse": 0}))
        for r in sel:
            k = "unparse" if r["status"] == "unparseable" else \
                ("hit" if r["status"] == "hit" else "miss")
            t[r["dim_class"]][r["checkpoint"]][k] += 1
        return t

    def report(label, sel):
        t = tally(sel)
        print(f"\n{'─'*66}\n{label}   n={len(sel)}")
        print(f"{'測點':<8}{'inferable':>12}{'non_inf':>11}{'差距':>10}{'unparse':>10}")
        gaps, ir, nr = [], [], []
        for c in CK:
            i, n = t["inferable"][c], t["non_inferable"][c]
            if not (i["hit"] + i["miss"] + i["unparse"]):
                continue
            ri, rn = rate(i), rate(n)
            tot = sum(i.values()) + sum(n.values())
            upr = (i["unparse"] + n["unparse"]) / tot if tot else 0
            gaps.append((c, (ri - rn) * 100))
            ir.append((c, ri)); nr.append((c, rn))
            print(f"{c:<8}{ri*100:>11.1f}%{rn*100:>10.1f}%{(ri-rn)*100:>+9.1f}pp{upr*100:>9.1f}%")

        if len(ir) >= 2:
            si = slope([x for x, _ in ir], [y for _, y in ir])
            sn = slope([x for x, _ in nr], [y for _, y in nr])
            final_gap = gaps[-1][1] if gaps else 0
            print(f"\n  inferable 斜率      {si:+.2f} pp / 10 輪")
            print(f"  non_inferable 斜率  {sn:+.2f} pp / 10 輪  （應接近 0）")
            print(f"  末測點差距          {final_gap:+.1f} pp  （門檻 {a.threshold}）")
            print("\n  判定：")
            c1 = si > 0
            c2 = final_gap >= a.threshold
            # 對照組只在「同步上升」時才算失敗；下降或持平皆通過。
            c3 = sn < max(2.0, si / 2) if si > 0 else False
            print(f"    inferable 上升趨勢     {'通過' if c1 else '未通過'}")
            print(f"    與對照組差距達門檻     {'通過' if c2 else '未通過'}")
            print(f"    對照組未同步上升       {'通過' if c3 else '未通過'}")
            print(f"    → H1 {'成立' if (c1 and c2 and c3) else '不成立'}")
            return {"slope_inferable": round(si, 2), "slope_non_inferable": round(sn, 2),
                    "final_gap": round(final_gap, 1),
                    "h1_supported": bool(c1 and c2 and c3),
                    "by_checkpoint": {str(c): {"inferable": round(dict(ir).get(c, 0), 3),
                                               "non_inferable": round(dict(nr).get(c, 0), 3)}
                                      for c in CK if c in dict(ir)}}
        return {}

    out = {"overall": report("全部合併", rows), "threshold": a.threshold}

    if a.by_author:
        out["by_author"] = {}
        for au in sorted(set(r["_author"] for r in rows)):
            sel = [r for r in rows if r["_author"] == au]
            out["by_author"][au] = report(f"出題方：{au}", sel)

    # 全測點皆中的題目 — 可能自帶答案
    per = defaultdict(list)
    for r in rows:
        if r["dim_class"] == "inferable":
            per[(r["_src"], r["dim_id"])].append(r["status"])
    always = [k for k, v in per.items() if len(v) >= 4 and all(s == "hit" for s in v)]
    never = [k for k, v in per.items() if len(v) >= 4 and all(s == "miss" for s in v)]
    if always:
        print(f"\n所有測點皆答對的 inferable 題 {len(always)} 道（疑似自帶答案）：")
        for s, d in always[:15]:
            print(f"  {s:<26}{d}")
    if never:
        print(f"\n所有測點皆答錯的 inferable 題 {len(never)} 道（疑似不可推論或 gold 有誤）：")
        for s, d in never[:15]:
            print(f"  {s:<26}{d}")
    out["always_correct"] = [f"{s}:{d}" for s, d in always]
    out["never_correct"] = [f"{s}:{d}" for s, d in never]

    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
