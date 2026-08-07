#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAV — 螺旋上升驗證 執行腳本

流程：
  Phase 1  逐條餵入 revealed（auto_confirm=True）
  Phase 2  每到 checkpoint，複製整個 data 目錄到臨時位置作答，答完刪除
           → 測驗零污染主記憶

用法:
  export CLAUDE_API_KEY=sk-ant-...   # 注意：不是 ANTHROPIC_API_KEY
  python3 sav_runner.py 題庫/GROK/P01_Grok.json --model claude-haiku-4-5-20251001
  python3 sav_runner.py "題庫/*/*.json" --model claude-haiku-4-5-20251001 --out results/

  # 不呼叫 API，只驗證流程（用 repo 內建 fallback）
  python3 sav_runner.py 題庫/GROK/P01_Grok.json --dry-run

放置位置：專案根目錄的 eval/sav/ 下。
"""
import json, os, sys, re, glob, time, shutil, argparse, tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.main import run_one_interaction  # noqa: E402
from src.middleware.endpoint import EndpointStore  # noqa: E402
from src.middleware.user_profile import ProfileStore  # noqa: E402

CHECKPOINTS = [10, 20, 30, 40, 50]

QUIZ_TMPL = """/free {question}

A. {A}
B. {B}
C. {C}
D. {D}

只回覆一個大寫字母，不要解釋。"""


def extract_choice(text):
    """從回覆抽取選項字母。抽不出回 None。"""
    if not text:
        return None
    t = str(text).strip()
    if len(t) == 1 and t.upper() in "ABCD":
        return t.upper()
    m = re.match(r'^\s*\**([ABCD])\**[\.\)、:：\s]', t)
    if m:
        return m.group(1)
    m = re.search(r'(?:答案|選項|我選|選擇)\s*(?:是|為)?\s*\**([ABCD])', t)
    if m:
        return m.group(1)
    # 全文只出現一個獨立的 A-D 才採用，避免誤抓
    cands = set(re.findall(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', t))
    return cands.pop() if len(cands) == 1 else None


def make_client(model, dry_run):
    if dry_run:
        from src.api.claude_client import LLMClient
        return LLMClient(api_key=None)  # 無 key → repo 內建 fallback 模板
    from src.api.claude_client import LLMClient
    return LLMClient(model=model)


def last_result(store):
    """取最後一筆 end 端點的 result 文字"""
    ends = store.ends()
    return ends[-1].get("result", "") if ends else ""


def run_quiz(card, work_dir, model, dry_run, checkpoint, calls):
    """在 data 目錄的臨時副本上作答，主記憶零污染"""
    tmp = tempfile.mkdtemp(prefix="sav_quiz_")
    try:
        shutil.copytree(work_dir, os.path.join(tmp, "data"), dirs_exist_ok=True)
        qs = EndpointStore(os.path.join(tmp, "data", "endpoints.jsonl"))
        qp = ProfileStore(os.path.join(tmp, "data", "user_profile.json"))
        qc = make_client(model, dry_run)
        rows = []
        for cls in ("inferable", "non_inferable"):
            for q in card.get(cls, []):
                o = q.get("options", {})
                if not all(k in o for k in "ABCD"):
                    continue
                prompt = QUIZ_TMPL.format(question=q["question"], **{k: o[k] for k in "ABCD"})
                t0 = time.time()
                try:
                    run_one_interaction(prompt, qs, qc, profile_store=qp,
                                        auto_confirm=True, mode="free")
                    raw = last_result(qs)
                    calls[0] += 2
                except Exception as e:
                    raw = f"__ERROR__ {e}"
                pick = extract_choice(raw)
                gold = q.get("gold")
                status = "unparseable" if pick is None else ("hit" if pick == gold else "miss")
                rows.append({
                    "persona_id": card.get("persona_id"),
                    "checkpoint": checkpoint,
                    "dim_id": q["dim_id"],
                    "dim_class": cls,
                    "gold": gold,
                    "extracted": pick,
                    "status": status,
                    "raw_response": str(raw)[:400],
                    "elapsed": round(time.time() - t0, 2),
                })
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_card(path, model, dry_run, out_dir):
    card = json.load(open(path, encoding="utf-8"))
    pid = card.get("persona_id", "?")
    tag = Path(path).stem
    rev = sorted(card.get("revealed", []), key=lambda r: r.get("session_order", 0))
    print(f"\n{'='*64}\n{tag}  persona={pid}  revealed={len(rev)}  model={model or 'DRY-RUN'}")

    work = tempfile.mkdtemp(prefix=f"sav_{tag}_")
    data_dir = os.path.join(work, "data")
    os.makedirs(data_dir, exist_ok=True)
    store = EndpointStore(os.path.join(data_dir, "endpoints.jsonl"))
    prof = ProfileStore(os.path.join(data_dir, "user_profile.json"))
    client = make_client(model, dry_run)

    calls = [0]
    all_rows = []
    t_start = time.time()
    try:
        for i, r in enumerate(rev, 1):
            try:
                run_one_interaction(str(r.get("content", "")), store, client,
                                    profile_store=prof, auto_confirm=True)
                calls[0] += 2
            except Exception as e:
                print(f"  互動失敗 {r.get('dim_id')}: {e}")
            if i in CHECKPOINTS:
                before = store.completed_count()
                rows = run_quiz(card, data_dir, model, dry_run, i, calls)
                after = store.completed_count()
                assert before == after, (
                    f"測驗污染主記憶！{before} → {after}。停止執行。")
                hit = sum(1 for x in rows if x["status"] == "hit")
                ih = sum(1 for x in rows if x["dim_class"] == "inferable" and x["status"] == "hit")
                nh = sum(1 for x in rows if x["dim_class"] == "non_inferable" and x["status"] == "hit")
                up = sum(1 for x in rows if x["status"] == "unparseable")
                print(f"  ck{i:>3}  inferable {ih}/10   non_inf {nh}/5   "
                      f"unparse {up}   累計呼叫 {calls[0]}")
                all_rows += rows
    finally:
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"{tag}_raw.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for x in all_rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  → {fp}  ({len(all_rows)} 筆, {calls[0]} 次呼叫, "
              f"{time.time()-t_start:.0f}s)")
        shutil.rmtree(work, ignore_errors=True)
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="人物卡路徑或 glob")
    ap.add_argument("--model", default=None,
                    help="必填（除非 --dry-run）。不給預設值，避免誤打高價模型")
    ap.add_argument("--dry-run", action="store_true", help="不呼叫 API，只驗證流程")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if not a.dry_run and not a.model:
        sys.exit("錯誤：--model 為必填。範例 --model claude-haiku-4-5-20251001")

    files = [f for f in sorted(glob.glob(a.pattern)) if "QC_report" not in os.path.basename(f)] \
        or ([a.pattern] if os.path.exists(a.pattern) else [])
    if not files:
        sys.exit(f"找不到卡片：{a.pattern}")

    print(f"卡片 {len(files)} 張  checkpoints={CHECKPOINTS}")
    for f in files:
        run_card(f, a.model, a.dry_run, a.out)
    print(f"\n完成。用 scorer.py 彙總：python3 scorer.py '{a.out}/*_raw.jsonl'")


if __name__ == "__main__":
    main()
