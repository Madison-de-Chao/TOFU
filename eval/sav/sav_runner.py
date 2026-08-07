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

v3 修正：
  - make_client 強制 fallback（原本 api_key=None 會回頭讀環境變數並打 Opus）
  - last_result 讀 end_data.result（原本讀端點頂層的 result，永遠取到空字串）
  - 測驗階段關閉 analyze_deviation（每題省一次呼叫）

v4 修正：
  - _sandbox 隔離詞性標記器（它自己讀 CLAUDE_API_KEY，不經過 make_client 的
    fallback 防線；不隔離的話 --dry-run 在設了 key 的機器上仍打真實 Haiku，
    且 data/words/ 會寫進真專案目錄）
  - 抽取答案前先剝除回覆中的題幹回聲（dry-run 實測抓到假陽性：選項行尾
    「…的選項」接下一行的字母，會誤中 extract_choice 規則三）
"""
import json, os, sys, re, glob, time, shutil, argparse, tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import _sandbox  # noqa: F401, E402  詞表隔離——必須在 src.main 之前
from src.main import run_one_interaction  # noqa: E402
from src.middleware.endpoint import EndpointStore  # noqa: E402
from src.middleware.user_profile import ProfileStore  # noqa: E402

CHECKPOINTS = [10, 20, 30, 40, 50]

# 兩種題幹。差別只在有沒有指稱錨定句。
#
# 為什麼需要 anchor：逗福把記憶注入 system prompt 時，一律標記為
# 「使用者畫像」「你之前提過」「根據我對你的認識」——claude_client.py
# 裡「使用者」出現 118 次，全部是第二人稱。而題幹問「這個人」「她」，
# 是第三人稱。模型手上有一份標成「你」的畫像，題目問「這個人」，
# 兩者接不上。這是題目與 prompt 的指稱不匹配，不是記憶沒注入。
#
# 注意：字串裡不放字面的 "/free"。模式由 run_one_interaction(mode="free")
# 指定，dispatcher 才負責剝指令前綴；直接送進去會污染 goal 推斷。

QUIZ_PLAIN = """{question}

A. {A}
B. {B}
C. {C}
D. {D}

只回覆一個大寫字母，不要解釋。"""

QUIZ_ANCHORED = """下面這題問的「這個人」「他」「她」，指的就是我本人。請根據你對我的了解作答。

{question}

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
    from src.api.claude_client import LLMClient
    if dry_run:
        # 不可只傳 api_key=None：LLMClient 會回頭讀環境變數 CLAUDE_API_KEY，
        # 且預設模型是 MODEL_ID = "claude-opus-4-6"。必須強制關掉。
        c = LLMClient(api_key="__DRYRUN_NO_API__")
        c.fallback_mode = True
        c._client = None
        return c
    return LLMClient(model=model)


def last_result(store):
    """取最後一筆 end 端點的 result 文字。

    注意欄位路徑：result 位於 end_data 內層，不在端點頂層。
    端點結構為 {endpoint_id, timestamp, type, event_id, start_data, end_data}
    """
    ends = store.ends()
    if not ends:
        return ""
    return (ends[-1].get("end_data") or {}).get("result", "") or ""


def run_quiz(card, work_dir, model, dry_run, checkpoint, calls, anchored=False):
    """在 data 目錄的臨時副本上作答，主記憶零污染"""
    tmp = tempfile.mkdtemp(prefix="sav_quiz_")
    try:
        shutil.copytree(work_dir, os.path.join(tmp, "data"), dirs_exist_ok=True)
        qs = EndpointStore(os.path.join(tmp, "data", "endpoints.jsonl"))
        qp = ProfileStore(os.path.join(tmp, "data", "user_profile.json"))
        qc = make_client(model, dry_run)
        # 關掉出口檢查：答案只有一個字母 → detect_deviation 判定「異常簡短」
        # → 每題多打一次 analyze_deviation。測驗階段這段分析毫無用處，
        # 且佔測驗總呼叫的一半。main.py 用 getattr(client,"analyze_deviation")
        # 取，設成 None 後 callable() 為 False，改走 _fallback_analyze_deviation
        # （純程式碼、零 API 呼叫；黑盒子仍會寫進臨時副本，無妨）。
        qc.analyze_deviation = None
        rows = []
        for cls in ("inferable", "non_inferable"):
            for q in card.get(cls, []):
                o = q.get("options", {})
                if not all(k in o for k in "ABCD"):
                    continue
                tmpl = QUIZ_ANCHORED if anchored else QUIZ_PLAIN
                prompt = tmpl.format(question=q["question"], **{k: o[k] for k in "ABCD"})
                t0 = time.time()
                try:
                    run_one_interaction(prompt, qs, qc, profile_store=qp,
                                        auto_confirm=True, mode="free")
                    raw = last_result(qs)
                    calls[0] += 1  # free 模式跳過復述；出口檢查已關閉
                except Exception as e:
                    raw = f"__ERROR__ {e}"
                # 先剝除回覆中的題幹回聲再抽取。模型（或 fallback 樣板）常會
                # 復述題目與選項，「…的選項\nB.」這類殘影會誤中抽取規則。
                cleaned = str(raw)
                for k in "ABCD":
                    cleaned = cleaned.replace(f"{k}. {o[k]}", "")
                cleaned = cleaned.replace(q["question"], "")
                pick = extract_choice(cleaned)
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
                    "anchored": anchored,
                    "elapsed": round(time.time() - t0, 2),
                })
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_card(path, model, dry_run, out_dir, anchored=False):
    card = json.load(open(path, encoding="utf-8"))
    pid = card.get("persona_id", "?")
    tag = Path(path).stem
    rev = sorted(card.get("revealed", []), key=lambda r: r.get("session_order", 0))
    print(f"\n{'='*64}\n{tag}  persona={pid}  revealed={len(rev)}  "
          f"model={model or 'DRY-RUN'}  anchor={'ON' if anchored else 'OFF'}")

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
                rows = run_quiz(card, data_dir, model, dry_run, i, calls, anchored)
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
        suffix = "_anchored" if anchored else ""
        fp = os.path.join(out_dir, f"{tag}{suffix}_raw.jsonl")
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
    ap.add_argument("--anchored", action="store_true",
                    help="題幹加指稱錨定句（「這個人指的就是我」）。"
                         "跑一次開、一次不開，差距即為指稱斷裂造成的損失。")
    a = ap.parse_args()

    if not a.dry_run and not a.model:
        sys.exit("錯誤：--model 為必填。範例 --model claude-haiku-4-5-20251001")

    files = [f for f in sorted(glob.glob(a.pattern)) if "QC_report" not in os.path.basename(f)] \
        or ([a.pattern] if os.path.exists(a.pattern) else [])
    if not files:
        sys.exit(f"找不到卡片：{a.pattern}")

    print(f"卡片 {len(files)} 張  checkpoints={CHECKPOINTS}")
    for f in files:
        run_card(f, a.model, a.dry_run, a.out, a.anchored)
    print(f"\n完成。用 scorer.py 彙總：python3 scorer.py '{a.out}/*_raw.jsonl'")


if __name__ == "__main__":
    main()
