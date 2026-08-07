#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人物卡題庫自動檢查 — 對應設計規範 v1 第 6 節檢查表 + 詞彙洩漏偵測"""
import json, glob, os, sys, re
from collections import Counter, defaultdict
import jieba

STOP = set("的 了 在 我 有 和 就 不 人 都 一 一個 上 也 很 到 說 要 去 你 會 著 沒有 看 好 自己 這 那 是 我們 他 她 它 但 而 又 還 才 把 被 讓 給 從 對 跟 與 或 因為 所以 如果 雖然 就是 可以 應該 覺得 知道 想 做 用 個 們 之 其 此 些 麼 什麼 怎麼 為 以 於 及 等 中 大 小 多 少 後 前 時 候 天 年 月 日 次 種 樣 事 東西 時候 現在 已經 還是 只是 這樣 那樣 一直 真的 比較 有點 一點 可能 應該".split())

def words(t):
    return set(w for w in jieba.cut(str(t)) if len(w) >= 2 and w not in STOP and not w.isdigit())

def overlap(opt_text, rev_words):
    """選項實詞在 revealed 詞袋中的覆蓋率"""
    w = words(opt_text)
    return len(w & rev_words) / len(w) if w else 0.0

def check(path):
    d = json.load(open(path, encoding='utf-8'))
    name = os.path.basename(path)
    author = d.get('author_model', '')
    pid = d.get('persona_id', '?')
    rev = d.get('revealed', [])
    inf = d.get('inferable', [])
    non = d.get('non_inferable', [])
    fails, warns = [], []

    # --- 數量 ---
    if len(rev) != 50: fails.append(f"revealed {len(rev)} 條（應 50）")
    if len(inf) != 10: fails.append(f"inferable {len(inf)} 題（應 10）")
    if len(non) != 5: fails.append(f"non_inferable {len(non)} 題（應 5）")

    # --- dim_id 完整性 ---
    rids = [r.get('dim_id') for r in rev]
    if len(set(rids)) != len(rids): fails.append("revealed dim_id 有重複")
    exp = [f"R{i:02d}" for i in range(1, len(rev)+1)]
    if rids != exp: warns.append("revealed dim_id 非 R01..Rnn 連號")

    # --- revealed 長度 ---
    bad_len = [r.get('dim_id') for r in rev if not (30 <= len(str(r.get('content',''))) <= 120)]
    if bad_len: warns.append(f"revealed 長度越界 {len(bad_len)} 條: {bad_len[:6]}")

    # --- revealed 條列式 / 自述 ---
    listish = [r.get('dim_id') for r in rev if re.search(r'[：:].*[、,].*[、,]', str(r.get('content','')))]
    if listish: warns.append(f"疑似條列式 {len(listish)} 條: {listish[:5]}")
    selfdesc = [r.get('dim_id') for r in rev if re.search(r'我是[一個]{0,2}[^。，]{0,8}的人', str(r.get('content','')))]
    if selfdesc: fails.append(f"人格自述 {len(selfdesc)} 條: {selfdesc}")

    rev_text = " ".join(str(r.get('content','')) for r in rev)
    rev_words = words(rev_text)

    # --- gold 位置分布 ---
    golds = [q.get('gold') for q in inf + non]
    dist = Counter(golds)
    if len(inf+non) == 15:
        off = [k for k in "ABCD" if not (3 <= dist.get(k,0) <= 4)]
        if off: fails.append(f"gold 位置分布失衡 {dict(dist)}")

    # --- inference_basis ---
    for q in inf:
        b = q.get('inference_basis', [])
        if len(b) < 2:
            fails.append(f"{q.get('dim_id')} inference_basis 少於 2 條")
        miss = [x for x in b if x not in rids]
        if miss:
            fails.append(f"{q.get('dim_id')} basis 指向不存在: {miss}")

    # --- 選項長度差距 ---
    for q in inf + non:
        o = q.get('options', {})
        L = [len(str(v)) for v in o.values()]
        if not L: continue
        if max(L) > min(L) * 1.5 + 2:
            warns.append(f"{q.get('dim_id')} 選項長度差距大 {L}")

    # --- 詞彙洩漏：gold vs 干擾項 對 revealed 的重疊度 ---
    leak_inf, leak_non = [], []
    for q, bucket, tag in [(q,'inf','I') for q in inf] + [(q,'non','N') for q in non]:
        o = q.get('options', {}); g = q.get('gold')
        if g not in o: continue
        og = overlap(o[g], rev_words)
        od = [overlap(v, rev_words) for k, v in o.items() if k != g]
        avg_d = sum(od)/len(od) if od else 0
        rec = (q.get('dim_id'), round(og,2), round(avg_d,2))
        if og - avg_d >= 0.34:
            (leak_inf if bucket=='inf' else leak_non).append(rec)

    if leak_inf:
        warns.append(f"inferable 疑似答案洩漏 {len(leak_inf)} 題 (gold重疊>干擾+0.34): {leak_inf[:4]}")
    if leak_non:
        fails.append(f"non_inferable 疑似非獨立 {len(leak_non)} 題: {leak_non[:4]}")

    # --- non_inferable 主題關聯 ---
    for q in non:
        qw = words(q.get('question',''))
        hit = qw & rev_words
        if len(hit) >= 3:
            warns.append(f"{q.get('dim_id')} 題幹與 revealed 詞彙重疊 {sorted(hit)[:5]}")

    # --- 難度標註 ---
    diff = Counter()
    for q in inf:
        m = re.search(r'\[(易|中|難)\]', str(q.get('rationale','')))
        diff[m.group(1) if m else '未標'] += 1
    if diff.get('未標', 0) > 0:
        warns.append(f"難度未標註 {diff['未標']} 題")
    elif (diff.get('易'), diff.get('中'), diff.get('難')) != (3,5,2):
        warns.append(f"難度分布 易{diff.get('易',0)}/中{diff.get('中',0)}/難{diff.get('難',0)}（應 3/5/2）")

    return dict(file=name, pid=pid, author=author[:22], fails=fails, warns=warns,
                dist=dict(dist), leak_inf=len(leak_inf), leak_non=len(leak_non))

if __name__ == "__main__":
    jieba.setLogLevel(60)
    files = sorted(glob.glob(sys.argv[1]))
    results = [check(f) for f in files if 'QC_report' not in f]
    print(f"{'檔案':<34}{'ID':<5}{'FAIL':>5}{'WARN':>6}  gold分布")
    print("-"*78)
    for r in results:
        print(f"{r['file']:<34}{r['pid']:<5}{len(r['fails']):>5}{len(r['warns']):>6}  {r['dist']}")
    print("\n" + "="*78)
    for r in results:
        if r['fails'] or r['warns']:
            print(f"\n### {r['file']}  [{r['author']}]")
            for x in r['fails']: print(f"  FAIL  {x}")
            for x in r['warns']: print(f"  warn  {x}")
    json.dump(results, open('/home/claude/qc_results.json','w'), ensure_ascii=False, indent=1)
