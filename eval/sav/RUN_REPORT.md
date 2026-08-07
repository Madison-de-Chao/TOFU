# SAV 執行回報 — 2026-08-07

依 `eval/sav/README.md` 執行。**第一步與第二步未能執行**，原因與已完成的部分如下。

---

## 結論先講

| 步驟 | 狀態 |
|---|---|
| 放置 `eval/` 到專案根目錄 | 完成 |
| 環境安裝 | 完成（`jieba` 需繞道，見下） |
| 附帶工具 `qc_cards.py` | 完成，20 張卡全數檢查 |
| 管線驗證（`--dry-run`） | 完成，250 次呼叫、5 個測點全通過污染斷言 |
| **第一步：單張真實 API 執行** | **未執行 — 環境沒有 API key** |
| **第二步：20 張真實 API 執行** | **未執行 — 同上** |
| 第三步：`scorer.py` 判分 | 機制已驗證，但**沒有可判的真實資料** |

沒有任何一個真實的 inferable／non_inferable 命中率數字產生。
下面所有數字都不是實驗結果，**不要拿去解讀 H1**。

---

## 阻塞原因：沒有 API key

執行環境中 `CLAUDE_API_KEY` 與 `ANTHROPIC_API_KEY` 皆未設定。
`sav_runner.py` 沒有 key 時會靜默走 fallback，不會報錯，跑出來的是純程式碼樣板。
在沒有 key 的情況下硬跑第一步，只會得到一份 unparse 100% 的假資料，
所以沒有跑，也沒有把任何檔案寫進 `results/`。

**要繼續，需要提供 `CLAUDE_API_KEY`。** 拿到之後第一步的指令原封不動可用。

---

## 一個會擋住第一步的文件錯誤（已修）

README 原本寫 `export ANTHROPIC_API_KEY=sk-...`，但
`src/api/claude_client.py` 第 439 行讀的是 `CLAUDE_API_KEY`：

```python
key = api_key or os.environ.get("CLAUDE_API_KEY")
if not key or Anthropic is None:
    self.fallback_mode = True   # 靜默降級，不拋例外
```

全 repo 沒有任何一處讀 `ANTHROPIC_API_KEY`（`tests/` 裡出現兩次，都是在
`os.environ.pop()` 清理殘留）。專案主 `README.md` 第 104 行用的也是 `CLAUDE_API_KEY`。

照原文設定會進 fallback 而不報錯，症狀就是 unparse 直接 100%——
也就是 README 表格裡「超過 10% 就停下來回報」那一格會被觸發，
但真正的原因是變數名字，不是逗福的回覆格式。已改掉 README 與 `sav_runner.py` 的 docstring。

---

## 已完成：管線驗證（`--dry-run`）

```bash
python3 eval/sav/sav_runner.py "eval/sav/personas/P01_grok.json" --dry-run --out <臨時目錄>
```

結果：**75 筆、250 次呼叫、14 秒，跑完沒有中斷。**

驗證到的三件事：

1. **路徑假設成立。** `Path(__file__).resolve().parents[2]` 在 `eval/sav/` 下正確解析到專案根目錄，`src.main` / `src.middleware.*` 全部匯入成功。
2. **測驗零污染主記憶。** 5 個測點的 `assert before == after` 全數通過，沒有觸發。
3. **`--model` 必填的硬規則有效。** 不給 `--model` 也不給 `--dry-run` 時直接退出，沒有預設值可誤用。

輸出**沒有**寫進 `eval/sav/results/`，寫在臨時目錄。fallback 資料混進 `results/`
會被 scorer 自動合併進真實資料，所以刻意隔開；`.gitignore` 也加了
`eval/sav/results/*_raw.jsonl` 防止誤commit。

`scorer.py` 也對這份 dry-run 輸出跑過一次，確認判分、`--by-author` 分組、
斜率計算、`summary.json` 輸出都正常。三個條件全部「未通過」、H1 不成立——
這是 fallback 全 unparse 的必然結果，不是實驗結論。

---

## dry-run 的 unparse 是 100%，而且這件事值得注意

README 第 74 行說「dry-run 時確實出現這個狀況，但真實 API 下會不會發生沒驗證過」。
真實 API 仍然沒驗證，但**讀過 prompt 組裝路徑後，這個風險比原本描述的更具體**。

`run_quiz()` 走的是 `mode="free"`，最後落到 `execute_task()`，system prompt 是
`TOFU_IDENTITY_PROMPT` + `OUTPUT_PROMPT_FREE`（`src/api/claude_client.py:154`、`1063`）。
`OUTPUT_PROMPT_FREE` 裡明文寫著：

- 「結構：[結論] → [關鍵假設 1-2 個] → [具體步驟 1-2 個] → [反駁條件 1 個]」
- 「4-8 句。你在跟人類說話。」
- 一整段「禁止的迴避句型」，要求給出具體可執行內容

而測驗題目在 user message 裡要求「只回覆一個大寫字母，不要解釋」。
**這兩者是直接衝突的，而且 system prompt 通常贏。**

也就是說，高 unparse 未必是 dry-run 專屬的假象，真實 API 下有結構性的理由會發生。

依 README 第 76 行「不要自行調整 prompt 硬修，先回報實際的 raw_response」，
這裡只回報、沒有動任何 prompt。真跑第一步時如果 unparse 超過 10%，
問題大概率在 `OUTPUT_PROMPT_FREE` 的格式要求蓋掉了單字母指令，
而不是 `extract_choice()` 的正則不夠寬。

---

## 已完成：`qc_cards.py` 格式檢查

20 張卡全部載入成功，條數與 dim_id 連號無誤。兩個 FAIL：

**1. `P02_gemini.json` — gold 位置分布失衡 `{A:2, B:5, C:6, D:2}`**

這是 README 第 142 行寫明的位置偏好探針，**未修改**。

**2. `P03_grok.json` — `N02` 疑似非獨立（重疊 0.5）**

這一張 README 沒有提到。N02 是對照組題目，對照組若不獨立，
會讓 non_inferable 基線偏離 25%，直接影響判定條件 3。
依「不要修改人物卡」的原則**未修改**，僅回報。
真跑完之後可以用 scorer 的「所有測點皆答對」清單交叉確認是不是真有問題。

**warn 層級（僅供參考，未處理）**

- 「疑似答案洩漏」共 26 題次，分布於 15 張卡。README 第 158 行已說明這一段
  訊噪比不好（詞袋重疊會把常用詞算成線索），當參考不當結論。
- `P01_gpt.json` 的 `I08` 被標為洩漏（重疊 1.0）——這正是 README 第 141 行
  描述的端點檢索探針，符合預期。
- revealed 長度越界 5 條、疑似條列式 4 條、選項長度差距 2 題、
  `P05_grok.json` 難度分布 易2/中5/難3（規範為 3/5/2）。

---

## 環境安裝備註

`pip install jieba` 在本環境（Debian 系 Python 3.11、setuptools 較新）失敗：

```
AttributeError: install_layout. Did you mean: 'install_platlib'?
```

jieba 0.42.1 的舊版 `setup.py` 與新 setuptools 不相容，PyPI 上也只有 sdist 沒有 wheel。
繞道方式（直接把套件目錄放進 site-packages）已寫進 README 的「環境」段落。
`anthropic` 0.120.2 正常安裝。

---

## 下一步

拿到 `CLAUDE_API_KEY` 之後：

```bash
export CLAUDE_API_KEY=sk-ant-...
python3 eval/sav/sav_runner.py "eval/sav/personas/P01_grok.json" \
    --model claude-haiku-4-5-20251001 \
    --out eval/sav/results
```

先看 unparse。超過 10% 就停下來回報 `raw_response`，不要接著跑第二步的 US$26。
