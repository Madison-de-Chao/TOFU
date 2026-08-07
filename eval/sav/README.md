# SAV 螺旋上升驗證 — 執行說明

## 這是什麼

驗證逗福「使用者理解隨互動輪數提升」這個宣稱。

做法：拿虛構人物卡，把 50 條這個人說過的話逐條餵進逗福，每 10 條停下來考 15 道題。

- **inferable**（10 題）：答案沒寫在那 50 條裡，但推得出來。如果逗福真的越讀越懂，命中率會上升。
- **non_inferable**（5 題）：答案跟那 50 條完全無關。這是**對照組**，命中率應該停在 25%（四選一的隨機值）。

兩條線拉開，宣稱成立。兩條線一起上升，代表題目有問題不是系統厲害。

---

## 放置位置

把整個 `eval/` 放到專案根目錄（與 `src/` 同層）：

```
tofu_mvp_model_b-main/
├── src/
├── tests/
├── data/
└── eval/
    └── sav/
        ├── sav_runner.py
        ├── scorer.py
        ├── qc_cards.py
        ├── personas/       ← 20 張人物卡
        └── results/        ← 輸出目錄（空的）
```

**路徑有依賴。** `sav_runner.py` 用 `Path(__file__).resolve().parents[2]` 找專案根目錄，也就是預設它在 `eval/sav/` 下。放別的地方要改這一行。

---

## 環境

```bash
pip install anthropic jieba
export CLAUDE_API_KEY=sk-ant-...
```

**環境變數名稱是 `CLAUDE_API_KEY`，不是 `ANTHROPIC_API_KEY`。**
`src/api/claude_client.py` 第 439 行只讀 `CLAUDE_API_KEY`；讀不到就靜默進入
fallback 模式（不呼叫 API、不報錯）。設錯名字的症狀是 unparse 直接 100%。

若 `pip install jieba` 在 Debian／Ubuntu 系的 Python 3.11+ 失敗
（`AttributeError: install_layout`，setuptools 與 jieba 舊版 setup.py 不相容），
改用：

```bash
pip download --no-binary :all: --no-deps jieba -d /tmp/jb
tar -xzf /tmp/jb/jieba-*.tar.gz -C /tmp/jb
cp -r /tmp/jb/jieba-*/jieba "$(python3 -c 'import site;print(site.getsitepackages()[0])')/"
```

---

## 執行步驟

### 第一步：先跑一張，不要直接開 20 張

```bash
cd tofu_mvp_model_b-main
python3 eval/sav/sav_runner.py "eval/sav/personas/P01_grok.json" \
    --model claude-haiku-4-5-20251001 \
    --out eval/sav/results
```

約 250 次 API 呼叫，Haiku 約 US$1.3，十幾分鐘。

**看 `unparse` 那一欄。**

輸出長這樣：

```
  ck 10  inferable 3/10   non_inf 1/5   unparse 0   累計呼叫 65
  ck 20  inferable 5/10   non_inf 1/5   unparse 1   累計呼叫 145
```

| unparse | 動作 |
|---|---|
| 0–10% | 正常，繼續第二步 |
| **超過 10%** | **停下來回報，不要繼續跑** |

unparse 代表逗福沒有回一個乾淨的 A/B/C/D。已知風險是：逗福的管線會在題目外包一層 TOFU Identity Prompt 再送給模型，這層包裝可能蓋掉「只回覆一個大寫字母」的指令。dry-run 時（fallback 模式）確實出現這個狀況，但真實 API 下會不會發生沒驗證過。

如果 unparse 偏高，不要自行調整 prompt 硬修，先回報實際的 `raw_response` 內容。

**這一張的分數不要解讀。** 單張卡的 non_inferable 每個測點只有 5 題，答對一題就差 20 個百分點，隨機波動會完全蓋過訊號。第一張只確認兩件事：跑得完、unparse 低。

### 第二步：跑完 20 張

```bash
python3 eval/sav/sav_runner.py "eval/sav/personas/*.json" \
    --model claude-haiku-4-5-20251001 \
    --out eval/sav/results
```

約 5,000 次呼叫，Haiku 約 US$26。

可以分批跑，`results/` 下每張卡一個 `_raw.jsonl`，scorer 會自動合併。

### 第三步：判分

```bash
python3 eval/sav/scorer.py "eval/sav/results/*_raw.jsonl" --by-author
```

不呼叫 API，純程式碼。

---

## 硬規則

**`--model` 是必填，不給會直接報錯退出。**

這是刻意的。同專案的 `eval/longmemeval_real_test.py` 第 228 行預設值是 `claude-opus-4-6`，同樣規模跑 Opus 約 US$5,000。不要沿用那個預設值，也不要幫 `sav_runner.py` 加預設值。

**測驗不得寫進主記憶。**

`run_quiz()` 會把整個 `data` 目錄複製到臨時位置作答，答完刪除。每個測點後有一行 `assert before == after` 檢查端點數沒變。

這個斷言如果觸發，程式會停。**不要把它註解掉。** 測驗一旦污染主記憶，第 20 輪的考題會變成第 30 輪的記憶，跑出來的上升曲線是假的。

---

## 判定標準

scorer 的三個條件，全過才算宣稱成立：

1. inferable 斜率為正
2. 末測點 inferable 與 non_inferable 差距 ≥ 15 個百分點
3. non_inferable 沒有同步上升

門檻可用 `--threshold` 改，但**跑完之後不准改**。事後調門檻讓結果通過，實驗就作廢了。

scorer 另外會列兩張清單：

- **所有測點都答對的 inferable 題** → 疑似答案自帶，該題要重寫
- **所有測點都答錯的 inferable 題** → 疑似 gold 標錯，或推論鏈不成立

---

## 已知限制與觀察點

1. **測不到收斂閘門。** `auto_confirm=True` 會讓流程走單輪、不進多輪補位迴圈（見 `src/main.py` 第 631-633 行 Copilot review #6）。v4.1 收斂閘門在這個測法下不會被觸發。

2. **人物卡是虛構的。** 真人的偏好會有矛盾與漂移，這裡測不到。

3. **兩張卡有結構特徵，保留不改，當作診斷探針**：

   - `P01_gpt.json` 的 I08 — 正解是「查研究條件限制」，而 R23 寫「我會先找研究對象、劑量和限制」。答案幾乎直接寫在記憶裡。**這題是檢驗端點檢索的探針**：逗福如果連這題都答不對，代表基本的記憶檢索有問題。
   - `P02_gemini.json` — 正解落點為 A2 / B5 / C6 / D2，集中在中間兩個字母（其餘 19 張皆為各 3–4）。**這張是檢驗位置偏好的探針**：如果逗福在這張的表現明顯高於其他卡，代表它有選 B/C 的傾向。

   **不要修改這兩張卡。** 它們的特徵是資訊，不是錯誤。跑完後看 scorer 列出的「所有測點皆答對／皆答錯」清單再判斷。

4. **出題方之間有系統性差異。** 規範要求至少 10 條 revealed 是純敘事、不承載線索。Grok 的 5 張確實有 10 條；ChatGPT 與 Gemini 的抽樣顯示每條都在傳遞特質。內容密度過高會讓表現虛高。用 `--by-author` 看四組曲線是否分歧。

---

## 附帶工具

```bash
python3 eval/sav/qc_cards.py "eval/sav/personas/*.json"
```

格式檢查（條數、dim_id 連號、gold 位置分布、選項長度、難度標註）。

它的「疑似答案洩漏」那一段訊噪比不好——用詞袋重疊當指標，會把「自己」「決定」「朋友」這類常用詞算成線索。當參考看，不要當結論。真正判定題目品質的是 scorer 跑完的「所有測點皆答對」清單。
