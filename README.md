# 逗福Tofu — 認知中間層 | Cognitive Middleware

> **問對問題，才有對的答案。**
> Ask the right question before getting the right answer.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-608%20passing-brightgreen)]()
[![Cost](https://img.shields.io/badge/cost-US%240.01%2Finteraction-blue)]()

---

## 這是什麼？

逗福Tofu 是一個坐在你和 AI 之間的認知中間層。

你跟 AI 說話之前，逗福先做三件事：
1. **復述確認** — 用自己的話重講一遍你的需求，確認聽懂了
2. **補位提問** — 問你一個你沒想到但很重要的問題
3. **確認後才記** — 你點頭了才寫進記憶，不是 AI 自己決定記什麼

用越久，逗福越懂你的思考盲區。第三次忘了考慮動線？它會主動提醒。

## What is this?

Tofu is a cognitive middleware layer that sits between you and your AI. Before the AI answers, Tofu:
1. **Restates** your request in its own words to confirm understanding
2. **Fills gaps** by asking questions you didn't think to ask
3. **Confirms before recording** — nothing is memorized until you approve

---

## 為什麼需要它？

大多數錯誤的決定，不是因為 AI 的答案差。是因為一開始就解錯了題。

你說「我想辦派對」，AI 立刻列出場地清單。
逗福先問：「這場派對是辦給誰的？公司形象還是朋友聚會？」
因為這兩種派對從頭到尾做法不一樣。

每一個 AI 都在搶著回答。沒有一個先停下來問你。逗福做的就是這個停頓。

---

## 核心數據

| 項目 | 數字 |
|------|------|
| 實測互動 | 244 筆，零錯誤 |
| 總費用 | US$2.57（約 NT$80） |
| 每次互動成本 | 約 US$0.01（約 NT$0.3） |
| 測試通過 | 608 項 |
| 推薦模型 | Claude Haiku（最便宜） |

### 小模型打贏大模型

同題測試 19 題，Haiku（US$0.01/次）+ 逗福的補位品質 **超過** Opus（US$2.05/次）。

- Haiku Level 3 補位率：58%
- Opus Level 3 補位率：53%
- 費用差距：**194 倍**

我們把逗福 + Haiku 的輸出拿給 10 個 AI 盲測。8 個判定是旗艦級品質。0 個猜到用的是最便宜的模型。

---

## 跟其他工具的差異

| | 一般 AI 記憶工具 | 逗福Tofu |
|---|---|---|
| 寫入確認 | AI 自己決定記什麼 | 復述確認，你同意才記 |
| 補位能力 | 無 | 七維度主動補位 |
| 品質標記 | 無 | 事實／推測／建議 三層標記 |
| 儲存趨勢 | 越用越多 O(n) | 收斂（只存起點和終點） |
| 跨模型 | 通常綁定單一 AI | 不綁定，換模型不丟記憶 |
| 中文優化 | 通常沒有 | jieba 中文分詞 + 中文停用詞 |

---

## 安裝

### 系統需求

- Python 3.10+
- 網路連線（API 呼叫）
- 硬碟 < 50MB

### 步驟

```bash
# 1. 下載
git clone https://github.com/madison-de-chao/tofu.git
cd tofu

# 2. 安裝套件
pip install -r requirements.txt
pip install jieba  # 選裝，中文分詞品質更好

# 3. 設定 API Key（推薦 Claude Haiku）
# 到 console.anthropic.com 取得 API key
export CLAUDE_API_KEY=sk-ant-你的key

# 4. 啟動
python src/main.py
```

沒有 git？到 GitHub 頁面點 **Code → Download ZIP**，解壓縮後進入資料夾。

不設定 API key 也能用——逗福會自動進入離線模式，核心功能（復述確認、補位提問、端點紀錄）照常運作，但不呼叫 AI。

### 驗證安裝

```bash
python -m pytest tests/ -v
# 預期：608 項測試全部通過
```

---

## 使用方式

```
你：我想去日本玩一個禮拜。

[逗福Tofu 復述]
你想安排一趟一週的日本旅行。
有沒有同行的人？年紀多大？這一週包含來回飛機時間嗎？

你：跟爸媽一起，70 歲，包含飛機。

[逗福Tofu 已確認]
目標：日本家族旅行
同行：父母（70 歲）
實際可用天數：約 5 天
注意：行程需適合年長者步調

（開始執行任務）
```

### 指令

| 指令 | 功能 |
|------|------|
| `/help` | 所有指令清單 |
| `/profile` | 逗福目前對你的理解 |
| `/baseline` | 你的高頻目標與常見盲區 |
| `/stats` | 互動數、修正率、品質分布 |
| `/history` | 最近 5 筆互動摘要 |
| `/export` | 端點紀錄匯出為 Markdown |
| `/reset` | 清除所有紀錄（需二次確認） |

---

## 技術架構

```
使用者輸入
    │
    ▼
┌──────────────┐
│  逗福Tofu    │ ← 認知中間層
│              │
│  1. 復述確認  │    用自己的話重講，確認理解正確
│  2. 補位提問  │    七維度檢查你漏了什麼
│  3. 端點紀錄  │    只存起點和終點，不存過程
│  4. 品質標記  │    事實／推測／建議 三層分類
│  5. 安全阻斷  │    趨勢偵測，不是關鍵字過濾
│              │
└──────┬───────┘
       │
       ▼
   LLM（任意模型）
       │
       ▼
   回覆給使用者
```

逗福不取代 AI，它確保問題在生成答案之前被正確建構。

記憶儲存在你本機的 `data/` 資料夾，不上傳。換電腦只要複製 `data/` 資料夾。換 AI 模型不丟記憶。

---

## 已知限制

這是技術預覽版（Public Beta）。以下是目前已知的限制，我們選擇公開而不是隱藏：

- **偏好提取率偏低**：244 筆互動提取到 45 個偏好（18.4%），隱式偏好是結構性難題
- **偏好清單可能有重複條目**：去重功能開發中
- **長記憶截斷**：端點超過 250 筆時，早期的具體細節會被壓縮
- **介面為 CLI**：目前是命令列操作，桌面版和網頁版製作中
- **英文停用詞覆蓋不完整**：英文場景下有噪音
- **未經大規模使用者驗證**：目前數據來自單一開發者的密集測試

---

## Roadmap

詳見 [ROADMAP.md](ROADMAP.md)

近期：
- [ ] 桌面版（一鍵安裝，不需要 Python）
- [ ] 網頁版（瀏覽器直接使用）
- [ ] 偏好去重與結構化
- [ ] 補位提問去重優化

中期：
- [ ] 多語系支援強化
- [ ] 使用者畫像匯入（從其他 AI 帶過來）
- [ ] 螺旋上升效果驗證（Round 2/3）

---

## 這是誰做的？

逗福Tofu 的創辦人[趙偉辰（默默超 MoMo Chao）](https://yyuniverse.com)不是工程師。他是行銷人。

他設計的不是程式碼，是一套思考框架。程式碼只是讓框架跑起來的工具。

逗福的核心發現：**決定品質的是問對問題的流程，不是模型的大小。** 最便宜的模型加上正確的框架，打贏最貴的旗艦模型。

## 體系歸屬

逗福Tofu 是[元壹宇宙 YuanYi Universe](https://yyuniverse.com) 的第一個工程化應用。

品牌生態：超烜創意 Maison de Chao → 元壹宇宙 YuanYi Universe → 虹靈御所 Rainbow Sanctuary → 逗福Tofu

工具免費，方法論來自元壹宇宙。

---

## 授權

[MIT License](LICENSE) — 自由使用、修改、散布。

---

## 回報問題

遇到問題請到 [Issues](../../issues) 回報。附上：
1. 你做了什麼
2. 預期行為
3. 實際行為
4. 錯誤訊息（如果有）

---

*逗福Tofu。問對問題，才有對的答案。*
*免費開源。*
*© 2024-2026 趙偉辰 / 超烜創意 Maison de Chao*
