# 逗福執行與整合指南 v1

適用：0.2.0b1。Python 3.10+。此 repo 是 CLI 與 Python 核心；網站既有線上服務不由本輪部署。

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,claude]'
cp .env.example .env
python -m src --help
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 啟動虛擬環境。若不需要 Claude SDK，可只安裝 `'.[dev]'`。可使用 `tofu`、`python -m src` 或 `python src/main.py` 進入相同 CLI。

`.env.example` 預設 offline。先測流程再編輯 `.env` 選用模型；設定的 model 必須是供應商帳戶可用的 ID。

## 後端設定

| 後端 | 必要設定 | 邊界 |
|---|---|---|
| offline | TOFU_PROVIDER=offline | 規則模板，不呼叫主流程及背景標記 API，不假裝完成 AI 分析 |
| Claude | TOFU_PROVIDER=claude、CLAUDE_API_KEY、TOFU_MODEL | 安裝 claude extra；可使用 Anthropic 專用 Batch runner |
| OpenAI 相容 | TOFU_PROVIDER=openai、OPENAI_API_KEY、TOFU_MODEL | 同步文字 Chat Completions；不支援 tools / streaming / Anthropic batch |
| 本機相容伺服器 | 同上，OPENAI_BASE_URL=http://127.0.0.1:PORT/v1 | 若伺服器不驗 key，依其文件設定非空 placeholder；TOFU 不自動探測端點 |

OpenAI 相容 API 預設送 `max_completion_tokens`；供應商若要求舊欄位，設 `TOFU_OPENAI_TOKEN_PARAMETER=max_tokens`。非本機 URL 必須 HTTPS；拒絕含帳密、query、fragment 的 base URL，不跟隨 redirect。每次 transport 最多三次嘗試；401 等非暫時錯誤立即失敗；429/5xx 有有限退避。錯誤訊息不回印 key、輸入或供應商 response body。

API 相容性依據：[OpenAI Chat Completions 參考](https://developers.openai.com/api/reference/resources/chat)、[Python urllib.request](https://docs.python.org/3/library/urllib.request.html)。此為 transport 契約來源，不是模型效能驗證。

## 五模式操作例

```text
我想辦一場 30 人的品牌活動
/free 幫我列出活動報到的三個步驟
/risk 評估戶外活動遇到下雨的備案風險
/propose 為 30 人品牌活動製作執行提案
/check 我收到一則活動免費抽獎訊息，要求先付手續費
```

default 會復述並等待確認；補位最多五輪。/free、/risk 直接處理原輸入；/propose 自問自答後交卷，最後生成最多三次（首次＋兩次重試）。/check 第一階段後輸入 1 查看完整分析、2 生成警示文字稿、3 結束。

`/check` 的分析是模型判讀與格式審核，沒有內建外部搜尋，不能宣稱連網查證成功。輸出中標示來源未獨立查證。

## 本機資料與案件隔離

使用環境變數 `TOFU_DATA_DIR` 為每個人／專案指定獨立目錄；啟動前設定。個別路徑變數仍優先。不要讓多名使用者共用同一目錄；此 CLI 沒有帳號或租戶驗證層。

| 資料 | 檔案／目錄 | 用途 |
|---|---|---|
| 互動端點 | endpoints.jsonl | 原話、確認、結果、品質與查表 audit |
| 畫像 | user_profile.json | 長期偏好、風格、反例 |
| 跨輪 Zone | zone_history.jsonl | signature / Zone 一致性 |
| 查核中繼 | sessions/ | /check 分階段狀態，24 小時 TTL |
| 詞索引 | words/ | 可選 Claude 詞性標記 |

完整資料存在本機；線上生成會送出當次原話、確認內容、選取歷史與策略摘要。常規歷史注入受 30 筆／9,000 字元限制；評測用 `answer_with_profile` 明確接收完整 haystack，不享有相同預算。`/check` 查表只留 metadata，不把私人歷史注入內容分析。OpenAI／offline 模式不使用背景 Claude 標記。

備份時結束所有寫入程序，再複製整個資料目錄。`/export` 輸出 Markdown，可能包含私人內容。`/reset` 只清除端點，不清獨立畫像／歷史；要完整清除，由使用者在停止程式後刪除指定資料目錄。資料不會因換模型而搬移。

## 舊記憶裁決

```text
/memory
/memory confirm <完整端點ID>
/memory supersede <完整端點ID>
```

過期記憶可降權參考，但檢索命中不代表人類確認。confirm 恢復 active，supersede 停用檢索；原文、起終端點及 status_history 仍保留。舊程式直接呼叫 mark_hit 預設仍相容舊行為，新整合請使用 `reactivate_pending=False`。

## 程式整合

```python
from src.api.factory import create_client
client = create_client("offline")
restate = client.generate_restate("規劃品牌活動", {}, ["budget", "scope"])
print(restate["restate_text"])
```

`BaseLLMClient`／factory 只負責模型層，**直接呼叫 execute_task 不代表已跑完治理流程**。完整單使用者 CLI 入口是 `src.main`；現有 run_one_interaction 支援程式呼叫，但 default 會讀取 stdin。不要將 `auto_confirm=True` 用在需要人類確認的正式服務中。多使用者 HTTP/session API 仍列後續工作，不能用 process 全域 I/O monkeypatch 架成網頁服務。

## 檢查

```bash
python -m pytest -q
python -m build
python -m src --version
```

測試使用合成資料與 mock，不消耗付費模型額度。CI 設 Python 3.10/3.12/3.13；本輪環境實測版本及結果見 VALIDATION_v1。成本護欄是既有固定每次呼叫估值，不是供應商硬性金額上限，正式額度仍應以供應商設定及帳單為準。
