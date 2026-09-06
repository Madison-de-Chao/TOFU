# 白皮書對齊工程驗證紀錄 v1

日期：2026-09-06。套件版本：0.2.0b1。起始 commit：a5917f6。環境：Linux、Python 3.12。

## 基線

- 未裝依賴的 unittest：682 個 tests、5 個 skipped。這只能驗證 fallback 分支，不可當成完整依賴驗收。
- 安裝依賴後的原始 pytest：678 passed、4 failed。4 個失敗來自 prompt-only fixture 初始化 Anthropic SDK，觸發本環境 SOCKS proxy 缺少 socksio；並非真的呼叫模型失敗。
- 修正 fixture 為不建立網路 SDK，保留所有原話傳遞與 prompt 斷言；來源測試同時更正「裸年份即來源」的錯誤期望。

## 最終結果

`python -m pytest tests/ -q`：**707 passed，18 subtests passed**，約 4 秒。

- 新增 25 個白皮書契約測試：Chat Completions 請求／回應、模型必填、認證錯誤、429 重試、連線重試上限、錯誤隱私、fallback、來源逐行隔離、反駁條件缺值、記憶預算、confirmed-write、check 隔離、偏好去重、The Killers 名稱誤判與 fenced code 保真。
- 既有 682 個測試繼續執行；未以刪除舊測試或略過不通過項目獲得綠燈。
- `python -m build` 成功產生 sdist / wheel；editable 安裝與命令列入口可執行。
- wheel 解壓至乾淨目錄、在 checkout 外載入實際 wheel 內的 src 模組與大詞典，CLI --version 成功。此驗證不依賴 editable checkout。
- 隔離資料目錄的 CLI subprocess：free / risk / propose / check 四模式全部寫入完成端點，皆包含 delivery_audit，exit code 0，無「[錯誤]」訊息。default 的復述／確認／取消仍由既有整合測試涵蓋。
- `--help`、`--version` 回傳 0；未知參數回傳 2。
- `git diff --check` 通過。

## 證據分層與限制

這是本輪本機實測的軟體行為紀錄。外部 API transport 使用 mock 回應，沒有使用 MOMO 的真實 API key、沒有消耗付費模型額度、沒有上傳使用者記憶。

通過單元測試不代表白皮書 244 筆／500 題／19 題模型比較已重現。不能據此宣稱零幻覺、醫療法律財務可用、情緒診斷有效、完整語意風險辨識或「越用越懂」的效益已驗證。

GitHub Actions 提供 Python 3.10 / 3.12 / 3.13 的遠端驗證；本機只有 3.12 實測。遠端結果以 PR checks 為準，不把配置了 CI 當成 CI 已成功。

## 仍需完成的驗收

1. 取得正式規格裁決：情緒兩種三軸、完整 CIP-X / 元動機方向／六步與八階 I/O、來源查證服務。
2. 以明確模型 ID、費用上限、合成題組與保存原始 log 的方式跑 Round 4 及跨模型實測。
3. 由非本輪作者的外部評估者驗證輸出與實際用戶價值。模型自評與本輪回歸測試不替代此步驟。
