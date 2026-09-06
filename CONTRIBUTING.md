# Contributing | 參與貢獻

感謝你對逗福Tofu 有興趣。

## 回報問題

到 [Issues](https://github.com/Madison-de-Chao/TOFU/issues) 開票，附上：

1. 你做了什麼（操作步驟）
2. 你預期會發生什麼
3. 實際發生了什麼
4. 錯誤訊息（如果有的話）
5. 你的環境（作業系統、Python 版本）

不方便公開的問題，可來信 [support@momo-chao.com](mailto:support@momo-chao.com)。

## 提交程式碼

1. Fork 這個 repo
2. 建立新分支（`git checkout -b fix/你的修改描述`）
3. 修改並確認測試通過（`python -m pytest tests/ -v`）
4. 提交 PR，說明你改了什麼、為什麼

## 開發環境

```bash
git clone https://github.com/madison-de-chao/tofu.git
cd tofu
python -m pip install -e '.[dev,claude]'
python -m pytest tests/ -v
```

## 設計原則

逗福的設計有幾個不會妥協的原則。提交 PR 前請確認你的修改不違反這些：

- **確認後才記** — 任何寫入使用者記憶的行為都必須經過復述確認，不允許 AI 自己決定記什麼
- **補位而非替代** — 逗福幫使用者想清楚，不替使用者做決定
- **本機優先** — 完整記憶保留本機；API 只傳當次需要的資料，需準確揭露傳輸範圍
- **模型中立** — 不綁定特定 AI 模型，換模型不影響記憶
- **誠實標記** — 事實、推測、建議必須分層標記，不混為一談

## 行為準則

請保持尊重和建設性的溝通。我們歡迎所有善意的參與。

---

有任何問題歡迎開 Issue 討論，或來信 [support@momo-chao.com](mailto:support@momo-chao.com)。
