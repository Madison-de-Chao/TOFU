# 逗福Tofu 公測版：開新 GitHub Repo 操作指南

**日期：** 2026-05-23
**目標：** 建立乾淨的 public repo，舊 repo 保留為 private 內部開發用

---

## 第一步：在 GitHub 上建新 Repo

1. 到 github.com，右上角點 **+** → **New repository**
2. 填寫：

| 欄位 | 填什麼 |
|------|--------|
| Repository name | `tofu` |
| Description | `認知中間層——AI 回答前先確認你問對了沒。Cognitive middleware that validates your question before AI answers it. Open source, US$0.01/interaction.` |
| Public / Private | **Public** |
| Add a README | **不勾** |
| Add .gitignore | **不勾** |
| Choose a license | **不勾** |

3. 點 **Create repository**
4. 建完後 GitHub 會顯示一組指令，先不動，往下看

---

## 第二步：本機準備乾淨資料夾

打開終端機（Mac: Terminal / Windows: PowerShell），執行：

```bash
# 建立新資料夾
mkdir tofu
cd tofu
```

---

## 第三步：從舊 repo 複製程式碼

只複製需要公開的檔案。**不要直接複製整個舊資料夾。**

```bash
# 假設舊 repo 在 ~/tofu_mvp_model_b（改成你實際的路徑）

# 複製程式碼
cp -r ~/tofu_mvp_model_b/src ./src
cp -r ~/tofu_mvp_model_b/tests ./tests
cp ~/tofu_mvp_model_b/requirements.txt ./requirements.txt
```

Windows PowerShell 版本：
```powershell
# 假設舊 repo 在 C:\Users\你的名字\tofu_mvp_model_b

Copy-Item -Recurse ~\tofu_mvp_model_b\src .\src
Copy-Item -Recurse ~\tofu_mvp_model_b\tests .\tests
Copy-Item ~\tofu_mvp_model_b\requirements.txt .\requirements.txt
```

### 也可能需要複製的（看你 repo 裡有什麼）：

```bash
# 如果有 data 資料夾的結構定義（但不要複製個人資料）
mkdir -p data

# 如果有其他必要的設定檔
cp ~/tofu_mvp_model_b/setup.py ./setup.py          # 如果有的話
cp ~/tofu_mvp_model_b/pyproject.toml ./pyproject.toml  # 如果有的話
```

---

## 第四步：放入公測版文件

把從 Claude 下載的那組檔案放進來：

```bash
# 把下載的檔案複製進 tofu 資料夾
# （改成你下載檔案的實際路徑）

cp ~/Downloads/README.md ./README.md
cp ~/Downloads/LICENSE ./LICENSE
cp ~/Downloads/ROADMAP.md ./ROADMAP.md
cp ~/Downloads/CONTRIBUTING.md ./CONTRIBUTING.md
cp ~/Downloads/CHANGELOG.md ./CHANGELOG.md
cp ~/Downloads/gitignore_template ./.gitignore
```

Windows PowerShell：
```powershell
Copy-Item ~\Downloads\README.md .\README.md
Copy-Item ~\Downloads\LICENSE .\LICENSE
Copy-Item ~\Downloads\ROADMAP.md .\ROADMAP.md
Copy-Item ~\Downloads\CONTRIBUTING.md .\CONTRIBUTING.md
Copy-Item ~\Downloads\CHANGELOG.md .\CHANGELOG.md
Copy-Item ~\Downloads\gitignore_template .\.gitignore
```

### 建立 data 資料夾佔位檔

使用者 clone 下來需要 `data/` 資料夾存在：

```bash
mkdir -p data
echo "# 逗福Tofu 資料目錄" > data/README.md
echo "端點紀錄與使用者畫像會存在這裡。這些資料只存在你的本機，不上傳。" >> data/README.md
```

---

## 第五步：安全檢查（重要）

在推送之前，確認沒有敏感內容混進來。

### 5.1 搜尋 API key

```bash
# Mac/Linux
grep -r "sk-ant-" .
grep -r "CLAUDE_API_KEY" . --include="*.py" --include="*.json"

# Windows PowerShell
Select-String -Path .\* -Pattern "sk-ant-" -Recurse
```

如果有找到任何結果，把那個 key 刪掉再繼續。

### 5.2 確認沒有個人端點資料

```bash
# 這些檔案不應該存在
ls data/endpoints.json      # 應該顯示「找不到」
ls data/user_profile.json   # 應該顯示「找不到」
```

### 5.3 確認沒有內部文件

```bash
# 搜尋不該出現在公開版的內容
grep -r "待MOMO決定" . --include="*.md"
grep -r "狀態快照" . --include="*.md"
grep -r "交接文件" . --include="*.md"
grep -r "頭腦資料庫" . --include="*.md"
```

以上搜尋應該全部零結果。有找到的話，把那個檔案刪掉或移走。

---

## 第六步：確認 README 裡的連結

打開 `README.md`，搜尋並修改：

1. `momo-chao/tofu_mvp_model_b` → 改成你的 **GitHub 帳號名/tofu**
   - 例如：`你的帳號/tofu`
   - 總共出現 1 次（安裝步驟的 git clone）

2. 確認 `yyuniverse.com` 可以打開

---

## 第七步：初始化 Git 並推送

```bash
# 初始化
git init
git add .

# 確認一下要提交的檔案清單
git status
```

看一下 `git status` 的列表：
- ✅ 應該看到：README.md, LICENSE, ROADMAP.md, CONTRIBUTING.md, CHANGELOG.md, .gitignore, src/, tests/, requirements.txt, data/README.md
- ❌ 不應該看到：endpoints.json, user_profile.json, 任何 `*_交接*.md`, `*_開發紀錄*.md`

確認沒問題後：

```bash
# 提交
git commit -m "v0.1.0-beta: 逗福Tofu 公測版首次公開發布"

# 連接到 GitHub（把 URL 換成你建的 repo）
git remote add origin https://github.com/你的帳號/tofu.git
git branch -M main
git push -u origin main
```

---

## 第八步：在 GitHub 上加 Topics 標籤

1. 到 repo 頁面
2. 點右邊齒輪圖示（About 旁邊）
3. 在 Topics 加上：

```
ai, cognitive-middleware, llm, memory-layer, prompt-engineering,
chinese-nlp, open-source, claude, haiku, question-validation
```

---

## 第九步：建立 Release

1. 到 repo 頁面，點右邊 **Releases**
2. 點 **Create a new release**
3. 填寫：

| 欄位 | 填什麼 |
|------|--------|
| Tag version | `v0.1.0-beta` |
| Target | `main` |
| Release title | `逗福Tofu v0.1.0 — 技術預覽版 Public Beta` |

4. 描述欄貼上：

```
## 逗福Tofu 首次公開發布

認知中間層。問對問題，才有對的答案。

### 核心功能
- 復述確認 — AI 回答前先確認理解正確
- 七維度補位提問 — 帶出你沒想到的問題
- 確認後才記 — 你同意才寫進記憶
- 品質三層標記 — 事實／推測／建議自動分類
- 安全趨勢偵測阻斷

### 數據
- 608 項測試通過
- 244 筆實測互動，零錯誤
- 每次互動約 US$0.01（推薦 Claude Haiku）
- 最便宜的模型 + 逗福，打贏最貴的旗艦模型（58% vs 53%，便宜 194 倍）

### 安裝
見 README.md

### 已知限制
技術預覽版。介面為 CLI，桌面版和網頁版製作中。
詳見 README.md「已知限制」章節。

---
免費開源 | MIT License
創辦人：趙偉辰（默默超 MoMo Chao）
yyuniverse.com
```

5. 勾選 **Set as a pre-release**
6. 點 **Publish release**

---

## 第十步：最終確認

- [ ] 用無痕模式（或手機）打開 repo URL，確認看到新 README
- [ ] 點 LICENSE 確認可以打開
- [ ] 點 data/ 資料夾確認只有 README.md，沒有個人資料
- [ ] Release 頁面有 v0.1.0-beta
- [ ] 搜尋 `sk-ant-` 零結果

### 可選：用另一台電腦測試完整流程

```bash
git clone https://github.com/你的帳號/tofu.git
cd tofu
pip install -r requirements.txt
python -m pytest tests/ -v
# 預期：608 項全過
```

---

## 舊 Repo 怎麼處理

舊的 `tofu_mvp_model_b` repo 不用動。保持 private，繼續當內部開發用。
所有開發紀錄、PR 歷史、交接文件都還在那邊。

新 repo `tofu` = 對外的乾淨門面。
舊 repo `tofu_mvp_model_b` = 內部的完整紀錄。

---

*操作過程中遇到任何問題，回來這個對話問我。*
