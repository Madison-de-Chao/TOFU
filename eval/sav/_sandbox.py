#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval 專用隔離設定。必須在 `from src.main import ...` 之前 import。

做兩件事，都是為了堵「繞過 runner 的 client 直接打 API / 寫真資料」的路：

1. 把 data/words/ 導向臨時目錄。
   WORDS_DIR 在 src.main import 時就定案（main.py 讀 TOFU_WORDS_DIR），
   不先設環境變數，詞性標記會寫進真專案的 data/words/，
   考題詞彙就混進主詞表——assert 只看端點數，抓不到這條。

2. 把詞性標記器換成離線實例。
   HaikuWordTagger 自己讀 CLAUDE_API_KEY（word_tagger.py），完全不經過
   runner 傳入的 client。不關掉的話，--dry-run 在設了 key 的機器上
   仍會每輪互動打一次真實 Haiku。eval 用不到詞表：詞表在
   run_one_interaction 裡只寫不讀，不進 prompt、不進檢索，
   關掉不影響測驗結果。
"""
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 1. words 目錄導向臨時位置——必須在 import src.main 之前生效
os.environ.setdefault("TOFU_WORDS_DIR", tempfile.mkdtemp(prefix="sav_words_"))

import src.main as _tofu_main  # noqa: E402
from src.middleware.word_tagger import HaikuWordTagger  # noqa: E402

# 2. 預先塞入離線 tagger；_get_word_pipeline 的懶初始化看到非 None 不會重建
_offline_tagger = HaikuWordTagger(api_key="__EVAL_OFFLINE__")
_offline_tagger._offline = True
_offline_tagger._client = None
_tofu_main._word_tagger = _offline_tagger
