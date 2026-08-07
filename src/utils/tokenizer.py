"""統一斷詞模組。

全專案的中文斷詞都從這裡走，不在各模組自己切。
v2.0 P0-1：使用 jieba 精確模式取代 2-gram。

jieba 未安裝時不會讓整個程式崩潰——降級回 2-gram 並印警告。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# 外來語與特殊複合詞表：jieba 預設詞典會拆錯的常用詞。
# 邏輯鏈 v0.3 第二層規則：外來語保留常用譯名不拆字。
# 只收錄 jieba 會拆錯的詞。jieba 已認識的詞不重複加。
# 資料來源：CASE_07~11 實際對話 + 維基百科外來語 + 教育部外來詞附錄。
_LOANWORDS = [
    # ════════════════════════════════════════
    # 一、食物飲料（CASE 對話高頻出現）
    # ════════════════════════════════════════
    "派對",         # party             → ['派','對']
    "拿鐵",         # latte             → ['拿','鐵']
    "味噌",         # miso              → ['味','噌']
    "拉麵",         # ramen             → ['拉','麵']
    "抹茶",         # matcha            → ['抹','茶']
    "鬆餅",         # waffle/pancake    → ['鬆','餅']
    "泡芙",         # puff              → ['泡','芙']
    "馬芬",         # muffin            → ['馬','芬']
    "塔可",         # taco              → ['塔','可']
    "燉飯",         # risotto           → ['燉','飯']
    "焗烤",         # gratin            → ['焗','烤']
    "酪梨",         # avocado           → ['酪','梨']    ← CASE 出現 5 次
    "榴槤",         # durian            → ['榴','槤']
    "萊姆",         # lime              → ['萊','姆']    ← CASE 出現 3 次
    "羅勒",         # basil             → ['羅','勒']
    "薑黃",         # turmeric          → ['薑','黃']
    "馬卡龍",       # macaron           → ['馬','卡龍']
    "馬丁尼",       # martini           → ['馬','丁尼']
    "天婦羅",       # tempura           → ['天婦','羅']
    "牛軋糖",       # nougat            → ['牛軋','糖']
    "太妃糖",       # toffee            → ['太妃','糖']
    "卡士達",       # custard           → ['卡士','達']
    "覆盆莓",       # raspberry         → ['覆盆','莓']
    "蘋果塔",       # apple tart        → ['蘋果','塔']
    "肉桂捲",       # cinnamon roll     → ['肉桂','捲']
    "玉米餅",       # tortilla          → ['玉米','餅']
    "甜辣醬",       # sweet chili sauce → ['甜辣','醬']
    "酸奶油",       # sour cream        → ['酸奶','油']  ← CASE 出現
    "杏仁奶",       # almond milk       → ['杏仁','奶']
    "氣泡水",       # sparkling water   → ['氣','泡水']
    "提拉米蘇",     # tiramisu          → ['提拉','米','蘇']
    "卡布奇諾",     # cappuccino        → ['卡布','奇諾']
    "焦糖瑪奇朵",   # caramel macchiato → ['焦糖','瑪奇朵']
    "莫札瑞拉",     # mozzarella        → ['莫札','瑞拉']
    "馬斯卡彭",     # mascarpone        → ['馬','斯卡','彭']
    "莫西多",       # mojito            → ['莫西','多']
    "血腥瑪麗",     # bloody mary       → ['血腥','瑪麗']
    "法拉費",       # falafel           → ['法拉','費']   ← CASE 出現
    "鷹嘴豆",       # chickpea          → ['鷹','嘴豆']   ← CASE 出現 18 次
    "羽衣甘藍",     # kale              → ['羽衣','甘藍'] ← CASE 出現
    "拖鞋麵包",     # ciabatta          → ['拖鞋','麵','包']
    "班乃迪克蛋",   # eggs benedict     → ['班','乃','迪克','蛋']
    # ════════════════════════════════════════
    # 二、科技工程
    # ════════════════════════════════════════
    "馬達",         # motor             → ['馬','達']
    "部落格",       # blog              → ['部落','格']
    "區塊鏈",       # blockchain        → ['區塊','鏈']  ← CASE 出現 2 次
    # ════════════════════════════════════════
    # 三、生活用品服飾
    # ════════════════════════════════════════
    "蕾絲",         # lace              → ['蕾','絲']
    "模特兒",       # model             → ['模特','兒']
    # ════════════════════════════════════════
    # 四、醫學科學
    # ════════════════════════════════════════
    "嗎啡",         # morphine          → ['嗎','啡']
    "維他命",       # vitamin           → ['維他','命']
    "阿斯匹靈",     # aspirin           → ['阿斯','匹靈']
    "盤尼西林",     # penicillin        → ['盤尼','西林']
    "阿摩尼亞",     # ammonia           → ['阿摩','尼亞']
    # ════════════════════════════════════════
    # 五、運動文化娛樂
    # ════════════════════════════════════════
    "馬拉松",       # marathon          → ['馬','拉松']
    "安可",         # encore            → ['安','可']
    "卡拉OK",       # karaoke           → ['卡拉','OK']
    "皮拉提斯",     # pilates           → ['皮拉','提斯'] ← CASE 出現
    "佛朗明哥",     # flamenco          → ['佛朗','明哥'] ← CASE 出現
    "脫口秀",       # talk show         → ['脫口','秀']   ← CASE 出現 3 次
    # ════════════════════════════════════════
    # 六、地名（CASE 對話實際出現）
    # ════════════════════════════════════════
    "義大利",       # Italy             → ['義','大利']   ← CASE 出現 14 次
    "好市多",       # Costco            → ['好市','多']
    "巴塞隆納",     # Barcelona         → ['巴塞隆','納'] ← CASE 出現
    "布達佩斯",     # Budapest          → ['布達','佩斯']
    "馬爾地夫",     # Maldives          → ['馬','爾','地夫']
    "聖托里尼",     # Santorini         → ['聖托','里尼']
    "布巴內什瓦",   # Bhubaneswar       → ['布巴','內什瓦'] ← CASE 出現 7 次
    "阿姆斯特丹",   # Amsterdam         → ['阿姆斯特','丹'] ← CASE 出現 5 次
    "峇里島",       # Bali              → ['峇','里島']
    "普吉島",       # Phuket            → ['普吉','島']
    # ════════════════════════════════════════
    # 七、日語借詞 + 度量衡
    # ════════════════════════════════════════
    "歐巴桑",       # obasan            → ['歐','巴桑']
    "宅急便",       # takkyubin         → ['宅急','便']
    "加侖",         # gallon            → ['加','侖']
]

# 繁中大詞典：jieba 內建 dict.txt 以簡體語料為主，直接切繁體會產生
# 「這禮/拜」「電費帳/單寄來」這類跨詞界碎片，污染興趣領域統計，
# 也拉低 top-k 檢索的詞彙重疊率。repo 內建 jieba 官方 dict.txt.big
# （繁簡合一，MIT 授權），存在即自動啟用。
#
# 環境變數 TOFU_JIEBA_DICT 可覆寫：
#   - 指向其他詞典檔路徑 → 用該詞典
#   - 設為 "default"      → 強制用 jieba 內建詞典（跳過 dict.txt.big）
#
# 成本：首次切詞多 ~2-3 秒建構前綴樹（jieba 懶載入 + 有磁碟快取），
# 之後無差異。
_BIG_DICT_PATH = Path(__file__).resolve().parents[2] / "resources" / "jieba" / "dict.txt.big"

# jieba 可選載入：未安裝時降級回 2-gram
_jieba = None
try:
    import jieba
    _jieba = jieba
    _dict_override = os.environ.get("TOFU_JIEBA_DICT", "")
    if _dict_override and _dict_override.lower() != "default":
        # 覆寫路徑不存在時不能讓 import 直接炸掉（set_dictionary 會拋例外，
        # 連帶所有依賴 tokenizer 的模組都無法載入）——警告後回退內建詞典。
        if os.path.isfile(_dict_override):
            _jieba.set_dictionary(_dict_override)
        else:
            print(
                f"[逗福Tofu 警告] TOFU_JIEBA_DICT 指向的詞典不存在："
                f"{_dict_override}，回退使用內建詞典。",
                file=sys.stderr,
                flush=True,
            )
            if _BIG_DICT_PATH.is_file():
                _jieba.set_dictionary(str(_BIG_DICT_PATH))
    elif not _dict_override and _BIG_DICT_PATH.is_file():
        _jieba.set_dictionary(str(_BIG_DICT_PATH))
    # 把外來語表註冊進 jieba，避免被切碎（邏輯鏈 v0.3 第二層規則）
    # add_word 必須在 set_dictionary 之後——換詞典會重置使用者詞表。
    # tag="n"：不給詞性的話 posseg 會標成 'x'，tokenize_nouns 的名詞性
    # 過濾就會把大詞典裡沒有的外來語（拿鐵、拉麵…）整批誤殺。
    # 外來語表全為名詞，統一標 n 是安全的。
    for _word in _LOANWORDS:
        _jieba.add_word(_word, tag="n")
except ImportError:
    print(
        "[逗福Tofu 警告] jieba 未安裝，斷詞將使用 2-gram 降級模式。"
        " 建議執行 pip install jieba 以獲得更好的斷詞效果。",
        file=sys.stderr,
        flush=True,
    )

# 停用詞表：中文部分保留原版，英文部分擴充至 197 個。
# 英文 stopwords 依據 CASE_07~11 1,220 條英文對話統計，補齊代名詞/介係詞/
# 連接詞/助動詞/副詞/縮寫殘留等高頻功能詞。
STOPWORDS = {
    # ── 中文停用詞（逐字保留原版）──
    "的", "了", "和", "與", "或", "是", "在", "我", "你", "他", "她", "它",
    "們", "這", "那", "有", "沒", "要", "想", "會", "能", "可以", "以", "也",
    "都", "就", "還", "很", "太", "更", "最", "把", "被", "讓", "給", "從", "到",
    "一", "個", "些", "種", "次", "裡", "中", "上", "下", "對", "為", "因", "為了",
    "所以", "但是", "但", "而", "以及", "一些", "一個", "一下", "下次", "這次",
    # ── 英文停用詞（197 個，按類別排列）──
    # 代名詞（主格/受格/所有格/反身）
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # 疑問/關係代名詞
    "who", "whom", "whose", "which", "what",
    # 指示代名詞
    "this", "that", "these", "those",
    # 冠詞
    "a", "an", "the",
    # 介係詞
    "in", "on", "at", "to", "for", "with", "by", "from", "of",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out", "up", "down", "off",
    "against", "among", "around", "along", "across", "behind", "beyond",
    "within", "without", "upon", "toward", "towards",
    # 連接詞
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    "if", "then", "than", "because", "since", "while", "although", "though",
    "unless", "until", "when", "where", "whether",
    # be 動詞
    "is", "am", "are", "was", "were", "be", "been", "being",
    # 助動詞
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    # 高頻無意義副詞
    "not", "no", "very", "really", "just", "also", "too", "already",
    "still", "even", "only", "ever", "never", "always", "often",
    "here", "there", "now", "again", "once",
    "much", "more", "most", "less", "least",
    "well", "quite", "rather", "pretty", "enough",
    # 限定詞/不定代名詞
    "all", "any", "some", "each", "every", "few", "many",
    "other", "another", "such", "own",
    # 功能性疑問副詞/連接副詞
    "how", "why", "as", "like",
    # 高頻輕動詞（語意由後面的名詞決定，本身不具區分力）
    "get", "got", "getting", "let",
    # 禮貌用語
    "yes", "yeah", "ok", "okay", "please", "thanks", "thank",
    # 縮寫殘留（jieba 切完 don't → ["don", "t"]，t 被 min_len 濾掉，don 留下）
    "ve", "ll", "re",
    "don", "doesn", "didn", "won", "wouldn",
    "couldn", "shouldn", "isn", "aren", "wasn", "weren", "hasn", "hadn",
}

_CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ENGLISH_RUN_RE = re.compile(r"[a-zA-Z]+")
_DIGITS_RE = re.compile(r"^\d+$")
_PUNCT_RE = re.compile(r"^[\s\W]+$")


def _should_keep_token(tok: str, min_token_len: int = 2) -> bool:
    """判斷 token 是否應保留。

    過濾純數字、純標點空白、長度不足、停用詞。
    min_token_len 預設 2（與原行為一致）；傳入 1 可保留單字 token。
    """
    tok = tok.strip()
    if not tok:
        return False
    if _DIGITS_RE.match(tok):
        return False
    if _PUNCT_RE.match(tok):
        return False
    if len(tok) < min_token_len:
        return False
    if tok in STOPWORDS:
        return False
    return True


def _tokenize_lightweight(text: str) -> list[str]:
    """輕量斷詞：中文拆單字、英文保留完整單詞。

    不做 jieba 斷詞；用於 jieba 不可用且 min_token_len < 2 的場景。
    """
    tokens: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text):
        tok = match.group(0)
        if _CHINESE_RUN_RE.fullmatch(tok):
            tokens.extend(list(tok))
        else:
            tokens.append(tok)
    return tokens


def _tokenize_2gram_fallback(text: str) -> list[str]:
    """2-gram 降級模式——與原 baseline.py 的 _tokenize 相同。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for run in _CHINESE_RUN_RE.findall(lowered):
        if len(run) < 2:
            continue
        tokens.extend(run[i: i + 2] for i in range(len(run) - 1))
    for run in _ENGLISH_RUN_RE.findall(lowered):
        tokens.append(run)
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]


def tokenize(text: str, min_token_len: int = 2) -> list[str]:
    """統一斷詞入口。

    中文：jieba 精確模式（可用時）/ 2-gram（降級時，min_token_len>=2）
          / 輕量單字模式（降級時，min_token_len<2）
    英文：保留完整單詞（小寫化）
    過濾：停用詞、最小 token 長度、純數字、純標點

    Args:
        text: 待斷詞文字。
        min_token_len: token 最小長度，預設 2（保持舊行為）。
            傳入 1 可保留單字 token（用於規則比對，如否定副詞偵測）。
    """
    if not text:
        return []

    lowered = text.lower()

    if _jieba is None:
        if min_token_len >= 2:
            return _tokenize_2gram_fallback(text)
        raw_tokens = _tokenize_lightweight(lowered)
    else:
        raw_tokens = _jieba.lcut(lowered)

    result: list[str] = []
    for tok in raw_tokens:
        if not _should_keep_token(tok, min_token_len=min_token_len):
            continue
        result.append(tok)

    return result


# 興趣領域用的名詞性詞性標記：
#   n* — 名詞族（n 一般 / nr 人名 / ns 地名 / nt 機構 / nz 專名）
#   vn — 動名詞（「繞路」「規劃」這類事件名）
#   eng — 英文詞
_NOUNISH_FLAG_PREFIXES = ("n", "vn", "eng")


def tokenize_nouns(text: str, min_token_len: int = 2) -> list[str]:
    """名詞性斷詞——興趣領域（domains）專用。

    一般 tokenize 保留所有實/虛詞供檢索比對；興趣領域若不過濾詞性，
    「放在」「這樣」「突然」「比較」這類功能詞會佔滿榜單（v0.8 實測
    50 輪互動的 domains 前 25 名中虛詞佔八成）。此函式用 jieba 詞性
    標註只保留名詞族 / 動名詞 / 英文詞。

    jieba 不可用時退回一般 tokenize（維持舊行為，不多做假設）。
    """
    if not text:
        return []
    if _jieba is None:
        return tokenize(text, min_token_len=min_token_len)

    import jieba.posseg as _pseg

    result: list[str] = []
    for pair in _pseg.lcut(text.lower()):
        word, flag = pair.word, pair.flag
        if not any(flag.startswith(p) for p in _NOUNISH_FLAG_PREFIXES):
            continue
        if not _should_keep_token(word, min_token_len=min_token_len):
            continue
        result.append(word)
    return result
