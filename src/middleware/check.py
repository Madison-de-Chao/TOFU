"""/check 模式的 session state 管理。

依 `docs/specs/逗福_v0_6_開發_Spec_v2.md` §1.6。

三階段流程：
1. 使用者打 ``/check <內容>`` → :func:`create_session` 建立 session state，
   寫到 ``data/sessions/check_<uuid>.json``
2. LLM 跑第一階段後，:func:`save_stage1_output` 把 summary 寫回 state
3. 使用者輸入 ``1`` / ``2`` / ``3`` → :func:`find_latest_session` 找最近 session，
   :func:`load_session` 讀回，:func:`discard_session` 清除 state

每次啟動 /check 時先做 :func:`sweep_old_sessions`（預設 24 小時），
避免 state 累積。檔案損壞／讀取失敗 → caller fallback（不報錯、
指示使用者重新打 ``/check``）。

設計原則：
- **純檔案系統**，不引入資料庫、不用 pickle。
- 路徑決定權在 caller：所有函式都接受 ``session_dir`` 參數，
  main.py 傳固定的 ``data/sessions/``，測試可以用 tmpdir。
- 讀寫都用 UTF-8 + ensure_ascii=False，保留中文內容可讀性。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


SESSION_FILE_PREFIX = "check_"
SESSION_FILE_SUFFIX = ".json"
DEFAULT_SESSION_TTL_HOURS = 24

# session_id 合法字元集（限制成 uuid4 的字面形式，避免遇到 ../ 這類路徑注入）
_SESSION_ID_RE = re.compile(r"^[0-9a-fA-F\-]{8,64}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def new_session_id() -> str:
    """產生新的 session id（uuid4）。"""
    return str(uuid.uuid4())


def _session_path(session_dir: str, session_id: str) -> str:
    """安全組合 session 檔案路徑，拒絕非法 session_id。"""
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"Illegal session_id: {session_id!r}")
    filename = f"{SESSION_FILE_PREFIX}{session_id}{SESSION_FILE_SUFFIX}"
    return os.path.join(session_dir, filename)


def ensure_session_dir(session_dir: str) -> None:
    """確保 session 目錄存在。"""
    os.makedirs(session_dir, exist_ok=True)


def create_session(
    session_dir: str,
    user_content: str,
    *,
    session_id: str | None = None,
    lookup_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """建立新的 /check session 並寫入磁碟。

    回傳含 ``session_id`` / ``user_content`` / ``created_at`` / ``mode`` 的 dict。
    ``stage1_output`` 初始為 ``None``，待第一階段 LLM 呼叫完成後由
    :func:`save_stage1_output` 寫入。
    """
    ensure_session_dir(session_dir)
    sid = session_id or new_session_id()
    session = {
        "session_id": sid,
        "user_content": user_content,
        "stage1_output": None,
        "created_at": _iso(_now()),
        "mode": "check",
        "lookup_audit": lookup_audit,
    }
    path = _session_path(session_dir, sid)
    _atomic_write_json(path, session)
    return session


def save_stage1_output(
    session_dir: str,
    session_id: str,
    stage1_output: str,
) -> dict[str, Any]:
    """把第一階段的 LLM 輸出回寫到 session state。"""
    session = load_session(session_dir, session_id)
    if session is None:
        raise FileNotFoundError(
            f"session {session_id} not found in {session_dir}"
        )
    session["stage1_output"] = stage1_output
    _atomic_write_json(_session_path(session_dir, session_id), session)
    return session


def load_session(
    session_dir: str,
    session_id: str,
) -> dict[str, Any] | None:
    """讀取 session state。

    失敗條件（任一觸發回傳 None，caller 應視為 fallback）：
    - 檔案不存在
    - 檔案損壞（不是合法 JSON）
    - 檔案欄位不完整（缺 session_id / user_content）
    """
    try:
        path = _session_path(session_dir, session_id)
    except ValueError:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("session_id") != session_id:
        return None
    if not isinstance(data.get("user_content"), str):
        return None
    return data


def find_latest_session(session_dir: str) -> dict[str, Any] | None:
    """依 mtime 找最近的 check_*.json session，回傳其內容。

    若目錄不存在或內無有效 session 檔，回傳 None。
    損壞的 session 檔會被略過，繼續找次新的。
    """
    if not os.path.isdir(session_dir):
        return None
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(session_dir):
        if not name.startswith(SESSION_FILE_PREFIX):
            continue
        if not name.endswith(SESSION_FILE_SUFFIX):
            continue
        full = os.path.join(session_dir, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        candidates.append((mtime, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if not isinstance(data.get("session_id"), str):
            continue
        if not isinstance(data.get("user_content"), str):
            continue
        return data
    return None


def discard_session(session_dir: str, session_id: str) -> bool:
    """刪除指定 session 的 state 檔，成功回傳 True。"""
    try:
        path = _session_path(session_dir, session_id)
    except ValueError:
        return False
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def sweep_old_sessions(
    session_dir: str,
    *,
    ttl_hours: int = DEFAULT_SESSION_TTL_HOURS,
    now: datetime | None = None,
) -> int:
    """掃 session 目錄刪除超過 ttl_hours 的 check_*.json。

    依照 ``created_at`` 欄位判斷（若欄位解析失敗或缺失，改用檔案 mtime）。
    回傳實際刪除的檔案數。
    """
    if not os.path.isdir(session_dir):
        return 0
    reference = now or _now()
    cutoff = reference - timedelta(hours=ttl_hours)
    removed = 0
    for name in os.listdir(session_dir):
        if not name.startswith(SESSION_FILE_PREFIX):
            continue
        if not name.endswith(SESSION_FILE_SUFFIX):
            continue
        full = os.path.join(session_dir, name)

        # 先嘗試讀 created_at
        created_at: datetime | None = None
        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
            ca_str = (data or {}).get("created_at")
            if isinstance(ca_str, str):
                try:
                    created_at = datetime.fromisoformat(ca_str)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    created_at = None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            created_at = None

        # 退回 mtime
        if created_at is None:
            try:
                mtime = os.path.getmtime(full)
                created_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
            except OSError:
                continue

        if created_at < cutoff:
            try:
                os.remove(full)
                removed += 1
            except OSError:
                pass
    return removed


def _atomic_write_json(path: str, payload: Any) -> None:
    """原子寫入：先寫 .tmp 再 rename。避免半寫檔。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


__all__ = [
    "SESSION_FILE_PREFIX",
    "SESSION_FILE_SUFFIX",
    "DEFAULT_SESSION_TTL_HOURS",
    "new_session_id",
    "ensure_session_dir",
    "create_session",
    "save_stage1_output",
    "load_session",
    "find_latest_session",
    "discard_session",
    "sweep_old_sessions",
]
