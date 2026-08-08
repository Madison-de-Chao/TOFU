"""端點紀錄模組。

端點（endpoint）是 TOFU 的核心資料單位。一次互動會產生一對：
- start：使用者輸入 + 復述 + 補位提問 + 確認/修正 + 最終的目標/動機/限制
- end：執行結果 + 滿意度 + 偏差描述

同一次互動的 start 與 end 共用同一個 event_id。
MVP 階段全部落地在單一 JSON 檔案，不使用資料庫。

公測版新增：
- JSON 損壞容錯：讀到非法 JSON 時自動備份為 .bak，從空列表重來。
- 原子寫入：先寫 .tmp 再 rename，避免中途斷電造成半寫檔案。

v0.5 新增（端點檢索與元動機偵測邏輯鏈）：
- 冷卻欄位：last_referenced / reference_count / round_last_referenced / status
  （status: active / pending_confirmation / superseded）
- 舊端點讀進來時自動補預設值（永不刪除、永不壓縮）
- mark_hit() 命中時更新計數並把 pending 恢復 active
- apply_cooldown() 90 天 OR 50 輪未命中 → pending_confirmation（降權，不移除）
- retrieve_top_k_endpoints() 強制查表 gate 的主力函式

v0.6 新增（PR-E：JSONL 底層 + 詞性分類檔，依 Spec v2 §3）：
- 雙格式支援：依 path 後綴自動切換
  * ``.jsonl`` → 新格式，append-only JSONL + filelock，O(1) 寫入
  * ``.json``  → 舊格式（單一 JSON array），整檔讀寫，向後相容既有測試
- 自動遷移：當 caller 傳入 ``.jsonl`` 路徑、但同位置存在 legacy ``.json`` 檔案時
  自動 migrate（轉成 jsonl 行、舊檔重命名為 ``.json.bak``）
- 原子寫入：``mark_hit`` / ``apply_cooldown`` / ``clear`` 等批次更新仍走
  「寫 tmp → rename」的原子路徑，jsonl 也是。
- ``filelock``：append/rewrite 期間以同一個 ``.lock`` 檔做跨 process 鎖；
  filelock 套件未安裝時退回為 per-path 的 thread-only lock（單 process
  場景仍正確，但無跨 process 保護，多 process 才會競爭）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from src.utils.tokenizer import tokenize as _tokenize

try:  # filelock 為 PR-E 後新增依賴；缺席時退回為 thread-only lock
    from filelock import FileLock as _FileLock  # type: ignore
    from filelock import Timeout as _FileLockTimeout  # type: ignore

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - 環境缺套件時的 fallback
    _FileLock = None  # type: ignore
    _FileLockTimeout = Exception  # type: ignore
    _HAS_FILELOCK = False


# 程序內回退鎖：filelock 缺席時用，按 path 分配各自的 threading.Lock
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _acquire_path_lock(path: str, timeout: float = 30.0):
    """回傳一個 context manager 鎖住 ``path`` 的寫入。

    優先使用 ``filelock.FileLock``（跨 process 安全）；缺席時退回 thread-only。
    """
    if _HAS_FILELOCK:
        return _FileLock(path + ".lock", timeout=timeout)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(path, threading.Lock())
    return lock


# v0.5：冷卻門檻（90 天 OR 50 輪未命中 → pending_confirmation）
COOLDOWN_DAYS = 90
COOLDOWN_ROUNDS = 50
# v0.5：第四層強制查表 gate 的預設 top-k
DEFAULT_TOP_K = 30
# 近因保底：最近 N 筆 start 端點無條件帶上（接續性）
RECENT_FLOOR_N = 2
# v0.5：pending_confirmation 端點的排序降權
_PENDING_PENALTY = 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_event_id() -> str:
    return str(uuid.uuid4())


def new_endpoint_id() -> str:
    return str(uuid.uuid4())


def _ensure_cooldown_fields(row: dict[str, Any]) -> dict[str, Any]:
    """為舊端點補上 v0.5 冷卻欄位的預設值。

    原則：只增不改。舊端點讀進記憶體時自動補 `status=active` / 其他為 `None` / `0`。
    補欄位本身不會單獨觸發寫檔；但因為此函式會就地修改傳入的 row，
    若後續流程對同一份資料執行 `_write(data)`（例如整檔覆寫的 append 類操作），
    補上的欄位也會一併被寫回磁碟。
    此函式回傳傳入的 row，方便 caller 鏈式使用。
    """
    if not isinstance(row, dict):
        return row
    row.setdefault("last_referenced", None)
    row.setdefault("reference_count", 0)
    row.setdefault("round_last_referenced", None)
    row.setdefault("status", "active")
    return row


class EndpointStore:
    """端點儲存。依 path 後綴自動切換格式（v0.6 PR-E）。

    - ``.jsonl`` → 新格式：append-only JSONL + filelock，O(1) 寫入。
      適合大量端點（成千上萬筆），不會因為單筆 append 重讀整檔。
    - ``.json``  → 舊格式：單一 JSON array，整檔讀寫。向後相容既有測試。

    自動遷移：當 path 為 ``.jsonl`` 而同位置存在 legacy ``<basename>.json``
    時，自動把 legacy 內容轉成 jsonl 行寫入新檔，並把 legacy 重命名為
    ``<basename>.json.bak``（一次性、不可逆，但 .bak 保留供回溯）。

    讀寫共通保證：
    - 讀：每次操作都重新讀檔，避免狀態飄移；損壞行/損壞檔自動 quarantine。
    - 寫：append 走 file lock + append mode；整檔覆寫走 tmp + rename 原子路徑。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        # 依後綴決定格式。預設為 json（向後相容）。
        self._format = "jsonl" if path.endswith(".jsonl") else "json"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        # JSONL 模式：先檢查是否需要從 legacy .json 遷移
        if self._format == "jsonl":
            self._maybe_migrate_legacy_json()

        if not os.path.exists(self.path):
            if self._format == "jsonl":
                # touch 一個空檔，後續 append 直接寫即可
                with _acquire_path_lock(self.path):
                    open(self.path, "a", encoding="utf-8").close()
            else:
                self._write([])

    # ------------------------------------------------------------------
    # PR-E：legacy .json → .jsonl 遷移
    # ------------------------------------------------------------------
    def _maybe_migrate_legacy_json(self) -> None:
        """若同位置有 legacy ``.json`` 而新 ``.jsonl`` 還沒生成，自動遷移。

        遷移步驟：
        1. 讀 legacy 整檔（單一 JSON array）。
        2. 把每筆 row 寫成 jsonl 一行到目標 ``.jsonl``。
        3. 把 legacy 重命名為 ``.json.bak`` 保留供回溯。

        失敗時不阻擋啟動：legacy 損壞 → 直接 quarantine 為 .bak，新檔留空。
        """
        if not self.path.endswith(".jsonl"):
            return
        legacy_path = self.path[:-1]  # endpoints.jsonl → endpoints.json
        if not os.path.exists(legacy_path):
            return
        if os.path.exists(self.path):
            # 已經有新檔，不再覆寫；legacy 視為陳舊，仍重命名為 .bak 避免混淆
            try:
                os.replace(legacy_path, legacy_path + ".bak")
            except OSError:
                pass
            return

        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if not isinstance(old_data, list):
                raise ValueError("legacy top-level JSON is not a list")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            print(
                f"[逗福Tofu 警告] legacy {os.path.basename(legacy_path)} 無法解析"
                f"（{exc}），重命名為 .bak 並從空 JSONL 重新開始。",
                file=sys.stderr,
                flush=True,
            )
            try:
                os.replace(legacy_path, legacy_path + ".bak")
            except OSError:
                pass
            return

        # 寫到新 JSONL（atomic：先寫 tmp，完成後 rename）
        tmp_path = self.path + ".tmp"
        with _acquire_path_lock(self.path):
            with open(tmp_path, "w", encoding="utf-8") as f:
                for row in old_data:
                    if not isinstance(row, dict):
                        continue
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass
            os.replace(tmp_path, self.path)
        # 把 legacy 重命名為 .bak（不刪，保留供回溯）
        try:
            os.replace(legacy_path, legacy_path + ".bak")
        except OSError as exc:  # pragma: no cover - 防禦
            print(
                f"[逗福Tofu 警告] 重命名 legacy 檔失敗：{exc}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[逗福Tofu] 已將 {os.path.basename(legacy_path)} 遷移為 "
            f"{os.path.basename(self.path)}（{len(old_data)} 筆），"
            f"原檔保留為 .bak。",
            file=sys.stderr,
            flush=True,
        )

    # ------------------------------------------------------------------
    def _read(self) -> list[dict[str, Any]]:
        if self._format == "jsonl":
            return self._read_jsonl()
        return self._read_legacy_json()

    def _read_jsonl(self) -> list[dict[str, Any]]:
        """逐行讀 JSONL；損壞行（無法 parse 或非 dict）跳過並警告，不 quarantine 整檔。"""
        if not os.path.exists(self.path):
            return []
        rows: list[dict[str, Any]] = []
        bad_lines = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue
                    if not isinstance(row, dict):
                        bad_lines += 1
                        continue
                    _ensure_cooldown_fields(row)
                    rows.append(row)
        except (UnicodeDecodeError, OSError) as exc:
            self._quarantine_corrupted(reason=f"jsonl read failed: {exc}")
            return []
        if bad_lines:
            print(
                f"[逗福Tofu 警告] {os.path.basename(self.path)} 跳過 "
                f"{bad_lines} 行損壞紀錄（不影響其他資料）。",
                file=sys.stderr,
                flush=True,
            )
        return rows

    def _read_legacy_json(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # 損壞 → 備份並從空列表重新開始
            self._quarantine_corrupted(reason=str(exc))
            return []

        if not isinstance(data, list):
            self._quarantine_corrupted(reason="top-level JSON is not a list")
            return []
        # v0.5：讀進來時補冷卻欄位預設值（不寫回磁碟）
        for row in data:
            _ensure_cooldown_fields(row)
        return data

    def _quarantine_corrupted(self, reason: str) -> None:
        """把損壞的端點檔備份為 .bak 並顯示警告。"""
        bak_path = self.path + ".bak"
        try:
            # 若已有 .bak，仍覆寫（最新狀態最有參考價值）
            shutil.copy2(self.path, bak_path)
        except FileNotFoundError:
            pass
        except Exception as exc:  # pragma: no cover - 防禦
            print(
                f"[逗福Tofu 警告] 備份損壞的 {os.path.basename(self.path)} 失敗：{exc}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[逗福Tofu 警告] {os.path.basename(self.path)} 損壞（{reason}），"
            f"已備份為 {os.path.basename(bak_path)}，"
            f"將從空檔重新開始。",
            file=sys.stderr,
            flush=True,
        )
        # 重新寫入一個乾淨的空狀態
        try:
            if self._format == "jsonl":
                with _acquire_path_lock(self.path):
                    open(self.path, "w", encoding="utf-8").close()
            else:
                self._write_atomic_legacy([])
        except Exception:  # pragma: no cover - 防禦
            pass

    # ------------------------------------------------------------------
    def _write_atomic_legacy(self, data: list[dict[str, Any]]) -> None:
        """legacy 單一 JSON array 的原子寫入：先寫 .tmp，完成後 rename。"""
        tmp_path = self.path + ".tmp"
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        # 寫前先 parse 一次，確保 payload 本身合法
        json.loads(payload)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            try:
                f.flush()
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        os.replace(tmp_path, self.path)

        # 寫入後再讀回驗證一次
        with open(self.path, "r", encoding="utf-8") as f:
            verify = json.load(f)
            if not isinstance(verify, list):
                raise ValueError("post-write verification failed: not a list")

    def _write_atomic_jsonl_nolock(self, data: list[dict[str, Any]]) -> None:
        """JSONL 整檔原子覆寫的核心：寫 .tmp 後 rename。

        呼叫端必須已持有 ``.lock``。供 `_rewrite_locked` 在同一鎖區間內
        完成 read-modify-write 時使用，避免巢狀重複取鎖。
        """
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in data:
                if not isinstance(row, dict):
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        os.replace(tmp_path, self.path)

    def _write_atomic_jsonl(self, data: list[dict[str, Any]]) -> None:
        """JSONL 的整檔原子覆寫（用於 clear 等不需 read-modify-write 的情境）。"""
        with _acquire_path_lock(self.path):
            self._write_atomic_jsonl_nolock(data)

    def _write(self, data: list[dict[str, Any]]) -> None:
        """整檔覆寫。jsonl 與 json 都走原子路徑，差別在儲存格式。"""
        if self._format == "jsonl":
            self._write_atomic_jsonl(data)
        else:
            self._write_atomic_legacy(data)

    def _rewrite_locked(
        self,
        mutate: Callable[[list[dict[str, Any]]], tuple[bool, Any]],
    ) -> Any:
        """在單一鎖區間內完成 read-modify-write，回傳 mutate 的結果值。

        ``mutate(data)`` 就地修改 ``data`` 並回傳 ``(changed, result)``；
        只有 ``changed`` 為真時才寫回磁碟。

        jsonl 格式下，整段「讀檔 → 改 → 覆寫」共用同一把 ``.lock``，因此
        mark_hit / apply_cooldown 不會再用過期快照蓋掉並行 ``append_*`` 寫入的
        新紀錄（修正先讀後鎖的 read-modify-write race）。legacy json 沿用既有
        的讀後原子覆寫路徑。
        """
        if self._format == "jsonl":
            with _acquire_path_lock(self.path):
                data = self._read_jsonl()
                changed, result = mutate(data)
                if changed:
                    self._write_atomic_jsonl_nolock(data)
                return result
        data = self._read_legacy_json()
        changed, result = mutate(data)
        if changed:
            self._write_atomic_legacy(data)
        return result

    def _append_row(self, row: dict[str, Any]) -> None:
        """單筆 append。

        - jsonl：直接 ``open(..., 'a')`` 寫一行（O(1)），filelock 保證原子性。
        - json：讀整檔 → 加一筆 → 整檔覆寫（O(N)，向後相容）。
        """
        if self._format == "jsonl":
            line = json.dumps(row, ensure_ascii=False) + "\n"
            with _acquire_path_lock(self.path):
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except (AttributeError, OSError):
                        pass
        else:
            data = self._read()
            data.append(row)
            self._write(data)

    # ------------------------------------------------------------------
    def all(self) -> list[dict[str, Any]]:
        return self._read()

    def starts(self) -> list[dict[str, Any]]:
        return [e for e in self._read() if e.get("type") == "start"]

    def ends(self) -> list[dict[str, Any]]:
        return [e for e in self._read() if e.get("type") == "end"]

    def completed_events(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """回傳所有已湊齊 start+end 的事件對。"""
        data = self._read()
        by_event: dict[str, dict[str, dict[str, Any]]] = {}
        for row in data:
            eid = row.get("event_id")
            if not eid:
                continue
            slot = by_event.setdefault(eid, {})
            slot[row.get("type", "")] = row
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for eid, slot in by_event.items():
            if "start" in slot and "end" in slot:
                out.append((slot["start"], slot["end"]))
        return out

    def completed_count(self) -> int:
        return len(self.completed_events())

    def clear(self) -> None:
        """清空所有端點紀錄（公開介面）。

        取代直接呼叫私有的 `_write([])`。實際上仍走原子寫入路徑，
        因此清空也不會留下半寫檔或不合法的 JSON / JSONL。
        """
        self._write([])

    # ------------------------------------------------------------------
    def append_start(
        self,
        event_id: str,
        start_data: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "endpoint_id": new_endpoint_id(),
            "timestamp": _now_iso(),
            "type": "start",
            "event_id": event_id,
            "start_data": start_data,
            "end_data": None,
        }
        _ensure_cooldown_fields(row)
        self._append_row(row)
        return row

    def append_end(
        self,
        event_id: str,
        end_data: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "endpoint_id": new_endpoint_id(),
            "timestamp": _now_iso(),
            "type": "end",
            "event_id": event_id,
            "start_data": None,
            "end_data": end_data,
        }
        _ensure_cooldown_fields(row)
        self._append_row(row)
        return row

    # ------------------------------------------------------------------
    # v0.5：知識冷卻
    # ------------------------------------------------------------------
    def mark_hit(
        self,
        endpoint_ids: list[str] | set[str],
        current_round: int,
    ) -> int:
        """把命中端點的冷卻計數器更新。

        - 更新 last_referenced、reference_count、round_last_referenced
        - status == "pending_confirmation" 自動恢復 "active"
        - "superseded" 不恢復（使用者已否認，要恢復需走顯式流程）
        回傳實際被更新的 row 數量。
        """
        if not endpoint_ids:
            return 0
        ids = set(endpoint_ids)
        now_iso = _now_iso()

        def _mutate(data: list[dict[str, Any]]) -> tuple[bool, int]:
            touched = 0
            for row in data:
                if row.get("endpoint_id") in ids:
                    _ensure_cooldown_fields(row)
                    row["last_referenced"] = now_iso
                    row["reference_count"] = int(row.get("reference_count") or 0) + 1
                    row["round_last_referenced"] = int(current_round)
                    if row.get("status") == "pending_confirmation":
                        row["status"] = "active"
                    touched += 1
            return touched > 0, touched

        return self._rewrite_locked(_mutate)

    def apply_cooldown(
        self,
        current_round: int,
        now: datetime | None = None,
        days_threshold: int = COOLDOWN_DAYS,
        rounds_threshold: int = COOLDOWN_ROUNDS,
    ) -> int:
        """掃全量端點套用冷卻規則。

        - 90 天 OR 50 輪未命中 → active → pending_confirmation
        - 永不刪除、永不移除
        - 已是 pending_confirmation / superseded 的不動
        回傳被新標記為 pending_confirmation 的 row 數量。
        """
        reference_time = now or datetime.now(timezone.utc)
        cutoff_time = reference_time - timedelta(days=days_threshold)

        def _mutate(data: list[dict[str, Any]]) -> tuple[bool, int]:
            changed = 0
            for row in data:
                _ensure_cooldown_fields(row)
                # 只對 start 端點（真正帶知識的）跑冷卻
                if row.get("type") != "start":
                    continue
                if row.get("status") != "active":
                    continue

                last_ref = row.get("last_referenced")
                round_last = row.get("round_last_referenced")

                # 沒被命中過 → 用 append 時間推 / 0 輪推
                time_based_stale = False
                last_ref_time: datetime | None = None
                if last_ref:
                    try:
                        last_ref_time = datetime.fromisoformat(last_ref)
                    except (TypeError, ValueError):
                        last_ref_time = None
                if last_ref_time is None:
                    # 沒被命中過：看端點自己的 timestamp
                    ts_str = row.get("timestamp")
                    if ts_str:
                        try:
                            last_ref_time = datetime.fromisoformat(ts_str)
                        except (TypeError, ValueError):
                            last_ref_time = None
                if last_ref_time is not None and last_ref_time < cutoff_time:
                    time_based_stale = True

                round_based_stale = False
                # 沒被命中過（round_last 為 None）→ 只有時間門檻生效，
                # 不走輪數門檻。否則 global round 一超過 50，所有剛建立、
                # 從未被 hit 的端點都會立刻被標 pending（#PR review P1）。
                if round_last is not None:
                    baseline_round = int(round_last)
                    if int(current_round) - baseline_round > rounds_threshold:
                        round_based_stale = True

                if time_based_stale or round_based_stale:
                    row["status"] = "pending_confirmation"
                    changed += 1
            return changed > 0, changed

        return self._rewrite_locked(_mutate)

    def current_round(self) -> int:
        """目前的互動輪數——取已完成事件數 + 1。

        start 寫入後、end 寫入前這段時間，current_round 會等於
        completed_count() + 1（代表「當前這一輪」）。
        """
        return self.completed_count() + 1

    # ------------------------------------------------------------------
    # P0-2：黑盒子回溯
    # ------------------------------------------------------------------
    def append_black_box(
        self,
        event_id: str,
        black_box_data: dict[str, Any],
    ) -> dict[str, Any]:
        """當偏差偵測觸發時，記錄一筆黑盒子回溯資料。

        黑盒子以獨立 row 儲存（type="black_box"），綁定同一個 event_id。
        平時不記錄中間過程，只在結尾與開頭對不上時才打開。
        """
        row = {
            "endpoint_id": new_endpoint_id(),
            "timestamp": _now_iso(),
            "type": "black_box",
            "event_id": event_id,
            "start_data": None,
            "end_data": None,
            "black_box": black_box_data,
        }
        _ensure_cooldown_fields(row)
        self._append_row(row)
        return row

    def black_boxes(self) -> list[dict[str, Any]]:
        """回傳所有黑盒子記錄，依 timestamp 遞增排序。"""
        rows = [r for r in self._read() if r.get("type") == "black_box"]
        return sorted(rows, key=lambda r: r.get("timestamp", ""))


def make_start_data(
    user_input: str,
    tofu_understanding: str,
    gap_questions: list[str],
    user_confirmation: str,
    user_modifications: str,
    goal: str,
    motivation: str,
    constraints: list[str],
    gap_categories: list[str] | None = None,
    *,
    unresolved_gaps: list[str] | None = None,
    convergence_reason: str | None = None,
    total_gap_rounds: int = 0,
) -> dict[str, Any]:
    """建構 start_data，欄位嚴格對應規格書 3.3。

    gap_categories 不是規格書原生欄位，但為了讓基線模組能純程式碼做頻率統計，
    在 start_data 裡附帶這個額外欄位（不影響原欄位完整性）。

    v4.1 收斂閘門（20260424_收斂閘門_實作規格_v1.md）新增三個可選欄位：
    - unresolved_gaps: 收斂時仍未解的補位維度（條件一 / 條件二觸發時才非空）
    - convergence_reason: "empty_streak" / "max_rounds" / "info_sufficient" / None
    - total_gap_rounds: 實際補位輪數（單輪為 1；多輪迴圈時為實際執行的輪數）
    舊 caller 不帶新 kwargs 時，三個欄位分別預設為 [] / None / 0，向後相容。
    """
    return {
        "user_input": user_input,
        "tofu_understanding": tofu_understanding,
        "gap_questions": list(gap_questions),
        "user_confirmation": user_confirmation,  # confirmed | modified
        "user_modifications": user_modifications,
        "goal": goal,
        "motivation": motivation,
        "constraints": list(constraints),
        "gap_categories": list(gap_categories or []),
        "unresolved_gaps": list(unresolved_gaps or []),
        "convergence_reason": convergence_reason,
        "total_gap_rounds": int(total_gap_rounds or 0),
    }


def make_end_data(
    result: str,
    user_satisfaction: str = "no_feedback",
    deviation_from_goal: str = "",
) -> dict[str, Any]:
    return {
        "result": result,
        "user_satisfaction": user_satisfaction,  # satisfied | unsatisfied | no_feedback
        "deviation_from_goal": deviation_from_goal,
    }


# ------------------------------------------------------------------
# 記憶編碼（Memory Encoding）
#
# 把端點紀錄壓縮成結構化標記。設計原則：
# 1. 關鍵名詞原封不動保留（片名、食材名、產品型號、人名、地名）
# 2. 廢話全刪（寒暄、語氣詞、重複表達）
# 3. 格式不像對話——防止 LLM 續寫
# 4. 純程式碼產出——不需要 LLM 參與
# 5. LLM 可直接讀懂——不需要額外解碼
#
# 這是逗福端點架構的自然延伸：
# 端點說「只記開頭結尾」，編碼說「開頭結尾也不記原文，記結構化標記」。
# ------------------------------------------------------------------

_KEY_NOUN_PATTERNS: list[re.Pattern[str]] = []


def _ensure_key_noun_patterns() -> list[re.Pattern[str]]:
    """延遲編譯 key noun 的正則表達式。"""
    global _KEY_NOUN_PATTERNS
    if _KEY_NOUN_PATTERNS:
        return _KEY_NOUN_PATTERNS

    _KEY_NOUN_PATTERNS = [
        # 大寫開頭的詞組（英文專有名詞）
        # "John Mulaney", "Sony A7R IV", "West Elm", "Netflix"
        re.compile(r'\b([A-Z][a-zA-Z\'-]+(?:\s+(?:[A-Z][a-zA-Z0-9\'-]*|[IVXLCDM]+|[A-Z0-9]{2,}))*)\b'),

        # 數字+單位或數字+名詞
        # "375°F", "24-70mm", "$500", "1300 followers", "3 times"
        re.compile(
            r'\b(\$?\d[\d,.\-]*\s*'
            r'(?:°[FC]|mm|cm|m|kg|lb|oz|ml|L|hrs?|mins?|%|dollars?'
            r'|followers?|days?|weeks?|months?|years?|people|persons?'
            r'|cups?|tbsp|tsp|times|pieces?|sessions?|pages?'
            r'|GB|TB|MB|K|k)?)\b',
            re.IGNORECASE,
        ),

        # 帶斜線的型號 "f/2.8"
        re.compile(r'\b(\w+/[\d.]+)\b'),

        # 全大寫縮寫 "RACI", "MCM", "NAS", "HEPA"
        re.compile(r'\b([A-Z]{2,}(?:\s+[IVXLCDM]+)?)\b'),
    ]
    return _KEY_NOUN_PATTERNS


def extract_key_nouns(text: str) -> list[str]:
    """從文字中提取關鍵名詞。純程式碼，不呼叫 LLM。

    提取規則：
    - 大寫開頭的詞組（英文專有名詞）
    - 數字+單位
    - 型號格式（含斜線）
    - 全大寫縮寫
    - 引號內的內容

    回傳去重後的排序列表。
    """
    if not text:
        return []

    patterns = _ensure_key_noun_patterns()
    nouns: set[str] = set()

    # 正則提取
    for pattern in patterns:
        for match in pattern.finditer(text):
            term = match.group(1).strip()
            # 過濾太短或太普通的
            if len(term) <= 1:
                continue
            # 過濾純數字（沒有單位的）
            if term.replace(",", "").replace(".", "").isdigit():
                continue
            # 過濾常見句首詞（大寫但不是專有名詞）
            _COMMON_STARTS = {
                "I", "The", "This", "That", "My", "We", "You", "He", "She",
                "It", "They", "What", "How", "Why", "When", "Where", "Who",
                "Yes", "No", "And", "But", "Or", "So", "If", "For", "Not",
                "Can", "Do", "Does", "Did", "Is", "Are", "Was", "Were",
                "Have", "Has", "Had", "Will", "Would", "Could", "Should",
                "May", "Might", "Any", "Some", "Just", "Also", "Very",
                "Thanks", "Thank", "Great", "Good", "Well", "Sure",
                "Maybe", "Really", "Actually", "Basically", "Definitely",
                "Here", "There", "Now", "Then", "Today", "Yesterday",
            }
            if term in _COMMON_STARTS:
                continue
            nouns.add(term)

    # 引號內的內容
    for match in re.finditer(r'["\u201c](.+?)["\u201d]', text):
        content = match.group(1).strip()
        if 1 < len(content) < 100:
            nouns.add(content)

    return sorted(nouns)


# ------------------------------------------------------------------
# v0.5+：關鍵名詞的「動詞方向」狀態偵測
# ------------------------------------------------------------------
# 從名詞附近的上下文判斷使用者對這個物件的關係方向——純程式碼，
# 不需要 LLM 判斷。這讓端點在編碼時就能標示每個 KEY 名詞的狀態，
# 之後 execute_task 看到 `Nikon D750 [abandoned]` 會知道不該把它
# 當作使用中的裝備；看到 `Rolleiflex [acquired]` 會知道是剛入手的。
#
# 狀態集：
#   acquired  — 剛獲得、買入、入手
#   divested  — 賣掉、出清、脫手
#   active    — 目前仍在使用
#   abandoned — 停用、放棄、不再用
#   mentioned — 上下文沒明確動作（預設）
#
# 選擇演算法：nearest-verb wins（嚴格 `<` 比較）。對每個名詞，從前後
# ±window 字元的上下文裡找距離名詞最近的動詞，用它的 status。多名詞
# 句子如「bought Rolleiflex but stopped using Leica M6」會正確把
# `stopped using` 歸給 Leica M6、`bought` 歸給 Rolleiflex。
#
# 迭代順序 abandoned→divested→acquired→active 的作用是在**距離相同**時
# 優先保留較強的負向訊號（`stopped using` 和 `using` 會同時命中同一個
# 名詞，應該被判為 abandoned 而不是 active）。動詞 phrase 也盡量用完整
# 形式（`stopped\s+using`）以便距離量測落在名詞旁邊。
# ------------------------------------------------------------------

_VERB_DIRECTION_ORDER: list[str] = [
    "abandoned",
    "divested",
    "acquired",
    "active",
]

_VERB_DIRECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "abandoned": [
        re.compile(
            # 完整的 phrase：動詞 + (using|playing|wearing...)，讓匹配
            # 延伸到名詞前一個 token，距離量測才不會輸給獨立的 `using`。
            r"(?:stopped\s+(?:using|playing|wearing|riding|driving|watching)"
            r"|quit\s+(?:using|playing)?"
            r"|no\s+longer\s+(?:use|using|play|playing)?"
            r"|abandoned"
            r"|gave\s+up\s+(?:using|on|playing)?"
            r"|got\s+rid\s+of"
            r"|don'?t\s+use\s+(?:it|them|that)?"
            r"|not\s+using"
            r"|used\s+to\s+use"
            r"|switched\s+(?:away\s+)?from"
            r"|replaced)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(不用了|停了|停用|淘汰|換掉|放棄|不再用|不再使用|"
            r"不用|沒在用|沒再用|換成)"
        ),
    ],
    "divested": [
        re.compile(
            r"(?:sold|sold\s+off|divested|traded\s+(?:in|away)|let\s+go\s+of)",
            re.IGNORECASE,
        ),
        re.compile(r"(賣了|賣掉|出清|脫手|賣出)"),
    ],
    "acquired": [
        re.compile(
            r"(?:bought|purchased|acquired|got\s+(?:a|the|my|an)|"
            r"just\s+got|picked\s+up|grabbed|ordered|received|"
            r"added\s+(?:a|the|an)|gifted)",
            re.IGNORECASE,
        ),
        re.compile(r"(買了|買到|入手|剛買|新買|新購|已購|下訂|訂了|拿到|收到)"),
    ],
    "active": [
        re.compile(
            r"(?:using|use|i\s+use|still\s+use|currently\s+use|"
            r"working\s+with|running|rely\s+on|shoot\s+with|play\s+on|"
            r"depend\s+on)",
            re.IGNORECASE,
        ),
        re.compile(r"(還在用|現在用|一直用|在用|用的是|目前用|持續用|都用|正在用)"),
    ],
}


# 名詞前後各看 30 字元的上下文，足以覆蓋「bought D750 last month」
# 這類短句，又不會跨到上一段話。
KEY_NOUN_CONTEXT_WINDOW = 30

# 語意子句邊界：偵測到這些就截斷 context，避免動詞跨子句污染其他名詞。
# 包含標點符號、中英文對比連接詞（但 / but / however / yet）、
# 英文列舉連接詞（and then 不算，只有 transition 詞才算）。
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:[。！？；;，,]"
    r"|\bbut\b|\bhowever\b|\byet\b|\bthough\b|\balthough\b|\bwhile\b"
    r"|但是|但|可是|然而|不過)",
    re.IGNORECASE,
)


def _detect_noun_status(
    text: str,
    noun: str,
    window: int = KEY_NOUN_CONTEXT_WINDOW,
) -> str:
    """從名詞的上下文判斷狀態。純程式碼，不呼叫 LLM。

    Args:
        text: 名詞出處的原文（通常是 user_input + result 的拼接）。
        noun: 要判斷的關鍵名詞。
        window: 前後各看多少字元。

    Returns:
        狀態代號：abandoned / divested / acquired / active / mentioned。
        沒命中任何動詞方向時回 "mentioned"。

    選擇演算法：找距離名詞最近的動詞匹配。這可以正確處理多名詞場景：
    "bought a Rolleiflex but stopped using Leica M6" 會把 `stopped`
    歸給 Leica M6、把 `bought` 歸給 Rolleiflex，不會互相污染。
    """
    if not text or not noun:
        return "mentioned"
    lower_text = text.lower()
    lower_noun = noun.lower()
    idx = lower_text.find(lower_noun)
    if idx < 0:
        return "mentioned"
    noun_end = idx + len(noun)

    # 先用 window 畫出粗略的候選範圍。
    raw_start = max(0, idx - window)
    raw_end = min(len(text), noun_end + window)

    # 再用語意子句邊界收緊範圍：如果 raw_start 到 idx 之間有邊界（句號/
    # 但/but...），把 start 往右推到邊界之後。右側同理往左縮。這樣
    # 「我買了 Nikon D750 配 24-70mm 鏡頭，但已經不用 Canon 5D 了」的
    # 24-70mm 不會吃到右邊 `不用` 的訊號。
    start = raw_start
    for m in _CLAUSE_BOUNDARY_RE.finditer(text, raw_start, idx):
        start = m.end()  # 取最右邊的 boundary，之後的才是和 noun 同子句
    end = raw_end
    right_match = _CLAUSE_BOUNDARY_RE.search(text, noun_end, raw_end)
    if right_match:
        end = right_match.start()

    context = text[start:end]
    noun_start_in_ctx = idx - start
    noun_end_in_ctx = noun_start_in_ctx + len(noun)

    best_status = "mentioned"
    best_distance = float("inf")
    # 對每個 status 類別下的每個 pattern，找**全部** match（一個
    # 上下文可能有多個匹配），取距離名詞最近的那個決定 status。
    for status in _VERB_DIRECTION_ORDER:
        for pattern in _VERB_DIRECTION_PATTERNS[status]:
            for m in pattern.finditer(context):
                m_start = m.start()
                m_end = m.end()
                if m_end <= noun_start_in_ctx:
                    distance = noun_start_in_ctx - m_end
                elif m_start >= noun_end_in_ctx:
                    distance = m_start - noun_end_in_ctx
                else:
                    # 動詞與名詞重疊（罕見）→ 距離視為 0
                    distance = 0
                if distance < best_distance:
                    best_distance = distance
                    best_status = status
    return best_status


def extract_key_nouns_with_status(
    text: str,
) -> list[dict[str, str]]:
    """提取關鍵名詞並附上動詞方向狀態。

    v0.5+ 改進：KEY 名詞不再只是字串列表，而是 `{item, status}` 結構。
    狀態由 `_detect_noun_status()` 從上下文判斷，不需要 LLM 介入。

    回傳依名詞排序的 list，每個元素 `{"item": ..., "status": ...}`。
    對 token 檢索（retrieve_top_k_endpoints）仍會用純字串版的
    `extract_key_nouns()`——那邊只需要 token 命中，不需要狀態。
    """
    items = extract_key_nouns(text)
    return [
        {"item": noun, "status": _detect_noun_status(text, noun)}
        for noun in items
    ]


def _format_key_noun_with_status(entry: dict[str, str]) -> str:
    """把一條 `{item, status}` 轉成 `noun [status]` 字串；
    狀態為 `mentioned`（預設）時省略標記以降低噪音。
    """
    item = entry.get("item", "")
    status = entry.get("status", "mentioned")
    if not item:
        return ""
    if status == "mentioned":
        return item
    return f"{item} [{status}]"


def _extract_first_sentence(text: str, max_len: int = 80) -> str:
    """取第一句有意義的話。保留關鍵名詞，不在詞中間截斷。

    規則：
    1. 找第一個句末標點，如果整句 <= max_len 就用
    2. 太長就找逗號截斷
    3. 都不行就找最後一個空格截斷（不切詞）
    """
    if not text:
        return ""

    text = text.strip()
    if len(text) <= max_len:
        return text

    # 找句末標點
    for i, ch in enumerate(text):
        if ch in ".?!。？！" and i > 10:
            sentence = text[: i + 1].strip()
            if len(sentence) <= max_len:
                return sentence
            break  # 第一句太長，往下走

    # 找逗號
    last_comma = -1
    for i, ch in enumerate(text):
        if ch in ",，、;" and 10 < i < max_len:
            last_comma = i
    if last_comma > 0:
        return text[:last_comma].strip()

    # 找空格（不切詞）
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.5:
        return cut[:last_space].strip()

    return cut.strip()


def encode_endpoint(
    user_input: str,
    result: str,
    topic: str,
    zone: str,
    session_date: str,
    preferences: list[dict[str, Any]],
    endpoint_index: int = 0,
    session_ref: str = "",
    turn_ref: str = "",
    derived_goal: str = "",
) -> str:
    """把一筆端點編碼成結構化標記。純程式碼。

    格式：
        [E{序號}] {日期} | {topic} | zone:{zone}
        GOAL: {使用者原話}
        DERIVED: {逗福理解的目標}   ← 僅在與原話不同時輸出（v0.9 修復二）
        RESULT: {一句短語}
        PREF_EXP: {明確偏好}
        PREF_IMP: {隱式偏好}
        AVOID: {排除偏好}
        KEY: {關鍵名詞}
        SRC: {session-turn ref}  ← 回查用

    v0.9 修復二：GOAL 必須是使用者原話。改寫過的 goal（例如把
    「我就直接取消，改天再說」改成「完成爬山活動」）會讓每一輪撈到
    這條端點的模型讀到與事實相反的記憶。逗福的詮釋有參考價值，
    保留在 DERIVED 欄位並標明來源，兩者衝突時以 GOAL 為準
    （優先序寫在 CODEBOOK 說明區塊）。
    """
    lines: list[str] = []

    # 標題行
    zone_char = zone[0].upper() if zone else "U"
    lines.append(
        f"[E{endpoint_index:03d}] {session_date} | {topic} | zone:{zone_char}"
    )

    # GOAL（原話）+ DERIVED（逗福詮釋，僅在不同時輸出，避免重複佔 token）
    # GOAL 上限 160：原話比改寫後的 goal 長，80 字上限可能把句尾的
    # 決定性轉折（「取消」「不要」「改天再說」）切掉，重演資訊遺失
    # （Copilot review #11）。實測 50 條敘事原話最長 42 字，160 對現有
    # 語料零成本。DERIVED 是壓縮過的詮釋，維持 80。
    g = _extract_first_sentence(user_input, max_len=160)
    r = _extract_first_sentence(result, max_len=100)
    lines.append(f"GOAL: {g}")
    if derived_goal and derived_goal.strip() != (user_input or "").strip():
        d = _extract_first_sentence(derived_goal, max_len=80)
        if d and d != g:
            lines.append(f"DERIVED: {d}")
    lines.append(f"RESULT: {r}")

    # PREF 欄位
    explicit = [p["item"] for p in preferences if p.get("type") == "explicit" and p.get("item")]
    implicit = [p["item"] for p in preferences if p.get("type") == "implicit" and p.get("item")]
    exclusion = [p["item"] for p in preferences if p.get("type") == "exclusion" and p.get("item")]
    if explicit:
        lines.append(f"PREF_EXP: {', '.join(explicit)}")
    if implicit:
        lines.append(f"PREF_IMP: {', '.join(implicit)}")
    if exclusion:
        lines.append(f"AVOID: {', '.join(exclusion)}")

    # KEY 欄位——user_input 和 result **分開**提取關鍵名詞，
    # 再合併去重。避免 user_input 的動詞（例如「stopped using my Leica」）
    # 跨過句界污染 result 的名詞（例如「Recommended Kodak Portra 400」）。
    # 附上由上下文動詞偵測出來的狀態標記（v0.5+）。
    merged_entries: list[dict[str, str]] = []
    seen_items: set[str] = set()
    for source in (user_input, result):
        for entry in extract_key_nouns_with_status(source):
            item = entry.get("item", "")
            if not item or item in seen_items:
                continue
            merged_entries.append(entry)
            seen_items.add(item)
    if merged_entries:
        rendered = [_format_key_noun_with_status(e) for e in merged_entries]
        rendered = [s for s in rendered if s]
        if rendered:
            lines.append(f"KEY: {', '.join(rendered)}")

    # SRC（回查用，不送 LLM 時可省略）
    if session_ref or turn_ref:
        lines.append(f"SRC: {session_ref}-{turn_ref}")

    return "\n".join(lines)


# 記憶編碼的 codebook（放在 system prompt 裡讓 LLM 理解格式）
MEMORY_CODEBOOK = """## Memory Format Legend
The records below are YOUR OWN memory — notes you (Tofu) wrote down
in past interactions with this user. They are not external input,
not a test, not fabricated data.
PREF_EXP: items the user explicitly said they like
PREF_IMP: items inferred from user behavior (bought, watched, spent time on)
AVOID: items the user wants to exclude or dislikes
KEY: exact nouns from original conversation — never ignore these
GOAL: the user's original words — highest authority
DERIVED: your own interpretation at the time. It may contain things
  the user never said — that is expected, it is your past inference.
  When GOAL and DERIVED conflict, GOAL wins.
RESULT: what you yourself replied in that interaction. Read it to
  keep continuity. Do not audit or disown it — if your past reply
  was off, just do better this round without announcing it."""


# v0.5：LLM 每次 call 送出的密碼字典（CODEBOOK），≈ 50 tokens
# 依 20260416_端點檢索與元動機偵測_邏輯鏈_v0_5.md 第一塊定義。
# 這條字串會注入 system prompt 讓 LLM 理解密碼表欄位的意義。
#
# v0.5+：KEY 名詞可能附帶 `[status]` 標記（純程式碼偵測，非 LLM 判斷），
# acquired / active / divested / abandoned 代表使用者對該物件的狀態方向。
CODEBOOK = """## CODEBOOK
The records below are YOUR OWN memory — notes you (Tofu) wrote down
in past interactions with this user. They are not external input,
not a test, not fabricated data.
GOAL: the user's original words — highest authority
DERIVED: your own interpretation at the time. It may contain things
  the user never said — that is expected, it is your past inference.
  When GOAL and DERIVED conflict, GOAL wins.
RESULT: what you yourself replied in that interaction. Read it to
  keep continuity. Do not audit or disown it — if your past reply
  was off, just do better this round without announcing it.
PREF_EXP: items the user explicitly said they like
PREF_IMP: items inferred from user behavior
AVOID: items the user wants to exclude
KEY: exact nouns from original conversation — never ignore these
  status markers: [acquired] just got, [active] still using,
  [divested] sold off, [abandoned] stopped using. No marker = mentioned.
SRC: source reference (for system use, not for response)"""


def _build_end_data_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """建立 event_id → end_data 的映射表。

    因為 append_start / append_end 把 start 和 end 存成獨立的 row，
    start row 的 end_data 永遠是 None。retrieval / encoding 需要
    result 文字（提升命中率和密碼表 RESULT 欄位品質），所以要先
    把同一 event_id 的 end row 的 end_data 查出來接上。
    """
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("type") == "end" and row.get("event_id"):
            ed = row.get("end_data")
            if isinstance(ed, dict):
                mapping[row["event_id"]] = ed
    return mapping


def _iter_start_endpoints(
    rows: list[dict[str, Any]],
    end_map: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """只取 start 類型、帶 start_data 的 row，並確保冷卻欄位存在。

    如果提供了 end_map（event_id → end_data），會把對應的 end_data
    接到 start row 上（就地修改記憶體副本，不寫回磁碟），讓下游的
    _collect_endpoint_terms 和 encode 能讀到 result 文字。
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type") != "start":
            continue
        if not row.get("start_data"):
            continue
        _ensure_cooldown_fields(row)
        # 補上配對的 end_data（如果有且尚未接上）
        if end_map and not row.get("end_data"):
            eid = row.get("event_id")
            if eid and eid in end_map:
                row["end_data"] = end_map[eid]
        out.append(row)
    return out


def _count_term_hits(query_tokens: set[str], ep_terms: set[str]) -> int:
    """計算查詢詞與端點詞之間的命中數。

    v0.6 PR-H audit #2 修：單純 set 交集會漏掉 jieba 黏詞狀況——
    - 查詢「攝影」→ tokens = {'攝影'}
    - 端點「學攝影」→ jieba 可能把它黏成 {'學攝影'}
    - 交集空 → 誤判為零命中

    修法：先跑精確交集（快、零誤判），針對沒命中的查詢詞再做一次子字串
    備援比對（query_token 是 endpoint_token 的子字串、或反之）。
    每個查詢詞最多算 1 次命中，避免黏詞被反覆計分膨脹分數。

    安全性：
    - 只對長度 ≥ 2 的 token 做子字串比對，避免「書」誤命中「讀書」
    - tokenizer 已排除停用詞與 min_token_len<2，這裡再加一層保險
    """
    if not query_tokens or not ep_terms:
        return 0
    exact = query_tokens & ep_terms
    hit_count = len(exact)
    for qt in query_tokens - exact:
        if len(qt) < 2:
            continue
        for et in ep_terms:
            if len(et) < 2:
                continue
            if qt in et or et in qt:
                hit_count += 1
                break  # 每個查詢詞最多 +1，避免黏詞膨脹
    return hit_count


def _collect_endpoint_terms(row: dict[str, Any]) -> set[str]:
    """把一個端點裡所有可供比對的詞蒐集起來。

    v0.5 第四層的比對來源：
    - start_data.goal / user_input / constraints / tofu_understanding
    - end_data.result（如果已寫入，提供更多命中）
    - KEY 名詞（由 extract_key_nouns 再跑一次，確保專有名詞直接參與比對）
    """
    sd = row.get("start_data") or {}
    text_parts: list[str] = []
    for key in ("goal", "user_input", "tofu_understanding", "motivation"):
        val = sd.get(key)
        if isinstance(val, str):
            text_parts.append(val)
    for c in sd.get("constraints") or []:
        if isinstance(c, str):
            text_parts.append(c)
    ed = row.get("end_data") or {}
    result = ed.get("result") if isinstance(ed, dict) else None
    if isinstance(result, str):
        text_parts.append(result)

    blob = " ".join(text_parts)
    tokens = set(_tokenize(blob))
    # KEY 名詞 lower-case 化加入以提升專有名詞命中率
    for kn in extract_key_nouns(blob):
        tokens.add(kn.lower())
    return tokens


def retrieve_top_k_endpoints(
    rows: list[dict[str, Any]],
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    recent_floor_n: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """第四層強制查表 Gate。

    回傳 (top_k 端點 list, 已知 gap_categories list)。
    - 端點依命中詞數排序（pending_confirmation 降權後排序）
    - 所有詞同等重要，一個詞就是一個命中
    - 即使查不到任何結果，也必須呼叫過此函式——查表動作不能省

    v0.5 的廣度保證由外層的策略摘要（baseline 全量自帶率統計）提供；
    此函式只管深度（跟這一題最相關的 top-k 具體事實）。
    """
    query_tokens = set(_tokenize(query or ""))
    end_map = _build_end_data_map(rows)
    starts = _iter_start_endpoints(rows, end_map=end_map)
    if not starts:
        return [], []

    scored: list[tuple[float, int, int, dict[str, Any]]] = []
    for idx, row in enumerate(starts):
        if row.get("status") == "superseded":
            # superseded 不參與檢索（使用者已否認）
            continue
        ep_terms = _collect_endpoint_terms(row)
        if not ep_terms:
            continue
        if query_tokens:
            hit = _count_term_hits(query_tokens, ep_terms)
        else:
            # 查詢詞為空：回退到全量取樣（依時間最新優先）
            hit = 1

        score = float(hit)
        if row.get("status") == "pending_confirmation":
            score *= _PENDING_PENALTY
        if score <= 0:
            continue
        ref_count = int(row.get("reference_count") or 0)
        # recency_rank：idx 越大越新，用它做第三排序鍵
        recency_rank = idx
        scored.append((score, ref_count, recency_rank, row))

    # 依 (score, reference_count, recency) 降序排列。
    # recency_rank 越大代表越新→同分時最近的端點排最前面。
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    top_rows = [row for _score, _rc, _rec, row in scored[: max(0, int(top_k))]]

    # === A：近因保底（recent_floor_n>0 才啟用）===
    # 先做完 top_k 截斷，再補近因——近因是「額外保證」，不受 top_k 上限約束；
    # 最終長度可為 top_k + 補入數（補入數 ≤ recent_floor_n）。
    if recent_floor_n > 0:
        got_ids = {r.get("endpoint_id") for r in top_rows}
        # starts 已是依出現順序；末尾 N 筆 = 最近 N 筆，排除 superseded
        recent = [
            r for r in starts if r.get("status") != "superseded"
        ][-recent_floor_n:]
        for r in reversed(recent):  # 最近的擺最前
            if r.get("endpoint_id") not in got_ids:
                top_rows.insert(0, r)  # 插最前 = 最高優先（延續相關性最高）
                got_ids.add(r.get("endpoint_id"))

    # 已知 gap_categories：把 top-k 裡每個端點的 gap_categories 合併
    # （改用補入後的 top_rows，順序無影響）
    known: list[str] = []
    seen: set[str] = set()
    for row in top_rows:
        sd = row.get("start_data") or {}
        for cat in sd.get("gap_categories") or []:
            if not cat:
                continue
            low = str(cat).strip().lower()
            if low and low not in seen:
                known.append(low)
                seen.add(low)
    return top_rows, known


def encode_retrieved_endpoints(
    rows: list[dict[str, Any]],
    session_date: str | None = None,
    all_rows: list[dict[str, Any]] | None = None,
) -> str:
    """把 retrieve_top_k_endpoints 的輸出壓成「密碼表」文字。

    每個端點走 encode_endpoint() 產出一段結構化標記，之間用空行隔開。
    v0.5 規定：送到 LLM 的不是端點原文，是這份密碼表——
    244 筆全量 ~73K chars → top-30 編碼後 ~9K chars，壓縮比 8:1 起跳。

    all_rows: 如果提供，用於建立 event_id → end_data 映射，
    確保 start-only rows 也能拿到配對的 result。retrieve_top_k_endpoints
    已在內部做過 join，但直接建構 rows list 的 caller 可能漏了。
    """
    if not rows:
        return ""
    # 如果 caller 提供了完整 rows，建 end_data 映射並補上未接的 end_data
    if all_rows:
        end_map = _build_end_data_map(all_rows)
        for row in rows:
            if not row.get("end_data") and row.get("event_id"):
                eid = row["event_id"]
                if eid in end_map:
                    row["end_data"] = end_map[eid]
    blocks: list[str] = []
    for idx, row in enumerate(rows):
        sd = row.get("start_data") or {}
        ed = row.get("end_data") or {}
        # date 取端點自身 timestamp 的 yyyy-mm-dd 部分
        ts = row.get("timestamp") or ""
        if ts and len(ts) >= 10:
            date_part = ts[:10]
        else:
            date_part = session_date or ""
        topic = sd.get("topic") or "general"
        zone = sd.get("zone") or "B"
        # preferences：v0.3 端點沒有這個欄位，從 start_data 拼一個最小版出來
        # 送 LLM 時至少保留 KEY / GOAL / RESULT 三個關鍵欄位
        raw_prefs = sd.get("preferences") or []
        if not isinstance(raw_prefs, list):
            raw_prefs = []

        # v0.9 修復二：原話優先。goal 是 LLM 改寫過的版本（「我就直接取消」
        # 會變成「完成爬山活動」），只能當 fallback（舊端點沒有 user_input）。
        block = encode_endpoint(
            user_input=sd.get("user_input") or sd.get("goal") or "",
            derived_goal=sd.get("goal") or "",
            result=(ed.get("result") if isinstance(ed, dict) else "") or "",
            topic=topic,
            zone=zone,
            session_date=date_part,
            preferences=raw_prefs,
            endpoint_index=idx + 1,
            session_ref=row.get("event_id") or "",
            turn_ref=str(row.get("round_last_referenced") or ""),
        )
        if row.get("status") == "pending_confirmation":
            block += "\nSTATUS: pending_confirmation"
        blocks.append(block)
    return "\n\n".join(blocks)
