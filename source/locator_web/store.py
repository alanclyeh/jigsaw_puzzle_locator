"""Puzzle Locator 本機應用的 SQLite 持久層。

關聯模型：projects 1:N pieces。
- projects：一個拼圖專案（名稱、列行數、總片數、完成圖路徑、每專案設定）。
- pieces：一次定位並儲存的單片（grid 位置、信心度、影像路徑、是否標記確認）。

設計原則：
- 僅用 Python 內建 sqlite3，無額外依賴。
- 每次操作開一條連線（FastAPI 多執行緒下最簡單且安全）；啟用外鍵級聯。
- 時間一律存 ISO8601 字串（UTC）。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    rows          INTEGER NOT NULL,
    cols          INTEGER NOT NULL,
    total_pieces  INTEGER NOT NULL,
    ref_path      TEXT,
    settings_json TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pieces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    conf        REAL    NOT NULL DEFAULT 0,
    ref_label   TEXT    NOT NULL,
    image_path  TEXT,
    confirmed   INTEGER NOT NULL DEFAULT 1,
    located_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pieces_project ON pieces(project_id);
"""

# 預設每專案設定（對應前端設定頁）
DEFAULT_SETTINGS = {
    "auto_crop": True,
    "grid_guide": True,
    "haptic": False,
    "frame_mode": 1,  # 0 小 / 1 建議 / 2 大
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """SQLite 資料存取。建構時自動建表（冪等）。"""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---------- projects ----------
    def create_project(self, name: str, rows: int, cols: int,
                        ref_path: Optional[str] = None,
                        settings: Optional[dict] = None) -> dict:
        ts = _now()
        merged = {**DEFAULT_SETTINGS, **(settings or {})}
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO projects(name,rows,cols,total_pieces,ref_path,settings_json,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (name, rows, cols, rows * cols, ref_path, json.dumps(merged), ts, ts),
            )
            pid = cur.lastrowid
        return self.get_project(pid)

    def list_projects(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
            return [self._project_dict(conn, r) for r in rows]

    def get_project(self, project_id: int) -> Optional[dict]:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            return self._project_dict(conn, r) if r else None

    def update_project_ref(self, project_id: int, ref_path: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE projects SET ref_path=?, updated_at=? WHERE id=?",
                         (ref_path, _now(), project_id))

    def update_settings(self, project_id: int, settings: dict) -> Optional[dict]:
        with self._connect() as conn:
            r = conn.execute("SELECT settings_json FROM projects WHERE id=?", (project_id,)).fetchone()
            if not r:
                return None
            merged = {**DEFAULT_SETTINGS, **json.loads(r["settings_json"]), **settings}
            conn.execute("UPDATE projects SET settings_json=?, updated_at=? WHERE id=?",
                         (json.dumps(merged), _now(), project_id))
        return self.get_project(project_id)

    def delete_project(self, project_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            return cur.rowcount > 0

    # ---------- pieces ----------
    def add_piece(self, project_id: int, row: int, col: int, conf: float,
                  ref_label: str, image_path: Optional[str] = None,
                  confirmed: bool = True) -> dict:
        ts = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO pieces(project_id,row,col,conf,ref_label,image_path,confirmed,located_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (project_id, row, col, conf, ref_label, image_path, 1 if confirmed else 0, ts),
            )
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (ts, project_id))
            pid = cur.lastrowid
            r = conn.execute("SELECT * FROM pieces WHERE id=?", (pid,)).fetchone()
            return self._piece_dict(r)

    def list_pieces(self, project_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pieces WHERE project_id=? ORDER BY located_at DESC, id DESC",
                (project_id,),
            ).fetchall()
            return [self._piece_dict(r) for r in rows]

    def set_piece_confirmed(self, piece_id: int, confirmed: bool) -> Optional[dict]:
        with self._connect() as conn:
            conn.execute("UPDATE pieces SET confirmed=? WHERE id=?",
                         (1 if confirmed else 0, piece_id))
            r = conn.execute("SELECT * FROM pieces WHERE id=?", (piece_id,)).fetchone()
            return self._piece_dict(r) if r else None

    def delete_piece(self, piece_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM pieces WHERE id=?", (piece_id,))
            return cur.rowcount > 0

    # ---------- helpers ----------
    def _project_dict(self, conn: sqlite3.Connection, r: sqlite3.Row) -> dict:
        confirmed = conn.execute(
            "SELECT COUNT(*) AS c FROM pieces WHERE project_id=? AND confirmed=1", (r["id"],)
        ).fetchone()["c"]
        total = r["total_pieces"]
        pct = round(confirmed / total * 100) if total else 0
        return {
            "id": r["id"],
            "name": r["name"],
            "rows": r["rows"],
            "cols": r["cols"],
            "total_pieces": total,
            "ref_path": r["ref_path"],
            "has_reference": bool(r["ref_path"]),
            "settings": json.loads(r["settings_json"]),
            "confirmed": confirmed,
            "remaining": total - confirmed,
            "pct": pct,
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }

    @staticmethod
    def _piece_dict(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "project_id": r["project_id"],
            "row": r["row"],
            "col": r["col"],
            "conf": r["conf"],
            "ref_label": r["ref_label"],
            "image_path": r["image_path"],
            "confirmed": bool(r["confirmed"]),
            "located_at": r["located_at"],
        }
