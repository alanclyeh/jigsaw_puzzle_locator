"""拼圖定位助手 Web App 的儲存層（SQLite + 檔案系統）。

職責切分：
- SQLite（單檔 `jp.db`）只存 metadata（專案、單片定位/確認紀錄）。
- 完成大圖與單片裁切圖存在檔案系統 `projects/{pid}/`，DB 僅記副檔名。

設計成可注入根目錄（`Store(root)`），測試以 tmp_path 建庫，不污染專案 data/。
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL,
  rows          INTEGER NOT NULL,
  cols          INTEGER NOT NULL,
  reference_ext TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pieces (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  image_ext   TEXT NOT NULL,
  pred_row    INTEGER,
  pred_col    INTEGER,
  confidence  REAL,
  method      TEXT,
  certain     INTEGER DEFAULT 0,
  region_hint TEXT,
  top_cells   TEXT,
  confirmed   INTEGER DEFAULT 0,
  final_row   INTEGER,
  final_col   INTEGER,
  created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pieces_project ON pieces(project_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """資料存取門面（DB + 影像檔）。所有寫入單一連線、序列化以避免 SQLite 並發問題。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.root / "projects"
        self.projects_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "jp.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------- 路徑 ----------
    def project_dir(self, pid: int) -> Path:
        d = self.projects_dir / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def reference_path(self, pid: int, ext: str) -> Path:
        return self.project_dir(pid) / f"reference{ext}"

    def piece_path(self, pid: int, piece_id: int, ext: str) -> Path:
        return self.project_dir(pid) / f"piece_{piece_id}{ext}"

    # ---------- 專案 ----------
    def create_project(self, name: str, rows: int, cols: int, reference_ext: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO projects(name, rows, cols, reference_ext, created_at) VALUES (?,?,?,?,?)",
            (name, rows, cols, reference_ext, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_project(self, pid: int) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        return self._project_with_progress(dict(row))

    def list_projects(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
        return [self._project_with_progress(dict(r)) for r in rows]

    def _project_with_progress(self, proj: dict) -> dict:
        total = int(proj["rows"]) * int(proj["cols"])
        confirmed = self._conn.execute(
            "SELECT COUNT(*) c FROM pieces "
            "WHERE project_id=? AND confirmed=1 AND final_row IS NOT NULL AND final_col IS NOT NULL",
            (proj["id"],),
        ).fetchone()["c"]
        piece_total = self._conn.execute(
            "SELECT COUNT(*) c FROM pieces WHERE project_id=?", (proj["id"],)
        ).fetchone()["c"]
        proj["pieces"] = total
        proj["confirmed"] = confirmed
        proj["remaining"] = total - confirmed
        proj["pct"] = round(confirmed / total * 100) if total else 0
        proj["captured"] = piece_total  # 已拍/已辨識的單片總數（含未確認）
        return proj

    def delete_project(self, pid: int) -> bool:
        row = self._conn.execute("SELECT id FROM projects WHERE id=?", (pid,)).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM pieces WHERE project_id=?", (pid,))
        self._conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        self._conn.commit()
        d = self.projects_dir / str(pid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        return True

    # ---------- 單片 ----------
    def create_piece(
        self,
        project_id: int,
        image_ext: str,
        pred_row: Optional[int],
        pred_col: Optional[int],
        confidence: Optional[float],
        method: Optional[str],
        certain: bool,
        region_hint: Optional[dict],
        top_cells: Optional[list],
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO pieces(project_id, image_ext, pred_row, pred_col, confidence, method, "
            "certain, region_hint, top_cells, confirmed, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,0,?)",
            (
                project_id,
                image_ext,
                pred_row,
                pred_col,
                confidence,
                method,
                1 if certain else 0,
                json.dumps(region_hint) if region_hint is not None else None,
                json.dumps(top_cells) if top_cells is not None else None,
                _now(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_piece(self, project_id: int, piece_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM pieces WHERE id=? AND project_id=?", (piece_id, project_id)
        ).fetchone()
        return self._piece_to_dict(row) if row else None

    def list_pieces(self, project_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM pieces WHERE project_id=? ORDER BY id DESC", (project_id,)
        ).fetchall()
        return [self._piece_to_dict(r) for r in rows]

    def confirm_piece(
        self, project_id: int, piece_id: int, row: int, col: int, confirmed: bool = True
    ) -> Optional[dict]:
        """確認/修正落點。confirmed=True 時，先把同格其他已確認片取消（每格唯一）。"""
        piece = self._conn.execute(
            "SELECT id FROM pieces WHERE id=? AND project_id=?", (piece_id, project_id)
        ).fetchone()
        if not piece:
            return None
        if confirmed:
            # 同格唯一：取消同 (row,col) 的其他已確認片
            self._conn.execute(
                "UPDATE pieces SET confirmed=0 WHERE project_id=? AND final_row=? AND final_col=? AND id<>?",
                (project_id, row, col, piece_id),
            )
            self._conn.execute(
                "UPDATE pieces SET confirmed=1, final_row=?, final_col=? WHERE id=?",
                (row, col, piece_id),
            )
        else:
            # 取消確認，保留 final_row/col 作為紀錄
            self._conn.execute(
                "UPDATE pieces SET confirmed=0, final_row=?, final_col=? WHERE id=?",
                (row, col, piece_id),
            )
        self._conn.commit()
        return self.get_piece(project_id, piece_id)

    def set_confirmed_flag(self, project_id: int, piece_id: int, confirmed: bool) -> Optional[dict]:
        """純粹切換已確認旗標（沿用 final_row/col），供清單 toggle 用。"""
        row = self._conn.execute(
            "SELECT final_row, final_col FROM pieces WHERE id=? AND project_id=?",
            (piece_id, project_id),
        ).fetchone()
        if not row:
            return None
        if row["final_row"] is None or row["final_col"] is None:
            return None  # 沒有落點不能標已確認
        if confirmed:
            self._conn.execute(
                "UPDATE pieces SET confirmed=0 WHERE project_id=? AND final_row=? AND final_col=? AND id<>?",
                (project_id, row["final_row"], row["final_col"], piece_id),
            )
        self._conn.execute(
            "UPDATE pieces SET confirmed=? WHERE id=? AND project_id=?",
            (1 if confirmed else 0, piece_id, project_id),
        )
        self._conn.commit()
        return self.get_piece(project_id, piece_id)

    def delete_piece(self, project_id: int, piece_id: int) -> bool:
        row = self._conn.execute(
            "SELECT image_ext FROM pieces WHERE id=? AND project_id=?", (piece_id, project_id)
        ).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM pieces WHERE id=? AND project_id=?", (piece_id, project_id))
        self._conn.commit()
        p = self.piece_path(project_id, piece_id, row["image_ext"])
        p.unlink(missing_ok=True)
        return True

    def confirmed_cells(self, project_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT final_row r, final_col c FROM pieces "
            "WHERE project_id=? AND confirmed=1 AND final_row IS NOT NULL AND final_col IS NOT NULL",
            (project_id,),
        ).fetchall()
        return [{"row": r["r"], "col": r["c"]} for r in rows]

    @staticmethod
    def _piece_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["certain"] = bool(d.get("certain"))
        d["confirmed"] = bool(d.get("confirmed"))
        for k in ("region_hint", "top_cells"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (ValueError, TypeError):
                    d[k] = None
        return d
