"""拼圖單片測試資料集採集 Web App 後端。

提供手機/桌機瀏覽器拍攝單片拼圖、手動裁切、輸入行列後，
以現有命名慣例 `pieces_c{col}_r{row}.jpg` 存入 data 目錄當測試資料。

與既有去背 / 定位流程無關，獨立成模組（不動 source/app.py）。
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# 專案根目錄（.../jp_locator），不依賴執行時的 cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STATIC_DIR = PROJECT_ROOT / "static" / "capture"
CONFIG_FILENAME = "project_config.json"

DEFAULT_CONFIG = {"rows": 40, "cols": 25, "total_pieces": 1000}

# 沿用 data/ 既有命名：pieces_c{col}_r{row}.jpg，序號版 pieces_c{col}_r{row}_{n}.jpg
_PIECE_RE = re.compile(r"^pieces_c(?P<col>\d+)_r(?P<row>\d+)(?:_(?P<seq>\d+))?\.jpg$", re.IGNORECASE)


def load_config(data_dir: Path) -> dict:
    """讀 project_config.json，缺失或損毀時回預設值。"""
    config_path = data_dir / CONFIG_FILENAME
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                "rows": int(data.get("rows", DEFAULT_CONFIG["rows"])),
                "cols": int(data.get("cols", DEFAULT_CONFIG["cols"])),
                "total_pieces": int(data.get("total_pieces", DEFAULT_CONFIG["total_pieces"])),
            }
        except (ValueError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def resolve_filename(data_dir: Path, col: int, row: int) -> Path:
    """算出不衝突的存檔路徑。

    首選 pieces_c{col}_r{row}.jpg；已存在則依序試 _1、_2…（加序號保留多版）。
    """
    base = data_dir / f"pieces_c{col}_r{row}.jpg"
    if not base.exists():
        return base
    seq = 1
    while True:
        candidate = data_dir / f"pieces_c{col}_r{row}_{seq}.jpg"
        if not candidate.exists():
            return candidate
        seq += 1


def list_captures(data_dir: Path) -> list[dict]:
    """掃 data/ 下所有單片圖，依 (col,row) 彙整每格樣本數與檔名。"""
    cells: dict[tuple[int, int], list[str]] = {}
    if data_dir.exists():
        for path in sorted(data_dir.glob("pieces_c*_r*.jpg")):
            m = _PIECE_RE.match(path.name)
            if not m:
                continue
            key = (int(m.group("col")), int(m.group("row")))
            cells.setdefault(key, []).append(path.name)
    return [
        {"col": col, "row": row, "count": len(files), "files": files}
        for (col, row), files in sorted(cells.items())
    ]


def create_app(data_dir: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR

    app = FastAPI(title="Jigsaw Piece Dataset Collector")

    @app.get("/")
    def index():
        index_html = static_dir / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="index.html 未找到")
        return FileResponse(index_html)

    @app.get("/api/config")
    def get_config():
        return load_config(data_dir)

    @app.get("/api/captures")
    def get_captures():
        return {"cells": list_captures(data_dir)}

    @app.post("/api/captures")
    async def post_capture(
        image: UploadFile,
        col: int = Form(...),
        row: int = Form(...),
    ):
        config = load_config(data_dir)
        if not (1 <= col <= config["cols"]):
            raise HTTPException(status_code=400, detail=f"col 須介於 1~{config['cols']}（收到 {col}）")
        if not (1 <= row <= config["rows"]):
            raise HTTPException(status_code=400, detail=f"row 須介於 1~{config['rows']}（收到 {row}）")

        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="影像內容為空")

        # 驗證可解碼（用 Pillow，避免硬綁 cv2）
        try:
            from PIL import Image

            with Image.open(io.BytesIO(contents)) as im:
                im.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="無法解析影像，請重拍")

        data_dir.mkdir(parents=True, exist_ok=True)
        target = resolve_filename(data_dir, col, row)
        target.write_bytes(contents)

        total_in_cell = next(
            (c["count"] for c in list_captures(data_dir) if c["col"] == col and c["row"] == row),
            0,
        )
        return JSONResponse(
            {
                "filename": target.name,
                "saved_path": str(target),
                "total_in_cell": total_in_cell,
            }
        )

    return app


app = create_app()
