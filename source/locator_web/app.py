"""Puzzle Locator 本機應用後端（FastAPI）。

服務行動端 UI（static/locator/index.html）並提供 REST API：
  - 專案 CRUD 與完成圖（reference）上傳/讀取
  - 單片定位（/locate，呼叫現有去背 + locate_piece 管線，不落地）
  - 已定位單片的儲存/查詢/切換確認/刪除

資料：SQLite（data/locator.db）+ 每專案影像目錄 data/projects/{id}/。
與既有 capture / 定位流程共用底層模組，但獨立成新 app（不動 source/capture/app.py）。
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from source.locator_web.locate_service import run_locate
from source.locator_web.store import Store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STATIC_DIR = PROJECT_ROOT / "static" / "locator"
DEFAULT_DB = DEFAULT_DATA_DIR / "locator.db"

MAX_UPLOAD_BYTES = 25_000_000  # 與 capture app 一致，避免綁 0.0.0.0 時記憶體耗盡


def _decode_upload(contents: bytes) -> np.ndarray:
    """解碼上傳影像為 BGR ndarray，套 EXIF 方向校正；失敗丟 400。"""
    if not contents:
        raise HTTPException(status_code=400, detail="影像內容為空")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="影像過大")
    try:
        im = Image.open(io.BytesIO(contents))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="無法解析影像，請重拍")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _save_jpeg(contents: bytes, target: Path, quality: int = 92) -> None:
    """重新編碼存 JPEG（套 EXIF 校正），確保內容與副檔名一致。"""
    try:
        im = Image.open(io.BytesIO(contents))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="無法解析影像")
    target.parent.mkdir(parents=True, exist_ok=True)
    im.save(target, "JPEG", quality=quality)


def create_app(db_path: Path | None = None,
               data_dir: Path | None = None,
               static_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR
    store = Store(db_path or DEFAULT_DB)
    projects_dir = data_dir / "projects"

    app = FastAPI(title="Puzzle Locator")

    def _project_dir(pid: int) -> Path:
        return projects_dir / str(pid)

    def _require_project(pid: int) -> dict:
        proj = store.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        return proj

    # ---------- static ----------
    @app.get("/")
    def index():
        index_html = static_dir / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="index.html 未找到")
        return FileResponse(index_html)

    # ---------- projects ----------
    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list_projects()}

    @app.post("/api/projects")
    async def create_project(
        name: str = Form(...),
        rows: int = Form(...),
        cols: int = Form(...),
        reference: Optional[UploadFile] = File(None),
    ):
        if not name.strip():
            raise HTTPException(status_code=400, detail="專案名稱不可為空")
        if rows < 1 or cols < 1:
            raise HTTPException(status_code=400, detail="列數與行數須為正整數")

        proj = store.create_project(name.strip(), rows, cols)
        if reference is not None:
            contents = await reference.read()
            if contents:
                if len(contents) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="完成圖過大")
                ref_path = _project_dir(proj["id"]) / "reference.jpg"
                _save_jpeg(contents, ref_path)
                store.update_project_ref(proj["id"], str(ref_path.relative_to(data_dir)))
                proj = store.get_project(proj["id"])
        return proj

    @app.get("/api/projects/{pid}")
    def get_project(pid: int):
        return _require_project(pid)

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: int):
        _require_project(pid)
        store.delete_project(pid)
        shutil.rmtree(_project_dir(pid), ignore_errors=True)
        return {"ok": True}

    @app.get("/api/projects/{pid}/reference")
    def get_reference(pid: int):
        proj = _require_project(pid)
        if not proj["ref_path"]:
            raise HTTPException(status_code=404, detail="此專案尚未上傳完成圖")
        path = data_dir / proj["ref_path"]
        if not path.exists():
            raise HTTPException(status_code=404, detail="完成圖檔案遺失")
        return FileResponse(path)

    @app.put("/api/projects/{pid}/settings")
    async def update_settings(pid: int, settings: dict):
        proj = store.update_settings(pid, settings or {})
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        return proj

    # ---------- locate（不落地，只回結果）----------
    @app.post("/api/projects/{pid}/locate")
    async def locate(pid: int, image: UploadFile = File(...)):
        proj = _require_project(pid)
        if not proj["ref_path"]:
            raise HTTPException(status_code=400, detail="此專案尚未上傳完成圖，無法定位")
        ref_path = data_dir / proj["ref_path"]
        ref_bgr = cv2.imread(str(ref_path))
        if ref_bgr is None:
            raise HTTPException(status_code=400, detail="完成圖無法讀取")

        piece_bgr = _decode_upload(await image.read())
        try:
            result = run_locate(ref_bgr, piece_bgr, rows=proj["rows"], cols=proj["cols"])
        except Exception as e:  # 定位流程例外不該打掛伺服器
            raise HTTPException(status_code=500, detail=f"定位失敗：{e}")
        return result

    # ---------- pieces ----------
    @app.get("/api/projects/{pid}/pieces")
    def list_pieces(pid: int):
        _require_project(pid)
        return {"pieces": store.list_pieces(pid)}

    @app.post("/api/projects/{pid}/pieces")
    async def add_piece(
        pid: int,
        row: int = Form(...),
        col: int = Form(...),
        conf: float = Form(0.0),
        confirmed: bool = Form(True),
        image: Optional[UploadFile] = File(None),
    ):
        proj = _require_project(pid)
        if not (1 <= row <= proj["rows"]):
            raise HTTPException(status_code=400, detail=f"row 須介於 1~{proj['rows']}")
        if not (1 <= col <= proj["cols"]):
            raise HTTPException(status_code=400, detail=f"col 須介於 1~{proj['cols']}")

        image_rel = None
        if image is not None:
            contents = await image.read()
            if contents:
                if len(contents) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="影像過大")
                fname = f"r{row}_c{col}.jpg"
                target = _project_dir(pid) / "pieces" / fname
                _save_jpeg(contents, target)
                image_rel = str(target.relative_to(data_dir))

        return store.add_piece(pid, row, col, conf, f"R{row}C{col}",
                               image_path=image_rel, confirmed=confirmed)

    @app.patch("/api/projects/{pid}/pieces/{piece_id}")
    def patch_piece(pid: int, piece_id: int, payload: dict):
        _require_project(pid)
        if "confirmed" not in payload:
            raise HTTPException(status_code=400, detail="缺少 confirmed 欄位")
        piece = store.set_piece_confirmed(piece_id, bool(payload["confirmed"]))
        if not piece:
            raise HTTPException(status_code=404, detail="找不到單片")
        return piece

    @app.delete("/api/projects/{pid}/pieces/{piece_id}")
    def delete_piece(pid: int, piece_id: int):
        _require_project(pid)
        if not store.delete_piece(piece_id):
            raise HTTPException(status_code=404, detail="找不到單片")
        return {"ok": True}

    return app


app = create_app()
