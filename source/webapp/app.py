"""拼圖定位助手 Web App 後端（FastAPI）。

整併既有定位器與單片採集流程，提供專案管理、拍片定位、確認標記、進度查詢。
資料存 SQLite + 檔案系統（見 source/webapp/store.py）；定位以依賴注入接入
（見 source/webapp/locate_service.py），測試可注入快速 stub。

規格見 doc/webapp_spec.md。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from source.webapp.store import Store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "webapp"
DEFAULT_STATIC_DIR = PROJECT_ROOT / "static" / "webapp"

MAX_UPLOAD_BYTES = 25_000_000
_ALLOWED_EXT = {"JPEG": ".jpg", "PNG": ".png"}


def _decode_upload(contents: bytes) -> tuple[Image.Image, str]:
    """解碼上傳影像、套 EXIF 方向校正，回傳 (PIL RGB, 規範副檔名)。非影像則丟 400。"""
    if not contents:
        raise HTTPException(status_code=400, detail="影像內容為空")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="影像過大")
    try:
        im = Image.open(io.BytesIO(contents))
        fmt = im.format or "JPEG"
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="無法解析影像，請重新上傳")
    ext = _ALLOWED_EXT.get(fmt, ".jpg")
    return im, ext


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def create_app(
    data_root: Path | str | None = None,
    static_dir: Path | str | None = None,
    locator: Optional[Callable[[np.ndarray, np.ndarray, Optional[int], Optional[int]], dict]] = None,
) -> FastAPI:
    """建立 App。`locator(reference_bgr, piece_bgr, rows, cols) -> dict` 可注入以加速測試。"""
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR
    store = Store(data_root)

    if locator is None:
        from source.webapp.locate_service import locate_piece_image as locator  # noqa: PLW2901

    app = FastAPI(title="Jigsaw Puzzle Locator")
    app.state.store = store

    # ---------- 靜態頁面 ----------
    @app.get("/")
    def index():
        index_html = static_dir / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="index.html 未找到")
        return FileResponse(index_html)

    # ---------- 專案 ----------
    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list_projects()}

    @app.post("/api/projects")
    async def create_project(
        name: str = Form(...),
        rows: int = Form(...),
        cols: int = Form(...),
        reference: UploadFile = None,  # type: ignore[assignment]
    ):
        name = (name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="專案名稱不可空白")
        if rows < 1 or cols < 1:
            raise HTTPException(status_code=400, detail="列數與行數須為正整數")
        if rows * cols > 100_000:
            raise HTTPException(status_code=400, detail="片數過大（rows×cols 上限 100000）")
        if reference is None:
            raise HTTPException(status_code=400, detail="缺少完成圖")
        contents = await reference.read()
        im, ext = _decode_upload(contents)

        pid = store.create_project(name, rows, cols, ext)
        ref_path = store.reference_path(pid, ext)
        im.save(ref_path, "JPEG" if ext == ".jpg" else "PNG", quality=92)
        return store.get_project(pid)

    @app.get("/api/projects/{pid}")
    def get_project(pid: int):
        proj = store.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        return proj

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: int):
        if not store.delete_project(pid):
            raise HTTPException(status_code=404, detail="找不到專案")
        return {"ok": True}

    @app.get("/api/projects/{pid}/reference")
    def get_reference(pid: int):
        proj = store.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        path = store.reference_path(pid, proj["reference_ext"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="完成圖遺失")
        return FileResponse(path)

    # ---------- 定位 ----------
    @app.post("/api/projects/{pid}/locate")
    async def locate(pid: int, image: UploadFile):
        proj = store.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        ref_path = store.reference_path(pid, proj["reference_ext"])
        ref_bgr = cv2.imread(str(ref_path))
        if ref_bgr is None:
            raise HTTPException(status_code=500, detail="無法讀取完成圖")

        contents = await image.read()
        im, ext = _decode_upload(contents)
        piece_bgr = _pil_to_bgr(im)

        try:
            res = locator(ref_bgr, piece_bgr, proj["rows"], proj["cols"])
        except Exception as e:  # 定位內部錯誤不應讓整支 API 500 無訊息
            raise HTTPException(status_code=500, detail=f"定位失敗：{e}")

        piece_id = store.create_piece(
            project_id=pid,
            image_ext=ext,
            pred_row=res.get("pred_row"),
            pred_col=res.get("pred_col"),
            confidence=res.get("confidence"),
            method=res.get("method"),
            certain=bool(res.get("certain")),
            region_hint=res.get("region_hint"),
            top_cells=res.get("top_cells"),
        )
        # 存單片裁切圖供 UI 與清單顯示
        im.save(store.piece_path(pid, piece_id, ext), "JPEG" if ext == ".jpg" else "PNG", quality=92)

        out = dict(res)
        out["piece_id"] = piece_id
        return JSONResponse(out)

    # ---------- 單片 ----------
    @app.get("/api/projects/{pid}/pieces")
    def list_pieces(pid: int):
        if not store.get_project(pid):
            raise HTTPException(status_code=404, detail="找不到專案")
        return {
            "pieces": store.list_pieces(pid),
            "confirmed_cells": store.confirmed_cells(pid),
        }

    @app.get("/api/projects/{pid}/pieces/{piece_id}/image")
    def get_piece_image(pid: int, piece_id: int):
        piece = store.get_piece(pid, piece_id)
        if not piece:
            raise HTTPException(status_code=404, detail="找不到單片")
        path = store.piece_path(pid, piece_id, piece["image_ext"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="單片影像遺失")
        return FileResponse(path)

    @app.post("/api/projects/{pid}/pieces/{piece_id}/confirm")
    def confirm_piece(
        pid: int,
        piece_id: int,
        row: int = Body(...),
        col: int = Body(...),
        confirmed: bool = Body(True),
    ):
        proj = store.get_project(pid)
        if not proj:
            raise HTTPException(status_code=404, detail="找不到專案")
        if confirmed:
            if not (1 <= row <= proj["rows"]):
                raise HTTPException(status_code=400, detail=f"列須介於 1~{proj['rows']}")
            if not (1 <= col <= proj["cols"]):
                raise HTTPException(status_code=400, detail=f"行須介於 1~{proj['cols']}")
        piece = store.confirm_piece(pid, piece_id, row, col, confirmed)
        if not piece:
            raise HTTPException(status_code=404, detail="找不到單片")
        return {"ok": True, "piece": piece, "project": store.get_project(pid)}

    @app.delete("/api/projects/{pid}/pieces/{piece_id}")
    def delete_piece(pid: int, piece_id: int):
        if not store.delete_piece(pid, piece_id):
            raise HTTPException(status_code=404, detail="找不到單片")
        return {"ok": True}

    return app


app = create_app()
