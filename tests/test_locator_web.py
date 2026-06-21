"""Puzzle Locator 行動端 App 後端測試（source/locator_web）。

locator_web 刻意不另寫後端，而是薄包裝重用 webapp（API 合約一致、見
tests/test_webapp.py 完整覆蓋）。因此這裡只測「接縫」：
  1. 模組常數指向獨立的 data/locator 與 static/locator（不污染 webapp）。
  2. GET / serve 的是 locator 前端（Claude Design 版），不是 webapp 那支。
  3. 用 locator 的 STATIC_DIR + 注入 stub 定位器，跑一次 建立→定位→確認，
     確認薄包裝確實組合出與 webapp 相同的行為。

全部寫入 tmp 目錄，不觸發真實 pose-sweep、不污染專案 data/。
"""
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from source.locator_web.app import DATA_ROOT, STATIC_DIR
from source.webapp.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _img_bytes(color=(120, 60, 200), size=(64, 64), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _stub_locator(pred=(3, 5), conf=0.9, certain=True):
    def _loc(reference_bgr, piece_bgr, rows, cols):
        r, c = pred
        return {
            "pred_row": r, "pred_col": c, "confidence": conf, "method": "template",
            "certain": certain, "region_hint": None,
            "top_cells": [{"grid_pos": [r, c], "score": conf, "rotation": 0}],
        }

    return _loc


def _client(tmp_path):
    """以 locator 的 STATIC_DIR + 隔離 tmp 資料 + stub 定位器建測試 client。"""
    app = create_app(data_root=tmp_path, static_dir=STATIC_DIR, locator=_stub_locator())
    return TestClient(app)


# ---------- 接縫 1：資料/靜態目錄獨立 ----------
def test_module_paths_isolated_from_webapp():
    assert DATA_ROOT == PROJECT_ROOT / "data" / "locator"
    assert STATIC_DIR == PROJECT_ROOT / "static" / "locator"
    # 與 webapp 的執行時資料互不干擾
    assert DATA_ROOT != PROJECT_ROOT / "data" / "webapp"


def test_real_app_importable():
    """模組層 app 能無誤建立（沿用 webapp.create_app），且不在 import 期觸發定位。"""
    from source.locator_web.app import app

    assert any(r.path == "/api/projects" for r in app.routes)


# ---------- 接縫 2：serve 的是 locator 前端 ----------
def test_serves_locator_frontend(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    html = r.text
    assert "<title>Puzzle Locator</title>" in html      # locator 版標題
    assert "source/locator_web" in html                  # 接真後端的 locator 前端
    assert "拼圖定位助手" not in html                      # 不是 webapp 那支前端


def test_frontend_file_exists_on_disk():
    assert (STATIC_DIR / "index.html").exists()


# ---------- 接縫 3：薄包裝組合出 webapp 行為（建立→定位→確認）----------
def test_end_to_end_compose(tmp_path):
    client = _client(tmp_path)
    # 建立專案
    r = client.post(
        "/api/projects",
        data={"name": "星空", "rows": "10", "cols": "8"},
        files={"reference": ("ref.jpg", _img_bytes((30, 40, 120), (200, 160)), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert (tmp_path / "jp.db").exists()  # 落在隔離的 tmp，不在 data/locator

    # 定位（stub 固定回 (3,5)）→ 產生未確認單片
    r = client.post(f"/api/projects/{pid}/locate",
                    files={"image": ("piece.jpg", _img_bytes((200, 30, 30)), "image/jpeg")})
    assert r.status_code == 200, r.text
    loc = r.json()
    assert (loc["pred_row"], loc["pred_col"]) == (3, 5)
    piece_id = loc["piece_id"]

    # 確認落點 → 計入進度
    r = client.post(f"/api/projects/{pid}/pieces/{piece_id}/confirm",
                    json={"row": 3, "col": 5, "confirmed": True})
    assert r.status_code == 200, r.text
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["confirmed"] == 1
    assert {"row": 3, "col": 5} in client.get(f"/api/projects/{pid}/pieces").json()["confirmed_cells"]
