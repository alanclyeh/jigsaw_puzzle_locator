"""拼圖定位助手 Web App 後端測試（對應 doc/webapp_spec.md 驗收條件 AC1~AC11）。

大多數測試注入「快速 stub 定位器」，秒級完成、不觸發 pose-sweep；
AC11 端到端測試標記 slow，使用真實定位器 + data/ 實拍片驗證管線串通。

全部寫入 tmp 目錄，不污染專案 data/webapp。
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from source.webapp.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _img_bytes(color=(120, 60, 200), size=(64, 64), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _stub_locator(pred=(3, 5), conf=0.9, certain=True, region=None, top=None):
    """產生一個固定回傳的快速定位器，簽名同 locate_service.locate_piece_image。"""

    def _loc(reference_bgr, piece_bgr, rows, cols):
        r, c = pred
        return {
            "pred_row": r,
            "pred_col": c,
            "confidence": conf,
            "method": "template",
            "certain": certain,
            "region_hint": region,
            "top_cells": top or [{"grid_pos": [r, c], "score": conf, "rotation": 0}],
        }

    return _loc


@pytest.fixture
def client(tmp_path):
    app = create_app(data_root=tmp_path, locator=_stub_locator())
    return TestClient(app)


def _create_project(client, name="星空 1000 片", rows=40, cols=25):
    return client.post(
        "/api/projects",
        data={"name": name, "rows": str(rows), "cols": str(cols)},
        files={"reference": ("ref.jpg", _img_bytes((30, 40, 120), (200, 160)), "image/jpeg")},
    )


def _locate(client, pid, color=(200, 30, 30)):
    return client.post(
        f"/api/projects/{pid}/locate",
        files={"image": ("piece.jpg", _img_bytes(color), "image/jpeg")},
    )


# ---------- AC1 建立專案 ----------
def test_create_project_persists(client, tmp_path):
    r = _create_project(client)
    assert r.status_code == 200, r.text
    body = r.json()
    pid = body["id"]
    assert body["name"] == "星空 1000 片"
    assert body["rows"] == 40 and body["cols"] == 25
    assert body["pieces"] == 1000 and body["confirmed"] == 0 and body["pct"] == 0
    # 檔案落地
    assert (tmp_path / "projects" / str(pid) / "reference.jpg").exists()
    assert (tmp_path / "jp.db").exists()
    # 可由列表取得
    got = client.get("/api/projects").json()["projects"]
    assert any(p["id"] == pid for p in got)


# ---------- AC2 參數驗證 ----------
def test_create_project_validation(client):
    # 缺圖
    r = client.post("/api/projects", data={"name": "x", "rows": "10", "cols": "10"})
    assert r.status_code in (400, 422)
    # rows<=0
    r = client.post(
        "/api/projects",
        data={"name": "x", "rows": "0", "cols": "10"},
        files={"reference": ("ref.jpg", _img_bytes(), "image/jpeg")},
    )
    assert r.status_code == 400
    # 空名稱
    r = client.post(
        "/api/projects",
        data={"name": "  ", "rows": "5", "cols": "5"},
        files={"reference": ("ref.jpg", _img_bytes(), "image/jpeg")},
    )
    assert r.status_code == 400
    # 非影像
    r = client.post(
        "/api/projects",
        data={"name": "x", "rows": "5", "cols": "5"},
        files={"reference": ("ref.jpg", b"not-an-image", "image/jpeg")},
    )
    assert r.status_code == 400


# ---------- AC3 取完成圖 ----------
def test_get_reference(client):
    pid = _create_project(client).json()["id"]
    r = client.get(f"/api/projects/{pid}/reference")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert client.get("/api/projects/9999/reference").status_code == 404


# ---------- AC4 定位 ----------
def test_locate_creates_unconfirmed_piece(client):
    pid = _create_project(client).json()["id"]
    r = _locate(client, pid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["piece_id"]
    assert 1 <= body["pred_row"] <= 40
    assert 1 <= body["pred_col"] <= 25
    assert body["certain"] is True
    # 建立了未確認片
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["confirmed"] == 0
    assert proj["captured"] == 1
    pieces = client.get(f"/api/projects/{pid}/pieces").json()["pieces"]
    assert len(pieces) == 1 and pieces[0]["confirmed"] is False
    # 單片圖可取
    img = client.get(f"/api/projects/{pid}/pieces/{body['piece_id']}/image")
    assert img.status_code == 200


# ---------- AC5 確認/修正 ----------
def test_confirm_and_correct(client):
    pid = _create_project(client).json()["id"]
    piece_id = _locate(client, pid).json()["piece_id"]
    # 修正成 row=10,col=12
    r = client.post(
        f"/api/projects/{pid}/pieces/{piece_id}/confirm",
        json={"row": 10, "col": 12, "confirmed": True},
    )
    assert r.status_code == 200, r.text
    proj = r.json()["project"]
    assert proj["confirmed"] == 1
    assert proj["pct"] == round(1 / 1000 * 100)
    piece = r.json()["piece"]
    assert piece["final_row"] == 10 and piece["final_col"] == 12 and piece["confirmed"] is True


def test_confirm_out_of_range(client):
    pid = _create_project(client, rows=10, cols=10).json()["id"]
    piece_id = _locate(client, pid).json()["piece_id"]
    r = client.post(
        f"/api/projects/{pid}/pieces/{piece_id}/confirm",
        json={"row": 99, "col": 1, "confirmed": True},
    )
    assert r.status_code == 400


# ---------- AC6 每格唯一 ----------
def test_one_piece_per_cell(client):
    pid = _create_project(client).json()["id"]
    p1 = _locate(client, pid, (200, 0, 0)).json()["piece_id"]
    p2 = _locate(client, pid, (0, 200, 0)).json()["piece_id"]
    client.post(f"/api/projects/{pid}/pieces/{p1}/confirm", json={"row": 5, "col": 5, "confirmed": True})
    client.post(f"/api/projects/{pid}/pieces/{p2}/confirm", json={"row": 5, "col": 5, "confirmed": True})
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["confirmed"] == 1  # 同格只算一片
    cells = client.get(f"/api/projects/{pid}/pieces").json()["confirmed_cells"]
    assert cells == [{"row": 5, "col": 5}]
    # p1 應被取消
    pieces = {p["id"]: p for p in client.get(f"/api/projects/{pid}/pieces").json()["pieces"]}
    assert pieces[p1]["confirmed"] is False
    assert pieces[p2]["confirmed"] is True


# ---------- AC7 進度計算 ----------
def test_progress_counts(client):
    pid = _create_project(client, rows=2, cols=2).json()["id"]  # total=4
    for i, (rr, cc) in enumerate([(1, 1), (1, 2), (2, 1)]):
        pidc = _locate(client, pid).json()["piece_id"]
        client.post(
            f"/api/projects/{pid}/pieces/{pidc}/confirm",
            json={"row": rr, "col": cc, "confirmed": True},
        )
    proj = client.get(f"/api/projects/{pid}").json()
    assert proj["pieces"] == 4
    assert proj["confirmed"] == 3
    assert proj["remaining"] == 1
    assert proj["pct"] == 75


# ---------- AC8 toggle 取消 ----------
def test_toggle_unconfirm(client):
    pid = _create_project(client, rows=2, cols=2).json()["id"]
    piece_id = _locate(client, pid).json()["piece_id"]
    client.post(f"/api/projects/{pid}/pieces/{piece_id}/confirm", json={"row": 1, "col": 1, "confirmed": True})
    assert client.get(f"/api/projects/{pid}").json()["confirmed"] == 1
    # 取消
    client.post(f"/api/projects/{pid}/pieces/{piece_id}/confirm", json={"row": 1, "col": 1, "confirmed": False})
    assert client.get(f"/api/projects/{pid}").json()["confirmed"] == 0


# ---------- AC9 刪除專案 ----------
def test_delete_project(client, tmp_path):
    pid = _create_project(client).json()["id"]
    _locate(client, pid)
    assert (tmp_path / "projects" / str(pid)).exists()
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404
    assert not (tmp_path / "projects" / str(pid)).exists()
    assert client.delete(f"/api/projects/{pid}").status_code == 404


# ---------- AC10 路徑安全 / 不存在 ----------
def test_missing_resources_404(client):
    pid = _create_project(client).json()["id"]
    assert client.get(f"/api/projects/{pid}/pieces/9999/image").status_code == 404
    assert client.post(
        f"/api/projects/{pid}/pieces/9999/confirm", json={"row": 1, "col": 1, "confirmed": True}
    ).status_code == 404
    assert client.delete(f"/api/projects/{pid}/pieces/9999").status_code == 404


def test_delete_piece(client):
    pid = _create_project(client).json()["id"]
    piece_id = _locate(client, pid).json()["piece_id"]
    assert client.delete(f"/api/projects/{pid}/pieces/{piece_id}").status_code == 200
    assert client.get(f"/api/projects/{pid}/pieces").json()["pieces"] == []


# ---------- AC11 端到端（真實定位器，slow）----------
@pytest.mark.slow
def test_end_to_end_real_locator(tmp_path):
    """用真實定位器跑一片實拍 + 完成圖，驗證管線串通（不要求命中正解）。"""
    ref = PROJECT_ROOT / "data" / "reference_puzzle.jpg"
    pieces_dir = PROJECT_ROOT / "data" / "eval_native"
    if not ref.exists() or not pieces_dir.exists():
        pytest.skip("缺少 data/reference_puzzle.jpg 或 data/eval_native")
    sample = next(iter(sorted(pieces_dir.glob("pieces_c*_r*.jpg"))), None)
    if sample is None:
        pytest.skip("eval_native 無樣本")

    app = create_app(data_root=tmp_path)  # 真實定位器
    c = TestClient(app)
    with open(ref, "rb") as f:
        pid = c.post(
            "/api/projects",
            data={"name": "e2e", "rows": "40", "cols": "25"},
            files={"reference": ("ref.jpg", f.read(), "image/jpeg")},
        ).json()["id"]
    with open(sample, "rb") as f:
        r = c.post(
            f"/api/projects/{pid}/locate",
            files={"image": ("piece.jpg", f.read(), "image/jpeg")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["piece_id"]
    if body["pred_row"] is not None:
        assert 1 <= body["pred_row"] <= 40
        assert 1 <= body["pred_col"] <= 25
