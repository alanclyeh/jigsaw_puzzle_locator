"""Puzzle Locator 本機應用後端測試：store 持久層 + FastAPI 端點。

全部寫入 tmp 目錄，不污染專案 data/。
不涵蓋實際 /locate 影像定位（重、慢，屬整合測試）；此處只驗證
未上傳完成圖時 /locate 應回 400 的把關邏輯。
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from source.locator_web.app import create_app
from source.locator_web.store import Store


def _jpeg_bytes(color=(120, 60, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------- store 層 ----------------
@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "locator.db")


def test_store_create_and_stats(store):
    p = store.create_project("星空 1000 片", 40, 25)
    assert p["total_pieces"] == 1000
    assert p["confirmed"] == 0 and p["remaining"] == 1000 and p["pct"] == 0
    assert p["settings"]["auto_crop"] is True  # 預設值帶入


def test_store_pieces_and_confirmed_count(store):
    p = store.create_project("t", 10, 10)
    store.add_piece(p["id"], 3, 5, 0.9, "R3C5", confirmed=True)
    pc = store.add_piece(p["id"], 7, 2, 0.5, "R7C2", confirmed=False)
    assert store.get_project(p["id"])["confirmed"] == 1
    store.set_piece_confirmed(pc["id"], True)
    assert store.get_project(p["id"])["confirmed"] == 2
    assert len(store.list_pieces(p["id"])) == 2
    store.delete_piece(pc["id"])
    assert store.get_project(p["id"])["confirmed"] == 1


def test_store_settings_merge(store):
    p = store.create_project("t", 5, 5)
    out = store.update_settings(p["id"], {"haptic": True})
    assert out["settings"]["haptic"] is True
    assert out["settings"]["auto_crop"] is True  # 既有值保留


def test_store_cascade_delete(store):
    p = store.create_project("t", 5, 5)
    store.add_piece(p["id"], 1, 1, 1.0, "R1C1")
    assert store.delete_project(p["id"]) is True
    assert store.list_pieces(p["id"]) == []  # 級聯刪除


# ---------------- API 層 ----------------
@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "locator.db", data_dir=tmp_path)
    return TestClient(app)


def test_api_create_list_get_delete(client):
    assert client.get("/api/projects").json() == {"projects": []}
    r = client.post("/api/projects", data={"name": "P", "rows": 40, "cols": 25})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert r.json()["has_reference"] is False
    assert len(client.get("/api/projects").json()["projects"]) == 1
    assert client.get(f"/api/projects/{pid}").json()["name"] == "P"
    assert client.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_api_create_with_reference(client):
    r = client.post(
        "/api/projects",
        data={"name": "P", "rows": 10, "cols": 10},
        files={"reference": ("ref.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    pid = r.json()["id"]
    assert r.json()["has_reference"] is True
    assert client.get(f"/api/projects/{pid}/reference").status_code == 200


def test_api_validation(client):
    assert client.post("/api/projects", data={"name": "  ", "rows": 10, "cols": 10}).status_code == 400
    assert client.post("/api/projects", data={"name": "x", "rows": 0, "cols": 10}).status_code == 400


def test_api_pieces_flow(client):
    pid = client.post("/api/projects", data={"name": "P", "rows": 10, "cols": 10}).json()["id"]
    # 越界 row/col → 400
    assert client.post(f"/api/projects/{pid}/pieces", data={"row": 99, "col": 1}).status_code == 400
    piece = client.post(f"/api/projects/{pid}/pieces",
                        data={"row": 3, "col": 5, "conf": 0.9, "confirmed": True}).json()
    assert piece["ref_label"] == "R3C5"
    assert client.get(f"/api/projects/{pid}").json()["confirmed"] == 1
    # PATCH 取消確認
    client.patch(f"/api/projects/{pid}/pieces/{piece['id']}", json={"confirmed": False})
    assert client.get(f"/api/projects/{pid}").json()["confirmed"] == 0
    # DELETE
    assert client.delete(f"/api/projects/{pid}/pieces/{piece['id']}").json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}/pieces").json()["pieces"] == []


def test_api_settings_put(client):
    pid = client.post("/api/projects", data={"name": "P", "rows": 10, "cols": 10}).json()["id"]
    out = client.put(f"/api/projects/{pid}/settings", json={"frame_mode": 2}).json()
    assert out["settings"]["frame_mode"] == 2


def test_api_locate_without_reference_400(client):
    pid = client.post("/api/projects", data={"name": "P", "rows": 10, "cols": 10}).json()["id"]
    r = client.post(f"/api/projects/{pid}/locate",
                    files={"image": ("p.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 400  # 未上傳完成圖
