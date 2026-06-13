"""採集 Web App 後端測試：命名 / 序號 / 行列範圍驗證。

全部寫入 tmp 目錄，不污染專案 data/。
"""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from source.capture.app import create_app, resolve_filename


def _jpeg_bytes(color=(120, 60, 200), size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    (tmp_path / "project_config.json").write_text(
        json.dumps({"rows": 40, "cols": 25, "total_pieces": 1000}), encoding="utf-8"
    )
    app = create_app(data_dir=tmp_path)
    return TestClient(app), tmp_path


def _post(client, col, row):
    return client.post(
        "/api/captures",
        data={"col": str(col), "row": str(row)},
        files={"image": ("piece.jpg", _jpeg_bytes(), "image/jpeg")},
    )


def test_config_endpoint(client):
    c, _ = client
    body = c.get("/api/config").json()
    assert body == {"rows": 40, "cols": 25, "total_pieces": 1000}


def test_save_uses_existing_naming(client):
    c, data_dir = client
    res = _post(c, col=3, row=26)
    assert res.status_code == 200
    assert res.json()["filename"] == "pieces_c3_r26.jpg"
    assert (data_dir / "pieces_c3_r26.jpg").exists()
    assert res.json()["total_in_cell"] == 1


def test_duplicate_cell_gets_sequence_suffix(client):
    c, data_dir = client
    assert _post(c, 5, 10).json()["filename"] == "pieces_c5_r10.jpg"
    assert _post(c, 5, 10).json()["filename"] == "pieces_c5_r10_1.jpg"
    assert _post(c, 5, 10).json()["filename"] == "pieces_c5_r10_2.jpg"
    assert _post(c, 5, 10).json()["total_in_cell"] == 4


def test_out_of_range_rejected(client):
    c, _ = client
    assert _post(c, col=99, row=1).status_code == 400   # col 超界
    assert _post(c, col=1, row=0).status_code == 400     # row 下界
    assert _post(c, col=26, row=40).status_code == 400   # col == cols+1


def test_invalid_image_rejected(client):
    c, _ = client
    res = c.post(
        "/api/captures",
        data={"col": "1", "row": "1"},
        files={"image": ("x.jpg", b"not-an-image", "image/jpeg")},
    )
    assert res.status_code == 400


def test_list_captures_groups_by_cell(client):
    c, _ = client
    _post(c, 2, 38)
    _post(c, 2, 38)
    _post(c, 3, 25)
    cells = c.get("/api/captures").json()["cells"]
    by_key = {(x["col"], x["row"]): x["count"] for x in cells}
    assert by_key[(2, 38)] == 2
    assert by_key[(3, 25)] == 1


def test_resolve_filename_direct(tmp_path):
    assert resolve_filename(tmp_path, 1, 1).name == "pieces_c1_r1.jpg"
    (tmp_path / "pieces_c1_r1.jpg").write_bytes(b"x")
    assert resolve_filename(tmp_path, 1, 1).name == "pieces_c1_r1_1.jpg"
