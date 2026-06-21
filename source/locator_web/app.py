"""Puzzle Locator 行動端 App 後端（FastAPI）。

把 `static/locator/index.html`（Claude Design 行動端 UI）接成可用的本機應用。
後端邏輯與「正式版」webapp 完全相同（專案 CRUD、拍片定位、確認標記、進度），
因此直接重用 `source.webapp.create_app`，僅換成獨立的資料根目錄與靜態頁：
  - 資料：data/locator/（SQLite jp.db + 各專案影像），與 webapp 的 data/webapp/ 互不干擾。
  - 靜態頁：static/locator/index.html。

決策脈絡見記憶 locator-web-local-backend；API 合約與資料層見 source/webapp/。
"""
from __future__ import annotations

from pathlib import Path

from source.webapp.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "locator"
STATIC_DIR = PROJECT_ROOT / "static" / "locator"

app = create_app(data_root=DATA_ROOT, static_dir=STATIC_DIR)
