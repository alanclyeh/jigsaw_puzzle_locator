# locator_web — Puzzle Locator 本機應用後端

把行動端 UI（[`static/locator/index.html`](../../static/locator/index.html)）接成可用的本機應用：建立專案、拍單片→裁切→辨識位置、記錄進度。與既有 `source/capture`（資料集採集）獨立，但共用底層去背 / 定位模組。

## 啟動
```bash
python scripts/run_locator_server.py            # https + 自簽憑證（手機開相機用）
python scripts/run_locator_server.py --no-tls   # http（桌機 localhost 測試）
```

## 組成
| 檔案 | 職責 |
|---|---|
| `store.py` | SQLite 持久層（`projects` 1:N `pieces`，每專案 `settings`）。僅用內建 `sqlite3`。 |
| `locate_service.py` | 包裝定位管線：去背 `segment_pieces`+`extract_piece_images` 取面積最大片 BGRA → `locate_piece(ref,piece,rows,cols)` → 可序列化 dict（含 `grid_pos` / `conf` / `top_cells` / `region_hint`）。 |
| `app.py` | FastAPI：服務前端 + REST API。 |

## 資料
- DB：`data/locator.db`
- 影像：`data/projects/{id}/reference.jpg`、`data/projects/{id}/pieces/r{row}_c{col}.jpg`
- 兩者皆 gitignore（執行期產物）。

## API
| Method | Path | 說明 |
|---|---|---|
| GET | `/` | 行動端 UI |
| GET/POST | `/api/projects` | 列出 / 建立（multipart：`name,rows,cols`，可選 `reference` 圖） |
| GET/DELETE | `/api/projects/{id}` | 詳情（含統計）/ 刪除（連帶影像） |
| GET | `/api/projects/{id}/reference` | 完成圖 |
| PUT | `/api/projects/{id}/settings` | 更新設定（部分合併） |
| POST | `/api/projects/{id}/locate` | 上傳裁切單片 → 回定位結果（**不落地**） |
| GET/POST | `/api/projects/{id}/pieces` | 列出 / 儲存已定位單片 |
| PATCH/DELETE | `/api/projects/{id}/pieces/{pid}` | 切換確認 / 刪除 |

## 設計重點（依專案實證，勿回退）
- **拍片走原生全解析度**（前端 `<input capture>` → 凍結 canvas → 固定框 pan/zoom → 1:1 裁切，`MIN_CROP_PX=600` 把關）。低解析度截圖會嚴重掉辨識率。
- **`/locate` 慢**（單片對上千片參考做全姿態掃描可達數秒～數十秒），前端以 spinner 覆蓋。
- **信心值不可靠**（錯位也常回高分），故結果頁提供 **Top-K 候選清單**供人工判讀。

## 測試
```bash
python3 -m pytest tests/test_locator_web.py -v
```
涵蓋 store 持久層與 API 端點；實際影像定位（重、慢）屬整合測試，不在單元測試內。
