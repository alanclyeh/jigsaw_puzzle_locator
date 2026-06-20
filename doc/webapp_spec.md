# 拼圖定位助手 Web App 規格書（spec-driven）

> 版本 v1.0.0 ｜ 2026-06-20 ｜ 對應設計檔 `design/Puzzle Locator (standalone).html`
>
> 本文件先於程式碼撰寫，定義「要做什麼、如何驗證」。實作完成後此文件即為驗收依據。

## 1. 目標與範圍

把既有的 CLI 定位原型（`scripts/locate_piece.py`）與單片採集網頁（`source/capture`）整併成一個
**可在本機執行、支援手機版型** 的正式 Web App，照 `design/` 設計檔還原 UI/流程。

核心使用情境：使用者拿著一片實體拼圖 → 用手機拍照 → App 建議它在完成大圖上的位置 →
使用者確認後標記 → 全圖與進度即時更新。

### 1.1 必要需求（來自任務指派）
1. **專案管理**：先支援單一專案的完整生命週期（建立 / 檢視 / 刪除）；資料模型支援多專案。
2. **拍單片**：流程與 `capture` 採集網頁一致（原生相機 → 自動框 → 拖曳/縮放微調 → 原解析度裁切）。
3. **建議位置**：把裁切後單片送進既有定位器，回傳建議的列/行與信心度。
4. **標記儲存已確認單片**：使用者可勾選「已確認」，該片落點寫入全圖。
5. **顯示整體進度**：已定位 / 待辨識 / 完成度（%），全圖以網格標示已完成格。

### 1.2 擴充需求（本次一併實作，提升易用性）
- **Top-K 候選 + 搜尋區塊提示**：定位不唯一時，列出前幾名候選與「大概範圍」，避免假確定（沿用 locator 既有能力與 CLAUDE.md「信心值不可靠」教訓）。
- **手動修正落點**：建議錯誤時，使用者可直接在確認前改成正確的列/行再儲存。
- **最近已確認清單**：可回溯、可取消勾選（toggle）。
- **拍攝輔助設定**：保留採集網頁的「拍攝指引」精神（淺白步驟），列為設定頁開關。

### 1.3 不在本次範圍
- 雲端部署、多人協作、帳號系統。
- 改動定位演算法本身（遵守 CLAUDE.md 既定規範，僅作為服務呼叫）。
- 批次指派 / 區域提示等已被 memory 排除的方向。

## 2. 技術選型與決策

| 決策 | 選擇 | 理由 |
|------|------|------|
| 後端框架 | FastAPI（沿用 `source/capture`） | 與既有採集 App 一致，TestClient 易測 |
| 資料庫 | **SQLite（stdlib `sqlite3`）** | 關聯查詢（專案↔單片）比 JSON 清楚；零額外相依、單檔、易備份 |
| 影像儲存 | 檔案系統 `data/webapp/projects/{pid}/` | 大圖/單片不入 DB，DB 只存 metadata 路徑 |
| 前端 | 單檔 SPA `static/webapp/index.html` | 與 `capture/index.html` 風格一致，免建置工具，可離線開啟 |
| 定位器注入 | `create_app(locator=...)` 可注入 | 真實 App 用慢速 pose-sweep 定位器；測試注入快速 stub，保持測試秒級 |

> **為何不沿用設計檔的 React/dc-runtime 打包檔**：該檔是 Claude artifact 的 bundler 產物（內含 React UMD + 字型），
> 依賴外部 CDN、非可維護原始碼。設計檔僅作為 **UI/流程的視覺與互動規格**；實作以原生 HTML/JS 還原，
> 可本機離線執行、無外部相依，符合「可在 local 運作」目標。

## 3. 資料模型（SQLite）

```sql
CREATE TABLE projects (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL,
  rows          INTEGER NOT NULL,         -- 列數
  cols          INTEGER NOT NULL,         -- 行數
  reference_ext TEXT NOT NULL,            -- 完成圖副檔名（.jpg/.png）
  created_at    TEXT NOT NULL             -- ISO8601
);

CREATE TABLE pieces (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  image_ext     TEXT NOT NULL,            -- 單片裁切圖副檔名
  -- 定位建議
  pred_row      INTEGER,                  -- 建議列（1-indexed），失敗為 NULL
  pred_col      INTEGER,
  confidence    REAL,                     -- 0~1
  method        TEXT,                     -- 'feature' | 'template'
  certain       INTEGER DEFAULT 0,        -- 定位器是否判為「確定」(1/0)
  region_hint   TEXT,                     -- JSON：不確定時的搜尋區塊
  top_cells     TEXT,                     -- JSON：Top-K 候選
  -- 使用者確認
  confirmed     INTEGER DEFAULT 0,        -- 是否已確認 (1/0)
  final_row     INTEGER,                  -- 確認/修正後的列
  final_col     INTEGER,
  created_at    TEXT NOT NULL
);
```

- **進度定義**：`confirmed_count = COUNT(pieces WHERE confirmed=1 且 final_row/col 不為 NULL)`；
  `total = rows * cols`；`pct = round(confirmed_count / total * 100)`。
- **每格唯一**：同一 `(final_row, final_col)` 同時間只應有一片已確認；新確認覆蓋舊的（取消舊片 confirmed）。

## 4. API 規格

所有路徑前綴 `/api`。錯誤回 `{"detail": "..."}` + 對應 HTTP 狀態碼。

| 方法 | 路徑 | 說明 | 主要回傳 |
|------|------|------|----------|
| GET  | `/projects` | 列出所有專案（含進度摘要） | `{projects:[{id,name,rows,cols,pieces,confirmed,pct,...}]}` |
| POST | `/projects` | 建立專案（multipart：name, rows, cols, reference 圖） | `{id, ...}` |
| GET  | `/projects/{id}` | 專案詳情 + 進度 | `{id,name,rows,cols,pieces,confirmed,remaining,pct}` |
| DELETE | `/projects/{id}` | 刪除專案（連同單片與檔案） | `{ok:true}` |
| GET  | `/projects/{id}/reference` | 取完成大圖 | image/* |
| POST | `/projects/{id}/locate` | 上傳單片裁切圖，跑定位，存為未確認片 | `{piece_id, pred_row, pred_col, confidence, certain, top_cells, region_hint, method}` |
| GET  | `/projects/{id}/pieces` | 列出單片（預設只列已確認 + 最近） | `{pieces:[...], confirmed_cells:[{row,col}]}` |
| GET  | `/projects/{id}/pieces/{pid}/image` | 取單片裁切圖 | image/* |
| POST | `/projects/{id}/pieces/{pid}/confirm` | 確認/修正落點（body: row, col, confirmed） | `{ok, piece}` |
| DELETE | `/projects/{id}/pieces/{pid}` | 刪除單片 | `{ok:true}` |

### 4.1 `/locate` 行為細節
- 後端對上傳影像做去背（`segment_pieces`+`extract_piece_images`）取最大片；去背不到則整張當前景（與 CLI 一致）。
- 呼叫 `locate_piece(reference, piece_bgra, rows, cols)`。
- `certain` 由 `region_hint is None and grid_pos is not None` 推得（沿用 locator 的 `_assess_position`）。
- 不論成敗都建立一筆 piece 記錄（confirmed=0），回傳 `piece_id` 供後續確認。
- 信心值僅供參考（CLAUDE.md / memory：定位器對錯誤位置也可能回 conf≈1.0），UI 須同時呈現 Top-K 與搜尋區塊。

## 5. 前端頁面（對應設計檔）

單頁 SPA，狀態機切換畫面，手機版型（直式手機框，桌機置中、窄寬皆可用 RWD）：

1. **HOME 我的拼圖**：專案卡片（縮圖/名稱/規格/進度條/進度文字）＋「新增拼圖專案」。
2. **SETUP 新增專案**：名稱、規格 preset、列/行、完成圖上傳、建立。
3. **APP**（含底部 4 分頁）
   - **拍照 Capture**：aim（相機）→ cropped（拖曳/縮放對框）→ recognizing → result（結果卡＋全圖/放大切換＋確認）。
   - **全圖 Map**：完成圖 + 網格 + 已定位格 + 最新片高亮 + 進度卡。
   - **進度 List**：統計卡（已定位/待辨識/完成度）＋ 最近已確認清單（可 toggle）。
   - **設定 Settings**：拍攝輔助開關、建議框大小、刪除專案。

互動沿用 `capture/index.html`：原生相機 file input（`capture=environment`）、四角取樣自動框、固定框 + 移動/縮放底圖、原解析度裁切（`MIN_CROP_PX` 品質閘）。

## 6. 驗收條件（如何驗證 / 對應測試）

以 `tests/test_webapp.py`（FastAPI TestClient，注入快速 stub 定位器）驗證：

- [AC1] 建立專案：POST /projects 帶圖成功，DB 與檔案落地，GET 回得到。
- [AC2] 參數驗證：rows/cols ≤ 0、缺圖、非影像 → 400。
- [AC3] 取完成圖：GET /reference 回 200 + 正確 content-type。
- [AC4] 定位：POST /locate 回傳 pred_row/col 在範圍內，建立未確認片。
- [AC5] 確認/修正：confirm 後 confirmed_count +1，pct 正確；修正 row/col 生效。
- [AC6] 每格唯一：對同格再確認另一片，舊片自動取消（confirmed_count 不重複累加）。
- [AC7] 進度：confirmed/remaining/pct 計算正確。
- [AC8] toggle：取消已確認片，進度回退。
- [AC9] 刪除專案：檔案與單片一併刪除，GET 回 404。
- [AC10] 路徑安全：piece/ref 取檔不可越界。
- [AC11] 端到端（真實定位器，標記 slow）：用 `data/eval_native` 一片實拍 + `data/reference_puzzle.jpg`，/locate 能回出範圍內網格（不要求命中，僅驗證管線串通）。

測試結束後須自動產生 `output/report.html`（沿用 `tests/conftest.py`，不可破壞）。

## 7. 執行方式

- 啟動：`python scripts/run_webapp_server.py --no-tls`（桌機 http://localhost:8000）
  或 `python scripts/run_webapp_server.py`（手機 https，自簽憑證，鏡頭需 secure context）。
- 測試：`python3 -m pytest tests/test_webapp.py -v`（快速）；加 `-m slow` 或不加標記跑端到端。

## 8. 實作歷程重點（決策日誌）

- 2026-06-20：解析設計檔（dc-runtime bundler 產物），抽出 `<x-dc>` 模板得到完整畫面/狀態機規格。
- 決定 SQLite + 檔案系統混合儲存；定位器以依賴注入方式接入，兼顧真實效果與測試速度。
- 前端以原生 HTML/JS 還原設計（不引入 React/CDN），確保本機離線可跑。
