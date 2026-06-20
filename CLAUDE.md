# Project: Jigsaw Puzzle Helper (jp_locator)

此文件為專案的 Agent 開發規則與環境說明檔，用以持久化用戶的開發偏好與自動化規範，未來的 Agent 會在 Session 啟動時自動加載並遵循。

## Tech Stack
- 語言：Python 3.10+ (相容 3.9+)
- 影像處理：OpenCV (cv2) >= 4.9.0
- 數值計算：NumPy >= 1.26.0
- 測試框架：pytest >= 8.0.0

## Commands
- **執行定位自動化測試**：`python3 -m pytest tests/test_localization.py -v`
- **執行定位 CLI 原型**：`python scripts/locate_piece.py <reference.jpg> <piece_photo.jpg>`
- **啟動 FastAPI 本地開發伺服器**：`uvicorn source.app:app --reload` (暫未擴展)
- **啟動單片採集 Web App**：`python scripts/run_capture_server.py`（手機 https）或 `--no-tls`（桌機 localhost）。用來拍單片、手動裁切、輸入行列，存成 `data/pieces_c{col}_r{row}.jpg` 測試資料。
- **啟動拼圖定位助手 Web App（正式版）**：`python scripts/run_webapp_server.py`（手機 https）或 `--no-tls`（桌機 localhost）。完整流程：建立專案 → 拍單片 → 建議位置 → 確認標記 → 全圖/進度。後端 FastAPI + SQLite（`data/webapp/jp.db`），規格見 [doc/webapp_spec.md](file:///Users/alan.yeh/Workspace/my_projects/jp_locator/doc/webapp_spec.md)。
- **執行 Web App 後端測試**：`python3 -m pytest tests/test_webapp.py -v`（快速，stub 定位器）；加 `-m "not slow"` 略過真實定位端到端測試。

## Code Conventions & Automation

### 1. 測試自動化報告 (Critical)
*   **規則**：每次執行 `pytest` 測試後，**必須自動在 `output/report.html` 產出精美的 HTML 格式報告**（整合測試總結、測項明細與設計建議）。
*   **實作機制**：此功能已透過 [tests/conftest.py](file:///Users/alan.yeh/Workspace/my_projects/jp_locator/tests/conftest.py) 中的 `pytest_terminal_summary` hook 自動化實作。
*   **邊界限制**：在後續的任何開發與重構中，**不可刪除、註解或破壞 `conftest.py` 中的自動報告生成邏輯**。若測試檔案有新增，報告應自動適配。

### 2. 幾何正規化演算法規範 (Locator) — v1.2.0（依真實照片實證修訂，詳見 doc/dev_log_plan2.md）
*   **主體量測**：提取碎片特徵與匹配前，必須使用 `_get_puzzle_body_rect()` + `_standardize_rotated_rect()` 取得初始矩形角，再以 `_measure_body()`（投影中位數）量測主體長寬。實測純開運算量測在實拍凸耳較寬時高估達 18%+，不可單獨使用。
*   **尺度正規化**：以 `scale_factor = L_grid / L_piece_body` 將碎片與大圖網格對齊至 1:1 後再比對，僅以小幅尺度掃描（±6%）容忍量測誤差，不做大範圍多尺度搜索。
*   **全姿態掃描（取代 4 向直角旋轉）**：實測 minAreaRect 對齊角在帶凸耳實拍遮罩上誤差達 ±40°，**禁止只測 4 個直角方向**。保底層必須以全 360°（3° 步進）姿態掃描，分數採中心對齊逐像素 max 累積。
*   **帶遮罩 ZNCC 匹配**：高紋理碎片必須使用 `_masked_zncc_map()`（FFT 分解的帶遮罩零均值正規化相關，亮度/對比不變）。**禁止以 `TM_CCORR_NORMED` 作為主要評分**（存在亮度方向偏置，深色碎片對任意亮區得高分）；僅低紋理碎片（灰階變異 < 10、ZNCC 退化）退用彩色帶遮罩 `TM_CCORR_NORMED`。
*   **歷史教訓（禁止重新引入）**：色彩直方圖網格 Top-K 預過濾（GT 排名實測 354~735/1000，碎片橫跨 4 網格使網格對齊直方圖失效）；對實拍碎片照依賴 SIFT 描述子（GT 區域 affine inliers 實測 = 0）。

## Git Workflow & Code Review (Critical)
*   **分支策略**：後續所有修改**禁止**直接 Push 到 `main` 分支。必須建立新的 feature 分支（例如 `feature/xxx`）進行開發。
*   **代碼審查 (Code Review)**：修改完成後，須先推送到 GitHub 建立 Pull Request (PR)，並將變更提交給另一個 Agent 進行 Review。
*   **合併機制**：審查通過後，方可將 PR 合併回 `main` 分支。

