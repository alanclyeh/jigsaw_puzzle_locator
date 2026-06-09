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

## Code Conventions & Automation

### 1. 測試自動化報告 (Critical)
*   **規則**：每次執行 `pytest` 測試後，**必須自動在 `output/report.html` 產出精美的 HTML 格式報告**（整合測試總結、測項明細與設計建議）。
*   **實作機制**：此功能已透過 [tests/conftest.py](file:///Users/alan.yeh/Workspace/my_projects/jp_locator/tests/conftest.py) 中的 `pytest_terminal_summary` hook 自動化實作。
*   **邊界限制**：在後續的任何開發與重構中，**不可刪除、註解或破壞 `conftest.py` 中的自動報告生成邏輯**。若測試檔案有新增，報告應自動適配。

### 2. 幾何正規化演算法規範 (Locator)
*   **外接矩形**：提取碎片特徵與匹配前，必須使用 `_get_puzzle_body_rect()` 削去凸耳，再使用 `_standardize_rotated_rect()` 將寬度對齊長邊以獲取唯一的標準旋轉角。
*   **尺度正規化**：以 `scale_factor = L_grid / L_piece` 將碎片與大圖網格對齊至 1:1 後再比對，避免多尺度搜索。
*   **4向直角旋轉**：只測試 `aligned_angle + [0, 90, 180, 270]` 四個方向。
*   **帶遮罩匹配**：必須在 `cv2.matchTemplate` 中傳入遮罩，並將搜尋區擴大 1.25 倍防止越界。

## Git Workflow & Code Review (Critical)
*   **分支策略**：後續所有修改**禁止**直接 Push 到 `main` 分支。必須建立新的 feature 分支（例如 `feature/xxx`）進行開發。
*   **代碼審查 (Code Review)**：修改完成後，須先推送到 GitHub 建立 Pull Request (PR)，並將變更提交給另一個 Agent 進行 Review。
*   **合併機制**：審查通過後，方可將 PR 合併回 `main` 分支。

