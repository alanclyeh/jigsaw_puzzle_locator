# Spec: Plan 2 (MVP) — 拼圖定位輔助 (Piece Localization)

## Objective

驗證一個**新方向**的核心前提：**給定一張「完成的大圖」(盒面圖 / 已拼好的參考圖) 與一張手機拍的「單一碎片」，程式能建議該碎片應拼在完成圖的哪個位置，並在完成圖上框出該區域。**

這是與 Step 0（碎片切割）不同方向的輔助功能。Step 0 解決「桌上散落的碎片是哪幾片」；本階段解決「手上這一片該放到完成圖的哪裡」。

本階段採**獨立原型 (CLI prototype)** 先驗證演算法可行性，**暫不接 API、不接前端**。

### 使用者場景
1. 使用者先拍一張**完成的大圖**作為參考 (reference)。
2. 用手機拍**單一碎片** (背景單純或桌面背景皆可)。
3. 程式：
   - 重用現有切割模組從單片照去背，取得乾淨碎片。
   - 在完成圖中定位該碎片落點。
   - 在完成圖上**框出建議區域**，附上信心度；低信心時框出多個候選區域供使用者判斷。

### Success Criteria
- [ ] 合成測試通過：從完成圖裁出高紋理 region 合成「假單片」(旋轉/縮放/色彩抖動)，預測框與真值 region 的 **IoU ≥ 0.6** 比例 ≥ 80%
- [ ] 高紋理片旋轉角誤差在 **±15°**
- [ ] 低紋理 (純色) 片觸發 template fallback：`method == "template"` 且回傳 ≥ 1 個候選框
- [ ] CLI 可執行：輸入完成圖 + 單片照 → 產出標註圖，人工檢視落點合理
- [ ] 單片照經 `segment_pieces` / `extract_piece_images` 取得 BGRA 後能正確定位
- [ ] pytest 自動化測試全部通過

## Tech Stack

| 項目 | 選擇 | 版本 | 備註 |
|------|------|------|------|
| 語言 | Python | 3.10+ | |
| 影像處理 | OpenCV (cv2) | >= 4.9.0 | `cv2.SIFT_create()` 自 4.4 起免費可用 |
| 數值運算 | NumPy | >= 1.26.0 | |
| 測試框架 | pytest | >= 8.0.0 | |

> 本階段不新增依賴，沿用現有 `requirements.txt`。

## 演算法：混合策略 (SIFT 為主，色彩/模板為輔)

### 為什麼選 SIFT 特徵匹配為主
「在大圖中找出這個小圖塊的位置」本質是 patch localization。SIFT 對**尺度、旋轉不變**，正好解決兩個核心難題：
- 手機拍的單片與它在完成圖中的大小未知 (尺度差異)。
- 單片擺放角度任意 (旋轉差異)。

### 主流程 (textured pieces)
1. **取得乾淨單片**：重用 `extract_piece_images()` 取得 BGRA 裁切圖，用 alpha 當遮罩 (只在碎片像素上抽特徵)。
2. 對單片與完成圖各做 `SIFT.detectAndCompute` (單片傳入 alpha 遮罩，避免背景雜訊特徵)。
3. `FLANN` 比對 + **Lowe ratio test** (0.7) 篩掉模糊匹配。
4. `cv2.findHomography(..., cv2.RANSAC)` 估計單應矩陣，取得 inliers。
5. 把單片四角經 Homography 投影到完成圖 → 取得**四邊形落點區域**，再取外接矩形畫框；由矩陣分解推得建議旋轉角。
6. **信心度** = RANSAC inlier 數與比例 (例如 inliers ≥ 12 且 ratio ≥ 0.3 視為可靠)。

### 低紋理片的退路 (fallback)
純色 / 弱紋理片 (如天空、單色背景) 特徵點不足，SIFT 會失敗。當 inliers 低於門檻時：
- 改用**色彩 + 多尺度模板匹配** (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`)，以單片主色直方圖先縮小候選範圍，再回傳**前 N 個候選方框**。
- 此時輸出標記為「低信心」，框出多個候選而非單一框。

## Commands

```bash
# 啟用虛擬環境
source .venv/bin/activate

# 跑定位測試
pytest tests/test_localization.py -v

# CLI 原型：完成圖 + 單片照 → 標註輸出圖 + 信心度
python scripts/locate_piece.py <reference.jpg> <piece_photo.jpg>

# 範例 (待使用者放入真實資料後)
python scripts/locate_piece.py data/reference_cat.jpg data/piece_cat_01.jpg
```

## Project Structure

延續現有 `sources/features/<feature>/` 慣例，本階段以 CLI 原型為主：

```
jigsaw_puzzle_helper/
├── data/
│   ├── reference_<puzzle>.jpg          ← 完成大圖 (使用者手動放入，待補)
│   ├── piece_<puzzle>_NN.jpg           ← 對應單片照 (待補)
│   └── output/
│       └── <piece>_located.jpg         ← 標註圖：完成圖上框出落點
├── sources/
│   └── features/
│       ├── segmentation/detector.py    ← 重用：segment_pieces / extract_piece_images
│       └── localization/               ← 【新增】
│           ├── __init__.py
│           └── locator.py              ← 核心：locate_piece(reference, piece_bgra)
├── scripts/
│   └── locate_piece.py                 ← 【新增】CLI 進入點
├── tests/
│   ├── test_localization.py            ← 【新增】合成資料 IoU 測試
│   └── _synthetic.py                   ← 【新增】合成單片產生器
└── docs/specs/
    └── plan2_mvp_piece_localization.md ← 本文件
```

### `locator.py` 介面 (草案)
```python
@dataclass
class LocateResult:
    quad: np.ndarray | None          # 完成圖上的四邊形落點 (4x2)，失敗為 None
    bounding_box: tuple[int, int, int, int] | None
    rotation_deg: float | None       # 由 homography 推得的建議旋轉角
    confidence: float                # 0~1
    method: str                      # "feature" | "template"
    candidates: list[tuple]          # 低信心時的多個候選框
    annotated_reference: np.ndarray  # 已畫框的完成圖

def locate_piece(reference: np.ndarray, piece_bgra: np.ndarray) -> LocateResult: ...
```

### 重用既有元件
- `sources/features/segmentation/detector.py`
  - `segment_pieces()` — 從手機單片照去背取碎片。
  - `extract_piece_images()` — 取 BGRA (alpha = 遮罩)，直接餵給 `locate_piece`。
- `scripts/locate_piece.py` 流程：讀完成圖 → 讀單片照 → `segment_pieces` + `extract_piece_images` 取最大片 → `locate_piece` → 存標註圖。

## Code Style

```python
# dataclass 用於結構化回傳值 (沿用 detector.py 慣例)
@dataclass
class LocateResult: ...

# function 命名：動詞開頭，清楚表達意圖
def locate_piece(reference: np.ndarray, piece_bgra: np.ndarray) -> LocateResult: ...
```

- 變數名稱與註解使用英文
- type hint 明確標註參數與回傳值
- 一個 function 做一件事，內部 helper 以 `_` 開頭 (對齊 `detector.py`)
- 不寫多餘的註解，程式碼本身說明意圖

## Testing Strategy

- **框架**：pytest
- **測試檔位置**：`tests/`

### 測試資料策略 (重要)
目前 `data/` **沒有完成大圖，也沒有單片↔完成圖配對**，無法直接做真實驗證。處理方式：

1. **核心演算法用合成資料自動驗證 (不需實拍)**：
   `tests/_synthetic.py` 從一張完成圖隨機裁一塊 region，做旋轉 / 縮放 / 輕微色彩抖動當作「假單片」，送入 `locate_piece`，再比對預測框與 ground-truth region 的 **IoU**。
2. **真實照片驗證屬手動步驟**：需使用者提供一張完成大圖 + 幾張對應的手機單片照，放入 `data/` (命名 `reference_<puzzle>.jpg` / `piece_<puzzle>_NN.jpg`)。

### 測試層級

| 層級 | 對象 | 方式 |
|------|------|------|
| Unit | `locate_piece()` (feature) | 合成高紋理假單片，驗證預測框與真值 IoU ≥ 0.6、旋轉角誤差 ±15° |
| Unit | `locate_piece()` (template fallback) | 合成純色假單片，驗證 `method == "template"` 且回傳 ≥ 1 候選框 |
| Integration | `segment_pieces` → `locate_piece` | 單片照去背取 BGRA 後能定位 (有真實資料時) |
| Manual | `scripts/locate_piece.py` | 人工檢視標註圖落點合理 |

## Boundaries

### Always
- 輸出結果到 `data/output/`，不覆蓋原始輸入圖
- 重用 `segmentation/detector.py` 既有函式取得乾淨單片，不另寫去背邏輯
- 低信心時明確標記並框出多個候選，不假裝唯一解

### Ask First
- 新增依賴套件
- 接 FastAPI 端點 / 前端 (屬下一階段)

### Never
- 不修改現有 `segmentation/`、`normalization/`、`validation/`、`app.py`
- 不修改 `data/` 下的原始輸入圖
- 忽略 `archives/` 目錄
- 不部署到遠端

## 後續 (本階段不做)
- 新增 `POST /api/locate` (上傳完成圖 + 單片 → 回傳落點框 / 候選)。
- 已知拼圖網格 (片數) 時，用網格約束尺度，提升低紋理片定位率。
- 前端：點選單片照、在完成圖上以高亮覆蓋顯示建議位置。
