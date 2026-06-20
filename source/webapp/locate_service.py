"""定位服務：把上傳的單片裁切圖跑過「去背 → 定位器」，回傳結構化建議。

刻意與 FastAPI 路由解耦，方便（a）真實 App 直接用、（b）測試以快速 stub 取代，
避免每個 API 測試都觸發秒級的 pose-sweep 定位（遵守 CLAUDE.md 全姿態掃描規範）。

回傳 dict 欄位對應 store.create_piece 與 /locate 回應：
  pred_row, pred_col, confidence, method, certain, region_hint, top_cells
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def locate_piece_image(
    reference_bgr: np.ndarray,
    piece_bgr: np.ndarray,
    rows: Optional[int],
    cols: Optional[int],
) -> dict:
    """去背取最大片後呼叫定位器；去背失敗則整張當前景（與 CLI 一致）。"""
    # 延遲匯入：讓 store/app 在沒有 OpenCV 重模組時也能載入；定位才需要
    from source.features.segmentation.detector import extract_piece_images, segment_pieces
    from source.features.localization.locator import locate_piece

    try:
        seg_res = segment_pieces(piece_bgr)
        piece_images = extract_piece_images(piece_bgr, seg_res)
    except Exception:
        piece_images = []
        seg_res = None

    if piece_images:
        areas = [p.area for p in seg_res.pieces]
        chosen = piece_images[int(np.argmax(areas))]
    else:
        # 整張影像當前景（全白遮罩）
        chosen = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2BGRA)
        chosen[:, :, 3] = 255

    result = locate_piece(reference_bgr, chosen, rows=rows, cols=cols)

    grid = result.grid_pos
    certain = grid is not None and result.region_hint is None
    return {
        "pred_row": int(grid[0]) if grid else None,
        "pred_col": int(grid[1]) if grid else None,
        "confidence": float(result.confidence),
        "method": result.method,
        "certain": bool(certain),
        "region_hint": result.region_hint,
        "top_cells": result.top_cells or None,
    }
