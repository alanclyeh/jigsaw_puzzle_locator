"""把現有定位流程包成 web 用、可序列化的服務函式。

沿用 scripts/locate_piece.py 的管線：
    去背 (segment_pieces + extract_piece_images) → 取面積最大片 (BGRA)
    → locate_piece(reference, piece_bgra, rows, cols)
找不到明顯碎片時，退回「整張照當前景」（全白遮罩），與 CLI 行為一致。
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from source.features.localization.locator import locate_piece
from source.features.segmentation.detector import (
    extract_piece_images,
    segment_pieces,
)


def _piece_bgra_from_photo(piece_bgr: np.ndarray) -> np.ndarray:
    """去背取面積最大的單片 BGRA；偵測不到時整張當前景。"""
    seg = segment_pieces(piece_bgr)
    pieces = extract_piece_images(piece_bgr, seg)
    if not pieces:
        bgra = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra
    max_idx = int(np.argmax([p.area for p in seg.pieces]))
    return pieces[max_idx]


def _ref_label(row: Optional[int], col: Optional[int]) -> Optional[str]:
    if row is None or col is None:
        return None
    return f"R{row}C{col}"


def run_locate(reference_bgr: np.ndarray, piece_bgr: np.ndarray,
               rows: int, cols: int, top_k: int = 5) -> dict[str, Any]:
    """執行定位並回傳可 JSON 序列化的結果。

    回傳欄位：
      grid_pos: [row, col] 或 None（1-indexed）
      ref_label: "R{row}C{col}" 或 None
      conf: 0~100 整數百分比（前端直接顯示）
      confidence_raw: 0~1 原始信心度
      suggested_rotation: 0/90/180/270 或 None
      method: "feature" | "template"
      top_cells: [{row, col, ref_label, score}]，依分數高到低，最多 top_k 筆
      region_hint: {row_range:[r0,r1], col_range:[c0,c1], note} 或 None（候選分散時）
    """
    piece_bgra = _piece_bgra_from_photo(piece_bgr)
    res = locate_piece(reference_bgr, piece_bgra, rows=rows, cols=cols)

    grid = list(res.grid_pos) if res.grid_pos else None
    row = grid[0] if grid else None
    col = grid[1] if grid else None

    top_cells = []
    for c in (res.top_cells or [])[:top_k]:
        gp = c.get("grid_pos")
        if not gp:
            continue
        r, cc = int(gp[0]), int(gp[1])
        top_cells.append({
            "row": r,
            "col": cc,
            "ref_label": _ref_label(r, cc),
            "score": round(float(c.get("score", 0.0)), 4),
        })

    region = None
    if res.region_hint:
        rh = res.region_hint
        region = {
            "row_range": [int(x) for x in rh["row_range"]] if rh.get("row_range") else None,
            "col_range": [int(x) for x in rh["col_range"]] if rh.get("col_range") else None,
            "note": rh.get("note"),
        }

    return {
        "grid_pos": grid,
        "ref_label": _ref_label(row, col),
        "conf": round(float(res.confidence) * 100),
        "confidence_raw": round(float(res.confidence), 4),
        "suggested_rotation": res.suggested_rotation,
        "method": res.method,
        "top_cells": top_cells,
        "region_hint": region,
    }
