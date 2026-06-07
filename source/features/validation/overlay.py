from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from sources.features.segmentation.detector import SegmentationResult
from sources.features.validation.metrics import _auto_canny


def draw_contour_overlay(image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = _auto_canny(gray)
    tolerance_px = 3
    dilated_edges = cv2.dilate(
        edge_map,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_px * 2 + 1, tolerance_px * 2 + 1)),
    )

    canvas = image.copy()
    h, w = canvas.shape[:2]

    edge_overlay = np.zeros_like(canvas)
    edge_overlay[edge_map > 0] = (0, 0, 180)
    canvas = cv2.addWeighted(canvas, 1.0, edge_overlay, 0.4, 0)

    for piece in result.pieces:
        points = piece.contour.reshape(-1, 2)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            x, y = int(p1[0]), int(p1[1])
            on_edge = (0 <= y < h and 0 <= x < w and dilated_edges[y, x] > 0)
            color = (255, 255, 0) if on_edge else (0, 255, 0)
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

    return canvas


def draw_edge_heatmap(image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = _auto_canny(gray)

    dist_map = distance_transform_edt(edge_map == 0)

    canvas = (image * 0.4).astype(np.uint8)

    for piece in result.pieces:
        points = piece.contour.reshape(-1, 2)
        h, w = image.shape[:2]

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            x, y = int(p1[0]), int(p1[1])

            if 0 <= y < h and 0 <= x < w:
                d = dist_map[y, x]
            else:
                d = 10

            if d <= 2:
                color = (0, 255, 0)
            elif d <= 5:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

    return canvas


def save_validation_overlays(
    image: np.ndarray,
    result: SegmentationResult,
    output_dir: Path,
    image_name: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    contour_img = draw_contour_overlay(image, result)
    p = output_dir / f"{image_name}_contour_overlay.jpg"
    cv2.imwrite(str(p), contour_img)
    saved.append(p)

    heatmap_img = draw_edge_heatmap(image, result)
    p = output_dir / f"{image_name}_edge_heatmap.jpg"
    cv2.imwrite(str(p), heatmap_img)
    saved.append(p)

    return saved
