from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field

from sources.features.segmentation.detector import SegmentationResult


@dataclass
class PieceQuality:
    index: int
    edge_alignment_ratio: float
    curvature_sign_reversals: int
    convex_defect_count: int
    shape_complexity: float
    solidity: float
    issues: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    image_name: str
    piece_qualities: list[PieceQuality]
    overall_pass: bool
    warnings: list[str] = field(default_factory=list)


def _auto_canny(gray: np.ndarray) -> np.ndarray:
    median_val = np.median(gray)
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    return cv2.Canny(gray, low, high)


def _compute_edge_alignment(
    contour: np.ndarray, edge_map: np.ndarray, tolerance_px: int = 3
) -> float:
    dilated = cv2.dilate(
        edge_map,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_px * 2 + 1, tolerance_px * 2 + 1)),
    )
    points = contour.reshape(-1, 2)
    h, w = dilated.shape[:2]
    on_edge = 0
    for x, y in points:
        if 0 <= y < h and 0 <= x < w and dilated[int(y), int(x)] > 0:
            on_edge += 1
    return on_edge / len(points) if len(points) > 0 else 0.0


def _compute_curvature_sign_reversals(contour: np.ndarray, sample_interval: int = 5) -> int:
    points = contour.reshape(-1, 2).astype(np.float64)
    n = len(points)
    if n < sample_interval * 3:
        return 0

    indices = np.arange(0, n, sample_interval)
    if len(indices) < 3:
        return 0

    sampled = points[indices]
    v1 = np.diff(sampled, axis=0)
    angles = np.arctan2(v1[:, 1], v1[:, 0])
    diffs = np.diff(angles)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0

    reversals = np.sum(np.diff(signs) != 0)
    return int(reversals)


def _compute_convex_defects(contour: np.ndarray, min_depth_ratio: float = 0.05) -> int:
    if len(contour) < 5:
        return 0

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0

    if defects is None:
        return 0

    area = cv2.contourArea(contour)
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
    min_depth = equiv_diameter * min_depth_ratio

    count = 0
    for d in defects:
        depth = d[0][3] / 256.0
        if depth > min_depth:
            count += 1
    return count


def _compute_shape_complexity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    return perimeter / np.sqrt(area)


def _compute_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return area / hull_area if hull_area > 0 else 0.0


def validate_segmentation(
    image: np.ndarray,
    result: SegmentationResult,
    image_name: str = "",
    edge_align_warn: float = 0.5,
    edge_align_fail: float = 0.3,
) -> ValidationReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = _auto_canny(gray)

    qualities = []
    warnings = []
    overall_pass = True

    for piece in result.pieces:
        contour = piece.contour
        issues = []

        alignment = _compute_edge_alignment(contour, edge_map)
        reversals = _compute_curvature_sign_reversals(contour)
        defect_count = _compute_convex_defects(contour)
        complexity = _compute_shape_complexity(contour)
        solidity = _compute_solidity(contour)

        if alignment < edge_align_fail:
            issues.append(f"edge alignment {alignment:.2f} below fail threshold {edge_align_fail}")
            overall_pass = False
        elif alignment < edge_align_warn:
            issues.append(f"edge alignment {alignment:.2f} below warn threshold {edge_align_warn}")
            warnings.append(f"piece {piece.index}: low edge alignment ({alignment:.2f})")

        contour_len = cv2.arcLength(contour, True)
        if contour_len > 0 and reversals / contour_len > 0.15:
            issues.append(f"high curvature sign reversals ({reversals}), possible jaggedness")
            warnings.append(f"piece {piece.index}: jagged contour")

        if defect_count == 0:
            issues.append("no convex defects — possible total feature loss")
            overall_pass = False
        elif defect_count < 2:
            issues.append(f"only {defect_count} convex defect(s) — possible tab/blank loss")
            warnings.append(f"piece {piece.index}: low defect count ({defect_count})")
        elif defect_count > 8:
            issues.append(f"{defect_count} convex defects — possible fragmentation")
            warnings.append(f"piece {piece.index}: high defect count ({defect_count})")

        if complexity < 4.0:
            issues.append(f"shape complexity {complexity:.2f} — possibly over-smoothed")
            warnings.append(f"piece {piece.index}: low shape complexity ({complexity:.2f})")

        qualities.append(PieceQuality(
            index=piece.index,
            edge_alignment_ratio=alignment,
            curvature_sign_reversals=reversals,
            convex_defect_count=defect_count,
            shape_complexity=complexity,
            solidity=solidity,
            issues=issues,
        ))

    return ValidationReport(
        image_name=image_name,
        piece_qualities=qualities,
        overall_pass=overall_pass,
        warnings=warnings,
    )


