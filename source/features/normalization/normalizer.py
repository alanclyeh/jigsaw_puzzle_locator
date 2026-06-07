from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sources.features.segmentation.detector import DetectedPiece, SegmentationResult


@dataclass
class NormalizedPiece:
    index: int
    image: np.ndarray = field(repr=False)
    rotation_angle: float
    original_size: tuple[int, int]
    scale_factor: float


def _compute_target_size(pieces: list[DetectedPiece]) -> int:
    if not pieces:
        return 200
    long_edges = []
    for piece in pieces:
        _, (w, h), _ = cv2.minAreaRect(piece.contour)
        long_edges.append(max(w, h))
    return int(np.median(long_edges))


def _normalize_angle(angle: float, w: float, h: float) -> float:
    # cv2.minAreaRect returns ((cx,cy), (w,h), angle)
    # We want the long edge horizontal, with minimal rotation (within ±45°)
    if w < h:
        angle += 90.0
    # Bring into [-45, 45] range
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return angle



def _rotate_and_crop_piece(
    image: np.ndarray, piece: DetectedPiece,
) -> tuple[np.ndarray, float]:
    """Rotate and crop a single piece.

    Returns (cropped_bgr, angle_degrees).
    """
    contour = piece.contour
    if len(contour) < 5:
        x, y, w, h = piece.bounding_box
        return image[y:y + h, x:x + w].copy(), 0.0

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    rot_angle = _normalize_angle(angle, rw, rh)

    # Square ROI: use the larger of minAreaRect diagonal and bbox diagonal
    x_bb, y_bb, w_bb, h_bb = piece.bounding_box
    diag_rect = np.sqrt(rw ** 2 + rh ** 2)
    diag_bbox = np.sqrt(w_bb ** 2 + h_bb ** 2)
    diag = int(np.ceil(max(diag_rect, diag_bbox) * 1.2))
    half = diag // 2
    # Use bbox center for better coverage when contour is incomplete
    cx_i = x_bb + w_bb // 2
    cy_i = y_bb + h_bb // 2

    img_h, img_w = image.shape[:2]

    # Source ROI bounds (may exceed image)
    src_y1, src_y2 = cy_i - half, cy_i + half
    src_x1, src_x2 = cx_i - half, cx_i + half

    # Destination bounds within the ROI canvas
    dst_y1 = max(0, -src_y1)
    dst_x1 = max(0, -src_x1)
    # Clamp source to image
    s_y1 = max(0, src_y1)
    s_x1 = max(0, src_x1)
    s_y2 = min(img_h, src_y2)
    s_x2 = min(img_w, src_x2)

    roi_h = src_y2 - src_y1
    roi_w = src_x2 - src_x1

    roi_bgr = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)

    copy_h = s_y2 - s_y1
    copy_w = s_x2 - s_x1
    roi_bgr[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = (
        image[s_y1:s_y2, s_x1:s_x2]
    )

    # Rotate around ROI center
    center = (roi_w // 2, roi_h // 2)
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rotated_bgr = cv2.warpAffine(
        roi_bgr, M, (roi_w, roi_h), borderValue=(0, 0, 0),
    )

    # Rotate the piece mask and use its bounds for cropping.
    # The mask may cover more area than the contour in edge regions.
    pmask = piece.mask
    if pmask is None:
        pmask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(pmask, [contour], -1, 255, -1)

    roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    roi_mask[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = (
        pmask[s_y1:s_y2, s_x1:s_x2]
    )
    rotated_mask = cv2.warpAffine(roi_mask, M, (roi_w, roi_h), borderValue=0)
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)

    # Use union of rotated contour bounds and rotated mask bounds
    contour_in_roi = contour.reshape(-1, 1, 2).astype(np.float32)
    contour_in_roi[:, :, 0] -= src_x1
    contour_in_roi[:, :, 1] -= src_y1
    rotated_pts = cv2.transform(contour_in_roi, M).reshape(-1, 2).astype(np.int32)
    cx1, cy1, cw, ch = cv2.boundingRect(rotated_pts)

    mask_coords = cv2.findNonZero(rotated_mask)
    if mask_coords is not None:
        mx1, my1, mw, mh = cv2.boundingRect(mask_coords)
        # Union of contour and mask bounds
        tx = min(cx1, mx1)
        ty = min(cy1, my1)
        tw = max(cx1 + cw, mx1 + mw) - tx
        th = max(cy1 + ch, my1 + mh) - ty
    else:
        tx, ty, tw, th = cx1, cy1, cw, ch

    margin = max(6, int(max(tw, th) * 0.10))
    tx = max(0, tx - margin)
    ty = max(0, ty - margin)
    tw = min(rotated_bgr.shape[1] - tx, tw + 2 * margin)
    th = min(rotated_bgr.shape[0] - ty, th + 2 * margin)

    cropped_bgr_raw = rotated_bgr[ty:ty + th, tx:tx + tw].copy()

    return cropped_bgr_raw, rot_angle


def normalize_pieces(
    image: np.ndarray,
    result: SegmentationResult,
    *,
    target_max_side: Optional[int] = None,
) -> list[NormalizedPiece]:
    if not result.pieces:
        return []

    if target_max_side is None:
        target_max_side = _compute_target_size(result.pieces)
    target_max_side = max(1, target_max_side)

    normalized = []
    for piece in result.pieces:
        bgr_raw, angle = _rotate_and_crop_piece(image, piece)

        h, w = bgr_raw.shape[:2]
        original_size = (w, h)
        scale = target_max_side / max(w, h)
        scale = max(0.1, min(2.0, scale))

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        bgr_raw = cv2.resize(bgr_raw, (new_w, new_h), interpolation=interp)

        normalized.append(NormalizedPiece(
            index=piece.index,
            image=bgr_raw,
            rotation_angle=angle,
            original_size=original_size,
            scale_factor=scale,
        ))

    return normalized


def save_normalized_pieces(
    pieces: list[NormalizedPiece],
    output_dir: Path,
    stem: str,
) -> dict:
    norm_dir = output_dir / f"{stem}_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    for piece in pieces:
        cv2.imwrite(str(norm_dir / f"piece_{piece.index:03d}.png"), piece.image)

    return {
        "normalized_dir": str(norm_dir),
        "count": len(pieces),
    }
