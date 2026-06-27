from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectedPiece:
    index: int
    contour: np.ndarray
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    area: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SegmentationResult:
    pieces: list[DetectedPiece]
    count: int
    annotated_image: np.ndarray

    @property
    def bounding_boxes(self) -> list[tuple[int, int, int, int]]:
        return [p.bounding_box for p in self.pieces]


def _sample_background_color(lab_image: np.ndarray) -> np.ndarray:
    h, w = lab_image.shape[:2]
    margin = int(min(h, w) * 0.03)
    border_pixels = np.vstack([
        lab_image[:margin, :].reshape(-1, 3),
        lab_image[-margin:, :].reshape(-1, 3),
        lab_image[margin:-margin, :margin].reshape(-1, 3),
        lab_image[margin:-margin, -margin:].reshape(-1, 3),
    ])
    return np.median(border_pixels, axis=0).astype(np.float32)


def _create_foreground_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    # --- Channel 1: Color distance from background ---
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_color = _sample_background_color(lab)

    diff = np.sqrt(np.sum((lab - bg_color) ** 2, axis=2))
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_thresh, _ = cv2.threshold(
        diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, color_mask = cv2.threshold(
        diff_u8, int(otsu_thresh * 0.5), 255, cv2.THRESH_BINARY
    )

    margin = max(5, int(min(h, w) * 0.02))
    border = np.concatenate([
        color_mask[:margin, :].ravel(),
        color_mask[-margin:, :].ravel(),
        color_mask[margin:-margin, :margin].ravel(),
        color_mask[margin:-margin, -margin:].ravel(),
    ])
    if np.mean(border) > 127:
        color_mask = cv2.bitwise_not(color_mask)

    # --- Channel 2: Saturation ---
    # Pieces have color (higher saturation), background is neutral gray (low saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    _, sat_mask = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    border_sat = np.concatenate([
        sat_mask[:margin, :].ravel(),
        sat_mask[-margin:, :].ravel(),
        sat_mask[margin:-margin, :margin].ravel(),
        sat_mask[margin:-margin, -margin:].ravel(),
    ])
    if np.mean(border_sat) > 127:
        sat_mask = cv2.bitwise_not(sat_mask)

    # --- Combine: union of color distance and saturation evidence ---
    mask = cv2.bitwise_or(color_mask, sat_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Bridge narrow internal gaps from high-contrast surface features
    bridge_size = max(7, int(min(h, w) * 0.025)) | 1
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge, iterations=1)

    # Fill holes inside each contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, contours, -1, 255, -1)

    return mask


def _watershed_split(
    component_mask: np.ndarray,
    image: np.ndarray,
    min_area: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a merged foreground component into individual pieces via watershed."""
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Try progressively lower thresholds until we find multiple peaks
    num_peaks = 0
    peak_labels = None
    for thresh in [0.5, 0.4, 0.3, 0.2]:
        _, peaks = cv2.threshold(dist_norm, thresh, 1.0, cv2.THRESH_BINARY)
        peaks_u8 = peaks.astype(np.uint8)
        num_peaks, peak_labels = cv2.connectedComponents(peaks_u8)
        if num_peaks > 2:
            break

    if num_peaks <= 2:
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return [(contours[0], component_mask)]
        return []

    markers = np.zeros(image.shape[:2], dtype=np.int32)
    markers[component_mask == 0] = 1
    for i in range(1, num_peaks):
        markers[peak_labels == i] = i + 1

    cv2.watershed(image, markers)

    # Collect all watershed regions
    all_regions: list[tuple[int, np.ndarray]] = []  # (area, mask)
    for i in range(2, num_peaks + 1):
        piece_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        piece_mask[markers == i] = 255
        region_area = cv2.countNonZero(piece_mask)
        if region_area < min_area:
            continue
        all_regions.append((region_area, piece_mask))

    if not all_regions:
        return []

    # Separate large pieces from small fragments
    max_area = max(a for a, _ in all_regions)
    size_threshold = max_area * 0.3

    large: list[np.ndarray] = []
    small: list[np.ndarray] = []
    for region_area, mask in all_regions:
        if region_area >= size_threshold:
            large.append(mask)
        else:
            small.append(mask)

    # Merge small fragments into nearest large piece (by centroid distance)
    for frag in small:
        if not large:
            break
        frag_coords = cv2.findNonZero(frag)
        if frag_coords is None:
            continue
        frag_cx = int(np.mean(frag_coords[:, 0, 0]))
        frag_cy = int(np.mean(frag_coords[:, 0, 1]))

        best_idx = 0
        best_dist = float("inf")
        for idx, lmask in enumerate(large):
            lcoords = cv2.findNonZero(lmask)
            if lcoords is None:
                continue
            lcx = int(np.mean(lcoords[:, 0, 0]))
            lcy = int(np.mean(lcoords[:, 0, 1]))
            d = (frag_cx - lcx) ** 2 + (frag_cy - lcy) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = idx
        large[best_idx] = cv2.bitwise_or(large[best_idx], frag)

    # Recover boundary pixels lost by watershed.
    # Dilate each piece mask and intersect with the component mask to reclaim
    # tab areas that watershed assigned to the wrong piece.
    # Use iterative claiming: each round, unclaimed component pixels adjacent
    # to a piece are added to that piece. If contested, the piece with more
    # adjacent pixels wins.
    unclaimed = component_mask.copy()
    for lmask in large:
        unclaimed = cv2.bitwise_and(unclaimed, cv2.bitwise_not(lmask))

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for _ in range(30):
        if cv2.countNonZero(unclaimed) == 0:
            break
        # Each piece tries to claim adjacent unclaimed pixels
        claims: list[np.ndarray] = []
        for lmask in large:
            dilated = cv2.dilate(lmask, dilate_kernel, iterations=1)
            claim = cv2.bitwise_and(dilated, unclaimed)
            claims.append(claim)
        # Resolve conflicts: pixel goes to the piece with the most overlap
        for px_idx in range(len(large)):
            contested = np.zeros_like(unclaimed)
            for other_idx in range(len(large)):
                if other_idx != px_idx:
                    contested = cv2.bitwise_or(contested, claims[other_idx])
            # Exclusive pixels: only this piece claims them
            exclusive = cv2.bitwise_and(claims[px_idx], cv2.bitwise_not(contested))
            large[px_idx] = cv2.bitwise_or(large[px_idx], exclusive)
            unclaimed = cv2.bitwise_and(unclaimed, cv2.bitwise_not(exclusive))

    # Extract contours from recovered masks
    results = []
    for mask in large:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            results.append((max(contours, key=cv2.contourArea), mask))

    return results


def _smooth_mask(mask: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
    """Smooth mask edges, then refine boundary with GrabCut if image provided."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    blurred = cv2.GaussianBlur(smoothed, (9, 9), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(smoothed)
    cv2.drawContours(filled, contours, -1, 255, -1)

    if image is not None:
        filled = _grabcut_refine(filled, image)

    return filled


def _grabcut_refine(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Refine mask boundary using GrabCut color model."""
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    definite_fg = cv2.erode(mask, erode_k, iterations=2)
    probable_region = cv2.dilate(mask, dilate_k, iterations=1)

    gc_mask = np.full(mask.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[probable_region > 0] = cv2.GC_PR_FGD
    gc_mask[mask > 0] = cv2.GC_PR_FGD
    gc_mask[definite_fg > 0] = cv2.GC_FGD

    if cv2.countNonZero(definite_fg) == 0 or np.count_nonzero(gc_mask == cv2.GC_BGD) == 0:
        return mask

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # 決定性化（非演算法變更）：grabCut 內部 GMM 以 cv::theRNG() 隨機初始化，導致同一張
    # 照片每次去背遮罩有 ~2% 差異。每次呼叫前固定全域 RNG seed，使每張圖去背結果可重現、
    # 與呼叫順序無關，消除下游定位（尤其金字塔粗掃對近平手片）的 run 間翻動。
    cv2.setRNGSeed(20240613)
    try:
        cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask

    result_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = np.zeros_like(mask)
    cv2.drawContours(result, contours, -1, 255, -1)
    return result


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0


def _nms(pieces: list[DetectedPiece], iou_threshold: float = 0.3) -> list[DetectedPiece]:
    if len(pieces) <= 1:
        return pieces
    pieces = sorted(pieces, key=lambda p: p.area, reverse=True)
    keep = []
    for piece in pieces:
        if all(
            _bbox_iou(piece.bounding_box, k.bounding_box) < iou_threshold
            for k in keep
        ):
            keep.append(piece)
    return keep


def segment_pieces(image: np.ndarray) -> SegmentationResult:
    h, w = image.shape[:2]
    image_area = h * w

    mask = _create_foreground_mask(image)

    min_piece_area = image_area * 0.003
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_piece_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8) * 255

        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        _, peaks = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
        num_peaks, _ = cv2.connectedComponents(peaks.astype(np.uint8))

        if num_peaks <= 2:
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                candidates.append((contours[0], component_mask))
        else:
            split = _watershed_split(component_mask, image, min_piece_area)
            valid = [(c, m) for c, m in split
                     if cv2.contourArea(c) >= area * 0.15]
            if len(valid) >= 2:
                candidates.extend(valid)
            else:
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    candidates.append((contours[0], component_mask))

    if not candidates:
        return SegmentationResult(
            pieces=[], count=0, annotated_image=image.copy()
        )

    # Median-based area filter
    areas = sorted([cv2.contourArea(c) for c, _ in candidates], reverse=True)
    median_area = areas[len(areas) // 2]
    area_min = median_area * 0.3
    area_max = median_area * 5.0

    pieces = []
    for contour, pmask in candidates:
        area = cv2.contourArea(contour)
        if area < area_min or area > area_max:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.4:
            continue

        # Smooth each piece mask for cleaner contours
        pmask = _smooth_mask(pmask, image)
        contours_smooth, _ = cv2.findContours(
            pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours_smooth:
            continue
        contour = max(contours_smooth, key=cv2.contourArea)
        bbox = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        pieces.append(DetectedPiece(
            index=len(pieces),
            contour=contour,
            bounding_box=bbox,
            area=area,
            mask=pmask,
        ))

    pieces = _nms(pieces)

    for i, piece in enumerate(pieces):
        piece.index = i

    annotated = image.copy()
    for piece in pieces:
        x, y, bw, bh = piece.bounding_box
        cv2.drawContours(annotated, [piece.contour], -1, (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
        cv2.putText(
            annotated, str(piece.index), (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

    return SegmentationResult(
        pieces=pieces,
        count=len(pieces),
        annotated_image=annotated,
    )


def extract_piece_images(
    image: np.ndarray, result: SegmentationResult
) -> list[np.ndarray]:
    piece_images = []
    for piece in result.pieces:
        x, y, w, h = piece.bounding_box

        if piece.mask is not None:
            cropped_mask = piece.mask[y:y + h, x:x + w]
        else:
            piece_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(piece_mask, [piece.contour], -1, 255, -1)
            cropped_mask = piece_mask[y:y + h, x:x + w]

        cropped_bgr = image[y:y + h, x:x + w]
        rgba = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = cropped_mask
        piece_images.append(rgba)

    return piece_images


def save_results(
    image_path: Path,
    output_dir: Path,
    result: SegmentationResult,
    piece_images: list[np.ndarray],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    annotated_path = output_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), result.annotated_image)

    pieces_dir = output_dir / stem
    pieces_dir.mkdir(exist_ok=True)
    for i, piece_img in enumerate(piece_images):
        piece_path = pieces_dir / f"piece_{i:03d}.png"
        cv2.imwrite(str(piece_path), piece_img)

    return {
        "annotated_image": str(annotated_path),
        "pieces_dir": str(pieces_dir),
        "count": result.count,
    }
