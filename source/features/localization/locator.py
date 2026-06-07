import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class LocateResult:
    quad: Optional[np.ndarray]          # 完成圖上的四邊形落點 (4x2)，失敗為 None
    bounding_box: Optional[Tuple[int, int, int, int]]  # 外接矩形 (x, y, w, h)，失敗為 None
    rotation_deg: Optional[float]       # 由 homography/模板匹配推得的建議旋轉角
    suggested_rotation: Optional[int]   # 規整至最近的 90 度倍數 (0, 90, 180, 270)
    confidence: float                  # 0~1 信心度
    method: str                        # "feature" | "template"
    candidates: List[Tuple[int, int, int, int, float]]  # 候選框清單 [(x, y, w, h, score), ...]
    grid_pos: Optional[Tuple[int, int]] # (row, col)，1-indexed
    annotated_reference: np.ndarray    # 已畫框 of 完成圖

def _draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1, dash_length: int = 8):
    """繪製虛線段的輔助函數"""
    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    pts_count = int(dist / dash_length)
    if pts_count == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return
        
    for i in range(pts_count):
        start_t = i / pts_count
        end_t = (i + 0.5) / pts_count
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * start_t), int(pt1[1] + (pt2[1] - pt1[1]) * start_t))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * end_t), int(pt1[1] + (pt2[1] - pt1[1]) * end_t))
        cv2.line(img, p1, p2, color, thickness)

def _draw_dashed_rectangle(img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 1, dash_length: int = 8):
    """繪製虛線矩形的輔助函數"""
    x, y, w, h = bbox
    pt1 = (x, y)
    pt2 = (x + w, y)
    pt3 = (x + w, y + h)
    pt4 = (x, y + h)
    _draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)
    _draw_dashed_line(img, pt2, pt3, color, thickness, dash_length)
    _draw_dashed_line(img, pt3, pt4, color, thickness, dash_length)
    _draw_dashed_line(img, pt4, pt1, color, thickness, dash_length)

def _get_grid_position(center_x: float, center_y: float, ref_w: int, ref_h: int, rows: int, cols: int) -> Tuple[int, int]:
    """計算給定中心點落在第幾行第幾列 (1-indexed)"""
    gw = ref_w / cols
    gh = ref_h / rows
    col = int(center_x // gw) + 1
    row = int(center_y // gh) + 1
    # 限制邊界
    col = max(1, min(cols, col))
    row = max(1, min(rows, row))
    return row, col

def _standardize_rotated_rect(rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    將 cv2.minAreaRect 的輸出標準化：
    確保 width 永遠是長邊，height 永遠是短邊，並計算將長邊旋轉至「水平」所需的角度 (aligned_angle)。
    """
    (cx, cy), (w, h), angle = rect
    
    # 確保 w 是長邊，h 是短邊
    if w < h:
        w, h = h, w
        # 當寬高互換時，角度需要進行跳變補償
        aligned_angle = angle + 90.0 if angle < 0 else angle - 90.0
    else:
        aligned_angle = angle
        
    # 將角度限制在 [-45, 45] 度之間，方便後續直角旋轉處理
    if aligned_angle < -45:
        aligned_angle += 90
    elif aligned_angle > 45:
        aligned_angle -= 90
        
    return (cx, cy), (w, h), aligned_angle

def _get_puzzle_body_rect(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    透過形態學開運算去除拼圖的凸耳，取得主體輪廓的最小外接矩形。
    """
    # 尋找原始輪廓以估算大小
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    _, (w, h), _ = rect
    
    # 動態設定結構元素大小（以短邊的 18% 作為開運算半徑，可有效切除凸耳）
    ksize = int(min(w, h) * 0.18)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(5, ksize) # 確保 kernel 夠大
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 重新尋找主體輪廓
    contours_body, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_body:
        return rect # Fallback 到原始矩形
        
    body_contour = max(contours_body, key=cv2.contourArea)
    return cv2.minAreaRect(body_contour)

def locate_piece(
    reference: np.ndarray,
    piece_bgra: np.ndarray,
    rows: Optional[int] = None,
    cols: Optional[int] = None
) -> LocateResult:
    """
    定位碎片在完成大圖中的位置。
    優先使用 SIFT 特徵匹配。若 SIFT 失敗，則使用色彩直方圖過濾與多尺度模板匹配退路。
    """
    ref_h, ref_w = reference.shape[:2]
    
    # 預設 rows 與 cols 劃分
    target_rows = rows if rows is not None else 15
    target_cols = cols if cols is not None else 15
    gw = ref_w / target_cols
    gh = ref_h / target_rows
    
    # 1. 準備單片資料
    piece_bgr = piece_bgra[:, :, :3]
    piece_alpha = piece_bgra[:, :, 3]
    
    # 計算前景點，以備後續定位投影
    fg_coords = np.argwhere(piece_alpha > 127)
    if len(fg_coords) > 0:
        y_indices, x_indices = fg_coords[:, 0], fg_coords[:, 1]
        px, py = np.min(x_indices), np.min(y_indices)
        pw, ph = np.max(x_indices) - px, np.max(y_indices) - py
        pts = np.float32([[px, py], [px + pw, py], [px + pw, py + ph], [px, py + ph]]).reshape(-1, 1, 2)
    else:
        ph, pw = piece_bgra.shape[:2]
        pts = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]]).reshape(-1, 1, 2)
        
    sift = cv2.SIFT_create()
    
    # 2. SIFT 第一層：大圖縮小後的特徵匹配 (以應對高解析度大圖特徵點過多、Lowe's ratio 被誤殺的問題)
    sift_success = False
    quad = None
    bounding_box = None
    rotation_deg = None
    suggested_rotation = None
    confidence = 0.0
    grid_pos = None
    
    # 決定縮放因子 (升至 2560 像素以保留更多細節特徵)
    max_dim = max(ref_h, ref_w)
    if max_dim > 2560:
        scale_factor = 2560.0 / max_dim
        ref_scaled = cv2.resize(reference, (int(ref_w * scale_factor), int(ref_h * scale_factor)))
    else:
        scale_factor = 1.0
        ref_scaled = reference
        
    kp_piece, des_piece = sift.detectAndCompute(piece_bgr, piece_alpha)
    kp_ref, des_ref = sift.detectAndCompute(ref_scaled, None)
    
    if des_piece is not None and des_ref is not None and len(kp_piece) >= 4 and len(kp_ref) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_piece, des_ref, k=2)
        
        # Lowe's ratio test (放寬至 0.80 以獲取更多真實匹配點)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.80 * n.distance:
                    good_matches.append(m)
                    
        print(f"[INTERNAL SIFT - 全圖降採樣] 縮放比例: {scale_factor:.3f}, kp_piece: {len(kp_piece)}, kp_ref: {len(kp_ref)}, good_matches: {len(good_matches)}")
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H_scaled, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_scaled is not None:
                inliers_count = int(np.sum(mask))
                inliers_ratio = inliers_count / len(good_matches)
                print(f"[INTERNAL SIFT - 全圖降採樣] H found. inliers: {inliers_count}, ratio: {inliers_ratio:.3f}")
                
                # 判定 SIFT 成功 (加嚴 inliers 門檻以避免誤匹配)
                if inliers_count >= 10 and inliers_ratio >= 0.35:
                    # 投影至降採樣大圖
                    quad_scaled = cv2.perspectiveTransform(pts, H_scaled).reshape(4, 2)
                    # 映射回原始大圖坐標系
                    quad_candidate = quad_scaled / scale_factor
                    
                    # --- 幾何合理性驗證 (防呆機制) ---
                    is_convex = cv2.isContourConvex(quad_candidate.astype(np.int32).reshape(-1, 1, 2))
                    quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                    expected_area = gw * gh
                    area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)
                    
                    x_min, y_min = np.min(quad_candidate, axis=0)
                    x_max, y_max = np.max(quad_candidate, axis=0)
                    boundary_ok = (x_min >= -gw and y_min >= -gh and x_max <= ref_w + gw and y_max <= ref_h + gh)
                    
                    if is_convex and area_ok and boundary_ok:
                        sift_success = True
                        confidence = min(1.0, inliers_count / 20.0 * 0.7 + inliers_ratio * 0.3)
                        quad = quad_candidate
                        bounding_box = (
                            max(0, int(x_min)),
                            max(0, int(y_min)),
                            min(ref_w, int(x_max - x_min)),
                            min(ref_h, int(y_max - y_min))
                        )
                        
                        rotation_rad = np.arctan2(H_scaled[1, 0], H_scaled[0, 0])
                        rotation_deg = np.degrees(rotation_rad) % 360
                        suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
                        
                        if rows is not None and cols is not None:
                            center_x = (x_min + x_max) / 2.0
                            center_y = (y_min + y_max) / 2.0
                            grid_pos = _get_grid_position(center_x, center_y, ref_w, ref_h, rows, cols)
                    else:
                        print(f"[INTERNAL SIFT - 全圖降採樣] Homography 幾何驗證失敗: is_convex={is_convex}, area_ok={area_ok}, boundary_ok={boundary_ok}")
            
            # 如果 Homography 失敗，嘗試 Affine 變換作為 Fallback
            if not sift_success:
                print("[INTERNAL SIFT - 全圖降採樣] 嘗試仿射變換 Fallback...")
                M, affine_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is not None:
                    inliers_count = int(np.sum(affine_mask))
                    inliers_ratio = inliers_count / len(good_matches)
                    print(f"[INTERNAL SIFT - 全圖降採樣] Affine found. inliers: {inliers_count}, ratio: {inliers_ratio:.3f}")
                    
                    if inliers_count >= 5 and inliers_ratio >= 0.30:
                        quad_scaled = cv2.transform(pts, M).reshape(4, 2)
                        quad_candidate = quad_scaled / scale_factor
                        
                        quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                        expected_area = gw * gh
                        area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)
                        
                        x_min, y_min = np.min(quad_candidate, axis=0)
                        x_max, y_max = np.max(quad_candidate, axis=0)
                        boundary_ok = (x_min >= -gw and y_min >= -gh and x_max <= ref_w + gw and y_max <= ref_h + gh)
                        
                        if area_ok and boundary_ok:
                            sift_success = True
                            confidence = min(1.0, inliers_count / 15.0 * 0.7 + inliers_ratio * 0.3)
                            quad = quad_candidate
                            bounding_box = (
                                max(0, int(x_min)),
                                max(0, int(y_min)),
                                min(ref_w, int(x_max - x_min)),
                                min(ref_h, int(y_max - y_min))
                            )
                            rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                            rotation_deg = np.degrees(rotation_rad) % 360
                            suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
                            
                            if rows is not None and cols is not None:
                                center_x = (x_min + x_max) / 2.0
                                center_y = (y_min + y_max) / 2.0
                                grid_pos = _get_grid_position(center_x, center_y, ref_w, ref_h, rows, cols)
                            print(f"[INTERNAL SIFT - 全圖降採樣] 仿射變換成功！網格: {grid_pos}")

    # 3. SIFT 失敗，準備退路機制。
    # 計算色彩直方圖：使用 BGR 三通道三維直方圖 (8x8x8=512 bins)
    # 這能徹底解決低飽和度/中性色在 HSV 空間下的 Hue 隨機噪聲問題
    hist_piece = cv2.calcHist([piece_bgr], [0, 1, 2], piece_alpha, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_piece, hist_piece, 0, 1, cv2.NORM_MINMAX)
    
    grid_scores = []
    for r in range(1, target_rows + 1):
        for c in range(1, target_cols + 1):
            gx = int((c - 1) * gw)
            gy = int((r - 1) * gh)
            g_w = int(c * gw) - gx
            g_h = int(r * gh) - gy
            
            grid_patch = reference[gy:gy+g_h, gx:gx+g_w]
            hist_grid = cv2.calcHist([grid_patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_grid, hist_grid, 0, 1, cv2.NORM_MINMAX)
            
            score = cv2.compareHist(hist_piece, hist_grid, cv2.HISTCMP_CORREL)
            grid_scores.append((score, (r, c), (gx, gy, g_w, g_h)))
            
    # 排序取得前 15 個相似網格 (擴大至 15 個，以增加對光照變異大的實拍單片網格之涵蓋率)
    grid_scores = sorted(grid_scores, key=lambda x: x[0], reverse=True)
    top_k_grids = grid_scores[:min(15, len(grid_scores))]
    
    # 局部 SIFT 嘗試 (只在直方圖前 15 名網格的局部區域進行匹配)
    if not sift_success and des_piece is not None and len(kp_piece) >= 4:
        print("[INTERNAL SIFT] 全圖降採樣匹配失敗，進入局部 SIFT 嘗試...")
        best_local_inliers = 0
        best_local_result = None
        
        for hist_score, (r, c), (gx, gy, g_w, g_h) in top_k_grids:
            expand_w = g_w // 2
            expand_h = g_h // 2
            sx = max(0, gx - expand_w)
            sy = max(0, gy - expand_h)
            ex = min(ref_w, gx + g_w + expand_w)
            ey = min(ref_h, gy + g_h + expand_h)
            
            patch = reference[sy:ey, sx:ex]
            if patch.shape[0] < 10 or patch.shape[1] < 10:
                continue
                
            kp_patch, des_patch = sift.detectAndCompute(patch, None)
            if des_patch is not None and len(kp_patch) >= 4:
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des_piece, des_patch, k=2)
                
                good = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                            
                if len(good) >= 4:
                    src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_patch[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    use_affine = False
                    quad_candidate = None
                    rotation_deg = None
                    suggested_rotation = None
                    current_inliers = 0
                    current_ratio = 0.0
                    local_sift_ok = False
                    
                    H_local, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H_local is not None:
                        inliers = int(np.sum(mask))
                        ratio = inliers / len(good)
                        if inliers >= 8 and ratio >= 0.40:
                            quad_local = cv2.perspectiveTransform(pts, H_local).reshape(4, 2)
                            quad_candidate = quad_local + np.float32([sx, sy])
                            
                            is_convex = cv2.isContourConvex(quad_candidate.astype(np.int32).reshape(-1, 1, 2))
                            quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                            expected_area = gw * gh
                            area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)
                            
                            if is_convex and area_ok:
                                local_sift_ok = True
                                current_inliers = inliers
                                current_ratio = ratio
                                rotation_rad = np.arctan2(H_local[1, 0], H_local[0, 0])
                                rotation_deg = np.degrees(rotation_rad) % 360
                                suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
                            else:
                                use_affine = True
                        else:
                            use_affine = True
                    else:
                        use_affine = True
                        
                    if use_affine:
                        M, affine_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                        if M is not None:
                            inliers = int(np.sum(affine_mask))
                            ratio = inliers / len(good)
                            if inliers >= 5 and ratio >= 0.35:
                                quad_local = cv2.transform(pts, M).reshape(4, 2)
                                quad_candidate = quad_local + np.float32([sx, sy])
                                
                                quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                                expected_area = gw * gh
                                area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)
                                
                                if area_ok:
                                    local_sift_ok = True
                                    current_inliers = inliers
                                    current_ratio = ratio
                                    rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                                    rotation_deg = np.degrees(rotation_rad) % 360
                                    suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
                                    
                    if local_sift_ok and current_inliers > best_local_inliers:
                        best_local_inliers = current_inliers
                        x_min, y_min = np.min(quad_candidate, axis=0)
                        x_max, y_max = np.max(quad_candidate, axis=0)
                        bbox_ref = (
                            max(0, int(x_min)),
                            max(0, int(y_min)),
                            min(ref_w, int(x_max - x_min)),
                            min(ref_h, int(y_max - y_min))
                        )
                        
                        center_x = (x_min + x_max) / 2.0
                        center_y = (y_min + y_max) / 2.0
                        grid_pos_pred = _get_grid_position(center_x, center_y, ref_w, ref_h, target_rows, target_cols)
                        
                        best_local_result = {
                            'quad': quad_candidate,
                            'bbox': bbox_ref,
                            'rot': rotation_deg,
                            'sugg_rot': suggested_rotation,
                            'conf': min(1.0, current_inliers / 15.0 * 0.7 + current_ratio * 0.3),
                            'grid': grid_pos_pred
                        }
                    else:
                        if not local_sift_ok and 'is_convex' in locals():
                            print(f"[INTERNAL SIFT - 局部] 網格 {r},{c} 幾何驗證未通過: is_convex={is_convex if 'is_convex' in locals() else 'N/A'}, area_ok={area_ok if 'area_ok' in locals() else 'N/A'}")
                                    
        if best_local_result is not None:
            print(f"[INTERNAL SIFT - 局部網格匹配成功] 網格: {best_local_result['grid']}, inliers: {best_local_inliers}")
            sift_success = True
            quad = best_local_result['quad']
            bounding_box = best_local_result['bbox']
            rotation_deg = best_local_result['rot']
            suggested_rotation = best_local_result['sugg_rot']
            confidence = best_local_result['conf']
            grid_pos = best_local_result['grid']

    annotated = reference.copy()
    
    if sift_success:
        # 繪製 SIFT 定位結果
        pts_poly = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_poly], True, (0, 255, 0), 3) # 綠色粗實線
        
        x, y, w, h = bounding_box
        text_y = max(30, y - 10)
        info_text = f"SIFT Match (Rot: {rotation_deg:.1f}deg, Sugg: {suggested_rotation}deg)"
        if grid_pos is not None:
            info_text += f" Grid: R{grid_pos[0]} C{grid_pos[1]}"
            
        cv2.putText(annotated, info_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return LocateResult(
            quad=quad,
            bounding_box=bounding_box,
            rotation_deg=rotation_deg,
            suggested_rotation=suggested_rotation,
            confidence=confidence,
            method="feature",
            candidates=[],
            grid_pos=grid_pos,
            annotated_reference=annotated
        )
        
    # 4. SIFT 完全失敗，進入色彩直方圖 + 幾何正規化模板匹配退路方案
    print("[INTERNAL TEMPLATE] SIFT 匹配完全失敗，啟動幾何正規化模板退路匹配...")
    print(f"[INTERNAL TEMPLATE] top_k_grids (score, grid, bbox):")
    for score, grid, bbox in top_k_grids[:5]:
        print(f"  - Grid {grid}: score={score:.4f}, bbox={bbox}")
    
    # 取得大圖網格長短邊
    L_grid = max(gw, gh)
    S_grid = min(gw, gh)
    
    # 取得去背碎片主體外接矩形並標準化
    body_rect = _get_puzzle_body_rect(piece_alpha)
    if body_rect is None:
        # Fallback to standard contours if morphology opening fails
        contours, _ = cv2.findContours(piece_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            body_rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        else:
            body_rect = ((piece_alpha.shape[1]/2.0, piece_alpha.shape[0]/2.0), (piece_alpha.shape[1], piece_alpha.shape[0]), 0.0)
            
    _, (w_b, h_b), aligned_angle = _standardize_rotated_rect(body_rect)
    w_b = max(1.0, w_b)
    h_b = max(1.0, h_b)
    
    R_piece = w_b / h_b
    R_grid = L_grid / S_grid if S_grid > 0 else 1.0
    
    clean_piece_bgr = piece_bgr.copy()
    clean_piece_bgr[piece_alpha < 127] = 0
    
    # 計算主體比例因子：以長邊比對為準
    scale_factor = L_grid / w_b
    
    # 幾何正規化縮放
    h_p, w_p = piece_bgr.shape[:2]
    new_w = max(4, int(w_p * scale_factor))
    new_h = max(4, int(h_p * scale_factor))
    
    resized_piece = cv2.resize(clean_piece_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(piece_alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    template_candidates = []
    rotations = [0, 90, 180, 270]
    
    for hist_score, (r, c), (gx, gy, g_w, g_h) in top_k_grids:
        # 1. 長寬比過濾：比例偏差過大則直接排除（放寬至 0.2 以包容複雜凸耳形狀）
        if abs(R_piece - R_grid) > 0.20:
            continue
            
        # 2. 擴大網格候選搜尋區（向外擴展 25% 以容納可能旋轉越界的凸耳）
        expand_w = int(g_w * 0.25)
        expand_h = int(g_h * 0.25)
        sx = max(0, gx - expand_w)
        sy = max(0, gy - expand_h)
        ex = min(ref_w, gx + g_w + expand_w)
        ey = min(ref_h, gy + g_h + expand_h)
        
        search_area = reference[sy:ey, sx:ex]
        if search_area.shape[0] < 10 or search_area.shape[1] < 10:
            continue
            
        best_match_val = -1.0
        best_match_loc = (0, 0)
        best_match_size = (0, 0)
        best_rot = 0.0
        
        # 3. 測試 4 個直角方向的帶遮罩匹配
        for angle_offset in rotations:
            total_angle = aligned_angle + angle_offset
            
            # 計算旋轉矩陣，重新計算邊界大小以防影像被裁切
            M = cv2.getRotationMatrix2D((new_w / 2.0, new_h / 2.0), total_angle, 1.0)
            cos_val = np.abs(M[0, 0])
            sin_val = np.abs(M[0, 1])
            rot_w = int((new_h * sin_val) + (new_w * cos_val))
            rot_h = int((new_h * cos_val) + (new_w * sin_val))
            
            M[0, 2] += (rot_w / 2.0) - (new_w / 2.0)
            M[1, 2] += (rot_h / 2.0) - (new_h / 2.0)
            
            rot_templ = cv2.warpAffine(resized_piece, M, (rot_w, rot_h))
            rot_mask = cv2.warpAffine(resized_mask, M, (rot_w, rot_h), flags=cv2.INTER_NEAREST)
            
            rot_templ[rot_mask < 127] = 0
            _, binary_mask = cv2.threshold(rot_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 確保模板尺寸不超過搜尋區域
            s_h, s_w = search_area.shape[:2]
            if rot_h > s_h or rot_w > s_w:
                continue
                
            res = cv2.matchTemplate(search_area, rot_templ, cv2.TM_CCORR_NORMED, mask=binary_mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_match_size = (rot_w, rot_h)
                best_rot = total_angle
                
        if best_match_val > 0.1:
            tx = sx + best_match_loc[0]
            ty = sy + best_match_loc[1]
            tw, th = best_match_size
            combined_score = hist_score * 0.3 + best_match_val * 0.7
            template_candidates.append({
                'bbox': (tx, ty, tw, th),
                'score': combined_score,
                'grid_pos': (r, c),
                'rot': best_rot
            })
            
    template_candidates = sorted(template_candidates, key=lambda x: x['score'], reverse=True)
    top_candidates = template_candidates[:min(3, len(template_candidates))]
    
    print(f"[INTERNAL TEMPLATE] top_candidates:")
    for idx, c in enumerate(top_candidates):
        print(f"  - Rank {idx+1}: score={c['score']:.4f}, bbox={c['bbox']}, grid={c['grid_pos']}, rot={c['rot']}")
        
    if top_candidates:
        best = top_candidates[0]
        bx, by, bw, bh = best['bbox']
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3) # 綠色粗實線
        
        info_text = f"Rank 1 (Score: {best['score']:.2f}, Rot: {best['rot']:.1f}deg)"
        if rows is not None and cols is not None:
            info_text += f" Grid: R{best['grid_pos'][0]} C{best['grid_pos'][1]}"
        cv2.putText(annotated, info_text, (bx, max(30, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        colors = [(0, 255, 255), (0, 165, 255)] # 黃色, 橘色
        for idx, cand in enumerate(top_candidates[1:]):
            cx, cy, cw, ch = cand['bbox']
            color = colors[min(idx, len(colors) - 1)]
            _draw_dashed_rectangle(annotated, (cx, cy, cw, ch), color, thickness=2, dash_length=8)
            
            cand_text = f"Rank {idx+2} ({cand['score']:.2f}, {cand['rot']:.1f}deg)"
            if rows is not None and cols is not None:
                cand_text += f" R{cand['grid_pos'][0]}C{cand['grid_pos'][1]}"
            cv2.putText(annotated, cand_text, (cx, max(20, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        bounding_box = best['bbox']
        quad = np.array([
            [bx, by],
            [bx + bw, by],
            [bx + bw, by + bh],
            [bx, by + bh]
        ], dtype=np.float32)
        rotation_deg = best['rot']
        suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
        confidence = best['score']
        grid_pos = best['grid_pos'] if (rows is not None and cols is not None) else None
        candidates_out = [(c['bbox'][0], c['bbox'][1], c['bbox'][2], c['bbox'][3], c['score']) for c in top_candidates]
    else:
        bounding_box = None
        quad = None
        rotation_deg = None
        suggested_rotation = None
        confidence = 0.0
        grid_pos = None
        candidates_out = []
        
    return LocateResult(
        quad=quad,
        bounding_box=bounding_box,
        rotation_deg=rotation_deg,
        suggested_rotation=suggested_rotation,
        confidence=confidence,
        method="template",
        candidates=candidates_out,
        grid_pos=grid_pos,
        annotated_reference=annotated
    )
