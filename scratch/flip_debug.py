import cv2
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.features.segmentation.detector import segment_pieces, extract_piece_images

data_dir = project_root / "data"
ref_path = data_dir / "reference_puzzle.jpg"
ref_img = cv2.imread(str(ref_path))

max_dim = max(ref_img.shape[:2])
scale_factor = 1600.0 / max_dim
ref_scaled = cv2.resize(ref_img, (int(ref_img.shape[1] * scale_factor), int(ref_img.shape[0] * scale_factor)))

sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(ref_scaled, None)

p_files = ["pieces_c3_r26.jpg", "pieces_c3_r25.jpg", "pieces_c23_r18.jpg", "pieces_c2_r38.jpg"]

for filename in p_files:
    p_path = data_dir / filename
    p_img = cv2.imread(str(p_path))
    
    seg_res = segment_pieces(p_img)
    piece_images = extract_piece_images(p_img, seg_res)
    
    if len(piece_images) == 0:
        continue
        
    max_idx = int(np.argmax([p.area for p in seg_res.pieces]))
    chosen_bgra = piece_images[max_idx]
    
    # 對單片影像進行水平翻轉
    flipped_bgra = cv2.flip(chosen_bgra, 1)
    
    print(f"\n--- 測試 {filename} (水平翻轉後) ---")
    
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rot_piece = flipped_bgra
        elif angle == 90:
            rot_piece = cv2.rotate(flipped_bgra, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot_piece = cv2.rotate(flipped_bgra, cv2.ROTATE_180)
        elif angle == 270:
            rot_piece = cv2.rotate(flipped_bgra, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        rot_bgr = rot_piece[:, :, :3]
        rot_alpha = rot_piece[:, :, 3]
        
        kp_p, des_p = sift.detectAndCompute(rot_bgr, rot_alpha)
        
        if des_p is not None and des_ref is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_p, des_ref, k=2)
            
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                        
            inliers = 0
            if len(good) >= 4:
                src_pts = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    inliers = int(np.sum(mask))
                    
            if inliers >= 4:
                print(f"  旋轉 {angle}°: good={len(good)}, inliers={inliers}")
            if inliers >= 8:
                inlier_dst_pts = dst_pts[mask.ravel() == 1]
                avg_x = np.mean(inlier_dst_pts[:, 0, 0]) / scale_factor
                avg_y = np.mean(inlier_dst_pts[:, 0, 1]) / scale_factor
                gw = ref_img.shape[1] / 25
                gh = ref_img.shape[0] / 40
                c = int(avg_x / gw) + 1
                r = int(avg_y / gh) + 1
                print(f"    [匹配成功] 大圖座標: x={avg_x:.1f}, y={avg_y:.1f} -> 網格: Row {r}, Col {c}")
