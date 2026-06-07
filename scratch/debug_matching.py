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
ref_h, ref_w = ref_img.shape[:2]

p_files = ["pieces_c3_r26.jpg", "pieces_c3_r25.jpg", "pieces_c23_r18.jpg", "pieces_c2_r38.jpg"]

sift = cv2.SIFT_create()

rows, cols = 40, 25
gw = ref_w / cols
gh = ref_h / rows

for filename in p_files:
    p_path = data_dir / filename
    p_img = cv2.imread(str(p_path))
    
    seg_res = segment_pieces(p_img)
    piece_images = extract_piece_images(p_img, seg_res)
    
    if len(piece_images) == 0:
        continue
        
    max_idx = int(np.argmax([p.area for p in seg_res.pieces]))
    chosen_bgra = piece_images[max_idx]
    piece_bgr = chosen_bgra[:, :, :3]
    piece_alpha = chosen_bgra[:, :, 3]
    
    kp_piece, des_piece = sift.detectAndCompute(piece_bgr, piece_alpha)
    
    if des_piece is None or len(kp_piece) < 4:
        print(f"{filename}: 單片特徵點不足。")
        continue
        
    print(f"\n=== 開始全網格掃描: {filename} ===")
    
    local_sift_results = []
    
    # 遍歷 1000 個網格
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            gx = int((c - 1) * gw)
            gy = int((r - 1) * gh)
            g_w = int(c * gw) - gx
            g_h = int(r * gh) - gy
            
            # 擴大 50%
            expand_w = g_w // 2
            expand_h = g_h // 2
            sx = max(0, gx - expand_w)
            sy = max(0, gy - expand_h)
            ex = min(ref_w, gx + g_w + expand_w)
            ey = min(ref_h, gy + g_h + expand_h)
            
            patch = ref_img[sy:ey, sx:ex]
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
                    
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        inliers = int(np.sum(mask))
                        ratio = inliers / len(good)
                        if inliers >= 4:
                            local_sift_results.append((inliers, ratio, (r, c)))
                            
    # 排序印出 inliers 前 5 名的網格
    local_sift_results = sorted(local_sift_results, key=lambda x: x[0], reverse=True)
    print(f"SIFT 局部匹配前 5 名網格:")
    for idx, (inliers, ratio, grid) in enumerate(local_sift_results[:5]):
        print(f"  {idx+1}. Grid {grid}: inliers={inliers}, ratio={ratio:.3f}")
