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

# 使用 pieces_c3_r26.jpg 作為代表
p_file = data_dir / "pieces_c3_r26.jpg"
p_img = cv2.imread(str(p_file))

# 進行去背
seg_res = segment_pieces(p_img)
piece_images = extract_piece_images(p_img, seg_res)

if len(piece_images) > 0:
    max_idx = int(np.argmax([p.area for p in seg_res.pieces]))
    chosen_bgra = piece_images[max_idx]
    piece_bgr = chosen_bgra[:, :, :3]
    piece_alpha = chosen_bgra[:, :, 3]
    
    # 提取特徵 (使用 SIFT，不限縮放)
    sift = cv2.SIFT_create()
    kp_piece, des_piece = sift.detectAndCompute(piece_bgr, piece_alpha)
    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
    
    if des_piece is not None and des_ref is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_piece, des_ref, k=2)
        
        # 使用放寬的 ratio 0.82 收集匹配點
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.82 * n.distance:
                    good_matches.append(m)
                    
        # 繪製匹配連線
        # 為了避免連線過多雜亂，我們只取 RANSAC 的 inliers
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
                print(f"找到 {len(inlier_matches)} 個 inliers。正在繪製連線並儲存...")
                
                # 繪製 inliers
                out_img = cv2.drawMatches(
                    piece_bgr, kp_piece,
                    ref_img, kp_ref,
                    inlier_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                output_dir = data_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / "visual_sift_match.jpg"
                cv2.imwrite(str(out_path), out_img)
                print(f"成功儲存視覺匹配連線圖至: {out_path.absolute()}")
                
                # 印出這幾個 inliers 在大圖上的具體坐標
                print("Inlier 座標點在大圖上:")
                for m in inlier_matches:
                    pt = kp_ref[m.trainIdx].pt
                    print(f"  - (x={pt[0]:.1f}, y={pt[1]:.1f})")
            else:
                print("RANSAC 估算單應性矩陣失敗。")
        else:
            print("好的匹配點太少，無法執行 RANSAC。")
else:
    print("找不到碎片。")
