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

p_file = data_dir / "pieces_c3_r26.jpg"
p_img = cv2.imread(str(p_file))

seg_res = segment_pieces(p_img)
piece_images = extract_piece_images(p_img, seg_res)

if len(piece_images) > 0:
    max_idx = int(np.argmax([p.area for p in seg_res.pieces]))
    chosen_bgra = piece_images[max_idx]
    piece_bgr = chosen_bgra[:, :, :3]
    piece_alpha = chosen_bgra[:, :, 3]
    
    # 網格
    rows, cols = 40, 25
    gw = ref_w / cols
    gh = ref_h / rows
    
    # 真值
    gt_row, gt_col = 26, 3
    gx = int((gt_col - 1) * gw)
    gy = int((gt_row - 1) * gh)
    g_w = int(gt_col * gw) - gx
    g_h = int(gt_row * gh) - gy
    
    gt_patch = ref_img[gy:gy+g_h, gx:gx+g_w]
    
    # 計算平均色彩 (只針對單片前景)
    avg_color_piece = cv2.mean(piece_bgr, piece_alpha)[:3]
    avg_color_gt = cv2.mean(gt_patch)[:3]
    
    print(f"單片前景平均色彩 (BGR): {avg_color_piece}")
    print(f"大圖網格 (26, 3) 平均色彩 (BGR): {avg_color_gt}")
    
    # 輸出圖片以供檢查
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 把單片前景以外部分填黑存檔
    clean_piece = piece_bgr.copy()
    clean_piece[piece_alpha < 127] = 0
    
    cv2.imwrite(str(output_dir / "debug_piece.jpg"), clean_piece)
    cv2.imwrite(str(output_dir / "debug_gt_patch.jpg"), gt_patch)
    
    print(f"已儲存 debug_piece.jpg 及 debug_gt_patch.jpg 到 {output_dir}")
    
    # 測試：如果我們在大圖上做「色彩相似度最優搜尋」（不分網格，以 100x100 的滑動視窗），
    # 在 (gx, gy) 周邊 200 像素內，是否能找到色彩相似度極高的地方？
    # 這可以驗證是否是因為「邊框錯位」或「行列數定義偏移」導致真值位置不對
    best_local_score = -1.0
    best_local_loc = (0, 0)
    
    # 對單片直方圖
    h_bins, s_bins = 16, 16
    piece_hsv = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2HSV)
    hist_piece = cv2.calcHist([piece_hsv], [0, 1], piece_alpha, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist_piece, hist_piece, 0, 1, cv2.NORM_MINMAX)
    
    # 在 (gx, gy) 周圍 200 像素內進行滑動視窗直方圖匹配
    search_r = 150
    for dy in range(-search_r, search_r, 10):
        for dx in range(-search_r, search_r, 10):
            ny, nx = gy + dy, gx + dx
            if 0 <= nx < ref_w - g_w and 0 <= ny < ref_h - g_h:
                patch = ref_img[ny:ny+g_h, nx:nx+g_w]
                patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                hist_patch = cv2.calcHist([patch_hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
                cv2.normalize(hist_patch, hist_patch, 0, 1, cv2.NORM_MINMAX)
                score = cv2.compareHist(hist_piece, hist_patch, cv2.HISTCMP_CORREL)
                if score > best_local_score:
                    best_local_score = score
                    best_local_loc = (nx, ny)
                    
    print(f"在真值周圍 {search_r} 像素內，最佳直方圖相似度為 {best_local_score:.4f}，出現在 x={best_local_loc[0]}, y={best_local_loc[1]}")
    print(f"原始網格真值為 x={gx}, y={gy}")
