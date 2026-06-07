import cv2
import numpy as np
import random
from typing import Tuple

def generate_synthetic_piece(
    reference_image: np.ndarray,
    bbox: Tuple[int, int, int, int],  # x, y, w, h in reference
    rotation_deg: float,
    scale: float,
    perspective_shift_ratio: float = 0.03,  # 0~5% 的四角隨機位移
    brightness_shift: float = 0.0,         # -0.15 ~ 0.15
    contrast_shift: float = 0.0,           # -0.15 ~ 0.15
    blur_kernel_size: int = 0              # 3, 5 等奇數，0 代表不模糊
) -> Tuple[np.ndarray, np.ndarray]:
    """
    從 reference_image 裁切指定 bbox 的區域，並套用幾何與光照變形，
    生成模擬手機拍的單片碎片圖 (BGRA)。
    
    回傳:
        piece_bgra: 變形後的單片圖 (BGRA)
        piece_mask: 單片的 alpha 遮罩 (單通道 uint8, 0 或 255)
    """
    x, y, w, h = bbox
    patch = reference_image[y:y+h, x:x+w].copy()
    
    # 建立原始 patch 的 mask (初始為全白，即整個矩形都是前景)
    patch_mask = np.full((h, w), 255, dtype=np.uint8)
    
    # 為了模擬拼圖的不規則邊緣，我們可以在邊緣做一個簡單的去背效果 (比如橢圓形)
    # 這也能避免 Homography 估算時邊緣矩形特徵太明顯
    center_x, center_y = w // 2, h // 2
    axes = (int(w * 0.45), int(h * 0.45))
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    patch_mask = cv2.bitwise_and(patch_mask, ellipse_mask)

    # 1. 幾何變換：建立仿射或透視變換矩陣
    # 我們將先進行透視變形，再進行旋轉與縮放
    # 原始四角點
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    
    # 加入隨機透視位移 (四角隨機微調)
    dst_pts = src_pts.copy()
    if perspective_shift_ratio > 0:
        max_shift_w = w * perspective_shift_ratio
        max_shift_h = h * perspective_shift_ratio
        for i in range(4):
            dst_pts[i][0] += random.uniform(-max_shift_w, max_shift_w)
            dst_pts[i][1] += random.uniform(-max_shift_h, max_shift_h)
            
    M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # 接著進行旋轉與縮放
    # 為了避免旋轉後影像被裁切，我們需要計算旋轉後的新邊界大小
    # 計算透視變形後的大小
    warped_patch = cv2.warpPerspective(patch, M_perspective, (w, h))
    warped_mask = cv2.warpPerspective(patch_mask, M_perspective, (w, h))
    
    # 旋轉與縮放矩陣
    center = (w / 2.0, h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    
    # 計算旋轉後的外接矩形大小
    cos_val = np.abs(M_rot[0, 0])
    sin_val = np.abs(M_rot[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    # 修正旋轉矩陣中的平移向量，使旋轉中心對齊新影像中心
    M_rot[0, 2] += (new_w / 2.0) - center[0]
    M_rot[1, 2] += (new_h / 2.0) - center[1]
    
    # 套用旋轉與縮放
    rotated_patch = cv2.warpAffine(warped_patch, M_rot, (new_w, new_h))
    rotated_mask = cv2.warpAffine(warped_mask, M_rot, (new_w, new_h), flags=cv2.INTER_NEAREST)
    
    # 2. 光照變形：作用於前景部分 (mask > 0)
    fg_mask = (rotated_mask > 127)
    
    # 調整亮度與對比度
    # Formula: new_img = img * (1 + contrast_shift) + brightness_shift * 255
    temp_patch = rotated_patch.astype(np.float32)
    if contrast_shift != 0:
        temp_patch[fg_mask] = temp_patch[fg_mask] * (1.0 + contrast_shift)
    if brightness_shift != 0:
        temp_patch[fg_mask] = temp_patch[fg_mask] + brightness_shift * 255.0
        
    temp_patch = np.clip(temp_patch, 0, 255).astype(np.uint8)
    
    # 套用高斯模糊
    if blur_kernel_size > 0:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blurred = cv2.GaussianBlur(temp_patch, (blur_kernel_size, blur_kernel_size), 0)
        temp_patch[fg_mask] = blurred[fg_mask]
        
    # 合成 BGRA
    piece_bgra = cv2.cvtColor(temp_patch, cv2.COLOR_BGR2BGRA)
    piece_bgra[:, :, 3] = rotated_mask
    
    return piece_bgra, rotated_mask
