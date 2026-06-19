#!/usr/bin/env python3
import cv2
import argparse
import sys
import json
from pathlib import Path
import numpy as np

# 加入專案路徑
sys.path.append(str(Path(__file__).parent.parent))

from source.features.segmentation.detector import segment_pieces, extract_piece_images
from source.features.localization.locator import locate_piece

def main():
    parser = argparse.ArgumentParser(description="拼圖定位輔助 CLI 工具 (MVP)")
    parser.add_argument("reference", type=str, help="完成的大圖 (盒面圖 / 參考圖) 路徑")
    parser.add_argument("piece_photo", type=str, help="手機拍攝的單片照路徑")
    parser.add_argument("--rows", type=int, default=None, help="拼圖總列數 (rows)")
    parser.add_argument("--cols", type=int, default=None, help="拼圖總行數 (cols)")
    parser.add_argument("--piece-index", type=int, default=-1, help="指定要定位的碎片 index (預設為最大片)")
    parser.add_argument("--config", type=str, default="data/project_config.json", help="專案設定檔路徑 (預設: data/project_config.json)")
    
    args = parser.parse_args()
    
    # 載入專案設定檔
    config_rows = None
    config_cols = None
    config_total = None
    config_path = Path(args.config)
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                config_rows = config_data.get("rows")
                config_cols = config_data.get("cols")
                config_total = config_data.get("total_pieces")
                print(f"已讀取設定檔 {config_path}: rows={config_rows}, cols={config_cols}, total_pieces={config_total}")
        except Exception as e:
            print(f"警告: 無法讀取設定檔 {config_path}，錯誤: {e}")
            
    # 決定最終使用的 rows 與 cols (優先使用命令列輸入，其次為設定檔)
    final_rows = args.rows if args.rows is not None else config_rows
    final_cols = args.cols if args.cols is not None else config_cols
    
    # 檢查 rows * cols 是否等於 total_pieces (防呆機制)
    if final_rows is not None and final_cols is not None and config_total is not None:
        if final_rows * final_cols != config_total:
            print(f"警告: rows ({final_rows}) * cols ({final_cols}) = {final_rows * final_cols}，與設定檔的 total_pieces ({config_total}) 不符！")
            
    ref_path = Path(args.reference)
    piece_path = Path(args.piece_photo)
    
    if not ref_path.exists():
        print(f"錯誤: 找不到完成大圖 {ref_path}")
        sys.exit(1)
    if not piece_path.exists():
        print(f"錯誤: 找不到單片照 {piece_path}")
        sys.exit(1)
        
    print("正在讀取影像...")
    ref_img = cv2.imread(str(ref_path))
    piece_img = cv2.imread(str(piece_path))
    
    if ref_img is None:
        print(f"錯誤: 無法讀取完成大圖 {ref_path}")
        sys.exit(1)
    if piece_img is None:
        print(f"錯誤: 無法讀取單片照 {piece_path}")
        sys.exit(1)
        
    print("正在執行去背處理...")
    # 進行去背
    seg_res = segment_pieces(piece_img)
    piece_images = extract_piece_images(piece_img, seg_res)
    
    chosen_piece_bgra = None
    piece_idx_used = -1
    
    if len(piece_images) == 0:
        print("提示: 去背模組未偵測到明顯碎片。將使用整張單片照直接定位 (設為全白遮罩)。")
        # 轉換為 BGRA，並把整個影像作為前景
        chosen_piece_bgra = cv2.cvtColor(piece_img, cv2.COLOR_BGR2BGRA)
        chosen_piece_bgra[:, :, 3] = 255
    else:
        print(f"去背成功: 共偵測到 {len(piece_images)} 個碎片。")
        # 印出所有碎片的資訊
        for idx, piece in enumerate(seg_res.pieces):
            print(f"  - 碎片 Index {idx}: 面積={piece.area:.1f}, Bbox={piece.bounding_box}")
            
        if args.piece_index >= 0:
            if args.piece_index < len(piece_images):
                piece_idx_used = args.piece_index
                chosen_piece_bgra = piece_images[piece_idx_used]
                print(f"使用使用者指定的碎片 Index {piece_idx_used}")
            else:
                print(f"錯誤: 指定的 index {args.piece_index} 超出偵測範圍 (0~{len(piece_images)-1})。")
                sys.exit(1)
        else:
            # 預設尋找面積最大的片
            max_idx = int(np.argmax([piece.area for piece in seg_res.pieces]))
            piece_idx_used = max_idx
            chosen_piece_bgra = piece_images[piece_idx_used]
            print(f"使用面積最大的碎片 (Index {piece_idx_used})")
            
    print("正在計算碎片在完成圖中的落點位置...")
    result = locate_piece(ref_img, chosen_piece_bgra, rows=final_rows, cols=final_cols)
    
    # 嘗試從單片照檔名解析行列真值 (如 piece_r3_c5.jpg, r03c05)
    import re
    stem = piece_path.stem
    gt_row = None
    gt_col = None
    match_r = re.search(r'r(?:ow)?\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
    match_c = re.search(r'c(?:ol)?\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
    if match_r and match_c:
        gt_row = int(match_r.group(1))
        gt_col = int(match_c.group(1))
        
    # 如果有真值且預測不合，在標註圖上用紅字提醒
    if gt_row is not None and gt_col is not None and result.grid_pos is not None:
        pred_row, pred_col = result.grid_pos
        if pred_row != gt_row or pred_col != gt_col:
            if result.bounding_box is not None:
                bx, by, bw, bh = result.bounding_box
                cv2.putText(
                    result.annotated_reference, 
                    f"MISMATCH (True Grid: R{gt_row} C{gt_col})", 
                    (bx, max(60, by - 35)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2
                )
    
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_out_path = output_dir / f"{stem}_clean.png"
    located_out_path = output_dir / f"{stem}_located.jpg"
    
    # 儲存結果
    cv2.imwrite(str(clean_out_path), chosen_piece_bgra)
    cv2.imwrite(str(located_out_path), result.annotated_reference)
    
    print("\n================ 定位結果 ================")
    print(f"定位方式: {'SIFT 特徵比對' if result.method == 'feature' else '色彩直方圖+模板匹配 fallback'}")
    print(f"信心度得分: {result.confidence:.3f}")
    
    if result.bounding_box is not None:
        print(f"落點方框 (x, y, w, h): {result.bounding_box}")
        if result.rotation_deg is not None:
            print(f"精確旋轉角: {result.rotation_deg:.1f} 度")
            print(f"建議旋轉方向: {result.suggested_rotation} 度 (0, 90, 180, 270)")
        
        # 顯示網格與真值比對
        if result.grid_pos is not None:
            print(f"預估拼圖網格位置: 第 {result.grid_pos[0]} 行，第 {result.grid_pos[1]} 列")
            if gt_row is not None and gt_col is not None:
                print(f"檔名真實網格位置: 第 {gt_row} 行，第 {gt_col} 列")
                if result.grid_pos == (gt_row, gt_col):
                    print("比對結果: 成功 (符合)")
                else:
                    print("比對結果: 失敗 (不符合) ❌")
        else:
            if final_rows is not None or final_cols is not None:
                print("預估拼圖網格位置: 無法推估 (定位失敗)")
            if gt_row is not None and gt_col is not None:
                print(f"檔名真實網格位置: 第 {gt_row} 行，第 {gt_col} 列")
    else:
        print("定位結果: 失敗 (未找到合適的落點區)")
        if gt_row is not None and gt_col is not None:
            print(f"檔名真實網格位置: 第 {gt_row} 行，第 {gt_col} 列")
        
    # 方案1：Top-K 候選清單 + 找不到精確位置時的搜尋區塊建議（單片+輔助情境）
    if getattr(result, "top_cells", None):
        print("\n---- 候選位置 Top-K（可由此清單挑選） ----")
        for i, c in enumerate(result.top_cells, 1):
            gp = c.get("grid_pos")
            if gp:
                print(f"  第 {i} 名: 第 {gp[0]} 行 第 {gp[1]} 列  "
                      f"(分數 {c.get('score', 0):.3f}, 旋轉 {c.get('rotation', 0):.0f} 度)")
    region = getattr(result, "region_hint", None)
    if region is not None:
        r0, r1 = region["row_range"]; c0, c1 = region["col_range"]
        print("\n⚠️ 未找到明確單一格，但前幾名集中，建議大概搜尋區塊（請在此範圍內逐格嘗試）：")
        print(f"   列 {r0}~{r1}、行 {c0}~{c1}（完成圖上以洋紅框標示）")
    elif len(getattr(result, "top_cells", []) or []) > 1:
        print("\nℹ️ 最可能為第 1 名（候選較分散）；若不符，請依上方清單試第 2、3 名。")

    print(f"\n[輸出成果儲存路徑]")
    print(f"1. 去背乾淨碎片: {clean_out_path.absolute()}")
    print(f"2. 標註落點完成圖: {located_out_path.absolute()}")
    print("==========================================")

if __name__ == "__main__":
    main()
