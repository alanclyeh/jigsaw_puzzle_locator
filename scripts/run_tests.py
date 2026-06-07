#!/usr/bin/env python3
import cv2
import numpy as np
import json
import re
import sys
from pathlib import Path

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.features.segmentation.detector import segment_pieces, extract_piece_images
from source.features.localization.locator import locate_piece

# 導入測試用以在內部執行
from tests.test_localization import (
    create_textured_image,
    create_low_texture_image,
    calc_iou
)
from tests._synthetic import generate_synthetic_piece

def run_unit_tests() -> dict:
    """手動執行單元測試並捕獲詳細結果，避免 pytest 外部相依與輸出收集問題"""
    results = {}
    
    # 測試 1: SIFT 高紋理定位
    try:
        ref = create_textured_image()
        gt_bbox = (100, 100, 80, 80)
        piece_bgra, _ = generate_synthetic_piece(
            ref, gt_bbox, 
            rotation_deg=30.0, 
            scale=1.0,
            perspective_shift_ratio=0.01,
            brightness_shift=0.05,
            contrast_shift=-0.05,
            blur_kernel_size=0
        )
        res = locate_piece(ref, piece_bgra, rows=3, cols=3)
        
        assert res.method == "feature", "應使用 SIFT 特徵比對"
        assert res.bounding_box is not None, "Bounding box 應不為 None"
        iou = calc_iou(res.bounding_box, gt_bbox)
        assert iou >= 0.6, f"IoU ({iou:.2f}) 應 >= 0.6"
        
        rot_diff = abs(res.rotation_deg - 30.0) % 360
        if rot_diff > 180:
            rot_diff = 360 - rot_diff
        assert rot_diff <= 15.0, f"旋轉誤差 ({rot_diff:.1f}) 應 <= 15度"
        
        results["test_sift_localization_success"] = {"status": "PASSED", "message": f"SIFT 定位成功。IoU: {iou:.3f}, 旋轉角: {res.rotation_deg:.1f}°"}
    except Exception as e:
        results["test_sift_localization_success"] = {"status": "FAILED", "message": str(e)}
        
    # 測試 2: Template 退路定位
    try:
        ref = create_low_texture_image()
        gt_bbox = (50, 180, 40, 40)
        piece_bgra, _ = generate_synthetic_piece(
            ref, gt_bbox, 
            rotation_deg=0.0, 
            scale=1.0,
            perspective_shift_ratio=0.0,
            brightness_shift=0.0,
            contrast_shift=0.0,
            blur_kernel_size=0
        )
        res = locate_piece(ref, piece_bgra, rows=2, cols=2)
        
        assert res.method == "template", "應觸發 template 退路"
        assert len(res.candidates) >= 1, "候選框數量應 >= 1"
        assert res.grid_pos == (2, 1), f"預期網格 (2, 1)，實際為 {res.grid_pos}"
        
        bx, by, bw, bh = res.bounding_box
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        assert 0 <= cx <= 150 and 150 <= cy <= 300, f"中心點 ({cx}, {cy}) 應在網格 (2, 1) 內"
        
        results["test_template_localization_fallback"] = {"status": "PASSED", "message": f"退路成功。網格: {res.grid_pos}"}
    except Exception as e:
        results["test_template_localization_fallback"] = {"status": "FAILED", "message": str(e)}
        
    return results

def main():
    print("==========================================")
    print("  開始執行拼圖定位自動化測試與真實資料驗證  ")
    print("==========================================")
    
    # 1. 讀取專案設定檔
    config_path = project_root / "data" / "project_config.json"
    rows, cols, total_pieces = 20, 30, 600
    version = "v1.0.0"
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                rows = config_data.get("rows", rows)
                cols = config_data.get("cols", cols)
                total_pieces = config_data.get("total_pieces", total_pieces)
                version = config_data.get("version", version)
                print(f"已載入專案配置: rows={rows}, cols={cols}, total_pieces={total_pieces}, 版號={version}")
        except Exception as e:
            print(f"警告: 無法載入設定檔，錯誤: {e}")
            
    # 2. 跑單元測試
    print("\n[第一階段] 執行模擬合成資料單元測試...")
    unit_results = run_unit_tests()
    for name, r in unit_results.items():
        print(f"  - {name}: {r['status']} ({r['message']})")
        
    # 3. 掃描真實資料
    print("\n[第二階段] 掃描 data/ 目錄下的實拍單片照...")
    data_dir = project_root / "data"
    ref_path = data_dir / "reference_puzzle.jpg"
    
    if not ref_path.exists():
        print(f"錯誤: 找不到參考大圖 {ref_path}，無法進行實拍資料定位！")
        sys.exit(1)
        
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        print(f"錯誤: 無法讀取大圖 {ref_path}")
        sys.exit(1)
        
    ref_h, ref_w = ref_img.shape[:2]
    
    # 找出所有符合 pieces_c*_r*.jpg 等命名格式的碎片
    piece_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p_file in data_dir.glob(ext):
            if p_file.name != "reference_puzzle.jpg" and not p_file.name.endswith("_clean.png") and not p_file.name.endswith("_located.jpg"):
                piece_files.append(p_file)
                
    real_results = []
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"找到 {len(piece_files)} 張待測試的單片相片。")
    
    for p_file in piece_files:
        stem = p_file.stem
        print(f"\n正在處理單片照: {p_file.name} ...")
        
        # 解析真實行列 (如 pieces_c23_r18.jpg)
        gt_row, gt_col = None, None
        match_r = re.search(r'r(?:ow)?\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
        match_c = re.search(r'c(?:ol)?\s*[-_]?\s*(\d+)', stem, re.IGNORECASE)
        if match_r and match_c:
            # 注意使用者的命名習慣中 c 在前，r 在後：pieces_c23_r18.jpg
            gt_row = int(match_r.group(1))
            gt_col = int(match_c.group(1))
            print(f"  -> 解析真值: Row {gt_row}, Col {gt_col}")
            
        # 進行去背
        p_img = cv2.imread(str(p_file))
        if p_img is None:
            print(f"  -> 錯誤: 無法讀取該相片")
            continue
            
        seg_res = segment_pieces(p_img)
        piece_images = extract_piece_images(p_img, seg_res)
        
        if len(piece_images) == 0:
            # Fallback 使用整張圖
            chosen_bgra = cv2.cvtColor(p_img, cv2.COLOR_BGR2BGRA)
            chosen_bgra[:, :, 3] = 255
            has_segmented = False
        else:
            # 選擇最大片
            max_idx = int(np.argmax([p.area for p in seg_res.pieces]))
            chosen_bgra = piece_images[max_idx]
            has_segmented = True
            
        # 進行定位
        res = locate_piece(ref_img, chosen_bgra, rows=rows, cols=cols)
        
        # 計算與真值網格的真實 IoU
        iou = 0.0
        grid_status = "UNKNOWN" # 符合, 不符合, 無法推算
        config_error = False
        
        if gt_row is not None and gt_col is not None:
            if gt_row > rows or gt_col > cols:
                print(f"  -> 警告: 檔名真值 (R{gt_row}, C{gt_col}) 超出了設定檔定義的範圍 (Rows:{rows}, Cols:{cols})！")
                config_error = True
                grid_status = "CONFIG_ERROR"
            else:
                # 計算真值網格的 bbox
                gw = ref_w / cols
                gh = ref_h / rows
                gx = int((gt_col - 1) * gw)
                gy = int((gt_row - 1) * gh)
                g_w = int(gt_col * gw) - gx
                g_h = int(gt_row * gh) - gy
                gt_bbox = (gx, gy, g_w, g_h)
                
                if res.bounding_box is not None:
                    iou = calc_iou(res.bounding_box, gt_bbox)
                    
                if res.grid_pos is not None:
                    if res.grid_pos == (gt_row, gt_col):
                        grid_status = "MATCHED"
                    else:
                        grid_status = "MISMATCHED"
                else:
                    grid_status = "FAILED"
                    
        # 繪製與儲存圖片
        clean_out = output_dir / f"{stem}_clean.png"
        located_out = output_dir / f"{stem}_located.jpg"
        
        # 若有真值且預測不合，在標註圖上用紅字提醒真值
        annotated_img = res.annotated_reference.copy()
        if grid_status == "MISMATCHED" and res.bounding_box is not None:
            bx, by, bw, bh = res.bounding_box
            cv2.putText(
                annotated_img, 
                f"MISMATCH (True Grid: R{gt_row} C{gt_col})", 
                (bx, max(60, by - 35)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 255), 2
            )
            
        cv2.imwrite(str(clean_out), chosen_bgra)
        cv2.imwrite(str(located_out), annotated_img)
        
        # 儲存此相片的測試數據
        real_results.append({
            "filename": p_file.name,
            "clean_img": f"output/{clean_out.name}",
            "located_img": f"output/{located_out.name}",
            "gt_pos": (gt_row, gt_col) if gt_row is not None else None,
            "pred_pos": res.grid_pos,
            "grid_status": grid_status,
            "config_error": config_error,
            "confidence": res.confidence,
            "method": res.method,
            "rotation_deg": res.rotation_deg,
            "suggested_rotation": res.suggested_rotation,
            "iou": iou,
            "has_segmented": has_segmented
        })
        
        print(f"  -> 定位完成。方式: {res.method}, 信心度: {res.confidence:.3f}, 網格比對: {grid_status}, IoU: {iou:.3f}")

    # 4. 統計與產生 HTML 測試報告
    print("\n[第三階段] 正在產生 HTML 測試報告...")
    
    total_real = len(real_results)
    matched_real = sum(1 for r in real_results if r["grid_status"] == "MATCHED")
    pass_rate_real = (matched_real / total_real * 100) if total_real > 0 else 0.0
    
    unit_total = len(unit_results)
    unit_passed = sum(1 for r in unit_results.values() if r["status"] == "PASSED")
    unit_pass_rate = (unit_passed / unit_total * 100) if unit_total > 0 else 0.0
    
    # 決定整體網頁背景與樣式 (使用精緻的 Sleek Dark Mode 搭配 HSL 配色)
    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>拼圖定位測試報告 - {version}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #4f46e5;
            --primary-glow: rgba(79, 70, 229, 0.4);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.15);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.15);
            --border-color: rgba(255, 255, 255, 0.08);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', 'Noto Sans TC', sans-serif;
            line-height: 1.6;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        
        .title-area h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #818cf8, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}
        
        .title-area p {{
            color: var(--text-muted);
            margin-top: 5px;
        }}
        
        .badge-version {{
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: 600;
            box-shadow: 0 0 15px var(--primary-glow);
            font-size: 0.95rem;
        }}
        
        /* Dashboard Grid */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--primary);
        }}
        
        .stat-card.success-card::before {{
            background-color: var(--success);
        }}
        
        .stat-card.warning-card::before {{
            background-color: var(--warning);
        }}
        
        .stat-info h3 {{
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 800;
        }}
        
        /* Circular Progress Chart */
        .progress-ring {{
            position: relative;
            width: 80px;
            height: 80px;
        }}
        
        .progress-ring-circle {{
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: conic-gradient(var(--success) {pass_rate_real}%, #1e293b 0);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .progress-ring-inner {{
            width: 64px;
            height: 64px;
            background-color: var(--card-bg);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
        }}
        
        /* Section styling */
        section {{
            margin-bottom: 45px;
        }}
        
        section h2 {{
            font-size: 1.6rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        /* Unit tests list */
        .unit-tests-list {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        .unit-test-item {{
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .unit-test-item:last-child {{
            border-bottom: none;
        }}
        
        .test-name {{
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .test-desc {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 4px;
        }}
        
        .status-badge {{
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        
        .status-passed {{
            background-color: var(--success-bg);
            color: var(--success);
        }}
        
        .status-failed {{
            background-color: var(--danger-bg);
            color: var(--danger);
        }}
        
        /* Real pieces grid */
        .pieces-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 25px;
        }}
        
        .piece-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .piece-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(79, 70, 229, 0.15);
            border-color: rgba(79, 70, 229, 0.3);
        }}
        
        .piece-header {{
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        
        .piece-title h4 {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #fff;
        }}
        
        .piece-title p {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}
        
        .tag-group {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        
        .badge-tag {{
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }}
        
        .badge-sift {{
            background-color: rgba(59, 130, 246, 0.15);
            color: #3b82f6;
        }}
        
        .badge-template {{
            background-color: rgba(139, 92, 246, 0.15);
            color: #8b5cf6;
        }}
        
        .badge-matched {{
            background-color: var(--success-bg);
            color: var(--success);
        }}
        
        .badge-mismatched {{
            background-color: var(--danger-bg);
            color: var(--danger);
        }}
        
        .badge-cfg-error {{
            background-color: var(--warning-bg);
            color: var(--warning);
        }}
        
        /* Card Images comparison styling */
        .image-viewer {{
            display: grid;
            grid-template-columns: 1fr 2fr;
            height: 180px;
            background-color: rgba(0,0,0,0.15);
            border-bottom: 1px solid var(--border-color);
        }}
        
        .viewer-box {{
            position: relative;
            height: 100%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .viewer-box:first-child {{
            border-right: 1px solid var(--border-color);
            background: radial-gradient(circle, #2d3748 0%, #1a202c 100%);
        }}
        
        .viewer-box img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.5s ease;
        }}
        
        .viewer-box:hover img {{
            transform: scale(1.1);
        }}
        
        .img-label {{
            position: absolute;
            bottom: 8px;
            left: 8px;
            background-color: rgba(11, 15, 25, 0.85);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            color: #e5e7eb;
            backdrop-filter: blur(4px);
        }}
        
        /* Card Details */
        .piece-details {{
            padding: 20px;
        }}
        
        .detail-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }}
        
        .detail-row:last-child {{
            margin-bottom: 0;
        }}
        
        .detail-lbl {{
            color: var(--text-muted);
        }}
        
        .detail-val {{
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-area">
                <h1>拼圖定位演算法測試報告</h1>
                <p>專案設定：{rows} 行 × {cols} 列 (共 {total_pieces} 片)</p>
            </div>
            <div class="badge-version">版號：{version}</div>
        </header>
        
        <!-- Dashboard Stats -->
        <div class="dashboard-grid">
            <div class="stat-card success-card">
                <div class="stat-info">
                    <h3>真實數據通過率</h3>
                    <div class="stat-value">{pass_rate_real:.1f}%</div>
                </div>
                <div class="progress-ring">
                    <div class="progress-ring-circle">
                        <div class="progress-ring-inner">{matched_real}/{total_real}</div>
                    </div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-info">
                    <h3>單元測試通過</h3>
                    <div class="stat-value">{unit_passed} / {unit_total}</div>
                </div>
                <div style="font-size: 2.5rem; color: var(--primary);">✓</div>
            </div>
            
            <div class="stat-card warning-card">
                <div class="stat-info">
                    <h3>定位錯誤數</h3>
                    <div class="stat-value">{total_real - matched_real}</div>
                </div>
                <div style="font-size: 2.5rem; color: var(--danger);">✗</div>
            </div>
        </div>
        
        <!-- Unit Tests Section -->
        <section>
            <h2>單元與整合測試狀態 (模擬合成)</h2>
            <div class="unit-tests-list">
        """
        
    for name, r in unit_results.items():
        desc = "驗證 SIFT 在旋轉、縮放與透視變形下的預測框 IoU 與角度推導" if "sift" in name else "驗證無紋理純色區域的 HSV 直方圖與多尺度模板匹配退路位置"
        badge_class = "status-passed" if r["status"] == "PASSED" else "status-failed"
        html_content += f"""
                <div class="unit-test-item">
                    <div>
                        <div class="test-name">{name}</div>
                        <div class="test-desc">{desc} - <span style="color: var(--text-main); font-style: italic;">{r['message']}</span></div>
                    </div>
                    <span class="status-badge {badge_class}">{r['status']}</span>
                </div>
        """
        
    html_content += """
            </div>
        </section>
        
        <!-- Real Pieces Validation Section -->
        <section>
            <h2>實拍拼圖碎片定位結果</h2>
            <div class="pieces-grid">
    """
    
    for r in real_results:
        # 標籤樣式
        method_badge = "badge-sift" if r["method"] == "feature" else "badge-template"
        method_name = "SIFT 特徵匹配" if r["method"] == "feature" else "直方圖+模板匹配"
        
        if r["grid_status"] == "MATCHED":
            status_badge = "badge-matched"
            status_text = "定位符合"
        elif r["grid_status"] == "MISMATCHED":
            status_badge = "badge-mismatched"
            status_text = "定位偏差 ❌"
        elif r["grid_status"] == "CONFIG_ERROR":
            status_badge = "badge-cfg-error"
            status_text = "行列設定衝突"
        else:
            status_badge = "badge-mismatched"
            status_text = "定位失敗"
            
        gt_str = f"R{r['gt_pos'][0]} C{r['gt_pos'][1]}" if r["gt_pos"] else "無"
        pred_str = f"R{r['pred_pos'][0]} C{r['pred_pos'][1]}" if r["pred_pos"] else "失敗"
        rot_str = f"{r['rotation_deg']:.1f}°" if r["rotation_deg"] is not None else "無"
        sugg_rot_str = f"{r['suggested_rotation']}°" if r["suggested_rotation"] is not None else "無"
        
        html_content += f"""
                <!-- Piece Card -->
                <div class="piece-card">
                    <div class="piece-header">
                        <div class="piece-title">
                            <h4>{r['filename']}</h4>
                            <p>去背狀態: {'已去背' if r['has_segmented'] else '無明顯邊緣 (原圖比對)'}</p>
                            <div class="tag-group">
                                <span class="badge-tag {method_badge}">{method_name}</span>
                                <span class="badge-tag {status_badge}">{status_text}</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Images -->
                    <div class="image-viewer">
                        <div class="viewer-box">
                            <img src="{r['clean_img']}" alt="乾淨單片">
                            <span class="img-label">去背單片</span>
                        </div>
                        <div class="viewer-box">
                            <img src="{r['located_img']}" alt="定位落點">
                            <span class="img-label">完成圖標註落點</span>
                        </div>
                    </div>
                    
                    <!-- Details -->
                    <div class="piece-details">
                        <div class="detail-row">
                            <span class="detail-lbl">檔名行列真值 (GT)</span>
                            <span class="detail-val">{gt_str}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-lbl">預估網格位置 (Pred)</span>
                            <span class="detail-val" style="color: { 'var(--success)' if r['grid_status'] == 'MATCHED' else 'var(--danger)' }">{pred_str}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-lbl">網格重合度 (IoU)</span>
                            <span class="detail-val">{r['iou']:.3f}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-lbl">推估旋轉角度</span>
                            <span class="detail-val">{rot_str} (建議: {sugg_rot_str})</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-lbl">演算法信心度</span>
                            <span class="detail-val">{r['confidence']:.3f}</span>
                        </div>
                    </div>
                </div>
        """
        
    html_content += """
            </div>
        </section>
    </div>
</body>
</html>
    """
    
    report_file = data_dir / f"report_{version}.html"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n[成功] 測試報告已成功輸出至: {report_file.absolute()}")
    except Exception as e:
        print(f"錯誤: 無法寫入 HTML 報告，錯誤: {e}")
        
    print("\n================ 測試結束 ================")

if __name__ == "__main__":
    main()
