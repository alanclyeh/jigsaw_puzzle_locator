#!/usr/bin/env python3
"""對 data/eval_native 跑多輪定位，驗證全數命中並統計辨識時間（毫秒級）。

命中判定：grid_pos 與檔名真值 (col,row) 誤差 ±1 格內（與 test_localization.py 一致）。
時間量測：僅計 locate_piece 全流程（含 segmentation + 定位），不含影像讀檔 / 編碼。

用法：
    python3 scripts/bench_eval_native.py [輪數預設3] [資料夾預設data/eval_native]
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.features.localization.locator import locate_piece
from source.features.segmentation.detector import segment_pieces, extract_piece_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_config():
    cfg = json.loads((DATA_DIR / "project_config.json").read_text(encoding="utf-8"))
    return cfg["rows"], cfg["cols"]


def collect(d: Path):
    cases = []
    for p in sorted(d.glob("pieces_c*_r*.jpg")):
        m = re.match(r"pieces_c(\d+)_r(\d+)", p.stem)
        if m:
            cases.append((p, int(m.group(1)), int(m.group(2))))  # path, gt_col, gt_row
    return cases


def locate(ref_img, piece_img, rows, cols):
    """量測範圍：segmentation -> extract -> locate_piece。"""
    seg = segment_pieces(piece_img)
    imgs = extract_piece_images(piece_img, seg)
    if imgs:
        idx = int(np.argmax([p.area for p in seg.pieces]))
        bgra = imgs[idx]
    else:
        bgra = cv2.cvtColor(piece_img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
    return locate_piece(ref_img, bgra, rows=rows, cols=cols)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "eval_native"
    rows, cols = load_config()
    ref_img = cv2.imread(str(DATA_DIR / "reference_puzzle.jpg"))
    if ref_img is None:
        sys.exit("找不到 data/reference_puzzle.jpg")

    cases = collect(d)
    print(f"參考大圖載入完成，網格 {rows} 列 × {cols} 行")
    print(f"資料夾 {d.name}：{len(cases)} 片，跑 {rounds} 輪\n")

    # 預讀所有單片，避免每輪重複 IO（時間量測只算定位流程）
    piece_imgs = {p: cv2.imread(str(p)) for p, _, _ in cases}

    # times[piece_stem] = [t_round1, t_round2, ...]
    times = {p.stem: [] for p, _, _ in cases}
    last_pred = {}

    for r in range(1, rounds + 1):
        print(f"{'='*78}\n■ Round {r}/{rounds}\n{'='*78}")
        hit = exact = ok5 = 0
        for p, gt_col, gt_row in cases:
            t0 = time.perf_counter()
            res = locate(ref_img, piece_imgs[p], rows, cols)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            times[p.stem].append(dt)
            if res is None or res.grid_pos is None:
                print(f"  ✗ {p.name:<22} 定位失敗  ({dt:7.0f} ms)")
                continue
            pr, pc = res.grid_pos
            err = max(abs(pr - gt_row), abs(pc - gt_col))
            is_hit = abs(pr - gt_row) <= 1 and abs(pc - gt_col) <= 1
            is_exact = (pr == gt_row and pc == gt_col)
            is_pass = abs(pr - gt_row) <= 5 and abs(pc - gt_col) <= 5  # ±5 合格標準
            hit += is_hit
            exact += is_exact
            ok5 += is_pass
            last_pred[p.stem] = (pr, pc, gt_row, gt_col, is_hit, is_exact, res.confidence, res.method)
            conf_tag = "綠/確定" if getattr(res, "region_hint", None) is None else "洋紅/找不到"
            mark = "✓" if is_pass else "✗"  # 以 ±5 合格為主判準
            tag = "精確" if is_exact else (f"±{err}")
            print(f"  {mark} {p.name:<22} 預測 r{pr:<2} c{pc:<2} / 真值 r{gt_row:<2} c{gt_col:<2}  "
                  f"[{tag:<4}|{'合格' if is_pass else '不合格'}|{conf_tag}] conf={res.confidence:.2f} ({dt:7.0f} ms)")
        n = len(cases)
        print(f"  ── 合格(±5): {ok5}/{n} = {100*ok5/n:.1f}%   命中(±1): {hit}/{n} = {100*hit/n:.1f}%   "
              f"精確: {exact}/{n} = {100*exact/n:.1f}%\n")

    # ── 時間統計總表 ──
    print(f"{'#'*78}\n# 辨識時間統計（{rounds} 輪，單位 ms）— 後續演算法優化基準\n{'#'*78}")
    print(f"{'piece':<22}{'min':>9}{'mean':>9}{'max':>9}   結果")
    all_means = []
    for p, gt_col, gt_row in cases:
        ts = times[p.stem]
        mn, mx, mean = min(ts), max(ts), sum(ts)/len(ts)
        all_means.append(mean)
        info = last_pred.get(p.stem)
        if info:
            pr, pc, gr, gc, is_hit, is_exact, conf, method = info
            tag = "精確" if is_exact else ("±1命中" if is_hit else "MISS")
        else:
            tag = "定位失敗"
        print(f"{p.stem:<22}{mn:>9.0f}{mean:>9.0f}{mx:>9.0f}   {tag}")
    total_mean = sum(all_means)
    print(f"{'-'*78}")
    print(f"{'每片平均':<22}{min(all_means):>9.0f}{total_mean/len(all_means):>9.0f}{max(all_means):>9.0f}")
    print(f"全部 {len(cases)} 片單輪總計平均：{total_mean/1000:.1f} s")


if __name__ == "__main__":
    main()
