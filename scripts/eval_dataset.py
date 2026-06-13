#!/usr/bin/env python3
"""對一個目錄內的單片照跑定位，統計命中率（與 test_localization.py 同標準）。

命中判定：grid_pos 與檔名真值 (col,row) 誤差 ±1 格內（與既有測試一致）。
另外回報精確命中（完全相符）數。

用法：
    python scripts/eval_dataset.py data/eval_512 data/eval_other
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


def eval_one(ref_img, piece_path, rows, cols):
    piece_img = cv2.imread(str(piece_path))
    if piece_img is None:
        return None
    seg = segment_pieces(piece_img)
    imgs = extract_piece_images(piece_img, seg)
    if imgs:
        idx = int(np.argmax([p.area for p in seg.pieces]))
        bgra = imgs[idx]
    else:
        bgra = cv2.cvtColor(piece_img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
    res = locate_piece(ref_img, bgra, rows=rows, cols=cols)
    return res


def run_dir(ref_img, d: Path, rows, cols):
    cases = []
    for p in sorted(d.glob("pieces_c*_r*.jpg")):
        m = re.match(r"pieces_c(\d+)_r(\d+)", p.stem)
        if m:
            cases.append((p, int(m.group(1)), int(m.group(2))))  # (path, gt_col, gt_row)

    print(f"\n{'='*72}\n■ {d.name}（{len(cases)} 筆）\n{'='*72}")
    hit = exact = 0
    for p, gt_col, gt_row in cases:
        t0 = time.time()
        res = eval_one(ref_img, p, rows, cols)
        dt = time.time() - t0
        if res is None or res.grid_pos is None:
            print(f"  ✗ {p.name:<22} 定位失敗  ({dt:.0f}s)")
            continue
        pr, pc = res.grid_pos
        derr = max(abs(pr - gt_row), abs(pc - gt_col))
        is_hit = abs(pr - gt_row) <= 1 and abs(pc - gt_col) <= 1
        is_exact = (pr == gt_row and pc == gt_col)
        hit += is_hit
        exact += is_exact
        mark = "✓" if is_hit else "✗"
        tag = "精確" if is_exact else (f"±{derr}" if is_hit else "MISS")
        print(f"  {mark} {p.name:<22} 預測 r{pr:<2} c{pc:<2} / 真值 r{gt_row:<2} c{gt_col:<2}  "
              f"[{tag}] conf={res.confidence:.2f} {res.method} ({dt:.0f}s)")
    n = len(cases)
    rate = 100 * hit / n if n else 0
    erate = 100 * exact / n if n else 0
    print(f"  ── 命中(±1): {hit}/{n} = {rate:.1f}%   精確: {exact}/{n} = {erate:.1f}%")
    return d.name, n, hit, exact, rate, erate


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [DATA_DIR / "eval_512", DATA_DIR / "eval_other"]
    rows, cols = load_config()
    ref_img = cv2.imread(str(DATA_DIR / "reference_puzzle.jpg"))
    if ref_img is None:
        sys.exit("找不到 data/reference_puzzle.jpg")
    print(f"參考大圖載入完成，網格 {rows} 列 × {cols} 行")

    summary = [run_dir(ref_img, d, rows, cols) for d in dirs if d.exists()]

    print(f"\n{'#'*72}\n# 總結（命中標準：±1 格內）\n{'#'*72}")
    print(f"{'分類':<14}{'筆數':>5}{'命中±1':>9}{'命中率':>9}{'精確':>7}{'精確率':>9}")
    for name, n, hit, exact, rate, erate in summary:
        print(f"{name:<14}{n:>5}{hit:>9}{rate:>8.1f}%{exact:>7}{erate:>8.1f}%")
    if len(summary) >= 2:
        best = max(summary, key=lambda s: (s[4], s[5]))
        print(f"\n→ 命中率較高：{best[0]}（{best[4]:.1f}%）")


if __name__ == "__main__":
    main()
