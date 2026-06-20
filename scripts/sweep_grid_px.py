#!/usr/bin/env python3
"""解析度上限掃描：量測 JP_GRID_PX（比對時每網格長邊像素上限）對命中率的實際影響。

JP_GRID_PX 控制比對前把原圖降採樣到「每格長邊約 N px」：
  64  → RS≈0.49（現行預設，僅用原圖約 24% 像素）
  96  → RS≈0.73
  128 → RS≈0.98（≈使用原圖完整原生解析度）

命中判定與 eval_dataset.py 一致：grid_pos 與檔名真值 ±1 格內為命中，完全相符為精確。
逐片記錄結果，輸出每檔成功率與「翻轉的片」對照，存成 output/grid_px_sweep.{json,md}。

用法：  python scripts/sweep_grid_px.py            # 預設掃 64/96/128，全資料集
        python scripts/sweep_grid_px.py 64 128    # 自訂上限
"""
import json
import os
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
DATASETS = ["eval_native", "eval_consistent", "eval_unsolvable"]


def load_config():
    cfg = json.loads((DATA_DIR / "project_config.json").read_text(encoding="utf-8"))
    return cfg["rows"], cfg["cols"]


def list_cases(d: Path):
    cases = []
    for p in sorted(d.glob("pieces_c*_r*.jpg")):
        m = re.match(r"pieces_c(\d+)_r(\d+)", p.stem)
        if m:
            cases.append((p, int(m.group(1)), int(m.group(2))))
    return cases


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
    return locate_piece(ref_img, bgra, rows=rows, cols=cols)


def main():
    caps = [int(a) for a in sys.argv[1:]] or [64, 96, 128]
    rows, cols = load_config()
    ref_img = cv2.imread(str(DATA_DIR / "reference_puzzle.jpg"))
    if ref_img is None:
        sys.exit("找不到 data/reference_puzzle.jpg")

    # results[cap][dataset] = list of dict(piece, gt, pred, hit, exact, conf, method, sec)
    results = {}
    for cap in caps:
        os.environ["JP_GRID_PX"] = str(float(cap))
        results[cap] = {}
        print(f"\n{'#'*72}\n# JP_GRID_PX = {cap}\n{'#'*72}")
        for name in DATASETS:
            d = DATA_DIR / name
            if not d.exists():
                continue
            cases = list_cases(d)
            rows_out = []
            for p, gt_col, gt_row in cases:
                t0 = time.time()
                res = eval_one(ref_img, p, rows, cols)
                dt = time.time() - t0
                if res is None or res.grid_pos is None:
                    rows_out.append(dict(piece=p.stem, gt=(gt_row, gt_col), pred=None,
                                         hit=False, exact=False, conf=0.0, method="FAIL", sec=dt))
                    print(f"  ✗ {p.name:<22} 定位失敗 ({dt:.0f}s)")
                    continue
                pr, pc = res.grid_pos
                hit = abs(pr - gt_row) <= 1 and abs(pc - gt_col) <= 1
                exact = (pr == gt_row and pc == gt_col)
                rows_out.append(dict(piece=p.stem, gt=(gt_row, gt_col), pred=(pr, pc),
                                     hit=hit, exact=exact, conf=res.confidence,
                                     method=res.method, sec=dt))
                mark = "✓" if hit else "✗"
                tag = "精確" if exact else ("±1" if hit else "MISS")
                print(f"  {mark} {p.name:<22} 預測 r{pr} c{pc} / 真值 r{gt_row} c{gt_col} "
                      f"[{tag}] ({dt:.0f}s)")
            results[cap][name] = rows_out
            n = len(rows_out)
            hit = sum(r["hit"] for r in rows_out)
            ex = sum(r["exact"] for r in rows_out)
            print(f"  ── {name}: 命中 {hit}/{n}={100*hit/n:.0f}%  精確 {ex}/{n}={100*ex/n:.0f}%")

    # ---- 匯總表 ----
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "grid_px_sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# 解析度上限 (JP_GRID_PX) 掃描結果\n",
             f"原圖原生網格長邊 ≈131px。cap=64 僅用約 24% 像素；cap=128 ≈ 完整原生解析度。\n",
             "## 各資料集命中率（±1 格）\n",
             "| 資料集 | " + " | ".join(f"cap={c}" for c in caps) + " |",
             "|---|" + "---|" * len(caps)]
    for name in DATASETS:
        cells = []
        for c in caps:
            rs = results[c].get(name, [])
            if not rs:
                cells.append("—"); continue
            n = len(rs); h = sum(r["hit"] for r in rs)
            cells.append(f"{h}/{n} ({100*h/n:.0f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # 總計
    cells = []
    for c in caps:
        allr = [r for name in DATASETS for r in results[c].get(name, [])]
        n = len(allr); h = sum(r["hit"] for r in allr)
        cells.append(f"{h}/{n} ({100*h/n:.0f}%)")
    lines.append(f"| **總計** | " + " | ".join(cells) + " |")

    # 精確率
    lines += ["\n## 各資料集精確率（完全相符）\n",
              "| 資料集 | " + " | ".join(f"cap={c}" for c in caps) + " |",
              "|---|" + "---|" * len(caps)]
    for name in DATASETS:
        cells = []
        for c in caps:
            rs = results[c].get(name, [])
            if not rs:
                cells.append("—"); continue
            n = len(rs); e = sum(r["exact"] for r in rs)
            cells.append(f"{e}/{n} ({100*e/n:.0f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # 平均耗時
    lines += ["\n## 平均單片耗時（秒）\n",
              "| cap | 平均秒 |", "|---|---|"]
    for c in caps:
        allr = [r for name in DATASETS for r in results[c].get(name, [])]
        avg = np.mean([r["sec"] for r in allr]) if allr else 0
        lines.append(f"| {c} | {avg:.1f} |")

    # 翻轉的片（相對最低 cap）
    base = caps[0]
    lines += [f"\n## 相對 cap={base} 的命中翻轉片\n"]
    flips = []
    for name in DATASETS:
        base_map = {r["piece"]: r for r in results[base].get(name, [])}
        for c in caps[1:]:
            for r in results[c].get(name, []):
                b = base_map.get(r["piece"])
                if b and b["hit"] != r["hit"]:
                    direction = "✅修正" if r["hit"] else "❌退步"
                    flips.append(f"- [{name}] {r['piece']}: cap{base}={'hit' if b['hit'] else 'miss'} → cap{c}={'hit' if r['hit'] else 'miss'} {direction}")
    lines += flips if flips else ["（無任何片在不同 cap 間改變命中結果）"]

    md = "\n".join(lines) + "\n"
    (out_dir / "grid_px_sweep.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"→ 已存 output/grid_px_sweep.md 與 .json")


if __name__ == "__main__":
    main()
