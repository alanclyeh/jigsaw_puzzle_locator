#!/usr/bin/env python3
"""L0 診斷（唯讀，不動 locator，不影響 eval_native）。

對 eval_native + eval_unsolvable 每片跑現行全姿態掃描，輸出分類表：
  - GT 格在 1000 格中的排名、GT 分數、Top1 分數、GT 是否進 Top-5
  - 分數分布：尖峰 vs 平台（Top1 附近 0.03 帶內的格數；越多越平台）
  - 平邊偵測（heuristic）：偵測到的平邊數，並對照檔名真值的邊/角性質驗證準確率
這張表決定 L1(邊形狀)/L2(強化評分)/L-UX(Top-K清單) 各能救幾片。

用法：python3 scripts/diag_l0.py [資料夾...]  (預設 eval_native eval_unsolvable)
"""
import json, re, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.features.localization import locator as L
from source.features.segmentation.detector import segment_pieces, extract_piece_images

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
cfg = json.loads((DATA / "project_config.json").read_text(encoding="utf-8"))
ROWS, COLS = cfg["rows"], cfg["cols"]
ref = cv2.imread(str(DATA / "reference_puzzle.jpg")); ref_h, ref_w = ref.shape[:2]
gw, gh = ref_w / COLS, ref_h / ROWS; L_grid = max(gw, gh)

BAND = 0.03          # Top1 附近此帶內的格數 → 平台指標
FLAT_RATIO = 0.12    # 邊緣輪廓起伏 < 短邊*此比例 → 視為平邊


def get_bgra(path):
    img = cv2.imread(str(path)); seg = segment_pieces(img); imgs = extract_piece_images(img, seg)
    if imgs:
        return imgs[int(np.argmax([p.area for p in seg.pieces]))]
    b = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA); b[:, :, 3] = 255; return b


def count_flat_edges(alpha):
    """偵測平邊數：對齊主體後，量四邊輪廓起伏，起伏很小者視為平邊。回傳 (平邊數, 四邊起伏比例)。"""
    body_w, body_h, ang = L._measure_body(alpha)
    h, w = alpha.shape[:2]
    diag = int(np.ceil(np.sqrt(h * h + w * w)))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    M[0, 2] += diag / 2.0 - w / 2.0; M[1, 2] += diag / 2.0 - h / 2.0
    rot = cv2.warpAffine(alpha, M, (diag, diag), flags=cv2.INTER_NEAREST) > 127
    ys, xs = np.where(rot)
    if len(xs) == 0:
        return 0, (0, 0, 0, 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    short = max(1.0, min(body_w, body_h))
    ratios = []
    # top: 每欄最上緣；bottom: 每欄最下緣（取中央 60% 欄避免角落凸耳干擾）
    cx = np.arange(int(x0 + 0.2 * (x1 - x0)), int(x1 - 0.2 * (x1 - x0)) + 1)
    cy = np.arange(int(y0 + 0.2 * (y1 - y0)), int(y1 - 0.2 * (y1 - y0)) + 1)
    for axis, idx in [("top", cx), ("bottom", cx), ("left", cy), ("right", cy)]:
        prof = []
        for i in idx:
            if axis in ("top", "bottom"):
                col = np.where(rot[:, i])[0]
                if len(col): prof.append(col.min() if axis == "top" else col.max())
            else:
                row = np.where(rot[i, :])[0]
                if len(row): prof.append(row.min() if axis == "left" else row.max())
        if len(prof) < 5:
            ratios.append(9.9); continue
        prof = np.array(prof)
        rng = np.percentile(prof, 90) - np.percentile(prof, 10)  # 抗離群起伏幅度
        ratios.append(rng / short)
    flats = sum(1 for r in ratios if r < FLAT_RATIO)
    return flats, tuple(round(r, 2) for r in ratios)


def diag_piece(path, gt_col, gt_row):
    bgra = get_bgra(path); pb, pa = bgra[:, :, :3], bgra[:, :, 3]
    bw, bh, _ = L._measure_body(pa); bl = max(bw, bh); sc = L_grid / bl if bl > 1 else 1.0
    nb = cv2.resize(pb, (max(4, int(pb.shape[1] * sc)), max(4, int(pb.shape[0] * sc))),
                    interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    na = cv2.resize(pa, (nb.shape[1], nb.shape[0]), interpolation=cv2.INTER_NEAREST)
    tstd = float(cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)[na > 127].std())

    acc, pose_map, pose_table, RS, *_ = L._global_pose_sweep(ref, nb, na, 1.0, L_grid, ROWS, COLS, gw, gh)
    ghs, gws = gh * RS, gw * RS; cells = []
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            y0, y1 = int((r - 1) * ghs), int(r * ghs); x0, x1 = int((c - 1) * gws), int(c * gws)
            sub = acc[y0:y1, x0:x1]
            if sub.size: cells.append((float(sub.max()), r, c))
    cells.sort(key=lambda e: -e[0])
    top1 = cells[0][0]
    gt_rank = next((i + 1 for i, (s, r, c) in enumerate(cells) if r == gt_row and c == gt_col), -1)
    gt_score = next((s for s, r, c in cells if r == gt_row and c == gt_col), 0.0)
    plateau = sum(1 for s, r, c in cells if s >= top1 - BAND)
    flats, ratios = count_flat_edges(pa)
    gt_type = ("CORNER" if (gt_col in (1, COLS) and gt_row in (1, ROWS))
               else ("EDGE" if (gt_col in (1, COLS) or gt_row in (1, ROWS)) else "INTERIOR"))
    det_type = "CORNER" if flats >= 2 else ("EDGE" if flats == 1 else "INTERIOR")
    return dict(name=path.name, gt=(gt_row, gt_col), gt_type=gt_type, det_type=det_type, flats=flats,
                ratios=ratios, rank=gt_rank, gt_score=gt_score, top1=top1, top1_cell=cells[0][1:],
                in_top5=gt_rank <= 5, plateau=plateau, tstd=tstd)


def collect(d):
    out = []
    for p in sorted(d.glob("pieces_c*_r*.jpg")):
        m = re.match(r"pieces_c(\d+)_r(\d+)", p.stem)
        if m: out.append((p, int(m.group(1)), int(m.group(2))))
    return out


def main():
    dirs = sys.argv[1:] or ["eval_native", "eval_unsolvable"]
    for dn in dirs:
        d = DATA / dn; cases = collect(d)
        print(f"\n{'='*120}\n資料夾 {dn}：{len(cases)} 片\n{'='*120}")
        print(f"{'piece':<20}{'GT':<9}{'真值型':<9}{'偵測型':<9}{'平邊':<5}{'rank':>6}{'GTscore':>9}"
              f"{'Top1':>7}{'進T5':>5}{'平台格':>7}{'tex':>6}")
        rows = []
        for p, c, r in cases:
            d_ = diag_piece(p, c, r); rows.append(d_)
            print(f"{d_['name']:<20}r{d_['gt'][0]}c{d_['gt'][1]:<5}{d_['gt_type']:<9}{d_['det_type']:<9}"
                  f"{d_['flats']:<5}{d_['rank']:>6}{d_['gt_score']:>9.3f}{d_['top1']:>7.3f}"
                  f"{'✓' if d_['in_top5'] else '✗':>4}{d_['plateau']:>7}{d_['tstd']:>6.0f}")
        n = len(rows)
        hit1 = sum(1 for x in rows if x['rank'] == 1)
        t5 = sum(1 for x in rows if x['in_top5'])
        flat_acc = sum(1 for x in rows if x['gt_type'] == x['det_type'])
        edge_recall = (sum(1 for x in rows if x['gt_type'] != 'INTERIOR' and x['det_type'] != 'INTERIOR')
                       / max(1, sum(1 for x in rows if x['gt_type'] != 'INTERIOR')))
        print(f"{'-'*120}")
        print(f"  GT rank=1（自動命中）: {hit1}/{n} = {100*hit1/n:.0f}%   "
              f"GT 進 Top-5（L-UX 可救）: {t5}/{n} = {100*t5/n:.0f}%")
        print(f"  平邊型態偵測 vs 真值 完全一致: {flat_acc}/{n}   "
              f"非內部片(邊/角)召回: {100*edge_recall:.0f}%（L1 上線門檻，目標 >90%）")


if __name__ == "__main__":
    main()
