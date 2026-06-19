#!/usr/bin/env python3
"""L2 快取式離線評估器：粗掃只跑一次存快取，迭代評分變體用快取秒出排名。

build  : 對 eval_native 各片做 segment+正規化+粗掃，存 top-K 候選格(含姿態)與 nb/na 快取。
         （KNOWN_HARD_STEMS 為本腳本獨立的「待救片」分類，與 test_localization.KNOWN_HARD 用途不同）
eval   : 讀快取，對 top-K 候選用「全解析度評分變體」重排，印每片 GT 排名 + 是否 ±1 命中。

用法：
  python3 scripts/l2_harness.py build
  python3 scripts/l2_harness.py eval <variant>   # variant: coarse|zncchi|gradhi|fused
"""
import json, re, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.features.localization import locator as L
from source.features.segmentation.detector import segment_pieces, extract_piece_images

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
CACHE = Path("/tmp/l2cache"); CACHE.mkdir(exist_ok=True)
cfg = json.loads((DATA / "project_config.json").read_text(encoding="utf-8"))
ROWS, COLS = cfg["rows"], cfg["cols"]
ref = cv2.imread(str(DATA / "reference_puzzle.jpg")); ref_h, ref_w = ref.shape[:2]
gw, gh = ref_w / COLS, ref_h / ROWS; L_grid = max(gw, gh)
TOPK = 80
KNOWN_HARD_STEMS = {"pieces_c10_r1","pieces_c10_r40","pieces_c17_r17","pieces_c1_r39",
    "pieces_c1_r40","pieces_c22_r40","pieces_c22_r5","pieces_c22_r8","pieces_c23_r7",
    "pieces_c24_r11","pieces_c25_r7"}  # 待救片；其餘為原應命中基準


def get_bgra(path):
    img = cv2.imread(str(path)); seg = segment_pieces(img); imgs = extract_piece_images(img, seg)
    if imgs: return imgs[int(np.argmax([p.area for p in seg.pieces]))]
    b = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA); b[:, :, 3] = 255; return b


ERODE_PX = int(__import__("os").environ.get("ERODE_PX", "0"))


def normalize(path):
    bgra = get_bgra(path); pb, pa = bgra[:, :, :3], bgra[:, :, 3]
    if ERODE_PX > 0:
        # 內縮遮罩去除實拍綠色背景光暈（污染 ZNCC）
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX*2+1, ERODE_PX*2+1))
        pa = cv2.erode(pa, k)
    bw, bh, _ = L._measure_body(pa); bl = max(bw, bh); sc = L_grid / bl if bl > 1 else 1.0
    nb = cv2.resize(pb, (max(4, int(pb.shape[1]*sc)), max(4, int(pb.shape[0]*sc))),
                    interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    na = cv2.resize(pa, (nb.shape[1], nb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return nb, na


def cases():
    out = []
    for p in sorted((DATA / "eval_native").glob("pieces_c*_r*.jpg")):
        m = re.match(r"pieces_c(\d+)_r(\d+)", p.stem)
        if m: out.append((p, int(m.group(1)), int(m.group(2))))
    return out


def build():
    for p, c, r in cases():
        nb, na = normalize(p)
        acc, pose_map, pose_table, RS = L._global_pose_sweep(ref, nb, na, 1.0, L_grid, ROWS, COLS, gw, gh)
        ghs, gws = gh * RS, gw * RS; cells = []
        for rr in range(1, ROWS + 1):
            for cc in range(1, COLS + 1):
                y0, y1 = int((rr-1)*ghs), int(rr*ghs); x0, x1 = int((cc-1)*gws), int(cc*gws)
                sub = acc[y0:y1, x0:x1]
                if sub.size == 0: continue
                li = np.unravel_index(int(np.argmax(sub)), sub.shape)
                cy, cx = y0+li[0], x0+li[1]; pidx = int(pose_map[cy, cx])
                ang, ds = (pose_table[pidx][0], pose_table[pidx][1]) if pidx >= 0 else (0.0, 1.0)
                cells.append((float(sub.max()), rr, cc, ang, ds))
        cells.sort(key=lambda e: -e[0])
        topk = np.array([[s, rr, cc, ang, ds] for s, rr, cc, ang, ds in cells[:TOPK]], dtype=np.float32)
        allc = np.array([[s, rr, cc, ang, ds] for s, rr, cc, ang, ds in cells], dtype=np.float32)
        np.savez(CACHE / f"{p.stem}.npz", nb=nb, na=na, topk=topk, allcells=allc,
                 RS=np.float32(RS), gt=np.array([r, c]))
        gtrank = next((i+1 for i,(s,rr,cc,a,d) in enumerate(cells) if rr==r and cc==c), -1)
        print(f"cached {p.stem:<20} coarse GTrank={gtrank}")


# ---- 全解析度評分變體 ----
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
ref_gray2 = ref_gray * ref_gray
# 梯度方向場（無極性）：cos2θ, sin2θ 以幅值加權
def _orient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    return (gx*gx - gy*gy)/mag, (2*gx*gy)/mag  # mag*cos2θ, mag*sin2θ
ref_c2, ref_s2 = _orient(ref_gray)
ref_c2sq, ref_s2sq = ref_c2*ref_c2, ref_s2*ref_s2


def score_cell(nb, na, r, c, a0, ds0, mode):
    fg = na > 127; clean = nb.copy(); clean[~fg] = 0
    cx, cy = (c-0.5)*gw, (r-0.5)*gh
    best = -2.0
    for ds in (ds0*0.94, ds0, ds0*1.06):
        pw = max(4, int(nb.shape[1]*ds)); ph = max(4, int(nb.shape[0]*ds))
        t_gray = cv2.cvtColor(cv2.resize(clean, (pw, ph)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        m0 = cv2.resize(na, (pw, ph), interpolation=cv2.INTER_NEAREST)
        for a in np.arange(a0-12, a0+12+1e-6, 2.0):
            rt, m, rw, rh = L._rotate_template(t_gray, m0, pw, ph, float(a))
            if m.sum() < 50: continue
            x0 = int(max(0, cx-rw)); y0 = int(max(0, cy-rh))
            x1 = int(min(ref_w, cx+rw)); y1 = int(min(ref_h, cy+rh))
            if x1-x0 <= rw or y1-y0 <= rh: continue
            s = -2.0
            if mode in ("zncchi", "fused"):
                z = L._masked_zncc_map(ref_gray[y0:y1, x0:x1], ref_gray2[y0:y1, x0:x1], rt, m)
                if z is not None: s = float(z.max()); zmax_pos = z
            if mode in ("gradhi", "fused"):
                tc2, ts2 = _orient(rt)
                zc = L._masked_zncc_map(ref_c2[y0:y1, x0:x1], ref_c2sq[y0:y1, x0:x1], tc2, m)
                zs = L._masked_zncc_map(ref_s2[y0:y1, x0:x1], ref_s2sq[y0:y1, x0:x1], ts2, m)
                if zc is not None and zs is not None:
                    g = (zc + zs) / 2.0
                    if mode == "gradhi":
                        s = float(g.max())
                    else:  # fused: 對齊後逐像素取 0.5*zncc+0.5*grad 再 max
                        zz = L._masked_zncc_map(ref_gray[y0:y1, x0:x1], ref_gray2[y0:y1, x0:x1], rt, m)
                        if zz is not None and zz.shape == g.shape:
                            s = float((0.5*zz + 0.5*g).max())
            best = max(best, s)
    return best


def _gt_type(r, c):
    edge_r = r in (1, ROWS); edge_c = c in (1, COLS)
    if edge_r and edge_c: return "corner"
    if edge_r or edge_c: return "edge"
    return "interior"


def _allowed(r, c, typ):
    """完美邊偵測下，該 GT 型態允許的候選格集合判定。"""
    on_ring = (r in (1, ROWS) or c in (1, COLS))
    on_corner = (r in (1, ROWS) and c in (1, COLS))
    if typ == "corner": return on_corner
    if typ == "edge": return on_ring
    return True


def evaluate_edgegt():
    """L1 天花板：用 GT 邊/角型態限制候選集，純粗掃分數排名。"""
    print(f"{'piece':<20}{'set':<7}{'GTtype':<9}{'全格rank':>8}{'限制後rank':>10}{'hit±1':>7}")
    n_hit = 0; n = 0; base_hit = 0; base_n = 0
    for f in sorted(CACHE.glob("pieces_c*.npz")):
        d = np.load(f, allow_pickle=True); allc = d["allcells"]; gt = d["gt"]; stem = f.stem
        gr, gc = int(gt[0]), int(gt[1]); typ = _gt_type(gr, gc)
        rows_all = sorted(range(len(allc)), key=lambda i: -allc[i][0])
        full_rank = next((i+1 for i,i2 in enumerate(rows_all) if int(allc[i2][1])==gr and int(allc[i2][2])==gc), 999)
        keep = [i for i in range(len(allc)) if _allowed(int(allc[i][1]), int(allc[i][2]), typ)]
        keep.sort(key=lambda i: -allc[i][0])
        cons_rank = next((j+1 for j,i in enumerate(keep) if int(allc[i][1])==gr and int(allc[i][2])==gc), 999)
        top1 = (int(allc[keep[0]][1]), int(allc[keep[0]][2])) if keep else (0,0)
        hit = abs(top1[0]-gr) <= 1 and abs(top1[1]-gc) <= 1
        is_base = stem not in KNOWN_HARD_STEMS
        n += 1; n_hit += hit
        if is_base: base_n += 1; base_hit += hit
        print(f"{stem:<20}{'BASE' if is_base else 'rescue':<7}{typ:<9}{full_rank:>8}{cons_rank:>10}{'✓' if hit else '✗':>6}")
    print("-"*62)
    print(f"L1天花板(完美邊偵測): 總命中(±1) {n_hit}/{n} = {100*n_hit/n:.0f}%   基準 {base_hit}/{base_n}")


def evaluate(variant):
    if variant == "edgegt":
        return evaluate_edgegt()
    print(f"{'piece':<20}{'set':<7}{'coarse':>7}{'->new':>7}{'hit±1':>7}{'GTscore':>9}{'Top1cell':>10}")
    n_hit = n_base_hit = 0; n = 0; base_n = 0
    files = sorted(CACHE.glob("pieces_c*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=True); nb, na = d["nb"], d["na"]; topk = d["topk"]; gt = d["gt"]
        gr, gc = int(gt[0]), int(gt[1]); stem = f.stem
        coarse_rank = next((i+1 for i,row in enumerate(topk) if int(row[1])==gr and int(row[2])==gc), 999)
        if variant == "coarse":
            order = sorted(range(len(topk)), key=lambda i: -topk[i][0])
        else:
            sc = [score_cell(nb, na, int(topk[i][1]), int(topk[i][2]), float(topk[i][3]), float(topk[i][4]), variant)
                  for i in range(len(topk))]
            order = sorted(range(len(topk)), key=lambda i: -sc[i])
        ranked = [(int(topk[i][1]), int(topk[i][2])) for i in order]
        new_rank = next((i+1 for i,(rr,cc) in enumerate(ranked) if rr==gr and cc==gc), 999)
        top1 = ranked[0] if ranked else (0,0)
        hit = abs(top1[0]-gr) <= 1 and abs(top1[1]-gc) <= 1
        is_base = stem not in KNOWN_HARD_STEMS
        n += 1; n_hit += hit
        if is_base: base_n += 1; n_base_hit += hit
        gts = float(topk[next((i for i,row in enumerate(topk) if int(row[1])==gr and int(row[2])==gc), 0)][0])
        print(f"{stem:<20}{'BASE' if is_base else 'rescue':<7}{coarse_rank:>7}{new_rank:>7}"
              f"{'✓' if hit else '✗':>6}{gts:>9.3f}  r{top1[0]}c{top1[1]}")
    print("-"*70)
    print(f"變體 {variant}: 總命中(±1) {n_hit}/{n} = {100*n_hit/n:.0f}%   "
          f"原基準片 {n_base_hit}/{base_n}（須維持 {base_n}/{base_n} 不退步）")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "build": build()
    else: evaluate(sys.argv[2] if len(sys.argv) > 2 else "zncchi")
