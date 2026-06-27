import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

@dataclass
class LocateResult:
    quad: Optional[np.ndarray]          # 完成圖上的四邊形落點 (4x2)，失敗為 None
    bounding_box: Optional[Tuple[int, int, int, int]]  # 外接矩形 (x, y, w, h)，失敗為 None
    rotation_deg: Optional[float]       # 由 homography/模板匹配推得的建議旋轉角
    suggested_rotation: Optional[int]   # 規整至最近的 90 度倍數 (0, 90, 180, 270)
    confidence: float                  # 0~1 信心度
    method: str                        # "feature" | "template"
    candidates: List[Tuple[int, int, int, int, float]]  # 候選框清單 [(x, y, w, h, score), ...]
    grid_pos: Optional[Tuple[int, int]] # (row, col)，1-indexed
    annotated_reference: np.ndarray    # 已畫框 of 完成圖
    # 方案1：Top-K 候選清單，供使用者從前幾名挑選（單片+輔助情境）。
    # 每項 {'grid_pos': (row, col), 'score': float, 'rotation': float}，依分數由高到低。
    top_cells: List[Dict[str, Any]] = field(default_factory=list)
    # 找不到明確單一位置時，建議的「大概搜尋區塊」（候選分散時才有值）：
    # {'row_range': (r0, r1), 'col_range': (c0, c1), 'note': str}；None 代表 grid_pos 可信。
    region_hint: Optional[Dict[str, Any]] = None



_CONFIDENT_AGREE = 5   # 前兩名落在彼此 ±5 格內 → 視為指向同一帶，判為「確定」
_SEARCH_MARGIN = 5     # 不確定時，於 rank1 週圍 ±5 格畫搜尋區塊（與 ±5 合格標準一致）
_SATURATION_BAND = 0.03    # 分數飽和判定：與最高分相差此值內者算「同分」
_SATURATION_PLATEAU = 100  # 同分網格數超過此值 → 視為分數飽和（碎片無鑑別資訊）


def _assess_position(top_cells: List[Dict[str, Any]], rows: int, cols: int, saturated: bool = False):
    """判定定位是否「確定」，並在不確定時回傳搜尋區塊／不可解原因。

    - 分數飽和（純色/低紋理，rank1 不可信）→ 不確定，回傳 reason='saturated'（無 ±5 區塊，
      因 rank1 本身無意義），後續標洋紅並提示「紋理不足、無法可靠定位」。
    - 確定（綠框）：只有單一候選，或前兩名落在彼此 ±5 格內（指向同一帶）。
    - 不確定（洋紅框）：前兩名分散 → 回傳 rank1 ±SEARCH_MARGIN 搜尋區塊供人工搜尋；
      只要正解落在此 ±5 範圍即視為合格。
    回傳 (confident: bool, region_hint: Optional[dict])。
    """
    if saturated:
        return False, {'reason': 'saturated',
                       'note': "紋理不足／分數飽和，rank1 不可信，無法可靠定位（建議人工判斷或跳過）"}
    cells = [c['grid_pos'] for c in top_cells if c.get('grid_pos')]
    if len(cells) <= 1:
        return True, None
    (r1, c1), (r2, c2) = cells[0], cells[1]
    if abs(r1 - r2) <= _CONFIDENT_AGREE and abs(c1 - c2) <= _CONFIDENT_AGREE:
        return True, None  # 前兩名指向同一帶 → 確定
    r0, rmax = max(1, r1 - _SEARCH_MARGIN), min(rows, r1 + _SEARCH_MARGIN)
    c0, cmax = max(1, c1 - _SEARCH_MARGIN), min(cols, c1 + _SEARCH_MARGIN)
    zone = {
        'row_range': (r0, rmax),
        'col_range': (c0, cmax),
        'note': f"未確定單一位置；建議在第1名週圍 列{r0}~{rmax}、行{c0}~{cmax}（±{_SEARCH_MARGIN}格）內搜尋",
    }
    return False, zone

def _draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1, dash_length: int = 8):
    """繪製虛線段的輔助函數"""
    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    pts_count = int(dist / dash_length)
    if pts_count == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return

    for i in range(pts_count):
        start_t = i / pts_count
        end_t = (i + 0.5) / pts_count
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * start_t), int(pt1[1] + (pt2[1] - pt1[1]) * start_t))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * end_t), int(pt1[1] + (pt2[1] - pt1[1]) * end_t))
        cv2.line(img, p1, p2, color, thickness)

def _draw_dashed_rectangle(img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 1, dash_length: int = 8):
    """繪製虛線矩形的輔助函數"""
    x, y, w, h = bbox
    pt1 = (x, y)
    pt2 = (x + w, y)
    pt3 = (x + w, y + h)
    pt4 = (x, y + h)
    _draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)
    _draw_dashed_line(img, pt2, pt3, color, thickness, dash_length)
    _draw_dashed_line(img, pt3, pt4, color, thickness, dash_length)
    _draw_dashed_line(img, pt4, pt1, color, thickness, dash_length)

def _get_grid_position(center_x: float, center_y: float, ref_w: int, ref_h: int, rows: int, cols: int) -> Tuple[int, int]:
    """計算給定中心點落在第幾行第幾列 (1-indexed)"""
    gw = ref_w / cols
    gh = ref_h / rows
    col = int(center_x // gw) + 1
    row = int(center_y // gh) + 1
    # 限制邊界
    col = max(1, min(cols, col))
    row = max(1, min(rows, row))
    return row, col

def _standardize_rotated_rect(rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    將 cv2.minAreaRect 的輸出標準化：
    確保 width 永遠是長邊，height 永遠是短邊，並計算將長邊旋轉至「水平」所需的角度 (aligned_angle)。
    """
    (cx, cy), (w, h), angle = rect

    # 確保 w 是長邊，h 是短邊
    if w < h:
        w, h = h, w
        # 當寬高互換時，角度需要進行跳變補償
        aligned_angle = angle + 90.0 if angle < 0 else angle - 90.0
    else:
        aligned_angle = angle

    # 將角度限制在 [-45, 45] 度之間，方便後續直角旋轉處理
    if aligned_angle < -45:
        aligned_angle += 90
    elif aligned_angle > 45:
        aligned_angle -= 90

    return (cx, cy), (w, h), aligned_angle

def _get_puzzle_body_rect(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    透過形態學開運算去除拼圖的凸耳，取得主體輪廓的最小外接矩形。
    """
    # 尋找原始輪廓以估算大小
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    _, (w, h), _ = rect

    # 動態設定結構元素大小（以短邊的 18% 作為開運算半徑，可有效切除凸耳）
    ksize = int(min(w, h) * 0.18)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(5, ksize) # 確保 kernel 夠大

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 重新尋找主體輪廓
    contours_body, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_body:
        return rect # Fallback 到原始矩形

    body_contour = max(contours_body, key=cv2.contourArea)
    return cv2.minAreaRect(body_contour)

def _measure_body(alpha: np.ndarray) -> Tuple[float, float, float]:
    """
    量測碎片主體 (削去凸耳) 的長寬與對齊角。
    流程：_get_puzzle_body_rect 取得初始矩形角度 → _standardize_rotated_rect 標準化 →
    旋轉至對齊後以「投影中位數」量測主體寬高。
    中位數對凸耳穩健：凸耳僅佔少數行/列，不會拉大中位數量測值；
    開運算量測在實拍凸耳較寬時會殘留，導致主體尺寸高估 (實測誤差可達 18%+)。
    """
    body_rect = _get_puzzle_body_rect(alpha)
    if body_rect is None:
        h, w = alpha.shape[:2]
        return float(w), float(h), 0.0
    _, _, aligned_angle = _standardize_rotated_rect(body_rect)

    h, w = alpha.shape[:2]
    diag = int(np.ceil(np.sqrt(h * h + w * w)))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), aligned_angle, 1.0)
    M[0, 2] += diag / 2.0 - w / 2.0
    M[1, 2] += diag / 2.0 - h / 2.0
    rot = cv2.warpAffine(alpha, M, (diag, diag), flags=cv2.INTER_NEAREST)

    bin_rot = rot > 127
    row_any = np.where(bin_rot.any(axis=1))[0]
    col_any = np.where(bin_rot.any(axis=0))[0]
    if len(row_any) == 0 or len(col_any) == 0:
        return float(w), float(h), aligned_angle

    widths = []
    for y in row_any:
        xs = np.where(bin_rot[y])[0]
        if len(xs) > 10:
            widths.append(xs[-1] - xs[0] + 1)
    heights = []
    for x in col_any:
        ys = np.where(bin_rot[:, x])[0]
        if len(ys) > 10:
            heights.append(ys[-1] - ys[0] + 1)

    if not widths or not heights:
        return float(w), float(h), aligned_angle
    return float(np.median(widths)), float(np.median(heights)), aligned_angle

def _masked_zncc_map(f: np.ndarray, f2: np.ndarray, t: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """
    以 FFT 分解計算「帶遮罩 ZNCC」(零均值正規化交叉相關) 的全圖滑動分數圖。
    對每個視窗皆做去均值與變異數正規化，因此對亮度/對比變化不敏感，
    解決 TM_CCORR_NORMED 的亮度方向偏置 (亮區永遠高分) 問題。
    f: 灰階搜尋影像 (float32), f2: f*f 預先計算, t: 模板灰階 (float32), m: 遮罩 (float32, 0/1)
    """
    n = float(m.sum())
    if n < 50:
        return None
    tm = t * m
    A = cv2.matchTemplate(f, tm, cv2.TM_CCORR)   # Σ m·t·w
    B = cv2.matchTemplate(f, m, cv2.TM_CCORR)    # Σ m·w
    C = cv2.matchTemplate(f2, m, cv2.TM_CCORR)   # Σ m·w²
    st = float(tm.sum())
    st2 = float((tm * t).sum())
    num = A - B * (st / n)
    var_f = np.maximum(C - (B * B) / n, 0.0)
    var_t = max(st2 - st * st / n, 1e-6)
    # FFT 浮點誤差在近零變異視窗可能使分數輕微越界，夾限避免 max 累積偏好虛高分
    return np.clip(num / (np.sqrt(var_f * var_t) + 1e-6), -1.0, 1.0)

def _build_corr_ctx(f: np.ndarray, f2: np.ndarray) -> Dict[str, Any]:
    """預先對搜尋影像做一次 FFT，供整輪姿態掃描重用（階段A加速核心）。

    帶遮罩 ZNCC 的三個相關項 A=corr(f,t·m)、B=corr(f,m)、C=corr(f²,m) 之中，
    影像側 f、f² 在 270 個姿態間完全不變，但 `cv2.matchTemplate` 每次呼叫都重算
    f/f² 的 FFT。此處把 F=DFT(f)、F2=DFT(f²) 預算一次，之後每姿態只需轉換小核
    （t·m 與 m）即可，省下絕大多數重複的影像 FFT（實測階段A ~1.8-2×）。
    """
    H, W = f.shape
    Nh = cv2.getOptimalDFTSize(H)
    Nw = cv2.getOptimalDFTSize(W)
    fp = np.zeros((Nh, Nw), np.float32)
    fp[:H, :W] = f
    f2p = np.zeros((Nh, Nw), np.float32)
    f2p[:H, :W] = f2
    return {
        "H": H, "W": W, "Nh": Nh, "Nw": Nw,
        "F": cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT),
        "F2": cv2.dft(f2p, flags=cv2.DFT_COMPLEX_OUTPUT),
    }


def _xcorr_ctx(ctx: Dict[str, Any], img_spec: np.ndarray, ker_spec: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """以預算頻譜計算「有效區」交叉相關，等價於 cv2.matchTemplate(., ., TM_CCORR)。

    對 (Nh,Nw) 做循環相關：模板零填於左上、影像零填於右下時，有效區
    [0:H-kh+1, 0:W-kw+1] 不會回繞，故與 matchTemplate 的有效輸出逐點相同。
    """
    prod = cv2.mulSpectrums(img_spec, ker_spec, 0, conjB=True)
    corr = cv2.idft(prod, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    return corr[:ctx["H"] - kh + 1, :ctx["W"] - kw + 1]


def _masked_zncc_map_ctx(ctx: Dict[str, Any], t: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """帶遮罩 ZNCC，但影像側 FFT 取自預算的 ctx（與 `_masked_zncc_map` 數學等價）。

    僅用於階段A 全圖掃描（f/f² 固定）。階段B 的小 ROI 因影像每次不同，仍用原
    `_masked_zncc_map`（matchTemplate 路徑），保持位元不變。
    """
    n = float(m.sum())
    if n < 50:
        return None
    assert t.shape == m.shape, "模板與遮罩須同形（_rotate_template 保證）"
    tm = t * m
    kh, kw = m.shape  # 旋轉後模板與遮罩同形
    Nh, Nw = ctx["Nh"], ctx["Nw"]
    tm_p = np.zeros((Nh, Nw), np.float32)
    tm_p[:kh, :kw] = tm
    m_p = np.zeros((Nh, Nw), np.float32)
    m_p[:kh, :kw] = m
    tm_spec = cv2.dft(tm_p, flags=cv2.DFT_COMPLEX_OUTPUT)
    m_spec = cv2.dft(m_p, flags=cv2.DFT_COMPLEX_OUTPUT)
    A = _xcorr_ctx(ctx, ctx["F"], tm_spec, kh, kw)   # Σ m·t·w
    B = _xcorr_ctx(ctx, ctx["F"], m_spec, kh, kw)    # Σ m·w
    C = _xcorr_ctx(ctx, ctx["F2"], m_spec, kh, kw)   # Σ m·w² (B、C 共用 m_spec)
    st = float(tm.sum())
    st2 = float((tm * t).sum())
    num = A - B * (st / n)
    var_f = np.maximum(C - (B * B) / n, 0.0)
    var_t = max(st2 - st * st / n, 1e-6)
    return np.clip(num / (np.sqrt(var_f * var_t) + 1e-6), -1.0, 1.0)


def _rotate_template(t: np.ndarray, m0: np.ndarray, pw: int, ph: int, a: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """將模板與遮罩旋轉 a 度，回傳 (旋轉模板 rt, 二值遮罩 m(float32), rw, rh)。"""
    M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), float(a), 1.0)
    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    rw = int(ph * sin_v + pw * cos_v)
    rh = int(ph * cos_v + pw * sin_v)
    M[0, 2] += rw / 2.0 - pw / 2.0
    M[1, 2] += rh / 2.0 - ph / 2.0
    rm = cv2.warpAffine(m0, M, (rw, rh), flags=cv2.INTER_NEAREST)
    m = (rm > 127).astype(np.float32)
    m = cv2.erode(m, np.ones((3, 3), np.float32))
    rt = cv2.warpAffine(t, M, (rw, rh))
    return rt, m, rw, rh


def _score_pose(f: np.ndarray, f2: Optional[np.ndarray], rt: np.ndarray,
                m: np.ndarray, use_zncc: bool,
                ctx: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """對單一旋轉姿態計算分數圖：高紋理用帶遮罩 ZNCC，低紋理退用彩色帶遮罩 CCORR_NORMED。

    ctx 提供時（階段A，f/f² 固定）走預算 FFT 重用路徑加速；ctx=None 時（階段B 小 ROI）
    走原 matchTemplate 路徑，與既有行為位元相同。
    """
    if use_zncc:
        if ctx is not None:
            return _masked_zncc_map_ctx(ctx, rt, m)
        return _masked_zncc_map(f, f2, rt, m)
    rt = rt.copy()
    rt[m < 0.5] = 0
    z = cv2.matchTemplate(f, rt, cv2.TM_CCORR_NORMED, mask=(m * 255).astype(np.uint8))
    return np.nan_to_num(z, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _global_pose_sweep(
    reference: np.ndarray,
    piece_bgr: np.ndarray,
    piece_alpha: np.ndarray,
    s0: float,
    L_grid: float,
    rows: int,
    cols: int,
    gw: float,
    gh: float,
    coarse_step: float = 12.0,
    angle_step: float = 3.0,
    refine_band: float = 9.0,
    top_n_refine: int = 60,  # 只精修 coarse 分數 top-N 格 (最終僅回傳 top-3，精修全部網格純浪費)；60 係對本專案 40×25=1000 格驗證，更大網格可按比例上調
    scale_steps: Tuple[float, ...] = (0.94, 1.0, 1.06),
) -> Tuple[np.ndarray, np.ndarray, list, float]:
    """
    全圖姿態掃描（粗→精兩階段，取代原本 360 度 3° 全圖全掃）：
      階段 A 粗掃：以 coarse_step(12°) 全 360 度 × 多尺度做全圖帶遮罩匹配，
                   分數以「中心對齊」逐像素 max 累積。此階段為主成本（實測 ~55%），
                   主要靠 JP_GRID_PX=64 降採樣壓低（階段A 76.7s→17.7s）。
                   （實測角度步進放寬至 18°/尺度砍至 1 檔會讓近平手案例排名翻轉、
                   eval_native 掉片，故保留 12°×3 尺度。）
      階段 B 精修：僅對 coarse 分數 top_n_refine(預設 60) 格做精修——最終只回傳 top-3，
                   精修全部網格純浪費（本專案 40×25=1000 格實測 ~63s@128 / ~16s@64 vs
                   top-60 ~2.5s；函式預設網格 15×15=225 格則對應更少）。
                   未精修的格仍保有粗掃分數、不影響候選排名，故此裁剪不傷命中率。
                   在各候選中心的小 ROI 內，以 angle_step(3°) 在粗掃最佳角 ±refine_band
                   範圍與多尺度精修，補回精確分數/旋轉角。ROI 卷積成本相對全圖可忽略。
    回傳 (分數累積圖, 姿態索引圖, 姿態表, 降採樣比例 RS)；姿態表項目: (angle, ds, rw, rh)。

    高紋理碎片使用帶遮罩 ZNCC (亮度不變)；
    低紋理 (灰階變異過低、ZNCC 退化) 碎片改用彩色帶遮罩 TM_CCORR_NORMED。
    """
    ref_h, ref_w = reference.shape[:2]
    # 降採樣使網格長邊約 64px。
    # 實證 (output/grid_px_sweep.md) cap 64→128 命中率不變(44%)，128 僅讓已命中片的
    # 落點精確率略升 (eval_native 91%→100%)，代價約 4 倍耗時(階段A 17.7s→76.7s)。
    # 速度優先設 64；需更精落點可由環境變數 JP_GRID_PX=128 覆寫。非法值（空字串/非數字/
    # 非正數）一律回退預設，避免定位中途因環境變數設錯而崩潰或產生退化結果。
    try:
        grid_px = float(os.environ.get("JP_GRID_PX", "64.0"))
        if not (grid_px > 0):
            raise ValueError
    except ValueError:
        grid_px = 64.0
    RS = min(1.0, grid_px / max(L_grid, 1e-6))
    ref_s = cv2.resize(reference, (max(8, int(ref_w * RS)), max(8, int(ref_h * RS))), interpolation=cv2.INTER_AREA) if RS < 1.0 else reference

    gray_piece = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
    fg = piece_alpha > 127
    texture_std = float(gray_piece[fg].std()) if fg.any() else 0.0
    use_zncc = texture_std >= 10.0

    if use_zncc:
        f = cv2.cvtColor(ref_s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f2 = f * f
        corr_ctx = _build_corr_ctx(f, f2)  # 階段A 全圖掃描重用的預算 FFT
    else:
        f = ref_s  # 彩色 CCORR_NORMED 模式
        f2 = None
        corr_ctx = None

    RH, RW = ref_s.shape[:2]
    acc = np.full((RH, RW), -2.0, np.float32)
    pose_map = np.full((RH, RW), -1, np.int32)
    pose_table = []

    clean_bgr = piece_bgr.copy()
    clean_bgr[~fg] = 0

    # 每個尺度預先備妥模板 (灰階供 ZNCC、彩色供 CCORR) 與遮罩，粗掃與精修共用。
    scale_data = []
    for ds in scale_steps:
        s = s0 * RS * ds
        pw = max(4, int(piece_bgr.shape[1] * s))
        ph = max(4, int(piece_bgr.shape[0] * s))
        t_bgr = cv2.resize(clean_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        t_gray = cv2.cvtColor(t_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m0 = cv2.resize(piece_alpha, (pw, ph), interpolation=cv2.INTER_NEAREST)
        tmpl = t_gray if use_zncc else t_bgr
        scale_data.append((float(ds), pw, ph, tmpl, m0))

    def _accumulate(z, rw, rh, a, ds):
        pose_idx = len(pose_table)
        pose_table.append((float(a), float(ds), rw, rh))
        zh, zw = z.shape
        oy, ox = rh // 2, rw // 2
        sub = acc[oy:oy + zh, ox:ox + zw]
        sub_pose = pose_map[oy:oy + zh, ox:ox + zw]
        better = z > sub
        sub[better] = z[better]
        sub_pose[better] = pose_idx

    # ── 階段 A：粗角度全圖掃描 ──
    for ds, pw, ph, tmpl, m0 in scale_data:
        for a in np.arange(0.0, 360.0, coarse_step):
            rt, m, rw, rh = _rotate_template(tmpl, m0, pw, ph, float(a))
            if rw >= RW or rh >= RH or m.sum() < 50:
                continue
            z = _score_pose(f, f2, rt, m, use_zncc, ctx=corr_ctx)
            if z is None:
                continue
            _accumulate(z, rw, rh, a, ds)

    # ── 階段 B：top-N 網格候選的局部 ROI 角度/尺度精修 ──
    ghs, gws = gh * RS, gw * RS
    cells = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            y0, y1 = int((r - 1) * ghs), int(r * ghs)
            x0, x1 = int((c - 1) * gws), int(c * gws)
            sub = acc[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            li = np.unravel_index(int(np.argmax(sub)), sub.shape)
            cells.append((float(sub[li]), y0 + li[0], x0 + li[1]))
    cells.sort(key=lambda e: -e[0])

    for _, cy, cx in cells[:top_n_refine]:
        pidx = int(pose_map[cy, cx])
        if pidx < 0:
            continue
        a0 = pose_table[pidx][0]
        for ds, pw, ph, tmpl, m0 in scale_data:
            for a in np.arange(a0 - refine_band, a0 + refine_band + 1e-6, angle_step):
                rt, m, rw, rh = _rotate_template(tmpl, m0, pw, ph, float(a))
                if rw >= RW or rh >= RH or m.sum() < 50:
                    continue
                # 在候選中心 (cx, cy) 周圍開一個小 ROI，容許定位點微幅移動。
                # 精修只在粗掃最佳角附近微調，定位點幾乎不動，視窗取小即可大幅降低 ROI 卷積成本。
                wnd = max(5, int(0.12 * max(rw, rh)))
                x0r = max(0, cx - wnd - rw // 2)
                y0r = max(0, cy - wnd - rh // 2)
                x1r = min(RW, cx + wnd + rw // 2 + 1)
                y1r = min(RH, cy + wnd + rh // 2 + 1)
                if x1r - x0r <= rw or y1r - y0r <= rh:
                    continue
                f_roi = f[y0r:y1r, x0r:x1r]
                f2_roi = f2[y0r:y1r, x0r:x1r] if use_zncc else None
                z = _score_pose(f_roi, f2_roi, rt, m, use_zncc)
                if z is None:
                    continue
                li = np.unravel_index(int(np.argmax(z)), z.shape)
                s2 = float(z[li])
                gx = x0r + li[1] + rw // 2
                gy = y0r + li[0] + rh // 2
                if 0 <= gy < RH and 0 <= gx < RW and s2 > acc[gy, gx]:
                    acc[gy, gx] = s2
                    pose_map[gy, gx] = len(pose_table)
                    pose_table.append((float(a), float(ds), rw, rh))

    return acc, pose_map, pose_table, RS

def locate_piece(
    reference: np.ndarray,
    piece_bgra: np.ndarray,
    rows: Optional[int] = None,
    cols: Optional[int] = None
) -> LocateResult:
    """
    定位碎片在完成大圖中的位置。
    第 1 層：尺度正規化後的全圖降採樣 SIFT 特徵匹配 (高紋理快速路徑)。
    第 2 層：全圖姿態掃描帶遮罩 ZNCC / CCORR 模板匹配 (實拍與低紋理保底)。
    """
    ref_h, ref_w = reference.shape[:2]

    # 預設 rows 與 cols 劃分
    target_rows = rows if rows is not None else 15
    target_cols = cols if cols is not None else 15
    gw = ref_w / target_cols
    gh = ref_h / target_rows
    L_grid = max(gw, gh)

    # 1. 準備單片資料
    piece_bgr = piece_bgra[:, :, :3]
    piece_alpha = piece_bgra[:, :, 3]

    # 1.5 尺度正規化 (CLAUDE.md 規範)：scale_factor = L_grid / L_piece
    # 以「投影中位數」量測主體 (對凸耳穩健)，將碎片縮放至與大圖網格 1:1，
    # 消除實拍單片與大圖網格間 3~4 倍的尺度落差。
    # rows/cols 未提供時以預設 15x15 網格正規化，使模板保底層的尺度假設一致成立
    body_w, body_h, aligned_angle = _measure_body(piece_alpha)
    body_long = max(body_w, body_h)
    piece_norm_scale = 1.0
    if body_long > 1.0:
        piece_norm_scale = L_grid / body_long
        if not (0.05 <= piece_norm_scale <= 20.0):
            piece_norm_scale = 1.0

    if abs(piece_norm_scale - 1.0) > 1e-3:
        norm_w = max(4, int(piece_bgr.shape[1] * piece_norm_scale))
        norm_h = max(4, int(piece_bgr.shape[0] * piece_norm_scale))
        interp = cv2.INTER_AREA if piece_norm_scale < 1.0 else cv2.INTER_CUBIC
        norm_bgr = cv2.resize(piece_bgr, (norm_w, norm_h), interpolation=interp)
        norm_alpha = cv2.resize(piece_alpha, (norm_w, norm_h), interpolation=cv2.INTER_NEAREST)
        print(f"[INTERNAL NORM] 尺度正規化: L_grid={L_grid:.1f}, body={body_w:.0f}x{body_h:.0f}, scale={piece_norm_scale:.3f} -> piece {norm_w}x{norm_h}")
    else:
        norm_bgr = piece_bgr
        norm_alpha = piece_alpha

    # 計算前景點 (以正規化後座標)，以備後續定位投影
    fg_coords = np.argwhere(norm_alpha > 127)
    if len(fg_coords) > 0:
        y_indices, x_indices = fg_coords[:, 0], fg_coords[:, 1]
        px, py = np.min(x_indices), np.min(y_indices)
        pw_, ph_ = np.max(x_indices) - px, np.max(y_indices) - py
        pts = np.float32([[px, py], [px + pw_, py], [px + pw_, py + ph_], [px, py + ph_]]).reshape(-1, 1, 2)
    else:
        ph_, pw_ = norm_bgr.shape[:2]
        pts = np.float32([[0, 0], [pw_ - 1, 0], [pw_ - 1, ph_ - 1], [0, ph_ - 1]]).reshape(-1, 1, 2)

    sift = cv2.SIFT_create()

    # 2. SIFT 第一層：大圖縮小後的特徵匹配
    sift_success = False
    quad = None
    bounding_box = None
    rotation_deg = None
    suggested_rotation = None
    confidence = 0.0
    grid_pos = None

    max_dim = max(ref_h, ref_w)
    if max_dim > 2560:
        ref_scale = 2560.0 / max_dim
        ref_scaled = cv2.resize(reference, (int(ref_w * ref_scale), int(ref_h * ref_scale)))
    else:
        ref_scale = 1.0
        ref_scaled = reference

    kp_piece, des_piece = sift.detectAndCompute(norm_bgr, norm_alpha)
    kp_ref, des_ref = sift.detectAndCompute(ref_scaled, None)

    if des_piece is not None and des_ref is not None and len(kp_piece) >= 4 and len(kp_ref) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_piece, des_ref, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.80 * n.distance:
                    good_matches.append(m)

        print(f"[INTERNAL SIFT] 縮放比例: {ref_scale:.3f}, kp_piece: {len(kp_piece)}, kp_ref: {len(kp_ref)}, good_matches: {len(good_matches)}")

        if len(good_matches) >= 4:
            src_pts = np.float32([kp_piece[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H_scaled, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_scaled is not None:
                inliers_count = int(np.sum(mask))
                inliers_ratio = inliers_count / len(good_matches)
                print(f"[INTERNAL SIFT] H found. inliers: {inliers_count}, ratio: {inliers_ratio:.3f}")

                if inliers_count >= 10 and inliers_ratio >= 0.35:
                    quad_scaled = cv2.perspectiveTransform(pts, H_scaled).reshape(4, 2)
                    quad_candidate = quad_scaled / ref_scale

                    is_convex = cv2.isContourConvex(quad_candidate.astype(np.int32).reshape(-1, 1, 2))
                    quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                    expected_area = gw * gh
                    area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)

                    x_min, y_min = np.min(quad_candidate, axis=0)
                    x_max, y_max = np.max(quad_candidate, axis=0)
                    boundary_ok = (x_min >= -gw and y_min >= -gh and x_max <= ref_w + gw and y_max <= ref_h + gh)

                    if is_convex and area_ok and boundary_ok:
                        sift_success = True
                        confidence = min(1.0, inliers_count / 20.0 * 0.7 + inliers_ratio * 0.3)
                        quad = quad_candidate
                        bx0, by0 = max(0, int(x_min)), max(0, int(y_min))
                        bounding_box = (
                            bx0,
                            by0,
                            min(ref_w - bx0, int(x_max - x_min)),
                            min(ref_h - by0, int(y_max - y_min))
                        )

                        rotation_rad = np.arctan2(H_scaled[1, 0], H_scaled[0, 0])
                        rotation_deg = np.degrees(rotation_rad) % 360
                        suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360

                        if rows is not None and cols is not None:
                            center_x = (x_min + x_max) / 2.0
                            center_y = (y_min + y_max) / 2.0
                            grid_pos = _get_grid_position(center_x, center_y, ref_w, ref_h, rows, cols)
                    else:
                        print(f"[INTERNAL SIFT] Homography 幾何驗證失敗: is_convex={is_convex}, area_ok={area_ok}, boundary_ok={boundary_ok}")

            # 如果 Homography 失敗，嘗試 Affine 變換作為 Fallback
            if not sift_success:
                M, affine_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is not None:
                    inliers_count = int(np.sum(affine_mask))
                    inliers_ratio = inliers_count / len(good_matches)
                    print(f"[INTERNAL SIFT] Affine found. inliers: {inliers_count}, ratio: {inliers_ratio:.3f}")

                    if inliers_count >= 5 and inliers_ratio >= 0.30:
                        quad_scaled = cv2.transform(pts, M).reshape(4, 2)
                        quad_candidate = quad_scaled / ref_scale

                        quad_area = cv2.contourArea(quad_candidate.astype(np.float32))
                        expected_area = gw * gh
                        area_ok = (0.4 * expected_area <= quad_area <= 2.5 * expected_area)

                        x_min, y_min = np.min(quad_candidate, axis=0)
                        x_max, y_max = np.max(quad_candidate, axis=0)
                        boundary_ok = (x_min >= -gw and y_min >= -gh and x_max <= ref_w + gw and y_max <= ref_h + gh)

                        if area_ok and boundary_ok:
                            sift_success = True
                            confidence = min(1.0, inliers_count / 15.0 * 0.7 + inliers_ratio * 0.3)
                            quad = quad_candidate
                            bx0, by0 = max(0, int(x_min)), max(0, int(y_min))
                            bounding_box = (
                                bx0,
                                by0,
                                min(ref_w - bx0, int(x_max - x_min)),
                                min(ref_h - by0, int(y_max - y_min))
                            )
                            rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                            rotation_deg = np.degrees(rotation_rad) % 360
                            suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360

                            if rows is not None and cols is not None:
                                center_x = (x_min + x_max) / 2.0
                                center_y = (y_min + y_max) / 2.0
                                grid_pos = _get_grid_position(center_x, center_y, ref_w, ref_h, rows, cols)

    annotated = reference.copy()

    if sift_success:
        pts_poly = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_poly], True, (0, 255, 0), 3)

        x, y, w, h = bounding_box
        text_y = max(30, y - 10)
        info_text = f"SIFT Match (Rot: {rotation_deg:.1f}deg, Sugg: {suggested_rotation}deg)"
        if grid_pos is not None:
            info_text += f" Grid: R{grid_pos[0]} C{grid_pos[1]}"

        cv2.putText(annotated, info_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        sift_top = ([{'grid_pos': grid_pos, 'score': confidence, 'rotation': rotation_deg}]
                    if grid_pos is not None else [])
        return LocateResult(
            quad=quad,
            bounding_box=bounding_box,
            rotation_deg=rotation_deg,
            suggested_rotation=suggested_rotation,
            confidence=confidence,
            method="feature",
            candidates=[],
            grid_pos=grid_pos,
            annotated_reference=annotated,
            top_cells=sift_top,
            region_hint=None,  # 特徵匹配成功＝精確，無需區塊建議
        )

    # 3. SIFT 失敗，進入全圖姿態掃描帶遮罩模板匹配保底方案。
    # 實測發現：實拍碎片 (光澤/模糊/印刷紋理) 的 SIFT 描述子與大圖完全無法對應，
    # 且 minAreaRect 對齊角在帶凸耳遮罩上誤差可達 ±40 度，故不可只測 4 個直角方向，
    # 必須以全 360 度掃描搭配亮度不變的帶遮罩 ZNCC 評分。
    print("[INTERNAL TEMPLATE] SIFT 匹配失敗，啟動全圖姿態掃描帶遮罩匹配...")

    s0 = 1.0  # norm_bgr 已正規化至網格尺度
    acc, pose_map, pose_table, RS = _global_pose_sweep(
        reference, norm_bgr, norm_alpha, s0, L_grid,
        target_rows, target_cols, gw, gh
    )

    # 以網格為單位取每格最高分，排序產生候選
    ghs, gws = gh * RS, gw * RS
    cell_entries = []
    for r in range(1, target_rows + 1):
        for c in range(1, target_cols + 1):
            y0, y1 = int((r - 1) * ghs), int(r * ghs)
            x0, x1 = int((c - 1) * gws), int(c * gws)
            sub = acc[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            local_idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            score = float(sub[local_idx])
            cy, cx = y0 + local_idx[0], x0 + local_idx[1]
            cell_entries.append((score, (r, c), (cx, cy)))

    cell_entries.sort(key=lambda e: -e[0])

    # 不可解誠實偵測：分數飽和（大量網格分數逼近最高分）代表碎片本身無鑑別資訊
    # （純色/低紋理片在帶遮罩相關下整片飽和 → conf≈1.0 卻定位錯）。此時 rank1 不可信，
    # 後續判為「找不到」標洋紅，避免給出過度自信的綠框假確定。
    plateau_count = 0
    if cell_entries:
        top_score = cell_entries[0][0]
        plateau_count = sum(1 for s, _, _ in cell_entries if s >= top_score - _SATURATION_BAND)
    score_saturated = plateau_count > _SATURATION_PLATEAU

    top_candidates = []
    for score, (r, c), (cx, cy) in cell_entries[:3]:
        pidx = int(pose_map[cy, cx])
        if pidx < 0 or score <= -1.0:
            continue
        angle, ds, rw, rh = pose_table[pidx]
        # 中心對齊累積圖上的 (cx, cy) 即為該姿態的模板中心
        bx = int((cx - rw / 2.0) / RS)
        by = int((cy - rh / 2.0) / RS)
        bw_full = int(rw / RS)
        bh_full = int(rh / RS)
        bx = max(0, bx)
        by = max(0, by)
        top_candidates.append({
            'bbox': (bx, by, min(ref_w - bx, bw_full), min(ref_h - by, bh_full)),
            'score': score,
            'grid_pos': (r, c),
            'rot': angle
        })

    print(f"[INTERNAL TEMPLATE] top_candidates:")
    for idx, cand in enumerate(top_candidates):
        print(f"  - Rank {idx+1}: score={cand['score']:.4f}, bbox={cand['bbox']}, grid={cand['grid_pos']}, rot={cand['rot']:.1f}")

    # 方案1：Top-K 候選清單（供使用者挑選）與「找不到精確位置時的搜尋區塊建議」
    top_cells = [{'grid_pos': c['grid_pos'], 'score': c['score'], 'rotation': c['rot']}
                 for c in top_candidates]
    # 確定→綠框；找不到（不確定）→洋紅框 + rank1 週圍 ±5 搜尋區塊
    confident, region_hint = True, None
    if rows is not None and cols is not None and top_cells:
        confident, region_hint = _assess_position(top_cells, target_rows, target_cols,
                                                  saturated=score_saturated)

    if top_candidates:
        best = top_candidates[0]
        bx, by, bw, bh = best['bbox']
        box_color = (0, 255, 0) if confident else (255, 0, 255)  # 綠=確定 / 洋紅=找不到
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 3)

        # 不確定時的洋紅標示：
        #   分散 → 畫 ±5 搜尋區塊框；分數飽和(紋理不足) → rank1 已是洋紅框，僅加註無法可靠定位
        if region_hint is not None and 'row_range' in region_hint:
            (r0, r1), (c0, c1) = region_hint['row_range'], region_hint['col_range']
            rx0, ry0 = int((c0 - 1) * gw), int((r0 - 1) * gh)
            rx1, ry1 = int(c1 * gw), int(r1 * gh)
            cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (255, 0, 255), 4)
            cv2.putText(annotated, f"Search zone R{r0}-{r1} C{c0}-{c1} (+/-5)",
                        (rx0, max(24, ry0 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        elif region_hint is not None and region_hint.get('reason') == 'saturated':
            cv2.putText(annotated, "Low texture - cannot localize reliably",
                        (bx, max(24, by - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        tag = "Confirmed" if confident else "Uncertain"
        info_text = f"Rank 1 [{tag}] (Score: {best['score']:.2f}, Rot: {best['rot']:.1f}deg)"
        if rows is not None and cols is not None:
            info_text += f" Grid: R{best['grid_pos'][0]} C{best['grid_pos'][1]}"
        cv2.putText(annotated, info_text, (bx, max(30, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        colors = [(0, 255, 255), (0, 165, 255)]
        for idx, cand in enumerate(top_candidates[1:]):
            cx_, cy_, cw_, ch_ = cand['bbox']
            color = colors[min(idx, len(colors) - 1)]
            _draw_dashed_rectangle(annotated, (cx_, cy_, cw_, ch_), color, thickness=2, dash_length=8)

            cand_text = f"Rank {idx+2} ({cand['score']:.2f}, {cand['rot']:.1f}deg)"
            if rows is not None and cols is not None:
                cand_text += f" R{cand['grid_pos'][0]}C{cand['grid_pos'][1]}"
            cv2.putText(annotated, cand_text, (cx_, max(20, cy_ - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        bounding_box = best['bbox']
        bx, by, bw, bh = bounding_box
        quad = np.array([
            [bx, by],
            [bx + bw, by],
            [bx + bw, by + bh],
            [bx, by + bh]
        ], dtype=np.float32)
        rotation_deg = best['rot']
        suggested_rotation = int(round(rotation_deg / 90.0) * 90) % 360
        confidence = max(0.0, min(1.0, best['score']))
        grid_pos = best['grid_pos'] if (rows is not None and cols is not None) else None
        candidates_out = [(c['bbox'][0], c['bbox'][1], c['bbox'][2], c['bbox'][3], c['score']) for c in top_candidates]
    else:
        bounding_box = None
        quad = None
        rotation_deg = None
        suggested_rotation = None
        confidence = 0.0
        grid_pos = None
        candidates_out = []

    return LocateResult(
        quad=quad,
        bounding_box=bounding_box,
        rotation_deg=rotation_deg,
        suggested_rotation=suggested_rotation,
        confidence=confidence,
        method="template",
        candidates=candidates_out,
        grid_pos=grid_pos,
        annotated_reference=annotated,
        top_cells=top_cells,
        region_hint=region_hint,
    )
