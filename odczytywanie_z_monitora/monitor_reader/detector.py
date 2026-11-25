"""Automatic detection and perspective rectification of the white-bezel LCD monitor."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WARP_WIDTH = 440
WARP_HEIGHT = 200


@dataclass
class DetectionResult:
    """Result of locating and rectifying the LCD display in a photo."""

    found: bool
    corners_lcd: np.ndarray | None  # (4, 2) float32 ordered TL, TR, BR, BL in full upright image
    warped_lcd: np.ndarray | None  # (WARP_HEIGHT, WARP_WIDTH, 3) BGR uint8
    crop_preview: np.ndarray | None  # Padded context crop with LCD outline
    score: float
    reason: str


def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners of an elongated horizontal LCD as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    # Sort clockwise in image coordinates (x right, y down)
    cyc = pts[np.argsort(angles)]

    # Identify the two longer opposite edges: (0,1)+(2,3) vs (1,2)+(3,0)
    e01 = np.linalg.norm(cyc[1] - cyc[0]) + np.linalg.norm(cyc[3] - cyc[2])
    e12 = np.linalg.norm(cyc[2] - cyc[1]) + np.linalg.norm(cyc[0] - cyc[3])

    if e01 >= e12:
        pair_a = (cyc[0], cyc[1])
        pair_b = (cyc[3], cyc[2])
    else:
        pair_a = (cyc[1], cyc[2])
        pair_b = (cyc[0], cyc[3])

    # Top edge has smaller average y
    if (pair_a[0][1] + pair_a[1][1]) <= (pair_b[0][1] + pair_b[1][1]):
        top_pair, bot_pair = pair_a, pair_b
    else:
        top_pair, bot_pair = pair_b, pair_a

    tl, tr = (top_pair[0], top_pair[1]) if top_pair[0][0] <= top_pair[1][0] else (top_pair[1], top_pair[0])
    bl, br = (bot_pair[0], bot_pair[1]) if bot_pair[0][0] <= bot_pair[1][0] else (bot_pair[1], bot_pair[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _sample_bezel_sides(
    l_chan: np.ndarray,
    s_chan: np.ndarray,
    box_pts: np.ndarray,
    lcd_l_med: float,
) -> tuple[bool, float, float]:
    """Check that all 4 sides around the LCD candidate look like a bright neutral bezel."""
    h, w = l_chan.shape
    ordered = _order_quad_tl_tr_br_bl(box_pts)
    tl, tr, br, bl = ordered
    width_vec = tr - tl
    height_vec = bl - tl
    lcd_w = float(np.linalg.norm(width_vec))
    lcd_h = float(np.linalg.norm(height_vec))
    if lcd_w < 12 or lcd_h < 6:
        return False, 0.0, 0.0

    # Sample strips outside the LCD on all 4 sides (15% to 40% of short side away from edges)
    pad_inner = 0.14
    pad_outer = 0.38
    side_contrasts: list[float] = []
    side_medians: list[float] = []
    side_sats: list[float] = []

    u_dir = width_vec / lcd_w
    v_dir = height_vec / lcd_h

    # For each side, sample a grid of points inside the bezel strip
    strips = [
        # (u_range, v_range) in normalized LCD coordinates [0..1]
        (np.linspace(0.15, 0.85, 7), np.linspace(-pad_outer * lcd_h / lcd_h, -pad_inner, 3)),  # top
        (np.linspace(0.15, 0.85, 7), np.linspace(1.0 + pad_inner, 1.0 + pad_outer, 3)),  # bottom
        (np.linspace(-pad_outer * (lcd_h / lcd_w), -pad_inner * (lcd_h / lcd_w), 3), np.linspace(0.15, 0.85, 5)),  # left
        (np.linspace(1.0 + pad_inner * (lcd_h / lcd_w), 1.0 + pad_outer * (lcd_h / lcd_w), 3), np.linspace(0.15, 0.85, 5)),  # right
    ]

    for us, vs in strips:
        uu, vv = np.meshgrid(us, vs)
        pts_x = np.round(tl[0] + uu * width_vec[0] + vv * height_vec[0]).astype(int)
        pts_y = np.round(tl[1] + uu * width_vec[1] + vv * height_vec[1]).astype(int)
        valid = (pts_x >= 0) & (pts_x < w) & (pts_y >= 0) & (pts_y < h)
        if valid.sum() < 4:
            return False, 0.0, 0.0
        l_vals = l_chan[pts_y[valid], pts_x[valid]].astype(np.float32)
        s_vals = s_chan[pts_y[valid], pts_x[valid]].astype(np.float32)
        med_l = float(np.median(l_vals))
        med_s = float(np.median(s_vals))
        side_medians.append(med_l)
        side_sats.append(med_s)
        side_contrasts.append(med_l - lcd_l_med)

    min_contrast = min(side_contrasts)
    max_sat = max(side_sats)
    bezel_std = float(np.std(side_medians))
    mean_contrast = float(np.mean(side_contrasts))

    # All 4 bezel sides must be substantially brighter than the LCD and neutral in color
    if min_contrast < 22.0:
        return False, 0.0, 0.0
    if max_sat > 55.0:
        return False, 0.0, 0.0

    # Uniformity bonus: bezel sides should have fairly similar brightness
    uniformity = max(0.0, 1.0 - bezel_std / 45.0)
    return True, mean_contrast, uniformity


def _intersect_lines(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    """Intersect two cv2.fitLine outputs (vx, vy, x0, y0)."""
    vx1, vy1, x1, y1 = [float(v) for v in line1.ravel()]
    vx2, vy2, x2, y2 = [float(v) for v in line2.ravel()]
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-6:
        return None
    rhs_x = x2 - x1
    rhs_y = y2 - y1
    t = (rhs_x * (-vy2) - rhs_y * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def _refine_lcd_corners_fullres(
    img_bgr: np.ndarray,
    coarse_quad: np.ndarray,
) -> np.ndarray:
    """Refine LCD corners to sub-pixel accuracy on a full-resolution upright crop."""
    h_img, w_img = img_bgr.shape[:2]
    ordered = _order_quad_tl_tr_br_bl(coarse_quad)
    lcd_w = float(np.linalg.norm(ordered[1] - ordered[0]))
    lcd_h = float(np.linalg.norm(ordered[3] - ordered[0]))

    pad_x = max(24.0, lcd_w * 0.35)
    pad_y = max(24.0, lcd_h * 0.55)
    x0 = max(0, int(np.floor(ordered[:, 0].min() - pad_x)))
    y0 = max(0, int(np.floor(ordered[:, 1].min() - pad_y)))
    x1 = min(w_img, int(np.ceil(ordered[:, 0].max() + pad_x)))
    y1 = min(h_img, int(np.ceil(ordered[:, 1].max() + pad_y)))

    crop_bgr = img_bgr[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        return ordered

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0.7)

    # Local quad in crop coordinates
    local_quad = ordered - np.array([x0, y0], dtype=np.float32)
    init_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(init_mask, np.round(local_quad).astype(np.int32), 255)

    # Sample interior & surrounding bezel to get a tight bimodal threshold
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(lcd_h * 0.45)), max(5, int(lcd_h * 0.45))))
    ring_mask = cv2.subtract(cv2.dilate(init_mask, kernel_dil), cv2.dilate(init_mask, np.ones((5, 5), np.uint8)))
    lcd_vals = gray[init_mask > 0]
    ring_vals = gray[ring_mask > 0]
    if lcd_vals.size < 20 or ring_vals.size < 20:
        return ordered

    # Threshold halfway between 75th percentile of LCD glass and 25th percentile of bezel
    lcd_hi = float(np.percentile(lcd_vals, 80))
    bezel_lo = float(np.percentile(ring_vals, 25))
    if bezel_lo <= lcd_hi + 8:
        thresh_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = (blur <= thresh_val).astype(np.uint8) * 255
    else:
        thresh_val = 0.5 * (lcd_hi + bezel_lo)
        mask = (blur <= thresh_val).astype(np.uint8) * 255

    # Restrict to dilated initial region to drop distant shadows/frames
    search_roi = cv2.dilate(init_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(lcd_w * 0.25)), max(9, int(lcd_h * 0.35)))))
    mask = cv2.bitwise_and(mask, search_roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return ordered

    target_c = local_quad.mean(axis=0)
    best_lbl = -1
    best_dist = 1e9
    expected_area = lcd_w * lcd_h
    for lbl in range(1, num_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < expected_area * 0.4 or area > expected_area * 1.8:
            continue
        dist = float(np.linalg.norm(centroids[lbl] - target_c))
        if dist < best_dist:
            best_dist = dist
            best_lbl = lbl

    if best_lbl < 0:
        return ordered

    comp_mask = (labels == best_lbl).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ordered
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        return ordered

    # Project boundary points into normalized local (u, v) coordinates of initial rect
    rect = cv2.minAreaRect(contour)
    rect_quad = _order_quad_tl_tr_br_bl(cv2.boxPoints(rect))
    tl, tr, br, bl = rect_quad
    u_vec = tr - tl
    v_vec = bl - tl
    u_len2 = float(np.dot(u_vec, u_vec))
    v_len2 = float(np.dot(v_vec, v_vec))
    if u_len2 < 16 or v_len2 < 16:
        return ordered

    rel = pts - tl
    u = (rel @ u_vec) / u_len2
    v = (rel @ v_vec) / v_len2

    # Partition edge pixels, trimming 14% near corners to avoid rounding bias
    top_pts = pts[(v < 0.16) & (u > 0.14) & (u < 0.86)]
    bot_pts = pts[(v > 0.84) & (u > 0.14) & (u < 0.86)]
    left_pts = pts[(u < 0.16) & (v > 0.16) & (v < 0.84)]
    right_pts = pts[(u > 0.84) & (v > 0.16) & (v < 0.84)]

    if min(len(top_pts), len(bot_pts), len(left_pts), len(right_pts)) < 6:
        refined_local = rect_quad
    else:
        l_top = cv2.fitLine(top_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        l_bot = cv2.fitLine(bot_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        l_left = cv2.fitLine(left_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        l_right = cv2.fitLine(right_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)

        c_tl = _intersect_lines(l_top, l_left)
        c_tr = _intersect_lines(l_top, l_right)
        c_br = _intersect_lines(l_bot, l_right)
        c_bl = _intersect_lines(l_bot, l_left)

        if any(c is None for c in (c_tl, c_tr, c_br, c_bl)):
            refined_local = rect_quad
        else:
            cand = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.float32)
            # Sanity check: intersections should stay near minAreaRect corners
            max_shift = max(lcd_w, lcd_h) * 0.25
            if np.max(np.linalg.norm(cand - rect_quad, axis=1)) > max_shift:
                refined_local = rect_quad
            else:
                refined_local = cand

    refined_full = refined_local + np.array([x0, y0], dtype=np.float32)
    return _order_quad_tl_tr_br_bl(refined_full)


def _build_crop_preview(img_bgr: np.ndarray, corners_lcd: np.ndarray) -> np.ndarray:
    """Create a padded context crop around the clock with the detected LCD polygon."""
    h, w = img_bgr.shape[:2]
    lcd_w = float(np.linalg.norm(corners_lcd[1] - corners_lcd[0]))
    lcd_h = float(np.linalg.norm(corners_lcd[3] - corners_lcd[0]))
    pad_x = max(40, int(lcd_w * 0.85))
    pad_y = max(40, int(lcd_h * 1.15))

    x0 = max(0, int(np.floor(corners_lcd[:, 0].min() - pad_x)))
    y0 = max(0, int(np.floor(corners_lcd[:, 1].min() - pad_y)))
    x1 = min(w, int(np.ceil(corners_lcd[:, 0].max() + pad_x)))
    y1 = min(h, int(np.ceil(corners_lcd[:, 1].max() + pad_y)))

    preview = img_bgr[y0:y1, x0:x1].copy()
    poly = np.round(corners_lcd - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    cv2.polylines(preview, [poly], isClosed=True, color=(0, 220, 60), thickness=2, lineType=cv2.LINE_AA)
    return preview


def detect_and_rectify_lcd(img_bgr: np.ndarray) -> DetectionResult:
    """Locate the white-bezel LCD monitor and warp its display to 440x200 BGR."""
    if img_bgr is None or img_bgr.size == 0:
        return DetectionResult(False, None, None, None, 0.0, "Pusty obraz")

    h_full, w_full = img_bgr.shape[:2]
    scale = min(1.0, 2000.0 / float(max(h_full, w_full)))
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(round(w_full * scale)), int(round(h_full * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    hs, ws = small.shape[:2]
    img_area = float(hs * ws)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Edge magnitude for internal digit texture scoring
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # Generate binary masks across multiple thresholding regimes
    masks: list[np.ndarray] = []
    for block_size in (25, 45, 75):
        if min(hs, ws) > block_size:
            th = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                8,
            )
            masks.append(th)

    for q in (15, 25, 35):
        val = float(np.percentile(l_chan, q))
        masks.append((l_chan <= val).astype(np.uint8) * 255)

    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu_mask)

    candidates: list[tuple[float, np.ndarray, tuple[float, float]]] = []
    seen_centers: list[tuple[float, float, float]] = []
    kernel3 = np.ones((3, 3), np.uint8)

    for raw_mask in masks:
        clean = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel3)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel3)
        contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            rel_area = area / img_area
            if rel_area < 0.00015 or rel_area > 0.25:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), _ = rect
            if rw < 10 or rh < 10:
                continue
            long_side = max(rw, rh)
            short_side = min(rw, rh)
            aspect = long_side / short_side
            if aspect < 1.55 or aspect > 2.95:
                continue

            rect_area = rw * rh
            rectangularity = area / rect_area
            if rectangularity < 0.72:
                continue

            # Deduplicate near-identical candidates
            is_dup = False
            for sx, sy, sscore in seen_centers:
                if abs(cx - sx) < long_side * 0.35 and abs(cy - sy) < short_side * 0.45:
                    is_dup = True
                    break

            box = cv2.boxPoints(rect).astype(np.float32)
            # Check interior properties
            x_min = max(0, int(np.floor(box[:, 0].min())))
            y_min = max(0, int(np.floor(box[:, 1].min())))
            x_max = min(ws, int(np.ceil(box[:, 0].max())))
            y_max = min(hs, int(np.ceil(box[:, 1].max())))
            if x_max - x_min < 8 or y_max - y_min < 6:
                continue

            local_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            cv2.fillConvexPoly(local_mask, np.round(box - np.array([x_min, y_min])).astype(np.int32), 255)
            inside = local_mask > 0
            if inside.sum() < 20:
                continue

            lcd_l_med = float(np.median(l_chan[y_min:y_max, x_min:x_max][inside]))
            lcd_s_med = float(np.median(s_chan[y_min:y_max, x_min:x_max][inside]))
            if lcd_s_med > 65.0 or lcd_l_med > 165.0:
                continue

            ok_bezel, mean_contrast, uniformity = _sample_bezel_sides(l_chan, s_chan, box, lcd_l_med)
            if not ok_bezel:
                continue

            # Internal texture energy (exclude outer 12% border)
            eroded = cv2.erode(local_mask, np.ones((3, 3), np.uint8)) > 0
            if eroded.sum() > 10:
                edge_energy = float(np.mean(grad_mag[y_min:y_max, x_min:x_max][eroded]))
            else:
                edge_energy = 0.0

            aspect_score = max(0.0, 1.0 - abs(aspect - 2.15) / 1.0)
            edge_score = min(1.0, edge_energy / 35.0)
            contrast_score = min(1.0, mean_contrast / 95.0)
            score = (
                2.0 * contrast_score
                + 1.3 * uniformity
                + 1.0 * rectangularity
                + 0.8 * aspect_score
                + 0.6 * edge_score
            )

            if not is_dup:
                seen_centers.append((cx, cy, score))
            candidates.append((score, box / scale, (cx / scale, cy / scale)))

    if not candidates:
        return DetectionResult(False, None, None, None, 0.0, "Nie wykryto prostokątnego ekranu LCD w jasnej ramce")

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_coarse_quad, _ = candidates[0]

    corners_lcd = _refine_lcd_corners_fullres(img_bgr, best_coarse_quad)

    # Perspective warp to canonical 440x200
    dst_pts = np.array(
        [
            [0.0, 0.0],
            [float(WARP_WIDTH - 1), 0.0],
            [float(WARP_WIDTH - 1), float(WARP_HEIGHT - 1)],
            [0.0, float(WARP_HEIGHT - 1)],
        ],
        dtype=np.float32,
    )
    mat = cv2.getPerspectiveTransform(corners_lcd, dst_pts)
    warped = cv2.warpPerspective(img_bgr, mat, (WARP_WIDTH, WARP_HEIGHT), flags=cv2.INTER_CUBIC)
    crop_preview = _build_crop_preview(img_bgr, corners_lcd)

    return DetectionResult(
        found=True,
        corners_lcd=corners_lcd,
        warped_lcd=warped,
        crop_preview=crop_preview,
        score=float(best_score),
        reason="OK",
    )
