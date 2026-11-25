"""Parametric offline 7-segment and 14-segment LCD decoders using NCC + segment verification."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from monitor_reader.preprocessor import PreprocessedLCD

_ALL_7SEGS = ("a", "b", "c", "d", "e", "f", "g")

# Standard 7-segment mapping: a(top), b(tr), c(br), d(bot), e(bl), f(tl), g(mid)
_SEG7_MAP: dict[str, tuple[str, ...]] = {
    "0": ("a", "b", "c", "d", "e", "f"),
    "1": ("b", "c"),
    "2": ("a", "b", "g", "e", "d"),
    "3": ("a", "b", "g", "c", "d"),
    "4": ("f", "g", "b", "c"),
    "5": ("a", "f", "g", "c", "d"),
    "6": ("a", "f", "e", "d", "c", "g"),
    "7": ("a", "b", "c"),
    "8": ("a", "b", "c", "d", "e", "f", "g"),
    "9": ("a", "b", "c", "d", "f", "g"),
}

# 14-segment mapping matching LCD weekday glyph style
_SEG14_MAP: dict[str, tuple[str, ...]] = {
    "A": ("a", "f", "e", "b", "c", "g1", "g2"),
    "D": ("a", "b", "c", "d", "i", "l"),
    "E": ("a", "f", "e", "d", "g1", "g2"),
    "F": ("a", "f", "e", "g1", "g2"),
    "H": ("f", "e", "b", "c", "g1", "g2"),
    "I": ("a", "d", "i", "l"),
    "M": ("f", "e", "b", "c", "h", "j"),
    "N": ("f", "e", "b", "c", "h", "m"),
    "O": ("a", "b", "c", "d", "e", "f"),
    "R": ("a", "f", "e", "b", "g1", "g2", "m"),
    "S": ("a", "f", "g1", "g2", "c", "d"),
    "T": ("a", "i", "l"),
    "U": ("f", "e", "d", "b", "c"),
    "W": ("f", "e", "b", "c", "d", "l"),
}

WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
DAYS_IN_MONTH = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


@dataclass
class DecodeResult:
    """Decoded parameters, per-field confidences, and ROI boxes for visualization."""

    time_str: str
    date_str: str
    weekday: str
    temperature: str
    conf_time: float
    conf_date: float
    conf_weekday: float
    conf_temperature: float
    confidence: float
    roi_boxes: dict[str, tuple[int, int, int, int]]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load layout and decoder configuration JSON."""
    cfg_path = Path(path) if path else Path(__file__).resolve().parent.parent / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_italic_slant(img: np.ndarray, slant_deg: float) -> np.ndarray:
    """Shear top edges slightly right relative to bottom edges."""
    if abs(slant_deg) < 1e-3:
        return img
    h, w = img.shape
    shear = math.tan(math.radians(slant_deg))
    mat = np.array([[1.0, -shear, shear * h * 0.5], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)


def _draw_segment_line(
    canvas: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    thickness: int,
) -> None:
    cv2.line(
        canvas,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        1.0,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _build_7seg_masks(
    width: int,
    height: int,
    thickness_ratio: float,
    slant_deg: float,
) -> dict[str, np.ndarray]:
    """Build soft boolean masks for each of the 7 segments (a..g)."""
    scale = 2
    w2, h2 = width * scale, height * scale
    t = max(2, int(round(min(w2, h2) * thickness_ratio)))
    mx = max(t // 2 + 1, int(round(w2 * 0.14)))
    my = max(t // 2 + 1, int(round(h2 * 0.09)))
    x0, x1 = float(mx), float(w2 - mx - 1)
    y0, y1 = float(my), float(h2 - my - 1)
    ym = 0.5 * (y0 + y1)
    gap = max(1.5, t * 0.35)

    coords = {
        "a": ((x0 + gap, y0), (x1 - gap, y0)),
        "b": ((x1, y0 + gap), (x1, ym - gap)),
        "c": ((x1, ym + gap), (x1, y1 - gap)),
        "d": ((x0 + gap, y1), (x1 - gap, y1)),
        "e": ((x0, ym + gap), (x0, y1 - gap)),
        "f": ((x0, y0 + gap), (x0, ym - gap)),
        "g": ((x0 + gap, ym), (x1 - gap, ym)),
    }

    masks: dict[str, np.ndarray] = {}
    for seg_name, (p1, p2) in coords.items():
        canvas = np.zeros((h2, w2), dtype=np.float32)
        _draw_segment_line(canvas, p1, p2, max(2, int(round(t * 0.9))))
        canvas = _apply_italic_slant(canvas, slant_deg)
        small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        masks[seg_name] = small > 0.35
    return masks


def render_7seg_template(
    digit: str,
    width: int,
    height: int,
    thickness_ratio: float = 0.20,
    slant_deg: float = 5.0,
    blur_sigma: float = 1.8,
) -> np.ndarray:
    """Render a parametric 7-segment digit template on a float32 [0..1] canvas."""
    scale = 2
    w2, h2 = width * scale, height * scale
    canvas = np.zeros((h2, w2), dtype=np.float32)
    segs = _SEG7_MAP.get(digit, ())
    if not segs:
        return np.zeros((height, width), dtype=np.float32)

    t = max(2, int(round(min(w2, h2) * thickness_ratio)))
    mx = max(t // 2 + 1, int(round(w2 * 0.14)))
    my = max(t // 2 + 1, int(round(h2 * 0.09)))
    x0, x1 = float(mx), float(w2 - mx - 1)
    y0, y1 = float(my), float(h2 - my - 1)
    ym = 0.5 * (y0 + y1)
    gap = max(1.0, t * 0.25)

    coords = {
        "a": ((x0 + gap, y0), (x1 - gap, y0)),
        "b": ((x1, y0 + gap), (x1, ym - gap)),
        "c": ((x1, ym + gap), (x1, y1 - gap)),
        "d": ((x0 + gap, y1), (x1 - gap, y1)),
        "e": ((x0, ym + gap), (x0, y1 - gap)),
        "f": ((x0, y0 + gap), (x0, ym - gap)),
        "g": ((x0 + gap, ym), (x1 - gap, ym)),
    }
    for s in segs:
        p1, p2 = coords[s]
        _draw_segment_line(canvas, p1, p2, t)

    canvas = _apply_italic_slant(canvas, slant_deg)
    small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0.1:
        small = cv2.GaussianBlur(small, (0, 0), blur_sigma)
    mx_val = float(small.max())
    if mx_val > 1e-6:
        small /= mx_val
    return small


def render_14seg_char_template(
    ch: str,
    width: int,
    height: int,
    thickness_ratio: float = 0.19,
    slant_deg: float = 3.0,
    blur_sigma: float = 1.6,
) -> np.ndarray:
    """Render a parametric 14-segment alphanumeric template on a float32 [0..1] canvas."""
    scale = 2
    w2, h2 = width * scale, height * scale
    canvas = np.zeros((h2, w2), dtype=np.float32)
    segs = _SEG14_MAP.get(ch.upper(), ())
    if not segs:
        return np.zeros((height, width), dtype=np.float32)

    t = max(2, int(round(min(w2, h2) * thickness_ratio)))
    mx = max(t // 2 + 1, int(round(w2 * 0.13)))
    my = max(t // 2 + 1, int(round(h2 * 0.09)))
    x0, x1 = float(mx), float(w2 - mx - 1)
    xm = 0.5 * (x0 + x1)
    y0, y1 = float(my), float(h2 - my - 1)
    ym = 0.5 * (y0 + y1)
    gap = max(1.0, t * 0.25)

    coords = {
        "a": ((x0 + gap, y0), (x1 - gap, y0)),
        "b": ((x1, y0 + gap), (x1, ym - gap)),
        "c": ((x1, ym + gap), (x1, y1 - gap)),
        "d": ((x0 + gap, y1), (x1 - gap, y1)),
        "e": ((x0, ym + gap), (x0, y1 - gap)),
        "f": ((x0, y0 + gap), (x0, ym - gap)),
        "g1": ((x0 + gap, ym), (xm - gap, ym)),
        "g2": ((xm + gap, ym), (x1 - gap, ym)),
        "i": ((xm, y0 + gap), (xm, ym - gap)),
        "l": ((xm, ym + gap), (xm, y1 - gap)),
        "h": ((x0 + gap, y0 + gap), (xm - gap, ym - gap)),
        "j": ((x1 - gap, y0 + gap), (xm + gap, ym - gap)),
        "k": ((x0 + gap, y1 - gap), (xm - gap, ym + gap)),
        "m": ((x1 - gap, y1 - gap), (xm + gap, ym + gap)),
    }
    for s in segs:
        p1, p2 = coords[s]
        _draw_segment_line(canvas, p1, p2, t)

    canvas = _apply_italic_slant(canvas, slant_deg)
    small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0.1:
        small = cv2.GaussianBlur(small, (0, 0), blur_sigma)
    mx_val = float(small.max())
    if mx_val > 1e-6:
        small /= mx_val
    return small


def render_weekday_template(
    day_str: str,
    char_w: int,
    char_gap: int,
    height: int,
    thickness_ratio: float = 0.19,
    slant_deg: float = 3.0,
    blur_sigma: float = 1.6,
) -> np.ndarray:
    """Render a 3-letter weekday word template in 14-segment style."""
    total_w = 3 * char_w + 2 * char_gap
    out = np.zeros((height, total_w), dtype=np.float32)
    for idx, ch in enumerate(day_str[:3]):
        x_start = idx * (char_w + char_gap)
        out[:, x_start : x_start + char_w] = render_14seg_char_template(
            ch,
            char_w,
            height,
            thickness_ratio=thickness_ratio,
            slant_deg=slant_deg,
            blur_sigma=blur_sigma,
        )
    return out


def _segment_agreement_score(
    patch: np.ndarray,
    seg_masks: dict[str, np.ndarray],
    digit: str,
) -> float:
    """Score agreement between patch segment intensities and the digit's 7-segment pattern."""
    on_segs = set(_SEG7_MAP[digit])
    bg = float(np.percentile(patch, 15))
    peak = float(np.percentile(patch, 96))
    rng = max(0.18, peak - bg)

    acts: dict[str, float] = {}
    for seg_name, mask in seg_masks.items():
        vals = patch[mask]
        if vals.size == 0:
            acts[seg_name] = 0.0
        else:
            acts[seg_name] = float(np.clip((np.percentile(vals, 80) - bg) / rng, 0.0, 1.0))

    # Mean activity match: ON segments should be bright, OFF segments should be dark
    total = 0.0
    for seg_name in _ALL_7SEGS:
        act = acts[seg_name]
        if seg_name in on_segs:
            total += act
        else:
            # Penalize bright OFF segments strongly
            total += 1.0 - min(1.0, act * 1.25)
    return 2.0 * (total / 7.0) - 1.0


def _score_slot_at_y(
    ink_map: np.ndarray,
    templates: dict[str, np.ndarray],
    seg_masks: dict[str, np.ndarray],
    allowed: tuple[str, ...],
    box_x: int,
    fixed_y: int,
    w: int,
    h: int,
    search_dx: int,
    allow_blank: bool = False,
    blank_ink_p95: float = 0.24,
    min_x: int | None = None,
) -> tuple[dict[str, float], int]:
    """Score allowed digits in a slot at fixed baseline y, searching dx in [-search_dx..+search_dx]."""
    ih, iw = ink_map.shape
    y0 = max(0, min(ih - h, fixed_y))
    want_x0 = box_x - search_dx
    if min_x is not None:
        want_x0 = max(want_x0, min_x)
    want_x1 = box_x + w + search_dx
    if want_x1 < want_x0 + w:
        want_x1 = want_x0 + w
    pad_left = max(0, -want_x0)
    pad_right = max(0, want_x1 - iw)
    x0 = max(0, want_x0)
    x1 = min(iw, want_x1)
    strip = ink_map[y0 : y0 + h, x0:x1]
    if pad_left > 0 or pad_right > 0:
        strip = cv2.copyMakeBorder(strip, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0.0)
    origin_x = want_x0

    scores: dict[str, float] = {}
    slot_x0 = max(0, box_x)
    slot_x1 = min(iw, box_x + w)
    nominal_patch = ink_map[y0 : y0 + h, slot_x0:slot_x1]
    if allow_blank and nominal_patch.size > 0:
        p95 = float(np.percentile(nominal_patch, 95))
        if p95 < blank_ink_p95:
            scores[""] = 0.85
        else:
            scores[""] = max(0.0, 0.30 - 0.5 * (p95 - blank_ink_p95))

    best_x = box_x
    best_s = -2.0
    if strip.shape[0] < h or strip.shape[1] < w:
        for d in allowed:
            if d != "":
                scores[d] = -1.0
        return scores, best_x

    for d in allowed:
        if d == "":
            continue
        res = cv2.matchTemplate(strip.astype(np.float32), templates[d].astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, max_ncc, _, max_loc = cv2.minMaxLoc(res)
        loc_x = int(max_loc[0])
        aligned_patch = strip[:, loc_x : loc_x + w]
        seg_sc = _segment_agreement_score(aligned_patch, seg_masks, d)
        combined = 0.65 * float(max_ncc) + 0.35 * seg_sc
        scores[d] = combined
        if combined > best_s:
            best_s = combined
            best_x = origin_x + loc_x
    return scores, best_x


def _score_row_joint_y(
    ink_map: np.ndarray,
    templates: dict[str, np.ndarray],
    seg_masks: dict[str, np.ndarray],
    slot_specs: list[tuple[int, tuple[str, ...], bool]],
    nominal_y: int,
    w: int,
    h: int,
    search_dx: int,
    search_dy: int,
    blank_ink_p95: float,
) -> tuple[list[dict[str, float]], list[int], int]:
    """Pick the best shared row baseline y across all slots in a field."""
    ih, _ = ink_map.shape
    best_y = nominal_y
    best_row_score = -1e9
    best_all_scores: list[dict[str, float]] = []
    best_xs: list[int] = [spec[0] for spec in slot_specs]

    for dy in range(-search_dy, search_dy + 1):
        cand_y = max(0, min(ih - h, nominal_y + dy))
        row_scores: list[dict[str, float]] = []
        row_xs: list[int] = []
        row_sum = 0.0
        prev_right_min: int | None = None
        for sx, allowed, allow_blank in slot_specs:
            sc, bx = _score_slot_at_y(
                ink_map,
                templates,
                seg_masks,
                allowed,
                sx,
                cand_y,
                w,
                h,
                search_dx,
                allow_blank=allow_blank,
                blank_ink_p95=blank_ink_p95,
                min_x=prev_right_min,
            )
            row_scores.append(sc)
            row_xs.append(bx)
            row_sum += max(sc.values())
            prev_right_min = bx + w - 3
        if row_sum > best_row_score:
            best_row_score = row_sum
            best_y = cand_y
            best_all_scores = row_scores
            best_xs = row_xs

    return best_all_scores, best_xs, best_y


def _ncc_best_shift(
    ink_map: np.ndarray,
    tmpl: np.ndarray,
    box_x: int,
    box_y: int,
    search_dx: int,
    search_dy: int,
) -> tuple[float, int, int]:
    """Compute max zero-mean normalized cross-correlation in a shift window."""
    ih, iw = ink_map.shape
    th, tw = tmpl.shape
    x0 = max(0, box_x - search_dx)
    y0 = max(0, box_y - search_dy)
    x1 = min(iw, box_x + tw + search_dx)
    y1 = min(ih, box_y + th + search_dy)

    patch = ink_map[y0:y1, x0:x1]
    if patch.shape[0] < th or patch.shape[1] < tw:
        return -1.0, box_x, box_y

    res = cv2.matchTemplate(patch.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), x0 + int(max_loc[0]), y0 + int(max_loc[1])


def _score_to_confidence(best_score: float, margin: float) -> float:
    """Map combined score and runner-up margin into a calibrated [0, 1] confidence."""
    s_term = float(np.clip((best_score - 0.15) / 0.65, 0.0, 1.0))
    m_term = float(np.clip(margin / 0.14, 0.0, 1.0))
    return float(np.clip(0.55 * s_term + 0.45 * m_term, 0.0, 1.0))


def _decode_time(prep: PreprocessedLCD, cfg: dict[str, Any]) -> tuple[str, float, tuple[int, int, int, int]]:
    tcfg = cfg["time"]
    w, h = int(tcfg["w"]), int(tcfg["h"])
    y = int(tcfg["y"])
    xs = [int(v) for v in tcfg["slots_x"]]
    sdx, sdy = int(tcfg["search_dx"]), int(tcfg["search_dy"])
    thick = float(tcfg["thickness_ratio"])
    slant = float(tcfg["slant_deg"])
    blur = float(tcfg["blur_sigma"])

    tmpls = {str(d): render_7seg_template(str(d), w, h, thick, slant, blur) for d in range(10)}
    seg_masks = _build_7seg_masks(w, h, thick, slant)

    slot_specs = [
        (xs[0], ("0", "1", "2"), True),
        (xs[1], tuple(str(i) for i in range(10)), False),
        (xs[2], tuple(str(i) for i in range(6)), False),
        (xs[3], tuple(str(i) for i in range(10)), False),
    ]
    all_scores, best_xs, best_y = _score_row_joint_y(
        prep.ink_top,
        tmpls,
        seg_masks,
        slot_specs,
        y,
        w,
        h,
        sdx,
        sdy,
        float(tcfg.get("blank_ink_p95", 0.28)),
    )
    s_h1, s_h2, s_m1, s_m2 = all_scores

    hh_cands: list[tuple[float, str]] = []
    for h1, sc1 in s_h1.items():
        for h2, sc2 in s_h2.items():
            hh = int(f"{h1}{h2}")
            if 0 <= hh <= 23:
                hh_cands.append((0.5 * (sc1 + sc2), f"{hh:02d}" if h1 != "" else h2))
    hh_cands.sort(key=lambda item: item[0], reverse=True)

    mm_cands: list[tuple[float, str]] = []
    for m1, sc1 in s_m1.items():
        for m2, sc2 in s_m2.items():
            mm_cands.append((0.5 * (sc1 + sc2), f"{m1}{m2}"))
    mm_cands.sort(key=lambda item: item[0], reverse=True)

    best_hh_s, best_hh = hh_cands[0]
    sec_hh_s = hh_cands[1][0] if len(hh_cands) > 1 else 0.0
    best_mm_s, best_mm = mm_cands[0]
    sec_mm_s = mm_cands[1][0] if len(mm_cands) > 1 else 0.0

    conf = min(
        _score_to_confidence(best_hh_s, best_hh_s - sec_hh_s),
        _score_to_confidence(best_mm_s, best_mm_s - sec_mm_s),
    )
    roi = (best_xs[0], best_y, best_xs[3] + w - best_xs[0], h)
    return f"{best_hh}:{best_mm}", conf, roi


def _decode_date(prep: PreprocessedLCD, cfg: dict[str, Any]) -> tuple[str, float, tuple[int, int, int, int]]:
    dcfg = cfg["date"]
    w, h = int(dcfg["w"]), int(dcfg["h"])
    y_bot = int(dcfg["dy_from_divider"])
    xs = [int(v) for v in dcfg["slots_x"]]
    sdx, sdy = int(dcfg["search_dx"]), int(dcfg["search_dy"])
    thick = float(dcfg["thickness_ratio"])
    slant = float(dcfg["slant_deg"])
    blur = float(dcfg["blur_sigma"])

    tmpls = {str(d): render_7seg_template(str(d), w, h, thick, slant, blur) for d in range(10)}
    seg_masks = _build_7seg_masks(w, h, thick, slant)

    slot_specs = [
        (xs[0], ("0", "1"), True),
        (xs[1], tuple(str(i) for i in range(10)), False),
        (xs[2], ("0", "1", "2", "3"), True),
        (xs[3], tuple(str(i) for i in range(10)), False),
    ]
    all_scores, best_xs, best_y = _score_row_joint_y(
        prep.ink_bottom,
        tmpls,
        seg_masks,
        slot_specs,
        y_bot,
        w,
        h,
        sdx,
        sdy,
        float(dcfg.get("blank_ink_p95", 0.22)),
    )
    s_m1, s_m2, s_d1, s_d2 = all_scores

    mm_cands: list[tuple[float, int]] = []
    for m1, sc1 in s_m1.items():
        for m2, sc2 in s_m2.items():
            m = int(f"{m1}{m2}")
            if 1 <= m <= 12:
                mm_cands.append((0.5 * (sc1 + sc2), m))
    mm_cands.sort(key=lambda item: item[0], reverse=True)

    best_score = -1e9
    sec_score = -1e9
    best_date = "01-01"
    for sc_m, m in mm_cands[:4]:
        max_d = DAYS_IN_MONTH[m]
        for d1, sc_d1 in s_d1.items():
            for d2, sc_d2 in s_d2.items():
                d = int(f"{d1}{d2}")
                if 1 <= d <= max_d:
                    sc = 0.5 * sc_m + 0.25 * (sc_d1 + sc_d2)
                    label = f"{m:02d}-{d:02d}"
                    if sc > best_score:
                        if label != best_date:
                            sec_score = best_score
                        best_score = sc
                        best_date = label
                    elif sc > sec_score and label != best_date:
                        sec_score = sc

    conf = _score_to_confidence(best_score, max(0.0, best_score - sec_score))
    roi = (best_xs[0], prep.divider_y + best_y, best_xs[3] + w - best_xs[0], h)
    return best_date, conf, roi


def _decode_weekday(prep: PreprocessedLCD, cfg: dict[str, Any]) -> tuple[str, float, tuple[int, int, int, int]]:
    wcfg = cfg["weekday"]
    y_bot = int(wcfg["dy_from_divider"])
    h = int(wcfg["h"])
    char_w = int(wcfg["char_w"])
    char_gap = int(wcfg["char_gap"])
    x_start = prep.vdiv1_x + int(wcfg["x_offset_from_vdiv1"])
    sdx, sdy = int(wcfg["search_dx"]), int(wcfg["search_dy"])

    word_scores: list[tuple[float, str, tuple[int, int]]] = []
    for day in WEEKDAYS:
        tmpl = render_weekday_template(
            day,
            char_w=char_w,
            char_gap=char_gap,
            height=h,
            thickness_ratio=float(wcfg["thickness_ratio"]),
            slant_deg=float(wcfg["slant_deg"]),
            blur_sigma=float(wcfg["blur_sigma"]),
        )
        sc, bx, by = _ncc_best_shift(prep.ink_bottom, tmpl, x_start, y_bot, sdx, sdy)
        word_scores.append((sc, day, (bx, by)))

    word_scores.sort(key=lambda item: item[0], reverse=True)
    best_s, best_day, best_xy = word_scores[0]
    sec_s = word_scores[1][0] if len(word_scores) > 1 else 0.0
    conf = _score_to_confidence(best_s, best_s - sec_s)

    total_w = 3 * char_w + 2 * char_gap
    roi = (best_xy[0], prep.divider_y + best_xy[1], total_w, h)
    return best_day, conf, roi


def _decode_temperature(prep: PreprocessedLCD, cfg: dict[str, Any]) -> tuple[str, float, tuple[int, int, int, int]]:
    tcfg = cfg["temperature"]
    w, h = int(tcfg["w"]), int(tcfg["h"])
    y_bot = int(tcfg["dy_from_divider"])
    xs = [int(v) for v in tcfg["slots_x"]]
    sdx, sdy = int(tcfg["search_dx"]), int(tcfg["search_dy"])
    min_t = int(tcfg.get("min_temp", -20))
    max_t = int(tcfg.get("max_temp", 60))
    thick = float(tcfg["thickness_ratio"])
    slant = float(tcfg["slant_deg"])
    blur = float(tcfg["blur_sigma"])

    tmpls = {str(d): render_7seg_template(str(d), w, h, thick, slant, blur) for d in range(10)}
    seg_masks = _build_7seg_masks(w, h, thick, slant)

    slot_specs = [
        (xs[0], tuple(str(i) for i in range(10)), True),
        (xs[1], tuple(str(i) for i in range(10)), False),
    ]
    all_scores, best_xs, best_y = _score_row_joint_y(
        prep.ink_bottom,
        tmpls,
        seg_masks,
        slot_specs,
        y_bot,
        w,
        h,
        sdx,
        sdy,
        float(tcfg.get("blank_ink_p95", 0.24)),
    )
    s_t1, s_t2 = all_scores

    minus_x0 = max(prep.vdiv2_x + 4, xs[0] - 18)
    minus_x1 = max(minus_x0 + 4, xs[0] - 2)
    minus_y0 = max(0, best_y + int(h * 0.36))
    minus_y1 = min(prep.ink_bottom.shape[0], best_y + int(h * 0.64))
    minus_patch = prep.ink_bottom[minus_y0:minus_y1, minus_x0:minus_x1]
    is_neg = minus_patch.size > 0 and float(np.mean(minus_patch)) > 0.45

    cands: list[tuple[float, str]] = []
    for d1, sc1 in s_t1.items():
        for d2, sc2 in s_t2.items():
            val = int(f"{d1}{d2}")
            if is_neg:
                val = -val
            if min_t <= val <= max_t:
                cands.append((0.5 * (sc1 + sc2), str(val)))

    cands.sort(key=lambda item: item[0], reverse=True)
    best_s, best_temp = cands[0]
    sec_s = cands[1][0] if len(cands) > 1 else 0.0
    conf = _score_to_confidence(best_s, best_s - sec_s)

    roi = (best_xs[0], prep.divider_y + best_y, best_xs[1] + w - best_xs[0], h)
    return best_temp, conf, roi


def decode_lcd(prep: PreprocessedLCD, config: dict[str, Any] | None = None) -> DecodeResult:
    """Decode Time, Date, Weekday, and Temperature from a preprocessed LCD."""
    cfg = config if config is not None else load_config()
    time_str, conf_time, roi_time = _decode_time(prep, cfg)
    date_str, conf_date, roi_date = _decode_date(prep, cfg)
    weekday, conf_week, roi_week = _decode_weekday(prep, cfg)
    temp_str, conf_temp, roi_temp = _decode_temperature(prep, cfg)

    overall = float(0.30 * conf_time + 0.25 * conf_date + 0.20 * conf_week + 0.25 * conf_temp)
    return DecodeResult(
        time_str=time_str,
        date_str=date_str,
        weekday=weekday,
        temperature=temp_str,
        conf_time=round(conf_time, 3),
        conf_date=round(conf_date, 3),
        conf_weekday=round(conf_week, 3),
        conf_temperature=round(conf_temp, 3),
        confidence=round(overall, 3),
        roi_boxes={
            "time": roi_time,
            "date": roi_date,
            "weekday": roi_week,
            "temperature": roi_temp,
        },
    )
