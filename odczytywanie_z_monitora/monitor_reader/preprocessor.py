"""LCD preprocessing: independent top/bottom band normalization and divider detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PreprocessedLCD:
    """Normalized ink maps and detected structural dividers for a 440x200 LCD."""

    gray: np.ndarray  # (200, 440) float32 [0..1]
    ink_top: np.ndarray  # (divider_y, 440) float32 [0..1]
    ink_bottom: np.ndarray  # (200 - divider_y, 440) float32 [0..1]
    ink_full: np.ndarray  # (200, 440) float32 [0..1]
    divider_y: int
    vdiv1_x: int
    vdiv2_x: int


def _extract_band_ink(
    band: np.ndarray,
    morph_size: int,
    margin_top: int,
    margin_bottom: int,
    margin_x: int,
) -> np.ndarray:
    """Estimate local bright background via replicate-padded closing and normalize ink."""
    pad = morph_size
    padded = cv2.copyMakeBorder(band, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    bg_padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    bg_padded = cv2.GaussianBlur(bg_padded, (0, 0), 1.5)
    bg = bg_padded[pad:-pad, pad:-pad]

    ink = np.clip(bg - band, 0.0, None)

    # Zero out outer border rim where LCD glass meets dark shadow/bezel edge
    h, w = ink.shape
    y0 = min(margin_top, h // 4)
    y1 = max(y0 + 1, h - min(margin_bottom, h // 4))
    x0 = min(margin_x, w // 4)
    x1 = max(x0 + 1, w - margin_x)

    interior = ink[y0:y1, x0:x1]
    lo = float(np.percentile(interior, 5.0))
    hi = float(np.percentile(interior, 99.5))
    denom = max(0.025, hi - lo)
    norm = np.clip((ink - lo) / denom, 0.0, 1.0).astype(np.float32)

    # Suppress extreme border pixels that can distort template matching
    if y0 > 0:
        norm[:y0, :] *= 0.2
    if y1 < h:
        norm[y1:, :] *= 0.2
    if x0 > 0:
        norm[:, :x0] *= 0.2
    if x1 < w:
        norm[:, x1:] *= 0.2
    return norm


def _find_horizontal_divider(gray: np.ndarray, default_y: int = 128) -> int:
    """Locate the horizontal divider separating time row from bottom parameter strip."""
    y_min, y_max = 116, 144
    # Horizontal line is dark across the width and marks transition into darker bottom band
    strip = gray[y_min - 3 : y_max + 4, 24:416]
    # Local vertical dark-line response: (above + below)/2 - row
    row_means = strip.mean(axis=1)
    best_y = default_y
    best_score = -1e9
    for idx in range(3, len(row_means) - 3):
        y = y_min - 3 + idx
        line_darkness = 0.5 * (row_means[idx - 2] + row_means[idx + 2]) - row_means[idx]
        step_down = float(np.mean(row_means[max(0, idx - 3) : idx]) - np.mean(row_means[idx + 1 : idx + 4]))
        prior = -0.002 * abs(y - default_y)
        score = line_darkness + 0.35 * max(0.0, step_down) + prior
        if score > best_score:
            best_score = score
            best_y = y
    return int(best_y)


def _find_vertical_divider(
    ink_bottom: np.ndarray,
    search_range: tuple[int, int],
    default_x: int,
) -> int:
    """Locate a thin vertical separator line inside the bottom strip."""
    h, w = ink_bottom.shape
    x0 = max(10, search_range[0])
    x1 = min(w - 10, search_range[1])
    if x1 <= x0 or h < 12:
        return default_x
    # Vertical separator spans most of the bottom strip height
    y_start = min(h - 4, max(2, int(h * 0.12)))
    y_end = max(y_start + 4, int(h * 0.90))
    profile = ink_bottom[y_start:y_end, x0:x1].mean(axis=0)
    # Smooth slightly across 3 px and add mild prior toward default_x
    xs = np.arange(x0, x1, dtype=np.float32)
    prior = -0.004 * np.abs(xs - float(default_x))
    best_idx = int(np.argmax(profile + prior))
    if profile[best_idx] < 0.08:
        return default_x
    return int(x0 + best_idx)


def preprocess_lcd(warped_lcd: np.ndarray, config: dict[str, Any] | None = None) -> PreprocessedLCD:
    """Preprocess a 440x200 BGR LCD image into normalized top/bottom ink maps."""
    layout = (config or {}).get("layout", {})
    default_div_y = int(layout.get("divider_y", 128))
    default_vdiv1_x = int(layout.get("vdiv1_x", 148))
    default_vdiv2_x = int(layout.get("vdiv2_x", 280))

    gray = cv2.cvtColor(warped_lcd, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    divider_y = _find_horizontal_divider(gray, default_y=default_div_y)

    # Top band and bottom band normalized independently (bottom sits in bezel shadow)
    top_band = gray[:divider_y, :]
    bot_band = gray[divider_y:, :]

    ink_top = _extract_band_ink(top_band, morph_size=23, margin_top=10, margin_bottom=4, margin_x=10)
    ink_bottom = _extract_band_ink(bot_band, morph_size=15, margin_top=4, margin_bottom=10, margin_x=10)

    ink_full = np.vstack([ink_top, ink_bottom])

    vdiv1_x = _find_vertical_divider(ink_bottom, (default_vdiv1_x - 8, default_vdiv1_x + 7), default_vdiv1_x)
    vdiv2_x = _find_vertical_divider(ink_bottom, (default_vdiv2_x - 14, default_vdiv2_x + 14), default_vdiv2_x)

    return PreprocessedLCD(
        gray=gray,
        ink_top=ink_top,
        ink_bottom=ink_bottom,
        ink_full=ink_full,
        divider_y=divider_y,
        vdiv1_x=vdiv1_x,
        vdiv2_x=vdiv2_x,
    )
