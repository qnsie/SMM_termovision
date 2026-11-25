"""EXIF-aware image loading, saving, and timestamp extraction utilities."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_EXIF_DATE_ORIGINAL = 36867
_EXIF_DATE_DIGITIZED = 36868
_EXIF_DATETIME = 306
_FILENAME_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image with EXIF orientation applied and return a BGR uint8 array."""
    file_path = Path(path)
    with Image.open(file_path) as img:
        transposed = ImageOps.exif_transpose(img)
        rgb = transposed.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_image_bgr(path: str | Path, img_bgr: np.ndarray) -> None:
    """Save a BGR uint8 image via Pillow so non-ASCII Windows paths work reliably."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path)


def _normalize_exif_timestamp(raw_val: object) -> str | None:
    if raw_val is None:
        return None
    if isinstance(raw_val, bytes):
        text = raw_val.decode("ascii", errors="ignore").strip("\x00 ").strip()
    else:
        text = str(raw_val).strip("\x00 ").strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text[:19], fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


def extract_photo_timestamp(path: str | Path) -> str:
    """Return photo timestamp (YYYY-MM-DD HH:MM:SS) from EXIF, filename, or mtime."""
    file_path = Path(path)
    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
            if exif:
                for tag in (_EXIF_DATE_ORIGINAL, _EXIF_DATE_DIGITIZED, _EXIF_DATETIME):
                    parsed = _normalize_exif_timestamp(exif.get(tag))
                    if parsed:
                        return parsed
                # DateTimeOriginal often lives in the Exif IFD (0x8769)
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd:
                    for tag in (_EXIF_DATE_ORIGINAL, _EXIF_DATE_DIGITIZED):
                        parsed = _normalize_exif_timestamp(exif_ifd.get(tag))
                        if parsed:
                            return parsed
    except Exception:
        pass

    match = _FILENAME_TS_RE.search(file_path.stem)
    if match:
        y, m, d, hh, mm, ss = match.groups()
        try:
            dt = datetime(int(y), int(m), int(d), int(hh), int(mm), int(ss))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    mtime = file_path.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
