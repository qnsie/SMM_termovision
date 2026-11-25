"""End-to-end processing pipeline, debug overlay rendering, and CSV export."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from monitor_reader.decoder import DecodeResult, decode_lcd, load_config
from monitor_reader.detector import detect_and_rectify_lcd
from monitor_reader.exif_utils import extract_photo_timestamp, load_image_bgr, save_image_bgr
from monitor_reader.preprocessor import PreprocessedLCD, preprocess_lcd

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_HEADERS = [
    "plik",
    "czas_zdjecia",
    "godzina",
    "data",
    "dzien_tygodnia",
    "temperatura_C",
    "pewnosc",
    "status",
]


@dataclass
class MeasurementRecord:
    """Single photo measurement result plus diagnostic images for preview/export."""

    filename: str
    filepath: str
    photo_timestamp: str
    time: str
    date: str
    weekday: str
    temperature: str
    confidence: float
    conf_time: float
    conf_date: float
    conf_weekday: float
    conf_temperature: float
    status: str
    error_reason: str = ""
    crop_preview: np.ndarray | None = None
    warped_lcd: np.ndarray | None = None
    ink_vis: np.ndarray | None = None
    roi_overlay: np.ndarray | None = None


def list_image_files(input_path: str | Path) -> list[Path]:
    """Return sorted image files from a directory (non-recursive) or single file path."""
    target = Path(input_path)
    if not target.exists():
        raise FileNotFoundError(f"Ścieżka nie istnieje: {target}")
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files = [
        p
        for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def draw_roi_overlay(
    warped_lcd: np.ndarray,
    dec: DecodeResult,
    prep: PreprocessedLCD,
) -> np.ndarray:
    """Draw structural dividers and labeled ROI boxes on the 440x200 LCD image."""
    vis = warped_lcd.copy()
    cv2.line(vis, (0, prep.divider_y), (vis.shape[1] - 1, prep.divider_y), (0, 185, 255), 1)
    cv2.line(vis, (prep.vdiv1_x, prep.divider_y), (prep.vdiv1_x, vis.shape[0] - 1), (0, 185, 255), 1)
    cv2.line(vis, (prep.vdiv2_x, prep.divider_y), (prep.vdiv2_x, vis.shape[0] - 1), (0, 185, 255), 1)

    styles = {
        "time": ((40, 220, 40), dec.time_str),
        "date": ((255, 190, 40), dec.date_str),
        "weekday": ((255, 110, 220), dec.weekday),
        "temperature": ((60, 140, 255), f"{dec.temperature}C"),
    }
    for key, (color, label) in styles.items():
        x, y, w, h = dec.roi_boxes[key]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(vis.shape[1] - 1, x + w)
        y1 = min(vis.shape[0] - 1, y + h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        text_y = max(12, y0 - 3) if key != "time" else min(vis.shape[0] - 4, y1 + 12)
        cv2.putText(
            vis,
            label,
            (max(2, x0), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def process_image(path: str | Path, config: dict[str, Any] | None = None) -> MeasurementRecord:
    """Run EXIF loading, LCD detection, preprocessing, and offline decoding on one photo."""
    cfg = config if config is not None else load_config()
    file_path = Path(path)
    ts = extract_photo_timestamp(file_path)

    try:
        img_bgr = load_image_bgr(file_path)
    except Exception as exc:
        return MeasurementRecord(
            filename=file_path.name,
            filepath=str(file_path),
            photo_timestamp=ts,
            time="",
            date="",
            weekday="",
            temperature="",
            confidence=0.0,
            conf_time=0.0,
            conf_date=0.0,
            conf_weekday=0.0,
            conf_temperature=0.0,
            status="NIE WYKRYTO",
            error_reason=f"Błąd wczytywania: {exc}",
        )

    det = detect_and_rectify_lcd(img_bgr)
    if not det.found or det.warped_lcd is None:
        return MeasurementRecord(
            filename=file_path.name,
            filepath=str(file_path),
            photo_timestamp=ts,
            time="",
            date="",
            weekday="",
            temperature="",
            confidence=0.0,
            conf_time=0.0,
            conf_date=0.0,
            conf_weekday=0.0,
            conf_temperature=0.0,
            status="NIE WYKRYTO",
            error_reason=det.reason,
        )

    prep = preprocess_lcd(det.warped_lcd, cfg)
    dec = decode_lcd(prep, cfg)

    low_thr = float(cfg.get("thresholds", {}).get("low_confidence", 0.42))
    status = "OK" if dec.confidence >= low_thr else "NISKA PEWNOŚĆ"

    ink_u8 = np.clip(prep.ink_full * 255.0, 0, 255).astype(np.uint8)
    ink_vis = cv2.cvtColor(ink_u8, cv2.COLOR_GRAY2BGR)
    roi_overlay = draw_roi_overlay(det.warped_lcd, dec, prep)

    return MeasurementRecord(
        filename=file_path.name,
        filepath=str(file_path),
        photo_timestamp=ts,
        time=dec.time_str,
        date=dec.date_str,
        weekday=dec.weekday,
        temperature=dec.temperature,
        confidence=dec.confidence,
        conf_time=dec.conf_time,
        conf_date=dec.conf_date,
        conf_weekday=dec.conf_weekday,
        conf_temperature=dec.conf_temperature,
        status=status,
        crop_preview=det.crop_preview,
        warped_lcd=det.warped_lcd,
        ink_vis=ink_vis,
        roi_overlay=roi_overlay,
    )


def save_debug_artifacts(record: MeasurementRecord, debug_dir: str | Path) -> None:
    """Save diagnostic PNGs (monitor crop, rectified LCD+ROI, ink map) for a record."""
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record.filename).stem
    if record.crop_preview is not None:
        save_image_bgr(out_dir / f"{stem}_detect.png", record.crop_preview)
    if record.roi_overlay is not None:
        save_image_bgr(out_dir / f"{stem}_lcd_roi.png", record.roi_overlay)
    if record.ink_vis is not None:
        save_image_bgr(out_dir / f"{stem}_ink.png", record.ink_vis)


def export_csv(
    records: list[MeasurementRecord],
    output_path: str | Path,
    delimiter: str = ",",
) -> None:
    """Export measurement records to a UTF-8 with BOM CSV file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(CSV_HEADERS)
        for rec in records:
            writer.writerow(
                [
                    rec.filename,
                    rec.photo_timestamp,
                    rec.time,
                    rec.date,
                    rec.weekday,
                    rec.temperature,
                    f"{rec.confidence:.3f}",
                    rec.status,
                ]
            )
