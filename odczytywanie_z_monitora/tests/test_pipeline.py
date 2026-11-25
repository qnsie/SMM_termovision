"""Automated tests for detection, rectification, offline decoding, CLI/CSV, and GUI."""

from __future__ import annotations

import csv
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import cv2
import numpy as np

import cli
from monitor_reader.decoder import (
    WEEKDAYS,
    decode_lcd,
    load_config,
    render_7seg_template,
    render_weekday_template,
)
from monitor_reader.detector import WARP_HEIGHT, WARP_WIDTH, detect_and_rectify_lcd
from monitor_reader.exif_utils import load_image_bgr
from monitor_reader.gui import MonitorReaderApp
from monitor_reader.pipeline import process_image
from monitor_reader.preprocessor import preprocess_lcd

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PHOTOS = ("20261125_144205.jpg", "20261125_144230.jpg")


def _render_synthetic_lcd(
    time_str: str,
    date_str: str,
    weekday: str,
    temp_val: int,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render a realistic noisy, shaded, downscaled-then-warped 440x200 LCD BGR image."""
    ink = np.zeros((WARP_HEIGHT, WARP_WIDTH), dtype=np.float32)

    # Horizontal and vertical structural dividers
    div_y = int(cfg["layout"]["divider_y"])
    vdiv1_x = int(cfg["layout"]["vdiv1_x"])
    vdiv2_x = int(cfg["layout"]["vdiv2_x"])
    ink[div_y - 1 : div_y + 1, 12:-12] = 0.75
    ink[div_y + 2 : -10, vdiv1_x - 1 : vdiv1_x + 1] = 0.65
    ink[div_y + 2 : -10, vdiv2_x - 1 : vdiv2_x + 1] = 0.65

    # Top row: time digits
    tcfg = cfg["time"]
    hh, mm = time_str.split(":")
    time_chars = [hh[0] if hh[0] != "0" else "0", hh[1], mm[0], mm[1]]
    for idx, ch in enumerate(time_chars):
        if ch == "":
            continue
        t = render_7seg_template(
            ch,
            int(tcfg["w"]),
            int(tcfg["h"]),
            float(tcfg["thickness_ratio"]),
            float(tcfg["slant_deg"]),
            0.8,
        )
        x0, y0 = int(tcfg["slots_x"][idx]), int(tcfg["y"])
        ink[y0 : y0 + int(tcfg["h"]), x0 : x0 + int(tcfg["w"])] = np.maximum(
            ink[y0 : y0 + int(tcfg["h"]), x0 : x0 + int(tcfg["w"])], t
        )

    # Bottom-left: date MM-DD
    dcfg = cfg["date"]
    mon, day = date_str.split("-")
    for idx, ch in enumerate([mon[0], mon[1], day[0], day[1]]):
        t = render_7seg_template(
            ch,
            int(dcfg["w"]),
            int(dcfg["h"]),
            float(dcfg["thickness_ratio"]),
            float(dcfg["slant_deg"]),
            0.6,
        )
        x0 = int(dcfg["slots_x"][idx])
        y0 = div_y + int(dcfg["dy_from_divider"])
        ink[y0 : y0 + int(dcfg["h"]), x0 : x0 + int(dcfg["w"])] = np.maximum(
            ink[y0 : y0 + int(dcfg["h"]), x0 : x0 + int(dcfg["w"])], t
        )

    # Bottom-middle: weekday
    wcfg = cfg["weekday"]
    wt = render_weekday_template(
        weekday,
        int(wcfg["char_w"]),
        int(wcfg["char_gap"]),
        int(wcfg["h"]),
        float(wcfg["thickness_ratio"]),
        float(wcfg["slant_deg"]),
        0.6,
    )
    wx = vdiv1_x + int(wcfg["x_offset_from_vdiv1"])
    wy = div_y + int(wcfg["dy_from_divider"])
    ink[wy : wy + wt.shape[0], wx : wx + wt.shape[1]] = np.maximum(
        ink[wy : wy + wt.shape[0], wx : wx + wt.shape[1]], wt
    )

    # Bottom-right: temperature
    temp_cfg = cfg["temperature"]
    temp_str = f"{abs(temp_val):02d}" if abs(temp_val) >= 10 else f" {abs(temp_val)}"
    for idx, ch in enumerate(temp_str):
        if ch == " ":
            continue
        t = render_7seg_template(
            ch,
            int(temp_cfg["w"]),
            int(temp_cfg["h"]),
            float(temp_cfg["thickness_ratio"]),
            float(temp_cfg["slant_deg"]),
            0.6,
        )
        x0 = int(temp_cfg["slots_x"][idx])
        y0 = div_y + int(temp_cfg["dy_from_divider"])
        ink[y0 : y0 + int(temp_cfg["h"]), x0 : x0 + int(temp_cfg["w"])] = np.maximum(
            ink[y0 : y0 + int(temp_cfg["h"]), x0 : x0 + int(temp_cfg["w"])], t
        )

    # Simulate physical LCD background + darker bottom bezel shadow + camera blur + JPEG noise
    bg = np.full((WARP_HEIGHT, WARP_WIDTH), 118.0, dtype=np.float32)
    bg[div_y:, :] = 88.0
    gray = bg - ink * 68.0
    gray[div_y:, :] = bg[div_y:, :] - ink[div_y:, :] * 46.0
    gray = np.clip(gray + rng.normal(0.0, 4.5, size=gray.shape), 10, 235).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Downsample to realistic ~104x49 camera pixel size, JPEG compress, and upsample back
    tiny = cv2.resize(bgr, (104, 49), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", tiny, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.resize(decoded, (WARP_WIDTH, WARP_HEIGHT), interpolation=cv2.INTER_CUBIC)


class TestMonitorReaderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT_DIR / "config.json")

    def test_real_sample_photos(self) -> None:
        for name in SAMPLE_PHOTOS:
            with self.subTest(photo=name):
                rec = process_image(ROOT_DIR / name, self.cfg)
                self.assertEqual(rec.status, "OK", f"Unexpected status for {name}: {rec}")
                self.assertEqual(rec.time, "14:41")
                self.assertEqual(rec.date, "11-25")
                self.assertEqual(rec.weekday, "WED")
                self.assertEqual(rec.temperature, "23")
                self.assertTrue(rec.photo_timestamp.startswith("2026-11-25 14:42:"))

    def test_angle_perspective_scale_lighting_invariance(self) -> None:
        base_img = load_image_bgr(ROOT_DIR / "20261125_144230.jpg")
        h, w = base_img.shape[:2]
        center = (w * 0.5, h * 0.5)

        variants: dict[str, np.ndarray] = {}
        for angle in (-10.0, 10.0):
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            variants[f"rot_{angle}"] = cv2.warpAffine(
                base_img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

        # Perspective off-axis tilt around the monitor region
        src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst_pts = np.array(
            [[w * 0.04, h * 0.02], [w * 0.96, h * 0.05], [w * 0.94, h * 0.96], [w * 0.02, h * 0.95]],
            dtype=np.float32,
        )
        pmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        variants["perspective_tilt"] = cv2.warpPerspective(
            base_img, pmat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Scale 0.85x and 1.15x
        variants["scale_085"] = cv2.resize(base_img, (int(w * 0.85), int(h * 0.85)), interpolation=cv2.INTER_AREA)
        variants["scale_115"] = cv2.resize(base_img, (int(w * 1.15), int(h * 1.15)), interpolation=cv2.INTER_CUBIC)

        # Brightness +/- 20%
        variants["bright_minus20"] = np.clip(base_img.astype(np.float32) * 0.80, 0, 255).astype(np.uint8)
        variants["bright_plus20"] = np.clip(base_img.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)

        for label, img in variants.items():
            with self.subTest(variant=label):
                det = detect_and_rectify_lcd(img)
                self.assertTrue(det.found, f"Detector failed on {label}: {det.reason}")
                prep = preprocess_lcd(det.warped_lcd, self.cfg)
                dec = decode_lcd(prep, self.cfg)
                self.assertEqual(dec.time_str, "14:41")
                self.assertEqual(dec.date_str, "11-25")
                self.assertEqual(dec.weekday, "WED")
                self.assertEqual(dec.temperature, "23")

    def test_synthetic_generalization_all_digits_and_weekdays(self) -> None:
        rng = np.random.default_rng(42)
        cases = [
            ("00:00", "01-09", "MON", 5),
            ("08:35", "04-17", "TUE", 18),
            ("12:49", "07-23", "WED", 24),
            ("16:52", "09-14", "THU", 29),
            ("19:26", "10-08", "FRI", 31),
            ("21:14", "11-25", "SAT", 37),
            ("23:58", "12-30", "SUN", 10),
        ]
        for time_str, date_str, weekday, temp_val in cases:
            with self.subTest(time=time_str, date=date_str, weekday=weekday, temp=temp_val):
                lcd = _render_synthetic_lcd(time_str, date_str, weekday, temp_val, self.cfg, rng)
                prep = preprocess_lcd(lcd, self.cfg)
                dec = decode_lcd(prep, self.cfg)
                self.assertEqual(dec.time_str, time_str)
                self.assertEqual(dec.date_str, date_str)
                self.assertEqual(dec.weekday, weekday)
                self.assertEqual(dec.temperature, str(temp_val))
        self.assertEqual(len(cases), len(WEEKDAYS))

    def test_cli_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_csv = tmp / "pomiary.csv"
            debug_dir = tmp / "debug"
            rc = cli.main(
                [
                    "--input",
                    str(ROOT_DIR),
                    "--output",
                    str(out_csv),
                    "--delimiter",
                    ";",
                    "--debug-dir",
                    str(debug_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_csv.exists())
            with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f, delimiter=";"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                rows[0],
                ["plik", "czas_zdjecia", "godzina", "data", "dzien_tygodnia", "temperatura_C", "pewnosc", "status"],
            )
            for row in rows[1:]:
                self.assertEqual(row[2], "14:41")
                self.assertEqual(row[3], "11-25")
                self.assertEqual(row[4], "WED")
                self.assertEqual(row[5], "23")
                self.assertEqual(row[7], "OK")
            for name in SAMPLE_PHOTOS:
                stem = Path(name).stem
                self.assertTrue((debug_dir / f"{stem}_detect.png").exists())
                self.assertTrue((debug_dir / f"{stem}_lcd_roi.png").exists())
                self.assertTrue((debug_dir / f"{stem}_ink.png").exists())

    def test_gui_smoke(self) -> None:
        root = tk.Tk()
        root.withdraw()
        try:
            app = MonitorReaderApp(root, initial_dir=ROOT_DIR)
            records = [process_image(ROOT_DIR / name, self.cfg) for name in SAMPLE_PHOTOS]
            app.load_records(records)
            root.update_idletasks()

            self.assertEqual(len(app.tree.get_children()), 2)
            app.var_edit_temp.set("24")
            self.assertTrue(app.apply_manual_edit())
            self.assertEqual(app.records[0].temperature, "24")
            self.assertEqual(app.records[0].status, "POPRAWIONO")

            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = Path(tmpdir) / "gui_out.csv"
                app.export_to_path(csv_path)
                self.assertTrue(csv_path.exists())
                content = csv_path.read_text(encoding="utf-8-sig")
                self.assertIn("POPRAWIONO", content)
                self.assertIn(",24,", content)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
