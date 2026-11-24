import os
import re
import tempfile
import time
from pathlib import Path

# Ensure Ultralytics can write its config even in restricted environments
if "YOLO_CONFIG_DIR" not in os.environ:
    _default_yolo_cfg = Path.home() / "AppData" / "Roaming" / "Ultralytics"
    try:
        _default_yolo_cfg.mkdir(parents=True, exist_ok=True)
        _probe = _default_yolo_cfg / ".write_test"
        _probe.touch()
        _probe.unlink()
    except OSError:
        os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_cfg")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial import cKDTree
from ultralytics import YOLO

from configs import settings
from src.face import ensure_model

DEFAULT_VIDEO_OUTPUT_DIR = settings.BASE_DIR / "outputs" / "video_analysis"
DEFAULT_CSV_OUTPUT_DIR = settings.BASE_DIR / "outputs" / "csv"
DEFAULT_OUTPUT_DIR = DEFAULT_VIDEO_OUTPUT_DIR


class ScaleOCRReader:
    """Odczytuje wartosci T_max i T_min z paska skali FLIR T420 za pomoca RapidOCR."""

    _NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

    def __init__(
        self,
        ocr_every_n_frames: int = 5,
        upscale_factor: int = 4,
        temp_bounds: tuple[float, float] = (-10.0, 100.0),
        default_fallback: tuple[float, float] = (20.0, 40.0),
    ) -> None:
        self.ocr_every_n_frames = max(1, int(ocr_every_n_frames))
        self.upscale_factor = max(2, int(upscale_factor))
        self.temp_bounds = temp_bounds
        self.default_fallback = default_fallback

        self.last_valid: tuple[float, float] | None = None
        self.attempts = 0
        self.valid_reads = 0
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    @staticmethod
    def _extract_scale_crops(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Wycina pola z liczbami T_max (gora-prawo) i T_min (dol-prawo)."""
        h, w = frame.shape[:2]
        x1 = int(round(0.872 * w))
        x2 = int(round(0.988 * w))
        top_y1, top_y2 = int(round(0.021 * h)), int(round(0.100 * h))
        bot_y1, bot_y2 = int(round(0.896 * h)), int(round(0.975 * h))
        return frame[top_y1:top_y2, x1:x2], frame[bot_y1:bot_y2, x1:x2]

    def _preprocess_ocr_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Powieksza wycinek, izoluje biale cyfry od kolorowego tla i odwraca do OCR."""
        # Biale cyfry maja wysokie wartosci we wszystkich trzech kanalach BGR
        neutral_white = crop_bgr.min(axis=2)
        up = cv2.resize(
            neutral_white,
            None,
            fx=self.upscale_factor,
            fy=self.upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        _, mask = cv2.threshold(up, 130, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        inv = 255 - mask
        padded = cv2.copyMakeBorder(inv, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    def _parse_float(self, ocr_result) -> float | None:
        if not ocr_result:
            return None
        items = ocr_result[0] if isinstance(ocr_result, tuple) else ocr_result
        if not items:
            return None

        joined = " ".join(str(item[1]) for item in items if len(item) >= 2)
        match = self._NUMBER_RE.search(joined)
        if not match:
            return None

        token = match.group(0).replace(",", ".")
        try:
            val = float(token)
        except ValueError:
            return None

        # Zabezpieczenie przed zgubieniem kropki dziesietnej (np. "352" -> 35.2)
        if "." not in token and 100.0 < val < 1000.0:
            val = val / 10.0
        return val

    def _is_valid_range(self, t_min: float | None, t_max: float | None) -> bool:
        if t_min is None or t_max is None:
            return False
        low, high = self.temp_bounds
        if not (low <= t_min <= high and low <= t_max <= high):
            return False
        return (t_max - t_min) >= 0.5

    def read_scale(self, frame: np.ndarray, frame_idx: int) -> tuple[float, float]:
        """Zwraca zwalidowany przedzial (T_min, T_max) dla biezacej klatki."""
        should_run = (
            self.last_valid is None
            or frame_idx <= 1
            or (frame_idx - 1) % self.ocr_every_n_frames == 0
        )
        if not should_run and self.last_valid is not None:
            return self.last_valid

        self.attempts += 1
        top_crop, bot_crop = self._extract_scale_crops(frame)
        ocr = self._get_ocr()

        top_proc = self._preprocess_ocr_crop(top_crop)
        bot_proc = self._preprocess_ocr_crop(bot_crop)

        t_max = self._parse_float(ocr(top_proc, use_det=True, use_cls=False, use_rec=True))
        t_min = self._parse_float(ocr(bot_proc, use_det=True, use_cls=False, use_rec=True))

        if self._is_valid_range(t_min, t_max):
            self.last_valid = (round(float(t_min), 2), round(float(t_max), 2))
            self.valid_reads += 1
            return self.last_valid

        return self.last_valid if self.last_valid is not None else self.default_fallback


class ColorbarTempMapper:
    """Mapuje kolory RGB z klatki FLIR na wartosci [0, 1] oraz C przy uzyciu paska barw."""

    def __init__(self) -> None:
        self.colorbar_box: tuple[int, int, int, int] | None = None
        self.colorbar_detected: bool = False
        self.palette_bgr: np.ndarray | None = None
        self.kdtree: cKDTree | None = None
        self.lut_norm: np.ndarray | None = None

    @staticmethod
    def _find_border_pairs(dark_cols: np.ndarray, x_offset: int) -> tuple[int, int] | None:
        """Znajduje lewa i prawa krawedz pionowej ramki wokol paska barw."""
        runs: list[tuple[int, int]] = []
        in_run = False
        start = 0
        for idx, is_dark in enumerate(dark_cols):
            if is_dark and not in_run:
                in_run = True
                start = idx
            elif not is_dark and in_run:
                in_run = False
                runs.append((start, idx - 1))
        if in_run:
            runs.append((start, len(dark_cols) - 1))

        for i in range(len(runs) - 1):
            left_end = runs[i][1]
            right_start = runs[i + 1][0]
            interior_width = right_start - (left_end + 1)
            if 4 <= interior_width <= 16:
                return x_offset + left_end + 1, x_offset + right_start
        return None

    def detect_colorbar(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Wykrywa wspolrzedne (x1, y1, x2, y2) wnetrza pionowego paska gradientu po prawej."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x_start = int(0.90 * w)
        y_mid_top, y_mid_bot = int(0.20 * h), int(0.80 * h)
        col_means = gray[y_mid_top:y_mid_bot, x_start:w].mean(axis=0)
        pair = self._find_border_pairs(col_means < 15.0, x_start)

        if pair is not None:
            x_l, x_r = pair
            x_in_l = x_l + 1 if (x_r - x_l) >= 4 else x_l
            x_in_r = x_r - 1 if (x_r - x_l) >= 4 else x_r

            y_search_top = int(0.11 * h)
            y_search_bot = int(0.875 * h)
            border_l = gray[y_search_top:y_search_bot, max(0, x_l - 2):x_l].mean(axis=1)
            border_r = gray[y_search_top:y_search_bot, x_r:min(w, x_r + 2)].mean(axis=1)
            interior_bright = gray[y_search_top:y_search_bot, x_in_l:x_in_r].mean(axis=1)

            dark_borders = (border_l < 15.0) & (border_r < 15.0)
            valid_top = np.where(dark_borders & (interior_bright > 25.0))[0]
            if valid_top.size > 0:
                y_top = y_search_top + int(valid_top[0])
                # Koniec paska: przed dolnym boksem tekstowym lub po zakonczeniu ciemnej ramki
                dark_rows_after = np.where(dark_borders[valid_top[0]:])[0]
                y_bot = y_top + int(dark_rows_after[-1]) + 1 if dark_rows_after.size > 0 else y_search_bot
                if (y_bot - y_top) >= 50:
                    palette = frame[y_top:y_bot, x_in_l:x_in_r].mean(axis=1)
                    if float(palette.std()) > 15.0:
                        return x_in_l, y_top, x_in_r, y_bot

        # Awaryjna proporcja typowa dla FLIR T420
        fx1, fx2 = int(round(0.956 * w)), int(round(0.978 * w))
        fy1, fy2 = int(round(0.125 * h)), int(round(0.875 * h))
        if fx2 > fx1 and (fy2 - fy1) >= 50:
            palette = frame[fy1:fy2, fx1:fx2].mean(axis=1)
            if float(palette.std()) > 15.0:
                return fx1, fy1, fx2, fy2
        return None

    def _build_lut(self, palette_bgr: np.ndarray) -> None:
        self.palette_bgr = palette_bgr
        self.kdtree = cKDTree(palette_bgr)
        centers = (np.arange(64, dtype=np.float32) * 4.0) + 2.0
        grid = np.stack(
            np.meshgrid(centers, centers, centers, indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        _, idx = self.kdtree.query(grid, workers=-1)
        n = len(palette_bgr)
        norm = ((n - 1 - idx) / max(1, n - 1)).astype(np.float32)
        self.lut_norm = norm.reshape(64, 64, 64)

    def update_from_frame(self, frame: np.ndarray) -> bool:
        """Aktualizuje LUT z paska barw jesli pasek jest obecny i sie zmienil."""
        box = self.detect_colorbar(frame)
        if box is None:
            if self.palette_bgr is None:
                self.colorbar_detected = False
            return self.colorbar_detected

        x1, y1, x2, y2 = box
        palette = frame[y1:y2, x1:x2].mean(axis=1).astype(np.float32)
        self.colorbar_box = box
        self.colorbar_detected = True

        if (
            self.palette_bgr is None
            or self.palette_bgr.shape != palette.shape
            or float(np.abs(self.palette_bgr - palette).mean()) > 15.0
        ):
            self._build_lut(palette)
        return True

    def frame_to_norm_u8(self, frame: np.ndarray) -> np.ndarray:
        """Konwertuje cala klatke do znormalizowanej mapy [0..255] przez LUT lub jasnosc (fallback)."""
        if self.colorbar_detected and self.lut_norm is not None:
            norm = self.lut_norm[
                frame[:, :, 0] >> 2,
                frame[:, :, 1] >> 2,
                frame[:, :, 2] >> 2,
            ]
            return np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def roi_to_temp_matrix(
        self,
        roi_bgr: np.ndarray,
        t_min: float,
        t_max: float,
    ) -> np.ndarray:
        """Przelicza piksele wycinka BGR na temperature [C] przez najblizszy kolor RGB na pasku."""
        if roi_bgr.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        rh, rw = roi_bgr.shape[:2]
        if self.colorbar_detected and self.kdtree is not None and self.palette_bgr is not None:
            pixels = roi_bgr.reshape(-1, 3).astype(np.float32)
            _, idx = self.kdtree.query(pixels)
            n = len(self.palette_bgr)
            norm = ((n - 1 - idx) / max(1, n - 1)).astype(np.float32).reshape(rh, rw)
        else:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            norm = gray / 255.0

        return t_min + norm * (t_max - t_min)


def _apply_gamma_inferno(norm_u8: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        adjusted = norm_u8
    else:
        adjusted = np.clip(((norm_u8.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(adjusted, cv2.COLORMAP_INFERNO)


def _select_valid_yolo_box(
    result,
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int, int, int], float] | None:
    if result is None or len(result.boxes) == 0:
        return None

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)

    for idx in order:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        x1_i = max(0, int(round(x1)))
        y1_i = max(0, int(round(y1)))
        x2_i = min(frame_w, int(round(x2)))
        y2_i = min(frame_h, int(round(y2)))

        bw = x2_i - x1_i
        bh = y2_i - y1_i
        if bw < 10 or bh < 10:
            continue

        cx = 0.5 * (x1_i + x2_i)
        cy = 0.5 * (y1_i + y2_i)
        if cx > 0.87 * frame_w or cy < 0.11 * frame_h:
            continue

        return (x1_i, y1_i, x2_i, y2_i), float(confs[idx])
    return None


def detect_face_yolo(
    frame: np.ndarray,
    norm_u8: np.ndarray,
    yolo_model: YOLO,
    device: int | str = 0,
    conf: float = 0.25,
    imgsz: int = 640,
    preprocess_mode: str = "temp_inferno",
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Wykrywa twarz/glowe modelem yolov8_thermal na przygotowanej klatce IR."""
    h, w = frame.shape[:2]

    if preprocess_mode == "raw":
        candidates = [frame]
    elif preprocess_mode == "gray_inferno":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates = [cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)]
    elif preprocess_mode == "temp_inferno":
        candidates = [
            _apply_gamma_inferno(norm_u8, 1.5),
            _apply_gamma_inferno(norm_u8, 1.0),
            _apply_gamma_inferno(norm_u8, 2.0),
        ]
    else:
        raise ValueError(f"Nieznany tryb preprocessingu YOLO: {preprocess_mode}")

    for img_in in candidates:
        res = yolo_model.predict(
            img_in,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        picked = _select_valid_yolo_box(res[0] if res else None, w, h)
        if picked is not None:
            return picked

    return None, 0.0


def _create_landmarker(mp_model_path: Path, mp_model_url: str) -> vision.FaceLandmarker:
    model_bytes = ensure_model(mp_model_path, mp_model_url)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _detect_eye_inner_points(
    source_gray: np.ndarray,
    crop_box: tuple[int, int, int, int],
    landmarker: vision.FaceLandmarker,
    clahe: cv2.CLAHE,
    eye_indices: list[int],
) -> list[tuple[int, int]] | None:
    h, w = source_gray.shape
    x1, y1, x2, y2 = crop_box
    crop = source_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    enhanced = clahe.apply(crop)
    upscaled = cv2.resize(enhanced, (256, 256), interpolation=cv2.INTER_CUBIC)
    rgb_crop = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)

    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop))
    if not res.face_landmarks:
        return None

    lms = res.face_landmarks[0]
    cw, ch = x2 - x1, y2 - y1
    pts: list[tuple[int, int]] = []
    for idx in eye_indices:
        px = int(np.clip(x1 + lms[idx].x * cw, 0, w - 1))
        py = int(np.clip(y1 + lms[idx].y * ch, 0, h - 1))
        pts.append((px, py))
    return pts


def _measure_eye_inner_temp(
    frame: np.ndarray,
    pts: list[tuple[int, int]],
    mapper: ColorbarTempMapper,
    t_min: float,
    t_max: float,
    radius: int = 2,
) -> float:
    h, w = frame.shape[:2]
    patch_maxima: list[float] = []
    for px, py in pts:
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius + 1)
        y2 = min(h, py + radius + 1)
        patch_temps = mapper.roi_to_temp_matrix(frame[y1:y2, x1:x2], t_min, t_max)
        if patch_temps.size > 0:
            patch_maxima.append(float(patch_temps.max()))
    return round(max(patch_maxima), 2) if patch_maxima else float("nan")


def _draw_hud_overlay(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    eye_pts: list[tuple[int, int]] | None,
    t_min: float,
    t_max: float,
    t_head_max: float,
    t_eye_inner_max: float,
    best_head_max: float,
    is_manual_scale: bool,
) -> None:
    h, w = frame.shape[:2]

    if bbox is not None:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        if np.isfinite(t_head_max):
            box_label = (
                f"T:{t_head_max:.1f}C O:{t_eye_inner_max:.1f}C"
                if np.isfinite(t_eye_inner_max)
                else f"T:{t_head_max:.1f}C"
            )
            (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            lx1 = max(0, min(bbox[0], w - tw - 6))
            ly1 = bbox[3] + 1 if (bbox[3] + th + 6) < h else max(0, bbox[1] - th - 5)
            lx2 = min(w, lx1 + tw + 6)
            ly2 = min(h, ly1 + th + 5)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
            cv2.putText(
                frame,
                box_label,
                (lx1 + 3, ly2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if eye_pts:
        for px, py in eye_pts:
            cv2.circle(frame, (px, py), 3, (0, 0, 0), -1)
            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

    # Kompaktowy 3-liniowy pasek HUD ponizej gornego paska FLIR
    hud_x1, hud_y1, hud_x2, hud_y2 = 4, 26, 200, 62
    sub = frame[hud_y1:hud_y2, hud_x1:hud_x2]
    overlay = np.zeros_like(sub)
    cv2.addWeighted(overlay, 0.65, sub, 0.35, 0, dst=sub)

    mode_tag = "MAN" if is_manual_scale else "OCR"
    head_str = f"{t_head_max:.1f}C" if np.isfinite(t_head_max) else "--"
    eye_str = f"{t_eye_inner_max:.1f}C" if np.isfinite(t_eye_inner_max) else "--"
    best_head_str = f"{best_head_max:.1f}C" if np.isfinite(best_head_max) else "--"

    cv2.putText(
        frame,
        f"Skala[{mode_tag}]: {t_min:.1f}-{t_max:.1f}C",
        (hud_x1 + 4, hud_y1 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Oczy: {eye_str} | Twarz: {head_str}",
        (hud_x1 + 4, hud_y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Max twarz: {best_head_str}",
        (hud_x1 + 4, hud_y1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )


def _resolve_cv2_path(path: str | Path) -> str:
    try:
        return os.path.relpath(str(path))
    except ValueError:
        return str(path)


def analyze_ir_video(
    video_path: str | Path,
    output_video_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    manual_temp_range: tuple[float, float] | None = None,
    yolo_model_path: str | Path = settings.YOLO_IR_MODEL_PATH,
    mp_model_path: str | Path = settings.FACE_MODEL_PATH,
    mp_model_url: str = settings.FACE_MODEL_URL,
    yolo_conf: float = 0.25,
    yolo_imgsz: int = 640,
    yolo_preprocess: str = "temp_inferno",
    device: int | str = 0,
    min_bbox_width_for_landmarks: int = 50,
    ocr_every_n_frames: int = 5,
    eye_patch_radius: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Analizuje nagranie IR FLIR T420: detekcja twarzy, skala OCR/manual, colorbar i temperatury."""
    t_start = time.perf_counter()

    video_path = Path(video_path)
    output_video_path = (
        Path(output_video_path)
        if output_video_path is not None
        else DEFAULT_VIDEO_OUTPUT_DIR / f"{video_path.stem}_analyzed.mp4"
    )
    output_csv_path = (
        Path(output_csv_path)
        if output_csv_path is not None
        else DEFAULT_CSV_OUTPUT_DIR / f"{video_path.stem}_results.csv"
    )
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if manual_temp_range is not None:
        if len(manual_temp_range) != 2 or manual_temp_range[1] <= manual_temp_range[0]:
            raise ValueError(f"Niepoprawny zakres manual_temp_range: {manual_temp_range}")
        manual_range = (float(manual_temp_range[0]), float(manual_temp_range[1]))
        ocr_reader = None
    else:
        manual_range = None
        ocr_reader = ScaleOCRReader(ocr_every_n_frames=ocr_every_n_frames)

    if not Path(yolo_model_path).exists():
        raise FileNotFoundError(f"Brak modelu YOLO: {yolo_model_path}")

    resolved_device: int | str = device
    if isinstance(device, int) and not torch.cuda.is_available():
        resolved_device = "cpu"

    yolo_model = YOLO(str(yolo_model_path))

    cap = cv2.VideoCapture(_resolve_cv2_path(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie mozna otworzyc wideo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        _resolve_cv2_path(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Nie mozna utworzyc wideo wyjsciowego: {output_video_path}")

    mapper = ColorbarTempMapper()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eye_indices = settings.KEY_LANDMARKS["eye_inner"]

    records: list[dict] = []
    frame_idx = 0
    colorbar_frames = 0
    best_head_max = float("nan")

    try:
        with _create_landmarker(Path(mp_model_path), mp_model_url) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if manual_range is not None:
                    t_min, t_max = manual_range
                else:
                    t_min, t_max = ocr_reader.read_scale(frame, frame_idx)

                if mapper.update_from_frame(frame):
                    colorbar_frames += 1

                norm_u8 = mapper.frame_to_norm_u8(frame)
                bbox, _ = detect_face_yolo(
                    frame=frame,
                    norm_u8=norm_u8,
                    yolo_model=yolo_model,
                    device=resolved_device,
                    conf=yolo_conf,
                    imgsz=yolo_imgsz,
                    preprocess_mode=yolo_preprocess,
                )

                face_detected = bbox is not None
                landmarks_detected = False
                bbox_width_px = 0
                t_head_max = float("nan")
                t_eye_inner_max = float("nan")
                eye_pts: list[tuple[int, int]] | None = None

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    bbox_width_px = int(x2 - x1)

                    roi_temps = mapper.roi_to_temp_matrix(frame[y1:y2, x1:x2], t_min, t_max)
                    if roi_temps.size > 0:
                        t_head_max = round(float(np.percentile(roi_temps, 98)), 2)
                        if not np.isfinite(best_head_max) or t_head_max > best_head_max:
                            best_head_max = t_head_max

                    if bbox_width_px >= min_bbox_width_for_landmarks:
                        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        eye_pts = _detect_eye_inner_points(
                            raw_gray,
                            bbox,
                            landmarker,
                            clahe,
                            eye_indices,
                        )
                        if eye_pts is None:
                            bh = y2 - y1
                            pad_box = (
                                max(0, int(x1 - 0.15 * bbox_width_px)),
                                max(0, int(y1 - 0.15 * bh)),
                                min(width, int(x2 + 0.15 * bbox_width_px)),
                                min(height, int(y2 + 0.15 * bh)),
                            )
                            eye_pts = _detect_eye_inner_points(
                                norm_u8,
                                pad_box,
                                landmarker,
                                clahe,
                                eye_indices,
                            )

                        if eye_pts is not None:
                            landmarks_detected = True
                            t_eye_inner_max = _measure_eye_inner_temp(
                                frame,
                                eye_pts,
                                mapper,
                                t_min,
                                t_max,
                                radius=eye_patch_radius,
                            )

                annotated = frame.copy()
                _draw_hud_overlay(
                    annotated,
                    bbox=bbox,
                    eye_pts=eye_pts,
                    t_min=t_min,
                    t_max=t_max,
                    t_head_max=t_head_max,
                    t_eye_inner_max=t_eye_inner_max,
                    best_head_max=best_head_max,
                    is_manual_scale=(manual_range is not None),
                )
                writer.write(annotated)

                records.append(
                    {
                        "frame": frame_idx,
                        "time_s": round(frame_idx / fps, 2),
                        "scale_min": t_min,
                        "scale_max": t_max,
                        "face_detected": face_detected,
                        "landmarks_detected": landmarks_detected,
                        "bbox_width_px": bbox_width_px,
                        "T_head_max": t_head_max,
                        "T_eye_inner_max": t_eye_inner_max,
                    }
                )
    finally:
        cap.release()
        writer.release()

    columns = [
        "frame",
        "time_s",
        "scale_min",
        "scale_max",
        "face_detected",
        "landmarks_detected",
        "bbox_width_px",
        "T_head_max",
        "T_eye_inner_max",
    ]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(output_csv_path, index=False)

    total_frames = len(df)
    yolo_hits = int(df["face_detected"].sum()) if total_frames else 0
    mp_hits = int(df["landmarks_detected"].sum()) if total_frames else 0
    eligible_mp = int((df["bbox_width_px"] >= min_bbox_width_for_landmarks).sum()) if total_frames else 0

    summary = {
        "video_path": str(video_path),
        "output_video_path": str(output_video_path),
        "output_csv_path": str(output_csv_path),
        "scale_mode": "Manual" if manual_range is not None else "Auto-OCR",
        "device": str(resolved_device),
        "total_frames": total_frames,
        "yolo_detected_frames": yolo_hits,
        "yolo_detection_rate_pct": round(100.0 * yolo_hits / total_frames, 2) if total_frames else 0.0,
        "mp_detected_frames": mp_hits,
        "mp_detection_rate_pct": round(100.0 * mp_hits / total_frames, 2) if total_frames else 0.0,
        "mp_eligible_frames": eligible_mp,
        "mp_eligible_rate_pct": round(100.0 * mp_hits / eligible_mp, 2) if eligible_mp else 0.0,
        "colorbar_detected_frames": colorbar_frames,
        "ocr_attempts": ocr_reader.attempts if ocr_reader is not None else 0,
        "ocr_valid_reads": ocr_reader.valid_reads if ocr_reader is not None else 0,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }
    return df, summary
