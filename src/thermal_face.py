import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from src.face import ensure_model, calculate_point_temp, calculate_roi_temp


def load_thermal_yolo(model_path: Path) -> YOLO:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Brak pliku wag modelu: {model_path}")
    return YOLO(str(model_path))


def _create_landmarker(mp_model_path: Path, mp_model_url: str) -> vision.FaceLandmarker:
    model_bytes = ensure_model(mp_model_path, mp_model_url)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _extract_points_from_crop(
    gray_img: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarker: vision.FaceLandmarker,
    clahe: cv2.CLAHE,
    key_landmarks: dict[str, list[int]],
) -> dict[str, list[tuple[int, int]]] | None:
    """Wycina twarz, podbija kontrast, powiększa do 256x256 i zwraca punkty w pikselach pełnego kadru."""
    h, w = gray_img.shape
    x1, y1, x2, y2 = bbox
    crop = gray_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Powiększenie wycinka do 256x256 jest kluczowe, żeby MediaPipe złapał punkty na niskiej rozdzielczości IR
    enhanced = clahe.apply(crop)
    upscaled = cv2.resize(enhanced, (256, 256), interpolation=cv2.INTER_CUBIC)
    rgb_crop = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)

    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop))
    if not res.face_landmarks:
        return None

    lms = res.face_landmarks[0]
    cw, ch = x2 - x1, y2 - y1
    points_px = {}

    for name, indices in key_landmarks.items():
        pts = []
        for idx in indices:
            px = int(np.clip(x1 + lms[idx].x * cw, 0, w - 1))
            py = int(np.clip(y1 + lms[idx].y * ch, 0, h - 1))
            pts.append((px, py))
        points_px[name] = pts

    return points_px


def detect_ir_photo_landmarks(
    thermal_matrix: np.ndarray,
    yolo_model: YOLO,
    mp_model_path: Path,
    mp_model_url: str,
    key_landmarks: dict[str, list[int]],
    yolo_conf: float = 0.4,
) -> dict | None:
    """Wykrywa prostokąt twarzy (YOLO) oraz punkty (MediaPipe) na zdjęciu IR i liczy prawdziwe °C."""
    norm_ir = cv2.normalize(thermal_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ir_color = cv2.applyColorMap(norm_ir, cv2.COLORMAP_INFERNO)

    yolo_res = yolo_model.predict(ir_color, conf=yolo_conf, verbose=False)
    if not yolo_res or len(yolo_res[0].boxes) == 0:
        return None

    h, w = thermal_matrix.shape
    best_idx = int(yolo_res[0].boxes.conf.argmax())
    x1, y1, x2, y2 = yolo_res[0].boxes.xyxy[best_idx].cpu().numpy().astype(int)
    bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    with _create_landmarker(mp_model_path, mp_model_url) as landmarker:
        points = _extract_points_from_crop(norm_ir, bbox, landmarker, clahe, key_landmarks)

    temps = {}
    if points:
        for name, pts in points.items():
            if name == "forehead":
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                f_box = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
                temps[name] = round(calculate_roi_temp(thermal_matrix, f_box, percentile=90), 2)
            else:
                vals = [calculate_point_temp(thermal_matrix, p, radius=1) for p in pts]
                temps[name] = round(float(np.mean(vals)), 2)

    return {"bbox": bbox, "points": points, "temps": temps}


def analyze_ir_video_landmarks(
    video_path: str | Path,
    output_video_path: str | Path,
    yolo_model: YOLO,
    mp_model_path: Path,
    mp_model_url: str,
    key_landmarks: dict[str, list[int]],
    temp_range: tuple[float, float] = (20.0, 40.0),
    yolo_conf: float = 0.4,
) -> pd.DataFrame:
    """Śledzi punkty twarzy na filmiku IR, rysuje je na wideo i zwraca przebieg temperatur w czasie."""
    try:
        safe_in = os.path.relpath(str(video_path))
        safe_out = os.path.relpath(str(output_video_path))
    except ValueError:
        safe_in, safe_out = str(video_path), str(output_video_path)

    cap = cv2.VideoCapture(safe_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie można otworzyć wideo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(safe_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    t_min, t_max = temp_range
    records = []
    frame_idx = 0

    colors_bgr = {
        "eye_inner": (255, 255, 0),   # cyjan
        "eye_outer": (0, 255, 255),   # żółty
        "mouth": (0, 255, 0),         # zielony
        "forehead": (255, 255, 255),  # biały
    }

    with _create_landmarker(mp_model_path, mp_model_url) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_map = t_min + (gray.astype(np.float32) / 255.0) * (t_max - t_min)

            row = {"frame": frame_idx, "time_s": round(frame_idx / fps, 2)}
            for name in key_landmarks:
                row[f"temp_{name}"] = np.nan

            yolo_res = yolo_model.predict(frame, conf=yolo_conf, verbose=False)
            if yolo_res and len(yolo_res[0].boxes) > 0:
                best_idx = int(yolo_res[0].boxes.conf.argmax())
                x1, y1, x2, y2 = yolo_res[0].boxes.xyxy[best_idx].cpu().numpy().astype(int)
                bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (120, 120, 120), 1)

                points = _extract_points_from_crop(gray, bbox, landmarker, clahe, key_landmarks)
                if points:
                    for name, pts in points.items():
                        for px, py in pts:
                            cv2.circle(frame, (px, py), 2, colors_bgr.get(name, (0, 255, 0)), -1)
                        row[f"temp_{name}"] = round(float(np.mean([temp_map[py, px] for px, py in pts])), 2)

            records.append(row)
            out.write(frame)

    cap.release()
    out.release()
    return pd.DataFrame(records)