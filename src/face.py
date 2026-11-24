import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def ensure_model(model_path: Path, model_url: str) -> bytes:
    """Pobiera model .task jeśli nie istnieje i zwraca jego bajty."""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, str(model_path))
    return model_path.read_bytes()


def detect_landmarks_norm(rgb_path: str, model_path: Path, model_url: str) -> list | None:
    """Wykrywa 478 punktów twarzy na RGB i zwraca listę znormalizowanych współrzędnych."""
    img_bytes = np.fromfile(str(rgb_path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Nie można otworzyć pliku: {rgb_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    model_bytes = ensure_model(model_path, model_url)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    return result.face_landmarks[0] if result.face_landmarks else None


def get_bbox_from_landmarks(landmarks: list, indices: list[int]) -> tuple[float, float, float, float]:
    """Wyznacza znormalizowany prostokąt (x_min, y_min, x_max, y_max) z wybranych punktów."""
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))


def map_point_to_ir(
    x_norm: float,
    y_norm: float,
    ir_shape: tuple[int, int],
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[int, int]:
    """Przelicza pojedynczy punkt z RGB na piksele macierzy IR (z korektą zoomu i paralaksy)."""
    h, w = ir_shape
    tx = (x_norm - 0.5) * scale + 0.5 + dx
    ty = (y_norm - 0.5) * scale + 0.5 + dy
    px = int(np.clip(tx * w, 0, w - 1))
    py = int(np.clip(ty * h, 0, h - 1))
    return px, py


def map_bbox_to_ir(
    norm_bbox: tuple[float, float, float, float],
    ir_shape: tuple[int, int],
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[int, int, int, int]:
    """Przelicza prostokąt (x_min, y_min, x_max, y_max) z RGB na piksele IR."""
    x1, y1 = map_point_to_ir(norm_bbox[0], norm_bbox[1], ir_shape, scale, dx, dy)
    x2, y2 = map_point_to_ir(norm_bbox[2], norm_bbox[3], ir_shape, scale, dx, dy)
    return x1, y1, x2, y2


def calculate_roi_temp(thermal_matrix: np.ndarray, bbox_px: tuple[int, int, int, int], percentile: int = 90) -> float:
    """Liczy średnią temperaturę z najcieplejszych pikseli w prostokącie (np. czoło)."""
    x_min, y_min, x_max, y_max = bbox_px
    roi = thermal_matrix[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return float("nan")
    threshold = np.percentile(roi, percentile)
    return float(roi[roi >= threshold].mean())


def calculate_point_temp(thermal_matrix: np.ndarray, pt_px: tuple[int, int], radius: int = 1) -> float:
    """Odczytuje maksymalną temperaturę z małego otoczenia punktu (np. 3x3 px wokół kącika oka)."""
    h, w = thermal_matrix.shape
    cx, cy = pt_px
    patch = thermal_matrix[max(0, cy - radius):min(h, cy + radius + 1),
                           max(0, cx - radius):min(w, cx + radius + 1)]
    return float(patch.max()) if patch.size > 0 else float("nan")