from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_IR_PATH = BASE_DIR / "data" / "test" / "ZDJECIA" / "FLIR2300.jpg"
TEST_RGB_PATH = BASE_DIR / "data" / "test" / "ZDJECIA" / "FLIR2301.jpg"

# Punkty czoła w MediaPipe FaceMesh
FOREHEAD_LANDMARKS = [10, 151, 338, 109, 67, 297]

# Średnia z najcieplejszych 10% pikseli w ROI
TOP_PERCENTILE = 90

FACE_MODEL_PATH = BASE_DIR / "configs" / "face_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Kalibracja kadru RGB -> IR (zoom i przesunięcie obiektywów FLIR T420)
IR_SCALE = 1.85
IR_OFFSET_X = 0.06
IR_OFFSET_Y = 0.03

KEY_LANDMARKS = {
    "eye_inner": [133, 362],   # wewnętrzne kąciki oczu (najlepszy punkt pomiarowy)
    "eye_outer": [33, 263],    # zewnętrzne kąciki oczu
    "mouth": [61, 291],        # kąciki ust
    "forehead": [10, 151, 338, 109, 67, 297],
}

YOLO_IR_MODEL_PATH = BASE_DIR / "configs" / "yolov8_thermal.pt"