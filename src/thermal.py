import flyr
import numpy as np


def extract_flir_data(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (macierz_temperatur_C, obraz_RGB) bez użycia exiftool."""
    thermogram = flyr.unpack(str(image_path))
    thermal_matrix = thermogram.celsius
    rgb_image = thermogram.optical
    return thermal_matrix, rgb_image