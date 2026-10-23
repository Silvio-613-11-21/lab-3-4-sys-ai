import cv2
import numpy as np


def imread(path: str):
    """cv2.imread, который понимает кириллицу в пути.

    На Windows cv2.imread возвращает None, если в пути есть русские буквы
    (C:\\Users\\Иван\\...). Поэтому файл сначала читается как байты,
    а потом декодируется средствами OpenCV.
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, TypeError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
