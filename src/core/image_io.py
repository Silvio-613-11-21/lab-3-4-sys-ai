"""Чтение/запись изображений и подготовка картинок для интерфейса.

Главная причина, по которой этот модуль вообще существует:
``cv2.imread`` на Windows **не умеет** открывать файлы, в пути которых есть
символы вне ASCII (кириллица в именах папок — ``C:\\Users\\Иван\\...``).
Функция возвращает ``None``, программа молча рисует пустое окно и кажется
«сломанной». Поэтому файл читается через ``numpy.fromfile`` +
``cv2.imdecode``, а пишется через ``cv2.imencode`` + ``tofile``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import PREVIEW_MAX_SIDE


# --------------------------------------------------------------------------
# Файловый ввод-вывод
# --------------------------------------------------------------------------
def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Прочитать изображение по пути, который может содержать кириллицу.

    Возвращает BGR-массив ``numpy`` или ``None``, если файл не читается.
    """
    if path is None:
        return None

    file_path = Path(str(path)).expanduser()
    if not file_path.is_file():
        return None

    try:
        # fromfile работает с любыми именами, в отличие от cv2.imread
        buffer = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError:
        return None

    if buffer.size == 0:
        return None

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        # Файл может быть в формате, который OpenCV не понимает,
        # но который умеет читать Pillow (например, CMYK JPEG).
        try:
            with Image.open(file_path) as pil_image:
                rgb = np.array(pil_image.convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001 - любой сбой Pillow = «файл не прочитан»
            return None
    return image


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Сохранить изображение по пути с возможной кириллицей."""
    if image is None:
        return False
    file_path = Path(str(path)).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    extension = file_path.suffix.lower() or ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return False

    try:
        buffer.tofile(str(file_path))
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------
# Преобразования для интерфейса Flet
# --------------------------------------------------------------------------
def encode_png(image: np.ndarray) -> bytes:
    """BGR-массив -> байты PNG."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Не удалось закодировать изображение в PNG")
    return buffer.tobytes()


def to_data_uri(image: np.ndarray, max_side: int | None = PREVIEW_MAX_SIDE) -> str:
    """BGR-массив -> ``data:image/png;base64,...`` для ``ft.Image``.

    Изображение при необходимости уменьшается: гонять в UI кадр 4000x3000
    бессмысленно и медленно — Flet передаёт его целиком в base64.
    """
    if image is None:
        return ""
    if max_side:
        image = fit_within(image, max_side, max_side)
    payload = base64.b64encode(encode_png(image)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def fit_within(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Пропорционально уменьшить изображение до указанных границ."""
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image
    scale = min(max_width / width, max_height / height)
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------------------------------
# Конвертация между OpenCV и Pillow
# --------------------------------------------------------------------------
def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """BGR-numpy -> PIL.Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """PIL.Image -> BGR-numpy."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def pil_to_data_uri(image: Image.Image, max_side: int | None = PREVIEW_MAX_SIDE) -> str:
    """PIL.Image -> data URI (для картинок, отрисованных через Pillow)."""
    if max_side:
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"
