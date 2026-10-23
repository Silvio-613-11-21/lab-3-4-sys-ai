import asyncio
import os
import shutil

import pytesseract
from PIL import Image

# Tesseract — внешняя программа. Путь не прописываем жёстко (как было
# c:\Program Files\...), а ищем сами: переменная окружения, PATH,
# стандартные каталоги установки.
_TESSERACT_CANDIDATES = [
    os.environ.get("TESSERACT_CMD", ""),
    shutil.which("tesseract") or "",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


def find_tesseract():
    """Вернуть путь к tesseract или None, если он не найден."""
    for path in _TESSERACT_CANDIDATES:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


TESSERACT_PATH = find_tesseract()


def available_languages():
    """Языки, установленные в системе (rus, eng, ...)."""
    if TESSERACT_PATH is None:
        return []
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return []


async def image_to_string(path: str, lang: str = "rus+eng"):
    return await asyncio.to_thread(sync_image_to_string, path, lang)


def sync_image_to_string(path: str, lang: str = "rus+eng"):
    if TESSERACT_PATH is None:
        return None, "Tesseract OCR не найден (см. README, раздел установки)"

    # Если запрошенного языка нет в системе — оставляем только установленные
    installed = available_languages()
    if installed:
        parts = [p for p in lang.split("+") if p in installed]
        lang = "+".join(parts) or installed[0]

    string = pytesseract.image_to_string(Image.open(path), lang=lang)
    print(string)  # по заданию текст дублируется в консоль

    coord = None
    return coord, string


# print(  pytesseract.image_to_boxes('image.png'))

# boxes =  pytesseract.image_to_boxes('image.png')
