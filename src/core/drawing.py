"""Отрисовка подписей на изображениях.

``cv2.putText`` умеет рисовать только латиницу (Hershey-шрифты), поэтому
русские подписи («Объект найден», «Контур №3») на картинке превращаются в
``?????``. Здесь текст рисуется через Pillow с настоящим TrueType-шрифтом,
а результат возвращается обратно в BGR, чтобы дальше работать с OpenCV.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Кандидаты на «универсальный» шрифт с кириллицей.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

_MONO_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


@lru_cache(maxsize=32)
def _load_font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Найти системный шрифт нужного размера (с кэшем)."""
    candidates = _MONO_CANDIDATES if mono else _FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # Последний шанс — встроенный шрифт Pillow (без кириллицы, но работает).
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] | None = None,
    padding: int = 6,
    mono: bool = False,
) -> np.ndarray:
    """Нарисовать строку (с поддержкой кириллицы) и вернуть новый BGR-массив.

    ``xy`` — левый верхний угол текста. Если задан ``background``,
    под текстом рисуется полупрозрачная плашка.
    """
    if image is None:
        return image

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(font_size, mono=mono)

    left, top = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((left, top), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Текст не должен вылезать за границу картинки
    left = max(0, min(left, pil_image.width - text_width - padding))
    top = max(0, min(top, pil_image.height - text_height - padding))

    if background is not None:
        draw.rectangle(
            [
                left - padding,
                top - padding,
                left + text_width + padding,
                top + text_height + padding,
            ],
            fill=tuple(int(value) for value in background),
        )

    draw.text((left, top), text, font=font, fill=tuple(int(v) for v in color))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_multiline(
    image: np.ndarray,
    lines: list[str],
    xy: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] | None = (0, 0, 0),
    padding: int = 8,
    line_spacing: int = 6,
    mono: bool = False,
) -> np.ndarray:
    """Нарисовать несколько строк подряд (используется для статуса и OCR)."""
    result = image
    x, y = xy
    for line in lines:
        result = draw_text(
            result,
            line,
            (x, y),
            font_size=font_size,
            color=color,
            background=background,
            padding=padding,
            mono=mono,
        )
        y += font_size + line_spacing + padding * 2
    return result


def draw_caption_panel(
    image: np.ndarray,
    lines: list[str],
    font_size: int = 24,
    margin: int = 12,
    background: tuple[int, int, int] = (28, 28, 28),
    color: tuple[int, int, int] = (235, 235, 235),
    mono: bool = True,
) -> np.ndarray:
    """Дописать под изображением панель с текстом (как в задании ЛР3).

    Задание требует, чтобы распознанный текст был «отражён на выходном
    изображении рядом с исходным текстом» — поэтому текст не просто
    возвращается в интерфейс, а вписывается в картинку снизу.
    """
    if image is None:
        return image
    if not lines:
        lines = [""]

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = _load_font(font_size, mono=mono)
    draw = ImageDraw.Draw(pil_image)

    max_text_width = max(pil_image.width - 2 * margin, 200)
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_line(draw, line, font, max_text_width))

    line_height = int(font_size * 1.35)
    panel_height = line_height * len(wrapped) + 2 * margin
    panel = Image.new("RGB", (pil_image.width, panel_height), background)
    panel_draw = ImageDraw.Draw(panel)

    y = margin / 2
    for line in wrapped:
        panel_draw.text((margin, y), line, font=font, fill=color)
        y += line_height

    result = Image.new("RGB", (pil_image.width, pil_image.height + panel_height))
    result.paste(pil_image, (0, 0))
    result.paste(panel, (0, pil_image.height))
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _wrap_line(draw: ImageDraw.ImageDraw, line: str, font, max_width: int) -> list[str]:
    """Перенос строки по словам под заданную ширину."""
    if not line:
        return [""]
    words = line.split(" ")
    result: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                result.append(current)
            current = word
    result.append(current)
    return result


def draw_badge(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (0, 200, 0),
    font_size: int = 20,
) -> np.ndarray:
    """Плашка-подпись (используется для нумерации контуров)."""
    return draw_text(
        image,
        text,
        xy,
        font_size=font_size,
        color=(255, 255, 255),
        background=color,
        padding=4,
    )


