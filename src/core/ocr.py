"""Распознавание текста через Tesseract OCR (лабораторная работа №3, часть 2).

Что делает модуль:

* ищет установленный Tesseract (Windows/Linux/macOS, переменная окружения
  ``TESSERACT_CMD``, ``PATH``, стандартные каталоги установки);
* проверяет, какие языковые пакеты реально доступны (``rus``, ``eng``);
* распознаёт текст (``pytesseract.image_to_data`` — вместе с координатами
  слов), печатает результат в консоль;
* рисует на выходном изображении рамки вокруг слов **и** панель с
  распознанным текстом — как требует задание («текст должен быть отражён
  на выходном изображении рядом с исходным текстом»).

Старый вариант программы содержал жёстко прописанный путь
``c:\\Program Files\\Tesseract-OCR\\tesseract.exe``. На любой другой машине
(или после переустановки) это ломало всё приложение при импорте модуля.
Теперь путь определяется автоматически, а отсутствие Tesseract — это
сообщение в интерфейсе, а не падение.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from ..config import OCR_MIN_SIDE
from .drawing import draw_caption_panel, draw_text


# --------------------------------------------------------------------------
# Поиск Tesseract
# --------------------------------------------------------------------------
@dataclass
class TesseractStatus:
    """Состояние установки Tesseract."""

    available: bool = False
    path: str | None = None
    version: str | None = None
    languages: list[str] = field(default_factory=list)
    message: str = ""


_WINDOWS_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
]


_STATUS_CACHE: TesseractStatus | None = None


def find_tesseract(force: bool = False) -> TesseractStatus:
    """Найти исполняемый файл Tesseract и узнать его версию/языки.

    Результат кэшируется, повторный вызов не запускает tesseract заново.
    """
    global _STATUS_CACHE

    if not force and _STATUS_CACHE is not None:
        return _STATUS_CACHE

    candidates: list[str] = []

    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)

    # Уже настроенный путь (например, если пользователь задал его вручную)
    current = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if current and current not in ("tesseract", ""):
        candidates.append(current)

    found_in_path = shutil.which("tesseract")
    if found_in_path:
        candidates.append(found_in_path)

    for candidate in _WINDOWS_CANDIDATES:
        candidates.append(candidate)

    # Типовые пути Linux/macOS
    candidates.extend(
        ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
    )

    status = TesseractStatus(message="Tesseract не найден")

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            continue
        pytesseract.pytesseract.tesseract_cmd = str(path)
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception as error:  # noqa: BLE001 - tesseract может не запуститься
            status = TesseractStatus(
                available=False,
                path=str(path),
                message=f"Tesseract найден, но не запускается: {error}",
            )
            continue

        try:
            languages = list(pytesseract.get_languages(config=""))
        except Exception:  # noqa: BLE001
            languages = []

        status = TesseractStatus(
            available=True,
            path=str(path),
            version=version,
            languages=sorted(languages),
            message=f"Tesseract {version} готов",
        )
        break

    _STATUS_CACHE = status
    return status


def reset_cache() -> None:
    """Сбросить кэш (кнопка «Проверить окружение»)."""
    global _STATUS_CACHE
    _STATUS_CACHE = None


def resolve_language(requested: str, available: list[str]) -> tuple[str, str]:
    """Подобрать язык из доступных пакетов.

    Возвращает ``(язык_для_tesseract, примечание)``.
    """
    requested = (requested or "rus+eng").strip()
    if not available:
        return requested, ""

    parts = [part for part in requested.split("+") if part]
    usable = [part for part in parts if part in available]
    missing = [part for part in parts if part not in available]

    note = ""
    if missing:
        note = (
            "Не установлены языковые пакеты: "
            + ", ".join(missing)
            + f". Доступно: {', '.join(available)}"
        )

    if usable:
        if len(usable) == 1 and len(parts) > 1:
            note = (note + " " if note else "") + f"Используется только «{usable[0]}»."
        return "+".join(usable), note

    fallback = "eng" if "eng" in available else available[0]
    note = (
        (note + " " if note else "")
        + f"Запрошенный язык недоступен, используется «{fallback}»."
    )
    return fallback, note


# --------------------------------------------------------------------------
# Распознавание
# --------------------------------------------------------------------------
@dataclass
class OcrResult:
    """Результат распознавания."""

    ok: bool = False
    text: str = ""
    image: np.ndarray | None = None
    words: list[dict] = field(default_factory=list)
    language: str = ""
    message: str = ""
    elapsed_ms: float = 0.0
    average_confidence: float = 0.0


def prepare_for_ocr(image: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Подготовка картинки: увеличение мелких изображений и бинаризация."""
    if image is None:
        return image
    result = image
    height, width = result.shape[:2]
    scale = 1.0
    if min(height, width) < OCR_MIN_SIDE:
        scale = OCR_MIN_SIDE / min(height, width)
        result = cv2.resize(
            result,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    if binarize:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def recognize(
    image: np.ndarray,
    language: str = "rus+eng",
    binarize: bool = True,
    draw_panel: bool = True,
) -> OcrResult:
    """Распознать текст на BGR-изображении.

    Текст печатается в консоль (требование задания) и возвращается в
    :class:`OcrResult`. На изображение наносятся рамки слов и панель с текстом.
    """
    started = time.perf_counter()
    result = OcrResult(language=language)

    if image is None:
        result.message = "Изображение не загружено"
        return result

    status = find_tesseract()
    if not status.available:
        result.message = (
            "Tesseract OCR не найден. Установите его и/или задайте переменную "
            "окружения TESSERACT_CMD (см. README)."
        )
        result.image = image.copy()
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    resolved_language, note = resolve_language(language, status.languages)
    result.language = resolved_language

    prepared = prepare_for_ocr(image, binarize=binarize)

    try:
        # image_to_data даёт слова и их координаты — нужно для рамок
        data = pytesseract.image_to_data(
            prepared, lang=resolved_language, output_type=Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        result.message = "Tesseract не найден по пути, указанному в настройках"
        result.image = image.copy()
        return result
    except Exception as error:  # noqa: BLE001 - пробрасываем текст ошибки в UI
        result.message = f"Ошибка распознавания: {error}"
        result.image = image.copy()
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    words = _extract_words(data, prepared.shape)
    result.words = words
    confidences = [word["confidence"] for word in words if word["confidence"] > 0]
    result.average_confidence = float(np.mean(confidences)) if confidences else 0.0

    text = " ".join(word["text"] for word in words).strip()
    if not text:
        # Иногда Tesseract лучше работает построчно — берём обычный строковый вывод
        try:
            text = pytesseract.image_to_string(prepared, lang=resolved_language).strip()
        except Exception:  # noqa: BLE001
            text = ""
    result.text = text
    result.ok = True

    annotated = prepared.copy()
    for word in words:
        x, y, width, height = word["left"], word["top"], word["width"], word["height"]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 200, 0), 2)

    annotated = draw_text(
        annotated,
        f"Tesseract {status.version} · язык: {resolved_language} · "
        f"слов: {len(words)} · ср. уверенность: {result.average_confidence:.0f}%",
        (12, 12),
        font_size=20,
        color=(255, 255, 255),
        background=(0, 120, 0),
        padding=6,
    )

    if draw_panel:
        lines = text.splitlines() or ["*текст не найден*"]
        if len(lines) > 40:
            lines = lines[:40] + ["… (текст обрезан)"]
        annotated = draw_caption_panel(annotated, lines, font_size=24)

    result.image = annotated
    result.message = note or f"Распознано слов: {len(words)}"
    result.elapsed_ms = (time.perf_counter() - started) * 1000

    # Требование методички: текст выводится в консоль
    print("=" * 60)
    print(f"[OCR] Язык: {resolved_language}")
    print(result.text if result.text else "*текст не найден*")
    print("=" * 60)

    return result


def _extract_words(data: dict, shape) -> list[dict]:
    """Преобразовать вывод ``image_to_data`` в список слов с координатами."""
    words = []
    count = len(data.get("text", []))
    for index in range(count):
        text = (data["text"][index] or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue
        words.append(
            {
                "text": text,
                "confidence": confidence,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
                "line": int(data["line_num"][index]),
                "block": int(data["block_num"][index]),
            }
        )
    return words


def language_hint() -> str:
    """Короткая строка для интерфейса: что доступно в системе."""
    status = find_tesseract()
    if not status.available:
        return "Tesseract: не найден"
    languages = ", ".join(status.languages) if status.languages else "нет данных"
    return f"Tesseract {status.version}: {languages}"
