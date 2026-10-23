"""Тесты модуля OCR (ЛР3, часть 2).

Tesseract — внешняя программа, которой может не быть на машине, поэтому
тесты не требуют её установки: проверяются поиск, подбор языка и обработка
результата (с подменой вывода Tesseract на синтетический).
"""

from __future__ import annotations

import numpy as np

from src.core import ocr
from src.core.ocr import (
    OcrResult,
    TesseractStatus,
    find_tesseract,
    prepare_for_ocr,
    recognize,
    resolve_language,
)


def test_find_tesseract_never_raises():
    status = find_tesseract(force=True)
    assert isinstance(status, TesseractStatus)
    assert isinstance(status.available, bool)
    assert status.message


def test_resolve_language_keeps_available_parts():
    language, note = resolve_language("rus+eng", ["eng", "osd", "rus"])
    assert language == "rus+eng"
    assert note == ""


def test_resolve_language_drops_missing_parts():
    language, note = resolve_language("rus+eng", ["eng"])
    assert language == "eng"
    assert "rus" in note


def test_resolve_language_falls_back_to_first_available():
    language, note = resolve_language("rus+eng", ["deu"])
    assert language == "deu"
    assert note


def test_prepare_for_ocr_upscales_small_images():
    small = np.full((100, 120, 3), 255, np.uint8)
    prepared = prepare_for_ocr(small, binarize=True)
    assert min(prepared.shape[:2]) >= ocr.OCR_MIN_SIDE


def test_recognize_without_tesseract_returns_message(text_image, monkeypatch):
    monkeypatch.setattr(
        ocr,
        "find_tesseract",
        lambda force=False: TesseractStatus(available=False, message="Tesseract не найден"),
    )
    result = recognize(text_image)

    assert isinstance(result, OcrResult)
    assert result.ok is False
    assert "Tesseract" in result.message
    assert result.image is not None  # исходная картинка не теряется


def test_recognize_full_path_with_mocked_tesseract(text_image, monkeypatch, capsys):
    """Полный путь: подменяем вывод pytesseract, проверяем текст и отрисовку."""
    monkeypatch.setattr(
        ocr,
        "find_tesseract",
        lambda force=False: TesseractStatus(
            available=True, path="/usr/bin/tesseract", version="5.3.0",
            languages=["eng", "rus"], message="ok",
        ),
    )

    fake_data = {
        "text": ["", "Hello", "World", "", "Привет"],
        "conf": ["-1", "95.5", "91.0", "-1", "88.0"],
        "left": [0, 10, 120, 0, 10],
        "top": [0, 10, 10, 0, 50],
        "width": [0, 100, 110, 0, 120],
        "height": [0, 40, 40, 0, 40],
        "line_num": [0, 0, 0, 0, 1],
        "block_num": [0, 1, 1, 0, 2],
    }
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda *a, **k: fake_data)

    result = recognize(text_image, language="rus+eng")

    assert result.ok is True
    assert result.text == "Hello World Привет"
    assert len(result.words) == 3
    assert result.average_confidence > 80
    assert result.image is not None
    assert result.image.shape[0] > text_image.shape[0]  # добавлена панель с текстом

    # Требование задания: текст печатается в консоль
    captured = capsys.readouterr()
    assert "Hello World Привет" in captured.out


def test_recognize_reports_tesseract_errors(text_image, monkeypatch):
    monkeypatch.setattr(
        ocr,
        "find_tesseract",
        lambda force=False: TesseractStatus(available=True, version="5", languages=["eng"]),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("нет языкового пакета")

    monkeypatch.setattr(ocr.pytesseract, "image_to_data", boom)

    result = recognize(text_image, language="rus+eng")
    assert result.ok is False
    assert "нет языкового пакета" in result.message


def test_recognize_rejects_empty_image():
    result = recognize(None)
    assert result.ok is False
    assert result.image is None
