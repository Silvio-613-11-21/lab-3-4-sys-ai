"""Тесты прикладного слоя: кэш изображений и фоновый обработчик потока."""

from __future__ import annotations

import time

import numpy as np

from src.core.image_io import imwrite_unicode
from src.logic.controller import ImageCache, LiveProcessor


def test_cache_reads_and_returns_same_object(tmp_path, sample_image):
    path = tmp_path / "образец.png"
    imwrite_unicode(path, sample_image)

    cache = ImageCache()
    first = cache.get(str(path))
    second = cache.get(str(path))

    assert first is not None
    assert first is second  # второй раз берём из кэша, без чтения с диска


def test_cache_returns_none_for_missing_path():
    cache = ImageCache()
    assert cache.get(None) is None
    assert cache.get("/нет/такого/файла.png") is None


def test_cache_invalidates_on_file_change(tmp_path):
    path = tmp_path / "кадр.png"
    first = np.zeros((10, 10, 3), np.uint8)
    imwrite_unicode(path, first)

    cache = ImageCache()
    assert cache.get(str(path)).sum() == 0

    second = np.full((10, 10, 3), 255, np.uint8)
    time.sleep(0.01)  # mtime имеет разрешение файловой системы
    imwrite_unicode(path, second)

    refreshed = cache.get(str(path))
    assert refreshed is not None
    assert refreshed.sum() > 0


def test_live_processor_produces_frames():
    frame = np.zeros((60, 80, 3), np.uint8)

    def provider():
        return frame

    def processor(image):
        return image, "кадр обработан"

    live = LiveProcessor(fps=50)
    live.start(provider, processor)
    try:
        deadline = time.time() + 3
        latest = None
        while time.time() < deadline:
            latest = live.latest()
            if latest is not None and latest.sequence >= 2:
                break
            time.sleep(0.02)
    finally:
        live.stop()

    assert latest is not None
    assert latest.data_uri.startswith("data:image/png;base64,")
    assert latest.status == "кадр обработан"
    assert live.is_running is False


def test_live_processor_survives_failing_processor():
    frame = np.zeros((30, 30, 3), np.uint8)

    def broken(_image):
        raise RuntimeError("сломано")

    live = LiveProcessor(fps=50)
    live.start(lambda: frame, broken)
    try:
        time.sleep(0.3)
        assert live.is_running is True
        assert "сломано" in live.error
    finally:
        live.stop()


def test_live_processor_handles_empty_provider():
    live = LiveProcessor(fps=50)
    live.start(lambda: None, lambda image: (image, "ok"))
    try:
        time.sleep(0.2)
        assert live.latest() is None
        assert live.is_running is True
    finally:
        live.stop()
