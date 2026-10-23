"""Тесты чтения/записи изображений и подготовки превью."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.image_io import (
    bgr_to_pil,
    fit_within,
    imread_unicode,
    imwrite_unicode,
    pil_to_bgr,
    to_data_uri,
)


def test_roundtrip_with_cyrillic_path(tmp_path, sample_image):
    """Главная регрессия: кириллица в пути не должна «ломать» cv2.imread."""
    folder = tmp_path / "Лабораторная 2"
    folder.mkdir()
    path = folder / "образец объекта.png"

    assert imwrite_unicode(path, sample_image) is True
    restored = imread_unicode(path)

    assert restored is not None
    assert restored.shape == sample_image.shape
    assert np.array_equal(restored, sample_image)


def test_read_missing_file_returns_none(tmp_path):
    assert imread_unicode(tmp_path / "нет-такого-файла.png") is None


def test_read_broken_file_returns_none(tmp_path):
    path = tmp_path / "битый.png"
    path.write_bytes(b"not an image")
    assert imread_unicode(path) is None


def test_data_uri_is_png_base64(sample_image):
    uri = to_data_uri(sample_image)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 100


def test_data_uri_downscales_large_image():
    big = np.zeros((2000, 3000, 3), np.uint8)
    uri = to_data_uri(big, max_side=500)
    # уменьшенное изображение должно кодироваться заметно короче исходного
    assert len(uri) < 200_000


def test_fit_within_keeps_aspect_ratio():
    image = np.zeros((400, 800, 3), np.uint8)
    resized = fit_within(image, 400, 400)
    assert resized.shape[1] == 400
    assert resized.shape[0] == 200


def test_pil_conversion_roundtrip(sample_image):
    restored = pil_to_bgr(bgr_to_pil(sample_image))
    assert np.array_equal(restored, sample_image)


@pytest.mark.parametrize("path_suffix", ["a.png", "a.jpg"])
def test_write_creates_parent_directories(tmp_path, sample_image, path_suffix):
    path = tmp_path / "вложенная" / "папка" / path_suffix
    assert imwrite_unicode(path, sample_image) is True
    assert path.is_file()
