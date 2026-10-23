"""Тесты контурного анализа (ЛР3, часть 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.contours import (
    ContourParams,
    analyze_contours,
    binarize,
    compare_contours,
    contour_to_ev,
    filter_contours,
    normalized_scalar_product,
    resample_contour,
)


# --------------------------------------------------------------------------
# Уравнивание контуров
# --------------------------------------------------------------------------
def test_resample_returns_requested_number_of_points():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    for count in (8, 16, 64, 128):
        assert len(resample_contour(square, count)) == count


def test_resample_is_uniform_in_arc_length():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    points = resample_contour(square, 8)
    lengths = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    assert np.allclose(lengths, lengths.mean(), atol=1e-3)


def test_resample_survives_degenerate_input():
    assert len(resample_contour(np.array([[5, 5]], np.float32), 10)) == 10


def test_elementary_vectors_of_closed_contour_sum_to_zero():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    vectors = contour_to_ev(resample_contour(square, 64))
    assert abs(vectors.sum()) < 1e-6
    assert len(vectors) == 64


# --------------------------------------------------------------------------
# NSP и ВКФ
# --------------------------------------------------------------------------
def test_nsp_of_identical_contours_is_one(text_image):
    params = ContourParams(min_area_ratio=0.0005)
    result = analyze_contours(text_image, params)
    reference = result.contours[0].equalized

    score, shift, angle = compare_contours(reference, reference)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert shift == 0
    assert abs(angle) < 1e-6


def test_nsp_is_invariant_to_cyclic_shift(text_image):
    params = ContourParams(min_area_ratio=0.0005)
    result = analyze_contours(text_image, params)
    reference = result.contours[0].equalized
    shifted = np.roll(reference, 13, axis=0)

    score, best_shift, _ = compare_contours(reference, shifted)
    # 1.0 не достигается ровно из-за повторного ресэмплинга сдвинутого контура
    assert score > 0.99
    assert best_shift != 0


def test_nsp_is_scale_invariant():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    big_square = square * 3.0
    score, _, _ = compare_contours(resample_contour(square, 64),
                                   resample_contour(big_square, 64))
    assert score == pytest.approx(1.0, abs=1e-4)


def test_nsp_detects_rotation_angle():
    import cv2

    square = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], np.float32)
    matrix = cv2.getRotationMatrix2D((100, 100), 30, 1.0)
    rotated = cv2.transform(square.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    score, _, angle = compare_contours(resample_contour(square, 64),
                                       resample_contour(rotated, 64))
    assert score > 0.95
    # угол поворота оценивается с точностью «примерно»
    assert abs(abs(angle) - 30) < 12 or abs(abs(angle) - 60) < 12 or abs(abs(angle) - 90) < 12


def test_nsp_between_different_shapes_is_lower_than_between_same():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.stack([100 + 60 * np.cos(angles), 100 + 60 * np.sin(angles)], axis=1)
    same, _, _ = compare_contours(resample_contour(square, 64), resample_contour(square, 64))
    different, _, _ = compare_contours(resample_contour(square, 64), resample_contour(circle, 64))
    assert same > different


def test_normalized_scalar_product_bounds():
    a = np.array([1 + 1j, 2 - 1j])
    assert normalized_scalar_product(a, a) == pytest.approx(1.0)
    assert -1.0001 <= normalized_scalar_product(a, -a) <= 1.0001


# --------------------------------------------------------------------------
# Пайплайн
# --------------------------------------------------------------------------
def test_pipeline_returns_all_steps(text_image):
    result = analyze_contours(text_image)

    keys = [step.key for step in result.steps]
    assert keys == [
        "source", "gray", "blur", "binary", "morph",
        "contours", "filtered", "equalized", "final",
    ]
    for step in result.steps:
        assert step.image is not None
        assert step.image.ndim == 3


def test_pipeline_finds_shapes_on_white_background(text_image):
    result = analyze_contours(text_image, ContourParams(min_area_ratio=0.0005))
    assert len(result.contours) >= 3
    assert result.image is not None
    biggest_area = max(info.area for info in result.contours)
    assert biggest_area > 10_000


def test_pipeline_with_reference_image(sample_image, text_image):
    result = analyze_contours(text_image, sample_image)
    assert result.reference is not None
    assert all(info.score is not None for info in result.contours)


def test_pipeline_handles_none():
    result = analyze_contours(None)
    assert result.steps == []
    assert "прочитать" in result.message.lower()


def test_auto_invert_polarity():
    """Тёмные объекты на светлом фоне не должны давать один контур-рамку."""
    params = ContourParams()
    gray = np.full((200, 200), 250, np.uint8)
    gray[50:150, 50:150] = 10
    binary, _, inverted = binarize(gray, params)

    assert inverted is True
    assert binary[100, 100] == 255   # объект стал «белым»
    assert binary[5, 5] == 0         # фон стал «чёрным»


def test_filter_removes_tiny_contours():
    big = np.array([[[10, 10]], [[200, 10]], [[200, 200]], [[10, 200]]], np.int32)
    tiny = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]], np.int32)

    filtered = filter_contours([big, tiny], ContourParams(min_area_ratio=0.001), 300 * 300)
    assert len(filtered) == 1
