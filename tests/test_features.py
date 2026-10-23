"""Тесты поиска объекта по ключевым точкам (ЛР2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.core.features import (
    MatchParams,
    available_detectors,
    find_object,
    ratio_test,
    surf_available,
)


def test_detector_registry_contains_sift_and_orb():
    detectors = available_detectors()
    assert "sift" in detectors
    assert "orb" in detectors
    # SURF есть только в сборках, где разрешены патентные алгоритмы
    assert ("surf" in detectors) is surf_available()


@pytest.mark.parametrize("detector", ["sift", "orb"])
def test_object_is_found_on_synthetic_scene(detector, sample_image, scene_image):
    result = find_object(sample_image, scene_image, MatchParams(detector=detector))

    assert result.found is True, result.message
    assert result.match_count > 4
    assert result.inlier_count >= 6
    assert result.image is not None

    # Найденная рамка должна попадать в область, куда мы вставили образец
    assert result.box is not None
    center_x, center_y = result.box.mean(axis=0)
    assert 450 < center_x < 950
    assert 250 < center_y < 650


def test_empty_image_returns_not_found(sample_image):
    result = find_object(sample_image, np.zeros((300, 300, 3), np.uint8))
    assert result.found is False
    assert result.image is not None


def test_none_input_is_handled():
    result = find_object(None, None)
    assert result.found is False
    assert "прочитать" in result.message.lower()


def test_ratio_test_skips_single_candidate_pairs():
    """knnMatch иногда возвращает одну пару — старый код падал на распаковке."""
    good = cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10.0)
    bad = cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=100.0)
    far = cv2.DMatch(_queryIdx=1, _trainIdx=2, _distance=200.0)

    result = ratio_test([[good], [good, far], [bad, bad]], ratio=0.75)
    assert len(result) == 1


def test_ratio_test_zero_distance_is_safe():
    first = cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0)
    second = cv2.DMatch(_queryIdx=0, _trainIdx=1, _distance=0.0)
    assert ratio_test([[first, second]], 0.75) == []


def test_same_image_matches_itself(sample_image):
    result = find_object(sample_image, sample_image, MatchParams(detector="sift"))
    assert result.found is True
    assert result.inlier_count > 20


def test_binarization_mode_does_not_crash(sample_image, scene_image):
    params = MatchParams(detector="sift", binarize=True)
    result = find_object(sample_image, scene_image, params)
    assert result.image is not None


def test_surf_flag_is_consistent():
    # Флаг должен быть булевым и не падать при повторных вызовах
    assert isinstance(surf_available(), bool)
    assert surf_available() == surf_available()
