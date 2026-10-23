"""Общие фикстуры: синтетические изображения для тестов.

Все картинки генерируются «на лету» через OpenCV/NumPy — в репозитории не
нужно хранить бинарные файлы, а тесты работают одинаково на любой машине.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest


def make_sample_image() -> np.ndarray:
    """Изображение-образец с характерным рисунком."""
    sample = np.full((220, 300, 3), 245, np.uint8)
    cv2.rectangle(sample, (10, 10), (290, 210), (30, 60, 200), -1)
    cv2.circle(sample, (80, 80), 40, (250, 240, 20), -1)
    cv2.rectangle(sample, (150, 120), (270, 190), (20, 200, 60), -1)
    cv2.putText(sample, "LAB2", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.line(sample, (0, 0), (300, 220), (255, 255, 255), 2)
    return sample


def make_scene_with_sample(sample: np.ndarray, angle: float = 25.0, scale: float = 0.85,
                           offset: tuple[int, int] = (500, 300)) -> np.ndarray:
    """Сцена: шумный фон + повёрнутый и масштабированный образец."""
    rng = np.random.default_rng(0)
    scene = np.full((720, 1100, 3), 130, np.uint8)
    noise = rng.normal(0, 18, (720, 1100, 3))
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    matrix = cv2.getRotationMatrix2D((sample.shape[1] / 2, sample.shape[0] / 2), angle, scale)
    warped = cv2.warpAffine(sample, matrix, (400, 320), borderValue=(130, 130, 130))
    mask = cv2.warpAffine(
        np.full(sample.shape[:2], 255, np.uint8), matrix, (400, 320)
    )
    x, y = offset
    region = scene[y : y + 320, x : x + 400]
    region[mask > 0] = warped[mask > 0]

    cv2.putText(scene, "FIND ME", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.rectangle(scene, (950, 50), (1060, 160), (0, 0, 0), -1)
    return scene


def make_text_image() -> np.ndarray:
    """«Страница» с текстом и геометрическими фигурами."""
    image = np.full((600, 900, 3), 255, np.uint8)
    cv2.putText(image, "HELLO WORLD", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    cv2.putText(image, "Tesseract OCR", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    cv2.rectangle(image, (60, 250), (260, 420), (0, 0, 0), -1)
    cv2.circle(image, (450, 340), 90, (0, 0, 0), -1)
    cv2.ellipse(image, (720, 340), (110, 70), 30, 0, 360, (0, 0, 0), -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


@pytest.fixture
def sample_image() -> np.ndarray:
    return make_sample_image()


@pytest.fixture
def scene_image(sample_image) -> np.ndarray:
    return make_scene_with_sample(sample_image)


@pytest.fixture
def text_image() -> np.ndarray:
    return make_text_image()
