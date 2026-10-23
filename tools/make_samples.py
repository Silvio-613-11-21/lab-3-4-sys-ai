"""Генератор демонстрационных изображений в папке ``samples/``.

Скрипт создаёт три картинки, на которых удобно показывать работу программы
и делать скриншоты для отчёта:

* ``samples/object_card.png``  — образец (визитка), которую будем искать;
* ``samples/scene_desk.png``   — сцена: та же визитка, повёрнутая и уменьшенная,
  лежит на столе вместе с другими предметами;
* ``samples/text_page.png``    — страница с русским и английским текстом
  для проверки OCR (ЛР3, часть 2).

Запуск::

    python tools/make_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.drawing import draw_text  # noqa: E402
from src.core.image_io import imwrite_unicode  # noqa: E402


def make_card() -> np.ndarray:
    """Визитка с надписями и рисунком (образец для поиска)."""
    card = np.full((300, 480, 3), 250, np.uint8)
    card = cv2.rectangle(card, (0, 0), (479, 299), (250, 250, 250), -1)
    card = cv2.rectangle(card, (0, 0), (479, 70), (120, 60, 30), -1)

    card = draw_text(card, "СИСТЕМЫ ИИ", (18, 14), font_size=30, color=(255, 255, 255))
    card = draw_text(card, "Лабораторная работа №2", (18, 92), font_size=22,
                     color=(40, 40, 40))
    card = draw_text(card, "Поиск объекта: SIFT + RANSAC", (18, 130), font_size=18,
                     color=(90, 90, 90))

    # «Логотип»: круг + квадрат + треугольник
    cv2.circle(card, (410, 210), 46, (60, 180, 75), -1, cv2.LINE_AA)
    cv2.rectangle(card, (330, 170), (390, 230), (200, 120, 40), -1, cv2.LINE_AA)
    triangle = np.array([[330, 275], [390, 275], [360, 235]], np.int32)
    cv2.fillPoly(card, [triangle], (50, 50, 200), cv2.LINE_AA)

    card = draw_text(card, "ivanov@example.ru", (18, 246), font_size=18,
                     color=(30, 30, 160))
    return card


def make_scene(card: np.ndarray) -> np.ndarray:
    """Стол с предметами, среди которых лежит визитка."""
    height, width = 900, 1400
    scene = np.zeros((height, width, 3), np.uint8)

    # Деревянный стол: вертикальный градиент + волокна
    for row in range(height):
        shade = int(70 + 45 * row / height)
        scene[row, :] = (shade - 25, shade - 10, shade + 10)
    rng = np.random.default_rng(7)
    fibers = rng.normal(0, 8, (height // 6, width))
    fibers = cv2.resize(fibers, (width, height), interpolation=cv2.INTER_LINEAR)
    scene = np.clip(scene.astype(np.float32) + fibers[:, :, None], 0, 255).astype(np.uint8)

    # Посторонние предметы
    cv2.rectangle(scene, (80, 120), (520, 260), (40, 90, 170), -1)
    cv2.rectangle(scene, (80, 120), (520, 260), (20, 50, 110), 4)
    cv2.circle(scene, (1180, 160), 110, (60, 60, 60), -1)
    cv2.circle(scene, (1180, 160), 70, (120, 120, 120), -1)
    cv2.rectangle(scene, (950, 620), (1330, 850), (90, 140, 90), -1)

    # Визитка: поворот + перспектива + уменьшение
    source = np.float32([[0, 0], [479, 0], [479, 299], [0, 299]])
    target = np.float32([[600, 430], [980, 380], [1010, 640], [620, 700]])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(card, matrix, (width, height))
    mask = cv2.warpPerspective(
        np.full(card.shape[:2], 255, np.uint8), matrix, (width, height)
    )

    # Лёгкая тень под визиткой
    shadow = cv2.GaussianBlur(mask, (31, 31), 0)
    scene = np.clip(scene.astype(np.float32) * (1 - 0.35 * (shadow / 255.0))[:, :, None], 0, 255)

    scene = scene.astype(np.uint8)
    scene[mask > 0] = warped[mask > 0]

    # Небольшой шум сенсора
    scene = np.clip(
        scene.astype(np.float32) + rng.normal(0, 4, scene.shape), 0, 255
    ).astype(np.uint8)
    return scene


def make_text_page() -> np.ndarray:
    """Страница с русским и английским текстом для OCR."""
    page = np.full((1100, 900, 3), 252, np.uint8)

    page = draw_text(page, "Лабораторная работа №3", (60, 50), font_size=34,
                     color=(20, 20, 20))
    page = draw_text(page, "Контурный анализ и Tesseract OCR", (60, 100), font_size=28,
                     color=(20, 20, 20))

    english = [
        "Contour analysis assumes that the shape of an object",
        "contains all necessary information about it. The contour",
        "is described by a sequence of elementary vectors and",
        "compared with normalized scalar product (NSP).",
    ]
    russian = [
        "Контурный анализ позволяет сравнивать объекты независимо",
        "от их положения, поворота и масштаба. Для этого контуры",
        "приводятся к одинаковой длине (уравнивание), а затем",
        "сравниваются по нормированному скалярному произведению.",
    ]

    y = 180
    for line in english:
        page = draw_text(page, line, (60, y), font_size=24, color=(25, 25, 25))
        y += 42
    y += 24
    for line in russian:
        page = draw_text(page, line, (60, y), font_size=24, color=(25, 25, 25))
        y += 42

    # Таблица и фигуры — материал для контурного анализа
    cv2.rectangle(page, (60, 560), (400, 760), (0, 0, 0), 3)
    cv2.line(page, (60, 660), (400, 660), (0, 0, 0), 2)
    cv2.line(page, (230, 560), (230, 760), (0, 0, 0), 2)
    page = draw_text(page, "Площадь", (80, 600), font_size=22, color=(20, 20, 20))
    page = draw_text(page, "Периметр", (250, 600), font_size=22, color=(20, 20, 20))

    cv2.circle(page, (620, 660), 110, (0, 0, 0), 3)
    cv2.rectangle(page, (520, 830), (720, 990), (0, 0, 0), 3)
    triangle = np.array([[800, 990], [880, 830], [760, 830]], np.int32)
    cv2.polylines(page, [triangle], True, (0, 0, 0), 3)

    return page


def main() -> None:
    folder = ROOT / "samples"
    folder.mkdir(exist_ok=True)

    card = make_card()
    scene = make_scene(card)
    page = make_text_page()

    files = [
        (folder / "object_card.png", card),
        (folder / "scene_desk.png", scene),
        (folder / "text_page.png", page),
    ]
    for path, image in files:
        ok = imwrite_unicode(path, image)
        print(f"{'OK ' if ok else 'ERR'} {path} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
