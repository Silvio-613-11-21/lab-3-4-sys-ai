"""Генератор примеров работы программы (готовые картинки для отчёта).

Скрипт прогоняет те же функции, что и интерфейс, на демонстрационных
изображениях из ``samples/`` и складывает результаты в ``samples/examples/``:

* ``lab2_sift.png``          — поиск объекта (SIFT + RANSAC);
* ``lab2_orb.png``           — то же для ORB (для сравнения детекторов);
* ``lab3_contours.png``      — контурный анализ (итоговый шаг);
* ``lab3_equalization.png``  — шаг «уравнивание контуров».

Запуск::

    python tools/make_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.contours import ContourParams, analyze_contours  # noqa: E402
from src.core.features import MatchParams, find_object  # noqa: E402
from src.core.image_io import imread_unicode, imwrite_unicode  # noqa: E402


def save(folder: Path, name: str, image: np.ndarray | None, note: str) -> None:
    if image is None:
        print(f"ERR {name}: пустая картинка")
        return
    path = folder / name
    ok = imwrite_unicode(path, image)
    print(f"{'OK ' if ok else 'ERR'} {path} ({note})")


def main() -> None:
    samples = ROOT / "samples"
    folder = samples / "examples"
    folder.mkdir(parents=True, exist_ok=True)

    card = imread_unicode(samples / "object_card.png")
    scene = imread_unicode(samples / "scene_desk.png")
    page = imread_unicode(samples / "text_page.png")

    if card is not None and scene is not None:
        for detector in ("sift", "orb"):
            result = find_object(card, scene, MatchParams(detector=detector))
            save(
                folder,
                f"lab2_{detector}.png",
                result.image,
                f"{result.message}; inliers={result.inlier_count}; "
                f"{result.elapsed_ms:.0f} мс",
            )

    if page is not None:
        result = analyze_contours(page, ContourParams(min_area_ratio=0.0002))
        save(folder, "lab3_contours.png", result.image, result.message)
        equalized = result.step("equalized")
        if equalized is not None:
            save(folder, "lab3_equalization.png", equalized.image, equalized.title)


if __name__ == "__main__":
    main()
