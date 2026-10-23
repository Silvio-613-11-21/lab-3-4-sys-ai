"""Поиск объекта на изображении по ключевым точкам (лабораторная работа №2).

Пайплайн:

1. перевод изображений в оттенки серого (+ опционально CLAHE/бинаризация);
2. поиск ключевых точек и дескрипторов (SIFT / SURF / ORB);
3. сопоставление дескрипторов (`knnMatch`, k=2) и ratio-тест Лоу;
4. оценка геометрии через гомографию (RANSAC) — она отсекает случайные
   совпадения и позволяет нарисовать рамку найденного объекта;
5. отрисовка результата.

Модуль не зависит от интерфейса: на входе — массивы OpenCV, на выходе —
датакласс :class:`MatchResult`. Такой код можно тестировать без GUI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import (
    DEFAULT_DETECTOR,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_INLIERS,
    DEFAULT_RANSAC_REPROJ,
    DEFAULT_RATIO,
)
from .drawing import draw_text


# --------------------------------------------------------------------------
# Параметры и результат
# --------------------------------------------------------------------------
@dataclass
class MatchParams:
    """Настройки распознавания, которые пользователь меняет в интерфейсе."""

    detector: str = DEFAULT_DETECTOR          # "sift" | "surf" | "orb"
    ratio: float = DEFAULT_RATIO              # порог ratio-теста Лоу
    max_features: int = DEFAULT_MAX_FEATURES
    binarize: bool = False                    # бинаризация Оцу перед поиском
    use_clahe: bool = True                    # выравнивание контраста
    show_markers: bool = True                 # рисовать ключевые точки
    unit_markers: bool = False                # «соединять подобные маркеры»
    ransac_reproj: float = DEFAULT_RANSAC_REPROJ
    min_inliers: int = DEFAULT_MIN_INLIERS


@dataclass
class MatchResult:
    """Результат сопоставления образца и сцены."""

    found: bool = False
    message: str = ""
    details: list[str] = field(default_factory=list)
    match_count: int = 0
    inlier_count: int = 0
    keypoints_sample: int = 0
    keypoints_scene: int = 0
    elapsed_ms: float = 0.0
    image: np.ndarray | None = None            # картинка с линиями совпадений
    homography: np.ndarray | None = None
    box: np.ndarray | None = None              # 4 угла объекта на сцене
    detector_name: str = ""


DETECTOR_NAMES = {
    "sift": "SIFT",
    "surf": "SURF",
    "orb": "ORB",
}


# --------------------------------------------------------------------------
# Создание детектора
# --------------------------------------------------------------------------
_SURF_AVAILABLE: bool | None = None


def surf_available() -> bool:
    """Реально ли можно создать SURF в текущей сборке OpenCV.

    Мало проверить наличие ``cv2.xfeatures2d.SURF_create``: в предсобранных
    колёсах OpenCV 5 функция присутствует, но выбрасывает
    ``error: This algorithm is patented and is excluded in this configuration``.
    Поэтому пробуем создать детектор один раз и запоминаем результат.
    """
    global _SURF_AVAILABLE
    if _SURF_AVAILABLE is not None:
        return _SURF_AVAILABLE

    try:
        if not hasattr(cv2, "xfeatures2d"):
            _SURF_AVAILABLE = False
        else:
            cv2.xfeatures2d.SURF_create(hessianThreshold=400.0)
            _SURF_AVAILABLE = True
    except Exception:  # noqa: BLE001 - cv2.error и любые сборочные сюрпризы
        _SURF_AVAILABLE = False
    return _SURF_AVAILABLE


def available_detectors() -> list[str]:
    """Список доступных в текущей сборке OpenCV детекторов."""
    detectors = ["sift", "orb"]
    if surf_available():
        detectors.insert(1, "surf")
    return detectors


def _create_detector(kind: str, max_features: int):
    """Вернуть (детектор, тип нормы дескрипторов)."""
    kind = (kind or DEFAULT_DETECTOR).lower()

    if kind == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
        return detector, cv2.NORM_HAMMING

    if kind == "surf":
        if not surf_available():
            # Мягкая деградация: вместо падения используем SIFT.
            detector = cv2.SIFT_create(nfeatures=max_features)
            return detector, cv2.NORM_L2
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400.0)
        return detector, cv2.NORM_L2

    # По умолчанию — SIFT
    detector = cv2.SIFT_create(nfeatures=max_features)
    return detector, cv2.NORM_L2


def _preprocess(image: np.ndarray, params: MatchParams) -> np.ndarray:
    """Серое изображение + опциональные CLAHE/бинаризация."""
    if image is None:
        return image

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if params.binarize:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif params.use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def _match_descriptors(kind: str, des1: np.ndarray, des2: np.ndarray, norm: int):
    """knnMatch (k=2) с FLANN для вещественных дескрипторов и BF для ORB."""
    if norm == cv2.NORM_HAMMING:
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        return matcher.knnMatch(des1, des2, k=2)

    # SIFT/SURF дают float-дескрипторы -> FLANN KDTree.
    # Если сборка OpenCV не принимает словарь параметров — откатываемся на BF.
    try:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return matcher.knnMatch(des1, des2, k=2)
    except Exception:  # noqa: BLE001
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        return matcher.knnMatch(des1, des2, k=2)


def ratio_test(matches, ratio: float) -> list:
    """Ratio-тест Лоу: оставляем пару, если она заметно лучше второй.

    ВАЖНО: ``knnMatch`` может вернуть для точки только одного соседа
    (если в другом изображении мало дескрипторов). Старый код делал
    ``for m, n in matches`` и падал с ``ValueError`` на такой паре —
    здесь такие элементы просто пропускаются.
    """
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if second.distance <= 0:
            continue
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def find_object(
    sample: np.ndarray,
    scene: np.ndarray,
    params: MatchParams | None = None,
) -> MatchResult:
    """Найти образец ``sample`` на изображении ``scene``.

    :param sample: изображение-образец (что ищем), BGR.
    :param scene: изображение-сцена (где ищем), BGR.
    """
    params = params or MatchParams()
    started = time.perf_counter()

    result = MatchResult(detector_name=DETECTOR_NAMES.get(params.detector, params.detector))

    if sample is None or scene is None:
        result.message = "Не удалось прочитать изображение"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    gray_sample = _preprocess(sample, params)
    gray_scene = _preprocess(scene, params)

    detector, norm = _create_detector(params.detector, params.max_features)
    kp1, des1 = detector.detectAndCompute(gray_sample, None)
    kp2, des2 = detector.detectAndCompute(gray_scene, None)

    result.keypoints_sample = len(kp1)
    result.keypoints_scene = len(kp2)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        result.message = (
            "Недостаточно ключевых точек. Попробуйте другое изображение "
            "или включите бинаризацию."
        )
        result.image = _compose_pair(sample, scene, params)
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    try:
        matches = _match_descriptors(params.detector, des1, des2, norm)
    except cv2.error as error:
        result.message = f"Ошибка сопоставления дескрипторов: {error}"
        result.image = _compose_pair(sample, scene, params)
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    good = ratio_test(matches, params.ratio)
    result.match_count = len(good)

    # --- Проверка геометрии ------------------------------------------------
    homography = None
    mask = None
    inliers = 0

    if len(good) >= 4:
        source_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            homography, mask = cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                params.ransac_reproj,
            )
        except cv2.error:
            homography, mask = None, None
        if mask is not None:
            inliers = int(mask.sum())

    result.inlier_count = inliers
    result.homography = homography

    # --- Рисуем ------------------------------------------------------------
    match_image = _draw_matches(sample, scene, kp1, kp2, good, mask, params)

    box = None
    if homography is not None:
        height, width = sample.shape[:2]
        corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(corners, homography)
        except cv2.error:
            projected = None
        if projected is not None and np.all(np.isfinite(projected)):
            box = projected.reshape(-1, 2)
            if _box_is_sane(box, scene.shape):
                offset = np.array([sample.shape[1], 0], dtype=np.float32)
                shifted = (box + offset).astype(np.int32)
                cv2.polylines(match_image, [shifted], True, (0, 255, 0), 3)
            else:
                box = None

    result.box = box
    result.found = bool(homography is not None and inliers >= params.min_inliers)

    # --- Текст статуса -----------------------------------------------------
    if result.found:
        result.message = f"ОБЪЕКТ НАЙДЕН: {inliers} устойчивых совпадений"
        color = (0, 255, 0)
    elif homography is not None:
        result.message = (
            f"Объект не подтверждён: {inliers} совпадений "
            f"(нужно ≥ {params.min_inliers})"
        )
        color = (0, 200, 255)
    else:
        result.message = f"Объект не найден: {len(good)} совпадений (мало для гомографии)"
        color = (0, 0, 255)

    match_image = draw_text(
        match_image,
        result.message,
        (12, 12),
        font_size=24,
        color=(255, 255, 255),
        background=color,
        padding=8,
    )

    result.image = match_image
    result.details = [
        f"Детектор: {result.detector_name}",
        f"Ключевых точек: образец — {result.keypoints_sample}, "
        f"сцена — {result.keypoints_scene}",
        f"Совпадений после ratio-теста ({params.ratio:.2f}): {result.match_count}",
        f"Устойчивых совпадений (RANSAC): {result.inlier_count}",
    ]
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    if result.found and box is not None:
        center_x = int(box[:, 0].mean())
        center_y = int(box[:, 1].mean())
        result.details.append(f"Центр объекта на сцене: ({center_x}, {center_y})")
    return result


# --------------------------------------------------------------------------
# Вспомогательная отрисовка
# --------------------------------------------------------------------------
def _draw_matches(sample, scene, kp1, kp2, good, mask, params: MatchParams):
    """``drawMatches`` + аккуратные флаги (см. настройки интерфейса)."""
    if params.show_markers and not params.unit_markers:
        flags = cv2.DrawMatchesFlags_DEFAULT
    elif params.show_markers and params.unit_markers:
        flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    else:
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    mask_list = None
    if mask is not None:
        mask_list = mask.ravel().astype(np.uint8).tolist()

    # drawMatches создаёт полотно высотой max(h1, h2) и заполняет «пустоту»
    # чёрным. Чтобы картинка выглядела аккуратно, дополняем меньшее
    # изображение нейтральным фоном (координаты ключевых точек не меняются).
    sample_padded = _pad_to_height(sample, max(sample.shape[0], scene.shape[0]))
    scene_padded = _pad_to_height(scene, sample_padded.shape[0])

    try:
        return cv2.drawMatches(
            sample_padded,
            kp1,
            scene_padded,
            kp2,
            good,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 100, 0),
            matchesMask=mask_list,
            flags=flags,
        )
    except cv2.error:
        # Совсем экзотический случай (разные типы/каналы) — отдаём пару картинок.
        return _compose_pair(sample, scene, params)


def _pad_to_height(image: np.ndarray, height: int, value: int = 45) -> np.ndarray:
    """Дополнить изображение снизу однотонным фоном до нужной высоты."""
    if image.shape[0] >= height:
        return image
    pad = height - image.shape[0]
    return cv2.copyMakeBorder(
        image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(value, value, value)
    )


def _compose_pair(sample: np.ndarray, scene: np.ndarray, params: MatchParams) -> np.ndarray:
    """Склеить образец и сцену рядом, если рисовать совпадения нечего."""
    height = max(sample.shape[0], scene.shape[0])
    canvas = np.zeros((height, sample.shape[1] + scene.shape[1], 3), dtype=np.uint8)
    canvas[: sample.shape[0], : sample.shape[1]] = sample
    canvas[: scene.shape[0], sample.shape[1] :] = scene
    return canvas


def _box_is_sane(box: np.ndarray, scene_shape) -> bool:
    """Отсечь «взрывные» гомографии (рамка на пол-экрана и больше)."""
    height, width = scene_shape[:2]
    tolerance = 0.5
    return bool(
        np.all(box[:, 0] > -width * tolerance)
        and np.all(box[:, 0] < width * (1 + tolerance))
        and np.all(box[:, 1] > -height * tolerance)
        and np.all(box[:, 1] < height * (1 + tolerance))
        and cv2.contourArea(box.astype(np.float32)) < width * height * 0.98
    )
