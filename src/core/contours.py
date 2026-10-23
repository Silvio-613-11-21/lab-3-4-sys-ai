"""Контурный анализ (лабораторная работа №3, часть 1).

Реализован полный пайплайн из задания:

1. загрузка изображения;
2. перевод в оттенки серого;
3. размытие (подавление высокочастотного шума);
4. бинаризация (Оцу или адаптивная);
5. удаление шумов морфологией (открытие + закрытие);
6. поиск контуров ``cv2.findContours``;
7. фильтрация контуров по площади / периметру / соотношению сторон;
8. **уравнивание** контуров: ресэмплинг до фиксированной длины ``k``;
9. сравнение контуров по нормированному скалярному произведению (NSP)
   с перебором циклических сдвигов (ВКФ) — то, что описано в теории;
10. отрисовка рамок и подписей.

Каждый шаг возвращается отдельной картинкой (:class:`ContourStep`),
чтобы интерфейс мог показать «визуальное сопровождение промежуточных
результатов обработки».
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import (
    DEFAULT_ADAPTIVE_BLOCK,
    DEFAULT_ADAPTIVE_C,
    DEFAULT_APPROX_EPS,
    DEFAULT_BLUR_KERNEL,
    DEFAULT_MAX_AREA_RATIO,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_RESAMPLE_POINTS,
)
from .drawing import draw_badge, draw_text, draw_multiline


# --------------------------------------------------------------------------
# Параметры и структуры данных
# --------------------------------------------------------------------------
@dataclass
class ContourParams:
    """Настройки контурного анализа."""

    blur_kernel: int = DEFAULT_BLUR_KERNEL
    adaptive: bool = False            # адаптивный порог вместо Оцу
    adaptive_block: int = DEFAULT_ADAPTIVE_BLOCK
    adaptive_c: int = DEFAULT_ADAPTIVE_C
    invert: bool = False              # объект светлый на тёмном фоне
    auto_invert: bool = True           # автоматически подобрать полярность
    morph_kernel: int = DEFAULT_MORPH_KERNEL
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO
    min_perimeter: float = 20.0
    approx_eps: float = DEFAULT_APPROX_EPS
    resample_points: int = DEFAULT_RESAMPLE_POINTS
    max_contours: int = 12


@dataclass
class ContourStep:
    """Один шаг обработки — картинка + подпись."""

    key: str
    title: str
    description: str
    image: np.ndarray | None = None


@dataclass
class ContourInfo:
    """Сведения об одном найденном контуре."""

    index: int
    area: float
    perimeter: float
    bbox: tuple[int, int, int, int]     # x, y, w, h
    points: np.ndarray                  # исходные точки контура (N, 2)
    equalized: np.ndarray | None = None  # после ресэмплинга (k, 2)
    score: float | None = None           # NSP относительно эталона (|NSP|)
    shift: int | None = None             # лучший циклический сдвиг
    angle: float | None = None           # оценка поворота по arg(NSP), градусы
    hu_score: float | None = None        # cv2.matchShapes (Hu), меньше — похожее


@dataclass
class ContourResult:
    """Результат полного пайплайна."""

    steps: list[ContourStep] = field(default_factory=list)
    contours: list[ContourInfo] = field(default_factory=list)
    image: np.ndarray | None = None
    reference: np.ndarray | None = None
    message: str = ""
    details: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def step(self, key: str) -> ContourStep | None:
        for item in self.steps:
            if item.key == key:
                return item
        return None


# --------------------------------------------------------------------------
# Пайплайн
# --------------------------------------------------------------------------
def analyze_contours(
    scene: np.ndarray,
    reference_image: np.ndarray | None = None,
    params: ContourParams | None = None,
) -> ContourResult:
    """Выполнить контурный анализ сцены (и, при наличии, образца)."""
    # Защита от «перепутанных» позиционных аргументов:
    # analyze_contours(image, ContourParams(...)) тоже должно работать.
    if isinstance(reference_image, ContourParams) and params is None:
        params, reference_image = reference_image, None

    params = params or ContourParams()
    started = time.perf_counter()
    result = ContourResult()

    if scene is None:
        result.message = "Не удалось прочитать изображение"
        return result

    steps = result.steps

    # --- 1. Исходное изображение ------------------------------------------
    steps.append(
        ContourStep(
            "source", "1. Исходное изображение",
            "Изображение-сцена, на котором ищутся объекты.", scene.copy(),
        )
    )

    # --- 2. Оттенки серого -------------------------------------------------
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY) if scene.ndim == 3 else scene.copy()
    steps.append(
        ContourStep(
            "gray", "2. Оттенки серого",
            "Цвет не нужен для формы: cv2.cvtColor(..., COLOR_BGR2GRAY).",
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        )
    )
    gray_work = gray

    # --- 3. Размытие -------------------------------------------------------
    blur_kernel = _odd(params.blur_kernel)
    if blur_kernel >= 3:
        gray_work = cv2.GaussianBlur(gray_work, (blur_kernel, blur_kernel), 0)
        steps.append(
            ContourStep(
                "blur", "3. Размытие (подавление шума)",
                f"Гауссово размытие, ядро {blur_kernel}x{blur_kernel}.",
                cv2.cvtColor(gray_work, cv2.COLOR_GRAY2BGR),
            )
        )

    # --- 4. Бинаризация ----------------------------------------------------
    binary, threshold_text, inverted = binarize(gray_work, params)
    steps.append(
        ContourStep(
            "binary", "4. Бинаризация",
            f"Порог: {threshold_text}"
            + (" (инвертировано: объекты темнее фона)" if inverted else ""),
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        )
    )

    # --- 5. Удаление шумов морфологией ------------------------------------
    kernel_size = max(1, _odd(params.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = binary.copy()
    if kernel_size >= 3:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    steps.append(
        ContourStep(
            "morph", "5. Удаление шумов (морфология)",
            f"Открытие + закрытие эллиптическим ядром {kernel_size}x{kernel_size}.",
            cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR),
        )
    )

    # --- 6. Поиск контуров -------------------------------------------------
    found = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_contours = list(found[0]) if isinstance(found, tuple) else list(found)
    all_contours_image = scene.copy()
    cv2.drawContours(all_contours_image, raw_contours, -1, (0, 255, 255), 2)
    steps.append(
        ContourStep(
            "contours", "6. Поиск контуров",
            f"cv2.findContours: найдено {len(raw_contours)} контуров "
            "(включая шум и вложенные области).",
            all_contours_image,
        )
    )

    # --- 7. Фильтрация -----------------------------------------------------
    image_area = float(scene.shape[0] * scene.shape[1])
    filtered = filter_contours(raw_contours, params, image_area)

    filtered_image = scene.copy()
    cv2.drawContours(filtered_image, [c for c in filtered], -1, (0, 255, 0), 2)
    steps.append(
        ContourStep(
            "filtered", "7. Фильтрация контуров",
            f"Оставлено {len(filtered)} контуров из {len(raw_contours)}: "
            f"площадь {params.min_area_ratio * 100:.2f}%–{params.max_area_ratio * 100:.0f}% "
            f"кадра, периметр ≥ {params.min_perimeter:.0f} px.",
            filtered_image,
        )
    )

    # --- Описание контуров -------------------------------------------------
    infos: list[ContourInfo] = []
    for index, contour in enumerate(filtered[: params.max_contours], start=1):
        perimeter = float(cv2.arcLength(contour, True))
        approx = contour
        if params.approx_eps > 0:
            approx = cv2.approxPolyDP(contour, params.approx_eps * perimeter, True)
        points = approx.reshape(-1, 2).astype(np.float32)
        if len(points) < 3:
            continue
        infos.append(
            ContourInfo(
                index=index,
                area=float(cv2.contourArea(contour)),
                perimeter=perimeter,
                bbox=tuple(int(v) for v in cv2.boundingRect(contour)),
                points=points,
            )
        )

    # --- 8. Уравнивание контуров ------------------------------------------
    reference_points = _reference_points(reference_image, scene, params, raw_contours, infos)

    equalize_image = scene.copy()
    cv2.drawContours(equalize_image, filtered[: params.max_contours], -1, (80, 80, 80), 1)
    for info in infos:
        info.equalized = resample_contour(info.points, params.resample_points)
        _draw_equalized(equalize_image, info.equalized, (255, 160, 0))
    if reference_points is not None:
        result.reference = reference_points
        _draw_equalized(equalize_image, reference_points, (0, 255, 255))
        equalize_image = draw_text(
            equalize_image,
            f"Эталонный контур: {len(reference_points)} точек (жёлтый)",
            (12, 12), font_size=20, color=(0, 0, 0), background=(0, 255, 255), padding=6,
        )
    steps.append(
        ContourStep(
            "equalized", "8. Уравнивание контуров",
            f"Ресэмплинг по длине дуги до {params.resample_points} точек: "
            "контуры разной длины приводятся к одному числу элементарных векторов.",
            equalize_image,
        )
    )

    # --- 9. Сравнение (NSP + ВКФ) -----------------------------------------
    if reference_points is not None:
        for info in infos:
            score, shift, angle = compare_contours(reference_points, info.equalized)
            info.score = score
            info.shift = shift
            info.angle = angle
            try:
                info.hu_score = float(
                    cv2.matchShapes(
                        reference_points.reshape(-1, 1, 2),
                        info.equalized.reshape(-1, 1, 2),
                        cv2.CONTOURS_MATCH_I1,
                        0.0,
                    )
                )
            except cv2.error:
                info.hu_score = None

    # --- 10. Итоговое изображение -----------------------------------------
    final_image = scene.copy()
    for info in infos:
        x, y, w, h = info.bbox
        color = _score_color(info.score)
        cv2.rectangle(final_image, (x, y), (x + w, y + h), color, 2)
        if info.equalized is not None:
            cv2.polylines(
                final_image, [info.equalized.astype(np.int32)], True, color, 1, cv2.LINE_AA
            )
        label = f"#{info.index}"
        if info.score is not None:
            label += f" NSP={info.score:.2f}"
        final_image = draw_badge(final_image, label, (x, max(0, y - 26)), color=color, font_size=18)

    summary = [f"Найдено объектов: {len(infos)}"]
    for info in infos[:6]:
        line = (
            f"#{info.index}: S={info.area:.0f} px², P={info.perimeter:.0f} px, "
            f"точек={len(info.points)}"
        )
        if info.score is not None:
            line += f", |NSP|={info.score:.3f} (сдвиг {info.shift}"
            if info.angle is not None:
                line += f", поворот ≈ {info.angle:+.0f}°"
            line += ")"
            if info.hu_score is not None:
                line += f", Hu={info.hu_score:.3f}"
        summary.append(line)

    final_image = draw_multiline(
        final_image,
        summary,
        (12, 12),
        font_size=18,
        color=(255, 255, 255),
        background=(0, 0, 0),
        padding=4,
    )
    steps.append(
        ContourStep(
            "final", "9. Результат",
            "Рамки вокруг найденных объектов, пронумерованные контуры и "
            "коэффициент сходства NSP относительно эталона.",
            final_image,
        )
    )

    result.contours = infos
    result.image = final_image
    result.message = f"Обработано контуров: {len(infos)} (из {len(raw_contours)} найденных)"
    result.details = [
        f"Контуров до фильтрации: {len(raw_contours)}",
        f"Контуров после фильтрации: {len(filtered)}",
        f"Длина вектор-контура после уравнивания: {params.resample_points}",
        f"Эталон: {'задан отдельным изображением' if reference_image is not None else 'самый крупный контур сцены'}",
    ]
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


# --------------------------------------------------------------------------
# Бинаризация
# --------------------------------------------------------------------------
def binarize(gray: np.ndarray, params: ContourParams) -> tuple[np.ndarray, str, bool]:
    """Получить бинарную маску объектов.

    Возвращает ``(маска, описание метода, была_ли_инверсия)``.

    Важный момент: у Оцу нет понятия «объект/фон» — он просто делит яркости.
    Тёмный текст на белой бумаге даёт белый фон, и ``findContours`` находит
    одну рамку вокруг всего листа. Поэтому при ``auto_invert`` мы смотрим,
    какая часть кадра стала «белой» (объектом), и при необходимости
    инвертируем маску.
    """
    invert_flag = cv2.THRESH_BINARY_INV if params.invert else cv2.THRESH_BINARY

    if params.adaptive:
        block = _odd(max(3, params.adaptive_block))
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, invert_flag, block, params.adaptive_c
        )
        description = f"адаптивный порог (блок {block}, C={params.adaptive_c})"
    else:
        _, binary = cv2.threshold(gray, 0, 255, invert_flag + cv2.THRESH_OTSU)
        description = "метод Оцу"

    inverted = False
    if params.auto_invert:  # и авто-инверсия по доле «объекта»
        object_ratio = float(np.mean(binary > 0))
        if object_ratio > 0.5:
            binary = cv2.bitwise_not(binary)
            inverted = True
    elif params.invert:
        inverted = True

    return binary, description, inverted


# --------------------------------------------------------------------------
# Фильтрация
# --------------------------------------------------------------------------
def filter_contours(contours, params: ContourParams, image_area: float) -> list:
    """Оставить только «осмысленные» контуры."""
    min_area = params.min_area_ratio * image_area
    max_area = params.max_area_ratio * image_area
    result = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < params.min_perimeter:
            continue
        _, _, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect = width / height
        if aspect > 25 or aspect < 1 / 25:
            continue
        result.append(contour)
    # Крупные объекты — первыми
    result.sort(key=cv2.contourArea, reverse=True)
    return result


# --------------------------------------------------------------------------
# Уравнивание (ресэмплинг) контуров
# --------------------------------------------------------------------------
def resample_contour(points: np.ndarray, count: int) -> np.ndarray:
    """Привести контур к ровно ``count`` точкам, равномерным по длине дуги.

    Если исходный контур длиннее — это ресэмплинг (прореживание), если
    короче — интерполяция. Оба случая описаны в методичке как «уравнивание».
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0) if len(points) else np.zeros((count, 2), np.float32)

    count = max(3, int(count))
    closed = np.vstack([points, points[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 0:
        return np.repeat(points[:1], count, axis=0)

    # count точек, равномерно распределённых по периметру (без дубля первой)
    targets = np.linspace(0.0, total, count, endpoint=False)
    result = np.zeros((count, 2), dtype=np.float32)
    for index, target in enumerate(targets):
        position = np.searchsorted(cumulative, target, side="right") - 1
        position = int(np.clip(position, 0, len(segment_lengths) - 1))
        segment = segment_lengths[position]
        ratio = 0.0 if segment == 0 else (target - cumulative[position]) / segment
        result[index] = closed[position] + ratio * (closed[position + 1] - closed[position])
    return result


def contour_to_ev(points: np.ndarray) -> np.ndarray:
    """Точки контура -> последовательность элементарных векторов (комплексные)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    shifted = np.roll(points, -1, axis=0)
    vectors = shifted - points
    return vectors[:, 0] + 1j * vectors[:, 1]


def scalar_product(ev_a: np.ndarray, ev_b: np.ndarray) -> complex:
    """Скалярное произведение вектор-контуров в комплексном виде.

    ``(G, K) = Σ ev_Gi * conj(ev_Ki)`` — именно так определена операция в
    методичке: произведение комплексных чисел ``(a+ib)(c-id)``.
    """
    norm_a = np.linalg.norm(ev_a)
    norm_b = np.linalg.norm(ev_b)
    if norm_a == 0 or norm_b == 0:
        return 0j
    return complex(np.sum(ev_a * np.conj(ev_b)) / (norm_a * norm_b))


def normalized_scalar_product(ev_a: np.ndarray, ev_b: np.ndarray) -> float:
    """NSP = (G, K) / (|G| |K|) в обычном (вещественном) смысле."""
    return float(np.clip(np.real(scalar_product(ev_a, ev_b)), -1.0, 1.0))


def compare_contours(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, int, float]:
    """Сравнить контуры с перебором циклических сдвигов (ВКФ).

    Возвращает ``(лучший |NSP|, лучший сдвиг, оценка поворота в градусах)``.

    Берётся модуль NSP: по теории ``|NSP| = 1`` тогда и только тогда, когда
    контуры отличаются только масштабом и/или поворотом, а аргумент
    комплексного NSP как раз равен углу этого поворота.

    Дополнительно перебирается обратное направление обхода: ``findContours``
    может вернуть контур в любую сторону.
    """
    if reference is None or candidate is None:
        return 0.0, 0, 0.0
    count = min(len(reference), len(candidate))
    if count < 4:
        return 0.0, 0, 0.0

    ev_reference = contour_to_ev(resample_contour(reference, count))
    ev_candidate = contour_to_ev(resample_contour(candidate, count))

    best_score = -1.0
    best_shift = 0
    best_angle = 0.0
    for direction in (1, -1):
        sequence = ev_candidate if direction == 1 else ev_candidate[::-1]
        for shift in range(count):
            value = scalar_product(ev_reference, np.roll(sequence, shift))
            score = abs(value)
            if score > best_score:
                best_score = score
                best_shift = shift
                # arg(NSP) — оценка поворота одного контура относительно другого
                best_angle = float(np.degrees(np.angle(value))) * direction
    return float(best_score), int(best_shift), best_angle


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------
def _reference_points(reference_image, scene, params: ContourParams, raw_contours, infos):
    """Получить эталонный вектор-контур."""
    if reference_image is not None:
        gray = (
            cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            if reference_image.ndim == 3
            else reference_image
        )
        kernel_size = max(1, _odd(params.blur_kernel))
        if kernel_size >= 3:
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        binary, _, _ = binarize(gray, params)
        found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(found[0]) if isinstance(found, tuple) else list(found)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            points = biggest.reshape(-1, 2).astype(np.float32)
            return resample_contour(points, params.resample_points)

    # Иначе эталоном становится самый крупный контур сцены
    if infos:
        return resample_contour(infos[0].points, params.resample_points)
    return None


def _draw_equalized(image: np.ndarray, points: np.ndarray, color) -> None:
    """Нарисовать уравненный контур и его точки."""
    if points is None or len(points) < 2:
        return
    polyline = points.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polyline], True, color, 1, cv2.LINE_AA)
    for x, y in points.astype(np.int32):
        cv2.circle(image, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def _score_color(score: float | None):
    """Цвет рамки по степени сходства с эталоном."""
    if score is None:
        return (0, 255, 0)
    if score >= 0.9:
        return (0, 255, 0)      # BGR: зелёный — почти совпадение
    if score >= 0.7:
        return (0, 200, 255)    # жёлтый — похоже
    return (0, 0, 255)          # красный — слабое сходство


def _odd(value: int) -> int:
    """Сделать число нечётным (требование ядер OpenCV)."""
    value = int(value)
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1

