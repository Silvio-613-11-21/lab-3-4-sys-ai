"""Интерфейс приложения (Flet 1.0).

Окно состоит из двух колонок:

* слева — переключатель режимов и настройки;
* справа — большое окно предпросмотра, статус и подробности.

Режимы:

* ``match``    — лабораторная №2: поиск объекта по ключевым точкам
                 (статичное изображение / камера / снимок экрана);
* ``contours`` — лабораторная №3, часть 1: контурный анализ с показом
                 промежуточных шагов;
* ``ocr``      — лабораторная №3, часть 2: Tesseract (rus/eng).

Тяжёлые вычисления выполняются в отдельных потоках (``asyncio.to_thread``
и :class:`src.logic.controller.LiveProcessor`), поэтому интерфейс не
«замерзает» на время распознавания.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
import traceback
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from ..config import (
    APP_TITLE,
    CAMERA_INDEX_DEFAULT,
    CAMERA_PROBE_MAX_INDEX,
    DEFAULT_APPROX_EPS,
    DEFAULT_BLUR_KERNEL,
    DEFAULT_DETECTOR,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_AREA_RATIO,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_OCR_LANG,
    DEFAULT_RATIO,
    DEFAULT_RESAMPLE_POINTS,
    IMAGE_EXTENSIONS,
    LIVE_PROCESS_FPS,
    OCR_LANGUAGES,
    UI_REFRESH_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from ..core.camera import CameraStream, probe_cameras
from ..core.contours import ContourParams, analyze_contours
from ..core.features import MatchParams, available_detectors, find_object, surf_available
from ..core.image_io import imwrite_unicode, to_data_uri
from ..core.ocr import find_tesseract, language_hint, recognize, reset_cache
from ..core.screen import ScreenCaptureError, grab_screen
from ..logic.controller import ImageCache, LiveProcessor
from .components.common import card, file_button, hint, path_label, section_title

# Прозрачный пиксель: чтобы ft.Image не ругался на пустой src
_PLACEHOLDER = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

MODES = {
    "match": "ЛР2 · Поиск объекта",
    "contours": "ЛР3 · Контуры",
    "ocr": "ЛР3 · Текст (OCR)",
}


class App:
    """Собирает интерфейс и связывает его с компьютерным зрением."""

    def __init__(self, page: ft.Page):
        self.page = page

        self.mode = "match"
        self.images = ImageCache()
        self.camera = CameraStream(CAMERA_INDEX_DEFAULT)
        self.live = LiveProcessor(LIVE_PROCESS_FPS)
        self.file_picker = ft.FilePicker()

        # Выбранные пользователем файлы
        self.sample_path: str | None = None      # образец объекта (ЛР2)
        self.scene_path: str | None = None       # сцена (ЛР2 и контуры)
        self.reference_path: str | None = None   # эталонный контур (ЛР3)
        self.ocr_path: str | None = None         # картинка для OCR

        self.last_image: np.ndarray | None = None   # что сохранять по кнопке
        self.contour_steps: dict[str, tuple[str, str, str]] = {}  # key -> (uri, title, desc)
        self._last_live_sequence = -1
        self._busy = False

        self._build()
        self._switch_mode("match")

    # ==================================================================
    # Построение интерфейса
    # ==================================================================
    def _build(self) -> None:
        page = self.page
        page.title = APP_TITLE
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = ft.Colors.GREY_900
        page.padding = 0
        page.spacing = 0
        try:
            page.window.width = WINDOW_WIDTH
            page.window.height = WINDOW_HEIGHT
            page.window.min_width = 900
            page.window.min_height = 600
        except Exception:  # noqa: BLE001 - настройки окна есть не во всех режимах
            pass

        # --- Левая колонка ------------------------------------------------
        self.mode_buttons: dict[str, ft.Button] = {}
        mode_row = ft.Row(spacing=6, wrap=True)
        for key, title in MODES.items():
            button = ft.Button(
                content=ft.Text(title, size=12),
                on_click=lambda e, mode=key: self._switch_mode(mode),
                bgcolor=ft.Colors.GREY_800,
            )
            self.mode_buttons[key] = button
            mode_row.controls.append(button)

        self.params_host = ft.Column(spacing=12, tight=True)
        self.log_field = ft.TextField(
            value="",
            read_only=True,
            multiline=True,
            min_lines=5,
            max_lines=8,
            text_size=11,
            label="Журнал",
        )

        left = ft.Container(
            width=430,
            bgcolor=ft.Colors.GREY_900,
            padding=16,
            content=ft.Column(
                controls=[
                    ft.Text("Системы искусственного интеллекта", size=18, weight=ft.FontWeight.BOLD),
                    ft.Text("ЛР2: поиск объекта · ЛР3: контуры и текст", size=12,
                            color=ft.Colors.GREY_400),
                    mode_row,
                    ft.Divider(height=1, color=ft.Colors.GREY_800),
                    self.params_host,
                    ft.Divider(height=1, color=ft.Colors.GREY_800),
                    ft.Button(
                        content="Проверить окружение",
                        icon=ft.Icons.HEALTH_AND_SAFETY,
                        on_click=self._on_check_environment,
                    ),
                    self.log_field,
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

        # --- Правая колонка ----------------------------------------------
        self.preview_image = ft.Image(
            src=_PLACEHOLDER,
            fit=ft.BoxFit.CONTAIN,
            expand=True,
        )
        self.preview_container = ft.Container(
            content=self.preview_image,
            expand=True,
            bgcolor=ft.Colors.BLACK,
            border=ft.Border.all(1, ft.Colors.GREY_700),
            border_radius=10,
            padding=4,
        )

        self.status_text = ft.Text(
            "Выберите изображения и нажмите «Распознать»",
            size=13,
            color=ft.Colors.GREY_300,
        )
        self.spinner = ft.ProgressRing(width=16, height=16, visible=False, stroke_width=2)

        self.details_field = ft.TextField(
            value="",
            read_only=True,
            multiline=True,
            min_lines=4,
            max_lines=7,
            text_size=12,
            label="Подробности",
        )

        self.btn_save = ft.Button(
            content="Сохранить результат",
            icon=ft.Icons.SAVE_ALT,
            on_click=self._on_save,
            bgcolor=ft.Colors.GREY_800,
        )

        right = ft.Container(
            expand=True,
            padding=16,
            content=ft.Column(
                controls=[
                    self.preview_container,
                    ft.Row(
                        controls=[self.spinner, self.status_text, ft.Container(expand=True), self.btn_save],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=10,
                    ),
                    self.details_field,
                ],
                expand=True,
                spacing=10,
            ),
        )

        page.add(ft.Row(controls=[left, right], expand=True, spacing=0))
        page.services.append(self.file_picker)
        page.on_close = self._on_close

        page.run_task(self._refresh_loop)

    # ==================================================================
    # Переключение режимов
    # ==================================================================
    def _switch_mode(self, mode: str) -> None:
        if mode == self.mode and self.params_host.controls:
            return
        self.mode = mode
        for key, button in self.mode_buttons.items():
            active = key == mode
            button.bgcolor = ft.Colors.BLUE_700 if active else ft.Colors.GREY_800
            button.content.color = ft.Colors.WHITE if active else ft.Colors.GREY_300

        if mode != "match":
            self._stop_live()

        self.params_host.controls = [self._build_params(mode)]
        self._set_status(f"Режим: {MODES[mode]}")
        self._set_details([])
        self._safe_update()

    def _build_params(self, mode: str) -> ft.Control:
        if mode == "match":
            return self._build_match_params()
        if mode == "contours":
            return self._build_contour_params()
        return self._build_ocr_params()

    # ------------------------------------------------------------------
    # ЛР2: параметры
    # ------------------------------------------------------------------
    def _build_match_params(self) -> ft.Control:
        self.lbl_sample = path_label(
            Path(self.sample_path).name if self.sample_path else "образец не выбран"
        )
        self.lbl_scene = path_label(
            Path(self.scene_path).name if self.scene_path else "сцена не выбрана"
        )

        detectors = available_detectors()
        self.dd_detector = ft.Dropdown(
            label="Детектор",
            value=DEFAULT_DETECTOR if DEFAULT_DETECTOR in detectors else detectors[0],
            options=[
                ft.DropdownOption(
                    key=name,
                    text={
                        "sift": "SIFT (масштабно-инвариантный)",
                        "surf": "SURF (быстрый)",
                        "orb": "ORB (очень быстрый)",
                    }.get(name, name.upper()),
                )
                for name in detectors
            ],
            text_size=12,
        )
        if not surf_available():
            detector_hint = hint(
                "SURF недоступен в этой сборке OpenCV (патентный алгоритм "
                "исключён) — используйте SIFT или ORB."
            )
        else:
            detector_hint = hint("SIFT точнее, ORB быстрее (удобно для камеры).")

        self.sl_ratio = ft.Slider(
            min=0.5, max=0.95, divisions=9, value=DEFAULT_RATIO,
            label="ratio = {value}", on_change=lambda e: None,
        )
        self.tf_features = ft.TextField(
            label="Максимум ключевых точек",
            value=str(DEFAULT_MAX_FEATURES),
            keyboard_type=ft.KeyboardType.NUMBER,
            input_filter=ft.NumbersOnlyInputFilter(),
            text_size=12,
        )
        self.cb_binarize = ft.Checkbox(label="Бинаризация Оцу перед поиском", value=False)
        self.cb_clahe = ft.Checkbox(label="Выравнивать контраст (CLAHE)", value=True)
        self.cb_markers = ft.Checkbox(label="Показывать ключевые точки", value=True)
        self.cb_unit = ft.Checkbox(label="Соединять подобные маркеры", value=False)

        self.dd_source = ft.Dropdown(
            label="Источник кадра",
            value="file",
            options=[
                ft.DropdownOption(key="file", text="Статичное изображение"),
                ft.DropdownOption(key="camera", text="Веб-камера (реальное время)"),
                ft.DropdownOption(key="screen", text="Снимок экрана"),
            ],
            text_size=12,
        )
        self.tf_camera = ft.TextField(
            label="Индекс камеры",
            value=str(CAMERA_INDEX_DEFAULT),
            keyboard_type=ft.KeyboardType.NUMBER,
            input_filter=ft.NumbersOnlyInputFilter(),
            text_size=12,
        )
        self.btn_probe = ft.Button(
            content="Найти камеры", icon=ft.Icons.VIDEOCAM, on_click=self._on_probe_cameras
        )
        self.btn_live = ft.Button(
            content="▶ Запустить поток",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_toggle_live,
            bgcolor=ft.Colors.GREEN_800,
        )
        self.btn_run_match = ft.Button(
            content="Распознать объект",
            icon=ft.Icons.SEARCH,
            on_click=self._on_run_match,
            bgcolor=ft.Colors.BLUE_700,
        )

        return ft.Column(
            controls=[
                card(
                    section_title("Изображения"),
                    ft.Row([file_button("Образец объекта…", self._on_pick_sample), self.lbl_sample],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Row([file_button("Сцена…", self._on_pick_scene), self.lbl_scene],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    hint("Образец — что ищем. Сцена — где ищем (статичная картинка или поток)."),
                ),
                card(
                    section_title("Параметры сопоставления"),
                    self.dd_detector,
                    detector_hint,
                    ft.Text("Порог ratio-теста Лоу", size=12),
                    self.sl_ratio,
                    hint("0.75 — стандарт: пара считается надёжной, если она "
                         "заметно лучше второго кандидата."),
                    self.tf_features,
                    self.cb_binarize,
                    self.cb_clahe,
                    self.cb_markers,
                    self.cb_unit,
                ),
                card(
                    section_title("Режим работы"),
                    self.dd_source,
                    ft.Row([self.tf_camera, self.btn_probe], spacing=8),
                    self.btn_live,
                    hint("Поток обрабатывается ~4 кадра/с в фоновом потоке, "
                         "интерфейс при этом остаётся отзывчивым."),
                    self.btn_run_match,
                ),
            ],
            spacing=12,
            tight=True,
        )

    # ------------------------------------------------------------------
    # ЛР3: контурный анализ
    # ------------------------------------------------------------------
    def _build_contour_params(self) -> ft.Control:
        self.lbl_contour_scene = path_label(
            Path(self.scene_path).name if self.scene_path else "сцена не выбрана"
        )
        self.lbl_reference = path_label(
            Path(self.reference_path).name
            if self.reference_path
            else "эталон не выбран (берётся крупнейший контур)"
        )

        self.sl_blur = ft.Slider(min=1, max=15, divisions=7, value=DEFAULT_BLUR_KERNEL,
                                 label="ядро размытия = {value}")
        self.sl_morph = ft.Slider(min=1, max=15, divisions=7, value=DEFAULT_MORPH_KERNEL,
                                  label="ядро морфологии = {value}")
        self.sl_min_area = ft.Slider(min=0.0005, max=0.05, divisions=20,
                                     value=DEFAULT_MIN_AREA_RATIO,
                                     label="мин. площадь = {value}")
        self.sl_resample = ft.Slider(min=16, max=256, divisions=15,
                                     value=DEFAULT_RESAMPLE_POINTS,
                                     label="точек после уравнивания = {value}")
        self.sl_approx = ft.Slider(min=0.0, max=0.08, divisions=8, value=DEFAULT_APPROX_EPS,
                                   label="эпсилон аппроксимации = {value}")
        self.cb_adaptive = ft.Checkbox(label="Адаптивный порог (вместо Оцу)", value=False)
        self.cb_invert = ft.Checkbox(label="Инвертировать (светлый объект на тёмном)", value=False)
        self.cb_auto_invert = ft.Checkbox(label="Авто-инверсия полярности", value=True)

        self.dd_step = ft.Dropdown(
            label="Показывать шаг обработки",
            value="final",
            options=[ft.DropdownOption(key="final", text="9. Результат")],
            text_size=12,
            on_select=self._on_step_selected,
        )
        self.btn_run_contours = ft.Button(
            content="Обработать контуры",
            icon=ft.Icons.ACCOUNT_TREE,
            on_click=self._on_run_contours,
            bgcolor=ft.Colors.BLUE_700,
        )

        return ft.Column(
            controls=[
                card(
                    section_title("Изображения"),
                    ft.Row([file_button("Сцена…", self._on_pick_scene), self.lbl_contour_scene],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Row([file_button("Эталонный объект…", self._on_pick_reference),
                            self.lbl_reference],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    hint("Эталон нужен для сравнения форм (NSP). Без него эталоном "
                         "становится самый крупный контур сцены."),
                ),
                card(
                    section_title("Предобработка"),
                    ft.Text("Размытие", size=12), self.sl_blur,
                    ft.Text("Морфология (шумоподавление)", size=12), self.sl_morph,
                    self.cb_adaptive,
                    self.cb_invert,
                    self.cb_auto_invert,
                ),
                card(
                    section_title("Контуры и сравнение"),
                    ft.Text("Минимальная площадь контура", size=12), self.sl_min_area,
                    ft.Text("Длина вектор-контура (уравнивание)", size=12), self.sl_resample,
                    ft.Text("Аппроксимация (доля периметра)", size=12), self.sl_approx,
                ),
                card(
                    section_title("Результат"),
                    self.btn_run_contours,
                    self.dd_step,
                    hint("Каждый шаг пайплайна можно посмотреть отдельно — "
                         "это визуальное сопровождение из задания."),
                ),
            ],
            spacing=12,
            tight=True,
        )

    # ------------------------------------------------------------------
    # ЛР3: OCR
    # ------------------------------------------------------------------
    def _build_ocr_params(self) -> ft.Control:
        self.lbl_ocr = path_label(
            Path(self.ocr_path).name if self.ocr_path else "изображение не выбрано"
        )

        tesseract_status = find_tesseract()
        if tesseract_status.available:
            status_hint = hint(language_hint())
        else:
            status_hint = ft.Text(
                "Tesseract OCR не найден — OCR работать не будет.\n"
                "Windows: установите tesseract-ocr и русский языковой пакет.\n"
                "Linux: sudo apt install tesseract-ocr tesseract-ocr-rus",
                size=11,
                color=ft.Colors.ORANGE_300,
            )

        self.dd_ocr_lang = ft.Dropdown(
            label="Язык распознавания",
            value=DEFAULT_OCR_LANG,
            options=[ft.DropdownOption(key=lang, text=lang) for lang in OCR_LANGUAGES],
            text_size=12,
        )
        self.cb_ocr_binarize = ft.Checkbox(label="Бинаризация Оцу перед OCR", value=True)
        self.btn_run_ocr = ft.Button(
            content="Распознать текст",
            icon=ft.Icons.TEXT_FIELDS,
            on_click=self._on_run_ocr,
            bgcolor=ft.Colors.BLUE_700,
        )
        self.btn_check_tesseract = ft.Button(
            content="Проверить Tesseract", icon=ft.Icons.REFRESH, on_click=self._on_check_tesseract
        )
        self.ocr_text_field = ft.TextField(
            value="",
            read_only=True,
            multiline=True,
            min_lines=5,
            max_lines=10,
            text_size=12,
            label="Распознанный текст",
        )

        return ft.Column(
            controls=[
                card(
                    section_title("Изображение с текстом"),
                    ft.Row([file_button("Выбрать изображение…", self._on_pick_ocr), self.lbl_ocr],
                           spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    status_hint,
                    self.btn_check_tesseract,
                ),
                card(
                    section_title("Параметры"),
                    self.dd_ocr_lang,
                    self.cb_ocr_binarize,
                    hint("Текст печатается в консоль и дублируется на выходном "
                         "изображении (панель под картинкой)."),
                    self.btn_run_ocr,
                ),
                card(section_title("Результат"), self.ocr_text_field),
            ],
            spacing=12,
            tight=True,
        )

    # ==================================================================
    # Вспомогательные методы
    # ==================================================================
    def _safe_update(self) -> None:
        try:
            self.page.update()
        except Exception:  # noqa: BLE001 - страница могла быть уже закрыта
            pass

    def _set_status(self, text: str, color=ft.Colors.GREY_300) -> None:
        self.status_text.value = text
        self.status_text.color = color
        self._safe_update()

    def _set_details(self, lines: list[str]) -> None:
        self.details_field.value = "\n".join(str(line) for line in lines)
        self._safe_update()

    def _show_image(self, image: np.ndarray | None) -> None:
        if image is None:
            return
        self.last_image = image
        self.preview_image.src = to_data_uri(image)
        self._safe_update()

    def _show_uri(self, uri: str) -> None:
        if not uri:
            return
        self.preview_image.src = uri
        self._safe_update()

    def _log(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        existing = self.log_field.value.splitlines() if self.log_field.value else []
        existing.append(f"[{stamp}] {message}")
        self.log_field.value = "\n".join(existing[-40:])
        self._safe_update()

    def _notify(self, message: str) -> None:
        try:
            self.page.show_dialog(ft.SnackBar(ft.Text(message)))
        except Exception:  # noqa: BLE001
            self._log(message)

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._busy = busy
        self.spinner.visible = busy
        if text:
            self._set_status(text, ft.Colors.LIGHT_BLUE_200)
        self._safe_update()

    def _handle_exception(self, where: str, error: Exception) -> None:
        self._log(f"ОШИБКА в {where}: {error}")
        traceback.print_exc()
        self._set_status(f"Ошибка: {error}", ft.Colors.RED_300)
        self._notify(f"{where}: {error}")
        self._set_busy(False)

    # ==================================================================
    # Выбор файлов
    # ==================================================================
    async def _pick_image(self) -> str | None:
        """Открыть диалог выбора картинки и вернуть путь."""
        try:
            files = await self.file_picker.pick_files(
                dialog_title="Выберите изображение",
                allowed_extensions=IMAGE_EXTENSIONS,
                allow_multiple=False,
            )
        except Exception as error:  # noqa: BLE001
            self._notify(f"Не удалось открыть диалог выбора файла: {error}")
            return None
        if not files:
            return None
        path = files[0].path
        if not path:
            self._notify(
                "Диалог вернул файл без пути (так бывает в веб-версии). "
                "Запустите приложение как настольное."
            )
            return None
        return path

    async def _on_pick_sample(self, event) -> None:
        path = await self._pick_image()
        if path:
            self.sample_path = path
            self.lbl_sample.value = Path(path).name
            self._log(f"Образец: {path}")
            self._safe_update()

    async def _on_pick_scene(self, event) -> None:
        path = await self._pick_image()
        if path:
            self.scene_path = path
            for label in ("lbl_scene", "lbl_contour_scene"):
                control = getattr(self, label, None)
                if control is not None:
                    control.value = Path(path).name
            self._log(f"Сцена: {path}")
            self._safe_update()
            self._preview_scene()

    async def _on_pick_reference(self, event) -> None:
        path = await self._pick_image()
        if path:
            self.reference_path = path
            self.lbl_reference.value = Path(path).name
            self._log(f"Эталон: {path}")
            self._safe_update()

    async def _on_pick_ocr(self, event) -> None:
        path = await self._pick_image()
        if path:
            self.ocr_path = path
            self.lbl_ocr.value = Path(path).name
            self._log(f"OCR: {path}")
            self._preview_scene(path)
            self._safe_update()

    def _preview_scene(self, path: str | None = None) -> None:
        image = self.images.get(path or self.scene_path)
        if image is not None:
            self._show_image(image)

    # ==================================================================
    # ЛР2: распознавание
    # ==================================================================
    def _match_params(self) -> MatchParams:
        try:
            max_features = int(self.tf_features.value or DEFAULT_MAX_FEATURES)
        except ValueError:
            max_features = DEFAULT_MAX_FEATURES

        return MatchParams(
            detector=self.dd_detector.value or DEFAULT_DETECTOR,
            ratio=float(self.sl_ratio.value),
            max_features=max(100, max_features),
            binarize=bool(self.cb_binarize.value),
            use_clahe=bool(self.cb_clahe.value),
            show_markers=bool(self.cb_markers.value),
            unit_markers=bool(self.cb_unit.value),
        )

    async def _on_run_match(self, event) -> None:
        if self._busy:
            return
        try:
            source = self.dd_source.value or "file"
            if source != "file":
                self._notify("Для потока нажмите «Запустить поток»")
                return

            sample = self.images.get(self.sample_path)
            scene = self.images.get(self.scene_path)

            if sample is None:
                self._notify("Сначала выберите изображение-образец")
                return
            if scene is None:
                self._notify("Сначала выберите изображение-сцену")
                return

            self._set_busy(True, "Идёт поиск объекта…")
            params = self._match_params()
            result = await asyncio.to_thread(find_object, sample, scene, params)

            self._show_image(result.image)
            color = ft.Colors.GREEN_300 if result.found else ft.Colors.ORANGE_300
            self._set_status(f"{result.message} · {result.elapsed_ms:.0f} мс", color)
            self._set_details(result.details)
            self._log(
                f"ЛР2: {result.message}; совпадений {result.match_count}, "
                f"inliers {result.inlier_count}"
            )
            self._set_busy(False)
        except Exception as error:  # noqa: BLE001
            self._handle_exception("поиске объекта", error)

    async def _on_probe_cameras(self, event) -> None:
        self._set_busy(True, "Ищу доступные камеры…")
        try:
            cameras = await asyncio.to_thread(probe_cameras, CAMERA_PROBE_MAX_INDEX)
        except Exception as error:  # noqa: BLE001
            self._handle_exception("поиске камер", error)
            return

        if cameras:
            self.tf_camera.value = str(cameras[0])
            self._set_status(f"Найдены камеры с индексами: {cameras}", ft.Colors.GREEN_300)
            self._log(f"Камеры: {cameras}")
        else:
            self._set_status("Камеры не найдены", ft.Colors.ORANGE_300)
            self._log("Камеры не найдены")
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Живой поток (камера или экран)
    # ------------------------------------------------------------------
    def _on_toggle_live(self, event) -> None:
        if self.live.is_running:
            self._stop_live()
            self._set_status("Поток остановлен")
            return

        source = self.dd_source.value or "camera"
        if source == "file":
            self._notify("Выберите источник «Веб-камера» или «Снимок экрана»")
            return

        if self.images.get(self.sample_path) is None:
            self._notify("Сначала выберите изображение-образец")
            return

        if source == "camera":
            try:
                index = int(self.tf_camera.value or CAMERA_INDEX_DEFAULT)
            except ValueError:
                index = CAMERA_INDEX_DEFAULT
            if self.camera.index != index or not self.camera.is_running:
                started = self.camera.restart(index)
            else:
                started = True
            if not started:
                self._set_status(self.camera.error, ft.Colors.RED_300)
                self._log(self.camera.error)
                self._notify(self.camera.error)
                return
            provider = self.camera.read
        else:
            provider = self._grab_screen_safe

        self._last_live_sequence = -1
        self.live.reset()
        self.live.start(provider, self._process_live_frame)

        self.btn_live.content = "■ Остановить поток"
        self.btn_live.icon = ft.Icons.STOP
        self.btn_live.bgcolor = ft.Colors.RED_800
        self._set_status("Поток запущен", ft.Colors.GREEN_300)
        self._log(f"Поток запущен: {source}")
        self._safe_update()

    def _stop_live(self) -> None:
        if self.live.is_running:
            self.live.stop()
        if self.camera.is_running:
            self.camera.stop()
        if hasattr(self, "btn_live"):
            self.btn_live.content = "▶ Запустить поток"
            self.btn_live.icon = ft.Icons.PLAY_ARROW
            self.btn_live.bgcolor = ft.Colors.GREEN_800
        self._safe_update()

    def _grab_screen_safe(self) -> np.ndarray | None:
        try:
            return grab_screen()
        except ScreenCaptureError as error:
            self.live.error = str(error)
            self._log(str(error))
            return None

    def _process_live_frame(self, frame: np.ndarray):
        """Обработчик кадра (выполняется в фоновом потоке)."""
        sample = self.images.get(self.sample_path)
        if sample is None:
            return frame, "Образец не выбран"
        try:
            params = self._match_params()
        except Exception:  # noqa: BLE001 - UI-контролы читаются из другого потока
            params = MatchParams()

        result = find_object(sample, frame, params)
        status = f"{result.message} · {result.elapsed_ms:.0f} мс"
        # result.image — это кадр с линиями совпадений и рамкой объекта,
        # поэтому в потоке видно и сцену, и результат поиска одновременно.
        return result.image, status

    async def _refresh_loop(self) -> None:
        """Забирать свежий кадр из обработчика и обновлять окно."""
        while True:
            await asyncio.sleep(1.0 / UI_REFRESH_FPS)
            try:
                if not self.live.is_running:
                    continue
                latest = self.live.latest()
                if latest is None or latest.sequence == self._last_live_sequence:
                    continue
                self._last_live_sequence = latest.sequence
                if latest.data_uri:
                    self._show_uri(latest.data_uri)
                self._set_status(f"{latest.status} · {latest.fps:.1f} кадр/с")
            except Exception:  # noqa: BLE001 - цикл не должен умирать
                await asyncio.sleep(0.5)

    # ==================================================================
    # ЛР3: контурный анализ
    # ==================================================================
    async def _on_run_contours(self, event) -> None:
        if self._busy:
            return
        try:
            scene = self.images.get(self.scene_path)
            if scene is None:
                self._notify("Сначала выберите изображение-сцену")
                return
            reference = self.images.get(self.reference_path)

            params = ContourParams(
                blur_kernel=int(self.sl_blur.value),
                adaptive=bool(self.cb_adaptive.value),
                invert=bool(self.cb_invert.value),
                auto_invert=bool(self.cb_auto_invert.value),
                morph_kernel=int(self.sl_morph.value),
                min_area_ratio=float(self.sl_min_area.value),
                approx_eps=float(self.sl_approx.value),
                resample_points=int(self.sl_resample.value),
            )

            self._set_busy(True, "Контурный анализ…")
            result = await asyncio.to_thread(analyze_contours, scene, reference, params)

            self.contour_steps = {
                step.key: (to_data_uri(step.image), step.title, step.description)
                for step in result.steps
                if step.image is not None
            }
            options = [
                ft.DropdownOption(key=step.key, text=step.title)
                for step in result.steps
                if step.image is not None
            ]
            self.dd_step.options = options
            self.dd_step.value = "final" if "final" in self.contour_steps else options[0].key
            self.dd_step.disabled = not options

            self._show_contour_step(self.dd_step.value)
            self._set_status(f"{result.message} · {result.elapsed_ms:.0f} мс",
                             ft.Colors.GREEN_300)
            self._set_details(result.details)
            self._log(f"ЛР3-контуры: {result.message}")
            self._set_busy(False)
        except Exception as error:  # noqa: BLE001
            self._handle_exception("контурном анализе", error)

    def _on_step_selected(self, event) -> None:
        key = self.dd_step.value
        if key:
            self._show_contour_step(key)

    def _show_contour_step(self, key: str) -> None:
        data = self.contour_steps.get(key)
        if not data:
            return
        uri, title, description = data
        self._show_uri(uri)
        self._set_status(title)
        current = self.details_field.value.splitlines() if self.details_field.value else []
        # описание шага + ранее показанные подробности
        extra = [line for line in current if not line.startswith("Шаг:")]
        self._set_details([f"Шаг: {title}", description] + extra[:6])

    # ==================================================================
    # ЛР3: OCR
    # ==================================================================
    async def _on_run_ocr(self, event) -> None:
        if self._busy:
            return
        try:
            image = self.images.get(self.ocr_path)
            if image is None:
                self._notify("Сначала выберите изображение с текстом")
                return

            language = self.dd_ocr_lang.value or DEFAULT_OCR_LANG
            binarize = bool(self.cb_ocr_binarize.value)

            self._set_busy(True, "Распознаю текст…")
            result = await asyncio.to_thread(recognize, image, language, binarize)

            self._show_image(result.image)
            self.ocr_text_field.value = result.text or "*текст не найден*"

            if result.ok:
                self._set_status(
                    f"Готово: {len(result.words)} слов, средняя уверенность "
                    f"{result.average_confidence:.0f}% · {result.elapsed_ms:.0f} мс",
                    ft.Colors.GREEN_300,
                )
            else:
                self._set_status(result.message, ft.Colors.ORANGE_300)

            self._set_details(
                [
                    f"Язык: {result.language}",
                    f"Слов: {len(result.words)}",
                    f"Средняя уверенность: {result.average_confidence:.1f}%",
                    result.message,
                    "Текст также выведен в консоль.",
                ]
            )
            self._log(f"ЛР3-OCR: {result.message}")
            self._set_busy(False)
        except Exception as error:  # noqa: BLE001
            self._handle_exception("распознавании текста", error)

    async def _on_check_tesseract(self, event) -> None:
        reset_cache()
        status = await asyncio.to_thread(find_tesseract, True)
        if status.available:
            self._notify(f"Tesseract найден: {status.path}")
            self._set_details(
                [
                    f"Tesseract: {status.version}",
                    f"Путь: {status.path}",
                    f"Языки: {', '.join(status.languages) or 'нет данных'}",
                ]
            )
        else:
            self._notify(status.message)
            self._set_details([status.message, "См. раздел «Установка Tesseract» в README."])
        self._safe_update()

    # ==================================================================
    # Диагностика окружения
    # ==================================================================
    async def _on_check_environment(self, event) -> None:
        self._set_busy(True, "Проверяю окружение…")
        try:
            cameras = await asyncio.to_thread(probe_cameras, CAMERA_PROBE_MAX_INDEX)
            tesseract = await asyncio.to_thread(find_tesseract, True)
        except Exception as error:  # noqa: BLE001
            self._handle_exception("проверке окружения", error)
            return

        tesseract_ok = "OK" if tesseract.available else "НЕ НАЙДЕН"
        screen_ok = "OK"
        try:
            grab_screen()
        except Exception as error:  # noqa: BLE001
            screen_ok = f"нет ({error})"

        lines = [
            f"Python: {sys.version.split()[0]} ({sys.platform})",
            f"Flet: {getattr(ft, '__version__', 'неизвестно')}",
            f"OpenCV: {cv2.__version__}",
            f"NumPy: {np.__version__}",
            f"Детекторы: {', '.join(name.upper() for name in available_detectors())}",
            f"Tesseract: {tesseract_ok}"
            + (f" · {tesseract.version} · {', '.join(tesseract.languages)}" if tesseract.available else ""),
            f"Камеры: {cameras if cameras else 'не найдены'}",
            f"Снимок экрана: {screen_ok}",
        ]
        self._set_details(lines)
        for line in lines:
            self._log(line)
        self._set_status("Проверка окружения завершена", ft.Colors.GREEN_300)
        self._set_busy(False)

    # ==================================================================
    # Сохранение
    # ==================================================================
    def _on_save(self, event) -> None:
        if self.last_image is None:
            self._notify("Нечего сохранять: сначала выполните обработку")
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path("results")
        path = folder / f"{self.mode}_{stamp}.png"
        if imwrite_unicode(path, self.last_image):
            absolute = path.resolve()
            self._set_status(f"Сохранено: {absolute}", ft.Colors.GREEN_300)
            self._log(f"Файл сохранён: {absolute}")
            self._notify(f"Сохранено: {absolute}")
        else:
            self._notify("Не удалось сохранить файл")

    # ==================================================================
    # Завершение работы
    # ==================================================================
    def _on_close(self, event) -> None:
        """Обязательно освобождаем камеру и поток при закрытии окна."""
        self._log("Закрытие: освобождаю камеру")
        try:
            self.live.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.camera.stop()
        except Exception:  # noqa: BLE001
            pass


def build(page: ft.Page) -> App:
    """Точка входа для ``main.py``."""
    return App(page)
