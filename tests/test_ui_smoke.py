"""Дымовые тесты интерфейса.

Flet-страницу можно «сымитировать»: интерфейсу нужны только ``add``,
``update``, ``services``, ``run_task`` и несколько свойств. Такой тест
ловит опечатки в именах контролов и ошибки сборки дерева UI без запуска
реального окна (полезно и в CI, где нет дисплея).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.core.features import MatchParams
from src.ui.app import MODES, App


class FakeWindow:
    width = None
    height = None
    min_width = None
    min_height = None


class FakePage:
    """Минимальная заглушка ``flet.Page``."""

    def __init__(self):
        self.title = None
        self.theme_mode = None
        self.bgcolor = None
        self.padding = None
        self.spacing = None
        self.window = FakeWindow()
        self.controls = []
        self.services = []
        self.dialogs = []
        self.on_close = None
        self.update_count = 0

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        self.update_count += 1

    def run_task(self, handler, *args, **kwargs):
        # Гасим бесконечный цикл обновления: в тесте он не нужен
        if inspect.iscoroutine(handler):
            handler.close()
        return None

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)


@pytest.fixture
def app():
    page = FakePage()
    instance = App(page)
    yield instance
    instance._stop_live()


def test_app_builds_main_layout(app):
    assert len(app.page.controls) == 1
    assert app.mode == "match"
    assert app.params_host.controls, "параметры режима должны быть построены"
    assert app.page.services, "FilePicker должен быть подключён к странице"


@pytest.mark.parametrize("mode", list(MODES))
def test_switch_mode_builds_parameters(app, mode):
    app._switch_mode(mode)
    assert app.mode == mode
    assert len(app.params_host.controls) == 1


def test_match_parameters_are_collected(app):
    app._switch_mode("match")
    params = app._match_params()

    assert isinstance(params, MatchParams)
    assert params.max_features >= 100
    assert 0.5 <= params.ratio <= 0.95


def test_status_and_details_update(app):
    app._set_status("проверка")
    app._set_details(["строка 1", "строка 2"])
    assert app.status_text.value == "проверка"
    assert "строка 1" in app.details_field.value


def test_show_image_updates_preview(app, sample_image):
    app._show_image(sample_image)
    assert app.last_image is sample_image
    assert app.preview_image.src.startswith("data:image/png;base64,")


def test_log_keeps_only_last_lines(app):
    for index in range(60):
        app._log(f"сообщение {index}")
    assert len(app.log_field.value.splitlines()) <= 40


def test_open_image_in_viewer_is_not_called_in_tests(app, tmp_path):
    # Проверяем только, что сохранение без картинки не падает
    app._on_save(None)
    assert app.last_image is None


def test_save_writes_file(app, sample_image, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app._show_image(sample_image)
    app._on_save(None)

    files = list((tmp_path / "results").glob("*.png"))
    assert files, "файл результата должен быть сохранён"
    assert app.last_image is not None


def test_close_releases_resources(app):
    app._on_close(None)
    assert app.live.is_running is False
    assert app.camera.is_running is False


def test_control_tree_serializes_for_flet_protocol(app):
    """Flet передаёт дерево контролов на клиент msgpack-ом.

    Если у контрола недопустимое значение поля, ошибка вылезет именно здесь —
    ещё до открытия окна.
    """
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    encoder = configure_encode_object_for_msgpack(BaseControl)
    for mode in MODES:
        app._switch_mode(mode)
        payload = msgpack.packb(app.page.controls, default=encoder)
        assert len(payload) > 100

    msgpack.packb(
        [app.status_text, app.details_field, app.preview_image, app.spinner],
        default=encoder,
    )


def test_live_frame_processing_without_sample(app):
    frame = np.zeros((50, 50, 3), np.uint8)
    image, status = app._process_live_frame(frame)
    assert image is frame
    assert "образец" in status.lower()
