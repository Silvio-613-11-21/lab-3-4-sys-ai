"""Снимок экрана как источник кадра.

Используется библиотека ``mss``: она кроссплатформенная, не тянет за собой
тяжёлые зависимости и работает без графического окружения X11-обёрток.
Если ``mss`` недоступна, пробуем ``Pillow.ImageGrab`` (Windows/macOS), и лишь
в последнюю очередь — ``pyautogui`` (если он установлен).

Раньше модуль импортировал ``pyautogui`` на верхнем уровне. На Linux без
работающего X-сервера (а также на Python 3.13+, где часть зависимостей
``pyautogui`` не собирается) этот импорт падал **при запуске программы**, и
всё приложение не стартовало. Теперь импорты ленивые и с запасными путями.
"""

from __future__ import annotations

import numpy as np


class ScreenCaptureError(RuntimeError):
    """Не удалось сделать снимок экрана."""


def grab_screen(monitor: int = 1) -> np.ndarray:
    """Сделать снимок экрана и вернуть BGR-массив.

    :param monitor: номер монитора для ``mss`` (1 — основной).
    """
    errors: list[str] = []

    # --- Вариант 1: mss ---------------------------------------------------
    try:
        import mss  # noqa: PLC0415 - импорт внутри функции, чтобы не падать при старте
        import mss.tools

        with mss.mss() as sct:
            monitors = sct.monitors
            index = monitor if 0 <= monitor < len(monitors) else (1 if len(monitors) > 1 else 0)
            shot = sct.grab(monitors[index])
            frame = np.array(shot)  # BGRA
        if frame.size:
            return frame[:, :, :3].copy()
    except Exception as error:  # noqa: BLE001
        errors.append(f"mss: {error}")

    # --- Вариант 2: Pillow.ImageGrab -------------------------------------
    try:
        from PIL import ImageGrab  # noqa: PLC0415

        image = ImageGrab.grab()
        rgb = np.array(image.convert("RGB"))
        return rgb[:, :, ::-1].copy()
    except Exception as error:  # noqa: BLE001
        errors.append(f"ImageGrab: {error}")

    # --- Вариант 3: pyautogui (если установлен) --------------------------
    try:
        import pyautogui  # noqa: PLC0415

        image = pyautogui.screenshot()
        rgb = np.array(image.convert("RGB"))
        return rgb[:, :, ::-1].copy()
    except Exception as error:  # noqa: BLE001
        errors.append(f"pyautogui: {error}")

    raise ScreenCaptureError(
        "Не удалось сделать снимок экрана. " + "; ".join(errors)
    )
