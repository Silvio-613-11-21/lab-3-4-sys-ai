"""Работа с веб-камерой в отдельном потоке.

Почему отдельный поток:

* ``cv2.VideoCapture.read()`` — блокирующая операция. Если вызывать её в
  цикле интерфейса, окно «замирает».
* Камера отдаёт 30 кадров/с, а распознавание SIFT на CPU занимает десятки
  миллисекунд. Нужен буфер «последний кадр»: обработчик всегда берёт самый
  свежий кадр, а устаревшие пропускает.

Ресурс камеры обязательно освобождается (``release``), иначе после закрытия
окна устройство остаётся занятым — именно из-за этого программа «через время
перестаёт работать»: камера не открывается повторно, пока жив старый процесс.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ..config import CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_PROBE_MAX_INDEX


def _backend_for_platform() -> int:
    """На Windows DirectShow открывается быстрее и стабильнее, чем MSMF."""
    import sys

    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class CameraStream:
    """Потокобезопасный «последний кадр» с камеры."""

    def __init__(self, index: int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._error = ""
        self._fps = 0.0
        self._frames = 0

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str:
        return self._error

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> bool:
        """Открыть камеру и запустить поток чтения кадров."""
        if self.is_running:
            return True

        self._error = ""
        cap = self._open_capture(self.index)
        if cap is None:
            self._error = (
                f"Камера с индексом {self.index} не открывается. "
                "Проверьте, что она не занята другим приложением, и попробуйте другой индекс."
            )
            return False

        self._cap = cap
        self._stop.clear()
        self._frames = 0
        self._thread = threading.Thread(target=self._reader_loop, name="camera-reader", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Остановить поток и освободить камеру."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._frame = None

    def read(self) -> np.ndarray | None:
        """Копия последнего кадра (или ``None``, если кадров ещё нет)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def restart(self, index: int | None = None) -> bool:
        """Перезапустить камеру, при необходимости с другим индексом."""
        self.stop()
        if index is not None:
            self.index = int(index)
        time.sleep(0.15)  # дать драйверу освободить устройство
        return self.start()

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _open_capture(self, index: int) -> cv2.VideoCapture | None:
        backend = _backend_for_platform()
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - не все бэкенды поддерживают
            pass

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None

        with self._lock:
            self._frame = frame
        return cap

    def _reader_loop(self) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                # Камера «отвалилась» (шнур, драйвер, занята другим процессом)
                self._error = "Поток с камеры прерван"
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame
                self._frame_time = time.perf_counter()
            self._frames += 1

            elapsed = time.perf_counter() - started
            if elapsed >= 1.0:
                self._fps = self._frames / elapsed
                self._frames = 0
                started = time.perf_counter()


def probe_cameras(max_index: int = CAMERA_PROBE_MAX_INDEX) -> list[int]:
    """Проверить индексы камер 0..max_index и вернуть доступные.

    На Linux/macOS обращение к несуществующему индексу может занять до
    секунды, поэтому вызывать эту функцию стоит из фонового потока.
    """
    available = []
    backend = _backend_for_platform()
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(index)
        cap.release()
    return available
