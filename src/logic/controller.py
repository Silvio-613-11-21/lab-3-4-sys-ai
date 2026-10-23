"""Прикладная логика: кэш изображений и фоновый обработчик видеопотока.

Здесь нет ни одной строчки, зависящей от Flet: этим модулем можно
пользоваться из консоли и покрывать его тестами.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import LIVE_PROCESS_FPS
from ..core.image_io import imread_unicode, to_data_uri


# --------------------------------------------------------------------------
# Кэш изображений
# --------------------------------------------------------------------------
class ImageCache:
    """Кэш прочитанных изображений: путь + время изменения -> массив.

    Повторное чтение файла с диска на каждом кадре камеры — лишняя работа.
    Кэш также защищает от гонок: файл мог быть удалён/перезаписан.
    """

    def __init__(self, max_items: int = 32):
        self._cache: dict[str, tuple[float, np.ndarray]] = {}
        self._max_items = max_items

    def get(self, path: str | None) -> np.ndarray | None:
        if not path:
            return None
        key = str(Path(str(path)))
        try:
            mtime = Path(key).stat().st_mtime
        except OSError:
            return None

        cached = self._cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        image = imread_unicode(key)
        if image is None:
            return None

        if len(self._cache) >= self._max_items:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (mtime, image)
        return image

    def clear(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------
# Результат обработки кадра
# --------------------------------------------------------------------------
@dataclass
class LiveFrame:
    """Один обработанный кадр потока."""

    sequence: int
    data_uri: str
    status: str
    fps: float


class LiveProcessor:
    """Фоновый обработчик видеопотока (камера или экран).

    Схема:

    ``frame_provider()`` -> кадр -> ``processor(frame)`` -> (картинка, статус)
    -> PNG в base64 -> «последний результат» для интерфейса.

    Обработка идёт в отдельном потоке, поэтому SIFT на кадре 1280x720 не
    блокирует кнопки и прокрутку. Интерфейс забирает только самый свежий
    результат (``latest``) и никогда не ждёт обработчик.
    """

    def __init__(self, fps: float = LIVE_PROCESS_FPS):
        self.fps = max(0.5, float(fps))
        self.error = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: LiveFrame | None = None
        self._sequence = 0
        self._measured_fps = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        frame_provider: Callable[[], np.ndarray | None],
        processor: Callable[[np.ndarray], tuple[np.ndarray | None, str]],
        fps: float | None = None,
    ) -> None:
        if self.is_running:
            return
        if fps is not None:
            self.fps = max(0.5, float(fps))
        self.error = ""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(frame_provider, processor),
            name="live-processor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None

    def latest(self) -> LiveFrame | None:
        with self._lock:
            return self._latest

    def reset(self) -> None:
        with self._lock:
            self._latest = None

    # ------------------------------------------------------------------
    def _loop(self, frame_provider, processor) -> None:
        started = time.perf_counter()
        frames = 0
        while not self._stop.is_set():
            cycle_started = time.perf_counter()
            try:
                frame = frame_provider()
            except Exception as error:  # noqa: BLE001 - экран/камера могут отвалиться
                self.error = f"Источник кадров: {error}"
                self._stop.wait(0.5)
                continue

            if frame is None:
                self._stop.wait(0.02)
                continue

            try:
                image, status = processor(frame)
            except Exception as error:  # noqa: BLE001 - обработка не должна убивать поток
                self.error = f"Обработка кадра: {error}"
                image, status = frame, f"Ошибка обработки: {error}"

            uri = to_data_uri(image) if image is not None else ""

            frames += 1
            elapsed = time.perf_counter() - started
            if elapsed >= 1.0:
                self._measured_fps = frames / elapsed
                frames = 0
                started = time.perf_counter()

            with self._lock:
                self._sequence += 1
                self._latest = LiveFrame(
                    sequence=self._sequence,
                    data_uri=uri,
                    status=status,
                    fps=self._measured_fps,
                )

            # держим заданный темп обработки, чтобы не грузить CPU на 100 %
            spent = time.perf_counter() - cycle_started
            rest = 1.0 / self.fps - spent
            if rest > 0:
                self._stop.wait(rest)
