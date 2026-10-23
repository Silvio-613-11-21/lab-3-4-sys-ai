import time
from dataclasses import dataclass

import cv2

from ..ui.ui import UI
from ..cv.cv_ import match_fingerprints
from ..cv.tesseract import image_to_string

from ..utils.cv2_to_flet_base64 import cv2_to_flet_base64
from ..utils.frameToPath import frame_to_path
from ..utils.screenToPath import frame_to_path_pil
from ..utils.imread import imread


def grab_screen():
    """Снимок экрана. pyautogui подключаем лениво: его импорт может падать
    (нет X-сервера / не установлен), а из-за этого раньше не стартовала
    вся программа. Если pyautogui недоступен — берём Pillow."""
    try:
        import pyautogui as pui
        return pui.screenshot()
    except Exception:
        from PIL import ImageGrab
        return ImageGrab.grab()


@dataclass
class Cache():
    src_1: str | None = None
    src_2: str | None = None

    cap_num: int = 0
    # Раньше камера открывалась прямо здесь — ещё при импорте модуля.
    # Теперь только по требованию, чтобы устройство не оставалось занятым.
    cap: cv2.VideoCapture | None = None

    textFl: bool = False
    _cap_try_at: float = 0.0   # когда последний раз пытались открыть камеру

    def reset_cap(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cap_num)

    def open_cap(self):
        if self.cap is not None and self.cap.isOpened():
            return self.cap
        # не переоткрываем камеру на каждом кадре: 1 попытка в 2 секунды
        if time.monotonic() - self._cap_try_at > 2.0:
            self._cap_try_at = time.monotonic()
            self.reset_cap()
        return self.cap

    def close_cap(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    markers = True
    unit_markers = True


cache = Cache()


def close():
    """Освободить камеру (вызывается при закрытии окна)."""
    cache.close_cap()


async def state_reader(ui: UI):
    state = ui.img_source.state

    cache.textFl = ui.imgs_loads.text_check.value or False

    if (cache.textFl == False):
        match state:
            case "load":
                ImageReader.images_update(ui)
            case "camera":
                ImageReader.camera_update(ui)
            case "screen":
                ImageReader.screen_update(ui)
    else:
        match state:
            case "load":
                await TextReader.images_update(ui)
            case "camera":
                await TextReader.camera_update(ui)
            case "screen":
                await TextReader.screen_update(ui)


class ImageReader:
    @staticmethod
    def images_update(ui: UI):
        src_1 = ui.imgs_loads.img_src_1
        src_2 = ui.imgs_loads.img_src_2
        markers = ui.markers_state.show_markers.value
        unit_markers = ui.markers_state.unit_markers.value

        if(src_1 and src_2):
                if(
                    src_1 == cache.src_1 and
                    src_2 == cache.src_2 and
                    markers == cache.markers and
                    unit_markers == cache.unit_markers
                    ):
                    return
                else:
                    cache.src_1 = src_1
                    cache.src_2 = src_2
                    cache.markers = markers or False
                    cache.unit_markers = unit_markers or False

                    len_ , match_img, points = match_fingerprints(src_1, src_2, cache.markers,  cache.unit_markers)
                    print(f"pr {points}")
                    img_base64 = cv2_to_flet_base64(match_img)
                    ui.cv_window.set_src( img_base64)

    @staticmethod
    def camera_update(ui: UI):
        cap_num = camera_index(ui)

        if (cap_num != cache.cap_num):
            cache.cap_num = cap_num
            cache.reset_cap()

        cap = cache.open_cap()
        if cap is None or not cap.isOpened():
            return

        ret, frame = cap.read()

        if not ret or frame is None:
            return

        src_1 = frame_to_path(frame)
        src_2 = ui.imgs_loads.img_src_2
        markers = ui.markers_state.show_markers.value
        unit_markers = ui.markers_state.unit_markers.value

        if(src_1 and src_2):
            len_ , match_img, points  = match_fingerprints(src_1, src_2, markers or False, unit_markers or False)
            img_base64 = cv2_to_flet_base64(match_img)
            ui.cv_window.set_src( img_base64)


    @staticmethod
    def screen_update(ui: UI):
        try:
            screenshot = grab_screen()
        except Exception as error:
            # нет графической сессии / запрещён доступ к экрану
            ui.cv_window.txt.value = f"Снимок экрана недоступен: {error}"
            return
        src_1 = frame_to_path_pil(screenshot)
        src_2 = ui.imgs_loads.img_src_2
        markers = ui.markers_state.show_markers.value
        unit_markers = ui.markers_state.unit_markers.value


        if(src_1 and src_2):
            len_ , match_img, points  = match_fingerprints(src_1, src_2, markers or False,  unit_markers or False )
            img_base64 = cv2_to_flet_base64(match_img)
            ui.cv_window.set_src( img_base64)


class TextReader:
    @staticmethod
    async def images_update(ui: UI):
        src_1 = ui.imgs_loads.img_src_1

        if(src_1):
            if(src_1 == cache.src_1):
                show_file(ui, src_1)
                return
            else:
                cache.src_1 = src_1

                ui.cv_window.txt.value = "Ждите... =)"
                ui.update()
                coord, txt = await image_to_string(src_1)
                if(txt == ''): txt = "*нет текста*"
                ui.cv_window.txt.value = txt
                show_file(ui, src_1)

    @staticmethod
    async def camera_update(ui: UI):
            cap_num = camera_index(ui)

            if (cap_num != cache.cap_num):
                cache.cap_num = cap_num
                cache.reset_cap()

            cap = cache.open_cap()
            if cap is None or not cap.isOpened():
                return

            ret, frame = cap.read()

            if not ret or frame is None:
                return

            src_1 = frame_to_path(frame)

            if(src_1):
                ui.cv_window.set_src(cv2_to_flet_base64(frame))
                ui.update()

                coord, txt = await image_to_string(src_1)
                ui.cv_window.txt.value = txt
                ui.update()

    @staticmethod
    async def screen_update(ui: UI):
            try:
                screenshot = grab_screen()
            except Exception as error:
                ui.cv_window.txt.value = f"Снимок экрана недоступен: {error}"
                return
            src_1 = frame_to_path_pil(screenshot)

            if(src_1):
                img = imread(src_1)
                ui.cv_window.set_src(cv2_to_flet_base64(img))
                ui.update()

                coord, txt = await image_to_string(src_1)
                if(txt == ''): txt = "*нет текста*"
                ui.cv_window.txt.value = txt
                ui.update()


def camera_index(ui: UI) -> int:
    """Номер камеры из поля ввода. Пустая строка/мусор -> берём сохранённый."""
    try:
        return int(ui.img_source.cap_num.value)
    except (TypeError, ValueError):
        return cache.cap_num


def show_file(ui: UI, path: str):
    """Показать локальный файл в ft.Image.

    Flet (особенно в браузере) не умеет показывать произвольный путь к файлу,
    поэтому картинка сначала читается и передаётся как base64.
    """
    image = imread(path)
    ui.cv_window.set_src(cv2_to_flet_base64(image) if image is not None else "")
