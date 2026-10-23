import flet as ft
from dataclasses import dataclass
import cv2
import pyautogui  as pui
import time 

from ..ui.ui import UI
from ..cv.cv_ import match_fingerprints
from ..cv.tesseract import image_to_string

from ..utils.cv2_to_flet_base64 import cv2_to_flet_base64
from ..utils.frameToPath import frame_to_path
from ..utils.screenToPath import frame_to_path_pil

@dataclass
class Cache():
    src_1: str | None = None
    src_2: str | None = None
    
    cap_num: int = 0 
    cap = cv2.VideoCapture(cap_num)

    textFl: bool = False 

    def reset_cap(self): 
        self.cap = cv2.VideoCapture(self.cap_num)

    markers = True
    unit_markers = True
        
cache =  Cache()

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
        cap_num = int(ui.img_source.cap_num.value)
        if not cap_num :
            cap_num = cache.cap_num

        if (cap_num != cache.cap_num):
            cache.cap_num = cap_num 
            cache.reset_cap() 

        ret, frame = cache.cap.read() 

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
        screenshot = pui.screenshot()
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
                ui.cv_window.set_src(src_1)
                return 
            else:
                cache.src_1 = src_1 

                ui.cv_window.txt.value = "Ждите... =)"
                ui.update()
                coord, txt = await image_to_string(src_1)
                if(txt == ''): txt = "*нет текста*" 
                ui.cv_window.txt.value = txt
                ui.update()

    @staticmethod
    async def camera_update(ui: UI):
            cap_num = int(ui.img_source.cap_num.value)
            if not cap_num :
                cap_num = cache.cap_num
    
            if (cap_num != cache.cap_num):
                cache.cap_num = cap_num 
                cache.reset_cap() 
    
            ret, frame = cache.cap.read() 
    
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
            screenshot = pui.screenshot()
            src_1 = frame_to_path_pil(screenshot)
            
            if(src_1):
                ui.cv_window.set_src(src_1)
                ui.update()

                coord, txt = await image_to_string(src_1)
                if(txt == ''): txt = "*нет текста*" 
                ui.cv_window.txt.value = txt
                ui.update()