import base64
from io import BytesIO
from PIL import Image as PILImage
import flet as ft
import cv2
import numpy as np

def cv2_to_flet_base64(cv2_img):
   
    if cv2_img is None:
        return ""
    # 1. BGR -> RGB
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # 2. Numpy -> PIL
    pil_img = PILImage.fromarray(rgb_img)
    # 3. PIL -> BytesIO
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    # 4. BytesIO -> base64
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_base64