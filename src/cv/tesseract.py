import pytesseract
from PIL import Image
import asyncio

pytesseract.pytesseract.tesseract_cmd  = r"c:\\Program Files\\Tesseract-OCR\\tesseract.exe"

async def image_to_string(path: str):
    return await asyncio.to_thread(sync_image_to_string, path)

def sync_image_to_string(path: str,):
    
    string = pytesseract.image_to_string(Image.open(path))

    coord = None
    return coord,  string 


# print(  pytesseract.image_to_boxes('image.png'))

# boxes =  pytesseract.image_to_boxes('image.png')

