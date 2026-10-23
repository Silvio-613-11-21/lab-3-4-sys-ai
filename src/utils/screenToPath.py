# utils/screenToPath.py
import tempfile, os

_tmp_dir = tempfile.mkdtemp(prefix="flet_screen_")
_screen_path = os.path.join(_tmp_dir, "screen.png")

def frame_to_path_pil(img, path: str = _screen_path) -> str:
    img.save(path, format="PNG")
    return path