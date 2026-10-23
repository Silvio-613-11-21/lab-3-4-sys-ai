import cv2
import tempfile
import os

_tmp_dir = tempfile.mkdtemp(prefix="flet_cam_")
_frame_path = os.path.join(_tmp_dir, "frame.png")   

def frame_to_path(frame, path: str = _frame_path) -> str:
    ok = cv2.imwrite(path, frame)
    if not ok:
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("Не удалось закодировать кадр")
        buf.tofile(path)
    return path