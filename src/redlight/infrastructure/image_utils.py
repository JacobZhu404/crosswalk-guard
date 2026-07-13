"""图像 I/O 与标注工具函数。

规避 OpenCV 中文路径问题，提供统一的图像保存/读取与置信度颜色映射。
"""

import numpy as np


def save_jpg(img: np.ndarray, out_path: str) -> bool:
    """字节写盘，规避 OpenCV cv2.imwrite 中文路径静默失败。

    Args:
        img: BGR 图像数组
        out_path: 输出 .jpg 路径

    Returns:
        是否成功写入
    """
    import cv2
    if img is None or getattr(img, "size", 0) == 0:
        return False
    try:
        ok, buf = cv2.imencode(".jpg", img)
    except cv2.error:
        return False
    if not ok:
        return False
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return True


def conf_color_bgr(conf: float) -> tuple:
    """置信度 -> BGR 颜色。

    - 绿 >= 0.85
    - 黄 0.6 - 0.85
    - 红 < 0.6

    Returns:
        (B, G, R) 元组
    """
    if conf >= 0.85:
        return (0, 180, 0)
    if conf >= 0.6:
        return (0, 215, 230)
    return (0, 0, 220)


def robust_imread(path: str) -> np.ndarray:
    """二进制读取后 cv2.imdecode，避免中文路径问题。

    Args:
        path: 图像文件路径

    Returns:
        BGR 图像数组，失败返回 None
    """
    import cv2
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
