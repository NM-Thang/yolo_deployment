import cv2
import numpy as np
from pathlib import Path


def _dynamic_letterbox(
    img: np.ndarray,
    min_size: int = 320,
    max_size: int = 640,
    stride: int = 32,
    pad_value: int = 114,
) -> np.ndarray:
    """Resize with aspect ratio + symmetric padding to a stride-aligned dynamic shape.

    Output H/W stay inside [min_size, max_size] and are divisible by stride.
    """
    h, w = img.shape[:2]

    # Keep image as close to original as possible.
    # Only downscale when the longer side exceeds max_size.
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
    else:
        resized_w = w
        resized_h = h

    if (resized_w, resized_h) != (w, h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    target_w = int(np.ceil(resized_w / stride) * stride)
    target_h = int(np.ceil(resized_h / stride) * stride)
    target_w = max(min_size, min(target_w, max_size))
    target_h = max(min_size, min(target_h, max_size))

    pad_w = max(target_w - resized_w, 0)
    pad_h = max(target_h - resized_h, 0)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)


def preprocess_image_single(image_path: str | Path, input_size: tuple = (640, 640)) -> np.ndarray:
    """Read and preprocess a single image for YOLOv8 (ONNX/Triton).

    Returns a tensor with shape (3, H, W).
    """
    img_path_str = str(image_path)
    img = cv2.imread(img_path_str)

    if img is None:
        raise ValueError(f"Cannot read image from {img_path_str}")

    max_size = int(min(input_size))
    img_dyn = _dynamic_letterbox(img, min_size=320, max_size=max_size, stride=32)
    img_rgb = cv2.cvtColor(img_dyn, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose((2, 0, 1))
    return img_chw.astype(np.float32) / 255.0


def preprocess_image(image_path: str | Path, input_size: tuple = (640, 640)) -> np.ndarray:
    """
    Read and preprocess the image for YOLOv8 (ONNX/Triton).
    """
    img_chw = preprocess_image_single(image_path, input_size=input_size)
    return np.expand_dims(img_chw, axis=0)