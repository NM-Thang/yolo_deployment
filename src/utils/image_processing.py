import cv2
import numpy as np
from pathlib import Path


def preprocess_image_single(image_path: str | Path, input_size: tuple = (640, 640)) -> np.ndarray:
    """Read and preprocess a single image for YOLOv8 (ONNX/Triton).

    Returns a tensor with shape (3, H, W).
    """
    img_path_str = str(image_path)
    img = cv2.imread(img_path_str)

    if img is None:
        raise ValueError(f"Cannot read image from {img_path_str}")

    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose((2, 0, 1))
    return img_chw.astype(np.float32) / 255.0


def preprocess_image(image_path: str | Path, input_size: tuple = (640, 640)) -> np.ndarray:
    """
    Read and preprocess the image for YOLOv8 (ONNX/Triton).
    """
    img_chw = preprocess_image_single(image_path, input_size=input_size)
    return np.expand_dims(img_chw, axis=0)