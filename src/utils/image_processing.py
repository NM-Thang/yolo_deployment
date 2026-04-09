import cv2
import numpy as np
from pathlib import Path

def preprocess_image(image_path: str | Path, input_size: tuple = (640, 640)) -> np.ndarray:
    """
    Read and preprocess the image for YOLOv8 (ONNX/Triton).
    """
    # Convert Path object to string for cv2.imread
    img_path_str = str(image_path)
    img = cv2.imread(img_path_str)
    
    if img is None:
        raise ValueError(f"Cannot read image from {img_path_str}")
    
    # Resize, convert color space (BGR to RGB)
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    img_chw = img_rgb.transpose((2, 0, 1))
    
    # Add batch dimension and normalize pixels to [0, 1]
    img_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32) / 255.0
    
    return img_tensor