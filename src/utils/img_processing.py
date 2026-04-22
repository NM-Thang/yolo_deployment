import cv2
import numpy as np
from pathlib import Path

from __future__ import annotations
from dataclasses import dataclass

import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


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
        img = cv2.resize(img, (resized_w, resized_h),
                         interpolation=cv2.INTER_LINEAR)

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


def img_preprocessing(image_path: str | Path | np.ndarray, input_size: tuple = (640, 640)) -> np.ndarray:
    """Read and preprocess a single image for YOLOv8 (ONNX/Triton).
    Returns a tensor with shape (3, H, W).
    """
    if not isinstance(image_path, np.ndarray):
        img_path_str = str(image_path)
        img = cv2.imread(img_path_str)
    else:
        img = image_path

    if img is None:
        raise ValueError(f"Cannot read image from {img_path_str}")

    min_size = int(min(input_size))
    max_size = int(max(input_size))
    img_dyn = _dynamic_letterbox(
        img, min_size=min_size, max_size=max_size, stride=32)

    img_rgb = cv2.cvtColor(img_dyn, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose((2, 0, 1))

    return img_chw.astype(np.float32) / 255.0


def batch_preprocessing(sources: list[Path] | list[np.ndarray], input_size: tuple[int, int] = (320, 640)) -> np.ndarray:
    tensors = [img_preprocessing(sources, input_size=input_size)
               for sources in sources]
    return np.stack(tensors, axis=0).astype(np.float32, copy=False)

DEFAULT_COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]


def _normalize_output(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    if output.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {output.shape}")

    if output.shape[0] in (84, 85) and output.shape[1] > output.shape[0]:
        output = output.T  # -> (num_boxes, 84/85)

    if output.shape[1] < 5:
        raise ValueError(f"YOLO output has too few columns: {output.shape}")

    return output.astype(np.float32, copy=False)


def postprocess_yolo(
        raw_output: np.ndarray,
        input_size: tuple[int, int],
        orig_size: tuple[int, int],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        top_k: int = 10,
        class_names: tuple[str, ...] = DEFAULT_COCO_CLASSES,
) -> list[Detection]:
    predictions = _normalize_output(raw_output)
    # predictions_tensor = torch.from_numpy(predictions.T[None, ...])
    predictions_tensor = torch.from_numpy(predictions.copy().T[None, ...])
    filtered = non_max_suppression(
        predictions_tensor,
        conf_thres=confidence_threshold,
        iou_thres=iou_threshold,
        max_det=top_k,
    )

    if not filtered or filtered[0] is None or len(filtered[0]) == 0:
        return []

    detections = filtered[0].cpu()
    detections[:, :4] = scale_boxes(
        (input_size[1], input_size[0]),
        detections[:, :4],
        (orig_size[1], orig_size[0]),
    ).round()

    results: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det.tolist()
        class_id = int(class_id)
        results.append(
            Detection(
                class_id=class_id,
                class_name=class_names[class_id] if class_id < len(
                    class_names) else str(class_id),
                score=float(score),
                box=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return results
