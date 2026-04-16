from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


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
		output = output.T

	if output.shape[1] < 5:
		raise ValueError(f"YOLO output has too few columns: {output.shape}")

	return output.astype(np.float32, copy=False)


def postprocess_yolo(
	raw_output: np.ndarray,
	image_size: tuple[int, int],
	confidence_threshold: float = 0.5,
	iou_threshold: float = 0.45,
	top_k: int = 10,
	class_names: tuple[str, ...] = DEFAULT_COCO_CLASSES,
) -> list[Detection]:
	predictions = _normalize_output(raw_output)
	# Ultralytics NMS expects [batch, num_boxes, 4 + num_classes]
	predictions_tensor = torch.from_numpy(predictions.T[None, ...])
	filtered = non_max_suppression(
		predictions_tensor,
		conf_thres=confidence_threshold,
		iou_thres=iou_threshold,
		max_det=top_k,
	)

	if not filtered or filtered[0] is None or len(filtered[0]) == 0:
		return []

	detections = filtered[0].cpu()
	detections[:, :4] = scale_boxes((640, 640), detections[:, :4], (image_size[1], image_size[0])).round()

	results: list[Detection] = []
	for det in detections:
		x1, y1, x2, y2, score, class_id = det.tolist()
		class_id = int(class_id)
		results.append(
			Detection(
				class_id=class_id,
				class_name=class_names[class_id] if class_id < len(class_names) else str(class_id),
				score=float(score),
				box=(float(x1), float(y1), float(x2), float(y2)),
			)
		)
	return results