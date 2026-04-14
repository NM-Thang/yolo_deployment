from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from utils.image_processing import preprocess_image


COCO_CLASSES = (
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


def _load_triton_client(protocol: str, server_url: str):
	if protocol == "grpc":
		tritonclient = importlib.import_module("tritonclient.grpc")
	elif protocol == "http":
		tritonclient = importlib.import_module("tritonclient.http")
	else:
		raise ValueError("protocol must be 'grpc' or 'http'")

	client_cls = tritonclient.InferenceServerClient
	return tritonclient, client_cls(url=server_url, verbose=False)


def _get_model_tensors(client, model_name: str) -> tuple[str, str]:
	metadata = client.get_model_metadata(model_name)
	if not metadata.inputs or not metadata.outputs:
		raise RuntimeError(f"Model '{model_name}' does not expose inputs/outputs")
	return metadata.inputs[0].name, metadata.outputs[0].name


def _prepare_image(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Cannot read image from {image_path}")
	height, width = image.shape[:2]
	return preprocess_image(image_path), (width, height)


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


def _xywh_to_xyxy(box: np.ndarray) -> tuple[float, float, float, float]:
	x_center, y_center, width, height = box.tolist()
	half_width = width / 2.0
	half_height = height / 2.0
	return (
		x_center - half_width,
		y_center - half_height,
		x_center + half_width,
		y_center + half_height,
	)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	left = max(box_a[0], box_b[0])
	top = max(box_a[1], box_b[1])
	right = min(box_a[2], box_b[2])
	bottom = min(box_a[3], box_b[3])

	inter_width = max(0.0, right - left)
	inter_height = max(0.0, bottom - top)
	inter_area = inter_width * inter_height

	area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
	area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
	union_area = area_a + area_b - inter_area
	if union_area <= 0.0:
		return 0.0
	return inter_area / union_area


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
	selected: list[Detection] = []
	for class_id in sorted({det.class_id for det in detections}):
		class_detections = sorted(
			(det for det in detections if det.class_id == class_id),
			key=lambda det: det.score,
			reverse=True,
		)

		while class_detections:
			best = class_detections.pop(0)
			selected.append(best)
			class_detections = [
				det for det in class_detections if _box_iou(np.array(best.box), np.array(det.box)) < iou_threshold
			]

	return sorted(selected, key=lambda det: det.score, reverse=True)


def postprocess_yolo(
	raw_output: np.ndarray,
	image_size: tuple[int, int],
	confidence_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	top_k: int = 20,
) -> list[Detection]:
	predictions = _normalize_output(raw_output)
	width, height = image_size
	scale_x = width / 640.0
	scale_y = height / 640.0

	detections: list[Detection] = []
	class_scores = predictions[:, 4:]

	for row_index, row in enumerate(predictions):
		scores = class_scores[row_index]
		class_id = int(np.argmax(scores))
		score = float(scores[class_id])
		if score < confidence_threshold:
			continue

		x1, y1, x2, y2 = _xywh_to_xyxy(row[:4])
		box = (
			max(0.0, x1 * scale_x),
			max(0.0, y1 * scale_y),
			min(float(width), x2 * scale_x),
			min(float(height), y2 * scale_y),
		)
		class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
		detections.append(Detection(class_id=class_id, class_name=class_name, score=score, box=box))

	detections = _nms(detections, iou_threshold)
	return detections[:top_k]


def run_triton_inference(
	server_url: str,
	model_name: str,
	image_path: Path,
	protocol: str = "grpc",
	input_name: str | None = None,
	output_name: str | None = None,
	confidence_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	top_k: int = 20,
) -> list[Detection]:
	tritonclient, client = _load_triton_client(protocol, server_url)

	if not client.is_server_ready():
		raise RuntimeError(f"Triton server is not ready at {server_url}")
	if not client.is_model_ready(model_name):
		raise RuntimeError(f"Model '{model_name}' is not ready on Triton server")

	inferred_input_name, inferred_output_name = _get_model_tensors(client, model_name)
	input_name = input_name or inferred_input_name
	output_name = output_name or inferred_output_name

	image_tensor, image_size = _prepare_image(image_path)

	if protocol == "grpc":
		infer_input = tritonclient.InferInput(input_name, image_tensor.shape, "FP32")
		infer_input.set_data_from_numpy(image_tensor)
		requested_output = tritonclient.InferRequestedOutput(output_name)
		response = client.infer(model_name, inputs=[infer_input], outputs=[requested_output])
		raw_output = response.as_numpy(output_name)
	else:
		infer_input = tritonclient.InferInput(input_name, image_tensor.shape, "FP32")
		infer_input.set_data_from_numpy(image_tensor)
		requested_output = tritonclient.InferRequestedOutput(output_name)
		response = client.infer(model_name, inputs=[infer_input], outputs=[requested_output])
		raw_output = response.as_numpy(output_name)

	if raw_output is None:
		raise RuntimeError(f"Model '{model_name}' returned no output named '{output_name}'")

	return postprocess_yolo(
		raw_output=raw_output,
		image_size=image_size,
		confidence_threshold=confidence_threshold,
		iou_threshold=iou_threshold,
		top_k=top_k,
	)


def _print_detections(detections: Iterable[Detection]) -> None:
	detections = list(detections)
	if not detections:
		print("No objects detected.")
		return

	print(f"Detected {len(detections)} objects:")
	for index, det in enumerate(detections, start=1):
		x1, y1, x2, y2 = det.box
		print(
			f"{index:02d}. {det.class_name} | score={det.score:.3f} | "
			f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
		)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run YOLOv8 inference through Triton")
	parser.add_argument("--server-url", default="localhost:8001", help="Triton server address")
	parser.add_argument("--model-name", default="yolov8_onnx", help="Triton model name")
	parser.add_argument(
		"--image-path",
		default="data/coco8/images/val/000000000049.jpg",
		help="Image path to run inference on",
	)
	parser.add_argument("--protocol", choices=("grpc", "http"), default="grpc", help="Triton protocol")
	parser.add_argument("--input-name", default=None, help="Optional override for model input name")
	parser.add_argument("--output-name", default=None, help="Optional override for model output name")
	parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")
	parser.add_argument("--top-k", type=int, default=20, help="Maximum number of detections to print")
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parent.parent
	image_path = Path(args.image_path)
	if not image_path.is_absolute():
		image_path = project_root / image_path

	detections = run_triton_inference(
		server_url=args.server_url,
		model_name=args.model_name,
		image_path=image_path,
		protocol=args.protocol,
		input_name=args.input_name,
		output_name=args.output_name,
		confidence_threshold=args.conf_threshold,
		iou_threshold=args.iou_threshold,
		top_k=args.top_k,
	)
	_print_detections(detections)


if __name__ == "__main__":
	main()
