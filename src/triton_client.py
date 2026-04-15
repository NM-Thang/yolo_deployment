from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from utils.image_processing import preprocess_image
from utils.inference_io import collect_image_paths


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

CLASS_NAMES = DEFAULT_COCO_CLASSES


@dataclass(frozen=True)
class Detection:
	class_id: int
	class_name: str
	score: float
	box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelTarget:
	model_name: str
	model_version: str | None = None


def _load_triton_client(protocol: str, server_url: str):
	if protocol == "grpc":
		tritonclient = importlib.import_module("tritonclient.grpc")
	elif protocol == "http":
		tritonclient = importlib.import_module("tritonclient.http")
	else:
		raise ValueError("protocol must be 'grpc' or 'http'")

	client_cls = tritonclient.InferenceServerClient
	return tritonclient, client_cls(url=server_url, verbose=False)


def _get_model_tensors(client, model_name: str, model_version: str | None = None) -> tuple[str, str]:
	version = model_version or ""
	metadata = client.get_model_metadata(model_name, model_version=version)
	if not metadata.inputs or not metadata.outputs:
		raise RuntimeError(f"Model '{model_name}' does not expose inputs/outputs")
	return metadata.inputs[0].name, metadata.outputs[0].name


def _prepare_image(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Cannot read image from {image_path}")
	height, width = image.shape[:2]
	return preprocess_image(image_path), (width, height)


def _load_class_names_from_yaml(data_yaml: Path) -> tuple[str, ...]:
	try:
		yaml = importlib.import_module("yaml")
	except ImportError:
		print("Warning: PyYAML is not installed, using default COCO classes.")
		return DEFAULT_COCO_CLASSES

	if not data_yaml.exists():
		print(f"Warning: data yaml not found: {data_yaml}. Using default COCO classes.")
		return DEFAULT_COCO_CLASSES

	try:
		with open(data_yaml, "r", encoding="utf-8") as file:
			data = yaml.safe_load(file) or {}
	except Exception as exc:
		print(f"Warning: cannot read {data_yaml}: {exc}. Using default COCO classes.")
		return DEFAULT_COCO_CLASSES

	names = data.get("names")
	if isinstance(names, dict):
		try:
			ordered = [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
		except Exception:
			ordered = [str(value) for value in names.values()]
		return tuple(ordered) if ordered else DEFAULT_COCO_CLASSES

	if isinstance(names, list):
		ordered = [str(value) for value in names]
		return tuple(ordered) if ordered else DEFAULT_COCO_CLASSES

	print(f"Warning: 'names' not found in {data_yaml}. Using default COCO classes.")
	return DEFAULT_COCO_CLASSES


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
		class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
		detections.append(Detection(class_id=class_id, class_name=class_name, score=score, box=box))

	detections = _nms(detections, iou_threshold)
	return detections[:top_k]


def run_triton_inference(
	server_url: str,
	model_name: str,
	image_path: Path,
	protocol: str = "grpc",
	model_version: str | None = None,
	input_name: str | None = None,
	output_name: str | None = None,
	confidence_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	top_k: int = 20,
) -> np.ndarray:
	tritonclient, client = _load_triton_client(protocol, server_url)

	if not client.is_server_ready():
		raise RuntimeError(f"Triton server is not ready at {server_url}")
	version = model_version or ""
	if not client.is_model_ready(model_name, model_version=version):
		raise RuntimeError(
			f"Model '{model_name}' (version='{version or 'latest'}') is not ready on Triton server"
		)

	inferred_input_name, inferred_output_name = _get_model_tensors(client, model_name, model_version=version)
	input_name = input_name or inferred_input_name
	output_name = output_name or inferred_output_name

	image_tensor, image_size = _prepare_image(image_path)

	infer_input = tritonclient.InferInput(input_name, image_tensor.shape, "FP32")
	infer_input.set_data_from_numpy(image_tensor)
	requested_output = tritonclient.InferRequestedOutput(output_name)
	response = client.infer(model_name, model_version=version, inputs=[infer_input], outputs=[requested_output])
	raw_output = response.as_numpy(output_name)

	if raw_output is None:
		raise RuntimeError(f"Model '{model_name}' returned no output named '{output_name}'")

	# Return raw Triton output directly for debugging.
	return raw_output


def run_multi_model_inference(
	server_url: str,
	model_targets: list[ModelTarget],
	image_path: Path,
	protocol: str,
	input_name: str | None,
	output_name: str | None,
	confidence_threshold: float,
	iou_threshold: float,
	top_k: int,
) -> list[tuple[ModelTarget, np.ndarray]]:
	results: list[tuple[ModelTarget, np.ndarray]] = []
	for target in model_targets:
		raw_output = run_triton_inference(
			server_url=server_url,
			model_name=target.model_name,
			model_version=target.model_version,
			image_path=image_path,
			protocol=protocol,
			input_name=input_name,
			output_name=output_name,
			confidence_threshold=confidence_threshold,
			iou_threshold=iou_threshold,
			top_k=top_k,
		)
		results.append((target, raw_output))
	return results


def _print_raw_output(raw_output: np.ndarray) -> None:
	raw_output = np.asarray(raw_output)
	print(f"Raw output shape: {raw_output.shape}, dtype: {raw_output.dtype}")
	flat = raw_output.reshape(-1)
	sample_count = min(10, flat.size)
	print(f"Raw output sample (first {sample_count}): {flat[:sample_count]}")


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
		"--model-version",
		default=None,
		help="Optional model version (for TensorRT model with multiple versions)",
	)
	parser.add_argument(
		"--run-three",
		action="store_true",
		help="Run 3 targets: yolov8_onnx latest, yolov8_trt version 1, yolov8_trt version 2",
	)
	parser.add_argument(
		"--image-path",
		default="data/coco8/images/val/000000000049.jpg",
		help="Image path to run inference on",
	)
	parser.add_argument("--image-dir", default=None, help="Optional directory of images to run inference on")
	parser.add_argument("--data-yaml", default="data/coco8.yaml", help="Dataset yaml file that contains class names")
	parser.add_argument("--protocol", choices=("grpc", "http"), default="grpc", help="Triton protocol")
	parser.add_argument("--input-name", default=None, help="Optional override for model input name")
	parser.add_argument("--output-name", default=None, help="Optional override for model output name")
	parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")
	parser.add_argument("--top-k", type=int, default=20, help="Maximum number of detections to print")
	return parser


def _run_for_single_image(args: argparse.Namespace, image_path: Path) -> None:
	if args.run_three:
		targets = [
			ModelTarget(model_name="yolov8_onnx", model_version=None),
			ModelTarget(model_name="yolov8_trt", model_version="1"),
			ModelTarget(model_name="yolov8_trt", model_version="2"),
		]
		all_results = run_multi_model_inference(
			server_url=args.server_url,
			model_targets=targets,
			image_path=image_path,
			protocol=args.protocol,
			input_name=args.input_name,
			output_name=args.output_name,
			confidence_threshold=args.conf_threshold,
			iou_threshold=args.iou_threshold,
			top_k=args.top_k,
		)

		for target, raw_output in all_results:
			version_text = target.model_version or "latest"
			print(f"\n=== Model: {target.model_name} | Version: {version_text} ===")
			_print_raw_output(raw_output)
	else:
		raw_output = run_triton_inference(
			server_url=args.server_url,
			model_name=args.model_name,
			model_version=args.model_version,
			image_path=image_path,
			protocol=args.protocol,
			input_name=args.input_name,
			output_name=args.output_name,
			confidence_threshold=args.conf_threshold,
			iou_threshold=args.iou_threshold,
			top_k=args.top_k,
		)
		_print_raw_output(raw_output)


def main() -> None:
	global CLASS_NAMES

	parser = build_arg_parser()
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parent.parent
	data_yaml = Path(args.data_yaml)
	if not data_yaml.is_absolute():
		data_yaml = project_root / data_yaml
	CLASS_NAMES = _load_class_names_from_yaml(data_yaml)
	print(f"Loaded {len(CLASS_NAMES)} class names.")

	image_path = Path(args.image_path)
	if not image_path.is_absolute():
		image_path = project_root / image_path

	image_dir = Path(args.image_dir) if args.image_dir else None
	if image_dir is not None and not image_dir.is_absolute():
		image_dir = project_root / image_dir

	image_paths = collect_image_paths(image_path=image_path, image_dir=image_dir)
	print(f"Total images: {len(image_paths)}")

	for idx, current_image_path in enumerate(image_paths, start=1):
		print(f"\n========== Image {idx}/{len(image_paths)}: {current_image_path} ==========")
		_run_for_single_image(args, current_image_path)


if __name__ == "__main__":
	main()
