from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np

from utils.image_processing import preprocess_image
from utils.inference_io import resolve_path
from utils.postprocessing import postprocess_yolo
from utils.visualization import show_detections


def load_triton_client(protocol: str, server_url: str):
	if protocol == "grpc":
		tritonclient = importlib.import_module("tritonclient.grpc")
		default_port = 8001
	elif protocol == "http":
		tritonclient = importlib.import_module("tritonclient.http")
		default_port = 8000
	else:
		raise ValueError("protocol must be 'grpc' or 'http'")

	url = f"{server_url}:{default_port}"

	return tritonclient, tritonclient.InferenceServerClient(url=url, verbose=False)


def get_model_tensors(client, model_name: str, model_version: str | None) -> tuple[str, str]:
	metadata = client.get_model_metadata(model_name, model_version)
	if not metadata.inputs or not metadata.outputs:
		raise RuntimeError(f"Model '{model_name}' does not expose inputs/outputs")
	return metadata.inputs[0].name, metadata.outputs[0].name


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Send one image to Triton and print the raw output")
	parser.add_argument("--host", default="localhost", help="Triton host name or IP")
	parser.add_argument("--model-name", default="yolov8_onnx", help="Triton model name")
	parser.add_argument("--model-version", default=None, help="Triton model version (default: latest)")
	parser.add_argument("--image-path", default="data/coco8/images/val/000000000036.jpg", help="Image path to run inference on")
	parser.add_argument("--protocol", choices=("grpc", "http"), default="grpc", help="Triton protocol")
	parser.add_argument("--input-name", default=None, help="Optional override for model input name")
	parser.add_argument("--output-name", default=None, help="Optional override for model output name")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	project_root = Path(__file__).resolve().parent.parent
	image_path = resolve_path(args.image_path, project_root)
	if image_path is None:
		raise ValueError("--image-path is required")

	tritonclient, client = load_triton_client(args.protocol, args.host)
	if not client.is_server_ready():
		raise RuntimeError(f"Triton server is not ready at {args.host}")
	if not client.is_model_ready(args.model_name, args.model_version):
		raise RuntimeError(f"Model '{args.model_name}' is not ready on Triton server")

	inferred_input_name, inferred_output_name = get_model_tensors(client, args.model_name, args.model_version)
	input_name = args.input_name or inferred_input_name
	output_name = args.output_name or inferred_output_name

	original_image = cv2.imread(str(image_path))
	if original_image is None:
		raise ValueError(f"Cannot read image from {image_path}")
	orig_size = (original_image.shape[1], original_image.shape[0])

	image_tensor = preprocess_image(image_path)
	input_size = (image_tensor.shape[3], image_tensor.shape[2])
	infer_input = tritonclient.InferInput(input_name, image_tensor.shape, "FP32")
	
	infer_input.set_data_from_numpy(image_tensor)
	requested_output = tritonclient.InferRequestedOutput(output_name)
	response = client.infer(args.model_name, inputs=[infer_input], outputs=[requested_output], model_version=args.model_version)
	raw_output = response.as_numpy(output_name)
	if raw_output is None:
		raise RuntimeError(f"Model '{args.model_name}' returned no output named '{output_name}'")

	print(f"Raw output shape: {np.asarray(raw_output).shape}")
	print(f"Raw output dtype: {np.asarray(raw_output).dtype}")

	detections = postprocess_yolo(
		raw_output=raw_output,
		input_size=input_size,
		orig_size=orig_size,
		confidence_threshold=0.5,
		iou_threshold=0.45,
		top_k=20,
	)

	if not detections:
		print("No objects detected.")
		return

	print(f"Detected {len(detections)} objects:")
	for index, det in enumerate(detections, start=1):
		x1, y1, x2, y2 = det.box
		print(
			f"{index:02d}. {det.class_name} | class_id={det.class_id} | "
			f"score={det.score:.3f} | box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
		)

	show_detections(image_path, detections)


if __name__ == "__main__":
	main()