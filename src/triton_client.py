from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import numpy as np

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

def run(batch_input: np.ndarray = None, host: str = "localhost", protocol: str = "grpc", model_name: str = "yolov8_onnx", model_version: str = None) -> np.ndarray:
	tritonclient, client = load_triton_client(protocol=protocol, server_url=host)
	if model_version is None:
		model_version = ""
	if not client.is_server_ready():
		raise RuntimeError(f"Triton server is not ready at {host}")
	if not client.is_model_ready(model_name, model_version):
		raise RuntimeError(f"Model '{model_name}' is not ready on Triton server")

	inferred_input_name, inferred_output_name = get_model_tensors(client, model_name, model_version)

	if batch_input is None:
		raise ValueError("batch_input is required for inference")

	image_tensor = batch_input
	infer_input = tritonclient.InferInput(inferred_input_name, image_tensor.shape, "FP32")
	infer_input.set_data_from_numpy(image_tensor)

	requested_output = tritonclient.InferRequestedOutput(inferred_output_name)
	response = client.infer(model_name, inputs=[infer_input], outputs=[requested_output], model_version=model_version)
	raw_output = response.as_numpy(inferred_output_name)

	if raw_output is None:
		raise RuntimeError(f"Model '{model_name}' returned no output named '{inferred_output_name}'")
	
	return raw_output