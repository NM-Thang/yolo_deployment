from __future__ import annotations

import argparse
import importlib
import numpy as np

class TritonClient:
	def __init__(self):
		args = argparse_args()
		self.host = args.host
		self.protocol = args.protocol
		self.model_name = args.model_name
		self.model_version = args.model_version

	def load_triton_client(self):
		if self.protocol == "grpc":
			tritonclient = importlib.import_module("tritonclient.grpc")
			default_port = 8001
		elif self.protocol == "http":
			tritonclient = importlib.import_module("tritonclient.http")
			default_port = 8000
		else:
			raise ValueError("protocol must be 'grpc' or 'http'")
		url = f"{self.host}:{default_port}"

		return tritonclient, tritonclient.InferenceServerClient(url=url, verbose=False)

	def get_model_tensors(self, client, model_name: str, model_version: str | None) -> tuple[str, str]:
		metadata = client.get_model_metadata(model_name, model_version)
		if not metadata.inputs or not metadata.outputs:
			raise RuntimeError(f"Model '{model_name}' does not expose inputs/outputs")
		
		return metadata.inputs[0].name, metadata.outputs[0].name

	def run(self, batch_input: np.ndarray) -> np.ndarray:
		tritonclient, client = self.load_triton_client()

		if not client.is_server_ready():
			raise RuntimeError(f"Triton server is not ready at {self.host}")
		if not client.is_model_ready(self.model_name, self.model_version):
			raise RuntimeError(f"Model '{self.model_name}' is not ready on Triton server")

		inferred_input_name, inferred_output_name = self.get_model_tensors(client, self.model_name, self.model_version)

		if batch_input is None:
			raise ValueError("batch_input is required for inference")

		image_tensor = batch_input
		infer_input = tritonclient.InferInput(inferred_input_name, image_tensor.shape, "FP32")
		infer_input.set_data_from_numpy(image_tensor)

		requested_output = tritonclient.InferRequestedOutput(inferred_output_name)
		response = client.infer(self.model_name, inputs=[infer_input], outputs=[requested_output], model_version=self.model_version)
		raw_output = response.as_numpy(inferred_output_name)

		if raw_output is None:
			raise RuntimeError(f"Model '{self.model_name}' returned no output named '{inferred_output_name}'")
		
		return raw_output
	
def argparse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Triton Client for YOLOv8 Inference")
	parser.add_argument("--host", type=str, default="localhost", help="Triton server host (default: localhost)")
	parser.add_argument("--protocol", type=str, choices=["grpc", "http"], default="grpc", help="Protocol to use for communication with Triton server (default: grpc)")
	parser.add_argument("--model-name", type=str, default="yolov8_trt", help="Name of the model to use for inference (default: yolov8_onnx)")
	parser.add_argument("--model-version", type=str, default="", help="Version of the model to use for inference (default: None)")
	
	return parser.parse_known_args()[0]	