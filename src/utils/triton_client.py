from __future__ import annotations

import argparse
import importlib
import numpy as np

class TritonClient:
	def __init__(self):
		args = argparse_args()
		self.host = args.host
		self.protocol = args.protocol
		self.tritonclient, self.client = self.load_triton_client()

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
	

	# def check_server_and_model(self):
	# 	if not self.client.is_server_ready():
	# 		return False
	# 	if not self.client.is_model_ready(self.model_name, self.model_version):
	# 		return False
		
	# 	return True
		

	# def get_model_tensors(self,) -> tuple[str, str]:
	# 	metadata = self.client.get_model_metadata(self.model_name, self.model_version)
	# 	if not metadata.inputs or not metadata.outputs:
	# 		raise RuntimeError(f"Model '{self.model_name}' does not expose inputs/outputs")
		
	# 	return metadata.inputs[0].name, metadata.outputs[0].name


	def run(self, input: np.ndarray, model_name: str, model_version: str ="", input_name: str = "images", output_name: str = "output0") -> np.ndarray:

		if input is None:
			raise ValueError("batch_input is required for inference")
		input = input.cpu().numpy() if hasattr(input, "cpu") else input

		image_tensor = input
		infer_input = self.tritonclient.InferInput(input_name, image_tensor.shape, "FP32")
		infer_input.set_data_from_numpy(image_tensor)

		requested_output = self.tritonclient.InferRequestedOutput(output_name)
		response = self.client.infer(model_name, inputs=[infer_input], outputs=[requested_output], model_version=model_version)
		raw_output = response.as_numpy(output_name)

		if raw_output is None:
			raise RuntimeError(f"Model '{model_name}' returned no output named '{output_name}'")
		
		return raw_output
	
def argparse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Triton Client for YOLOv8 Inference")
	parser.add_argument("--host", type=str, default="localhost", help="Triton server host")
	parser.add_argument("--protocol", type=str, choices=["grpc", "http"], default="grpc", help="Protocol to use for communication with Triton server")
	
	return parser.parse_known_args()[0]	