from __future__ import annotations

import argparse
import importlib
import numpy as np

class AsyncTritonClient:
    def __init__(self):
        args = argparse_args()
        self.host = args.host
        self.protocol = args.protocol
        self.model_name = args.model_name
        self.model_version = args.model_version

        self.tritonclient = None
        self.client = None
        self.inferred_input_name = None
        self.inferred_output_name = None

    @classmethod
    async def create(cls) -> "AsyncTritonClient":
        """
        Factory method to initialize the asynchronous client.
        Must be used because __init__ cannot be async.
        
        Usage: client = await AsyncTritonClient.create()
        """
        instance = cls()
        
        instance.tritonclient, instance.client = instance._load_async_client()
        
        await instance._check_server_and_model()
        instance.inferred_input_name, instance.inferred_output_name = await instance._get_model_tensors()
        
        return instance

    def _load_async_client(self):
        """
        Dynamically loads the AIO (Asynchronous I/O) version of the Triton client.
        """
        if self.protocol == "grpc":
            tritonclient = importlib.import_module("tritonclient.grpc.aio")
            default_port = 8001
        elif self.protocol == "http":
            tritonclient = importlib.import_module("tritonclient.http.aio")
            default_port = 8000
        else:
            raise ValueError("protocol must be 'grpc' or 'http'")
            
        url = f"{self.host}:{default_port}"
        
        return tritonclient, tritonclient.InferenceServerClient(url=url, verbose=False)
    
    async def _check_server_and_model(self):
        """
        Asynchronously checks if the server and the specific model are ready.
        """
        if not await self.client.is_server_ready():
            raise RuntimeError("Triton server is not ready")
        
        if not await self.client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model '{self.model_name}' is not ready")

    async def _get_model_tensors(self) -> tuple[str, str]:
        """
        Asynchronously fetches model metadata to determine input and output names.
        """
        metadata = await self.client.get_model_metadata(self.model_name, self.model_version)
        
        if not metadata.inputs or not metadata.outputs:
            raise RuntimeError(f"Model '{self.model_name}' does not expose inputs/outputs")
        
        return metadata.inputs[0].name, metadata.outputs[0].name

    async def async_run(self, batch_input: np.ndarray) -> np.ndarray:
        """
        Sends an asynchronous inference request to the Triton server without blocking the event loop.
        """
        if batch_input is None:
            raise ValueError("batch_input is required for inference")
            
        if batch_input.ndim == 3:
            batch_input = np.expand_dims(batch_input, axis=0)

        infer_input = self.tritonclient.InferInput(self.inferred_input_name, batch_input.shape, "FP32")
        infer_input.set_data_from_numpy(batch_input)

        requested_output = self.tritonclient.InferRequestedOutput(self.inferred_output_name)
        
        response = await self.client.infer(
            self.model_name, 
            inputs=[infer_input], 
            outputs=[requested_output], 
            model_version=self.model_version
        )
        
        raw_output = response.as_numpy(self.inferred_output_name)
        
        if raw_output is None:
            raise RuntimeError(f"Model '{self.model_name}' returned no output named '{self.inferred_output_name}'")
            
        return raw_output


def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton Client for YOLOv8 Inference")
    parser.add_argument("--host", type=str, default="localhost", help="Triton server host (default: localhost)")
    parser.add_argument("--protocol", type=str, choices=["grpc", "http"], default="grpc", help="Protocol to use for communication (default: grpc)")
    parser.add_argument("--model-name", type=str, default="yolov8_trt", help="Name of the model (default: yolov8_trt)")
    parser.add_argument("--model-version", type=str, default="", help="Version of the model (default: '')")
    
    return parser.parse_known_args()[0]