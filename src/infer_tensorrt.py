import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from utils.inference_io import collect_image_paths, preprocess_batch, resolve_path


def _load_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python module not found. Install TensorRT Python bindings in this environment."
        ) from exc


def _load_engine(engine_path: Path) -> tuple[Any, Any]:
    trt = _load_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return trt, engine


def _infer_input_hw(input_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(input_shape) == 4:
        h, w = input_shape[2], input_shape[3]
        if h > 0 and w > 0:
            return int(w), int(h)
    return (640, 640)


def _resolve_effective_batch_size(input_shape: tuple[int, ...], requested_batch_size: int, total_images: int) -> int:
    if len(input_shape) < 4:
        return 1

    batch_dim = input_shape[0]
    if batch_dim in (-1, 0):
        return max(1, requested_batch_size)

    fixed_batch = int(batch_dim)
    if total_images > 1 and fixed_batch == 1:
        print("Engine has fixed batch size = 1; falling back to one image per inference call.")
        return 1
    return fixed_batch


def _binding_index(engine: Any, tensor_name: str) -> int:
    for idx in range(engine.num_bindings):
        if engine.get_binding_name(idx) == tensor_name:
            return idx
    raise ValueError(f"Tensor name not found in engine bindings: {tensor_name}")


def _run_engine_batch(trt: Any, engine: Any, context: Any, batch_tensor: np.ndarray) -> list[np.ndarray]:
    try:
        importlib.import_module("pycuda.autoinit")
        cuda = importlib.import_module("pycuda.driver")
    except ImportError as exc:
        raise ImportError(
            "Missing PyCUDA. Install it in this environment to run TensorRT local inference."
        ) from exc

    input_idx = None
    output_indices: list[int] = []
    for idx in range(engine.num_bindings):
        if engine.binding_is_input(idx):
            input_idx = idx
        else:
            output_indices.append(idx)

    if input_idx is None:
        raise RuntimeError("No input binding found in TensorRT engine.")

    # Ensure batch_tensor is contiguous and correct dtype
    input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
    input_host = np.ascontiguousarray(batch_tensor.astype(input_dtype, copy=False))
    
    print(f"  Input batch shape: {input_host.shape}, dtype: {input_host.dtype}, nbytes: {input_host.nbytes}")
    
    if not context.set_binding_shape(input_idx, tuple(input_host.shape)):
        raise RuntimeError(f"Failed to set input shape: {tuple(input_host.shape)}")

    device_buffers: list[Any] = [None] * engine.num_bindings

    # Allocate output buffers AFTER setting input shape
    host_outputs: list[np.ndarray] = []
    output_devices = []
    for out_idx in output_indices:
        out_shape = tuple(context.get_binding_shape(out_idx))
        out_dtype = trt.nptype(engine.get_binding_dtype(out_idx))
        print(f"  Output {out_idx} shape: {out_shape}, dtype: {out_dtype}")
        out_host = np.empty(out_shape, dtype=out_dtype)
        out_device = cuda.mem_alloc(out_host.nbytes)

        host_outputs.append(out_host)
        output_devices.append(out_device)
        device_buffers[out_idx] = out_device

    # Now allocate input buffer and set binding
    input_device = cuda.mem_alloc(input_host.nbytes) 
    device_buffers[input_idx] = input_device
    print(f"  Allocated input GPU memory: {input_host.nbytes} bytes")

    cuda.memcpy_htod(input_device, input_host)
    if not context.execute_v2(bindings=device_buffers):
        raise RuntimeError("TensorRT execute_v2 failed.")

    for host_output, out_device in zip(host_outputs, output_devices):
        cuda.memcpy_dtoh(host_output, out_device)
    return host_outputs


def test_tensorrt_local(
    engine_path: Path,
    image_path: Path,
    image_dir: Path | None = None,
    batch_size: int = 8,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine path does not exist: {engine_path}")

    print(f"Loading TensorRT engine from: {engine_path}")
    trt, engine = _load_engine(engine_path)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    input_name = next(engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i))
    output_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    input_idx = _binding_index(engine, input_name)
    input_shape = tuple(engine.get_binding_shape(input_idx))

    image_paths = collect_image_paths(image_path=image_path, image_dir=image_dir)
    input_size = _infer_input_hw(input_shape)
    effective_batch_size = _resolve_effective_batch_size(input_shape, batch_size, len(image_paths))

    print(f"Input name: {input_name}")
    print(f"Input shape (engine): {input_shape}")
    print(f"Output names: {output_names}")
    print(f"Preprocess input size (W, H): {input_size}")
    print(f"Total images: {len(image_paths)}")
    print(f"Effective batch size: {effective_batch_size}")

    for batch_start in range(0, len(image_paths), effective_batch_size):
        batch_paths = image_paths[batch_start : batch_start + effective_batch_size]
        batch_tensor = preprocess_batch(batch_paths, input_size=input_size)

        print(
            f"Running TensorRT inference for batch "
            f"{batch_start // effective_batch_size + 1} with {len(batch_paths)} image(s)..."
        )
        outputs = _run_engine_batch(trt=trt, engine=engine, context=context, batch_tensor=batch_tensor)
        for idx, output in enumerate(outputs, start=1):
            print(f"  Output {idx} shape: {output.shape}, dtype: {output.dtype}")


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local TensorRT (.plan) inference")
    parser.add_argument("--engine-path", default="models/tensorrt/2/model.plan", help="Path to TensorRT .plan engine")
    parser.add_argument(
        "--image-path",
        default="data/coco8/images/val/000000000049.jpg",
        help="Image path used for inference",
    )
    parser.add_argument("--image-dir", default=None, help="Optional directory of images to infer")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size when --image-dir is used")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    engine_path = resolve_path(args.engine_path, project_root)
    if engine_path is None:
        raise ValueError("--engine-path is required")

    image_path = resolve_path(args.image_path, project_root)
    if image_path is None:
        raise ValueError("--image-path is required")

    image_dir = resolve_path(args.image_dir, project_root)

    test_tensorrt_local(
        engine_path=engine_path,
        image_path=image_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
    )