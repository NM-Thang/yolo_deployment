import argparse
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort

from utils.inference_io import collect_image_paths, resolve_path
from utils.img_processing import batch_preprocessing

DEFAULT_MODELS = {
    "ultralytics": "yolov8n.onnx",
    "torch": "yolov8n_torch.onnx",
}


def _build_providers(provider: str) -> list[str]:
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def test_onnx_local(
    model_path: Path,
    image_path: Path,
    provider: str = "cpu",
    gpu_id: int = 0,
    image_dir: Path | None = None,
    batch_size: int = 8,
):
    if provider == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

    print(f"Loading ONNX model from: {model_path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), options=options, providers=_build_providers(provider))

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [out.name for out in session.get_outputs()]
    image_paths = collect_image_paths(image_path=image_path, image_dir=image_dir)

    print(f"Provider mode: {provider}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Output names: {output_names}")
    print(f"Total images: {len(image_paths)}")

    supports_dynamic_batch = len(input_shape) > 0 and input_shape[0] in (None, "None", "batch", -1)
    effective_batch_size = batch_size if supports_dynamic_batch else 1
    if not supports_dynamic_batch and len(image_paths) > 1:
        print("Model input has a fixed batch dimension; falling back to one image per inference call.")

    for batch_start in range(0, len(image_paths), effective_batch_size):
        batch_paths = image_paths[batch_start : batch_start + effective_batch_size]
        batch_tensor = batch_preprocessing(batch_paths)

        outputs = session.run(None, {input_name: batch_tensor})
        for idx, output in enumerate(outputs, start=1):
            print(f"  Output {idx} shape: {output.shape}")


def infer_by_mode(
    project_root: Path,
    mode: str,
    image_path: Path,
    provider: str,
    gpu_id: int,
    model_path: Path | None = None,
    image_dir: Path | None = None,
    batch_size: int = 8,
):
    selected_model = model_path or (project_root / "models" / DEFAULT_MODELS[mode])
    print(f"Mode: {mode}")
    test_onnx_local(
        model_path=selected_model,
        image_path=image_path,
        provider=provider,
        gpu_id=gpu_id,
        image_dir=image_dir,
        batch_size=batch_size,
    )


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local inference for ONNX models")
    parser.add_argument("--mode", choices=("ultralytics", "torch"), default="ultralytics")
    parser.add_argument("--model-path", default=None, help="Optional custom ONNX model path")
    parser.add_argument(
        "--image-path",
        default="data/coco8/images/val/000000000049.jpg",
        help="Image path used for inference",
    )
    parser.add_argument("--image-dir", default=None, help="Optional directory of images to infer")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size when --image-dir is used")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for CUDA provider")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent

    image_path = resolve_path(args.image_path, project_root)
    if image_path is None:
        raise ValueError("--image-path is required")

    model_path = resolve_path(args.model_path, project_root)
    image_dir = resolve_path(args.image_dir, project_root)

    infer_by_mode(
        project_root,
        mode=args.mode,
        image_path=image_path,
        provider=args.provider,
        gpu_id=args.gpu_id,
        model_path=model_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
    )