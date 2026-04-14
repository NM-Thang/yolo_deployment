import argparse
import os
from pathlib import Path

import onnxruntime as ort

from utils.image_processing import preprocess_image


def _build_providers(provider: str) -> list[str]:
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def test_onnx_local(model_path: Path, image_path: Path, provider: str = "cpu", gpu_id: int = 0):
    if provider == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
    
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=_build_providers(provider))

    input_tensor = preprocess_image(image_path)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    print(f"Provider mode: {provider}")
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")
    print("Running ONNX inference...")
    outputs = session.run(None, {input_name: input_tensor})
    print(f"Success! Number of outputs: {len(outputs)}")
    for idx, output in enumerate(outputs, start=1):
        print(f"  Output {idx} shape: {output.shape}")


def infer_ultralytics_export(project_root: Path, image_path: Path, provider: str, gpu_id: int, model_path: Path | None = None):
    selected_model = model_path or (project_root / "models" / "yolov8n.onnx")
    print("Mode: ultralytics")
    test_onnx_local(model_path=selected_model, image_path=image_path, provider=provider, gpu_id=gpu_id)


def infer_torch_export(project_root: Path, image_path: Path, provider: str, gpu_id: int, model_path: Path | None = None):
    selected_model = model_path or (project_root / "models" / "yolov8n_torch.onnx")
    print("Mode: torch")
    test_onnx_local(model_path=selected_model, image_path=image_path, provider=provider, gpu_id=gpu_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local inference for ONNX models")
    parser.add_argument("--mode", choices=("ultralytics", "torch"), default="ultralytics")
    parser.add_argument("--model-path", default=None, help="Optional custom ONNX model path")
    parser.add_argument(
        "--image-path",
        default="data/coco8/images/val/000000000049.jpg",
        help="Image path used for inference",
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID for CUDA provider")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = project_root / image_path

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path

    if args.mode == "ultralytics":
        infer_ultralytics_export(project_root, image_path=image_path, provider=args.provider, gpu_id=args.gpu_id, model_path=model_path)
    else:
        infer_torch_export(project_root, image_path=image_path, provider=args.provider, gpu_id=args.gpu_id, model_path=model_path)