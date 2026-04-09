import onnxruntime as ort
from pathlib import Path
from utils.image_processing import preprocess_image

def test_onnx_local(model_path: Path, image_path: Path):
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    input_tensor = preprocess_image(image_path)
    input_name = session.get_inputs()[0].name
    
    print("Running ONNX inference...")
    outputs = session.run(None, {input_name: input_tensor})
    print(f"Success! ONNX output shape: {outputs[0].shape}")

if __name__ == "__main__":
    # Dynamically resolve absolute paths
    project_root = Path(__file__).resolve().parent.parent
    
    test_onnx_local(
        model_path=project_root / "models" / "yolov8n.onnx", 
        image_path=project_root / "data" / "coco8/images/val/000000000049.jpg"
    )