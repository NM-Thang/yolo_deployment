from pathlib import Path
from ultralytics import YOLO

def test_tensorrt_local(engine_path: Path, image_path: Path):
    print(f"Loading TensorRT model from: {engine_path}")
    trt_model = YOLO(str(engine_path))
    
    print("Running TensorRT inference...")
    results = trt_model(str(image_path), verbose=False)
    print(f"Success! Found {len(results[0].boxes)} objects.")

if __name__ == "__main__":
    # Dynamically resolve absolute paths
    project_root = Path(__file__).resolve().parent.parent
    
    test_tensorrt_local(
        engine_path=project_root / "models" / "yolov8n.engine", 
        image_path=project_root / "data" / "coco8/images/val/000000000049.jpg"
    )