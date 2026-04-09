import shutil
from pathlib import Path
from ultralytics import YOLO

def export_models():
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = model_dir / "yolov8n.pt"
    
    print("[1/3] Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt") 
    
    if not pt_path.exists():
        shutil.move("yolov8n.pt", pt_path) 
    

    model = YOLO(str(pt_path))

    # Convert to ONNX
    print("\n[2/3] Exporting to ONNX (Dynamic)...")
    model.export(
        format="onnx",
        dynamic=True,  #dynamic shape
        simplify=True  #optimize model
    )
    
    # Convert to TensorRT
    print("\n[3/3] Exporting to TensorRT (FP16)...")
    model.export(
        format="engine",
        dynamic=True,
        half=True, #FP32 -> FP16
        workspace=4
    )
    print(f"\nModel export completed! Files are saved in: {model_dir}")

if __name__ == "__main__":
    export_models()