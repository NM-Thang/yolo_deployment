import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import shutil
from pathlib import Path
import torch
from ultralytics import YOLO
def export_models():

    # Force initialize CUDA and verify status
    print(f"--- GPU DIAGNOSTICS ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("WARNING: CUDA not detected via torch. Attempting force init...")
        try:
            torch.cuda.init()
            device = 0
            print("CUDA force initialized successfully.")
        except Exception as e:
            print(f"ERROR: Could not initialize CUDA: {e}")
            device = 'cpu'
    print(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"-----------------------\n")

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
        simplify=True,  #optimize model
        device=device
    )
    
    # Convert to TensorRT
    print("\n[3/3] Exporting to TensorRT (FP16)...")
    model.export(
        format="engine",
        dynamic=True,
        half=True, #FP32 -> FP16
        workspace=4,
        device=device
    )
    print(f"\nModel export completed! Files are saved in: {model_dir}")

if __name__ == "__main__":
    export_models()