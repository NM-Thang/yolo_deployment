import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def export_models(export_onnx: bool, export_engine: bool, use_half: bool):
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
    
    if not pt_path.exists():
        print("[1/3] File yolov8n.pt not found in models/ directory. Downloading...")
        _ = YOLO("yolov8n.pt") 
        
        if Path("yolov8n.pt").exists():
            shutil.move("yolov8n.pt", pt_path)

    model = YOLO(str(pt_path))

    if export_onnx:
        print("\n[2/3] Exporting to ONNX (Dynamic)...")
        model.export(
            format="onnx",
            dynamic=True,  # Enable dynamic shape/batch
            simplify=True, # Optimize the ONNX graph
            opset=12,
            device=device
        )
    else:
        print("\n[2/3] Skipping ONNX export (--onnx flag not provided).")
        
    if export_engine:
        print("\n[3/3] Exporting to TensorRT (FP16)...")
        try:
            model.export(
                format="engine",
                dynamic=True,
                half=use_half,  # Convert FP32 to FP16 for speed optimization
                workspace=4,   # Max workspace size in GB
                device=device  # Explicitly assign GPU device
            )
        except Exception as e:
             print(f"ERROR: TensorRT export failed (likely due to missing NVIDIA GPU). Details: {e}")
    else:
        print("\n[3/3] Skipping TensorRT export (--engine flag not provided).")

    print(f"\nModel export process finished! Target directory: {model_dir}")

if __name__ == "__main__":
    # Setup Argument Parser for CLI flags
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX and/or TensorRT formats.")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    parser.add_argument("--engine", action="store_true", help="Export to TensorRT format")
    parser.add_argument("--all", action="store_true", help="Export to BOTH formats")
    parser.add_argument("--half", action="store_true", help="Export TensorRT in FP16 precision (default is FP32)")
    args = parser.parse_args()
    
    # Determine which formats to export based on user flags
    do_onnx = args.onnx or args.all
    do_engine = args.engine or args.all
    
    # If the user runs the script without ANY flags, default to doing BOTH
    if not (do_onnx or do_engine):
        print("WARNING: No export format specified. Defaulting to BOTH (--all).")
        print("HINT: Use --onnx or --engine to export specifically.\n")
        do_onnx = True
        do_engine = True

    # Call the main function with the parsed flags
    export_models(export_onnx=do_onnx, export_engine=do_engine, use_half=args.half)