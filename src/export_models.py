import os
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def export_models(export_onnx: bool, export_torch: bool, export_engine: bool, use_half: bool, selected_device: str | None):

    # Force initialize CUDA and verify status
    print(f"--- GPU DIAGNOSTICS ---")
    print(f"Selected device argument: {selected_device}")

    if selected_device and selected_device.lower() != "cpu":
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            device = selected_device
            try:
                print(f"Device Name: {torch.cuda.get_device_name(int(selected_device))}")
            except ValueError:
                print(f"Device Name: {selected_device}")
        else:
            print("WARNING: Selected GPU is not visible to torch. Falling back to CPU.")
            device = "cpu"
    else:
        print("No GPU selected. Using CPU by default.")
        device = "cpu"
    print(f"-----------------------\n")

    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = model_dir / "yolov8n.pt"
    model = YOLO(str(pt_path))
    model.eval()    

    if export_onnx and not export_torch:
        print("\n[1/2] Exporting to ONNX (Dynamic)... with ultralytics export method")
        model.export(
            format="onnx",
            dynamic=True,  
            simplify=True, 
            opset=18,
            device=device
        )
    elif export_torch:
        print("\n[1/2] Exporting to ONNX (Dynamic)... with torch export method")
        # batch = torch.export.Dim("batch", min=1, max=1024)

        torch.onnx.export(
            model.model.to(device),  
            torch.randn(1, 3, 640, 640).to(device), 
            str(model_dir / "yolov8n_torch.onnx"),  
            export_params=True,
            opset_version=18,        
            do_constant_folding=True,
            input_names=['input'],    
            output_names=['output'],  
            # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            # dynamic_shapes={'x': {0: batch}}

            dynamic_axes = {
                'input': {
                    0: 'batch_size', 
                    2: 'height',      
                    3: 'width'        
                },
                'output': {
                    0: 'batch_size'
                    # note: output dynamic axes can be more complex due to YOLO's variable output shape, 
                    # so we only set batch_size here. Height and width are typically fixed for ONNX export. 
                    # Adjust as needed based on your model's output structure.
                }
            }
        )
    else :
        print("\n[1/2] Skipping ONNX export (--onnx flag not provided).")
        
        
    if export_engine:
        print("\n[2/2] Exporting to TensorRT (FP16)... with ultralytics export method")
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
        print("\n[2/2] Skipping TensorRT export (--engine flag not provided).")

    print(f"\nModel export process finished! Target directory: {model_dir}")

if __name__ == "__main__":
    # Setup Argument Parser for CLI flags
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX and/or TensorRT formats.")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    parser.add_argument("--torch", action="store_true", help="export to onnx format with torch export")
    parser.add_argument("--engine", action="store_true", help="Export to TensorRT format")
    parser.add_argument("--all", action="store_true", help="Export to BOTH formats")
    parser.add_argument("--half", action="store_true", help="Export TensorRT in FP16 precision (default is FP32)")
    parser.add_argument("--device", default="cpu", help="Choose GPU from terminal, for example --device 0 or --device 1. Use cpu to force CPU.",)

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
    export_models(
        export_onnx=do_onnx,
        export_torch=args.torch,
        export_engine=do_engine,
        use_half=args.half,
        selected_device=args.device,
    )