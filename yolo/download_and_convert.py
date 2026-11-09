"""Download an Ultralytics YOLO checkpoint, export it to ONNX, and
write the bundled manifest (see manifest_util.py) next to the ONNX.

The export settings that decide the ONNX contract (opset, E2E) are
explicit arguments here, not Ultralytics defaults, so the exported
model's format can be reproduced and recorded in the manifest.
"""
import argparse
import os
import sys
from pathlib import Path

# Exit early if dependencies are missing with a helpful error message
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' package is not installed.")
    print("Please install the required dependencies first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_util import write_manifest

# Opset 17 matches the RF-DETR export in this repository and is
# comfortably inside the Java ONNX Runtime's supported range.
DEFAULT_OPSET = 17


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="yolov8n-seg",
                        help="Model name or checkpoint (default: yolov8n-seg)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                        help=f"ONNX opset for the export (default: {DEFAULT_OPSET})")
    parser.add_argument("--end2end", action="store_true",
                        help="Export an end-to-end (NMS-fused) model instead of the normal one")
    args = parser.parse_args()

    model_name = args.model
    if model_name == "yolov8n-seg":
        print(f"No model name provided. Defaulting to '{model_name}'")

    # Ensure the model name has a .pt extension for loading/downloading
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    # Define directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Change the working directory to 'models' so all downloads and exports
    # are forced to save inside the 'models' directory.
    os.chdir(models_dir)

    pt_filename = os.path.basename(model_name)
    print(f"Working directory changed to: {os.getcwd()}")
    print(f"Target PyTorch model: {pt_filename}")
    print("Initializing YOLO model (this will automatically download the .pt file if not present)...")

    # Load and download model
    model = YOLO(pt_filename)

    print(f"Exporting model to ONNX format (opset={args.opset}, end2end={args.end2end})...")
    # Export the model. Since current working directory is models_dir,
    # the exported ONNX file will be saved in models_dir.
    onnx_path = Path(model.export(format="onnx", opset=args.opset, end2end=args.end2end))

    print(f"Model successfully exported to: {onnx_path}")

    # Resolve class names
    base_name = os.path.splitext(pt_filename)[0]
    txt_filename = f"{base_name}.txt"

    num_classes = None
    print(f"Extracting class names to: {txt_filename}")
    if hasattr(model, 'names') and model.names:
        with open(txt_filename, "w", encoding="utf-8") as f:
            for idx in sorted(model.names.keys()):
                class_name = model.names[idx]
                f.write(f"{class_name}\n")
        num_classes = len(model.names)
        print("Class names exported successfully.")
    else:
        print("Warning: Could not find class names list in the model.")

    # Record the export contract next to the ONNX file. The Java side
    # prefers the manifest's task/num_classes over shape heuristics.
    manifest_path = write_manifest(
        onnx_path,
        variant=base_name,
        task=getattr(model, "task", None) or "",
        tool="ultralytics",
        tool_version=ultralytics.__version__,
        checkpoint=pt_filename,
        opset=args.opset,
        end2end=args.end2end,
        num_classes=num_classes,
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
