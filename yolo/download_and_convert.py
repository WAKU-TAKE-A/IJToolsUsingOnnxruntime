import os
import sys

# Exit early if dependencies are missing with a helpful error message
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' package is not installed.")
    print("Please install the required dependencies first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def main():
    # Default model to download and convert
    default_model = "yolov8n-seg"
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = default_model
        print(f"No model name provided. Defaulting to '{default_model}'")

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

    print("Exporting model to ONNX format...")
    # Export the model. Since current working directory is models_dir,
    # the exported ONNX file will be saved in models_dir.
    onnx_path = model.export(format="onnx")
    
    print(f"Model successfully exported to: {onnx_path}")

    # Resolve class names
    base_name = os.path.splitext(pt_filename)[0]
    txt_filename = f"{base_name}.txt"

    print(f"Extracting class names to: {txt_filename}")
    if hasattr(model, 'names') and model.names:
        with open(txt_filename, "w", encoding="utf-8") as f:
            for idx in sorted(model.names.keys()):
                class_name = model.names[idx]
                f.write(f"{class_name}\n")
        print("Class names exported successfully.")
    else:
        print("Warning: Could not find class names list in the model.")

if __name__ == "__main__":
    main()
