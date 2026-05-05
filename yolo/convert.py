from ultralytics import YOLO
import sys

def convert(model_path):
    # This will download the .pt file to the current directory if it's not found
    model = YOLO(model_path)
    
    print("Exporting to ONNX format...")
    # Export the model. 
    # opset=12 is generally well-supported by OpenCV's DNN module.
    path = model.export(format='onnx', opset=12)
    
    print(f"Model exported to: {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    convert(model_path)