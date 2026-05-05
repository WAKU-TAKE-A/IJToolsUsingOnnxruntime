import onnx
import sys
import os

def inspect_onnx(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return

    print(f"Inspecting ONNX model: {model_path}")
    model = onnx.load(model_path)
    
    print("\n[Inputs]")
    for input in model.graph.input:
        name = input.name
        shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in input.type.tensor_type.shape.dim]
        # Element types: 1=FLOAT, 2=UINT8, 3=INT8, 4=UINT16, 5=INT16, 6=INT32, 7=INT64, etc.
        elem_type = input.type.tensor_type.elem_type
        print(f"  Name: {name}, Shape: {shape}, Type: {elem_type}")
        
    print("\n[Outputs]")
    for output in model.graph.output:
        name = output.name
        shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in output.type.tensor_type.shape.dim]
        elem_type = output.type.tensor_type.elem_type
        print(f"  Name: {name}, Shape: {shape}, Type: {elem_type}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <model_path.onnx>")
    else:
        inspect_onnx(sys.argv[1])
