import onnx
import sys
import os
import ast

def extract_names(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return

    # Generate output path (replace .onnx with .txt)
    base_path = os.path.splitext(model_path)[0]
    output_path = base_path + ".txt"

    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    
    # Ultralytics stores names in metadata_props
    names = None
    for prop in model.metadata_props:
        if prop.key == 'names':
            names = prop.value
            break
    
    if names:
        # names is usually a string representation of a dict: "{0: 'person', 1: 'bicycle', ...}"
        try:
            names_dict = ast.literal_eval(names)
            with open(output_path, 'w', encoding='utf-8') as f:
                # Ensure we write in order of indices
                for i in sorted(names_dict.keys()):
                    f.write(f"{names_dict[i]}\n")
            print(f"Successfully extracted {len(names_dict)} names to {output_path}")
        except Exception as e:
            print(f"Error parsing names metadata: {e}")
            print(f"Raw metadata: {names}")
    else:
        print(f"No 'names' metadata found in the ONNX model: {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_names.py <model_path.onnx>")
    else:
        extract_names(sys.argv[1])
