from ultralytics import YOLO
import sys
import os

def trim_onnx(model_path, keep_classes):
    try:
        import onnx
        from onnx import helper
        from onnx import TensorProto
        import ast
    except ImportError:
        print("Error: 'onnx' module is required for trimming classes. Please 'pip install onnx'")
        return

    print(f"Trimming ONNX model to keep only classes: {keep_classes}")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Assuming the output is named 'output0' and shape is (1, 4+num_classes, num_anchors)
    original_output = graph.output[0]
    
    # Define indices to keep: 0, 1, 2, 3 (bbox) + (cls + 4) for cls in keep_classes
    indices = [0, 1, 2, 3] + [c + 4 for c in keep_classes]
    
    # Create Constant node for indices
    indices_tensor = helper.make_tensor('gather_indices', TensorProto.INT64, [len(indices)], indices)
    indices_node = helper.make_node('Constant', [], ['gather_indices_out'], value=indices_tensor)
    
    # Create Gather node
    gather_node = helper.make_node(
        'Gather',
        inputs=[original_output.name, 'gather_indices_out'],
        outputs=['new_output0'],
        axis=1
    )
    
    # Add nodes to graph
    graph.node.extend([indices_node, gather_node])
    
    # Update output info
    new_output = helper.make_tensor_value_info(
        'new_output0', 
        original_output.type.tensor_type.elem_type, 
        ['batch', len(indices), 'num_anchors'] # Dynamic shapes
    )
    graph.output.remove(original_output)
    graph.output.append(new_output)
    
    # Metadata update
    meta = model.metadata_props
    for prop in meta:
        if prop.key == 'names':
            # ultralytics saves dict as string like "{0: 'person', 1: 'bicycle'}"
            try:
                names_dict = ast.literal_eval(prop.value)
                new_names = {i: names_dict[c] for i, c in enumerate(keep_classes) if c in names_dict}
                prop.value = str(new_names)
            except Exception as e:
                print(f"Warning: Could not update names metadata: {e}")
    
    onnx.checker.check_model(model)
    # Save the trimmed model (overwrite)
    onnx.save(model, model_path)
    print("Trimming completed.")

def convert(model_path, classes_file=None):
    # This will download the .pt file to the current directory if it's not found
    model = YOLO(model_path)
    
    print("Exporting to ONNX format...")
    # Export the model. 
    # opset=12 is generally well-supported by OpenCV's DNN module.
    path = model.export(format='onnx', opset=12)
    print(f"Model exported to: {path}")

    # Process class trimming if requested
    if classes_file and os.path.exists(classes_file):
        keep_classes = []
        with open(classes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    keep_classes.append(int(line))
        
        if keep_classes:
            trim_onnx(path, keep_classes)
        else:
            print(f"Warning: No valid class IDs found in {classes_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <model_path> [classes_file]")
        sys.exit(1)

    model_path = sys.argv[1]
    classes_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert(model_path, classes_file)