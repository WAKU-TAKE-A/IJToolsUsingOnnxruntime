import onnx
from onnx import helper
from onnx import TensorProto
import json

def trim_onnx(model_path, out_path, keep_classes):
    model = onnx.load(model_path)
    graph = model.graph
    
    # Assuming the output is named 'output0' and shape is (1, 84, 8400)
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
        ['batch', len(indices), 'num_anchors'] # dynamic or keep original
    )
    graph.output.remove(original_output)
    graph.output.append(new_output)
    
    import ast
    meta = model.metadata_props
    for prop in meta:
        if prop.key == 'names':
            names_dict = ast.literal_eval(prop.value)
            new_names = {i: names_dict[c] for i, c in enumerate(keep_classes)}
            prop.value = str(new_names)
    
    onnx.checker.check_model(model)
    onnx.save(model, out_path)

trim_onnx('yolov8n.onnx', 'yolov8n_trimmed.onnx', [0, 2, 50, 69])
print("Done")
