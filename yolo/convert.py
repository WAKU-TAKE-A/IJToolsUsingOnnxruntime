"""Legacy conversion path (opset 12) with optional class trimming.

Phase 3, plan A (minimal): trimming never overwrites the exported
ONNX. It writes ``<name>_trimmed.onnx`` next to it, the requested
class IDs are validated (duplicates and range), and every produced
ONNX finishes with a mandatory InferenceSession smoke run plus a
bundled manifest (see manifest_util.py). The duplicate trim
implementation that used to live in test/test_onnx_trim.py is
integrated here.
"""
import ast
import os
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_util import write_manifest

# Legacy export setting; validate the resulting ONNX with the Java/ImageJ plugin.
OPSET = 12


def read_keep_classes(classes_file, total_classes):
    """Reads and validates the class IDs to keep (one per line).

    A missing file, a non-numeric line, duplicates and out-of-range
    IDs are hard errors: silently accepting them used to produce
    models whose class layout nobody had actually asked for.
    """
    path = Path(classes_file)
    if not path.exists():
        raise SystemExit(f"Error: classes file not found: {classes_file}")
    keep = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if not line.lstrip("+-").isdigit():
            raise SystemExit(f"Error: {classes_file}:{lineno}: not a class ID: {line!r}")
        keep.append(int(line))
    if not keep:
        raise SystemExit(f"Error: no class IDs found in {classes_file}")
    duplicates = sorted({c for c in keep if keep.count(c) > 1})
    if duplicates:
        raise SystemExit(f"Error: duplicate class IDs in {classes_file}: {duplicates}")
    out_of_range = [c for c in keep if c < 0 or c >= total_classes]
    if out_of_range:
        raise SystemExit(
            f"Error: class IDs out of range [0, {total_classes}) in {classes_file}: {out_of_range}")
    return keep


def class_count_of(model_path):
    """Total class count of an exported detection ONNX, from the names
    metadata when present, else from the output shape (channels - 4)."""
    import onnx

    model = onnx.load(str(model_path))
    for prop in model.metadata_props:
        if prop.key == "names":
            try:
                names = ast.literal_eval(prop.value)
                if isinstance(names, dict) and names:
                    return len(names)
            except Exception:
                pass
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(dims) >= 2 and dims[1].dim_value > 4:
        return dims[1].dim_value - 4
    raise SystemExit(f"Error: cannot determine the class count of {model_path}")


def metadata_task(model_path):
    import onnx

    model = onnx.load(str(model_path))
    for prop in model.metadata_props:
        if prop.key == "task":
            return prop.value
    return ""


def trim_onnx(model_path, keep_classes):
    """Adds an axis-1 Gather keeping the bbox channels and the given
    classes. Writes ``<name>_trimmed.onnx``; the input file is never
    modified. Returns the trimmed path."""
    import onnx
    from onnx import TensorProto, helper

    model_path = Path(model_path)
    print(f"Trimming ONNX model to keep only classes: {keep_classes}")
    model = onnx.load(str(model_path))
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
        ['batch', len(indices), 'num_anchors']  # Dynamic shapes
    )
    graph.output.remove(original_output)
    graph.output.append(new_output)

    # Metadata update: renumber the kept class names from 0, in request order
    for prop in model.metadata_props:
        if prop.key == 'names':
            try:
                names_dict = ast.literal_eval(prop.value)
                prop.value = str({i: names_dict[c] for i, c in enumerate(keep_classes)})
            except Exception as e:
                print(f"Warning: Could not update names metadata: {e}")

    onnx.checker.check_model(model)
    trimmed_path = model_path.with_name(model_path.stem + "_trimmed.onnx")
    onnx.save(model, str(trimmed_path))
    print(f"Trimmed model written to: {trimmed_path} (original left unchanged)")
    return trimmed_path


def smoke_run(onnx_path, expected_classes=None):
    """Mandatory post-conversion check: the ONNX must load in
    onnxruntime and run once on a zero input. For a trimmed detection
    model the output channel count is asserted as well."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {}
    for info in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in info.shape]
        dtype = np.float32 if info.type == "tensor(float)" else np.float64
        inputs[info.name] = np.zeros(shape, dtype=dtype)
    outputs = session.run(None, inputs)
    if expected_classes is not None:
        channels = outputs[0].shape[1]
        if channels != 4 + expected_classes:
            raise SystemExit(
                f"Error: smoke run of {onnx_path} produced {channels} channels, "
                f"expected {4 + expected_classes}")
    print(f"Smoke run OK: {onnx_path} -> {[list(o.shape) for o in outputs]}")


def convert(model_path, classes_file=None):
    # This will download the .pt file to the current directory if it's not found
    model = YOLO(model_path)

    print("Exporting to ONNX format...")
    path = Path(model.export(format='onnx', opset=OPSET))
    print(f"Model exported to: {path}")

    produced = path
    num_classes = None
    if classes_file:
        total = class_count_of(path)
        keep_classes = read_keep_classes(classes_file, total)
        produced = trim_onnx(path, keep_classes)
        num_classes = len(keep_classes)
    else:
        try:
            num_classes = class_count_of(path)
        except SystemExit:
            num_classes = None

    smoke_run(produced, expected_classes=num_classes if classes_file else None)
    manifest_path = write_manifest(
        produced,
        variant=produced.stem,
        task=getattr(model, "task", None) or metadata_task(produced),
        tool="ultralytics",
        tool_version=ultralytics.__version__,
        checkpoint=str(model_path),
        opset=OPSET,
        num_classes=num_classes,
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <model_path> [classes_file]")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
