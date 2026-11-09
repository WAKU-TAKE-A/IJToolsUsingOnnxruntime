"""Write the bundled manifest that accompanies an exported ONNX model.

The manifest format is shared with RF-DETR/export_rfdetr.py: provenance
(checkpoint and conversion tool), the conversion tool and its version,
the opset, input/output shapes, the ONNX custom metadata and the file
SHA-256. It is written as ``<model>.onnx.manifest.json`` next to the
ONNX file. The Java side (ModelManifest / ModelContract) reads
``num_classes`` and ``task`` from it and prefers them over shape
heuristics, so the top-level ``task`` key is written before the
``metadata`` block on purpose.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(onnx_path: Path, *, variant: str, task: str, tool: str,
                   tool_version: str, checkpoint: str | None = None,
                   opset: int | None = None, end2end: bool | None = None,
                   num_classes: int | None = None) -> Path:
    """Exports the manifest for an ONNX file and returns its path."""
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    info = {
        "variant": variant,
        "task": task,
        "tool": tool,
        "tool_version": tool_version,
        "checkpoint": checkpoint,
        "onnx": onnx_path.name,
        "sha256": sha256(onnx_path),
        "inputs": {i.name: list(i.shape) for i in session.get_inputs()},
        "outputs": {o.name: list(o.shape) for o in session.get_outputs()},
        "metadata": dict(session.get_modelmeta().custom_metadata_map),
    }
    if opset is not None:
        info["opset"] = opset
    if end2end is not None:
        info["end2end"] = end2end
    if num_classes is not None:
        info["num_classes"] = num_classes
    manifest_path = onnx_path.with_name(onnx_path.name + ".manifest.json")
    manifest_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
