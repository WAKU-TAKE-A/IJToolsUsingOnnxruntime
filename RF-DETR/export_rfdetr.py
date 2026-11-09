"""Download Apache-2.0 RF-DETR weights and prepare raw ONNX models for the Java plugins.

Run with RF-DETR/.venv/Scripts/python.exe. All generated assets stay in RF-DETR/models.
No Roboflow account, API key, training, or rfdetr-plus dependency is needed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODELS = {
    "detection-nano": ("RFDETRNano", "detect"),
    "detection-small": ("RFDETRSmall", "detect"),
    "detection-medium": ("RFDETRMedium", "detect"),
    "detection-large": ("RFDETRLarge", "detect"),
    "segmentation-nano": ("RFDETRSegNano", "segment"),
    "segmentation-small": ("RFDETRSegSmall", "segment"),
    "segmentation-medium": ("RFDETRSegMedium", "segment"),
    "segmentation-large": ("RFDETRSegLarge", "segment"),
    "segmentation-xlarge": ("RFDETRSegXLarge", "segment"),
    "segmentation-2xlarge": ("RFDETRSeg2XLarge", "segment"),
    "keypoint-preview": ("RFDETRKeypointPreview", "pose"),
}
DEFAULTS = ["detection-nano", "segmentation-nano", "keypoint-preview"]
SAMPLES = {
    "dog.jpg": "https://media.roboflow.com/dog.jpg",
    "person.jpg": "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000785.jpg",
}


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "IJTools-RFDETR/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    partial.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_assets() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    fetch("https://raw.githubusercontent.com/roboflow/rf-detr/1.11.2/LICENSE", MODEL_DIR / "LICENSE-RF-DETR.txt")
    records = []
    for filename, url in SAMPLES.items():
        target = MODEL_DIR / "samples" / filename
        fetch(url, target)
        records.append({"file": filename, "url": url, "sha256": sha256(target)})
    (MODEL_DIR / "samples" / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def export_one(name: str, checkpoint: Path | None = None) -> None:
    import onnx
    import onnxruntime as ort
    import rfdetr
    from rfdetr.assets.coco_classes import COCO_CLASSES, COCO_CLASS_NAMES

    cls_name, task = MODELS[name]
    destination = MODEL_DIR / name
    destination.mkdir(exist_ok=True)
    kwargs = {"device": "cpu"}
    if checkpoint:
        kwargs["pretrain_weights"] = str(checkpoint.resolve())
    model = getattr(rfdetr, cls_name)(**kwargs)
    args = model.model.args
    config = model.model_config
    names = list(model.class_names)
    counts = list(getattr(args, "num_keypoints_per_class", []) or [])
    postprocess = model.model.postprocess
    # Capture schema before export (export may copy or temporarily modify the model).
    top_k = int(postprocess.num_select)
    alpha = float(getattr(postprocess, "trace_alpha", 0))
    path = Path(model.export(output_dir=str(destination), format="onnx", output_name="model",
                             opset_version=17, batch_size=1, dynamic_batch=False, verbose=False))
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    label_shape = next(o for o in proto.graph.output if o.name == "labels").type.tensor_type.shape.dim
    slots = label_shape[2].dim_value
    if not slots:
        raise RuntimeError("Export must have a static class dimension")
    slot_names = [f"class_{i}" for i in range(slots)]
    sparse_coco = int(getattr(args, "num_classes", len(names))) > len(names) and names == list(COCO_CLASS_NAMES)
    if sparse_coco:
        for category, label in zip(COCO_CLASSES, names):
            slot_names[category] = label
        # Sparse COCO slot 90 is toothbrush, not background. Do not drop it.
        background = -1
    elif counts and counts[0] == 0:
        active_slots = [i for i, count in enumerate(counts) if count > 0]
        for slot, label in zip(active_slots, names):
            slot_names[slot] = label
        background = 0
    else:
        for i, label in enumerate(names):
            slot_names[i] = label
        background = slots - 1 if slots > len(names) else -1
    if background >= 0:
        slot_names[background] = "__background__"
    metadata = {p.key: p.value for p in proto.metadata_props}
    license_name = "Apache-2.0" if checkpoint is None else "User-supplied checkpoint; see its license"
    metadata.update({
        "names": json.dumps(dict(enumerate(slot_names)), ensure_ascii=False),
        "task": task,
        "ijtools.rfdetr.contract": "1",
        "ijtools.rfdetr.task": task,
        "ijtools.rfdetr.num_select": str(top_k),
        "ijtools.rfdetr.background": str(background),
        "ijtools.rfdetr.keypoint_counts": ",".join(map(str, counts)),
        "ijtools.rfdetr.trace_alpha": str(alpha),
        "ijtools.rfdetr.version": importlib.metadata.version("rfdetr"),
        "ijtools.rfdetr.variant": name,
        "license": license_name,
    })
    onnx.helper.set_model_props(proto, metadata)
    onnx.save(proto, str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    info = {
        "variant": name, "class": cls_name, "task": task, "license": license_name,
        "tool": "rfdetr", "tool_version": importlib.metadata.version("rfdetr"),
        "rfdetr_version": importlib.metadata.version("rfdetr"),
        "checkpoint": str(config.pretrain_weights), "onnx": str(path.relative_to(MODEL_DIR)),
        "sha256": sha256(path),
        "inputs": {i.name: i.shape for i in session.get_inputs()},
        "outputs": {o.name: o.shape for o in session.get_outputs()},
        "metadata": metadata,
    }
    (destination / "manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": name, "onnx": str(path), "outputs": info["outputs"]}), flush=True)
    del session, proto, model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULTS)
    parser.add_argument("--all", action="store_true", help="Download/export all 11 Apache variants (large downloads)")
    parser.add_argument("--checkpoint", type=Path, help="Custom checkpoint; select exactly one matching variant")
    parser.add_argument("--samples-only", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Use the virtual environment: RF-DETR/.venv/Scripts/python.exe")
    selected = list(MODELS) if args.all else args.models
    if args.checkpoint and (len(selected) != 1 or not args.checkpoint.is_file()):
        parser.error("--checkpoint requires an existing file and exactly one --models variant")
    # Set before importing rfdetr: checkpoint and backbone caches also belong in MODEL_DIR.
    os.environ["RF_HOME"] = str(MODEL_DIR / "weights")
    os.environ["TORCH_HOME"] = str(MODEL_DIR / "cache" / "torch")
    os.environ["HF_HOME"] = str(MODEL_DIR / "cache" / "huggingface")
    prepare_assets()
    if not args.samples_only:
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        for name in selected:
            export_one(name, args.checkpoint)


if __name__ == "__main__":
    main()
