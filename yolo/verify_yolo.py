"""Build with Java 17, run the YoloSmoke harness, and compare its pre-NMS
detections with a Python reference (same letterbox + output parse).

Mirrors RF-DETR/verify_rfdetr.py: the build and harness steps are the
same shape, and the comparison happens before NMS (the harness itself
checks the post-NMS ResultsTable). The reference covers the standard
letterbox detection formats (YOLO_Object_Pixel / YOLO_Object_Normalized);
E2E and YOLOX formats need their own references and are rejected here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = Path(__file__).resolve().parent
MODEL_DIR = YOLO_DIR / "models"
SAMPLE_DIR = MODEL_DIR / "samples"
SAMPLE_SOURCES = {
    "bus.jpg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
}
SCORE_THRESHOLD = 0.3


def ensure_sample(image_name: str) -> Path:
    path = SAMPLE_DIR / image_name
    if path.exists():
        return path
    url = SAMPLE_SOURCES.get(image_name)
    if url is None:
        raise SystemExit(f"Sample image not found and no download source is known: {path}")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample image {image_name} from {url}")
    urllib.request.urlretrieve(url, path)
    sources_path = SAMPLE_DIR / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    sources[image_name] = {"url": url, "source": "Ultralytics example assets"}
    sources_path.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return path


def letterbox_reference(image, target_w: int, target_h: int):
    """The Java Preprocessor letterbox (YOLO_LETTERBOX): bilinear resize,
    gray-114 padding, RGB, divided by 255. Returns (tensor, scale, pad_left, pad_top)."""
    import numpy as np
    from PIL import Image

    scale = min(target_w / image.width, target_h / image.height)
    scaled_w = int(math.floor(image.width * scale + 0.5))
    scaled_h = int(math.floor(image.height * scale + 0.5))
    pad_left = (target_w - scaled_w) // 2
    pad_top = (target_h - scaled_h) // 2
    resized = image.resize((scaled_w, scaled_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(resized, (pad_left, pad_top))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), scale, pad_left, pad_top


def num_classes_for(model_path: Path, output: "np.ndarray", session) -> int:
    """Class count with the same precedence as the Java side: bundled
    manifest first, then the embedded names metadata, then the shape."""
    manifest_path = model_path.with_name(model_path.name + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("num_classes"):
            return int(manifest["num_classes"])
    names = session.get_modelmeta().custom_metadata_map.get("names")
    if names:
        try:
            parsed = eval(names, {"__builtins__": {}}, {})  # ultralytics stores a dict literal
            if isinstance(parsed, dict):
                return len(parsed)
        except Exception:
            pass
    d1, d2 = output.shape[1], output.shape[2]
    return min(d1, d2) - 4


def parse_reference(output, num_classes: int, normalized: bool,
                    image_w: int, image_h: int, input_w: int, input_h: int,
                    scale: float, pad_left: int, pad_top: int):
    """Pre-NMS candidates in the Java YoloDecoder.parseYOLO order:
    per anchor, the best class above the threshold, letterbox undone,
    clamped to the image."""
    import numpy as np

    data = output[0]
    d1, d2 = data.shape
    channel_first = d1 == 4 + num_classes and d1 < d2
    rows = []
    anchors = d2 if channel_first else d1
    for i in range(anchors):
        cx, cy, w, h = (data[0][i], data[1][i], data[2][i], data[3][i]) if channel_first \
            else (data[i][0], data[i][1], data[i][2], data[i][3])
        scores = data[4:4 + num_classes, i] if channel_first else data[i, 4:4 + num_classes]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < SCORE_THRESHOLD:
            continue
        if normalized:
            bw, bh = w * input_w, h * input_h
            bx, by = cx * input_w - bw / 2, cy * input_h - bh / 2
        else:
            bw, bh = w, h
            bx, by = cx - bw / 2, cy - bh / 2
        rx = min(max((bx - pad_left) / scale, 0.0), image_w)
        ry = min(max((by - pad_top) / scale, 0.0), image_h)
        rw = min(max(bw / scale, 0.0), image_w - rx)
        rh = min(max(bh / scale, 0.0), image_h - ry)
        if rw <= 0 or rh <= 0:
            continue
        rows.append((class_id, score, rx, ry, rw, rh))
    return rows


def verify(model_name: str, format_label: str, image_name: str) -> dict:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if format_label not in ("YOLO_Object_Pixel", "YOLO_Object_Normalized"):
        raise SystemExit(f"verify_yolo.py covers the standard detection formats only, not {format_label}")
    model_path = MODEL_DIR / model_name
    stem = model_name[:-len(".onnx")] if model_name.endswith(".onnx") else model_name
    destination = MODEL_DIR / "verification" / "java" / stem
    image = Image.open(ensure_sample(image_name)).convert("RGB")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    _, _, input_h, input_w = session.get_inputs()[0].shape
    reference_input, scale, pad_left, pad_top = letterbox_reference(image, input_w, input_h)
    java_input = np.fromfile(destination / "input.f32", dtype="<f4").reshape(reference_input.shape)
    input_error = float(np.max(np.abs(java_input - reference_input)))
    # ImageJ and PIL bilinear resampling differ slightly; the tensor must
    # still agree closely everywhere.
    np.testing.assert_allclose(java_input, reference_input, atol=2e-2, rtol=0)
    raw = session.run(None, {session.get_inputs()[0].name: reference_input})[0]
    classes = num_classes_for(model_path, raw, session)
    expected = parse_reference(raw, classes, format_label == "YOLO_Object_Normalized",
                               image.width, image.height, input_w, input_h,
                               scale, pad_left, pad_top)
    actual = np.atleast_1d(np.genfromtxt(destination / "detections.csv", delimiter=",", names=True))
    assert len(actual) == len(expected), f"pre-NMS count differs: Java {len(actual)} vs Python {len(expected)}"
    expected_labels = np.array([row[0] for row in expected])
    expected_scores = np.array([row[1] for row in expected])
    expected_boxes = np.array([row[2:] for row in expected])
    np.testing.assert_array_equal(actual["ClassID"], expected_labels)
    np.testing.assert_allclose(actual["Score"], expected_scores, atol=1e-2, rtol=1e-3)
    actual_boxes = np.column_stack([actual[c] for c in ["X", "Y", "W", "H"]])
    np.testing.assert_allclose(actual_boxes, expected_boxes, atol=1.5, rtol=1e-3)
    report = {"model": model_name, "format": format_label, "detections": len(actual),
              "input_max_abs_error": input_error,
              "score_max_abs_error": float(np.max(np.abs(actual["Score"] - expected_scores))) if len(actual) else 0.0,
              "box_max_abs_error_pixels": float(np.max(np.abs(actual_boxes - expected_boxes))) if len(actual) else 0.0}
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path, default=Path(r"C:\Program Files\Zulu\zulu-17"))
    parser.add_argument("--maven", type=Path, default=Path(r"C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd"))
    parser.add_argument("--imagej-jar", type=Path, default=Path.home() / ".m2/repository/net/imagej/ij/1.54k/ij-1.54k.jar")
    parser.add_argument("--model", default="yolov8n.onnx", help="ONNX file name under yolo/models")
    parser.add_argument("--format", default="YOLO_Object_Pixel", help="ModelContract label for the model")
    parser.add_argument("--image", default="bus.jpg", help="Sample image name under yolo/models/samples")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-java", action="store_true", help="Recheck existing Java output against Python")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run with yolo/.venv/Scripts/python.exe")
    env = os.environ.copy()
    env["JAVA_HOME"] = str(args.java_home)
    if not args.skip_build:
        subprocess.run([str(args.maven), "-q", "package"], cwd=ROOT, env=env, check=True)
    if not args.skip_java:
        classpath = os.pathsep.join(map(str, [ROOT / "target/test-classes",
                                              ROOT / "target/IJTools_UsingOnnxruntime.jar", args.imagej_jar]))
        result = subprocess.run([str(args.java_home / "bin/java.exe"), "-Djava.awt.headless=true", "-Xmx3g",
                                 "-cp", classpath,
                                 "io.github.wakutakea.ijtoolsusingonnxruntime.YoloSmoke",
                                 str(MODEL_DIR), args.model, args.format, args.image],
                                cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        (MODEL_DIR / "verification").mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / "verification/java.log").write_text(result.stdout, encoding="utf-8")
        result.check_returncode()
    reports = [verify(args.model, args.format, args.image)]
    (MODEL_DIR / "verification/report.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("PASS: Java inference and the Python reference agree (pre-NMS).")


if __name__ == "__main__":
    main()
