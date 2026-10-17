"""Build with Java 17, run ImageJ/ONNX smoke tests, and compare with RF-DETR's official Python processing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from export_rfdetr import ROOT, MODEL_DIR, DEFAULTS


def verify(name: str) -> dict:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import torch
    from rfdetr.export._runtime.preprocess import preprocess_to_nchw
    from rfdetr.models.postprocess import PostProcess

    destination = MODEL_DIR / "verification" / name
    image_name = "person.jpg" if name == "keypoint-preview" else "dog.jpg"
    image = Image.open(MODEL_DIR / "samples" / image_name).convert("RGB")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(MODEL_DIR / name / "model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    _, _, height, width = session.get_inputs()[0].shape
    reference_input = preprocess_to_nchw(image, height, width)
    java_input = np.fromfile(destination / "input.f32", dtype="<f4").reshape(reference_input.shape)
    input_error = float(np.max(np.abs(java_input - reference_input)))
    np.testing.assert_allclose(java_input, reference_input, atol=3e-5, rtol=1e-5)
    raw = dict(zip([o.name for o in session.get_outputs()], session.run(None, {session.get_inputs()[0].name: reference_input})))
    predictions = {"pred_boxes": torch.from_numpy(raw["dets"]), "pred_logits": torch.from_numpy(raw["labels"])}
    if "masks" in raw:
        predictions["pred_masks"] = torch.from_numpy(raw["masks"])
    if "keypoints" in raw:
        predictions["pred_keypoints"] = torch.from_numpy(raw["keypoints"])
    counts = [int(n) for n in meta["ijtools.rfdetr.keypoint_counts"].split(",") if n]
    processor = PostProcess(num_select=int(meta["ijtools.rfdetr.num_select"]),
                            num_keypoints_per_class=counts, trace_alpha=float(meta["ijtools.rfdetr.trace_alpha"]))
    result = processor(predictions, torch.tensor([[image.height, image.width]]), score_threshold=.3)[0]
    keep = result["scores"].numpy() > .3
    background = int(meta["ijtools.rfdetr.background"])
    if background >= 0:
        keep &= result["labels"].numpy() != background
    expected_labels = result["labels"].numpy()[keep]
    expected_scores = result["scores"].numpy()[keep]
    scale = np.array([image.width,image.height,image.width,image.height])
    expected_boxes = np.clip(result["boxes"].numpy()[keep], 0, scale)
    expected_boxes[:,2:] -= expected_boxes[:,:2]
    actual = np.atleast_1d(np.genfromtxt(destination / "detections.csv", delimiter=",", names=True))
    np.testing.assert_array_equal(actual["ClassID"], expected_labels)
    np.testing.assert_allclose(actual["Score"], expected_scores, atol=2e-4, rtol=1e-4)
    actual_boxes = np.column_stack([actual[c] for c in ["X","Y","W","H"]])
    np.testing.assert_allclose(actual_boxes, expected_boxes, atol=.05, rtol=1e-4)
    report = {"variant":name, "detections":len(actual), "input_max_abs_error":input_error,
              "score_max_abs_error":float(np.max(np.abs(actual["Score"]-expected_scores))),
              "box_max_abs_error_pixels":float(np.max(np.abs(actual_boxes-expected_boxes)))}
    if "masks" in result:
        mismatch = []
        for i, expected in enumerate(result["masks"].numpy()[keep]):
            observed = np.asarray(Image.open(destination / f"mask-{i}.png")) > 0
            if expected.ndim == 3:
                expected = expected[0]
            error = float(np.mean(observed != expected))
            assert error < .001, f"Mask mismatch: {error:.6%}"
            mismatch.append(error)
        report["mask_max_pixel_mismatch_fraction"] = max(mismatch)
    if "keypoints" in result:
        expected = result["keypoints"].numpy()[keep]
        observed = np.atleast_1d(np.genfromtxt(destination / "keypoints.csv", delimiter=",", names=True))
        observed = np.column_stack([observed[c] for c in ["X","Y","Score"]]).reshape(expected.shape)
        np.testing.assert_allclose(observed, expected, atol=.05, rtol=1e-4)
        report["keypoint_max_abs_error"] = float(np.max(np.abs(observed-expected)))
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java-home", type=Path, default=Path(r"C:\Program Files\Zulu\zulu-17"))
    parser.add_argument("--maven", type=Path, default=Path(r"C:\Program Files\Apache NetBeans\java\maven\bin\mvn.cmd"))
    parser.add_argument("--imagej-jar", type=Path, default=Path.home()/".m2/repository/net/imagej/ij/1.54k/ij-1.54k.jar")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-java", action="store_true", help="Recheck existing Java output against Python")
    parser.add_argument("--macro", action="store_true", help="Also test ImageJ batch macros in an isolated plugins directory (requires AWT)")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run with RF-DETR/.venv/Scripts/python.exe")
    env = os.environ.copy()
    env["JAVA_HOME"] = str(args.java_home)
    if not args.skip_build:
        subprocess.run([str(args.maven),"-q","package"], cwd=ROOT,env=env,check=True)
    if not args.skip_java:
        classpath = os.pathsep.join(map(str,[ROOT/"target/test-classes",ROOT/"target/IJTools_UsingOnnxruntime.jar",args.imagej_jar]))
        result = subprocess.run([str(args.java_home/"bin/java.exe"),"-Djava.awt.headless=true","-Xmx3g","-cp",classpath,
                                "io.github.wakutakea.ijtoolsusingonnxruntime.RfDetrSmoke",str(MODEL_DIR)],
                                cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,flush=True)
        (MODEL_DIR/"verification/java.log").write_text(result.stdout,encoding="utf-8")
        result.check_returncode()
    if args.macro:
        isolated = MODEL_DIR / "verification" / "imagej"
        plugins = isolated / "plugins"
        plugins.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "target/IJTools_UsingOnnxruntime.jar", plugins / "IJTools_UsingOnnxruntime.jar")
        result = subprocess.run([str(args.java_home / "bin/java.exe"), f"-Dplugins.dir={isolated}", "-Xmx3g",
                                 "-jar", str(args.imagej_jar), "-batch", str(ROOT / "test/test_rfdetr.ijm"), str(ROOT)],
                                cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (MODEL_DIR / "verification/macro.log").write_text(result.stdout, encoding="utf-8")
        result.check_returncode()
        for name in DEFAULTS:
            assert f"PASS: {name} ROIs=" in result.stdout, result.stdout
        print("PASS: ImageJ batch macros for all three tasks (isolated plugins directory).", flush=True)
    import torch
    torch.set_num_threads(4)
    reports = [verify(name) for name in DEFAULTS]
    (MODEL_DIR/"verification/report.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
    print("PASS: Java inference, ImageJ presentation, and official Python preprocessing/postprocessing agree.")


if __name__ == "__main__":
    main()
