# IJToolsUsingOnnxruntime

ONNX Runtime based plugins for ImageJ, specifically optimized for YOLO-based computer vision tasks.

## Features

- **High-Performance Inference**: Powered by Microsoft ONNX Runtime for Java.
- **YOLO Support**: 
  - Object Detection (Standard & End-to-End)
  - Pose Estimation (Standard & End-to-End)
  - Image Classification
- **Interactive UI**: Standard ImageJ GenericDialogs with real-time parameter validation using `DialogListener`.
- **Macro Compatibility**: Full support for ImageJ macro recording and headless execution.
- **Model Inspection**: Built-in tools to inspect ONNX model metadata and structure.
- **Label Extraction**: Utility script to extract class names directly from model metadata.

## Installation

1. Build the project using Maven:
   ```powershell
   mvn clean package -DskipTests
   ```
2. Copy the resulting `target/IJTools_UsingOnnxruntime.jar` to your ImageJ `plugins` folder.
3. Restart ImageJ.

## Usage

### 1. Model Loading
Go to `Plugins > ORT > 1st Read and Release`. Select your `.onnx` model and specify the format (e.g., `YOLO_Object_Pixel` or `YOLO_Pose`).

### 2. Inference
Open an image and go to:
- `Plugins > ORT > 2nd Detection` for object detection.
- `Plugins > ORT > 2nd Pose` for pose estimation.
- `Plugins > ORT > 2nd Classify` for image classification.

### 3. Inspection
Use `Plugins > ORT > Inspect` to view the input/output nodes and metadata of an ONNX file.

## Developer Tools

Check the `test/` and `yolo/` folders for utility scripts:
- `test/test_all_models.ijm`: Comprehensive macro test suite.
- `yolo/inspect_onnx.py`: CLI model inspector.
- `yolo/extract_names.py`: CLI label extractor.

See `test/README_CLI_Testing.md` for detailed CLI testing instructions.

## License
MIT License
