# Development History - IJToolsUsingOnnxruntime

## [0.1.0] - 2026-05-05

### Added
- Initial implementation of ONNX Runtime plugins for ImageJ.
- **ORT_1st_ReadAndRelease**: Model management and slot-based loading.
- **ORT_2nd_Detection**: YOLO object detection (supports Standard and E2E models).
- **ORT_2nd_Pose**: YOLO pose estimation (supports Standard and E2E models).
- **ORT_2nd_Classify**: YOLO image classification.
- **ORT_Inspect**: Utility to inspect ONNX model metadata within ImageJ.
- **OrtUtil**: Core utility class for image preprocessing, NMS, and UI helpers.
- **MyOrtSession**: Wrapper for OrtSession with YOLO-specific logic.

### Changed
- Refactored all plugin dialogs to use the `DialogListener` pattern for improved macro compatibility and interactive UI behavior.
- Standardized image preprocessing with statistical debug logging.

### Tools & Testing
- Added `test/` directory with comprehensive macros for automated inference testing.
- Added `yolo/` directory with Python scripts for model inspection and label extraction.
- Added `README_CLI_Testing.md` for headless environment configuration.

---
*Note: This project follows a pure Java/ImageJ architecture for ONNX inference, independent of OpenCV.*
