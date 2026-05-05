// test_inference.ijm
print("\\Clear");

baseDir = "C:/Users/admin/Documents/NetBeansProjects/IJToolsUsingOnnxruntime/test/";
imagePath = baseDir + "test01.jpg";
modelName = "yolov8s.onnx";
formatLabel = "YOLO_Object_Pixel";

print("Starting Inference Test...");
print("Image: " + imagePath);
print("Model: " + modelName);

// 1. Open Image
open(imagePath);
if (nImages == 0) {
    print("Error: Could not open image.");
    eval("script", "System.exit(1);");
}

// 2. Load Model
print("Loading model...");
run("1st Read and Release", "action=read_model model_path=[" + baseDir + modelName + "] model_format=[" + formatLabel + "] slot=0 enable_log=true");

// 3. Run Detection
print("Running detection...");
run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table=true show_roi_manager=true enable_log=true");

// 4. Check Results
print("--------------------------------------------------");
print("Detection Results:");
n = nResults;
print("Count: " + n);

if (n > 0) {
    for (i = 0; i < n; i++) {
        label = getResultLabel(i);
        conf  = getResult("Confidence", i);
        x     = getResult("X", i);
        y     = getResult("Y", i);
        print("  " + i + ": " + label + " (conf=" + conf + ") at [" + x + ", " + y + "]");
    }
} else {
    print("No objects detected.");
}
print("--------------------------------------------------");

// Cleanup
run("1st Read and Release", "action=release_all enable_log=true");
print("Test finished.");

// Print the log to console
print("IMAGEJ_LOG_START");
print(getInfo("log"));
print("IMAGEJ_LOG_END");

run("Quit");
