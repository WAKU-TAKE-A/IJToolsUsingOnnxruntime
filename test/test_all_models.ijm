// test_all_inference.ijm
print("\\Clear");

baseDir = "C:/Users/admin/Documents/NetBeansProjects/IJToolsUsingOnnxruntime/test/";
imagePath = baseDir + "test01.jpg";

models = newArray(
    "yolov8s.onnx,YOLO_Object_Pixel,Detection",
    "yolo26s.onnx,YOLO_Object_E2E,Detection",
    "yolov8s-pose.onnx,YOLO_Pose,Pose",
    "yolo11s-pose.onnx,YOLO_Pose_E2E,Pose"
);

print("Starting Comprehensive Inference Test...");
print("Image: " + imagePath);

for (i = 0; i < models.length; i++) {
    item = split(models[i], ",");
    modelName = item[0];
    formatLabel = item[1];
    type = item[2];

    print("--------------------------------------------------");
    print("Testing Model: " + modelName + " (" + type + ")");

    // 1. Open Image
    open(imagePath);
    
    // 2. Load Model
    print("  Loading...");
    run("1st Read and Release", "action=read_model model_path=[" + baseDir + modelName + "] model_format=[" + formatLabel + "] slot=0 enable_log=false");

    // 3. Run Inference
    if (type == "Detection") {
        print("  Running Detection...");
        run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table=true show_roi_manager=true enable_log=false");
    } else {
        print("  Running Pose...");
        run("2nd Pose", "slot=0 score_threshold=0.3 nms_threshold=0.45 kpt_threshold=0.5 show_roi_manager=true enable_log=false");
    }

    // 4. Check Results
    if (type == "Detection") {
        n = nResults;
        print("  Result: " + n + " objects detected.");
        if (n > 0) {
            for (j = 0; j < Math.min(n, 3); j++) {
                print("    - " + getResultLabel(j) + " (conf=" + getResult("Confidence", j) + ")");
            }
        }
    } else {
        // For Pose, we look at ROI Manager count (objects + keypoints)
        n = roiManager("count");
        print("  Result: " + n + " ROIs in manager (objects + keypoints).");
    }

    // Cleanup
    run("1st Read and Release", "action=release_all enable_log=false");
    roiManager("reset");
    run("Clear Results");
    close();
}

print("--------------------------------------------------");
print("All Tests finished.");
run("Quit");
