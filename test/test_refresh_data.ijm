// test_refresh_data.ijm
print("\\Clear");

baseDir = "C:/Users/admin/Documents/NetBeansProjects/IJToolsUsingOnnxruntime/test/";
imagePath = baseDir + "test01.jpg";
modelName = "yolov8s.onnx";
formatLabel = "YOLO_Object_Pixel";

print("Testing 'enable_refresh_data' functionality...");

open(imagePath);

// 1. Load Model
run("1st Read and Release", "action=read_model model_path=[" + baseDir + modelName + "] model_format=[" + formatLabel + "] slot=0 enable_log=false");

// --- CASE 1: enable_refresh_data = true (Default) ---
print("Case 1: enable_refresh_data = true (Clear data before inference)");
run("Clear Results");
roiManager("reset");

// Run 1st time
print("  Running 1st time...");
run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table show_roi_manager enable_refresh_data");
n1 = nResults;
r1 = roiManager("count");
print("  First run: Results=" + n1 + ", ROIs=" + r1);

// Run 2nd time (with refresh)
run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table show_roi_manager enable_refresh_data");
n2 = nResults;
r2 = roiManager("count");
print("  Second run (refresh=true): Results=" + n2 + ", ROIs=" + r2);

if (n2 == n1 && r2 == r1) {
    print("  PASS: Data was refreshed (count stayed same).");
} else {
    print("  FAIL: Data was NOT refreshed correctly! (n2=" + n2 + ", r2=" + r2 + ")");
}

// --- CASE 2: enable_refresh_data = false (Append data) ---
print("Case 2: enable_refresh_data = false (Append data)");
run("Clear Results");
roiManager("reset");

// Run 1st time
run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table show_roi_manager");
n1 = nResults;
r1 = roiManager("count");
print("  First run: Results=" + n1 + ", ROIs=" + r1);

// Run 2nd time (without refresh -> append)
run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.45 show_results_table show_roi_manager");
n2 = nResults;
r2 = roiManager("count");
print("  Second run (refresh=false): Results=" + n2 + ", ROIs=" + r2);

if (n2 == n1 * 2 && r2 == r1 * 2) {
    print("  PASS: Data was appended (count doubled).");
} else {
    print("  FAIL: Data was NOT appended correctly! (n2=" + n2 + ", r2=" + r2 + ")");
}

// Cleanup
run("1st Read and Release", "action=release_all enable_log=false");
run("Quit");
