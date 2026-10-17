// Run with ImageJ -macro or -batch. Pass the project root as the macro argument.
// Example: java -Dplugins.dir=C:/tools/ImageJ -jar C:/tools/ImageJ/ij.jar -batch test/test_rfdetr.ijm C:/path/to/project
base = getArgument();
if (base == "") exit("Pass the project root as the macro argument.");
base = replace(base, "\\", "/");
if (!endsWith(base, "/")) base = base + "/";
models = newArray("detection-nano", "segmentation-nano", "keypoint-preview");
formats = newArray("RF_DETR_Object", "RF_DETR_Segment", "RF_DETR_Pose");
commands = newArray("2nd Detection", "2nd Segmentation", "2nd Pose");
for (i=0; i<models.length; i++) {
    sample = "dog.jpg";
    if (i == 2) sample = "person.jpg";
    open(base + "RF-DETR/models/samples/" + sample);
    run("1st Read and Release", "action=read_model model_path=[" + base + "RF-DETR/models/" + models[i] + "/model.onnx] model_format=" + formats[i] + " slot=0 enable_log");
    options = "slot=0 score_threshold=0.3 nms_threshold=0.45 enable_refresh_data enable_log show_roi_manager";
    if (i == 0) options = options + " show_results_table";
    if (i == 1) options = options + " mask_threshold=0.5 overlay_alpha=80 show_results_table show_overlay";
    if (i == 2) options = options + " kpt_threshold=0.5";
    run(commands[i], options);
    count = roiManager("count");
    if (count == 0) exit("FAIL: No ROIs for " + models[i]);
    if (i != 2 && nResults == 0) exit("FAIL: No results for " + models[i]);
    if (i == 0) {
        run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=1 show_results_table show_roi_manager enable_refresh_data");
        rawCount = nResults;
        run("2nd Detection", "slot=0 score_threshold=0.3 nms_threshold=0.10 show_results_table show_roi_manager enable_refresh_data");
        if (rawCount != 7 || nResults != 6 || roiManager("count") != 6)
            exit("FAIL: dog.jpg NMS should reduce 7 detections to 6");
        personCount = 0;
        for (j=0; j<nResults; j++) {
            if (getResult("ClassID", j) == 1) personCount++;
        }
        if (personCount != 1) exit("FAIL: Duplicate person remains after NMS");
        print("PASS: dog.jpg NMS 7 -> 6, person 2 -> 1");
        count = roiManager("count");
    }
    print("PASS: " + models[i] + " ROIs=" + count);
    roiManager("reset");
    run("1st Read and Release", "action=release_all");
    close();
}
print("RF-DETR macro smoke test finished.");
