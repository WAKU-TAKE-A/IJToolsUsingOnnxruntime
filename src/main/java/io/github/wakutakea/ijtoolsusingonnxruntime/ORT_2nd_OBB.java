// ORT_2nd_OBB.java - Oriented Bounding Box (OBB) detection using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.util.ArrayList;
import java.util.List;

/**
 * 2nd step: Oriented Bounding Box (OBB) detection inference.
 *
 * Supported formats:
 *   YOLO_OBB     : standard output [1, 5+numClasses, numAnchors] or [1, numAnchors, 5+numClasses]
 *                  Layout per anchor: cx, cy, w, h, class0..classN-1, angle(rad)
 *   YOLO_OBB_E2E : E2E output [1, N, 7]  (two possible sub-layouts, auto-detected)
 *                  Layout A (x1y1x2y2): x1, y1, x2, y2, score, classId, angle(rad)
 *                  Layout B (cxcywh):   cx, cy, w,  h,  score, classId, angle(rad)
 *
 * Output: PolygonRoi (4 rotated corners) added to ROI Manager and/or Overlay.
 */
public class ORT_2nd_OBB extends AbstractOrt2ndPlugin {

    private static int     slotChoice        = 0;
    private static double  scoreThreshold    = 0.3;
    private static double  nmsThreshold      = 0.45;
    private static boolean showResultsTable  = true;
    private static boolean showRoiManager    = true;
    private static boolean showOverlay       = true;
    private static boolean enableRefreshData = true;
    private static boolean enableLog         = true;

    // ---------------------------------------------------------------
    // AbstractOrt2ndPlugin hooks
    // ---------------------------------------------------------------

    @Override
    protected int getSlotChoice() { return slotChoice; }

    @Override
    protected void setSlotChoice(int v) { slotChoice = v; }

    @Override
    protected boolean isEnableRefreshData() { return enableRefreshData; }

    @Override
    protected void setEnableRefreshData(boolean v) { enableRefreshData = v; }

    @Override
    protected boolean isEnableLog() { return enableLog; }

    @Override
    protected void setEnableLog(boolean v) { enableLog = v; }

    @Override
    protected String dialogTitle() {
        return "2nd OBB v" + OrtUtil.VERSION;
    }

    @Override
    protected void addTaskFields(GenericDialog gd) {
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addCheckbox("show_results_table",  showResultsTable);
        gd.addCheckbox("show_roi_manager",    showRoiManager);
        gd.addCheckbox("show_overlay",        showOverlay);
    }

    @Override
    protected void readTaskFields(GenericDialog gd) {
        scoreThreshold    = gd.getNextNumber();
        nmsThreshold      = gd.getNextNumber();
        showResultsTable  = gd.getNextBoolean();
        showRoiManager    = gd.getNextBoolean();
        showOverlay       = gd.getNextBoolean();
    }

    @Override
    protected boolean validateTaskDialog() {
        if (scoreThreshold < 0 || scoreThreshold > 1) { IJ.showStatus("Score threshold must be 0-1"); return false; }
        if (nmsThreshold   < 0 || nmsThreshold   > 1) { IJ.showStatus("NMS threshold must be 0-1");   return false; }
        return true;
    }

    // ---------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = requireLoadedSession();
        if (s == null) return;
        if (contractOf(s).task() != ModelContract.Task.OBB
                || contractOf(s).family() != ModelContract.Family.YOLO) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not an OBB model.");
            return;
        }

        int targetW    = s.getInputWidth();
        int targetH    = s.getInputHeight();
        int imgW       = ip.getWidth();
        int imgH       = ip.getHeight();

        // 1. Preprocess (contract-driven; tensor + coordinate transform in one pass)
        Preprocessor.Result prep = prepareInput(s, ip);
        float[] inputData = prep.tensor();

        // 2. Inference
        long timer = startTimer();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = elapsedMillis(timer);

            OnnxTensor t0    = (OnnxTensor) result.get(0);
            float[][][] raw0 = (float[][][]) t0.getValue();
            float[][] data   = raw0[0];

            int numClasses = s.getNumClasses();

            List<Detection> candidates = YoloDecoder.decodeObb(
                    data, numClasses, s.getContract(), (float) scoreThreshold, prep.transform());

            // 3. NMS per the contract's policy (axis-aligned envelope IoU
            //    approximation, as before; E2E formats pass through)
            List<Detection> nmsResults = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("OBB candidates (pre-NMS):  " + candidates.size());
                IJ.log("OBB candidates (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // 4. Output
            ResultsRenderer.renderObb(nmsResults, s, imp, inferenceTime,
                    renderOptions(0.5, 0.5, showResultsTable, showRoiManager, showOverlay, 80, false));

        } catch (Throwable t) {
            handleInferenceFailure("OBB inference failed: ", t);
        }
    }

    // ---------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------
}
