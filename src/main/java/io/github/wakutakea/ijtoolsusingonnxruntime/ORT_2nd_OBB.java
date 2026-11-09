// ORT_2nd_OBB.java - Oriented Bounding Box (OBB) detection using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.Color;
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
        if (s.getModelType() != MyOrtSession.ModelType.OBB) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not an OBB model.");
            return;
        }

        int targetW    = s.getInputWidth();
        int targetH    = s.getInputHeight();
        int imgW       = ip.getWidth();
        int imgH       = ip.getHeight();
        String imgTitle = imp.getShortTitle();

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
                    data, numClasses, s.getCoordFormat(), (float) scoreThreshold, prep.transform());

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
            outputObbResults(nmsResults, imgW, imgH, imgTitle, s, inferenceTime);

        } catch (Throwable t) {
            handleInferenceFailure("OBB inference failed: ", t);
        }
    }

    // ---------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------

    private void outputObbResults(List<Detection> results,
                                  int imgW, int imgH,
                                  String imageTitle, MyOrtSession s, long inferenceTime) {

        ij.plugin.frame.RoiManager rm = null;
        if (showRoiManager) rm = OrtUtil.getRoiManager(enableRefreshData, false);

        ij.gui.Overlay overlay = null;
        if (showOverlay) {
            overlay = imp.getOverlay();
            if (overlay == null) overlay = new ij.gui.Overlay();
            if (enableRefreshData) overlay.clear();
        }

        ij.measure.ResultsTable rt = null;
        if (showResultsTable) rt = OrtUtil.getResultsTable(enableRefreshData);

        for (Detection r : results) {
            float score = r.score();
            int classId = r.classId();
            float angle = r.angle();   // radians
            // Center/size in original space (restored by the decoder)
            float rw = r.width();
            float rh = r.height();
            float cx = r.x() + rw / 2;
            float cy = r.y() + rh / 2;

            if (rw <= 0 || rh <= 0) continue;

            // Compute 4 rotated corners in original image space
            float hw = rw / 2.0f;
            float hh = rh / 2.0f;

            float cosA = (float) Math.cos(angle);
            float sinA = (float) Math.sin(angle);

            // Corners relative to center (top-left, top-right, bottom-right, bottom-left)
            float[] dx = {-hw,  hw,  hw, -hw};
            float[] dy = {-hh, -hh,  hh,  hh};

            int[] px = new int[4];
            int[] py = new int[4];
            for (int k = 0; k < 4; k++) {
                px[k] = Math.round(cx + dx[k] * cosA - dy[k] * sinA);
                py[k] = Math.round(cy + dx[k] * sinA + dy[k] * cosA);
                // clamp to image bounds
                px[k] = Math.max(0, Math.min(px[k], imgW - 1));
                py[k] = Math.max(0, Math.min(py[k], imgH - 1));
            }

            PolygonRoi roi = new PolygonRoi(px, py, 4, Roi.POLYGON);
            String nameStr = s.resolveClassName(classId);
            roi.setName(nameStr + " (" + String.format("%.2f", score) + ")");
            Color clsColor = OrtUtil.getColorForClass(classId);
            roi.setStrokeColor(clsColor);

            if (showRoiManager && rm != null) rm.addRoi(roi);

            if (showOverlay && overlay != null) overlay.add(roi);

            if (showResultsTable && rt != null) {
                rt.incrementCounter();
                rt.addValue("Image",          imageTitle);
                rt.addValue("Slot",           slotChoice);
                rt.addValue("ModelName",      s.getModelName());
                rt.addValue("Label",          nameStr);
                rt.addValue("Confidence",     score);
                rt.addValue("CX",             cx);
                rt.addValue("CY",             cy);
                rt.addValue("W",              rw);
                rt.addValue("H",              rh);
                rt.addValue("Angle(rad)",     angle);
                rt.addValue("Angle(deg)",     (double)(angle * 180.0f / (float)Math.PI));
                rt.addValue("InferenceTime(ms)", inferenceTime);
            }
        }

        if (showOverlay && overlay != null) {
            imp.setOverlay(overlay);
            imp.draw();
        }

        if (showResultsTable && rt != null && !IJ.isMacro()) rt.show("Results");
    }
}
