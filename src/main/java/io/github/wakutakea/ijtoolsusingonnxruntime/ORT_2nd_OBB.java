// ORT_2nd_OBB.java - Oriented Bounding Box (OBB) detection using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.AWTEvent;
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
public class ORT_2nd_OBB implements ExtendedPlugInFilter, DialogListener {

    private static int     slotChoice        = 0;
    private static double  scoreThreshold    = 0.3;
    private static double  nmsThreshold      = 0.45;
    private static boolean showResultsTable  = true;
    private static boolean showRoiManager    = true;
    private static boolean showOverlay       = true;
    private static boolean enableRefreshData = true;
    private static boolean enableLog         = true;

    private ImagePlus imp;

    @Override
    public int setup(String arg, ImagePlus imp) {
        this.imp = imp;
        return DOES_ALL;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        GenericDialog gd = new GenericDialog("2nd OBB v" + OrtUtil.VERSION);
        gd.addChoice("slot",             OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addCheckbox("show_results_table",  showResultsTable);
        gd.addCheckbox("show_roi_manager",    showRoiManager);
        gd.addCheckbox("show_overlay",        showOverlay);
        gd.addCheckbox("enable_refresh_data", enableRefreshData);
        gd.addCheckbox("enable_log",          enableLog);
        gd.addMessage(OrtUtil.getSlotStatusText());

        gd.addDialogListener(this);
        gd.showDialog();

        if (gd.wasCanceled()) return DONE;

        slotChoice        = gd.getNextChoiceIndex();
        scoreThreshold    = gd.getNextNumber();
        nmsThreshold      = gd.getNextNumber();
        showResultsTable  = gd.getNextBoolean();
        showRoiManager    = gd.getNextBoolean();
        showOverlay       = gd.getNextBoolean();
        enableRefreshData = gd.getNextBoolean();
        enableLog         = gd.getNextBoolean();

        return DOES_ALL;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        slotChoice        = gd.getNextChoiceIndex();
        scoreThreshold    = gd.getNextNumber();
        nmsThreshold      = gd.getNextNumber();
        showResultsTable  = gd.getNextBoolean();
        showRoiManager    = gd.getNextBoolean();
        showOverlay       = gd.getNextBoolean();
        enableRefreshData = gd.getNextBoolean();
        enableLog         = gd.getNextBoolean();

        if (gd.invalidNumber()) { IJ.showStatus("Invalid number"); return false; }
        if (scoreThreshold < 0 || scoreThreshold > 1) { IJ.showStatus("Score threshold must be 0-1"); return false; }
        if (nmsThreshold   < 0 || nmsThreshold   > 1) { IJ.showStatus("NMS threshold must be 0-1");   return false; }
        return true;
    }

    @Override
    public void setNPasses(int nPasses) {}

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = OrtUtil.getSlot(slotChoice);
        if (s == null || !s.isLoaded()) {
            OrtUtil.logError(this.getClass().getSimpleName(), "No model loaded in slot " + slotChoice);
            return;
        }
        if (s.getModelType() != MyOrtSession.ModelType.OBB) {
            OrtUtil.logError(this.getClass().getSimpleName(),
                "Model Type Mismatch - The model in slot " + slotChoice + " is not an OBB model.");
            return;
        }

        int targetW    = s.getInputWidth();
        int targetH    = s.getInputHeight();
        int imgW       = ip.getWidth();
        int imgH       = ip.getHeight();
        String imgTitle = imp.getShortTitle();

        // 1. Preprocess
        float[] inputData = OrtUtil.preprocess(ip, targetW, targetH, false);
        float[] lbParams  = OrtUtil.calcLetterboxParams(imgW, imgH, targetW, targetH);
        float scale   = lbParams[0];
        float padLeft = lbParams[1];
        float padTop  = lbParams[2];

        // 2. Inference
        long startTime = System.currentTimeMillis();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = System.currentTimeMillis() - startTime;

            OnnxTensor t0    = (OnnxTensor) result.get(0);
            float[][][] raw0 = (float[][][]) t0.getValue();
            float[][] data   = raw0[0];

            int d1 = data.length;
            int d2 = data[0].length;
            int numClasses = s.getNumClasses();

            if (enableLog) {
                IJ.log("[OBB v2.0] Format: " + s.getCoordFormat()
                    + "  tensor: [" + d1 + ", " + d2 + "]"
                    + "  numClasses: " + numClasses);
                // Print first 3 rows of raw output for diagnostics
                int dbgRows = Math.min(d1, 3);
                int dbgCols = Math.min(d2, 8);
                for (int r = 0; r < dbgRows; r++) {
                    StringBuilder sb = new StringBuilder("  row[").append(r).append("]: ");
                    for (int c = 0; c < dbgCols; c++) sb.append(String.format("%.3f ", data[r][c]));
                    IJ.log(sb.toString());
                }
            }

            // Candidates float[11] = [bx, by, env_w, env_h, score, classId, angle, cx, cy, w, h]
            // We compute the axis-aligned envelope (bx,by,env_w,env_h) strictly for the IoU NMS approximation.
            List<float[]> candidates = new ArrayList<>();

            if (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_OBB_E2E) {
                // E2E format: [1, N, 7] = cx, cy, w, h, score, classId, angle
                for (int i = 0; i < d1; i++) {
                    float score = data[i][4];
                    if (score < scoreThreshold) continue;
                    float cx = data[i][0];
                    float cy = data[i][1];
                    float w  = data[i][2];
                    float h  = data[i][3];
                    int classId = (int) data[i][5];
                    float angle = data[i][6];
                    
                    if (w <= 0 || h <= 0) continue;

                    // Calculate axis-aligned envelope for NMS
                    float cosA = (float) Math.abs(Math.cos(angle));
                    float sinA = (float) Math.abs(Math.sin(angle));
                    float envW = w * cosA + h * sinA;
                    float envH = w * sinA + h * cosA;
                    float bx = cx - envW / 2.0f;
                    float by = cy - envH / 2.0f;

                    candidates.add(new float[]{bx, by, envW, envH, score, classId, angle, cx, cy, w, h});
                }
            } else {
                // Standard format: [1, 5+numClasses, numAnchors] or [1, numAnchors, 5+numClasses]
                // Per anchor: cx,cy,w,h, class0..classN-1, angle
                int C = 5 + numClasses;
                boolean channelFirst = (d1 == C && d1 < d2);
                int numAnchors = channelFirst ? d2 : d1;

                float overallMaxScore = -1.0f;

                for (int i = 0; i < numAnchors; i++) {
                    float cx = channelFirst ? data[0][i] : data[i][0];
                    float cy = channelFirst ? data[1][i] : data[i][1];
                    float w  = channelFirst ? data[2][i] : data[i][2];
                    float h  = channelFirst ? data[3][i] : data[i][3];

                    float maxScore = -1.0f;
                    int   maxClass = 0;
                    for (int c = 0; c < numClasses; c++) {
                        float sc = channelFirst ? data[4 + c][i] : data[i][4 + c];
                        if (sc > maxScore) { maxScore = sc; maxClass = c; }
                    }
                    
                    if (maxScore > overallMaxScore) {
                        overallMaxScore = maxScore;
                    }

                    if (maxScore < scoreThreshold) continue;

                    float angle = channelFirst ? data[4 + numClasses][i] : data[i][4 + numClasses];
                    
                    // Calculate axis-aligned envelope for NMS
                    float cosA = (float) Math.abs(Math.cos(angle));
                    float sinA = (float) Math.abs(Math.sin(angle));
                    float envW = w * cosA + h * sinA;
                    float envH = w * sinA + h * cosA;
                    float bx = cx - envW / 2.0f;
                    float by = cy - envH / 2.0f;

                    candidates.add(new float[]{bx, by, envW, envH, maxScore, maxClass, angle, cx, cy, w, h});
                }
                
                if (enableLog) {
                    IJ.log("  Max score across all anchors: " + overallMaxScore);
                }
            }

            // 3. NMS (axis-aligned IoU approximation is sufficient for moderate angles)
            List<float[]> nmsResults = perClassNms(candidates, (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("OBB candidates (pre-NMS):  " + candidates.size());
                IJ.log("OBB candidates (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // 4. Output
            outputObbResults(nmsResults, scale, padLeft, padTop, imgW, imgH, imgTitle, s, inferenceTime);

        } catch (Throwable t) {
            t.printStackTrace();
            OrtUtil.logError(this.getClass().getSimpleName(), "OBB inference failed: " + t.toString());
        }
    }

    // ---------------------------------------------------------------
    // NMS
    // ---------------------------------------------------------------

    private List<float[]> perClassNms(List<float[]> candidates, float nmsTh) {
        List<float[]> results = new ArrayList<>();
        int maxClass = -1;
        for (float[] c : candidates) if ((int) c[5] > maxClass) maxClass = (int) c[5];
        for (int cls = 0; cls <= maxClass; cls++) {
            List<float[]> clsCandidates = new ArrayList<>();
            for (float[] c : candidates) if ((int) c[5] == cls) clsCandidates.add(c);
            if (!clsCandidates.isEmpty()) results.addAll(OrtUtil.nms(clsCandidates, nmsTh));
        }
        return results;
    }

    // ---------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------

    private void outputObbResults(List<float[]> results,
                                  float scale, float padLeft, float padTop,
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

        for (float[] r : results) {
            float score = r[4];
            int classId = (int) r[5];
            float angle = r[6];   // radians
            float cx_lb = r[7];
            float cy_lb = r[8];
            float w_lb  = r[9];
            float h_lb  = r[10];

            // Restore directly from letterbox center to original space
            float cx = (cx_lb - padLeft) / scale;
            float cy = (cy_lb - padTop) / scale;
            float rw = w_lb / scale;
            float rh = h_lb / scale;

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
