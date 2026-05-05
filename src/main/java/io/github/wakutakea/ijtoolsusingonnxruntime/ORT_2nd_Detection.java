// ORT_2nd_Detection.java - Object detection using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.AWTEvent;
import java.util.ArrayList;
import java.util.List;

/**
 * 2nd step: Object detection inference.
 */
public class ORT_2nd_Detection implements ExtendedPlugInFilter, DialogListener {

    private static int     slotChoice     = 0;
    private static double  scoreThreshold = 0.3;
    private static double  nmsThreshold   = 0.45;
    private static boolean showResultsTable = true;
    private static boolean showRoiManager   = true;
    private static boolean enableLog        = true;

    private ImagePlus imp;

    @Override
    public int setup(String arg, ImagePlus imp) {
        this.imp = imp;
        return DOES_ALL;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        MyOrtSession s = OrtUtil.getSlot(slotChoice);
        
        // If in macro mode and model is already loaded, skip dialog
        if (IJ.isMacro() && s != null && s.isLoaded()) {
            return DOES_ALL;
        }

        GenericDialog gd = new GenericDialog("2nd Detection");
        gd.addChoice("slot", OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addCheckbox("show_results_table", showResultsTable);
        gd.addCheckbox("show_roi_manager",   showRoiManager);
        gd.addCheckbox("enable_log",         enableLog);
        gd.addMessage(OrtUtil.getSlotStatusText());
        
        gd.addDialogListener(this);
        gd.showDialog();
        
        if (gd.wasCanceled()) return DONE;
        
        return DOES_ALL;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        slotChoice     = gd.getNextChoiceIndex();
        scoreThreshold = gd.getNextNumber();
        nmsThreshold   = gd.getNextNumber();
        showResultsTable = gd.getNextBoolean();
        showRoiManager   = gd.getNextBoolean();
        enableLog        = gd.getNextBoolean();

        if (gd.invalidNumber()) return false;
        if (scoreThreshold < 0 || scoreThreshold > 1) return false;
        if (nmsThreshold < 0 || nmsThreshold > 1) return false;

        return true;
    }

    @Override
    public void setNPasses(int nPasses) {}

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = OrtUtil.getSlot(slotChoice);
        if (s == null || !s.isLoaded()) {
            IJ.error("No model loaded in slot " + slotChoice);
            return;
        }

        int targetW = s.getInputWidth();
        int targetH = s.getInputHeight();
        int imgW    = ip.getWidth();
        int imgH    = ip.getHeight();
        String imageTitle = imp.getShortTitle();

        // 1. Preprocess
        float[] inputData = OrtUtil.preprocess(ip, targetW, targetH);
        float[] lbParams  = OrtUtil.calcLetterboxParams(imgW, imgH, targetW, targetH);
        float scale   = lbParams[0];
        float padLeft = lbParams[1];
        float padTop  = lbParams[2];

        // 2. Inference
        long startTime = System.currentTimeMillis();
        System.out.println("DEBUG: Running inference for " + imageTitle + " using slot " + slotChoice);
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = System.currentTimeMillis() - startTime;

            // 3. Parse output tensor
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][][] raw = (float[][][]) outTensor.getValue(); // [1, d1, d2]
            float[][] data = raw[0]; // [d1, d2]

            List<float[]> candidates = new ArrayList<>();

            if (s.getModelType() == MyOrtSession.ModelType.YOLOX) {
                parseYOLOX(data, s.getNumClasses(), imgW, imgH,
                           (float) scoreThreshold, scale, padLeft, padTop, targetW, candidates);
            } else if (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_OBJECT_E2E) {
                parseYOLO_E2E(data, imgW, imgH,
                              (float) scoreThreshold, scale, padLeft, padTop, candidates);
            } else {
                parseYOLO(data, s.getNumClasses(), s.getCoordFormat(), imgW, imgH,
                          (float) scoreThreshold, scale, padLeft, padTop, targetW, targetH, candidates);
            }

            // 4. Per-class NMS (skip for E2E: model already performed NMS)
            List<float[]> nmsResults;
            boolean isE2E = (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_OBJECT_E2E);
            if (isE2E) {
                nmsResults = candidates;
            } else {
                nmsResults = perClassNms(candidates, (float) nmsThreshold);
            }

            System.out.println("DEBUG: Inference completed in " + inferenceTime + " ms");
            System.out.println("DEBUG: Candidates (pre-NMS): " + candidates.size());
            System.out.println("DEBUG: Candidates (post-NMS): " + nmsResults.size());

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Detections (pre-NMS):  " + candidates.size());
                IJ.log("Detections (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // 5. Output
            outputResults(nmsResults, s, imageTitle, inferenceTime);

        } catch (Throwable t) {
            t.printStackTrace();
            IJ.error("Detection inference failed: " + t.toString());
        }
    }

    // ---------------------------------------------------------------
    // YOLO parser
    // ---------------------------------------------------------------

    private void parseYOLO(float[][] data, int numClasses, MyOrtSession.CoordFormat fmt,
                           int imgW, int imgH, float scoreTh,
                           float scale, float padLeft, float padTop,
                           int inputW, int inputH, List<float[]> out) {

        int d1 = data.length;    
        int d2 = data[0].length; 
        int C  = 4 + numClasses;

        boolean channelFirst = (d1 == C && d1 < d2);
        int numAnchors = channelFirst ? d2 : d1;

        for (int i = 0; i < numAnchors; i++) {
            float cx = channelFirst ? data[0][i] : data[i][0];
            float cy = channelFirst ? data[1][i] : data[i][1];
            float w  = channelFirst ? data[2][i] : data[i][2];
            float h  = channelFirst ? data[3][i] : data[i][3];

            float maxScore = -1;
            int   maxClass = 0;
            for (int c = 0; c < numClasses; c++) {
                float sc = channelFirst ? data[4 + c][i] : data[i][4 + c];
                if (sc > maxScore) { maxScore = sc; maxClass = c; }
            }
            if (maxScore < scoreTh) continue;

            float bx, by, bw, bh;
            if (fmt == MyOrtSession.CoordFormat.YOLO_NORMALIZED) {
                bw = w * inputW;
                bh = h * inputH;
                bx = cx * inputW - bw / 2;
                by = cy * inputH - bh / 2;
            } else { 
                bw = w;
                bh = h;
                bx = cx - bw / 2;
                by = cy - bh / 2;
            }

            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new float[]{rx, ry, rw, rh, maxScore, maxClass});
        }
    }

    private void parseYOLO_E2E(float[][] data, int imgW, int imgH, float scoreTh,
                               float scale, float padLeft, float padTop,
                               List<float[]> out) {
        int numDetections = data.length;
        for (int i = 0; i < numDetections; i++) {
            float x1      = data[i][0];
            float y1      = data[i][1];
            float x2      = data[i][2];
            float y2      = data[i][3];
            float score   = data[i][4];
            int   classId = (int) data[i][5];

            if (score < scoreTh) continue;

            float bw = x2 - x1;
            float bh = y2 - y1;
            float[] restored = OrtUtil.restoreBbox(x1, y1, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new float[]{rx, ry, rw, rh, score, classId});
        }
    }

    private void parseYOLOX(float[][] data, int numClasses, int imgW, int imgH,
                            float scoreTh, float scale, float padLeft, float padTop,
                            int inputW, List<float[]> out) {
        int numAnchors = data.length;
        for (int i = 0; i < numAnchors; i++) {
            float x1 = data[i][0];
            float y1 = data[i][1];
            float x2 = data[i][2];
            float y2 = data[i][3];
            float objScore = data[i][4];
            
            float maxClsScore = -1;
            int   maxClass    = 0;
            for (int c = 0; c < numClasses; c++) {
                float sc = data[i][5 + c];
                if (sc > maxClsScore) { maxClsScore = sc; maxClass = c; }
            }
            float totalScore = objScore * maxClsScore;
            if (totalScore < scoreTh) continue;

            float bw = x2 - x1;
            float bh = y2 - y1;
            float[] restored = OrtUtil.restoreBbox(x1, y1, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new float[]{rx, ry, rw, rh, totalScore, maxClass});
        }
    }

    private List<float[]> perClassNms(List<float[]> candidates, float nmsTh) {
        List<float[]> finalResults = new ArrayList<>();
        int maxClass = -1;
        for (float[] c : candidates) if ((int)c[5] > maxClass) maxClass = (int)c[5];
        
        for (int cls = 0; cls <= maxClass; cls++) {
            List<float[]> clsCandidates = new ArrayList<>();
            for (float[] c : candidates) if ((int)c[5] == cls) clsCandidates.add(c);
            if (!clsCandidates.isEmpty()) {
                finalResults.addAll(OrtUtil.nms(clsCandidates, nmsTh));
            }
        }
        return finalResults;
    }

    private void outputResults(List<float[]> results, MyOrtSession s, String imageTitle, long inferenceTime) {
        if (showResultsTable) {
            ij.measure.ResultsTable rt = OrtUtil.getResultsTable(false);
            for (float[] r : results) {
                rt.incrementCounter();
                rt.addValue("Image", imageTitle);
                rt.addValue("Slot",  slotChoice);
                rt.addValue("Label", s.resolveClassName((int)r[5]));
                rt.addValue("Confidence", r[4]);
                rt.addValue("X", r[0]);
                rt.addValue("Y", r[1]);
                rt.addValue("W", r[2]);
                rt.addValue("H", r[3]);
                rt.addValue("Inference(ms)", inferenceTime);
            }
            if (!IJ.isMacro()) rt.show("Results");
        }

        if (showRoiManager) {
            ij.plugin.frame.RoiManager rm = OrtUtil.getRoiManager(false, false);
            for (float[] r : results) {
                ij.gui.Roi roi = new ij.gui.Roi(r[0], r[1], r[2], r[3]);
                roi.setName(s.resolveClassName((int)r[5]) + " (" + String.format("%.2f", r[4]) + ")");
                roi.setStrokeColor(OrtUtil.getColorForClass((int)r[5]));
                rm.addRoi(roi);
            }
        }
    }
}
