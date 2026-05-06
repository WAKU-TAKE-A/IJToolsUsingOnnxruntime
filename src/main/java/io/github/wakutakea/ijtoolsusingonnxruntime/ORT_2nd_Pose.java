// ORT_2nd_Pose.java - Pose estimation using ONNX Runtime
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
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

/**
 * 2nd step: Pose estimation inference.
 */
public class ORT_2nd_Pose implements ExtendedPlugInFilter, DialogListener {

    private static int     slotChoice     = 0;
    private static double  scoreThreshold = 0.3;
    private static double  nmsThreshold   = 0.45;
    private static double  kptThreshold   = 0.50;
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
        if (IJ.isMacro() && s != null && s.isLoaded()) {
            return DOES_ALL;
        }

        GenericDialog gd = new GenericDialog("2nd Pose");
        gd.addChoice("slot", OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addNumericField("kpt_threshold",   kptThreshold,   2);
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
        kptThreshold   = gd.getNextNumber();
        showRoiManager   = gd.getNextBoolean();
        enableLog        = gd.getNextBoolean();

        if (gd.invalidNumber()) {
            IJ.showStatus("Invalid number");
            return false;
        }
        if (scoreThreshold < 0 || scoreThreshold > 1) {
            IJ.showStatus("Score threshold must be 0-1");
            return false;
        }
        if (nmsThreshold < 0 || nmsThreshold > 1) {
            IJ.showStatus("NMS threshold must be 0-1");
            return false;
        }
        if (kptThreshold < 0 || kptThreshold > 1) {
            IJ.showStatus("Keypoint threshold must be 0-1");
            return false;
        }

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

        // Verify if the model is suitable for Pose
        if (s.getModelType() != MyOrtSession.ModelType.POSE) {
            OrtUtil.logError(this.getClass().getSimpleName(), "Model Type Mismatch - The model in slot " + slotChoice + " is not a Pose model (" + s.getModelType() + ").");
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
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = System.currentTimeMillis() - startTime;

            // 3. Parse output
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][][] raw = (float[][][]) outTensor.getValue();
            float[][] data = raw[0];

            List<float[]> candidates = new ArrayList<>();
            if (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_POSE_E2E) {
                parsePose_E2E(data, imgW, imgH, (float) scoreThreshold, scale, padLeft, padTop, candidates);
            } else {
                parsePose(data, imgW, imgH, (float) scoreThreshold, scale, padLeft, padTop, targetW, targetH, candidates);
            }

            // 4. Per-class NMS (usually only 1 class 'person' for pose, but generic)
            List<float[]> results;
            if (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_POSE_E2E) {
                results = candidates;
            } else {
                results = OrtUtil.nms(candidates, (float) nmsThreshold);
            }

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Pose Detections: " + results.size());
                IJ.log("=".repeat(60));
            }

            // 5. Output
            outputResults(results, s, imageTitle);

        } catch (Throwable t) {
            t.printStackTrace();
            OrtUtil.logError(this.getClass().getSimpleName(), "Pose inference failed " + t.toString());
        }
    }

    private void parsePose(float[][] data, int imgW, int imgH, float scoreTh,
                           float scale, float padLeft, float padTop,
                           int inputW, int inputH, List<float[]> out) {
        int d1 = data.length;
        int d2 = data[0].length;
        int C  = data.length; 

        boolean channelFirst = (d1 > d2); // Simplified check for pose [C, N]
        int numAnchors = channelFirst ? d2 : d1;
        int numKpts = ( (channelFirst ? d1 : d2) - 5 ) / 3;

        for (int i = 0; i < numAnchors; i++) {
            float cx    = channelFirst ? data[0][i] : data[i][0];
            float cy    = channelFirst ? data[1][i] : data[i][1];
            float w     = channelFirst ? data[2][i] : data[i][2];
            float h     = channelFirst ? data[3][i] : data[i][3];
            float score = channelFirst ? data[4][i] : data[i][4];

            if (score < scoreTh) continue;

            float bw = w;
            float bh = h;
            float bx = cx - bw / 2;
            float by = cy - bh / 2;

            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);
            
            float[] res = new float[5 + numKpts * 3];
            res[0] = Math.max(0, Math.min(restored[0], imgW));
            res[1] = Math.max(0, Math.min(restored[1], imgH));
            res[2] = Math.max(0, Math.min(restored[2], imgW - res[0]));
            res[3] = Math.max(0, Math.min(restored[3], imgH - res[1]));
            res[4] = score;

            for (int k = 0; k < numKpts; k++) {
                float kx = channelFirst ? data[5 + k*3][i]   : data[i][5 + k*3];
                float ky = channelFirst ? data[5 + k*3 + 1][i] : data[i][5 + k*3 + 1];
                float ks = channelFirst ? data[5 + k*3 + 2][i] : data[i][5 + k*3 + 2];
                
                float[] rk = OrtUtil.restorePoint(kx, ky, scale, padLeft, padTop);
                res[5 + k*3]     = rk[0];
                res[5 + k*3 + 1] = rk[1];
                res[5 + k*3 + 2] = ks;
            }
            out.add(res);
        }
    }

    private void parsePose_E2E(float[][] data, int imgW, int imgH, float scoreTh,
                               float scale, float padLeft, float padTop,
                               List<float[]> out) {
        int numDetections = data.length;
        int numKpts = (data[0].length - 6) / 3;

        for (int i = 0; i < numDetections; i++) {
            float x1      = data[i][0];
            float y1      = data[i][1];
            float x2      = data[i][2];
            float y2      = data[i][3];
            float score   = data[i][4];
            
            if (score < scoreTh) continue;

            float bw = x2 - x1;
            float bh = y2 - y1;
            float[] restored = OrtUtil.restoreBbox(x1, y1, bw, bh, scale, padLeft, padTop);
            
            float[] res = new float[5 + numKpts * 3];
            res[0] = Math.max(0, Math.min(restored[0], imgW));
            res[1] = Math.max(0, Math.min(restored[1], imgH));
            res[2] = Math.max(0, Math.min(restored[2], imgW - res[0]));
            res[3] = Math.max(0, Math.min(restored[3], imgH - res[1]));
            res[4] = score;

            for (int k = 0; k < numKpts; k++) {
                float kx = data[i][6 + k*3];
                float ky = data[i][6 + k*3 + 1];
                float ks = data[i][6 + k*3 + 2];
                
                float[] rk = OrtUtil.restorePoint(kx, ky, scale, padLeft, padTop);
                res[5 + k*3]     = rk[0];
                res[5 + k*3 + 1] = rk[1];
                res[5 + k*3 + 2] = ks;
            }
            out.add(res);
        }
    }

    private void outputResults(List<float[]> results, MyOrtSession s, String imageTitle) {
        if (showRoiManager) {
            ij.plugin.frame.RoiManager rm = OrtUtil.getRoiManager(false, false);
            for (float[] r : results) {
                ij.gui.Roi roi = new ij.gui.Roi(r[0], r[1], r[2], r[3]);
                roi.setName("person (" + String.format("%.2f", r[4]) + ")");
                roi.setStrokeColor(Color.YELLOW);
                rm.addRoi(roi);
                
                int numKpts = (r.length - 5) / 3;
                for (int k = 0; k < numKpts; k++) {
                    float kx = r[5 + k*3];
                    float ky = r[5 + k*3 + 1];
                    float ks = r[5 + k*3 + 2];
                    if (ks > kptThreshold) {
                        ij.gui.PointRoi p = new ij.gui.PointRoi(kx, ky);
                        p.setName("kp" + k);
                        rm.addRoi(p);
                    }
                }
            }
        }
    }
    
    private static java.awt.Color[] COLORS = { java.awt.Color.RED, java.awt.Color.GREEN, java.awt.Color.BLUE, java.awt.Color.YELLOW, java.awt.Color.CYAN, java.awt.Color.MAGENTA };
    private java.awt.Color getColor(int i) { return COLORS[i % COLORS.length]; }
}
