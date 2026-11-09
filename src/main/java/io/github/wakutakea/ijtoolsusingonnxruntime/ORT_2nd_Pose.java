// ORT_2nd_Pose.java - Pose estimation using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.Color;
import java.util.List;

/**
 * 2nd step: Pose estimation inference.
 */
public class ORT_2nd_Pose extends AbstractOrt2ndPlugin {

    private static int     slotChoice     = 0;
    private static double  scoreThreshold = 0.3;
    private static double  nmsThreshold   = 0.45;
    private static double  kptThreshold   = 0.50;
    private static boolean showRoiManager   = true;
    private static boolean enableRefreshData = true;
    private static boolean enableLog        = true;

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
        return "2nd Pose v" + OrtUtil.VERSION;
    }

    @Override
    protected void addTaskFields(GenericDialog gd) {
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addNumericField("kpt_threshold",   kptThreshold,   2);
        gd.addCheckbox("show_roi_manager",   showRoiManager);
    }

    @Override
    protected void readTaskFields(GenericDialog gd) {
        scoreThreshold = gd.getNextNumber();
        nmsThreshold   = gd.getNextNumber();
        kptThreshold   = gd.getNextNumber();
        showRoiManager = gd.getNextBoolean();
    }

    @Override
    protected boolean validateTaskDialog() {
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

    // ---------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = requireLoadedSession();
        if (s == null) return;

        if (s.getModelType() == MyOrtSession.ModelType.RF_DETR_POSE) {
            RfDetrResults.run(s, ip, imp, slotChoice, scoreThreshold, nmsThreshold, 0.5, kptThreshold,
                    false, showRoiManager, false, 80, false, enableRefreshData, enableLog);
            return;
        }

        // Verify if the model is suitable for Pose
        if (s.getModelType() != MyOrtSession.ModelType.POSE) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Pose model (" + s.getModelType() + ").");
            return;
        }

        int imgW    = ip.getWidth();
        int imgH    = ip.getHeight();
        String imageTitle = imp.getShortTitle();

        // 1. Preprocess (contract-driven; tensor + coordinate transform in one pass)
        Preprocessor.Result prep = prepareInput(s, ip);
        float[] inputData = prep.tensor();

        // 2. Inference
        long timer = startTimer();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = elapsedMillis(timer);

            // 3. Parse output
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][][] raw = (float[][][]) outTensor.getValue();
            float[][] data = raw[0];

            List<Detection> candidates = YoloDecoder.decodePose(
                    data, s.getCoordFormat(), imgW, imgH, (float) scoreThreshold, prep.transform());

            // 4. NMS per the contract's policy (class-agnostic; E2E passes through)
            List<Detection> results = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Pose Detections: " + results.size());
                IJ.log("=".repeat(60));
            }

            // 5. Output
            outputResults(results, s, imageTitle);

        } catch (Throwable t) {
            handleInferenceFailure("Pose inference failed ", t);
        }
    }

    private void outputResults(List<Detection> results, MyOrtSession s, String imageTitle) {
        if (showRoiManager) {
            ij.plugin.frame.RoiManager rm = OrtUtil.getRoiManager(enableRefreshData, false);
            for (Detection r : results) {
                ij.gui.Roi roi = new ij.gui.Roi(r.x(), r.y(), r.width(), r.height());
                roi.setName("person (" + String.format("%.2f", r.score()) + ")");
                roi.setStrokeColor(Color.YELLOW);
                rm.addRoi(roi);

                float[][] keypoints = r.keypoints();
                if (keypoints != null) {
                    for (int k = 0; k < keypoints.length; k++) {
                        float kx = keypoints[k][0];
                        float ky = keypoints[k][1];
                        float ks = keypoints[k][2];
                        if (ks > kptThreshold) {
                            ij.gui.PointRoi p = new ij.gui.PointRoi(kx, ky);
                            p.setName("kp" + k);
                            rm.addRoi(p);
                        }
                    }
                }
            }
        }
    }

    private static java.awt.Color[] COLORS = { java.awt.Color.RED, java.awt.Color.GREEN, java.awt.Color.BLUE, java.awt.Color.YELLOW, java.awt.Color.CYAN, java.awt.Color.MAGENTA };
    private java.awt.Color getColor(int i) { return COLORS[i % COLORS.length]; }
}
