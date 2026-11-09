// ORT_2nd_Pose.java - Pose estimation using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
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
        OrtModel s = requireLoadedSession();
        if (s == null) return;

        if (s.getContract() == ModelContract.RF_DETR_POSE) {
            RfDetrResults.run(s, ip, imp,
                    renderOptions(0.5, kptThreshold, false, showRoiManager, false, 80, false),
                    scoreThreshold, nmsThreshold);
            return;
        }

        // Verify if the model is suitable for Pose
        if (contractOf(s).task() != ModelContract.Task.POSE
                || contractOf(s).family() != ModelContract.Family.YOLO) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Pose model (" + s.getContract().label() + ").");
            return;
        }

        int imgW    = ip.getWidth();
        int imgH    = ip.getHeight();

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
                    data, s.getContract(), imgW, imgH, (float) scoreThreshold, prep.transform());

            // 4. NMS per the contract's policy (class-agnostic; E2E passes through)
            List<Detection> results = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Pose Detections: " + results.size());
                IJ.log("=".repeat(60));
            }

            // 5. Output
            ResultsRenderer.renderPose(results, s, imp, inferenceTime,
                    renderOptions(0.5, kptThreshold, true, showRoiManager, false, 80, false));

        } catch (Throwable t) {
            handleInferenceFailure("Pose inference failed ", t);
        }
    }
}
