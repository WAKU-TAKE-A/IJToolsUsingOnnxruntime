// ORT_2nd_Segmentation.java - Instance segmentation using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.util.ArrayList;
import java.util.List;

/**
 * 2nd step: Instance segmentation inference.
 */
public class ORT_2nd_Segmentation extends AbstractOrt2ndPlugin {

    private static int     slotChoice       = 0;
    private static double  scoreThreshold   = 0.3;
    private static double  nmsThreshold     = 0.45;
    private static double  maskThreshold    = 0.50;
    private static boolean showResultsTable = true;
    private static boolean showRoiManager   = true;
    private static boolean showOverlay      = true;
    private static int     overlayAlpha     = 80;
    private static boolean exportMask       = false;
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
        return "2nd Segmentation v" + OrtUtil.VERSION;
    }

    @Override
    protected void addTaskFields(GenericDialog gd) {
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addNumericField("mask_threshold",  maskThreshold,  2);
        gd.addCheckbox("show_results_table",  showResultsTable);
        gd.addCheckbox("show_roi_manager",    showRoiManager);
        gd.addCheckbox("show_overlay",        showOverlay);
        gd.addNumericField("overlay_alpha",   overlayAlpha, 0);
        gd.addCheckbox("export_mask",         exportMask);
    }

    @Override
    protected void readTaskFields(GenericDialog gd) {
        scoreThreshold   = gd.getNextNumber();
        nmsThreshold     = gd.getNextNumber();
        maskThreshold    = gd.getNextNumber();
        showResultsTable = gd.getNextBoolean();
        showRoiManager   = gd.getNextBoolean();
        showOverlay      = gd.getNextBoolean();
        overlayAlpha     = (int) gd.getNextNumber();
        exportMask       = gd.getNextBoolean();
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
        if (maskThreshold < 0 || maskThreshold > 1) {
            IJ.showStatus("Mask threshold must be 0-1");
            return false;
        }
        if (overlayAlpha < 0 || overlayAlpha > 255) {
            IJ.showStatus("Overlay alpha must be 0-255");
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

        if (s.getContract() == ModelContract.RF_DETR_SEGMENT) {
            RfDetrResults.run(s, ip, imp,
                    renderOptions(maskThreshold, 0.5, showResultsTable, showRoiManager, showOverlay, overlayAlpha, exportMask),
                    scoreThreshold, nmsThreshold);
            return;
        }

        if (contractOf(s).task() != ModelContract.Task.SEGMENTATION
                || contractOf(s).family() != ModelContract.Family.YOLO) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Segmentation model.");
            return;
        }

        int targetW = s.getInputWidth();
        int targetH = s.getInputHeight();
        int imgW    = ip.getWidth();
        int imgH    = ip.getHeight();

        // 1. Preprocess (contract-driven; tensor + coordinate transform in one pass)
        Preprocessor.Result prep = prepareInput(s, ip);
        float[] inputData = prep.tensor();
        Preprocessor.Transform tf = prep.transform();

        // 2. Inference
        long timer = startTimer();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = elapsedMillis(timer);

            if (result.size() < 2) {
                OrtUtil.logError(this.getClass().getSimpleName(), "YOLO Segmentation requires at least 2 outputs, got " + result.size());
                return;
            }

            // Dynamically identify box and proto tensors based on shapes
            OnnxTensor t0 = (OnnxTensor) result.get(0);
            OnnxTensor t1 = (OnnxTensor) result.get(1);
            long[] shape0 = t0.getInfo().getShape();
            long[] shape1 = t1.getInfo().getShape();

            OnnxTensor boxTensor = null;
            OnnxTensor protoTensor = null;

            if (shape0.length == 3 && shape1.length == 4) {
                boxTensor = t0;
                protoTensor = t1;
            } else if (shape0.length == 4 && shape1.length == 3) {
                boxTensor = t1;
                protoTensor = t0;
            } else {
                boxTensor = t0;
                protoTensor = t1;
            }

            // Output 0: Box & weights [1, C, numAnchors] or [1, numAnchors, C]
            float[][][] raw0 = (float[][][]) boxTensor.getValue();
            float[][] data0 = raw0[0];

            // Output 1: Prototype masks [1, 32, protoH, protoW]
            float[][][][] raw1 = (float[][][][]) protoTensor.getValue();
            float[][][] proto = raw1[0];

            int numClasses = s.getNumClasses();

            List<Detection> candidates = YoloDecoder.decodeSegmentation(
                    data0, numClasses, s.getContract(), (float) scoreThreshold, tf);

            // NMS per the contract's policy (E2E formats pass through)
            List<Detection> nmsResults = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Segmentation candidates (pre-NMS):  " + candidates.size());
                IJ.log("Segmentation candidates (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // Post-processing & Output Visualizations
            ResultsRenderer.renderSegmentation(nmsResults, s, imp, inferenceTime,
                    renderOptions(maskThreshold, 0.5, showResultsTable, showRoiManager, showOverlay, overlayAlpha, exportMask),
                    new ResultsRenderer.ProtoContext(proto, targetW, targetH, tf));

        } catch (Throwable t) {
            handleInferenceFailure("Segmentation inference failed: ", t);
        }
    }
}
