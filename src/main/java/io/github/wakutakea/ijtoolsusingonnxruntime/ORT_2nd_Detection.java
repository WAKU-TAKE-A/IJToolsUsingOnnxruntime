// ORT_2nd_Detection.java - Object detection using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.util.List;

/**
 * 2nd step: Object detection inference.
 */
public class ORT_2nd_Detection extends AbstractOrt2ndPlugin {

    private static int     slotChoice     = 0;
    private static double  scoreThreshold = 0.3;
    private static double  nmsThreshold   = 0.45;
    private static boolean showResultsTable = true;
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
        return "2nd Detection v" + OrtUtil.VERSION;
    }

    @Override
    protected void addTaskFields(GenericDialog gd) {
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addCheckbox("show_results_table", showResultsTable);
        gd.addCheckbox("show_roi_manager",   showRoiManager);
    }

    @Override
    protected void readTaskFields(GenericDialog gd) {
        scoreThreshold   = gd.getNextNumber();
        nmsThreshold     = gd.getNextNumber();
        showResultsTable = gd.getNextBoolean();
        showRoiManager   = gd.getNextBoolean();
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
        return true;
    }

    // ---------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = requireLoadedSession();
        if (s == null) return;

        if (s.getModelType() == MyOrtSession.ModelType.RF_DETR_DETECTION) {
            RfDetrResults.run(s, ip, imp, slotChoice, scoreThreshold, nmsThreshold, 0.5, 0.5,
                    showResultsTable, showRoiManager, false, 80, false, enableRefreshData, enableLog);
            return;
        }

        // Verify if the model is suitable for Detection
        if (s.getModelType() != MyOrtSession.ModelType.YOLO && s.getModelType() != MyOrtSession.ModelType.YOLOX) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Detection model (" + s.getModelType() + ").");
            return;
        }

        int targetW = s.getInputWidth();
        int targetH = s.getInputHeight();
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

            // 3. Parse output tensor
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][][] raw = (float[][][]) outTensor.getValue(); // [1, d1, d2]
            float[][] data = raw[0]; // [d1, d2]

            List<Detection> candidates = YoloDecoder.decodeDetection(
                    data, isYolox(s), s.getCoordFormat(), s.getNumClasses(), imgW, imgH,
                    (float) scoreThreshold, prep.transform(), targetW, targetH);

            // 4. NMS per the contract's policy (E2E formats pass through)
            List<Detection> nmsResults = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Detections (pre-NMS):  " + candidates.size());
                IJ.log("Detections (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // 5. Output
            outputResults(nmsResults, s, imageTitle, inferenceTime);

        } catch (Throwable t) {
            handleInferenceFailure("Detection inference failed ", t);
        }
    }

    private void outputResults(List<Detection> results, MyOrtSession s, String imageTitle, long inferenceTime) {
        if (showResultsTable) {
            ij.measure.ResultsTable rt = OrtUtil.getResultsTable(enableRefreshData);
            for (Detection r : results) {
                rt.incrementCounter();
                rt.addValue("Image", imageTitle);
                rt.addValue("Slot",  slotChoice);
                rt.addValue("Label", s.resolveClassName(r.classId()));
                rt.addValue("Confidence", r.score());
                rt.addValue("X", r.x());
                rt.addValue("Y", r.y());
                rt.addValue("W", r.width());
                rt.addValue("H", r.height());
                rt.addValue("Inference(ms)", inferenceTime);
            }
            rt.show("Results");
        }

        if (showRoiManager) {
            ij.plugin.frame.RoiManager rm = OrtUtil.getRoiManager(enableRefreshData, false);
            for (Detection r : results) {
                ij.gui.Roi roi = new ij.gui.Roi(r.x(), r.y(), r.width(), r.height());
                roi.setName(s.resolveClassName(r.classId()) + " (" + String.format("%.2f", r.score()) + ")");
                roi.setStrokeColor(OrtUtil.getColorForClass(r.classId()));
                rm.addRoi(roi);
            }
        }
    }
}
