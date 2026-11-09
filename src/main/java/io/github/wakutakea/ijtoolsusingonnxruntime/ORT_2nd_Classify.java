// ORT_2nd_Classify.java - Image classification using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.gui.GenericDialog;
import ij.measure.ResultsTable;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;

/**
 * 2nd step: Image classification inference.
 */
public class ORT_2nd_Classify extends AbstractOrt2ndPlugin {

    private static int     slotChoice   = 0;
    private static int     topK         = 1;
    private static boolean enableRefreshData = true;
    private static boolean enableLog    = true;

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
        return "2nd Classify v" + OrtUtil.VERSION;
    }

    @Override
    protected void addTaskFields(GenericDialog gd) {
        gd.addNumericField("top_k", topK, 0);
    }

    @Override
    protected void readTaskFields(GenericDialog gd) {
        topK = (int) gd.getNextNumber();
    }

    @Override
    protected boolean validateTaskDialog() {
        if (topK < 1) {
            IJ.showStatus("Top-K must be at least 1");
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

        // Verify if the model is suitable for Classification
        if (contractOf(s).task() != ModelContract.Task.CLASSIFICATION) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Classification model (" + s.getContract().label() + ").");
            return;
        }

        // 1. Preprocess
        float[] inputData = preprocess(s, ip);

        // 2. Inference
        long timer = startTimer();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = elapsedMillis(timer);

            // 3. Parse output (Softmax/Logits)
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][] raw = (float[][]) outTensor.getValue(); // [1, numClasses]
            float[] probs = raw[0];

            // 4. Get top-K
            java.util.List<ClassifyDecoder.Entry> entries = ClassifyDecoder.decode(probs, topK);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                for (int i = 0; i < entries.size(); i++) {
                    ClassifyDecoder.Entry e = entries.get(i);
                    IJ.log("Top " + (i+1) + ": " + s.resolveClassName(e.classId()) + " (" + String.format("%.4f", e.score()) + ")");
                }
                IJ.log("=".repeat(60));
            }

            // 5. Output to Results Table
            ResultsRenderer.renderClassification(entries, s, imp, inferenceTime,
                    renderOptions(0.5, 0.5, true, false, false, 80, false));

        } catch (Throwable t) {
            handleInferenceFailure("Classification failed ", t);
        }
    }

}
