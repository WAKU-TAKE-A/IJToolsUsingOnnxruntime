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
        if (s.getModelType() != MyOrtSession.ModelType.CLASSIFICATION) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Classification model (" + s.getModelType() + ").");
            return;
        }

        String imageTitle = imp.getShortTitle();

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
            int[] topIndices = getTopK(probs, topK);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                for (int i = 0; i < topIndices.length; i++) {
                    int idx = topIndices[i];
                    IJ.log("Top " + (i+1) + ": " + s.resolveClassName(idx) + " (" + String.format("%.4f", probs[idx]) + ")");
                }
                IJ.log("=".repeat(60));
            }

            // 5. Output to Results Table
            ResultsTable rt = OrtUtil.getResultsTable(enableRefreshData);
            for (int i = 0; i < topIndices.length; i++) {
                int idx = topIndices[i];
                rt.incrementCounter();
                rt.addValue("Image", imageTitle);
                rt.addValue("Rank",  i + 1);
                rt.addValue("Label", s.resolveClassName(idx));
                rt.addValue("Score", probs[idx]);
                rt.addValue("Inference(ms)", inferenceTime);
            }
            rt.show("Results");

        } catch (Throwable t) {
            handleInferenceFailure("Classification failed ", t);
        }
    }

    private int[] getTopK(float[] probs, int k) {
        int n = probs.length;
        int actualK = Math.min(k, n);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        // Simple selection sort for top-K (fine for typical class counts)
        for (int i = 0; i < actualK; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (probs[indices[j]] > probs[indices[maxIdx]]) {
                    maxIdx = j;
                }
            }
            int temp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = temp;
        }

        int[] result = new int[actualK];
        System.arraycopy(indices, 0, result, 0, actualK);
        return result;
    }
}
