// ORT_2nd_Classify.java - Image classification using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.measure.ResultsTable;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.AWTEvent;

/**
 * 2nd step: Image classification inference.
 */
public class ORT_2nd_Classify implements ExtendedPlugInFilter, DialogListener {

    private static int     slotChoice   = 0;
    private static int     topK         = 1;
    private static boolean enableLog    = true;

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

        GenericDialog gd = new GenericDialog("2nd Classify");
        gd.addChoice("slot", OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addNumericField("top_k", topK, 0);
        gd.addCheckbox("enable_log", enableLog);
        gd.addMessage(OrtUtil.getSlotStatusText());

        gd.addDialogListener(this);
        gd.showDialog();

        if (gd.wasCanceled()) return DONE;
        
        return DOES_ALL;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        slotChoice = gd.getNextChoiceIndex();
        topK       = (int) gd.getNextNumber();
        enableLog  = gd.getNextBoolean();

        if (gd.invalidNumber()) {
            IJ.showStatus("Invalid number");
            return false;
        }
        if (topK < 1) {
            IJ.showStatus("Top-K must be at least 1");
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

        // Verify if the model is suitable for Classification
        if (s.getModelType() != MyOrtSession.ModelType.CLASSIFICATION) {
            OrtUtil.logError(this.getClass().getSimpleName(), "Model Type Mismatch - The model in slot " + slotChoice + " is not a Classification model (" + s.getModelType() + ").");
            return;
        }

        int targetW = s.getInputWidth();
        int targetH = s.getInputHeight();
        String imageTitle = imp.getShortTitle();

        // 1. Preprocess
        float[] inputData = OrtUtil.preprocess(ip, targetW, targetH);

        // 2. Inference
        long startTime = System.currentTimeMillis();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = System.currentTimeMillis() - startTime;

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
            ResultsTable rt = OrtUtil.getResultsTable(false);
            for (int i = 0; i < topIndices.length; i++) {
                int idx = topIndices[i];
                rt.incrementCounter();
                rt.addValue("Image", imageTitle);
                rt.addValue("Rank",  i + 1);
                rt.addValue("Label", s.resolveClassName(idx));
                rt.addValue("Score", probs[idx]);
                rt.addValue("Inference(ms)", inferenceTime);
            }
            if (!IJ.isMacro()) rt.show("Results");

        } catch (Throwable t) {
            t.printStackTrace();
            OrtUtil.logError(this.getClass().getSimpleName(), "Classification failed " + t.toString());
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
