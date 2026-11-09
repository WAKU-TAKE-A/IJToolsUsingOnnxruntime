// ORT_2nd_Segmentation.java - Instance segmentation using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.process.ImageProcessor;
import ij.process.ByteProcessor;
import ai.onnxruntime.*;
import java.awt.Color;
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

        if (s.getModelType() == MyOrtSession.ModelType.RF_DETR_SEGMENTATION) {
            RfDetrResults.run(s, ip, imp, slotChoice, scoreThreshold, nmsThreshold, maskThreshold, 0.5,
                    showResultsTable, showRoiManager, showOverlay, overlayAlpha, exportMask,
                    enableRefreshData, enableLog);
            return;
        }

        if (s.getModelType() != MyOrtSession.ModelType.SEGMENTATION) {
            logError("Model Type Mismatch - The model in slot " + slotChoice + " is not a Segmentation model.");
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
            int protoH = proto[0].length;
            int protoW = proto[0][0].length;

            int numClasses = s.getNumClasses();

            List<Detection> candidates = YoloDecoder.decodeSegmentation(
                    data0, numClasses, s.getCoordFormat(), (float) scoreThreshold, tf);

            // NMS per the contract's policy (E2E formats pass through)
            List<Detection> nmsResults = PostProcessors.applyNms(candidates, contractOf(s), (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Segmentation candidates (pre-NMS):  " + candidates.size());
                IJ.log("Segmentation candidates (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // Post-processing & Output Visualizations
            outputSegmentationResults(nmsResults, proto, protoH, protoW, targetW, targetH, tf, imgW, imgH, imageTitle, s, inferenceTime);

        } catch (Throwable t) {
            handleInferenceFailure("Segmentation inference failed: ", t);
        }
    }

    private void outputSegmentationResults(List<Detection> results, float[][][] proto, int protoH, int protoW,
                                            int targetW, int targetH, Preprocessor.Transform tf,
                                            int imgW, int imgH, String imageTitle, MyOrtSession s, long inferenceTime) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();

        ij.plugin.frame.RoiManager rm = null;
        if (showRoiManager) {
            rm = OrtUtil.getRoiManager(enableRefreshData, false);
        }

        ij.gui.Overlay overlay = null;
        if (showOverlay) {
            overlay = imp.getOverlay();
            if (overlay == null) {
                overlay = new ij.gui.Overlay();
            }
            if (enableRefreshData) {
                overlay.clear();
            }
        }

        ByteProcessor globalMaskProcessor = null;
        if (exportMask) {
            globalMaskProcessor = new ByteProcessor(imgW, imgH);
        }

        ij.measure.ResultsTable rt = null;
        if (showResultsTable) {
            rt = OrtUtil.getResultsTable(enableRefreshData);
        }

        for (Detection r : results) {
            MaskRef.ProtoMask pm = (MaskRef.ProtoMask) r.mask();
            float bx = pm.boxX();
            float by = pm.boxY();
            float bw = pm.boxW();
            float bh = pm.boxH();
            float score = r.score();
            int classId = r.classId();

            float[] coeffs = pm.coeffs();

            // Box in original image coordinates (restored by the decoder), clipped as before
            float rx = Math.max(0, Math.min(r.x(), imgW));
            float ry = Math.max(0, Math.min(r.y(), imgH));
            float rw = Math.max(0, Math.min(r.width(), imgW - rx));
            float rh = Math.max(0, Math.min(r.height(), imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            int wLimit = (int) Math.ceil(rw);
            int hLimit = (int) Math.ceil(rh);
            if (wLimit <= 0 || hLimit <= 0) continue;

            ByteProcessor localMask = new ByteProcessor(wLimit, hLimit);

            // Populate local mask pixels
            for (int dy = 0; dy < hLimit; dy++) {
                for (int dx = 0; dx < wLimit; dx++) {
                    float x = rx + dx;
                    float y = ry + dy;

                    // Map back to letterbox input space
                    float lx = x * scale + padLeft;
                    float ly = y * scale + padTop;

                    if (lx < 0 || lx >= targetW || ly < 0 || ly >= targetH) continue;

                    // Map to prototype mask space
                    float px = lx * (float) protoW / targetW;
                    float py = ly * (float) protoH / targetH;

                    // Crop to bounding box in prototype space to prevent leakage
                    float pbx1 = bx * (float) protoW / targetW;
                    float pby1 = by * (float) protoH / targetH;
                    float pbx2 = (bx + bw) * (float) protoW / targetW;
                    float pby2 = (by + bh) * (float) protoH / targetH;

                    if (px < pbx1 || px > pbx2 || py < pby1 || py > pby2) continue;

                    // Bilinear interpolation
                    int x0 = (int) Math.floor(px);
                    int y0 = (int) Math.floor(py);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    float fdx = px - x0;
                    float fdy = py - y0;

                    float v00 = getProtoVal(proto, coeffs, y0, x0, protoH, protoW);
                    float v01 = getProtoVal(proto, coeffs, y0, x1, protoH, protoW);
                    float v10 = getProtoVal(proto, coeffs, y1, x0, protoH, protoW);
                    float v11 = getProtoVal(proto, coeffs, y1, x1, protoH, protoW);

                    float v = (1.0f - fdy) * ((1.0f - fdx) * v00 + fdx * v01) + fdy * ((1.0f - fdx) * v10 + fdx * v11);
                    float sigVal = 1.0f / (1.0f + (float) Math.exp(-v));

                    if (sigVal > maskThreshold) {
                        localMask.set(dx, dy, 255);
                    }
                }
            }

            // Convert to ROI
            localMask.setThreshold(127, 255, ImageProcessor.NO_LUT_UPDATE);
            ij.gui.Roi roi = new ij.plugin.filter.ThresholdToSelection().convert(localMask);
            if (roi != null) {
                roi.setLocation(rx, ry);
                String nameStr = s.resolveClassName(classId);
                roi.setName(nameStr + " (" + String.format("%.2f", score) + ")");
                Color clsColor = OrtUtil.getColorForClass(classId);
                roi.setStrokeColor(clsColor);

                if (showRoiManager && rm != null) {
                    rm.addRoi(roi);
                }

                if (showOverlay && overlay != null) {
                    Color fillColor = new Color(clsColor.getRed(), clsColor.getGreen(), clsColor.getBlue(), overlayAlpha);
                    roi.setFillColor(fillColor);
                    overlay.add(roi);
                }

                if (exportMask && globalMaskProcessor != null) {
                    for (int dy = 0; dy < hLimit; dy++) {
                        for (int dx = 0; dx < wLimit; dx++) {
                            if (localMask.get(dx, dy) == 255) {
                                globalMaskProcessor.set((int)rx + dx, (int)ry + dy, 255);
                            }
                        }
                    }
                }
            }

            // Record in ResultsTable
            if (showResultsTable && rt != null) {
                rt.incrementCounter();
                rt.addValue("Image", imageTitle);
                rt.addValue("Slot",  slotChoice);
                rt.addValue("ModelName", s.getModelName());
                rt.addValue("Label", s.resolveClassName(classId));
                rt.addValue("Confidence", score);
                rt.addValue("X", rx);
                rt.addValue("Y", ry);
                rt.addValue("W", rw);
                rt.addValue("H", rh);
                rt.addValue("InferenceTime(ms)", inferenceTime);
            }
        }

        if (showOverlay && overlay != null) {
            imp.setOverlay(overlay);
            imp.draw();
        }

        if (showResultsTable && rt != null && !IJ.isMacro()) {
            rt.show("Results");
        }

        if (exportMask && globalMaskProcessor != null) {
            ImagePlus maskImp = new ImagePlus(imageTitle + "_mask", globalMaskProcessor);
            maskImp.show();
        }
    }

    private float getProtoVal(float[][][] proto, float[] coeffs, int y, int x, int protoH, int protoW) {
        y = Math.max(0, Math.min(y, protoH - 1));
        x = Math.max(0, Math.min(x, protoW - 1));
        float sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            sum += coeffs[j] * proto[j][y][x];
        }
        return sum;
    }
}
