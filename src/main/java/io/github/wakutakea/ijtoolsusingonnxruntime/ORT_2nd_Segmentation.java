// ORT_2nd_Segmentation.java - Instance segmentation using ONNX Runtime
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import ij.process.ByteProcessor;
import ai.onnxruntime.*;
import java.awt.AWTEvent;
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

/**
 * 2nd step: Instance segmentation inference.
 */
public class ORT_2nd_Segmentation implements ExtendedPlugInFilter, DialogListener {

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

    private ImagePlus imp;

    @Override
    public int setup(String arg, ImagePlus imp) {
        this.imp = imp;
        return DOES_ALL;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        GenericDialog gd = new GenericDialog("2nd Segmentation v" + OrtUtil.VERSION);
        gd.addChoice("slot", OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addNumericField("score_threshold", scoreThreshold, 2);
        gd.addNumericField("nms_threshold",   nmsThreshold,   2);
        gd.addNumericField("mask_threshold",  maskThreshold,  2);
        gd.addCheckbox("show_results_table",  showResultsTable);
        gd.addCheckbox("show_roi_manager",    showRoiManager);
        gd.addCheckbox("show_overlay",        showOverlay);
        gd.addNumericField("overlay_alpha",   overlayAlpha, 0);
        gd.addCheckbox("export_mask",         exportMask);
        gd.addCheckbox("enable_refresh_data", enableRefreshData);
        gd.addCheckbox("enable_log",          enableLog);
        gd.addMessage(OrtUtil.getSlotStatusText());
        
        gd.addDialogListener(this);
        gd.showDialog();
        
        if (gd.wasCanceled()) return DONE;

        slotChoice       = gd.getNextChoiceIndex();
        scoreThreshold   = gd.getNextNumber();
        nmsThreshold     = gd.getNextNumber();
        maskThreshold    = gd.getNextNumber();
        showResultsTable = gd.getNextBoolean();
        showRoiManager   = gd.getNextBoolean();
        showOverlay      = gd.getNextBoolean();
        overlayAlpha     = (int) gd.getNextNumber();
        exportMask       = gd.getNextBoolean();
        enableRefreshData= gd.getNextBoolean();
        enableLog        = gd.getNextBoolean();
        
        return DOES_ALL;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        slotChoice       = gd.getNextChoiceIndex();
        scoreThreshold   = gd.getNextNumber();
        nmsThreshold     = gd.getNextNumber();
        maskThreshold    = gd.getNextNumber();
        showResultsTable = gd.getNextBoolean();
        showRoiManager   = gd.getNextBoolean();
        showOverlay      = gd.getNextBoolean();
        overlayAlpha     = (int) gd.getNextNumber();
        exportMask       = gd.getNextBoolean();
        enableRefreshData= gd.getNextBoolean();
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

    @Override
    public void setNPasses(int nPasses) {}

    @Override
    public void run(ImageProcessor ip) {
        MyOrtSession s = OrtUtil.getSlot(slotChoice);
        if (s == null || !s.isLoaded()) {
            OrtUtil.logError(this.getClass().getSimpleName(), "No model loaded in slot " + slotChoice);
            return;
        }

        if (s.getModelType() != MyOrtSession.ModelType.SEGMENTATION) {
            OrtUtil.logError(this.getClass().getSimpleName(), "Model Type Mismatch - The model in slot " + slotChoice + " is not a Segmentation model.");
            return;
        }

        int targetW = s.getInputWidth();
        int targetH = s.getInputHeight();
        int imgW    = ip.getWidth();
        int imgH    = ip.getHeight();
        String imageTitle = imp.getShortTitle();

        // 1. Preprocess
        float[] inputData = OrtUtil.preprocess(ip, targetW, targetH, false);
        float[] lbParams  = OrtUtil.calcLetterboxParams(imgW, imgH, targetW, targetH);
        float scale   = lbParams[0];
        float padLeft = lbParams[1];
        float padTop  = lbParams[2];

        // 2. Inference
        long startTime = System.currentTimeMillis();
        try (OrtSession.Result result = s.runInference(inputData)) {
            long inferenceTime = System.currentTimeMillis() - startTime;

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

            int d1 = data0.length;
            int d2 = data0[0].length;
            int numClasses = s.getNumClasses();

            List<float[]> candidates = new ArrayList<>();

            if (s.getCoordFormat() == MyOrtSession.CoordFormat.YOLO_SEGMENT_E2E) {
                // E2E format: [1, N, 38] = x1,y1,x2,y2 (letterbox px), score, classId, 32 coeffs
                int numDetections = d1;  // e.g. 300
                for (int i = 0; i < numDetections; i++) {
                    float score   = data0[i][4];
                    if (score < scoreThreshold) continue;

                    float x1 = data0[i][0];
                    float y1 = data0[i][1];
                    float x2 = data0[i][2];
                    float y2 = data0[i][3];
                    int   classId = (int) data0[i][5];

                    float[] cand = new float[6 + 32];
                    cand[0] = x1;          // bx = x1 in letterbox space
                    cand[1] = y1;          // by = y1 in letterbox space
                    cand[2] = x2 - x1;    // bw
                    cand[3] = y2 - y1;    // bh
                    cand[4] = score;
                    cand[5] = classId;
                    for (int j = 0; j < 32; j++) cand[6 + j] = data0[i][6 + j];
                    candidates.add(cand);
                }
            } else {
                // Standard format: [1, 4+numClasses+32, numAnchors] or [1, numAnchors, 4+numClasses+32]
                int C = 4 + numClasses + 32;
                boolean channelFirst = (d1 == C && d1 < d2);
                int numAnchors = channelFirst ? d2 : d1;

                for (int i = 0; i < numAnchors; i++) {
                    float cx = channelFirst ? data0[0][i] : data0[i][0];
                    float cy = channelFirst ? data0[1][i] : data0[i][1];
                    float w  = channelFirst ? data0[2][i] : data0[i][2];
                    float h  = channelFirst ? data0[3][i] : data0[i][3];

                    float maxScore = -1.0f;
                    int maxClass = 0;
                    for (int c = 0; c < numClasses; c++) {
                        float score = channelFirst ? data0[4 + c][i] : data0[i][4 + c];
                        if (score > maxScore) { maxScore = score; maxClass = c; }
                    }
                    if (maxScore < scoreThreshold) continue;

                    float bx = cx - w / 2.0f;
                    float by = cy - h / 2.0f;
                    float bw = w;
                    float bh = h;

                    float[] cand = new float[6 + 32];
                    cand[0] = bx;
                    cand[1] = by;
                    cand[2] = bw;
                    cand[3] = bh;
                    cand[4] = maxScore;
                    cand[5] = maxClass;
                    for (int j = 0; j < 32; j++) {
                        cand[6 + j] = channelFirst ? data0[4 + numClasses + j][i] : data0[i][4 + numClasses + j];
                    }
                    candidates.add(cand);
                }
            }

            // NMS
            List<float[]> nmsResults = perClassNms(candidates, (float) nmsThreshold);

            if (enableLog) {
                IJ.log("Inference time: " + inferenceTime + " ms");
                IJ.log("Segmentation candidates (pre-NMS):  " + candidates.size());
                IJ.log("Segmentation candidates (post-NMS): " + nmsResults.size());
                IJ.log("=".repeat(60));
            }

            // Post-processing & Output Visualizations
            outputSegmentationResults(nmsResults, proto, protoH, protoW, targetW, targetH, scale, padLeft, padTop, imgW, imgH, imageTitle, s, inferenceTime);

        } catch (Throwable t) {
            t.printStackTrace();
            OrtUtil.logError(this.getClass().getSimpleName(), "Segmentation inference failed: " + t.toString());
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

    private void outputSegmentationResults(List<float[]> results, float[][][] proto, int protoH, int protoW, 
                                            int targetW, int targetH, float scale, float padLeft, float padTop,
                                            int imgW, int imgH, String imageTitle, MyOrtSession s, long inferenceTime) {
        
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

        for (float[] r : results) {
            float bx = r[0];
            float by = r[1];
            float bw = r[2];
            float bh = r[3];
            float score = r[4];
            int classId = (int) r[5];

            float[] coeffs = new float[32];
            System.arraycopy(r, 6, coeffs, 0, 32);

            // Restore box to original image coordinates
            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
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
