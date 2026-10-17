// MIT License
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.*;
import ij.measure.ResultsTable;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;
import java.awt.GraphicsEnvironment;

/** ImageJ presentation shared by the three RF-DETR task entry points. */
public final class RfDetrResults {
    private RfDetrResults() {}

    public static void run(MyOrtSession session, ImageProcessor ip, ImagePlus image, int slot,
            double scoreThreshold, double nmsThreshold, double maskThreshold, double keypointThreshold,
            boolean table, boolean manager, boolean overlay, int alpha, boolean exportMask,
            boolean refresh, boolean log) {
        try {
            RfDetrInference.Prediction prediction = RfDetrInference.predict(session, ip, (float) scoreThreshold);
            int beforeNms = prediction.detections().size();
            prediction = RfDetrInference.applyNms(prediction, (float) nmsThreshold);
            render(prediction, session, image, slot, maskThreshold, keypointThreshold, table, manager,
                    overlay, alpha, exportMask, refresh);
            if (log) IJ.log("RF-DETR " + session.getModelName() + ": " + beforeNms + " -> "
                    + prediction.detections().size() + " detections after class-aware NMS (IoU="
                    + nmsThreshold + "), " + prediction.inferenceMillis() + " ms inference");
        } catch (Exception ex) {
            OrtUtil.logError("RF-DETR", ex.toString());
        }
    }

    public static void render(RfDetrInference.Prediction prediction, MyOrtSession session, ImagePlus image,
            int slot, double maskThreshold, double keypointThreshold, boolean table, boolean manager,
            boolean showOverlay, int alpha, boolean exportMask, boolean refresh) {
        int width = image.getWidth(), height = image.getHeight();
        ResultsTable rt = table ? OrtUtil.getResultsTable(refresh) : null;
        // Batch-mode macros can still use ImageJ's batch ROI Manager. A truly headless JVM cannot create AWT frames.
        RoiManager rm = null;
        if (manager && !GraphicsEnvironment.isHeadless()) {
            rm = RoiManager.getInstance2();
            if (rm == null) rm = new RoiManager();
            if (refresh) rm.reset();
            if (!IJ.isMacro()) rm.setVisible(true);
        }
        Overlay display = showOverlay ? image.getOverlay() : null;
        if (showOverlay && display == null) display = new Overlay();
        if (showOverlay && refresh) display.clear();
        ByteProcessor union = exportMask ? new ByteProcessor(width, height) : null;
        for (RfDetrInference.Detection d : prediction.detections()) {
            Color color = OrtUtil.getColorForClass(d.classId());
            String label = session.resolveClassName(d.classId());
            Roi roi;
            if (prediction.masks() != null) {
                ByteProcessor binary = new ByteProcessor(width, height);
                float[][] mask = prediction.masks()[d.query()];
                // RF-DETR masks span the entire image; unlike YOLO prototypes they are not cropped to boxes.
                for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
                    if (RfDetrInference.sigmoid(RfDetrInference.maskValue(mask, x, y, width, height)) > maskThreshold) {
                        binary.set(x, y, 255);
                        if (union != null) union.set(x, y, 255);
                    }
                }
                binary.setThreshold(255, 255, ImageProcessor.NO_LUT_UPDATE);
                roi = new ThresholdToSelection().convert(binary);
            } else roi = new Roi(d.x(), d.y(), d.width(), d.height());
            if (roi != null) {
                roi.setName(label + " (" + String.format("%.2f", d.score()) + ")");
                roi.setStrokeColor(color);
                if (prediction.masks() != null) roi.setFillColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha));
                if (rm != null) rm.addRoi(roi);
                if (display != null) display.add(roi);
            }
            if (d.keypoints() != null) {
                for (int k = 0; k < d.keypoints().length; k++) {
                    float[] p = d.keypoints()[k];
                    if (!Float.isFinite(p[0]) || !Float.isFinite(p[1]) || !(p[2] > keypointThreshold)) continue;
                    PointRoi point = new PointRoi(p[0], p[1]);
                    point.setName(label + " kp" + k);
                    point.setStrokeColor(color);
                    if (rm != null) rm.addRoi(point);
                    if (display != null) display.add(point);
                }
            }
            if (rt != null) {
                rt.incrementCounter();
                rt.addValue("Image", image.getShortTitle());
                rt.addValue("Slot", slot);
                rt.addValue("ModelName", session.getModelName());
                rt.addValue("Label", label);
                rt.addValue("ClassID", d.classId());
                rt.addValue("Confidence", d.score());
                rt.addValue("X", d.x()); rt.addValue("Y", d.y());
                rt.addValue("W", d.width()); rt.addValue("H", d.height());
                rt.addValue("InferenceTime(ms)", prediction.inferenceMillis());
            }
        }
        if (display != null) image.setOverlay(display);
        if (rt != null && !IJ.isMacro() && !GraphicsEnvironment.isHeadless()) rt.show("Results");
        if (union != null) {
            ImagePlus maskImage = new ImagePlus(image.getShortTitle() + "_mask", union);
            if (!GraphicsEnvironment.isHeadless()) maskImage.show();
        }
        image.updateAndDraw();
    }
}
