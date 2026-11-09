// ResultsRenderer.java - Unified results-table / ROI / overlay output for all tasks
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.gui.PointRoi;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.measure.ResultsTable;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;
import java.awt.GraphicsEnvironment;
import java.util.List;

/**
 * The single output stage for every task (Phase 2). All tasks write
 * the unified column set, in this order:
 * {@code Image, Slot, ModelName, Label, ClassID, Confidence, X, Y, W, H,
 * InferenceTime(ms)}; OBB appends its task-specific columns
 * ({@code CX, CY, Angle(rad), Angle(deg)}) after the common ones, and
 * classification uses its own layout
 * ({@code Image, Slot, ModelName, Rank, Label, Score, InferenceTime(ms)}).
 * The per-task rendering bodies are ported unchanged from the former
 * per-plugin output methods and {@code RfDetrResults.render}.
 */
public final class ResultsRenderer {
    private ResultsRenderer() {}

    /** Prototype-mask context a YOLO segmentation rendering needs. */
    public record ProtoContext(float[][][] proto, int inputW, int inputH, Preprocessor.Transform transform) {}

    // ---------------------------------------------------------------
    // Shared row writer (the unified column set)
    // ---------------------------------------------------------------

    private static void addRow(ResultsTable rt, String imageTitle, int slot, String modelName,
                               String label, int classId, float confidence,
                               float x, float y, float w, float h, long inferenceTime) {
        rt.incrementCounter();
        rt.addValue("Image", imageTitle);
        rt.addValue("Slot", slot);
        rt.addValue("ModelName", modelName);
        rt.addValue("Label", label);
        rt.addValue("ClassID", classId);
        rt.addValue("Confidence", confidence);
        rt.addValue("X", x);
        rt.addValue("Y", y);
        rt.addValue("W", w);
        rt.addValue("H", h);
        rt.addValue("InferenceTime(ms)", inferenceTime);
    }

    private static void addCommonRow(ResultsTable rt, String imageTitle, RenderOptions o,
                                     MyOrtSession s, Detection d, long inferenceTime) {
        addRow(rt, imageTitle, o.slot(), s.getModelName(), s.resolveClassName(d.classId()),
               d.classId(), d.score(), d.x(), d.y(), d.width(), d.height(), inferenceTime);
    }

    // ---------------------------------------------------------------
    // Detection
    // ---------------------------------------------------------------

    public static void renderDetections(List<Detection> results, MyOrtSession s, ImagePlus imp,
                                        long inferenceTime, RenderOptions o) {
        if (o.showResultsTable()) {
            ResultsTable rt = OrtUtil.getResultsTable(o.enableRefreshData());
            for (Detection r : results) {
                addCommonRow(rt, imp.getShortTitle(), o, s, r, inferenceTime);
            }
            rt.show("Results");
        }

        if (o.showRoiManager()) {
            RoiManager rm = OrtUtil.getRoiManager(o.enableRefreshData(), false);
            for (Detection r : results) {
                Roi roi = new Roi(r.x(), r.y(), r.width(), r.height());
                roi.setName(s.resolveClassName(r.classId()) + " (" + String.format("%.2f", r.score()) + ")");
                roi.setStrokeColor(OrtUtil.getColorForClass(r.classId()));
                rm.addRoi(roi);
            }
        }
    }

    // ---------------------------------------------------------------
    // Pose (results table is new in Phase 2; keypoints stay ROI-only)
    // ---------------------------------------------------------------

    public static void renderPose(List<Detection> results, MyOrtSession s, ImagePlus imp,
                                  long inferenceTime, RenderOptions o) {
        if (o.showResultsTable()) {
            ResultsTable rt = OrtUtil.getResultsTable(o.enableRefreshData());
            for (Detection r : results) {
                addCommonRow(rt, imp.getShortTitle(), o, s, r, inferenceTime);
            }
            rt.show("Results");
        }

        if (o.showRoiManager()) {
            RoiManager rm = OrtUtil.getRoiManager(o.enableRefreshData(), false);
            for (Detection r : results) {
                Roi roi = new Roi(r.x(), r.y(), r.width(), r.height());
                roi.setName("person (" + String.format("%.2f", r.score()) + ")");
                roi.setStrokeColor(Color.YELLOW);
                rm.addRoi(roi);

                float[][] keypoints = r.keypoints();
                if (keypoints != null) {
                    for (int k = 0; k < keypoints.length; k++) {
                        float kx = keypoints[k][0];
                        float ky = keypoints[k][1];
                        float ks = keypoints[k][2];
                        if (ks > o.kptThreshold()) {
                            PointRoi p = new PointRoi(kx, ky);
                            p.setName("kp" + k);
                            rm.addRoi(p);
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Segmentation (YOLO prototypes)
    // ---------------------------------------------------------------

    public static void renderSegmentation(List<Detection> results, MyOrtSession s, ImagePlus imp,
                                          long inferenceTime, RenderOptions o, ProtoContext ctx) {
        float[][][] proto = ctx.proto();
        int protoH = proto[0].length;
        int protoW = proto[0][0].length;
        int targetW = ctx.inputW();
        int targetH = ctx.inputH();
        Preprocessor.Transform tf = ctx.transform();
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int imgW = imp.getWidth();
        int imgH = imp.getHeight();
        String imageTitle = imp.getShortTitle();
        double maskThreshold = o.maskThreshold();
        int overlayAlpha = o.overlayAlpha();
        boolean showRoiManager = o.showRoiManager();
        boolean showOverlay = o.showOverlay();
        boolean showResultsTable = o.showResultsTable();
        boolean exportMask = o.exportMask();
        boolean enableRefreshData = o.enableRefreshData();

        RoiManager rm = null;
        if (showRoiManager) {
            rm = OrtUtil.getRoiManager(enableRefreshData, false);
        }

        Overlay overlay = null;
        if (showOverlay) {
            overlay = imp.getOverlay();
            if (overlay == null) {
                overlay = new Overlay();
            }
            if (enableRefreshData) {
                overlay.clear();
            }
        }

        ByteProcessor globalMaskProcessor = null;
        if (exportMask) {
            globalMaskProcessor = new ByteProcessor(imgW, imgH);
        }

        ResultsTable rt = null;
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
            Roi roi = new ThresholdToSelection().convert(localMask);
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

            // Record in ResultsTable (box values are the clipped ones, as before)
            if (showResultsTable && rt != null) {
                addRow(rt, imageTitle, o.slot(), s.getModelName(), s.resolveClassName(classId),
                       classId, score, rx, ry, rw, rh, inferenceTime);
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

    private static float getProtoVal(float[][][] proto, float[] coeffs, int y, int x, int protoH, int protoW) {
        y = Math.max(0, Math.min(y, protoH - 1));
        x = Math.max(0, Math.min(x, protoW - 1));
        float sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            sum += coeffs[j] * proto[j][y][x];
        }
        return sum;
    }

    // ---------------------------------------------------------------
    // OBB (common columns carry the left-top conversion; CX/CY/Angle follow)
    // ---------------------------------------------------------------

    public static void renderObb(List<Detection> results, MyOrtSession s, ImagePlus imp,
                                 long inferenceTime, RenderOptions o) {
        int imgW = imp.getWidth();
        int imgH = imp.getHeight();
        String imageTitle = imp.getShortTitle();
        boolean showRoiManager = o.showRoiManager();
        boolean showOverlay = o.showOverlay();
        boolean showResultsTable = o.showResultsTable();
        boolean enableRefreshData = o.enableRefreshData();

        RoiManager rm = null;
        if (showRoiManager) rm = OrtUtil.getRoiManager(enableRefreshData, false);

        Overlay overlay = null;
        if (showOverlay) {
            overlay = imp.getOverlay();
            if (overlay == null) overlay = new Overlay();
            if (enableRefreshData) overlay.clear();
        }

        ResultsTable rt = null;
        if (showResultsTable) rt = OrtUtil.getResultsTable(enableRefreshData);

        for (Detection r : results) {
            float score = r.score();
            int classId = r.classId();
            float angle = r.angle();   // radians
            // Center/size in original space (restored by the decoder)
            float rw = r.width();
            float rh = r.height();
            float cx = r.x() + rw / 2;
            float cy = r.y() + rh / 2;

            if (rw <= 0 || rh <= 0) continue;

            // Compute 4 rotated corners in original image space
            float hw = rw / 2.0f;
            float hh = rh / 2.0f;

            float cosA = (float) Math.cos(angle);
            float sinA = (float) Math.sin(angle);

            // Corners relative to center (top-left, top-right, bottom-right, bottom-left)
            float[] dx = {-hw,  hw,  hw, -hw};
            float[] dy = {-hh, -hh,  hh,  hh};

            int[] px = new int[4];
            int[] py = new int[4];
            for (int k = 0; k < 4; k++) {
                px[k] = Math.round(cx + dx[k] * cosA - dy[k] * sinA);
                py[k] = Math.round(cy + dx[k] * sinA + dy[k] * cosA);
                // clamp to image bounds
                px[k] = Math.max(0, Math.min(px[k], imgW - 1));
                py[k] = Math.max(0, Math.min(py[k], imgH - 1));
            }

            PolygonRoi roi = new PolygonRoi(px, py, 4, Roi.POLYGON);
            String nameStr = s.resolveClassName(classId);
            roi.setName(nameStr + " (" + String.format("%.2f", score) + ")");
            Color clsColor = OrtUtil.getColorForClass(classId);
            roi.setStrokeColor(clsColor);

            if (showRoiManager && rm != null) rm.addRoi(roi);

            if (showOverlay && overlay != null) overlay.add(roi);

            if (showResultsTable && rt != null) {
                // Common columns first (X/Y are the left-top conversion
                // CX-W/2, CY-H/2, i.e. the Detection box itself), then
                // the OBB-specific columns.
                addRow(rt, imageTitle, o.slot(), s.getModelName(), nameStr,
                       classId, score, r.x(), r.y(), rw, rh, inferenceTime);
                rt.addValue("CX",             cx);
                rt.addValue("CY",             cy);
                rt.addValue("Angle(rad)",     angle);
                rt.addValue("Angle(deg)",     (double)(angle * 180.0f / (float)Math.PI));
            }
        }

        if (showOverlay && overlay != null) {
            imp.setOverlay(overlay);
            imp.draw();
        }

        if (showResultsTable && rt != null && !IJ.isMacro()) rt.show("Results");
    }

    // ---------------------------------------------------------------
    // Classification
    // ---------------------------------------------------------------

    public static void renderClassification(List<ClassifyDecoder.Entry> entries, MyOrtSession s,
                                            ImagePlus imp, long inferenceTime, RenderOptions o) {
        ResultsTable rt = OrtUtil.getResultsTable(o.enableRefreshData());
        for (int i = 0; i < entries.size(); i++) {
            ClassifyDecoder.Entry e = entries.get(i);
            rt.incrementCounter();
            rt.addValue("Image", imp.getShortTitle());
            rt.addValue("Slot",  o.slot());
            rt.addValue("ModelName", s.getModelName());
            rt.addValue("Rank",  i + 1);
            rt.addValue("Label", s.resolveClassName(e.classId()));
            rt.addValue("Score", e.score());
            rt.addValue("InferenceTime(ms)", inferenceTime);
        }
        rt.show("Results");
    }

    // ---------------------------------------------------------------
    // RF-DETR (ported from RfDetrResults.render)
    // ---------------------------------------------------------------

    public static void renderRfDetr(List<Detection> detections, float[][][] masks, long inferenceMillis,
                                    MyOrtSession session, ImagePlus image, RenderOptions o) {
        int slot = o.slot();
        double maskThreshold = o.maskThreshold();
        double keypointThreshold = o.kptThreshold();
        boolean table = o.showResultsTable();
        boolean manager = o.showRoiManager();
        boolean showOverlay = o.showOverlay();
        int alpha = o.overlayAlpha();
        boolean exportMask = o.exportMask();
        boolean refresh = o.enableRefreshData();

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
        for (Detection d : detections) {
            Color color = OrtUtil.getColorForClass(d.classId());
            String label = session.resolveClassName(d.classId());
            Roi roi;
            if (masks != null) {
                ByteProcessor binary = new ByteProcessor(width, height);
                float[][] mask = masks[d.query()];
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
                if (masks != null) roi.setFillColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha));
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
                addCommonRow(rt, image.getShortTitle(), o, session, d, inferenceMillis);
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
