// MIT License
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import java.util.ArrayList;
import java.util.List;

/**
 * RF-DETR entry point shared by the three RF-DETR task plugins:
 * predict, apply the contract's NMS policy, render. Presentation
 * itself lives in {@link ResultsRenderer}; the display options are a
 * {@link RenderOptions} record instead of the former boolean list
 * (P1-4).
 */
public final class RfDetrResults {
    private RfDetrResults() {}

    public static void run(MyOrtSession session, ImageProcessor ip, ImagePlus image, RenderOptions o,
            double scoreThreshold, double nmsThreshold) {
        try {
            RfDetrInference.Prediction prediction = RfDetrInference.predict(session, ip, (float) scoreThreshold);
            int beforeNms = prediction.detections().size();
            List<Detection> detections = PostProcessors.applyNms(toUnified(prediction),
                    session.getContract(), (float) nmsThreshold);
            ResultsRenderer.renderRfDetr(detections, prediction.masks(), prediction.inferenceMillis(),
                    session, image, o);
            if (o.enableLog()) IJ.log("RF-DETR " + session.getModelName() + ": " + beforeNms + " -> "
                    + detections.size() + " detections after class-aware NMS (IoU="
                    + nmsThreshold + "), " + prediction.inferenceMillis() + " ms inference");
        } catch (Exception ex) {
            OrtUtil.logError("RF-DETR", ex.toString());
        }
    }

    /**
     * Compatibility entry point for the former positional-argument
     * render call (kept for existing callers/tests); delegates to
     * {@link ResultsRenderer#renderRfDetr}.
     */
    public static void render(RfDetrInference.Prediction prediction, MyOrtSession session, ImagePlus image,
            int slot, double maskThreshold, double keypointThreshold, boolean table, boolean manager,
            boolean showOverlay, int alpha, boolean exportMask, boolean refresh) {
        RenderOptions o = RenderOptions.builder(slot)
                .maskThreshold(maskThreshold)
                .kptThreshold(keypointThreshold)
                .showResultsTable(table)
                .showRoiManager(manager)
                .showOverlay(showOverlay)
                .overlayAlpha(alpha)
                .exportMask(exportMask)
                .enableRefreshData(refresh)
                .build();
        ResultsRenderer.renderRfDetr(toUnified(prediction), prediction.masks(),
                prediction.inferenceMillis(), session, image, o);
    }

    /** Converts a legacy Prediction's detections to unified Detections (masks referenced by query). */
    static List<Detection> toUnified(RfDetrInference.Prediction prediction) {
        List<Detection> out = new ArrayList<>(prediction.detections().size());
        for (RfDetrInference.Detection d : prediction.detections()) {
            MaskRef mask = prediction.masks() != null ? new MaskRef.FullMask(d.query()) : null;
            out.add(new Detection(d.classId(), d.score(), d.x(), d.y(), d.width(), d.height(),
                                  0f, d.keypoints(), mask, d.query()));
        }
        return out;
    }
}
