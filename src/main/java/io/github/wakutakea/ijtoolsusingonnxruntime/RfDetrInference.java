// MIT License
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ai.onnxruntime.*;
import ij.process.ImageProcessor;
import java.util.*;

/** RF-DETR raw ONNX contract: stretch-resize, ImageNet RGB, sigmoid and global top-K; no NMS. */
public final class RfDetrInference {
    private RfDetrInference() {}

    public record Detection(int query, int classId, float score, float x, float y,
                            float width, float height, float[][] keypoints) {}
    public record Prediction(List<Detection> detections, float[][][] masks, long inferenceMillis) {}

    public static String validate(MyOrtSession s) throws OrtException {
        Map<String, String> meta = s.getMetadata().getCustomMetadata();
        if (!"1".equals(meta.get("ijtools.rfdetr.contract"))) {
            return "RF-DETR requires an export prepared by RF-DETR/export_rfdetr.py (contract metadata missing).";
        }
        String task = switch (s.getModelType()) {
            case RF_DETR_DETECTION -> "detect";
            case RF_DETR_SEGMENTATION -> "segment";
            case RF_DETR_POSE -> "pose";
            default -> "unknown";
        };
        if (!task.equals(meta.get("ijtools.rfdetr.task"))) return "Selected RF-DETR task does not match the model.";
        Map<String, NodeInfo> inputs = s.getInputInfo();
        if (inputs.size() != 1) return "RF-DETR requires one RGB NCHW input.";
        TensorInfo input = tensorInfo(inputs.values().iterator().next());
        if (input == null || input.type != OnnxJavaType.FLOAT) return "RF-DETR input must be float32.";
        long[] in = input.getShape();
        if (in.length != 4 || in[0] != 1 || in[1] != 3 || in[2] <= 0 || in[3] <= 0)
            return "RF-DETR requires a static [1,3,H,W] input. Re-export with RF-DETR/export_rfdetr.py.";
        Map<String, NodeInfo> outputs = s.getOutputInfo();
        TensorInfo boxes = tensorInfo(outputs.get("dets"));
        TensorInfo labels = tensorInfo(outputs.get("labels"));
        if (!rank(boxes, 3) || !rank(labels, 3)) return "RF-DETR requires named dets and labels float32 tensors.";
        long[] b = boxes.getShape(), l = labels.getShape();
        if (b[0] != 1 || l[0] != 1 || b[2] != 4 || b[1] <= 0 || b[1] != l[1] || l[2] <= 0)
            return "Invalid RF-DETR box/class tensor shapes.";
        try {
            int topK = Integer.parseInt(meta.get("ijtools.rfdetr.num_select"));
            int background = Integer.parseInt(meta.get("ijtools.rfdetr.background"));
            if (topK <= 0 || background < -1 || background >= l[2]) return "Invalid RF-DETR selection metadata.";
            if (task.equals("pose")) {
                TensorInfo kp = tensorInfo(outputs.get("keypoints"));
                if (!rank(kp, 4) || kp.getShape()[0] != 1 || kp.getShape()[1] != b[1]
                        || kp.getShape()[3] < 3) return "Invalid RF-DETR keypoint tensor.";
                int[] counts = keypointCounts(meta);
                int max = Arrays.stream(counts).max().orElse(0);
                double alpha = Double.parseDouble(meta.get("ijtools.rfdetr.trace_alpha"));
                if (max <= 0 || counts.length > l[2] || Arrays.stream(counts).anyMatch(n -> n < 0)
                        || (long) counts.length * max != kp.getShape()[2] || !Double.isFinite(alpha) || alpha < 0
                        || (alpha > 0 && kp.getShape()[3] < 7)) return "Invalid RF-DETR keypoint schema.";
            }
        } catch (RuntimeException ex) { return "Invalid or incomplete RF-DETR metadata: " + ex.getMessage(); }
        if (task.equals("segment")) {
            TensorInfo masks = tensorInfo(outputs.get("masks"));
            if (!rank(masks, 4) || masks.getShape()[0] != 1 || masks.getShape()[1] != b[1]
                    || masks.getShape()[2] <= 0 || masks.getShape()[3] <= 0) return "Invalid RF-DETR mask tensor.";
        }
        return null;
    }

    private static TensorInfo tensorInfo(NodeInfo node) {
        return node != null && node.getInfo() instanceof TensorInfo ? (TensorInfo) node.getInfo() : null;
    }
    private static boolean rank(TensorInfo t, int rank) {
        return t != null && t.type == OnnxJavaType.FLOAT && t.getShape().length == rank;
    }
    private static int[] keypointCounts(Map<String, String> meta) {
        return Arrays.stream(meta.get("ijtools.rfdetr.keypoint_counts").split(",")).mapToInt(Integer::parseInt).toArray();
    }

    /** Half-pixel bilinear interpolation without antialiasing, on float RGB values. */
    public static float[] preprocess(ImageProcessor ip, int width, int height) {
        // The implementation lives in Preprocessor (Phase 2, STRETCH
        // variant of the RF-DETR spec); this method delegates to it.
        return Preprocessor.apply(ip, PreprocessSpec.RFDETR_STRETCH, width, height).tensor();
    }

    public static Prediction predict(MyOrtSession s, ImageProcessor ip, float threshold) throws OrtException {
        String error = validate(s);
        if (error != null) throw new IllegalArgumentException(error);
        float[] input = preprocess(ip, s.getInputWidth(), s.getInputHeight());
        Map<String, String> meta = s.getMetadata().getCustomMetadata();
        long start = System.nanoTime();
        try (OrtSession.Result out = s.runInference(input)) {
            long millis = (System.nanoTime() - start) / 1_000_000;
            float[][] boxes = ((float[][][]) out.get("dets").orElseThrow().getValue())[0];
            float[][] logits = ((float[][][]) out.get("labels").orElseThrow().getValue())[0];
            float[][][] masks = s.getModelType() == MyOrtSession.ModelType.RF_DETR_SEGMENTATION
                    ? ((float[][][][]) out.get("masks").orElseThrow().getValue())[0] : null;
            float[][][] keypoints = s.getModelType() == MyOrtSession.ModelType.RF_DETR_POSE
                    ? ((float[][][][]) out.get("keypoints").orElseThrow().getValue())[0] : null;
            List<io.github.wakutakea.ijtoolsusingonnxruntime.Detection> decoded = RfDetrDecoder.decode(
                    boxes, logits, keypoints, ip.getWidth(), ip.getHeight(), threshold,
                    Integer.parseInt(meta.get("ijtools.rfdetr.num_select")),
                    Integer.parseInt(meta.get("ijtools.rfdetr.background")),
                    keypoints == null ? new int[0] : keypointCounts(meta),
                    keypoints == null ? 0 : Double.parseDouble(meta.get("ijtools.rfdetr.trace_alpha")));
            List<Detection> detections = new ArrayList<>(decoded.size());
            for (io.github.wakutakea.ijtoolsusingonnxruntime.Detection d : decoded) detections.add(fromUnified(d));
            return new Prediction(detections, masks, millis);
        }
    }

    /** Converts the unified Detection back to this class's legacy record (used by Prediction). */
    static Detection fromUnified(io.github.wakutakea.ijtoolsusingonnxruntime.Detection d) {
        return new Detection(d.query(), d.classId(), d.score(), d.x(), d.y(),
                             d.width(), d.height(), d.keypoints());
    }

    /** Optional shared postprocessing, retaining each query's mask and keypoints. */
    public static Prediction applyNms(Prediction prediction, float threshold) {
        List<Detection> kept = Nms.suppress(prediction.detections(), threshold,
                d -> new Nms.Box(d.x(), d.y(), d.width(), d.height(), d.score(), d.classId()), true);
        return new Prediction(kept, prediction.masks(), prediction.inferenceMillis());
    }

    static List<Detection> decode(float[][] boxes, float[][] logits, float[][][] keypoints,
            int width, int height, float threshold, int topK, int background, int[] counts, double traceAlpha) {
        // The implementation lives in RfDetrDecoder (Phase 2); this facade
        // keeps the legacy record type for existing callers and tests.
        List<io.github.wakutakea.ijtoolsusingonnxruntime.Detection> decoded = RfDetrDecoder.decode(
                boxes, logits, keypoints, width, height, threshold, topK, background, counts, traceAlpha);
        List<Detection> output = new ArrayList<>(decoded.size());
        for (io.github.wakutakea.ijtoolsusingonnxruntime.Detection d : decoded) output.add(fromUnified(d));
        return output;
    }

    static float sigmoid(double x) { return RfDetrDecoder.sigmoid(x); }

    /** Sample mask logits with align_corners=false, matching RF-DETR's mask upsampling. */
    public static float maskValue(float[][] mask, int x, int y, int width, int height) {
        int h = mask.length, w = mask[0].length;
        float scaleX = (float) w / width, scaleY = (float) h / height;
        float sx = Math.max(0, (x + 0.5f) * scaleX - 0.5f);
        float sy = Math.max(0, (y + 0.5f) * scaleY - 0.5f);
        int x0 = (int) sx, y0 = (int) sy, x1 = Math.min(x0+1,w-1), y1 = Math.min(y0+1,h-1);
        float fx = sx-x0, fy = sy-y0;
        return (mask[y0][x0]*(1-fx)+mask[y0][x1]*fx)*(1-fy)
                +(mask[y1][x0]*(1-fx)+mask[y1][x1]*fx)*fy;
    }
}
