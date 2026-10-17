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
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid input size");
        int[] rgb = (int[]) ip.convertToRGB().getPixels();
        int srcW = ip.getWidth(), srcH = ip.getHeight(), plane = width * height;
        float[] result = new float[3 * plane];
        float[] mean = {0.485f, 0.456f, 0.406f}, std = {0.229f, 0.224f, 0.225f};
        float scaleX = (float) srcW / width, scaleY = (float) srcH / height;
        for (int y = 0; y < height; y++) {
            float sy = Math.max(0, (y + 0.5f) * scaleY - 0.5f);
            int y0 = (int) sy, y1 = Math.min(y0 + 1, srcH - 1);
            float fy = sy - y0;
            for (int x = 0; x < width; x++) {
                float sx = Math.max(0, (x + 0.5f) * scaleX - 0.5f);
                int x0 = (int) sx, x1 = Math.min(x0 + 1, srcW - 1);
                float fx = sx - x0;
                for (int c = 0; c < 3; c++) {
                    int shift = 16 - c * 8;
                    float a = ((rgb[y0 * srcW + x0] >> shift) & 255) / 255f;
                    float b = ((rgb[y0 * srcW + x1] >> shift) & 255) / 255f;
                    float d = ((rgb[y1 * srcW + x0] >> shift) & 255) / 255f;
                    float e = ((rgb[y1 * srcW + x1] >> shift) & 255) / 255f;
                    float v = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
                    result[c * plane + y * width + x] = (v - mean[c]) / std[c];
                }
            }
        }
        return result;
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
            List<Detection> detections = decode(boxes, logits, keypoints, ip.getWidth(), ip.getHeight(), threshold,
                    Integer.parseInt(meta.get("ijtools.rfdetr.num_select")),
                    Integer.parseInt(meta.get("ijtools.rfdetr.background")),
                    keypoints == null ? new int[0] : keypointCounts(meta),
                    keypoints == null ? 0 : Double.parseDouble(meta.get("ijtools.rfdetr.trace_alpha")));
            return new Prediction(detections, masks, millis);
        }
    }

    private record Candidate(int query, int cls, float score) {}

    /** Optional shared postprocessing, retaining each query's mask and keypoints. */
    public static Prediction applyNms(Prediction prediction, float threshold) {
        List<Detection> kept = Nms.suppress(prediction.detections(), threshold,
                d -> new Nms.Box(d.x(), d.y(), d.width(), d.height(), d.score(), d.classId()), true);
        return new Prediction(kept, prediction.masks(), prediction.inferenceMillis());
    }

    static List<Detection> decode(float[][] boxes, float[][] logits, float[][][] keypoints,
            int width, int height, float threshold, int topK, int background, int[] counts, double traceAlpha) {
        if (!Float.isFinite(threshold) || threshold < 0 || threshold > 1 || topK < 1)
            throw new IllegalArgumentException("Invalid RF-DETR score threshold/top-K");
        // Bound memory to topK even for custom models with many queries/classes.
        Comparator<Candidate> best = Comparator.comparingDouble((Candidate c) -> -c.score)
                .thenComparingInt(Candidate::query).thenComparingInt(Candidate::cls);
        PriorityQueue<Candidate> queue = new PriorityQueue<>(best.reversed());
        for (int q = 0; q < logits.length; q++) {
            for (int c = 0; c < logits[q].length; c++) {
                if (c == background || !Float.isFinite(logits[q][c])) continue;
                Candidate candidate = new Candidate(q, c, sigmoid(logits[q][c]));
                if (queue.size() < topK) queue.add(candidate);
                else if (best.compare(candidate, queue.peek()) < 0) { queue.poll(); queue.add(candidate); }
            }
        }
        List<Candidate> selected = new ArrayList<>(queue);
        selected.sort(best);
        List<Detection> result = new ArrayList<>();
        int maxK = Arrays.stream(counts).max().orElse(0);
        for (Candidate c : selected) {
            float score = c.score;
            float[][] kp = null;
            if (keypoints != null) {
                if (c.cls >= counts.length || counts[c.cls] == 0) continue;
                int count = counts[c.cls], offset = c.cls * maxK;
                kp = new float[count][3];
                for (int k = 0; k < count; k++) {
                    float[] raw = keypoints[c.query][offset + k];
                    kp[k] = new float[]{raw[0] * width, raw[1] * height, sigmoid(raw[2])};
                }
                if (traceAlpha > 0) score = fusedScore(score, keypoints[c.query], offset, count, traceAlpha);
            }
            if (!(score > threshold)) continue;
            float[] b = boxes[c.query];
            if (b.length != 4 || !finite(b) || b[2] <= 0 || b[3] <= 0) continue;
            float x1 = clamp((b[0] - b[2] / 2) * width, width);
            float y1 = clamp((b[1] - b[3] / 2) * height, height);
            float x2 = clamp((b[0] + b[2] / 2) * width, width);
            float y2 = clamp((b[1] + b[3] / 2) * height, height);
            if (x2 <= x1 || y2 <= y1) continue;
            result.add(new Detection(c.query, c.cls, score, x1, y1, x2-x1, y2-y1, kp));
        }
        return result;
    }

    private static float fusedScore(float score, float[][] kp, int offset, int count, double alpha) {
        double logNumerator = Double.NEGATIVE_INFINITY, logDenominator = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < count; k++) {
            float[] p = kp[offset + k];
            double a = -2.0 * p[4], b = -2.0 * p[6];
            double trace = logAdd(logAdd(a, b), 2 * Math.log(Math.max(Math.abs(p[5]), 1e-12)) + a + b);
            double weight = Math.log(Math.max(sigmoid(p[2]), 1e-12));
            logNumerator = logAdd(logNumerator, trace + weight);
            logDenominator = logAdd(logDenominator, weight);
        }
        return Math.min(Math.nextDown(1f), sigmoid(Math.log(score) - alpha * (logNumerator - logDenominator)));
    }
    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }
    static float sigmoid(double x) { return (float) (1.0 / (1.0 + Math.exp(-x))); }
    private static float clamp(float x, int max) { return Math.max(0, Math.min(max, x)); }
    private static boolean finite(float[] values) {
        for (float v : values) if (!Float.isFinite(v)) return false;
        return true;
    }

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
