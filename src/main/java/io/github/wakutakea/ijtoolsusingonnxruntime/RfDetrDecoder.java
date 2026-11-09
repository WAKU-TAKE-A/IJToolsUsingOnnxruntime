// RfDetrDecoder.java - Decodes RF-DETR raw outputs into unified Detections
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * RF-DETR decoder, moved out of {@link RfDetrInference} (Phase 2):
 * sigmoid scoring, global top-K selection across queries/classes
 * (background excluded), optional pose score fusion from keypoint
 * covariance traces, and boxes clamped to the image. The arithmetic
 * is unchanged. RF-DETR performs no NMS here; the optional extra NMS
 * is {@link PostProcessors} with the contract's OPTIONAL policy.
 * Detections keep their query index; segmentation masks live in the
 * prediction-wide array and are referenced by query at render time.
 */
public final class RfDetrDecoder {
    private RfDetrDecoder() {}

    private record Candidate(int query, int cls, float score) {}

    public static List<Detection> decode(float[][] boxes, float[][] logits, float[][][] keypoints,
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
            result.add(new Detection(c.cls, score, x1, y1, x2 - x1, y2 - y1, 0f, kp, null, c.query));
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
}
