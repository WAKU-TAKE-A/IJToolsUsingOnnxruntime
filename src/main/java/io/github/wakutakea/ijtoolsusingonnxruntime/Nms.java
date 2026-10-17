// MIT License
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/** Greedy, confidence-ordered NMS for axis-aligned boxes. Original payloads are retained. */
public final class Nms {
    private Nms() {}

    public record Box(double x, double y, double width, double height, double score, int classId) {}
    private record Candidate<T>(T value, Box box) {}

    /** Suppress boxes only when IoU is strictly above the threshold. Input order breaks score ties. */
    public static <T> List<T> suppress(List<T> values, float threshold, Function<T, Box> adapter,
            boolean classAware) {
        if (!Float.isFinite(threshold) || threshold < 0 || threshold > 1)
            throw new IllegalArgumentException("NMS threshold must be finite and between 0 and 1");
        List<Candidate<T>> candidates = new ArrayList<>(values.size());
        for (T value : values) {
            Box b = adapter.apply(value);
            if (Double.isFinite(b.x) && Double.isFinite(b.y) && Double.isFinite(b.width)
                    && Double.isFinite(b.height) && Double.isFinite(b.score) && b.width > 0 && b.height > 0)
                candidates.add(new Candidate<>(value, b));
        }
        candidates.sort(Comparator.comparingDouble((Candidate<T> c) -> c.box.score).reversed());
        List<T> result = new ArrayList<>();
        boolean[] suppressed = new boolean[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;
            Candidate<T> current = candidates.get(i);
            result.add(current.value);
            for (int j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j]) continue;
                Box other = candidates.get(j).box;
                if ((!classAware || current.box.classId == other.classId) && iou(current.box, other) > threshold)
                    suppressed[j] = true;
            }
        }
        return result;
    }

    /** Array format: x, y, width, height, confidence, classId, optional task-specific payload. */
    public static List<float[]> perClass(List<float[]> boxes, float threshold) {
        return suppress(boxes, threshold, b -> new Box(b[0], b[1], b[2], b[3], b[4], (int) b[5]), true);
    }

    /** Also accepts five-element arrays used by single-class pose estimation. */
    public static List<float[]> classAgnostic(List<float[]> boxes, float threshold) {
        return suppress(boxes, threshold, b -> new Box(b[0], b[1], b[2], b[3], b[4], 0), false);
    }

    public static double iou(Box a, Box b) {
        double iw = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
        double ih = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
        double intersection = iw * ih;
        double union = a.width * a.height + b.width * b.height - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
