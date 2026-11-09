// PostProcessors.java - Contract-driven post-processing, in one place
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.List;

/**
 * Post-processing stages driven by {@link ModelContract}. NMS used to
 * be assembled per plugin (per-class vs class-agnostic, E2E skips added
 * ad hoc), which is how Segmentation/OBB ended up double-applying NMS
 * to E2E models; the policy now comes from the contract and the
 * application lives here only.
 */
public final class PostProcessors {
    private PostProcessors() {}

    /**
     * Applies the NMS policy declared by the contract. For OBB the IoU
     * boxes are the axis-aligned envelopes of the rotated rectangles,
     * computed exactly as the former per-plugin NMS did (rotated IoU is
     * out of scope).
     */
    public static List<Detection> applyNms(List<Detection> detections, ModelContract contract, float threshold) {
        boolean obbEnvelope = contract.task() == ModelContract.Task.OBB;
        return applyNms(detections, contract.nmsPolicy(), obbEnvelope, threshold);
    }

    /** Applies an explicit policy (also used directly by tests). */
    public static List<Detection> applyNms(List<Detection> detections, NmsPolicy policy,
                                           boolean obbEnvelope, float threshold) {
        return switch (policy) {
            case NONE, NONE_E2E -> detections;
            case PER_CLASS, OPTIONAL ->
                    Nms.suppress(detections, threshold, d -> box(d, obbEnvelope), true);
            case CLASS_AGNOSTIC ->
                    Nms.suppress(detections, threshold, d -> box(d, obbEnvelope), false);
        };
    }

    private static Nms.Box box(Detection d, boolean obbEnvelope) {
        if (!obbEnvelope) {
            return new Nms.Box(d.x(), d.y(), d.width(), d.height(), d.score(), d.classId());
        }
        float cx = d.x() + d.width() / 2;
        float cy = d.y() + d.height() / 2;
        float cosA = (float) Math.abs(Math.cos(d.angle()));
        float sinA = (float) Math.abs(Math.sin(d.angle()));
        float envW = d.width() * cosA + d.height() * sinA;
        float envH = d.width() * sinA + d.height() * cosA;
        return new Nms.Box(cx - envW / 2, cy - envH / 2, envW, envH, d.score(), d.classId());
    }
}
