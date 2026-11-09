// MaskRef.java - Reference to a segmentation mask payload, per decoder family
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * How a {@link Detection}'s segmentation mask is materialized. The two
 * families store masks differently, so the record only carries a
 * reference; the renderer resolves it.
 */
public sealed interface MaskRef {

    /**
     * YOLO segmentation: the mask is {@code coeffs} combined with the
     * prototype tensor at render time. {@code boxX/boxY/boxW/boxH} are
     * the candidate box in letterboxed input space, needed to crop the
     * prototype region (the Detection bbox itself is already restored
     * to original image coordinates).
     */
    record ProtoMask(float[] coeffs, float boxX, float boxY, float boxW, float boxH) implements MaskRef {}

    /**
     * RF-DETR segmentation: the mask is {@code masks[query]} of the
     * prediction-wide mask array held by the caller.
     */
    record FullMask(int query) implements MaskRef {}
}
