// Detection.java - Unified intermediate representation for one detected object
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * The single intermediate representation every decoder produces and the
 * renderer / NMS post-processing consume, for both YOLO and RF-DETR.
 *
 * <ul>
 *   <li>{@code x, y, width, height}: axis-aligned bounding box in
 *       original image coordinates (letterbox/stretch already undone).
 *       For OBB this is the unrotated rectangle itself
 *       (center &plusmn; size/2); the rotation is in {@code angle}.</li>
 *   <li>{@code angle}: OBB rotation in radians; 0 for other tasks.</li>
 *   <li>{@code keypoints}: pose only, {@code [k] = {x, y, score}} in
 *       original image coordinates; null otherwise.</li>
 *   <li>{@code mask}: segmentation only; a {@link MaskRef} describing
 *       where the mask payload lives (decoder-specific storage stays
 *       outside this record). Null otherwise.</li>
 *   <li>{@code query}: RF-DETR query index (keeps the association with
 *       the prediction-wide mask array); -1 for YOLO.</li>
 * </ul>
 */
public record Detection(int classId, float score,
                        float x, float y, float width, float height,
                        float angle, float[][] keypoints, MaskRef mask, int query) {

    /** Convenience constructor for tasks without angle/keypoints/mask/query payloads. */
    public Detection(int classId, float score, float x, float y, float width, float height) {
        this(classId, score, x, y, width, height, 0f, null, null, -1);
    }
}
