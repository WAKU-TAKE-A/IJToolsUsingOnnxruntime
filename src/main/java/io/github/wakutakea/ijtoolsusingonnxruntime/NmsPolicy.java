// NmsPolicy.java - How NMS is applied to a model's decoded candidates
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * NMS policy declared by a {@link ModelContract} and executed in one
 * place by {@link PostProcessors}. Keeping the policy in the contract
 * (instead of per-plugin if statements) is what prevents regressions
 * like the Segmentation/OBB double-NMS on E2E models.
 */
public enum NmsPolicy {
    /** Suppress duplicates within each class separately. */
    PER_CLASS,
    /** Suppress duplicates regardless of class. */
    CLASS_AGNOSTIC,
    /** E2E models already ran NMS internally; pass candidates through. */
    NONE_E2E,
    /**
     * Optional extra NMS (RF-DETR: the decoder itself performs no NMS;
     * this is an add-on driven by the dialog threshold, where 1.0 keeps
     * every box and therefore effectively disables it).
     */
    OPTIONAL,
    /** No NMS stage at all (classification). */
    NONE
}
