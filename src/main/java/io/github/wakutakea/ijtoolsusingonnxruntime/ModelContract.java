// ModelContract.java - Single source of truth for model formats and their contracts
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.TensorInfo;
import java.util.Map;

/**
 * The list of selectable model formats in "1st Read and Release", and
 * the single definition of everything a format implies.
 *
 * <p>This enum is the source of truth for the format labels and their
 * indices. The declaration order must exactly match the historical
 * {@code FORMAT_LABELS} order in {@code ORT_1st_ReadAndRelease}: macros
 * may record the format by index or by label string, so neither the
 * order nor the label strings may change.
 *
 * <p>Each contract declares the task, the model family, the
 * preprocessing spec and the NMS policy of the format, and owns the
 * format validation ({@link #validate}) and the class-count estimation
 * ({@link #resolveNumClasses}) for its formats. Adding a format is:
 * one enum line here, one branch in the matching decoder, and a test.
 * (Phase 3: this enum replaced the former MyOrtSession.ModelType /
 * CoordFormat pair; a loaded model now carries exactly one contract.)
 */
public enum ModelContract {

    YOLO_OBJECT_PIXEL    ("YOLO_Object_Pixel",       Task.DETECTION,     Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.PER_CLASS),
    YOLO_OBJECT_NORMALIZED("YOLO_Object_Normalized", Task.DETECTION,     Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.PER_CLASS),
    YOLO_OBJECT_E2E      ("YOLO_Object_E2E",         Task.DETECTION,     Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.NONE_E2E),
    YOLO_CLASS           ("YOLO_Class",              Task.CLASSIFICATION, Family.YOLO,   PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.NONE),
    YOLO_POSE            ("YOLO_Pose",               Task.POSE,          Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.CLASS_AGNOSTIC),
    YOLO_POSE_E2E        ("YOLO_Pose_E2E",           Task.POSE,          Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.NONE_E2E),
    YOLO_SEGMENT         ("YOLO_Segment",            Task.SEGMENTATION,  Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.PER_CLASS),
    YOLO_SEGMENT_E2E     ("YOLO_Segment_E2E",        Task.SEGMENTATION,  Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.NONE_E2E),
    YOLO_OBB             ("YOLO_OBB",                Task.OBB,           Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.PER_CLASS),
    YOLO_OBB_E2E         ("YOLO_OBB_E2E",            Task.OBB,           Family.YOLO,    PreprocessSpec.YOLO_LETTERBOX,  NmsPolicy.NONE_E2E),
    YOLOX_OBJECT_UNDECODED("YOLOX_Object_Undecoded", Task.DETECTION,     Family.YOLO,    PreprocessSpec.YOLOX_LETTERBOX, NmsPolicy.PER_CLASS),
    RF_DETR_OBJECT       ("RF_DETR_Object",          Task.DETECTION,     Family.RF_DETR, PreprocessSpec.RFDETR_STRETCH,  NmsPolicy.OPTIONAL),
    RF_DETR_SEGMENT      ("RF_DETR_Segment",         Task.SEGMENTATION,  Family.RF_DETR, PreprocessSpec.RFDETR_STRETCH,  NmsPolicy.OPTIONAL),
    RF_DETR_POSE         ("RF_DETR_Pose",            Task.POSE,          Family.RF_DETR, PreprocessSpec.RFDETR_STRETCH,  NmsPolicy.OPTIONAL);

    /** The inference task a format belongs to. */
    public enum Task { DETECTION, CLASSIFICATION, POSE, SEGMENTATION, OBB }

    /** The model family a format belongs to. */
    public enum Family { YOLO, RF_DETR }

    /** Result of {@link #resolveNumClasses}: the count and the rule that produced it. */
    public record NumClasses(int value, String basis) {}

    private final String label;
    private final Task task;
    private final Family family;
    private final PreprocessSpec preprocess;
    private final NmsPolicy nmsPolicy;

    ModelContract(String label, Task task, Family family,
                  PreprocessSpec preprocess, NmsPolicy nmsPolicy) {
        this.label = label;
        this.task = task;
        this.family = family;
        this.preprocess = preprocess;
        this.nmsPolicy = nmsPolicy;
    }

    /** Dialog label, identical to the historical FORMAT_LABELS entry. */
    public String label() {
        return label;
    }

    public Task task() {
        return task;
    }

    public Family family() {
        return family;
    }

    /** Preprocessing contract of this format. */
    public PreprocessSpec preprocess() {
        return preprocess;
    }

    /** NMS policy of this format, executed by {@link PostProcessors}. */
    public NmsPolicy nmsPolicy() {
        return nmsPolicy;
    }

    /** Whether this format's decoder is the YOLOX grid decoder. */
    public boolean isYolox() {
        return this == YOLOX_OBJECT_UNDECODED;
    }

    /** Whether this format is an end-to-end (NMS-free, per-row class id) variant. */
    public boolean isE2e() {
        return nmsPolicy == NmsPolicy.NONE_E2E;
    }

    // ---------------------------------------------------------------
    // Format validation (the contract's responsibility)
    // ---------------------------------------------------------------

    /**
     * Checks whether a loaded model's output structure and metadata
     * match this contract.
     *
     * @param model the loaded model (its session must be open)
     * @return a warning message if a mismatch is found, otherwise null
     */
    public String validate(MyOrtSession model) throws OrtException {
        if (family == Family.RF_DETR) {
            // Strict contract-metadata validation; never shape guessing.
            return RfDetrInference.validate(model);
        }

        // A bundled manifest (written by the Python export scripts) is
        // stronger evidence than shape heuristics: its task must agree
        // with the selected format.
        ModelManifest manifest = model.getManifest();
        if (manifest != null && manifest.task() != null && manifest.task() != task) {
            return "Warning: Bundled manifest task is '" + manifest.taskLabel()
                 + "', but " + label + " was selected.";
        }

        Map<String, NodeInfo> outMap = model.getOutputInfo();
        if (outMap.isEmpty()) return null;
        NodeInfo ni = outMap.values().iterator().next();
        if (!(ni.getInfo() instanceof TensorInfo)) return null;
        long[] shape = ((TensorInfo) ni.getInfo()).getShape();

        // 1. Check dimensionality
        if (shape.length == 2) {
            // [batch, classes] -> Classification
            if (task != Task.CLASSIFICATION) {
                return "Warning: Model output is 2D, but " + label + " was selected.\n"
                     + "This model is likely a Classification model.";
            }
        } else if (shape.length == 3) {
            // [batch, ...] -> Detection, Pose or Segmentation
            if (task == Task.CLASSIFICATION) {
                return "Warning: Model output is 3D, but Classification was selected.\n"
                     + "This model is likely a Detection, Pose or Segmentation model.";
            }
        }

        // 2. Check metadata 'task' (Ultralytics specific)
        Map<String, String> custom = model.getMetadata().getCustomMetadata();
        String metaTask = custom.get("task");
        if (metaTask != null && !metaTask.isEmpty()) {
            metaTask = metaTask.toLowerCase();
            if (metaTask.contains("detect") && task != Task.DETECTION) {
                return "Warning: Model metadata task is '" + metaTask + "', but " + label + " was selected.";
            }
            if (metaTask.contains("pose") && task != Task.POSE) {
                return "Warning: Model metadata task is '" + metaTask + "', but " + label + " was selected.";
            }
            if (metaTask.contains("class") && task != Task.CLASSIFICATION) {
                return "Warning: Model metadata task is '" + metaTask + "', but " + label + " was selected.";
            }
            if (metaTask.contains("segment") && task != Task.SEGMENTATION) {
                return "Warning: Model metadata task is '" + metaTask + "', but " + label + " was selected.";
            }
            if (metaTask.contains("obb") && task != Task.OBB) {
                return "Warning: Model metadata task is '" + metaTask + "', but " + label + " was selected.";
            }
        }

        // 3. Check outputs for Segmentation
        if (task == Task.SEGMENTATION) {
            if (outMap.size() < 2) {
                return "Warning: Model output count is " + outMap.size() + ", but SEGMENTATION was selected.\n"
                     + "YOLO Segmentation models typically have at least 2 outputs.";
            }
        }

        return null;
    }

    // ---------------------------------------------------------------
    // Class-count estimation (the contract's responsibility)
    // ---------------------------------------------------------------

    /**
     * Estimates the class count from the model's output tensor shapes,
     * using the rule of this contract. The returned basis string states
     * the rule and the shape it was applied to, so callers can log why
     * the count is what it is. (A bundled manifest, when present, takes
     * precedence over this estimation; see {@code MyOrtSession.load}.)
     */
    public NumClasses resolveNumClasses(MyOrtSession model) throws OrtException {
        Map<String, NodeInfo> outMap = model.getOutputInfo();
        if (outMap.isEmpty()) return new NumClasses(0, "model has no outputs");
        NodeInfo ni = outMap.values().iterator().next();
        if (!(ni.getInfo() instanceof TensorInfo)) return new NumClasses(0, "first output is not a tensor");
        long[] shape = ((TensorInfo) ni.getInfo()).getShape();
        String shapeText = java.util.Arrays.toString(shape);

        int n;
        String rule;
        switch (this) {
            case RF_DETR_OBJECT, RF_DETR_SEGMENT, RF_DETR_POSE -> {
                // Class count is the last dimension of the "labels" output.
                n = 0;
                rule = "RF-DETR labels output dimension";
                NodeInfo labels = outMap.get("labels");
                if (labels != null && labels.getInfo() instanceof TensorInfo) {
                    long[] dims = ((TensorInfo) labels.getInfo()).getShape();
                    if (dims.length == 3 && dims[2] > 0) n = (int) dims[2];
                }
            }
            case YOLO_OBJECT_E2E, YOLO_SEGMENT_E2E, YOLO_OBB_E2E -> {
                // E2E rows carry their class id; the count is not in the
                // shape and is resolved from metadata class names.
                n = 0;
                rule = "E2E layout " + shapeText + ": count not derivable from shape, resolved from metadata class names";
            }
            case YOLOX_OBJECT_UNDECODED -> {
                // [1, numAnchors, 5+numClasses]
                n = shape.length >= 3 ? (int) (shape[2] - 5) : 0;
                rule = "YOLOX output " + shapeText + ": last dim - 5";
            }
            case YOLO_CLASS -> {
                // [1, numClasses]
                n = shape.length >= 2 ? (int) shape[1] : 0;
                rule = "classification output " + shapeText + ": dim 1";
            }
            case YOLO_POSE, YOLO_POSE_E2E -> {
                // Person only
                n = 1;
                rule = "pose models have a single person class";
            }
            case YOLO_SEGMENT -> {
                // [1, 4+numClasses+32, numAnchors] or transposed
                n = shape.length >= 3 ? (int) ((Math.min(shape[1], shape[2])) - 4 - 32) : 0;
                rule = "segmentation output " + shapeText + ": min(dim1, dim2) - 4 - 32 mask coefficients";
            }
            case YOLO_OBB -> {
                // [1, 5+numClasses, numAnchors] or transposed (cx,cy,w,h, classes..., angle)
                n = shape.length >= 3 ? (int) ((Math.min(shape[1], shape[2])) - 5) : 0;
                rule = "OBB output " + shapeText + ": min(dim1, dim2) - 5";
            }
            default -> {
                // YOLO_OBJECT_PIXEL / YOLO_OBJECT_NORMALIZED:
                // [1, 4+numClasses, numAnchors] or transposed
                n = shape.length >= 3 ? (int) ((Math.min(shape[1], shape[2])) - 4) : 0;
                rule = "detection output " + shapeText + ": min(dim1, dim2) - 4";
            }
        }
        if (n < 0) n = 0;
        return new NumClasses(n, rule);
    }

    // ---------------------------------------------------------------
    // Lookup
    // ---------------------------------------------------------------

    /** All labels in declaration (= historical index) order, for dialog choices. */
    public static String[] labels() {
        ModelContract[] values = values();
        String[] labels = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            labels[i] = values[i].label;
        }
        return labels;
    }

    /**
     * Returns the contract for a dialog choice index.
     * Out-of-range indices fall back to the first contract, matching the
     * {@code default} branch of the historical resolve* switch statements.
     */
    public static ModelContract fromIndex(int index) {
        ModelContract[] values = values();
        if (index < 0 || index >= values.length) {
            return values[0];
        }
        return values[index];
    }

    /** Returns the contract with the given dialog label, or null if there is none. */
    public static ModelContract fromLabel(String label) {
        for (ModelContract c : values()) {
            if (c.label.equals(label)) {
                return c;
            }
        }
        return null;
    }
}
