// ModelContract.java - Single source of truth for model format labels and indices
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * The list of selectable model formats in "1st Read and Release".
 *
 * <p>This enum is the source of truth for the format labels and their
 * indices. The declaration order must exactly match the historical
 * {@code FORMAT_LABELS} order in {@code ORT_1st_ReadAndRelease}: macros
 * may record the format by index or by label string, so neither the
 * order nor the label strings may change.
 *
 * <p>Each contract also declares the task, the preprocessing spec and
 * the NMS policy of the format (Phase 2), so those decisions live in
 * exactly one place. The legacy {@link MyOrtSession.ModelType} /
 * {@link MyOrtSession.CoordFormat} pair is still carried for the
 * callers that branch on it; removing it is Phase 3 work.
 */
public enum ModelContract {

    YOLO_OBJECT_PIXEL    ("YOLO_Object_Pixel",       MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_PIXEL,
                          Task.DETECTION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.PER_CLASS),
    YOLO_OBJECT_NORMALIZED("YOLO_Object_Normalized", MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_NORMALIZED,
                          Task.DETECTION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.PER_CLASS),
    YOLO_OBJECT_E2E      ("YOLO_Object_E2E",         MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_OBJECT_E2E,
                          Task.DETECTION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.NONE_E2E),
    YOLO_CLASS           ("YOLO_Class",              MyOrtSession.ModelType.CLASSIFICATION,    MyOrtSession.CoordFormat.YOLO_PIXEL,
                          Task.CLASSIFICATION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.NONE),
    YOLO_POSE            ("YOLO_Pose",               MyOrtSession.ModelType.POSE,              MyOrtSession.CoordFormat.YOLO_POSE,
                          Task.POSE, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.CLASS_AGNOSTIC),
    YOLO_POSE_E2E        ("YOLO_Pose_E2E",           MyOrtSession.ModelType.POSE,              MyOrtSession.CoordFormat.YOLO_POSE_E2E,
                          Task.POSE, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.NONE_E2E),
    YOLO_SEGMENT         ("YOLO_Segment",            MyOrtSession.ModelType.SEGMENTATION,      MyOrtSession.CoordFormat.YOLO_SEGMENT,
                          Task.SEGMENTATION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.PER_CLASS),
    YOLO_SEGMENT_E2E     ("YOLO_Segment_E2E",        MyOrtSession.ModelType.SEGMENTATION,      MyOrtSession.CoordFormat.YOLO_SEGMENT_E2E,
                          Task.SEGMENTATION, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.NONE_E2E),
    YOLO_OBB             ("YOLO_OBB",                MyOrtSession.ModelType.OBB,               MyOrtSession.CoordFormat.YOLO_OBB,
                          Task.OBB, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.PER_CLASS),
    YOLO_OBB_E2E         ("YOLO_OBB_E2E",            MyOrtSession.ModelType.OBB,               MyOrtSession.CoordFormat.YOLO_OBB_E2E,
                          Task.OBB, PreprocessSpec.YOLO_LETTERBOX, NmsPolicy.NONE_E2E),
    YOLOX_OBJECT_UNDECODED("YOLOX_Object_Undecoded", MyOrtSession.ModelType.YOLOX,             MyOrtSession.CoordFormat.YOLOX_UNDECODED,
                          Task.DETECTION, PreprocessSpec.YOLOX_LETTERBOX, NmsPolicy.PER_CLASS),
    RF_DETR_OBJECT       ("RF_DETR_Object",          MyOrtSession.ModelType.RF_DETR_DETECTION, MyOrtSession.CoordFormat.RF_DETR,
                          Task.DETECTION, PreprocessSpec.RFDETR_STRETCH, NmsPolicy.OPTIONAL),
    RF_DETR_SEGMENT      ("RF_DETR_Segment",         MyOrtSession.ModelType.RF_DETR_SEGMENTATION, MyOrtSession.CoordFormat.RF_DETR,
                          Task.SEGMENTATION, PreprocessSpec.RFDETR_STRETCH, NmsPolicy.OPTIONAL),
    RF_DETR_POSE         ("RF_DETR_Pose",            MyOrtSession.ModelType.RF_DETR_POSE,      MyOrtSession.CoordFormat.RF_DETR,
                          Task.POSE, PreprocessSpec.RFDETR_STRETCH, NmsPolicy.OPTIONAL);

    /** The inference task a format belongs to. */
    public enum Task { DETECTION, CLASSIFICATION, POSE, SEGMENTATION, OBB }

    /** The model family a format belongs to. */
    public enum Family { YOLO, RF_DETR }

    private final String label;
    private final MyOrtSession.ModelType modelType;
    private final MyOrtSession.CoordFormat coordFormat;
    private final Task task;
    private final PreprocessSpec preprocess;
    private final NmsPolicy nmsPolicy;

    ModelContract(String label, MyOrtSession.ModelType modelType, MyOrtSession.CoordFormat coordFormat,
                  Task task, PreprocessSpec preprocess, NmsPolicy nmsPolicy) {
        this.label = label;
        this.modelType = modelType;
        this.coordFormat = coordFormat;
        this.task = task;
        this.preprocess = preprocess;
        this.nmsPolicy = nmsPolicy;
    }

    /** Dialog label, identical to the historical FORMAT_LABELS entry. */
    public String label() {
        return label;
    }

    public MyOrtSession.ModelType modelType() {
        return modelType;
    }

    public MyOrtSession.CoordFormat coordFormat() {
        return coordFormat;
    }

    public Task task() {
        return task;
    }

    /** YOLO for the YOLO/YOLOX formats, RF_DETR for the RF-DETR formats. */
    public Family family() {
        return switch (modelType) {
            case RF_DETR_DETECTION, RF_DETR_SEGMENTATION, RF_DETR_POSE -> Family.RF_DETR;
            default -> Family.YOLO;
        };
    }

    /** Preprocessing contract of this format. */
    public PreprocessSpec preprocess() {
        return preprocess;
    }

    /** NMS policy of this format, executed by {@link PostProcessors}. */
    public NmsPolicy nmsPolicy() {
        return nmsPolicy;
    }

    /**
     * Finds the contract for a loaded session's type/format pair. Every
     * (ModelType, CoordFormat) combination a session can hold maps to
     * exactly one contract; returns null for impossible combinations.
     */
    public static ModelContract of(MyOrtSession.ModelType modelType, MyOrtSession.CoordFormat coordFormat) {
        for (ModelContract c : values()) {
            if (c.modelType == modelType && c.coordFormat == coordFormat) {
                return c;
            }
        }
        return null;
    }

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
}
