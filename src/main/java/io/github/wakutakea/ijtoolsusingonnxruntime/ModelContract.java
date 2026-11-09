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
 * <p>For now each contract only maps to the existing
 * {@link MyOrtSession.ModelType} / {@link MyOrtSession.CoordFormat} pair.
 * Callers still branch on those enums; richer contract data (preprocess
 * spec, decoder kind, NMS policy) is added in later phases.
 */
public enum ModelContract {

    YOLO_OBJECT_PIXEL    ("YOLO_Object_Pixel",       MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_PIXEL),
    YOLO_OBJECT_NORMALIZED("YOLO_Object_Normalized", MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_NORMALIZED),
    YOLO_OBJECT_E2E      ("YOLO_Object_E2E",         MyOrtSession.ModelType.YOLO,              MyOrtSession.CoordFormat.YOLO_OBJECT_E2E),
    YOLO_CLASS           ("YOLO_Class",              MyOrtSession.ModelType.CLASSIFICATION,    MyOrtSession.CoordFormat.YOLO_PIXEL),
    YOLO_POSE            ("YOLO_Pose",               MyOrtSession.ModelType.POSE,              MyOrtSession.CoordFormat.YOLO_POSE),
    YOLO_POSE_E2E        ("YOLO_Pose_E2E",           MyOrtSession.ModelType.POSE,              MyOrtSession.CoordFormat.YOLO_POSE_E2E),
    YOLO_SEGMENT         ("YOLO_Segment",            MyOrtSession.ModelType.SEGMENTATION,      MyOrtSession.CoordFormat.YOLO_SEGMENT),
    YOLO_SEGMENT_E2E     ("YOLO_Segment_E2E",        MyOrtSession.ModelType.SEGMENTATION,      MyOrtSession.CoordFormat.YOLO_SEGMENT_E2E),
    YOLO_OBB             ("YOLO_OBB",                MyOrtSession.ModelType.OBB,               MyOrtSession.CoordFormat.YOLO_OBB),
    YOLO_OBB_E2E         ("YOLO_OBB_E2E",            MyOrtSession.ModelType.OBB,               MyOrtSession.CoordFormat.YOLO_OBB_E2E),
    YOLOX_OBJECT_UNDECODED("YOLOX_Object_Undecoded", MyOrtSession.ModelType.YOLOX,             MyOrtSession.CoordFormat.YOLOX_UNDECODED),
    RF_DETR_OBJECT       ("RF_DETR_Object",          MyOrtSession.ModelType.RF_DETR_DETECTION, MyOrtSession.CoordFormat.RF_DETR),
    RF_DETR_SEGMENT      ("RF_DETR_Segment",         MyOrtSession.ModelType.RF_DETR_SEGMENTATION, MyOrtSession.CoordFormat.RF_DETR),
    RF_DETR_POSE         ("RF_DETR_Pose",            MyOrtSession.ModelType.RF_DETR_POSE,      MyOrtSession.CoordFormat.RF_DETR);

    private final String label;
    private final MyOrtSession.ModelType modelType;
    private final MyOrtSession.CoordFormat coordFormat;

    ModelContract(String label, MyOrtSession.ModelType modelType, MyOrtSession.CoordFormat coordFormat) {
        this.label = label;
        this.modelType = modelType;
        this.coordFormat = coordFormat;
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
