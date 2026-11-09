// ORT_1st_ReadAndRelease.java - Manages ONNX model loading and releasing across slots
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import ai.onnxruntime.*;
import java.awt.AWTEvent;
import java.awt.Choice;
import java.awt.TextField;
import java.io.File;
import java.util.Map;
import java.util.Vector;

import static ij.plugin.filter.PlugInFilter.DONE;
import static ij.plugin.filter.PlugInFilter.NO_IMAGE_REQUIRED;

/**
 * 1st step: Model management for ORT-based inference plugins.
 */
public class ORT_1st_ReadAndRelease implements ExtendedPlugInFilter, DialogListener {

    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------

    private static final String[] ACTION_LABELS = {
        "read_model",
        "release_model",
        "release_all"
    };

    // Format labels and indices are owned by ModelContract (declaration
    // order = the historical index order below). This array is only the
    // dialog-facing view of ModelContract.labels().
    //   0 YOLO_Object_Pixel, 1 YOLO_Object_Normalized, 2 YOLO_Object_E2E,
    //   3 YOLO_Class, 4 YOLO_Pose, 5 YOLO_Pose_E2E, 6 YOLO_Segment,
    //   7 YOLO_Segment_E2E, 8 YOLO_OBB, 9 YOLO_OBB_E2E,
    //   10 YOLOX_Object_Undecoded, 11 RF_DETR_Object, 12 RF_DETR_Segment,
    //   13 RF_DETR_Pose
    private static final String[] FORMAT_LABELS = ModelContract.labels();

    // ---------------------------------------------------------------
    // Persistent dialog state (static)
    // ---------------------------------------------------------------

    private static int     actionChoice = 0;
    private static String  modelPath    = "";
    private static int     formatChoice = 0;
    private static int     slotChoice   = 0;
    private static boolean enableLog    = true;

    // ---------------------------------------------------------------
    // ExtendedPlugInFilter
    // ---------------------------------------------------------------

    @Override
    public int setup(String arg, ImagePlus imp) {
        return NO_IMAGE_REQUIRED;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        GenericDialog gd = new GenericDialog("1st Read and Release v" + OrtUtil.VERSION);

        gd.addChoice("action",       ACTION_LABELS,           ACTION_LABELS[actionChoice]);
        gd.addFileField("model_path", modelPath, 40);
        gd.addChoice("model_format", FORMAT_LABELS,           FORMAT_LABELS[formatChoice]);
        gd.addChoice("slot",         OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[slotChoice]);
        gd.addCheckbox("enable_log", enableLog);
        gd.addMessage(OrtUtil.getSlotStatusText());
        gd.addMessage("Note: Paths with quotes will be automatically trimmed.");

        gd.addDialogListener(this);

        gd.showDialog();
        if (gd.wasCanceled()) return DONE;

        actionChoice = gd.getNextChoiceIndex();
        modelPath = OrtUtil.trimQuotes(gd.getNextString());
        formatChoice = gd.getNextChoiceIndex();
        slotChoice = gd.getNextChoiceIndex();
        enableLog = gd.getNextBoolean();

        // Final validation before run()
        if (actionChoice == 0) { // read_model
            if (OrtUtil.isNullOrEmpty(modelPath) || !new File(modelPath).exists()) {
                OrtUtil.logError(this.getClass().getSimpleName(), "Model file not found " + modelPath);
                return DONE;
            }
        } else if (actionChoice == 1) { // release_model
            MyOrtSession s = OrtUtil.getSlot(slotChoice);
            if (s == null || !s.isLoaded()) {
                OrtUtil.logError(this.getClass().getSimpleName(), "Slot " + slotChoice + " is empty.");
                return DONE;
            }
        }

        return NO_IMAGE_REQUIRED;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        actionChoice = gd.getNextChoiceIndex();
        modelPath    = OrtUtil.trimQuotes(gd.getNextString());
        formatChoice = gd.getNextChoiceIndex();
        slotChoice   = gd.getNextChoiceIndex();
        enableLog    = gd.getNextBoolean();

        // Enable/Disable fields based on action
        Vector<?> choices = gd.getChoices();
        Vector<?> strFields = gd.getStringFields();
        
        boolean isRead    = (actionChoice == 0);
        boolean isRelease = (actionChoice == 1);

        if (strFields != null && !strFields.isEmpty()) ((TextField)strFields.get(0)).setEnabled(isRead);
        if (choices != null && choices.size() >= 2) ((Choice)choices.get(1)).setEnabled(isRead);
        if (choices != null && choices.size() >= 3) ((Choice)choices.get(2)).setEnabled(isRead || isRelease);

        return true;
    }

    @Override
    public void setNPasses(int nPasses) {}

    @Override
    public void run(ImageProcessor ip) {
        switch (actionChoice) {
            case 0: readModel();    break;
            case 1: releaseModel(); break;
            case 2: releaseAll();   break;
        }
    }

    // ---------------------------------------------------------------
    // Actions
    // ---------------------------------------------------------------

    private void readModel() {
        try {
            MyOrtSession existing = OrtUtil.getSlot(slotChoice);
            if (existing != null && existing.isLoaded()) {
                existing.release();
            }

            MyOrtSession.ModelType   type  = resolveModelType(formatChoice);
            MyOrtSession.CoordFormat coord = resolveCoordFormat(formatChoice);

            MyOrtSession s = new MyOrtSession();
            s.load(modelPath, type, coord);

            // Handle dynamic input shape
            if (s.getInputWidth() <= 0 || s.getInputHeight() <= 0) {
                if (IJ.isMacro()) {
                    s.setInputWidth(640);
                    s.setInputHeight(640);
                } else {
                    IJ.showStatus("Dynamic shape detected. Please enter input size.");
                    GenericDialog gd2 = new GenericDialog("Input Size");
                    gd2.addNumericField("input_width",  640, 0);
                    gd2.addNumericField("input_height", 640, 0);
                    gd2.showDialog();
                    if (gd2.wasCanceled()) return;
                    s.setInputWidth( (int) gd2.getNextNumber());
                    s.setInputHeight((int) gd2.getNextNumber());
                }
                if (s.getInputWidth() <= 0 || s.getInputHeight() <= 0) {
                    OrtUtil.logError(this.getClass().getSimpleName(), "Input dimensions must be positive.");
                    return;
                }
            }

            s.loadClassNamesFromMetadata();

            // Validate if the selected format matches the model's structure/metadata.
            // A mismatch is always a load failure, regardless of model family
            // (YOLO / RF-DETR) or macro execution: continuing would let the 2nd
            // plugins interpret the output tensor with the wrong format.
            String warning = s.validateFormat();
            if (warning != null) {
                OrtUtil.logError(this.getClass().getSimpleName(), "Model Type Mismatch - " + warning);
                s.release();
                return; // Stop loading if there's a mismatch
            }

            OrtUtil.setSlot(slotChoice, s);

            if (enableLog) logModelLoaded(s);
            IJ.showStatus("Slot " + slotChoice + ": " + s.getModelName() + " loaded.");

        } catch (Throwable t) {
            t.printStackTrace();
            OrtUtil.logError(this.getClass().getSimpleName(), "Failed to load model " + t.toString());
        }
    }

    private void releaseModel() {
        MyOrtSession s = OrtUtil.getSlot(slotChoice);
        if (s != null) {
            String name = s.getModelName();
            s.release();
            OrtUtil.setSlot(slotChoice, null);
            if (enableLog) IJ.log("Slot " + slotChoice + ": released. (" + name + ")");
            IJ.showStatus("Slot " + slotChoice + ": released.");
        }
    }

    private void releaseAll() {
        for (int i = 0; i < OrtUtil.MAX_SLOTS; i++) {
            MyOrtSession s = OrtUtil.getSlot(i);
            if (s != null && s.isLoaded()) {
                s.release();
                OrtUtil.setSlot(i, null);
            }
        }
        if (enableLog) IJ.log("All slots released.");
        IJ.showStatus("All slots released.");
    }

    private void logModelLoaded(MyOrtSession s) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("=".repeat(60)).append("\n");
            sb.append("Model Load Complete:\n");
            sb.append("  Slot:       ").append(slotChoice).append("\n");
            sb.append("  File:       ").append(s.getModelName()).append("\n");
            sb.append("  Format:     ").append(FORMAT_LABELS[formatChoice]).append("\n");
            sb.append("  Input Size: ").append(s.getInputWidth())
              .append(" x ").append(s.getInputHeight()).append("\n");
            sb.append("  Classes:    ").append(s.getNumClasses()).append("\n");

            sb.append("\n[Inputs]\n");
            for (Map.Entry<String, NodeInfo> e : s.getInputInfo().entrySet()) {
                appendTensorInfo(sb, e.getKey(), e.getValue());
            }
            sb.append("\n[Outputs]\n");
            for (Map.Entry<String, NodeInfo> e : s.getOutputInfo().entrySet()) {
                appendTensorInfo(sb, e.getKey(), e.getValue());
            }
            sb.append("=".repeat(60));
            IJ.log(sb.toString());
        } catch (Exception e) {
            IJ.log("(Logging failed: " + e.getMessage() + ")");
        }
    }

    private void appendTensorInfo(StringBuilder sb, String name, NodeInfo info) {
        sb.append("  Name:  ").append(name).append("\n");
        if (info.getInfo() instanceof TensorInfo) {
            TensorInfo ti = (TensorInfo) info.getInfo();
            long[] shape  = ti.getShape();
            sb.append("  Shape: [");
            for (int i = 0; i < shape.length; i++) {
                sb.append(shape[i] < 0 ? "?" : shape[i]);
                if (i < shape.length - 1) sb.append(", ");
            }
            sb.append("]\n");
            sb.append("  Type:  ").append(ti.type.toString()).append("\n");
        }
    }

    private MyOrtSession.ModelType resolveModelType(int choice) {
        return ModelContract.fromIndex(choice).modelType();
    }

    private MyOrtSession.CoordFormat resolveCoordFormat(int choice) {
        return ModelContract.fromIndex(choice).coordFormat();
    }
}
