// AbstractOrt2ndPlugin.java - Shared skeleton for the ORT "2nd" inference plugins
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.ImageProcessor;
import java.awt.AWTEvent;

/**
 * Template-method base class for the 2nd-step inference plugins
 * (Detection / Classify / Pose / Segmentation / OBB).
 *
 * <p>The base class owns the skeleton that used to be copy-pasted into
 * every subclass: {@code setup}, the dialog flow (slot choice first,
 * then the task-specific fields, then {@code enable_refresh_data} /
 * {@code enable_log}, then the slot status message), the common dialog
 * validation entry point, session lookup with the shared "no model
 * loaded" error, YOLO-family preprocessing helpers, inference timing
 * (unified on {@code System.nanoTime()}), and failure handling.
 *
 * <p>Macro compatibility contract: the dialog item names and the order
 * in which {@code getNext*()} values are read are exactly the historical
 * ones. Subclasses add their task fields between the slot choice and the
 * two common checkboxes, and read them back in the same relative order
 * (per value type) as before.
 *
 * <p>Dialog state stays in per-subclass {@code static} fields (as
 * before, so values persist between invocations); the base class only
 * accesses the three common values through the abstract accessors.
 */
public abstract class AbstractOrt2ndPlugin implements ExtendedPlugInFilter, DialogListener {

    /** The image the plugin was invoked on (set by {@link #setup}). */
    protected ImagePlus imp;

    // ---------------------------------------------------------------
    // Dialog state accessors (backed by each subclass's static fields)
    // ---------------------------------------------------------------

    protected abstract int getSlotChoice();

    protected abstract void setSlotChoice(int slotChoice);

    protected abstract boolean isEnableRefreshData();

    protected abstract void setEnableRefreshData(boolean enableRefreshData);

    protected abstract boolean isEnableLog();

    protected abstract void setEnableLog(boolean enableLog);

    // ---------------------------------------------------------------
    // Template hooks implemented by each task plugin
    // ---------------------------------------------------------------

    /** Dialog window title, e.g. "2nd Detection v0.9.5". */
    protected abstract String dialogTitle();

    /** Adds the task-specific dialog items (after "slot", before the common checkboxes). */
    protected abstract void addTaskFields(GenericDialog gd);

    /**
     * Reads back the task-specific values added by {@link #addTaskFields}.
     * Must read numbers/booleans in the same relative order as the
     * historical per-class code (the two common booleans are read by the
     * base class afterwards).
     */
    protected abstract void readTaskFields(GenericDialog gd);

    /** Task-specific range validation for dialogItemChanged (after the invalid-number check). */
    protected abstract boolean validateTaskDialog();

    // ---------------------------------------------------------------
    // ExtendedPlugInFilter skeleton
    // ---------------------------------------------------------------

    @Override
    public int setup(String arg, ImagePlus imp) {
        this.imp = imp;
        return DOES_ALL;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        GenericDialog gd = new GenericDialog(dialogTitle());
        gd.addChoice("slot", OrtUtil.buildSlotLabels(), OrtUtil.buildSlotLabels()[getSlotChoice()]);
        addTaskFields(gd);
        gd.addCheckbox("enable_refresh_data", isEnableRefreshData());
        gd.addCheckbox("enable_log", isEnableLog());
        gd.addMessage(OrtUtil.getSlotStatusText());

        gd.addDialogListener(this);
        gd.showDialog();

        if (gd.wasCanceled()) return DONE;

        // Parse values (important for macro support). Read order is part
        // of the macro contract and must not change.
        setSlotChoice(gd.getNextChoiceIndex());
        readTaskFields(gd);
        setEnableRefreshData(gd.getNextBoolean());
        setEnableLog(gd.getNextBoolean());

        return DOES_ALL;
    }

    @Override
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        setSlotChoice(gd.getNextChoiceIndex());
        readTaskFields(gd);
        setEnableRefreshData(gd.getNextBoolean());
        setEnableLog(gd.getNextBoolean());

        if (gd.invalidNumber()) {
            IJ.showStatus("Invalid number");
            return false;
        }
        return validateTaskDialog();
    }

    @Override
    public void setNPasses(int nPasses) {}

    // ---------------------------------------------------------------
    // Shared run() helpers
    // ---------------------------------------------------------------

    /** Logs an error prefixed with the concrete plugin class name (as the old per-class code did). */
    protected final void logError(String message) {
        OrtUtil.logError(this.getClass().getSimpleName(), message);
    }

    /**
     * Returns the session in the selected slot, or null after logging the
     * shared "No model loaded" error.
     */
    protected final MyOrtSession requireLoadedSession() {
        MyOrtSession s = OrtUtil.getSlot(getSlotChoice());
        if (s == null || !s.isLoaded()) {
            logError("No model loaded in slot " + getSlotChoice());
            return null;
        }
        return s;
    }

    /** Whether the session's model uses the YOLOX preprocessing variant (BGR, [0,255]). */
    protected static boolean isYolox(MyOrtSession s) {
        return s.getModelType() == MyOrtSession.ModelType.YOLOX;
    }

    /** The model contract of a loaded session (never null for sessions loaded via 1st). */
    protected static ModelContract contractOf(MyOrtSession s) {
        return ModelContract.of(s.getModelType(), s.getCoordFormat());
    }

    /**
     * Runs the session contract's preprocessing once and returns both
     * the input tensor and the coordinate transform decoders need.
     */
    protected static Preprocessor.Result prepareInput(MyOrtSession s, ImageProcessor ip) {
        return Preprocessor.apply(ip, contractOf(s).preprocess(),
                                  s.getInputWidth(), s.getInputHeight());
    }

    /** YOLO-family letterbox preprocessing with the session's input size. */
    protected static float[] preprocess(MyOrtSession s, ImageProcessor ip) {
        return prepareInput(s, ip).tensor();
    }

    /** Letterbox parameters [scale, padLeft, padTop] for the session's input size. */
    protected static float[] letterboxParams(MyOrtSession s, ImageProcessor ip) {
        Preprocessor.Transform t = Preprocessor.transformFor(
                contractOf(s).preprocess(), ip.getWidth(), ip.getHeight(),
                s.getInputWidth(), s.getInputHeight());
        return new float[]{t.scale(), t.padLeft(), t.padTop()};
    }

    /** Starts an inference timer. Timing is unified on nanoTime in all 2nd plugins. */
    protected static long startTimer() {
        return System.nanoTime();
    }

    /** Elapsed whole milliseconds since {@link #startTimer()}. */
    protected static long elapsedMillis(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    /** Shared failure handling for inference: stack trace plus a Log error line. */
    protected final void handleInferenceFailure(String message, Throwable t) {
        t.printStackTrace();
        logError(message + t.toString());
    }
}
