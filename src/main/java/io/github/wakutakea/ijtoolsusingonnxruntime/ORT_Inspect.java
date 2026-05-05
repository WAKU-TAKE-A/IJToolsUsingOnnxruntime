// ORT_Inspect.java - Inspects ONNX model metadata
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
import java.io.File;
import java.util.Map;

/**
 * Utility plugin to inspect ONNX model structure and metadata.
 */
public class ORT_Inspect implements ExtendedPlugInFilter, DialogListener {

    private static String  modelPath    = "";
    private static boolean showMetadata = true;

    @Override
    public int setup(String arg, ImagePlus imp) {
        return NO_IMAGE_REQUIRED;
    }

    @Override
    public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
        GenericDialog gd = new GenericDialog("Inspect ONNX Model");
        gd.addFileField("model_path", modelPath, 40);
        gd.addCheckbox("show_metadata", showMetadata);
        gd.addMessage("Note: Paths with quotes will be automatically trimmed.");
        
        gd.addDialogListener(this);
        gd.showDialog();
        
        if (gd.wasCanceled()) return DONE;

        if (OrtUtil.isNullOrEmpty(modelPath) || !new File(modelPath).exists()) {
            IJ.error("Model file not found: " + modelPath);
            return DONE;
        }

        return NO_IMAGE_REQUIRED;
    }

    @Override
    @SuppressWarnings("rawtypes")
    public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
        modelPath    = OrtUtil.trimQuotes(gd.getNextString());
        showMetadata = gd.getNextBoolean();
        return true;
    }

    @Override
    public void setNPasses(int nPasses) {}

    @Override
    public void run(ImageProcessor ip) {
        try (OrtEnvironment env = OrtUtil.getEnv();
             OrtSession s = env.createSession(modelPath)) {

            StringBuilder sb = new StringBuilder();
            sb.append("=".repeat(60)).append("\n");
            sb.append("ONNX Model Inspection:\n");
            sb.append("  File: ").append(new File(modelPath).getName()).append("\n");

            if (showMetadata) {
                sb.append("\n[Metadata]\n");
                OnnxModelMetadata meta = s.getMetadata();
                sb.append("  Producer: ").append(meta.getProducerName()).append("\n");
                sb.append("  Version:  ").append(meta.getVersion()).append("\n");
                sb.append("  Graph:    ").append(meta.getGraphName()).append("\n");
                for (Map.Entry<String, String> entry : meta.getCustomMetadata().entrySet()) {
                    sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
                }
            }

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
            IJ.error("Inspection failed: " + e.getMessage());
        }
    }

    private void appendTensorInfo(StringBuilder sb, String name, NodeInfo info) {
        sb.append("  Name:  ").append(name).append("\n");
        if (info.getInfo() instanceof TensorInfo) {
            TensorInfo ti = (TensorInfo) info.getInfo();
            long[] shape = ti.getShape();
            sb.append("  Shape: [");
            for (int i = 0; i < shape.length; i++) {
                sb.append(shape[i] < 0 ? "?" : shape[i]);
                if (i < shape.length - 1) sb.append(", ");
            }
            sb.append("]\n");
            sb.append("  Type:  ").append(ti.type.toString()).append("\n");
        }
    }
}
