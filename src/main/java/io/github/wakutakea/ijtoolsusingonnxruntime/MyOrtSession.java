// MyOrtSession.java - Holds one ONNX Runtime session and performs inference
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ai.onnxruntime.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.*;

/*
 * The MIT License
 *
 * Copyright 2026 WAKU-TAKE-A.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

public class MyOrtSession {

    // ---------------------------------------------------------------
    // Enums
    // ---------------------------------------------------------------

    public enum ModelType {
        YOLO,
        YOLOX,
        CLASSIFICATION,
        POSE,
        SEGMENTATION,
        OBB,
    }

    public enum CoordFormat {
        YOLO_PIXEL,
        YOLO_NORMALIZED,
        YOLO_OBJECT_E2E,
        YOLOX_UNDECODED,
        YOLO_POSE,
        YOLO_POSE_E2E,
        YOLO_SEGMENT,
        YOLO_SEGMENT_E2E,
        YOLO_OBB,
        YOLO_OBB_E2E,
    }

    // ---------------------------------------------------------------
    // Fields
    // ---------------------------------------------------------------

    private OrtSession  session;
    private String      modelPath;
    private String      modelName;
    private ModelType   modelType;
    private CoordFormat coordFormat;
    private int         inputWidth;
    private int         inputHeight;
    private String[]    classNames;
    private int         numClasses;
    private boolean     loaded;

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    public MyOrtSession() {
        loaded = false;
    }

    // ---------------------------------------------------------------
    // Load / Release
    // ---------------------------------------------------------------

    /**
     * Loads an ONNX model and initializes the session.
     * Input shape is auto-detected; -1 means dynamic (caller must set manually).
     *
     * @param modelPath absolute path to the .onnx file
     * @param type      model task type
     * @param coord     coordinate format for output parsing
     */
    public void load(String modelPath, ModelType type, CoordFormat coord) throws OrtException {
        release(); // release any existing session first

        OrtEnvironment env = OrtUtil.getEnv();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        this.session     = env.createSession(modelPath, opts);
        this.modelPath   = modelPath;
        this.modelName   = new File(modelPath).getName();
        this.modelType   = type;
        this.coordFormat = coord;

        detectInputShape();
        resolveNumClasses();

        this.loaded = true;
    }

    /**
     * Checks if the selected model format/type matches the actual output tensor and metadata.
     * @return a warning message if a mismatch is found, otherwise null.
     */
    public String validateFormat() throws OrtException {
        if (session == null) return null;
        
        Map<String, NodeInfo> outMap = session.getOutputInfo();
        if (outMap.isEmpty()) return null;
        NodeInfo ni = outMap.values().iterator().next();
        if (!(ni.getInfo() instanceof TensorInfo)) return null;
        long[] shape = ((TensorInfo) ni.getInfo()).getShape();

        // 1. Check dimensionality
        if (shape.length == 2) {
            // [batch, classes] -> Classification
            if (modelType != ModelType.CLASSIFICATION) {
                return "Warning: Model output is 2D, but " + modelType + " was selected.\n"
                     + "This model is likely a Classification model.";
            }
        } else if (shape.length == 3) {
            // [batch, ...] -> Detection, Pose or Segmentation
            if (modelType == ModelType.CLASSIFICATION) {
                return "Warning: Model output is 3D, but Classification was selected.\n"
                     + "This model is likely a Detection, Pose or Segmentation model.";
            }
        }

        // 2. Check metadata 'task' (Ultralytics specific)
        Map<String, String> custom = session.getMetadata().getCustomMetadata();
        String task = custom.get("task");
        if (task != null && !task.isEmpty()) {
            task = task.toLowerCase();
            if (task.contains("detect") && (modelType != ModelType.YOLO && modelType != ModelType.YOLOX)) {
                return "Warning: Model metadata task is '" + task + "', but " + modelType + " was selected.";
            }
            if (task.contains("pose") && modelType != ModelType.POSE) {
                return "Warning: Model metadata task is '" + task + "', but " + modelType + " was selected.";
            }
            if (task.contains("class") && modelType != ModelType.CLASSIFICATION) {
                return "Warning: Model metadata task is '" + task + "', but " + modelType + " was selected.";
            }
            if (task.contains("segment") && modelType != ModelType.SEGMENTATION) {
                return "Warning: Model metadata task is '" + task + "', but " + modelType + " was selected.";
            }
            if (task.contains("obb") && modelType != ModelType.OBB) {
                return "Warning: Model metadata task is '" + task + "', but " + modelType + " was selected.";
            }
        }

        // 3. Check outputs for Segmentation
        if (modelType == ModelType.SEGMENTATION) {
            if (outMap.size() < 2) {
                return "Warning: Model output count is " + outMap.size() + ", but SEGMENTATION was selected.\n"
                     + "YOLO Segmentation models typically have at least 2 outputs.";
            }
        }

        return null;
    }

    /** Detects input width/height from the session. Sets -1 if dynamic. */
    private void detectInputShape() throws OrtException {
        this.inputWidth  = -1;
        this.inputHeight = -1;
        Map<String, NodeInfo> infoMap = session.getInputInfo();
        if (infoMap.isEmpty()) return;
        NodeInfo ni = infoMap.values().iterator().next();
        if (!(ni.getInfo() instanceof TensorInfo)) return;
        long[] shape = ((TensorInfo) ni.getInfo()).getShape();
        // Expected: [batch, channels, H, W]
        if (shape.length >= 4) {
            this.inputHeight = shape[2] > 0 ? (int) shape[2] : -1;
            this.inputWidth  = shape[3] > 0 ? (int) shape[3] : -1;
        }
    }

    /** Infers numClasses from the output tensor shape heuristic. */
    private void resolveNumClasses() throws OrtException {
        this.numClasses = 0;
        Map<String, NodeInfo> outMap = session.getOutputInfo();
        if (outMap.isEmpty()) return;
        NodeInfo ni = outMap.values().iterator().next();
        if (!(ni.getInfo() instanceof TensorInfo)) return;
        long[] shape = ((TensorInfo) ni.getInfo()).getShape();

        switch (modelType) {
            case YOLO:
                if (coordFormat == CoordFormat.YOLO_OBJECT_E2E) {
                    // E2E: [1, N, 6] - numClasses not derivable from shape;
                    // will be resolved from metadata class names
                    numClasses = 0;
                } else if (shape.length >= 3) {
                    // [1, 4+numClasses, numAnchors] or [1, numAnchors, 4+numClasses]
                    long d1 = shape[1], d2 = shape[2];
                    numClasses = (int) ((d1 < d2 ? d1 : d2) - 4);
                }
                break;
            case YOLOX:
                // [1, numAnchors, 5+numClasses]
                if (shape.length >= 3) numClasses = (int) (shape[2] - 5);
                break;
            case CLASSIFICATION:
                // [1, numClasses]
                if (shape.length >= 2) numClasses = (int) shape[1];
                break;
            case POSE:
                // Person only
                numClasses = 1;
                break;
            case SEGMENTATION:
                if (coordFormat == CoordFormat.YOLO_SEGMENT_E2E) {
                    // E2E: [1, N, 38] = x1,y1,x2,y2,score,classId,32coeffs
                    // numClasses not derivable from shape; resolved from metadata class names
                    numClasses = 0;
                } else if (shape.length >= 3) {
                    // [1, 4+numClasses+32, numAnchors] or [1, numAnchors, 4+numClasses+32]
                    long d1 = shape[1], d2 = shape[2];
                    numClasses = (int) ((d1 < d2 ? d1 : d2) - 4 - 32);
                }
                break;
            case OBB:
                if (coordFormat == CoordFormat.YOLO_OBB_E2E) {
                    // E2E: [1, N, 7] = x1,y1,x2,y2,score,classId,angle
                    // numClasses not derivable from shape; resolved from metadata class names
                    numClasses = 0;
                } else if (shape.length >= 3) {
                    // [1, 5+numClasses, numAnchors] or [1, numAnchors, 5+numClasses]
                    // Layout: cx,cy,w,h, class0..classN-1, angle  -> 4+N+1 = 5+N
                    long d1 = shape[1], d2 = shape[2];
                    numClasses = (int) ((d1 < d2 ? d1 : d2) - 5);
                }
                break;
            default:
                numClasses = 0;
        }
        if (numClasses < 0) numClasses = 0;
    }

    /** Closes the ORT session and marks this slot as unloaded. */
    public void release() {
        if (session != null) {
            try { session.close(); } catch (Exception e) { /* ignore */ }
            session = null;
        }
        loaded = false;
    }

    // ---------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------

    /**
     * Runs inference on a pre-processed NCHW float array.
     * Caller is responsible for closing the returned Result.
     *
     * @param inputData float[] of length 3 * inputHeight * inputWidth in NCHW order
     * @return OrtSession.Result (must be closed by caller)
     */
    public OrtSession.Result runInference(float[] inputData) throws OrtException {
        OrtEnvironment env   = OrtUtil.getEnv();
        long[]   shape       = {1, 3, inputHeight, inputWidth};
        OnnxTensor tensor    = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
        String inputName     = session.getInputNames().iterator().next();
        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
        OrtSession.Result result = session.run(inputs);
        tensor.close();
        return result;
    }

    // ---------------------------------------------------------------
    // Session info access (for logging)
    // ---------------------------------------------------------------

    public Map<String, NodeInfo> getInputInfo()  throws OrtException { return session.getInputInfo(); }
    public Map<String, NodeInfo> getOutputInfo() throws OrtException { return session.getOutputInfo(); }
    public OnnxModelMetadata     getMetadata()   throws OrtException { return session.getMetadata(); }

    // ---------------------------------------------------------------
    // Class name resolution
    // ---------------------------------------------------------------

    /**
     * Loads class names from an external text file.
     * Format: one class name per line; lines starting with '#' are comments.
     * Encoding: UTF-8 first, then Shift-JIS on failure.
     */
    public void loadClassNamesFromFile(String filePath) {
        String[] encodings = {"UTF-8", "Shift_JIS"};
        for (String enc : encodings) {
            List<String> names = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filePath), enc))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty() && !line.startsWith("#")) names.add(line);
                }
                if (!names.isEmpty()) {
                    classNames = names.toArray(new String[0]);
                    return;
                }
            } catch (Exception ignored) {}
        }
    }

    /**
     * Attempts to load class names from model metadata (key: "names").
     * Falls back silently if not present.
     */
    public void loadClassNamesFromMetadata() {
        if (session == null) return;
        try {
            Map<String, String> custom = session.getMetadata().getCustomMetadata();
            String raw = custom.get("names");
            if (raw == null || raw.isEmpty()) return;
            // Format may be "{0: 'cat', 1: 'dog', ...}" or "cat\ndog\n..."
            List<String> names = new ArrayList<>();
            if (raw.startsWith("{")) {
                // Parse dict-style: {0: 'cat', 1: 'dog'}
                raw = raw.replaceAll("[{}]", "");
                String[] entries = raw.split(",");
                Arrays.sort(entries, (a, b) -> {
                    try {
                        int ia = Integer.parseInt(a.split(":")[0].trim());
                        int ib = Integer.parseInt(b.split(":")[0].trim());
                        return Integer.compare(ia, ib);
                    } catch (Exception e) { return 0; }
                });
                for (String entry : entries) {
                    String[] kv = entry.split(":", 2);
                    if (kv.length == 2) {
                        names.add(kv[1].trim().replaceAll("['\"]", ""));
                    }
                }
            } else {
                // Plain newline-separated
                for (String line : raw.split("\n")) {
                    String t = line.trim();
                    if (!t.isEmpty()) names.add(t);
                }
            }
            if (!names.isEmpty()) classNames = names.toArray(new String[0]);
        } catch (Exception ignored) {}
    }

    /**
     * Returns the class name for a given index.
     * Falls back to "class_N" if names are not loaded or index is out of range.
     */
    public String resolveClassName(int idx) {
        if (classNames != null && idx >= 0 && idx < classNames.length) {
            return classNames[idx];
        }
        return "class_" + idx;
    }

    // ---------------------------------------------------------------
    // Getters / Setters
    // ---------------------------------------------------------------

    public boolean     isLoaded()          { return loaded; }
    public String      getModelPath()      { return modelPath; }
    public String      getModelName()      { return modelName; }
    public ModelType   getModelType()      { return modelType; }
    public CoordFormat getCoordFormat()    { return coordFormat; }
    public int         getInputWidth()     { return inputWidth; }
    public int         getInputHeight()    { return inputHeight; }
    public String[]    getClassNames()     { return classNames; }
    public int         getNumClasses()     { return numClasses; }

    public void setInputWidth(int w)               { this.inputWidth  = w; }
    public void setInputHeight(int h)              { this.inputHeight = h; }
    public void setClassNames(String[] names)      { this.classNames  = names; }
}
