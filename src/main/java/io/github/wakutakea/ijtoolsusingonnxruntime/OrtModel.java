// OrtModel.java - Holds one ONNX Runtime session and performs inference
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ai.onnxruntime.*;
import ij.IJ;
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

public class OrtModel {

    // ---------------------------------------------------------------
    // Fields
    // ---------------------------------------------------------------

    private OrtSession    session;
    private String        modelPath;
    private String        modelName;
    private ModelContract contract;
    private ModelManifest manifest;
    private int           inputWidth;
    private int           inputHeight;
    private String[]      classNames;
    private int           numClasses;
    private boolean       loaded;

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    public OrtModel() {
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
     * @param contract  the model format contract the user selected
     */
    public void load(String modelPath, ModelContract contract) throws OrtException {
        release(); // release any existing session first

        OrtEnvironment env = OrtUtil.getEnv();
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            this.session = env.createSession(modelPath, opts);
        }
        this.modelPath = modelPath;
        this.modelName = new File(modelPath).getName();
        this.contract  = contract;

        detectInputShape();

        // A bundled manifest (YOLO exports) outranks shape heuristics
        // for the class count. RF-DETR contracts never consult one:
        // their contract stays the strict ONNX-metadata validation.
        this.manifest = contract.family() == ModelContract.Family.YOLO
                ? ModelManifest.findFor(modelPath) : null;
        resolveNumClasses();

        this.loaded = true;
    }

    /**
     * Checks if the selected format matches the actual output tensor
     * and metadata. The check itself is the contract's responsibility
     * ({@link ModelContract#validate}); see there for the rules.
     *
     * @return a warning message if a mismatch is found, otherwise null.
     */
    public String validate() throws OrtException {
        if (session == null) return null;
        return contract.validate(this);
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

    /**
     * Resolves the class count: the bundled manifest when it records
     * one, otherwise the contract's shape-based estimation. Either way
     * the basis is logged, so a wrong count can be traced to its source.
     */
    private void resolveNumClasses() throws OrtException {
        if (manifest != null && manifest.numClasses() > 0) {
            this.numClasses = manifest.numClasses();
            IJ.log("OrtModel: numClasses=" + numClasses + " for " + modelName
                    + " (bundled manifest " + manifest.path().getFileName() + ")");
            return;
        }
        ModelContract.NumClasses estimated = contract.resolveNumClasses(this);
        this.numClasses = estimated.value();
        IJ.log("OrtModel: numClasses=" + numClasses + " for " + modelName
                + " (estimated: " + estimated.basis() + ")");
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
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            String inputName = session.getInputNames().iterator().next();
            return session.run(Collections.singletonMap(inputName, tensor));
        }
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

    public boolean       isLoaded()        { return loaded; }
    public boolean isRfDetr() {
        return contract != null && contract.family() == ModelContract.Family.RF_DETR;
    }
    public String        getModelPath()    { return modelPath; }
    public String        getModelName()    { return modelName; }
    public ModelContract getContract()     { return contract; }
    public ModelManifest getManifest()     { return manifest; }
    public int           getInputWidth()   { return inputWidth; }
    public int           getInputHeight()  { return inputHeight; }
    public String[]      getClassNames()   { return classNames; }
    public int           getNumClasses()   { return numClasses; }

    public void setInputWidth(int w)               { this.inputWidth  = w; }
    public void setInputHeight(int h)              { this.inputHeight = h; }
    public void setClassNames(String[] names)      { this.classNames  = names; }
}
