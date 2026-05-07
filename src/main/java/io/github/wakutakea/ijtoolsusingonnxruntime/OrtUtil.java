// MIT License
// OrtUtil.java - Common utilities for ORT-based ImageJ plugins

package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.WindowManager;
import ij.measure.ResultsTable;
import ij.plugin.frame.RoiManager;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import ij.process.Blitter;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.awt.Color;
import java.awt.Frame;
import java.util.ArrayList;
import java.util.List;

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

public class OrtUtil {

    public static final String VERSION   = "0.9.1";
    public static final int    MAX_SLOTS = 5;

    private static OrtEnvironment   env   = null;
    private static MyOrtSession[] slots   = new MyOrtSession[MAX_SLOTS];

    // ---------------------------------------------------------------
    // Environment / Slot management
    // ---------------------------------------------------------------

    public static OrtEnvironment getEnv() throws OrtException {
        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }
        return env;
    }

    public static MyOrtSession getSlot(int index) {
        if (index < 0 || index >= MAX_SLOTS)
            throw new IllegalArgumentException("Slot index out of range: " + index);
        return slots[index];
    }

    public static void setSlot(int index, MyOrtSession session) {
        if (index < 0 || index >= MAX_SLOTS)
            throw new IllegalArgumentException("Slot index out of range: " + index);
        slots[index] = session;
    }

    /** Returns a formatted string listing all slot states for GenericDialog.addMessage(). */
    public static String getSlotStatusText() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < MAX_SLOTS; i++) {
            MyOrtSession s = slots[i];
            if (s != null && s.isLoaded()) {
                sb.append(i).append(": ").append(s.getModelName());
            } else {
                sb.append(i).append(": (empty)");
            }
            if (i < MAX_SLOTS - 1) sb.append("\n");
        }
        return sb.toString();
    }

    /** Builds slot choice labels ["0", "1", ..., "MAX_SLOTS-1"]. */
    public static String[] buildSlotLabels() {
        String[] labels = new String[MAX_SLOTS];
        for (int i = 0; i < MAX_SLOTS; i++) labels[i] = String.valueOf(i);
        return labels;
    }

    // ---------------------------------------------------------------
    // Pre-processing: letterbox + ARGB -> NCHW float[]
    // ---------------------------------------------------------------

    /**
     * Converts an ImageProcessor to a float[] in NCHW format with letterbox padding.
     * Padding color: RGB(114, 114, 114).
     *
     * @param ip      source ImageProcessor (any type, internally converted to ColorProcessor)
     * @param targetW model input width
     * @param targetH model input height
     * @return float array of length 3 * targetH * targetW in NCHW order (R, G, B planes)
     */
    public static float[] preprocess(ImageProcessor ip, int targetW, int targetH, boolean isYolox) {
        // Ensure ColorProcessor
        ColorProcessor cp = (ip instanceof ColorProcessor)
                ? (ColorProcessor) ip
                : (ColorProcessor) ip.convertToRGB();

        int srcW = cp.getWidth();
        int srcH = cp.getHeight();

        // Compute letterbox scale and padding
        double scale   = Math.min((double) targetW / srcW, (double) targetH / srcH);
        int scaledW    = (int) Math.round(srcW * scale);
        int scaledH    = (int) Math.round(srcH * scale);
        int padLeft    = (targetW - scaledW) / 2;
        int padTop     = (targetH - scaledH) / 2;

        // Resize
        cp.setInterpolationMethod(ImageProcessor.BILINEAR);
        ImageProcessor resized = cp.resize(scaledW, scaledH);

        // Create padded canvas
        ColorProcessor padded = new ColorProcessor(targetW, targetH);
        padded.setColor(new Color(114, 114, 114));
        padded.fill();
        padded.copyBits(resized, padLeft, padTop, Blitter.COPY);

        // ARGB int[] -> NCHW float[]
        int[] pixels    = (int[]) padded.getPixels();
        int   planeSize = targetH * targetW;
        float[] nchw    = new float[3 * planeSize];
        float sum = 0, max = -1.0f, min = Float.MAX_VALUE;
        for (int i = 0; i < planeSize; i++) {
            int px = pixels[i];
            float r = ((px >> 16) & 0xFF);
            float g = ((px >>  8) & 0xFF);
            float b = ( px        & 0xFF);

            if (isYolox) {
                // YOLOX: BGR order, [0, 255] range
                nchw[              i] = b;
                nchw[planeSize   + i] = g;
                nchw[planeSize*2 + i] = r;
            } else {
                // YOLO: RGB order, [0.0, 1.0] range
                r /= 255.0f;
                g /= 255.0f;
                b /= 255.0f;
                nchw[              i] = r;
                nchw[planeSize   + i] = g;
                nchw[planeSize*2 + i] = b;
            }
            
            float pixelSum = r + g + b;
            sum += pixelSum;
            max = Math.max(max, Math.max(r, Math.max(g, b)));
            min = Math.min(min, Math.min(r, Math.min(g, b)));
        }
        System.out.println("DEBUG: Preprocess done. Shape: [1, 3, " + targetW + ", " + targetH + "], isYolox: " + isYolox);
        System.out.println("DEBUG: Input Range: [" + min + ", " + max + "], Mean: " + (sum / (3 * planeSize)));

        return nchw;
    }

    /**
     * Returns letterbox parameters: [scale, padLeft, padTop].
     * Used to restore bounding box coordinates to the original image space.
     */
    public static float[] calcLetterboxParams(int srcW, int srcH, int targetW, int targetH) {
        double scale = Math.min((double) targetW / srcW, (double) targetH / srcH);
        int scaledW  = (int) Math.round(srcW * scale);
        int scaledH  = (int) Math.round(srcH * scale);
        int padLeft  = (targetW - scaledW) / 2;
        int padTop   = (targetH - scaledH) / 2;
        return new float[]{(float) scale, padLeft, padTop};
    }

    /**
     * Restores a bounding box from letterboxed space to original image space.
     *
     * @param x       top-left x in letterboxed space (cx - w/2)
     * @param y       top-left y in letterboxed space (cy - h/2)
     * @param w       width in letterboxed space
     * @param h       height in letterboxed space
     * @param scale   scale factor from calcLetterboxParams
     * @param padLeft horizontal padding from calcLetterboxParams
     * @param padTop  vertical padding from calcLetterboxParams
     * @return float[] {x, y, w, h} in original image space
     */
    public static float[] restoreBbox(float x, float y, float w, float h,
                                      float scale, float padLeft, float padTop) {
        return new float[]{
            (x - padLeft) / scale,
            (y - padTop)  / scale,
            w / scale,
            h / scale
        };
    }

    /**
     * Restores a single keypoint coordinate from letterboxed space to original image space.
     */
    public static float[] restorePoint(float px, float py, float scale, float padLeft, float padTop) {
        return new float[]{(px - padLeft) / scale, (py - padTop) / scale};
    }

    // ---------------------------------------------------------------
    // NMS
    // ---------------------------------------------------------------

    /**
     * Custom NMS (Non-Maximum Suppression).
     * Each element of boxes is float[]{x, y, w, h, confidence, classId}.
     *
     * @param boxes        list of detection candidates
     * @param nmsThreshold IoU threshold for suppression
     * @return surviving detections sorted by confidence (descending)
     */
    public static List<float[]> nms(List<float[]> boxes, float nmsThreshold) {
        boxes.sort((a, b) -> Float.compare(b[4], a[4]));
        List<float[]>  result     = new ArrayList<>();
        boolean[]      suppressed = new boolean[boxes.size()];

        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            result.add(boxes.get(i));
            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                if (iou(boxes.get(i), boxes.get(j)) > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    private static float iou(float[] a, float[] b) {
        float ax2 = a[0] + a[2], ay2 = a[1] + a[3];
        float bx2 = b[0] + b[2], by2 = b[1] + b[3];
        float iw  = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
        float ih  = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
        float inter = iw * ih;
        float union = a[2] * a[3] + b[2] * b[3] - inter;
        return (union <= 0) ? 0 : inter / union;
    }

    // ---------------------------------------------------------------
    // Color for class
    // ---------------------------------------------------------------

    /** Generates a distinct color for a given class ID using the golden-ratio HSB method. */
    public static Color getColorForClass(int classId) {
        float hue = (classId * 0.618033988749895f) % 1.0f;
        return Color.getHSBColor(hue, 1.0f, 1.0f);
    }

    // ---------------------------------------------------------------
    // ResultsTable / RoiManager helpers
    // ---------------------------------------------------------------

    public static ResultsTable getResultsTable(boolean reset) {
        ResultsTable rt = ResultsTable.getResultsTable();
        if (rt == null) {
            rt = new ResultsTable();
        }
        ij.plugin.filter.Analyzer.setResultsTable(rt);
        if (reset) {
            rt.reset();
        }
        return rt;
    }

    public static RoiManager getRoiManager(boolean reset, boolean showNone) {
        RoiManager rm = RoiManager.getInstance();
        if (rm == null) {
            rm = new RoiManager();
        }
        rm.setVisible(true);
        if (reset) {
            rm.reset();
        }
        if (showNone) rm.runCommand("Show None");
        return rm;
    }

    // ---------------------------------------------------------------
    // String utilities
    // ---------------------------------------------------------------

    /**
     * Removes surrounding single or double quotes from a file path string.
     * Also trims leading/trailing whitespace.
     */
    public static String trimQuotes(String str) {
        if (str == null || str.isEmpty()) return str;
        String s = str.trim();
        if (s.length() >= 2 && s.startsWith("\"") && s.endsWith("\""))
            return s.substring(1, s.length() - 1);
        if (s.length() >= 2 && s.startsWith("'")  && s.endsWith("'"))
            return s.substring(1, s.length() - 1);
        return s;
    }

    public static boolean isNullOrEmpty(String s) {
        return s == null || s.isEmpty() || s.isBlank();
    }

    /**
     * Generates grid and stride arrays for YOLOX decoding.
     * @return int[][] { gridArray(x,y,x,y...), strideArray(s,s...) }
     */
    public static int[][] makeGridStride(int imgSize, int[] strides) {
        List<Integer> gridList   = new ArrayList<>();
        List<Integer> strideList = new ArrayList<>();

        for (int s : strides) {
            int gridH = imgSize / s;
            int gridW = imgSize / s;
            for (int y = 0; y < gridH; y++) {
                for (int x = 0; x < gridW; x++) {
                    gridList.add(x);
                    gridList.add(y);
                    strideList.add(s);
                }
            }
        }

        int[] grid   = new int[gridList.size()];
        int[] stride = new int[strideList.size()];
        for (int i = 0; i < gridList.size();   i++) grid[i]   = gridList.get(i);
        for (int i = 0; i < strideList.size(); i++) stride[i] = strideList.get(i);

        return new int[][]{grid, stride};
    }

    /**
     * Logs an error message to the ImageJ Log window.
     * Format: ClassName error: Message
     */
    public static void logError(String className, String msg) {
        ij.IJ.log(className + " error: " + msg);
    }
}
