// YoloDecoder.java - Decodes YOLO-family output tensors into unified Detections
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.ArrayList;
import java.util.List;

/**
 * YOLO-family decoders, moved out of the ORT_2nd_* plugins (Phase 2).
 * The parsing arithmetic is unchanged from the original per-plugin
 * {@code parse*} methods and run()-inline parsing; only the output
 * representation changed from layout-dependent {@code float[]} rows to
 * {@link Detection}. Segmentation candidates are restored to original
 * image coordinates here (unclipped; clipping remains a rendering
 * concern, as before), with the letterbox-space box kept in the
 * {@link MaskRef.ProtoMask} for prototype cropping.
 */
public final class YoloDecoder {
    private YoloDecoder() {}

    // ---------------------------------------------------------------
    // Detection (YOLO / YOLO E2E / YOLOX)
    // ---------------------------------------------------------------

    public static List<Detection> decodeDetection(float[][] data,
                                                  ModelContract contract, int numClasses,
                                                  int imgW, int imgH, float scoreTh,
                                                  Preprocessor.Transform tf, int inputW, int inputH) {
        List<Detection> out = new ArrayList<>();
        if (contract.isYolox()) {
            parseYOLOX(data, numClasses, imgW, imgH, scoreTh, tf, inputW, out);
        } else if (contract == ModelContract.YOLO_OBJECT_E2E) {
            parseYOLO_E2E(data, imgW, imgH, scoreTh, tf, out);
        } else {
            parseYOLO(data, numClasses, contract, imgW, imgH, scoreTh, tf, inputW, inputH, out);
        }
        return out;
    }

    private static void parseYOLO(float[][] data, int numClasses, ModelContract contract,
                                  int imgW, int imgH, float scoreTh,
                                  Preprocessor.Transform tf, int inputW, int inputH,
                                  List<Detection> out) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int d1 = data.length;
        int d2 = data[0].length;
        int C  = 4 + numClasses;

        boolean channelFirst = (d1 == C && d1 < d2);
        int numAnchors = channelFirst ? d2 : d1;

        for (int i = 0; i < numAnchors; i++) {
            float cx = channelFirst ? data[0][i] : data[i][0];
            float cy = channelFirst ? data[1][i] : data[i][1];
            float w  = channelFirst ? data[2][i] : data[i][2];
            float h  = channelFirst ? data[3][i] : data[i][3];

            float maxScore = -1;
            int   maxClass = 0;
            for (int c = 0; c < numClasses; c++) {
                float sc = channelFirst ? data[4 + c][i] : data[i][4 + c];
                if (sc > maxScore) { maxScore = sc; maxClass = c; }
            }
            if (maxScore < scoreTh) continue;

            float bx, by, bw, bh;
            if (contract == ModelContract.YOLO_OBJECT_NORMALIZED) {
                bw = w * inputW;
                bh = h * inputH;
                bx = cx * inputW - bw / 2;
                by = cy * inputH - bh / 2;
            } else {
                bw = w;
                bh = h;
                bx = cx - bw / 2;
                by = cy - bh / 2;
            }

            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new Detection(maxClass, maxScore, rx, ry, rw, rh));
        }
    }

    private static void parseYOLO_E2E(float[][] data, int imgW, int imgH, float scoreTh,
                                      Preprocessor.Transform tf, List<Detection> out) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int numDetections = data.length;
        for (int i = 0; i < numDetections; i++) {
            float x1      = data[i][0];
            float y1      = data[i][1];
            float x2      = data[i][2];
            float y2      = data[i][3];
            float score   = data[i][4];
            int   classId = (int) data[i][5];

            if (score < scoreTh) continue;

            float bw = x2 - x1;
            float bh = y2 - y1;
            float[] restored = OrtUtil.restoreBbox(x1, y1, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new Detection(classId, score, rx, ry, rw, rh));
        }
    }

    private static void parseYOLOX(float[][] data, int numClasses, int imgW, int imgH,
                                   float scoreTh, Preprocessor.Transform tf,
                                   int inputW, List<Detection> out) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int[] strides = {8, 16, 32};
        int[][] gridStride = OrtUtil.makeGridStride(inputW, strides);
        int[] grid   = gridStride[0];
        int[] stride = gridStride[1];

        int numAnchors = data.length;
        for (int i = 0; i < numAnchors; i++) {
            float tx  = data[i][0];
            float ty  = data[i][1];
            float tw  = data[i][2];
            float th  = data[i][3];
            float obj = data[i][4];

            float maxClsScore = -1;
            int   maxClass    = 0;
            for (int c = 0; c < numClasses; c++) {
                float sc = data[i][5 + c];
                if (sc > maxClsScore) { maxClsScore = sc; maxClass = c; }
            }

            // Confidence calculation consistent with previous OCV implementation
            float confidence = (float)(Math.sqrt(Math.max(0, obj)) * maxClsScore);
            if (confidence < scoreTh) continue;

            // Decode grid coordinates
            int gx = grid[i * 2];
            int gy = grid[i * 2 + 1];
            int s  = stride[i];

            double cx = (tx + gx) * s;
            double cy = (ty + gy) * s;
            double w  = Math.exp(tw) * s;
            double h  = Math.exp(th) * s;

            float bx = (float)(cx - w / 2);
            float by = (float)(cy - h / 2);
            float bw = (float)w;
            float bh = (float)h;

            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);
            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));
            if (rw <= 0 || rh <= 0) continue;

            out.add(new Detection(maxClass, confidence, rx, ry, rw, rh));
        }
    }

    // ---------------------------------------------------------------
    // Pose
    // ---------------------------------------------------------------

    public static List<Detection> decodePose(float[][] data, ModelContract contract,
                                             int imgW, int imgH, float scoreTh,
                                             Preprocessor.Transform tf) {
        List<Detection> out = new ArrayList<>();
        if (contract == ModelContract.YOLO_POSE_E2E) {
            parsePose_E2E(data, imgW, imgH, scoreTh, tf, out);
        } else {
            parsePose(data, imgW, imgH, scoreTh, tf, out);
        }
        return out;
    }

    private static void parsePose(float[][] data, int imgW, int imgH, float scoreTh,
                                  Preprocessor.Transform tf, List<Detection> out) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int d1 = data.length;
        int d2 = data[0].length;

        boolean channelFirst = (d1 > d2); // Simplified check for pose [C, N]
        int numAnchors = channelFirst ? d2 : d1;
        int numKpts = ( (channelFirst ? d1 : d2) - 5 ) / 3;

        for (int i = 0; i < numAnchors; i++) {
            float cx    = channelFirst ? data[0][i] : data[i][0];
            float cy    = channelFirst ? data[1][i] : data[i][1];
            float w     = channelFirst ? data[2][i] : data[i][2];
            float h     = channelFirst ? data[3][i] : data[i][3];
            float score = channelFirst ? data[4][i] : data[i][4];

            if (score < scoreTh) continue;

            float bw = w;
            float bh = h;
            float bx = cx - bw / 2;
            float by = cy - bh / 2;

            float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, scale, padLeft, padTop);

            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));

            float[][] keypoints = new float[numKpts][3];
            for (int k = 0; k < numKpts; k++) {
                float kx = channelFirst ? data[5 + k*3][i]   : data[i][5 + k*3];
                float ky = channelFirst ? data[5 + k*3 + 1][i] : data[i][5 + k*3 + 1];
                float ks = channelFirst ? data[5 + k*3 + 2][i] : data[i][5 + k*3 + 2];

                float[] rk = OrtUtil.restorePoint(kx, ky, scale, padLeft, padTop);
                keypoints[k][0] = rk[0];
                keypoints[k][1] = rk[1];
                keypoints[k][2] = ks;
            }
            out.add(new Detection(0, score, rx, ry, rw, rh, 0f, keypoints, null, -1));
        }
    }

    private static void parsePose_E2E(float[][] data, int imgW, int imgH, float scoreTh,
                                      Preprocessor.Transform tf, List<Detection> out) {
        float scale = tf.scale(), padLeft = tf.padLeft(), padTop = tf.padTop();
        int numDetections = data.length;
        int numKpts = (data[0].length - 6) / 3;

        for (int i = 0; i < numDetections; i++) {
            float x1      = data[i][0];
            float y1      = data[i][1];
            float x2      = data[i][2];
            float y2      = data[i][3];
            float score   = data[i][4];

            if (score < scoreTh) continue;

            float bw = x2 - x1;
            float bh = y2 - y1;
            float[] restored = OrtUtil.restoreBbox(x1, y1, bw, bh, scale, padLeft, padTop);

            float rx = Math.max(0, Math.min(restored[0], imgW));
            float ry = Math.max(0, Math.min(restored[1], imgH));
            float rw = Math.max(0, Math.min(restored[2], imgW - rx));
            float rh = Math.max(0, Math.min(restored[3], imgH - ry));

            float[][] keypoints = new float[numKpts][3];
            for (int k = 0; k < numKpts; k++) {
                float kx = data[i][6 + k*3];
                float ky = data[i][6 + k*3 + 1];
                float ks = data[i][6 + k*3 + 2];

                float[] rk = OrtUtil.restorePoint(kx, ky, scale, padLeft, padTop);
                keypoints[k][0] = rk[0];
                keypoints[k][1] = rk[1];
                keypoints[k][2] = ks;
            }
            out.add(new Detection(0, score, rx, ry, rw, rh, 0f, keypoints, null, -1));
        }
    }

    // ---------------------------------------------------------------
    // Segmentation (box/coefficient tensor only; prototypes stay with the caller)
    // ---------------------------------------------------------------

    public static List<Detection> decodeSegmentation(float[][] data0, int numClasses,
                                                     ModelContract contract, float scoreTh,
                                                     Preprocessor.Transform tf) {
        List<Detection> out = new ArrayList<>();
        int d1 = data0.length;
        int d2 = data0[0].length;

        if (contract == ModelContract.YOLO_SEGMENT_E2E) {
            // E2E format: [1, N, 38] = x1,y1,x2,y2 (letterbox px), score, classId, 32 coeffs
            int numDetections = d1;  // e.g. 300
            for (int i = 0; i < numDetections; i++) {
                float score   = data0[i][4];
                if (score < scoreTh) continue;

                float x1 = data0[i][0];
                float y1 = data0[i][1];
                float x2 = data0[i][2];
                float y2 = data0[i][3];
                int   classId = (int) data0[i][5];

                float[] coeffs = new float[32];
                for (int j = 0; j < 32; j++) coeffs[j] = data0[i][6 + j];
                out.add(restoredSegDetection(classId, score, x1, y1, x2 - x1, y2 - y1, coeffs, tf));
            }
        } else {
            // Standard format: [1, 4+numClasses+32, numAnchors] or [1, numAnchors, 4+numClasses+32]
            int C = 4 + numClasses + 32;
            boolean channelFirst = (d1 == C && d1 < d2);
            int numAnchors = channelFirst ? d2 : d1;

            for (int i = 0; i < numAnchors; i++) {
                float cx = channelFirst ? data0[0][i] : data0[i][0];
                float cy = channelFirst ? data0[1][i] : data0[i][1];
                float w  = channelFirst ? data0[2][i] : data0[i][2];
                float h  = channelFirst ? data0[3][i] : data0[i][3];

                float maxScore = -1.0f;
                int maxClass = 0;
                for (int c = 0; c < numClasses; c++) {
                    float score = channelFirst ? data0[4 + c][i] : data0[i][4 + c];
                    if (score > maxScore) { maxScore = score; maxClass = c; }
                }
                if (maxScore < scoreTh) continue;

                float bx = cx - w / 2.0f;
                float by = cy - h / 2.0f;

                float[] coeffs = new float[32];
                for (int j = 0; j < 32; j++) {
                    coeffs[j] = channelFirst ? data0[4 + numClasses + j][i] : data0[i][4 + numClasses + j];
                }
                out.add(restoredSegDetection(maxClass, maxScore, bx, by, w, h, coeffs, tf));
            }
        }
        return out;
    }

    /** Restores a segmentation candidate's box to original image coordinates (unclipped). */
    private static Detection restoredSegDetection(int classId, float score,
                                                  float bx, float by, float bw, float bh,
                                                  float[] coeffs, Preprocessor.Transform tf) {
        float[] restored = OrtUtil.restoreBbox(bx, by, bw, bh, tf.scale(), tf.padLeft(), tf.padTop());
        return new Detection(classId, score, restored[0], restored[1], restored[2], restored[3],
                             0f, null, new MaskRef.ProtoMask(coeffs, bx, by, bw, bh), -1);
    }

    // ---------------------------------------------------------------
    // OBB
    // ---------------------------------------------------------------

    public static List<Detection> decodeObb(float[][] data, int numClasses,
                                            ModelContract contract, float scoreTh,
                                            Preprocessor.Transform tf) {
        List<Detection> out = new ArrayList<>();
        int d1 = data.length;
        int d2 = data[0].length;

        if (contract == ModelContract.YOLO_OBB_E2E) {
            // E2E format: [1, N, 7] = cx, cy, w, h, score, classId, angle
            for (int i = 0; i < d1; i++) {
                float score = data[i][4];
                if (score < scoreTh) continue;
                float cx = data[i][0];
                float cy = data[i][1];
                float w  = data[i][2];
                float h  = data[i][3];
                int classId = (int) data[i][5];
                float angle = data[i][6];

                if (w <= 0 || h <= 0) continue;

                out.add(restoredObbDetection(classId, score, cx, cy, w, h, angle, tf));
            }
        } else {
            // Standard format: [1, 5+numClasses, numAnchors] or [1, numAnchors, 5+numClasses]
            // Per anchor: cx,cy,w,h, class0..classN-1, angle
            int C = 5 + numClasses;
            boolean channelFirst = (d1 == C && d1 < d2);
            int numAnchors = channelFirst ? d2 : d1;

            for (int i = 0; i < numAnchors; i++) {
                float cx = channelFirst ? data[0][i] : data[i][0];
                float cy = channelFirst ? data[1][i] : data[i][1];
                float w  = channelFirst ? data[2][i] : data[i][2];
                float h  = channelFirst ? data[3][i] : data[i][3];

                float maxScore = -1.0f;
                int   maxClass = 0;
                for (int c = 0; c < numClasses; c++) {
                    float sc = channelFirst ? data[4 + c][i] : data[i][4 + c];
                    if (sc > maxScore) { maxScore = sc; maxClass = c; }
                }
                if (maxScore < scoreTh) continue;

                float angle = channelFirst ? data[4 + numClasses][i] : data[i][4 + numClasses];

                out.add(restoredObbDetection(maxClass, maxScore, cx, cy, w, h, angle, tf));
            }
        }
        return out;
    }

    /** Restores an OBB candidate to original image coordinates, as the former output code did. */
    private static Detection restoredObbDetection(int classId, float score,
                                                  float cxLb, float cyLb, float wLb, float hLb,
                                                  float angle, Preprocessor.Transform tf) {
        float cx = (cxLb - tf.padLeft()) / tf.scale();
        float cy = (cyLb - tf.padTop()) / tf.scale();
        float rw = wLb / tf.scale();
        float rh = hLb / tf.scale();
        return new Detection(classId, score, cx - rw / 2, cy - rh / 2, rw, rh, angle, null, null, -1);
    }
}
