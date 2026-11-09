package io.github.wakutakea.ijtoolsusingonnxruntime;

import static org.junit.Assert.*;

import ij.process.ColorProcessor;
import java.util.List;
import org.junit.Test;

/**
 * Deterministic synthetic-tensor tests for the Phase 2 decoders and
 * the contract-driven NMS. Identity letterbox transform (scale 1, no
 * padding) unless a test states otherwise, so expected boxes equal the
 * tensor values.
 */
public class DecoderTest {
    private static final Preprocessor.Transform IDENTITY = new Preprocessor.Transform(1f, 0f, 0f, true);
    private static final float EPS = 1e-6f;

    // ---------------------------------------------------------------
    // YOLO detection: standard / E2E / YOLOX
    // ---------------------------------------------------------------

    @Test public void yoloStandardChannelFirst() {
        // [4+2 classes][7 anchors] (channel-first detection needs anchors > channels,
        // as in real models; anchors 2..6 stay below threshold)
        float[][] data = {
            {50, 10, 10, 0, 0, 0, 0},   // cx
            {50, 10, 10, 0, 0, 0, 0},   // cy
            {20, 10, 10, 0, 0, 0, 0},   // w
            {10, 10, 10, 0, 0, 0, 0},   // h
            {0.1f, 0.8f, 0.2f, 0, 0, 0, 0},  // class 0 scores
            {0.9f, 0.2f, 0.1f, 0, 0, 0, 0},  // class 1 scores
        };
        List<Detection> out = YoloDecoder.decodeDetection(data, false,
                MyOrtSession.CoordFormat.YOLO_PIXEL, 2, 100, 100, 0.5f, IDENTITY, 100, 100);
        assertEquals(2, out.size());
        Detection a = out.get(0);
        assertEquals(1, a.classId());
        assertEquals(0.9f, a.score(), EPS);
        assertEquals(40f, a.x(), EPS); assertEquals(45f, a.y(), EPS);
        assertEquals(20f, a.width(), EPS); assertEquals(10f, a.height(), EPS);
        Detection b = out.get(1);
        assertEquals(0, b.classId());
        assertEquals(5f, b.x(), EPS); assertEquals(5f, b.y(), EPS);
    }

    @Test public void yoloE2eAndClipping() {
        float[][] data = {
            {10, 10, 30, 40, 0.9f, 2},
            {0, 0, 5, 5, 0.4f, 0},       // below threshold
            {90, 90, 120, 120, 0.8f, 0}, // clipped to the 100x100 image
        };
        List<Detection> out = YoloDecoder.decodeDetection(data, false,
                MyOrtSession.CoordFormat.YOLO_OBJECT_E2E, 3, 100, 100, 0.5f, IDENTITY, 100, 100);
        assertEquals(2, out.size());
        assertEquals(2, out.get(0).classId());
        assertEquals(20f, out.get(0).width(), EPS);
        assertEquals(30f, out.get(0).height(), EPS);
        assertEquals(90f, out.get(1).x(), EPS);
        assertEquals(10f, out.get(1).width(), EPS);
        assertEquals(10f, out.get(1).height(), EPS);
    }

    @Test public void yoloxGridDecode() {
        int anchors = 80 * 80 + 40 * 40 + 20 * 20;
        float[][] data = new float[anchors][7]; // 5 + 2 classes, all zero
        // Anchor 0: grid (0,0), stride 8. obj=1, class0=0.9 -> conf 0.9, box (4,4,8,8)
        data[0] = new float[]{1, 1, 0, 0, 1.0f, 0.9f, 0.1f};
        List<Detection> out = YoloDecoder.decodeDetection(data, true,
                MyOrtSession.CoordFormat.YOLOX_UNDECODED, 2, 640, 640, 0.5f, IDENTITY, 640, 640);
        assertEquals(1, out.size());
        Detection d = out.get(0);
        assertEquals(0, d.classId());
        assertEquals(0.9f, d.score(), EPS);
        assertEquals(4f, d.x(), EPS); assertEquals(4f, d.y(), EPS);
        assertEquals(8f, d.width(), EPS); assertEquals(8f, d.height(), EPS);
    }

    // ---------------------------------------------------------------
    // Pose
    // ---------------------------------------------------------------

    @Test public void poseKeypointsRestored() {
        int channels = 5 + 17 * 3;
        float[][] data = new float[channels][1];
        data[0][0] = 50; data[1][0] = 50; data[2][0] = 20; data[3][0] = 40; data[4][0] = 0.9f;
        data[5][0] = 50; data[6][0] = 50; data[7][0] = 0.95f;  // keypoint 0
        data[8][0] = 40; data[9][0] = 60; data[10][0] = 0.3f;  // keypoint 1
        List<Detection> out = YoloDecoder.decodePose(data,
                MyOrtSession.CoordFormat.YOLO_POSE, 100, 100, 0.5f, IDENTITY);
        assertEquals(1, out.size());
        Detection d = out.get(0);
        assertEquals(40f, d.x(), EPS); assertEquals(30f, d.y(), EPS);
        assertEquals(17, d.keypoints().length);
        assertArrayEquals(new float[]{50, 50, 0.95f}, d.keypoints()[0], EPS);
        assertArrayEquals(new float[]{40, 60, 0.3f}, d.keypoints()[1], EPS);
    }

    @Test public void poseNmsIsClassAgnostic() {
        Detection a = new Detection(0, 0.9f, 10, 10, 20, 20);
        Detection b = new Detection(0, 0.8f, 10, 10, 20, 20);
        List<Detection> out = PostProcessors.applyNms(List.of(a, b), ModelContract.YOLO_POSE, 0.45f);
        assertEquals(1, out.size());
        assertSame(a, out.get(0));
    }

    // ---------------------------------------------------------------
    // Segmentation: mask coefficients survive decoding and NMS
    // ---------------------------------------------------------------

    @Test public void segmentationRetainsCoefficients() {
        int channels = 4 + 1 + 32;
        float[][] data = new float[channels][40]; // anchors > channels, as in real models
        data[0][0] = 50; data[1][0] = 50; data[2][0] = 20; data[3][0] = 20;
        data[4][0] = 0.9f; // the single class score
        for (int j = 0; j < 32; j++) data[5 + j][0] = j * 0.01f;
        data[4][1] = 0.1f; // second anchor below threshold
        // Non-identity transform: scale 0.5, pad (10, 20)
        Preprocessor.Transform tf = new Preprocessor.Transform(0.5f, 10f, 20f, true);
        List<Detection> out = YoloDecoder.decodeSegmentation(data, 1,
                MyOrtSession.CoordFormat.YOLO_SEGMENT, 0.5f, tf);
        assertEquals(1, out.size());
        Detection d = out.get(0);
        // Restored: ((40-10)/0.5, (40-20)/0.5, 20/0.5, 20/0.5)
        assertEquals(60f, d.x(), EPS); assertEquals(40f, d.y(), EPS);
        assertEquals(40f, d.width(), EPS); assertEquals(40f, d.height(), EPS);
        assertTrue(d.mask() instanceof MaskRef.ProtoMask);
        MaskRef.ProtoMask pm = (MaskRef.ProtoMask) d.mask();
        // Letterbox-space box is retained for prototype cropping
        assertEquals(40f, pm.boxX(), EPS); assertEquals(40f, pm.boxY(), EPS);
        assertEquals(20f, pm.boxW(), EPS); assertEquals(20f, pm.boxH(), EPS);
        for (int j = 0; j < 32; j++) assertEquals(j * 0.01f, pm.coeffs()[j], EPS);
    }

    @Test public void segmentationE2e() {
        float[][] data = new float[1][38];
        data[0][0] = 10; data[0][1] = 10; data[0][2] = 30; data[0][3] = 50;
        data[0][4] = 0.9f; data[0][5] = 3;
        for (int j = 0; j < 32; j++) data[0][6 + j] = 1f;
        List<Detection> out = YoloDecoder.decodeSegmentation(data, 4,
                MyOrtSession.CoordFormat.YOLO_SEGMENT_E2E, 0.5f, IDENTITY);
        assertEquals(1, out.size());
        assertEquals(3, out.get(0).classId());
        assertEquals(20f, out.get(0).width(), EPS);
        assertEquals(40f, out.get(0).height(), EPS);
        assertEquals(1f, ((MaskRef.ProtoMask) out.get(0).mask()).coeffs()[31], EPS);
    }

    // ---------------------------------------------------------------
    // OBB: angle survives decoding; envelope NMS still suppresses
    // ---------------------------------------------------------------

    @Test public void obbRetainsAngle() {
        float[][] data = new float[7][8]; // 5 + 2 classes, channel-first (anchors > channels)
        data[0][0] = 50; data[1][0] = 50; data[2][0] = 30; data[3][0] = 10;
        data[4][0] = 0.2f; data[5][0] = 0.9f; data[6][0] = 0.5f; // angle last
        List<Detection> out = YoloDecoder.decodeObb(data, 2,
                MyOrtSession.CoordFormat.YOLO_OBB, 0.5f, IDENTITY);
        assertEquals(1, out.size());
        Detection d = out.get(0);
        assertEquals(1, d.classId());
        assertEquals(35f, d.x(), EPS); assertEquals(45f, d.y(), EPS);
        assertEquals(30f, d.width(), EPS); assertEquals(10f, d.height(), EPS);
        assertEquals(0.5f, d.angle(), EPS);
    }

    @Test public void obbE2eAndEnvelopeNms() {
        float[][] data = {{50, 50, 30, 10, 0.9f, 1, -0.25f}};
        List<Detection> out = YoloDecoder.decodeObb(data, 2,
                MyOrtSession.CoordFormat.YOLO_OBB_E2E, 0.5f, IDENTITY);
        assertEquals(1, out.size());
        assertEquals(-0.25f, out.get(0).angle(), EPS);
        assertEquals(35f, out.get(0).x(), EPS);

        Detection dup = new Detection(1, 0.8f, 35, 45, 30, 10, -0.25f, null, null, -1);
        List<Detection> kept = PostProcessors.applyNms(List.of(out.get(0), dup),
                ModelContract.YOLO_OBB_E2E, 0.45f);
        assertEquals(2, kept.size()); // E2E policy passes through
        List<Detection> suppressed = PostProcessors.applyNms(List.of(out.get(0), dup),
                ModelContract.YOLO_OBB, 0.45f);
        assertEquals(1, suppressed.size()); // envelope IoU = 1 for identical boxes
    }

    // ---------------------------------------------------------------
    // NMS policies
    // ---------------------------------------------------------------

    @Test public void nmsPolicies() {
        Detection a = new Detection(0, 0.9f, 10, 10, 20, 20);
        Detection sameClass = new Detection(0, 0.8f, 10, 10, 20, 20);
        Detection otherClass = new Detection(1, 0.8f, 10, 10, 20, 20);
        // PER_CLASS: same class suppressed, other class kept
        assertEquals(1, PostProcessors.applyNms(List.of(a, sameClass), ModelContract.YOLO_OBJECT_PIXEL, 0.45f).size());
        assertEquals(2, PostProcessors.applyNms(List.of(a, otherClass), ModelContract.YOLO_OBJECT_PIXEL, 0.45f).size());
        // NONE_E2E: untouched
        List<Detection> in = List.of(a, sameClass);
        assertSame(in, PostProcessors.applyNms(in, ModelContract.YOLO_OBJECT_E2E, 0.45f));
    }

    // ---------------------------------------------------------------
    // Classification: ranked entries, selection-sort tie order pinned
    // ---------------------------------------------------------------

    @Test public void classifyTopK() {
        List<ClassifyDecoder.Entry> out = ClassifyDecoder.decode(new float[]{0.1f, 0.5f, 0.2f}, 2);
        assertEquals(2, out.size());
        assertEquals(1, out.get(0).classId());
        assertEquals(2, out.get(1).classId());
        // Tie order of the ported selection sort: [0.3, 0.3, 0.9] -> classes 2, 1, 0
        List<ClassifyDecoder.Entry> ties = ClassifyDecoder.decode(new float[]{0.3f, 0.3f, 0.9f}, 3);
        assertEquals(2, ties.get(0).classId());
        assertEquals(1, ties.get(1).classId());
        assertEquals(0, ties.get(2).classId());
    }

    // ---------------------------------------------------------------
    // RF-DETR: NMS must not break query / mask / keypoint association
    // ---------------------------------------------------------------

    @Test public void rfDetrNmsRetainsQueryAssociations() {
        // 3 queries x 2 classes; boxes normalized cx, cy, w, h
        float[][] boxes = {
            {0.5f, 0.5f, 0.4f, 0.4f},  // q0
            {0.5f, 0.5f, 0.4f, 0.4f},  // q1: identical box, same class -> suppressed
            {0.1f, 0.1f, 0.1f, 0.1f},  // q2: separate box, class 1
        };
        float[][] logits = {{3f, 0f}, {2.5f, 0f}, {0f, 3f}};
        // Keypoint rows are class-major: counts {2, 2} -> 4 rows per query (class * maxK + k)
        float[][][] keypoints = new float[3][4][3];
        for (int q = 0; q < 3; q++) {
            for (int r = 0; r < 4; r++) keypoints[q][r] = new float[]{0.1f * (q + 1), 0.2f, 0f};
        }
        float[][][] masks = new float[3][2][2];
        for (int q = 0; q < 3; q++) masks[q][0][0] = 100 + q; // per-query marker

        List<Detection> decoded = RfDetrDecoder.decode(boxes, logits, keypoints,
                100, 50, 0.5f, 10, -1, new int[]{2, 2}, 0);
        assertEquals(3, decoded.size());

        List<Detection> kept = PostProcessors.applyNms(decoded, ModelContract.RF_DETR_OBJECT, 0.45f);
        assertEquals(2, kept.size());
        boolean[] seenQuery = new boolean[3];
        for (Detection d : kept) {
            seenQuery[d.query()] = true;
            // The mask and keypoints reachable via this detection still belong to its query
            assertEquals(100 + d.query(), masks[d.query()][0][0], EPS);
            assertEquals(0.1f * (d.query() + 1) * 100, d.keypoints()[0][0], EPS);
            assertEquals(0.2f * 50, d.keypoints()[0][1], EPS);
        }
        assertTrue(seenQuery[0]);
        assertFalse(seenQuery[1]); // suppressed duplicate
        assertTrue(seenQuery[2]);

        // The legacy facade keeps the same query/keypoint association
        List<RfDetrInference.Detection> legacy = RfDetrInference.decode(boxes, logits, keypoints,
                100, 50, 0.5f, 10, -1, new int[]{2, 2}, 0);
        assertEquals(3, legacy.size());
        assertEquals(0, legacy.get(0).query());
        assertEquals(10f, legacy.get(0).keypoints()[0][0], EPS);
    }

    // ---------------------------------------------------------------
    // Preprocessor: letterbox tensor layout (YOLO and YOLOX specs)
    // ---------------------------------------------------------------

    @Test public void preprocessorLetterboxSpecs() {
        ColorProcessor cp = new ColorProcessor(1, 1);
        cp.set(0, 0, 0xFF0000); // pure red
        float[] yolo = Preprocessor.apply(cp, PreprocessSpec.YOLO_LETTERBOX, 1, 1).tensor();
        assertArrayEquals(new float[]{1f, 0f, 0f}, yolo, EPS); // RGB, [0,1]
        float[] yolox = Preprocessor.apply(cp, PreprocessSpec.YOLOX_LETTERBOX, 1, 1).tensor();
        assertArrayEquals(new float[]{0f, 0f, 255f}, yolox, EPS); // BGR, [0,255]
    }
}
