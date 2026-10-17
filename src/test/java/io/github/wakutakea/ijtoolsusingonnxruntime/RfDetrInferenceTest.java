package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.process.ColorProcessor;
import org.junit.Test;
import static org.junit.Assert.*;

public class RfDetrInferenceTest {
    @Test public void usesImageNetRgbWithoutLetterbox() {
        ColorProcessor image = new ColorProcessor(2, 1, new int[]{0xff0000, 0x0000ff});
        float[] tensor = RfDetrInference.preprocess(image, 1, 1);
        assertArrayEquals(new float[]{(0.5f-.485f)/.229f, (0-.456f)/.224f, (.5f-.406f)/.225f}, tensor, 1e-6f);
    }
    @Test public void stretchesNonSquareImagesWithHalfPixelSampling() {
        ColorProcessor image = new ColorProcessor(2, 1, new int[]{0xff0000, 0x0000ff});
        float[] tensor = RfDetrInference.preprocess(image, 4, 2);
        assertEquals((.75f-.485f)/.229f, tensor[1], 1e-6);
        assertEquals(tensor[1], tensor[5], 0);
        assertEquals((.25f-.406f)/.225f, tensor[17], 1e-6);
    }
    @Test public void selectsGlobalQueryClassPairsAndDoesNotSuppressOverlaps() {
        float[][] boxes = {{.5f,.5f,.5f,.5f},{.5f,.5f,.5f,.5f}};
        float[][] logits = {{4,3},{2,-9}};
        var result = RfDetrInference.decode(boxes, logits, null, 100, 50, .5f, 3, -1, new int[0], 0);
        assertEquals(3, result.size());
        assertEquals(0, result.get(0).query());
        assertEquals(1, result.get(1).classId());
        assertEquals(1, result.get(2).query());
        assertEquals(25, result.get(0).x(), 0);
        assertEquals(12.5f, result.get(0).y(), 0);
    }
    @Test public void clipsBoxEdgesInsteadOfShiftingWidth() {
        var result = RfDetrInference.decode(new float[][]{{0,.5f,.4f,1}}, new float[][]{{5}}, null,
                100,50,.5f,1,-1,new int[0],0);
        assertEquals(0, result.get(0).x(), 0);
        assertEquals(20, result.get(0).width(), 1e-5);
    }
    @Test public void keepsSparseCocoClass90AndExcludesExplicitBackground() {
        float[][] logits = new float[1][91];
        java.util.Arrays.fill(logits[0], -20);
        logits[0][90] = 10;
        float[][] boxes = {{.5f,.5f,1,1}};
        var sparse = RfDetrInference.decode(boxes,logits,null,10,10,.5f,1,-1,new int[0],0);
        assertEquals(90, sparse.get(0).classId());
        assertTrue(RfDetrInference.decode(boxes,logits,null,10,10,.5f,1,90,new int[0],0).isEmpty());
    }
    @Test public void gathersKeypointsByQueryAndClassSchema() {
        float[][] boxes = {{.5f,.5f,1,1}};
        float[][][] kp = {{{.1f,.2f,5},{.7f,.8f,5}}};
        var result = RfDetrInference.decode(boxes,new float[][]{{-8,8}},kp,100,50,.5f,1,-1,new int[]{1,1},0);
        assertEquals(70, result.get(0).keypoints()[0][0], 1e-5);
        assertEquals(40, result.get(0).keypoints()[0][1], 1e-5);
    }
    @Test public void fusesKeypointUncertaintyBeforeThresholding() {
        float[][][] kp = {{{.5f,.5f,0,0,0,0,0}}};
        // Unit precision gives covariance trace=2. At alpha=1, fused score = p/(2+p).
        var result = RfDetrInference.decode(new float[][]{{.5f,.5f,1,1}},new float[][]{{0}},kp,
                10,10,.1f,1,-1,new int[]{1},1);
        assertEquals(.2, result.get(0).score(), 1e-6);
        assertTrue(RfDetrInference.decode(new float[][]{{.5f,.5f,1,1}},new float[][]{{0}},kp,
                10,10,.3f,1,-1,new int[]{1},1).isEmpty());
    }
    @Test public void interpolatesMaskLogitsBeforeThresholding() {
        float[][] mask = {{-4,4},{-4,4}};
        assertEquals(-4, RfDetrInference.maskValue(mask,0,0,4,4), 0);
        assertEquals(-2, RfDetrInference.maskValue(mask,1,1,4,4), 0);
        assertEquals(2, RfDetrInference.maskValue(mask,2,2,4,4), 0);
        assertEquals(4, RfDetrInference.maskValue(mask,3,3,4,4), 0);
    }
    @Test public void rejectsNanThreshold() {
        try {
            RfDetrInference.decode(new float[][]{{0,0,1,1}},new float[][]{{0}},null,1,1,
                    Float.NaN,1,-1,new int[0],0);
            fail("NaN threshold must fail");
        } catch (IllegalArgumentException expected) { }
    }
}
