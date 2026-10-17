package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.List;
import org.junit.Test;
import static org.junit.Assert.*;

public class NmsTest {
    @Test public void removesLowerConfidenceDuplicateWithoutMutatingInput() {
        float[] low = {0,0,10,10,.4f,1}, high = {1,1,10,10,.9f,1};
        List<float[]> original = List.of(low,high);
        var result = Nms.perClass(original,.1f);
        assertEquals(1,result.size());
        assertSame(high,result.get(0));
        assertSame(low,original.get(0));
    }
    @Test public void preservesDifferentClassesAndSparseClassIds() {
        float[] car = {0,0,10,10,.9f,3}, truck = {0,0,10,10,.8f,8};
        float[] sparse = {0,0,10,10,.7f,1_000_000};
        assertEquals(3,Nms.perClass(List.of(car,truck,sparse),0).size());
        assertEquals(1,Nms.classAgnostic(List.of(car,truck,sparse),0).size());
    }
    @Test public void keepsDisjointBoxesAndStableScoreTies() {
        float[] first = {0,0,10,10,.8f,1}, same = {0,0,10,10,.8f,1};
        float[] separate = {10,0,10,10,.8f,1};
        var kept = Nms.perClass(List.of(first,same,separate),0);
        assertEquals(2,kept.size());
        assertSame(first,kept.get(0));
        assertSame(separate,kept.get(1));
    }
    @Test public void thresholdOneKeepsIdenticalBoxes() {
        float[] a = {0,0,10,10,.9f,1}, b = {0,0,10,10,.8f,1};
        assertEquals(2,Nms.perClass(List.of(a,b),1).size());
        assertEquals(1,Nms.perClass(List.of(a,b),Math.nextDown(1f)).size());
    }
    @Test public void ignoresInvalidBoxesAndRejectsInvalidThresholds() {
        float[] a = {0,0,0,10,.9f,1}, b = {0,0,10,10,Float.NaN,1};
        assertTrue(Nms.perClass(List.of(a,b),.45f).isEmpty());
        for (float threshold : new float[]{Float.NaN,Float.POSITIVE_INFINITY,-.1f,1.1f}) {
            try { Nms.perClass(List.of(),threshold); fail("Invalid threshold accepted"); }
            catch (IllegalArgumentException expected) { }
        }
    }
    @Test public void supportsSingleClassPoseAndRetainsAdditionalPayload() {
        float[] pose = {0,0,10,10,.9f,20,30,.8f};
        float[] duplicate = {0,0,10,10,.8f};
        assertSame(pose,Nms.classAgnostic(List.of(duplicate,pose),.45f).get(0));
    }
    @Test public void rfDetrFilteringRetainsQueryMasksAndKeypoints() {
        float[][] points = {{3,4,.9f}};
        var high = new RfDetrInference.Detection(7,1,.9f,0,0,10,10,points);
        var low = new RfDetrInference.Detection(2,1,.3f,0,0,10,10,null);
        float[][][] masks = new float[8][1][1];
        var input = new RfDetrInference.Prediction(List.of(low,high),masks,123);
        var output = RfDetrInference.applyNms(input,.1f);
        assertEquals(1,output.detections().size());
        assertSame(high,output.detections().get(0));
        assertSame(points,output.detections().get(0).keypoints());
        assertSame(masks,output.masks());
        assertEquals(123,output.inferenceMillis());
        assertEquals(2,input.detections().size());
    }
}
