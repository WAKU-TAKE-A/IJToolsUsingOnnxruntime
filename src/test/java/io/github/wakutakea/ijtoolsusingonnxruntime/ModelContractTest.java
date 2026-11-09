package io.github.wakutakea.ijtoolsusingonnxruntime;

import static org.junit.Assert.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;

/**
 * Phase 3 contract tests: the label/index order is a macro
 * compatibility contract, and the bundled-manifest reader must prefer
 * the per-model sibling file and map export task strings.
 */
public class ModelContractTest {

    @Test public void labelsKeepHistoricalOrder() {
        assertArrayEquals(new String[]{
                "YOLO_Object_Pixel", "YOLO_Object_Normalized", "YOLO_Object_E2E",
                "YOLO_Class", "YOLO_Pose", "YOLO_Pose_E2E", "YOLO_Segment",
                "YOLO_Segment_E2E", "YOLO_OBB", "YOLO_OBB_E2E",
                "YOLOX_Object_Undecoded", "RF_DETR_Object", "RF_DETR_Segment",
                "RF_DETR_Pose",
        }, ModelContract.labels());
    }

    @Test public void indexAndLabelLookupRoundTrip() {
        for (int i = 0; i < ModelContract.values().length; i++) {
            ModelContract c = ModelContract.values()[i];
            assertSame(c, ModelContract.fromIndex(i));
            assertSame(c, ModelContract.fromLabel(c.label()));
        }
        assertSame(ModelContract.values()[0], ModelContract.fromIndex(-1));
        assertSame(ModelContract.values()[0], ModelContract.fromIndex(999));
        assertNull(ModelContract.fromLabel("no such format"));
    }

    @Test public void manifestSiblingFileIsRead() throws Exception {
        Path dir = Files.createTempDirectory("manifest-test");
        Path model = dir.resolve("yolov8n-seg.onnx");
        Files.write(model, new byte[0]);
        Files.writeString(dir.resolve("yolov8n-seg.onnx.manifest.json"),
                "{\"task\": \"segment\", \"num_classes\": 80, \"sha256\": \"abc\"}",
                StandardCharsets.UTF_8);
        ModelManifest m = ModelManifest.findFor(model.toString());
        assertNotNull(m);
        assertEquals(80, m.numClasses());
        assertEquals("segment", m.taskLabel());
        assertEquals(ModelContract.Task.SEGMENTATION, m.task());
    }

    @Test public void manifestDirectoryFallbackAndSiblingPrecedence() throws Exception {
        Path dir = Files.createTempDirectory("manifest-test");
        Path model = dir.resolve("model.onnx");
        Files.write(model, new byte[0]);
        Files.writeString(dir.resolve("manifest.json"),
                "{\"task\": \"detect\", \"num_classes\": 3}", StandardCharsets.UTF_8);
        ModelManifest fallback = ModelManifest.findFor(model.toString());
        assertNotNull(fallback);
        assertEquals(3, fallback.numClasses());
        assertEquals(ModelContract.Task.DETECTION, fallback.task());

        Files.writeString(dir.resolve("model.onnx.manifest.json"),
                "{\"task\": \"obb\", \"num_classes\": 5}", StandardCharsets.UTF_8);
        ModelManifest sibling = ModelManifest.findFor(model.toString());
        assertNotNull(sibling);
        assertEquals(5, sibling.numClasses());
        assertEquals(ModelContract.Task.OBB, sibling.task());
    }

    @Test public void manifestAbsentOrContentless() throws Exception {
        Path dir = Files.createTempDirectory("manifest-test");
        Path model = dir.resolve("lonely.onnx");
        Files.write(model, new byte[0]);
        assertNull(ModelManifest.findFor(model.toString()));
        Files.writeString(dir.resolve("manifest.json"), "{\"unrelated\": true}",
                StandardCharsets.UTF_8);
        assertNull(ModelManifest.findFor(model.toString()));
    }
}
