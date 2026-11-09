package io.github.wakutakea.ijtoolsusingonnxruntime;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.measure.ResultsTable;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** Real-model smoke harness for YOLO detection, driven by {@code yolo/verify_yolo.py}.
 * Mirrors {@link RfDetrSmoke}: dumps the preprocessed input tensor and the
 * decoded pre-NMS detections so Python can compare against its own
 * reference, then exercises the real plugin entry and checks the
 * ResultsTable size equals the post-NMS count at the same thresholds.
 * Not a JUnit test; run it through verify_yolo.py with a real model.
 *
 * <p>Arguments: {@code [modelsDir] [onnxName] [formatLabel] [imageName]},
 * defaults {@code yolo/models}, {@code yolov8n.onnx},
 * {@code YOLO_Object_Pixel}, {@code bus.jpg} (looked up under
 * {@code modelsDir/samples/}). Outputs go to
 * {@code modelsDir/verification/java/<model stem>/}.</p>
 */
public final class YoloSmoke {
    private static final float SCORE_THRESHOLD = 0.3f;
    private static final float NMS_THRESHOLD = 0.45f;

    private YoloSmoke() {}

    public static void main(String[] args) throws Exception {
        Path root = Path.of(args.length > 0 ? args[0] : "yolo/models");
        String onnxName = args.length > 1 ? args[1] : "yolov8n.onnx";
        String formatLabel = args.length > 2 ? args[2] : "YOLO_Object_Pixel";
        String imageName = args.length > 3 ? args[3] : "bus.jpg";
        Path modelPath = root.resolve(onnxName);
        Path imagePath = root.resolve("samples").resolve(imageName);
        String stem = onnxName.endsWith(".onnx")
                ? onnxName.substring(0, onnxName.length() - ".onnx".length()) : onnxName;
        Path output = root.resolve("verification").resolve("java").resolve(stem);
        Files.createDirectories(output);

        ModelContract contract = ModelContract.fromLabel(formatLabel);
        if (contract == null || !formatLabel.equals(contract.label()))
            throw new AssertionError("Unknown format label: " + formatLabel);
        if (contract.task() != ModelContract.Task.DETECTION || contract.family() != ModelContract.Family.YOLO)
            throw new AssertionError("YoloSmoke covers YOLO detection formats only: " + formatLabel);

        OrtModel session = new OrtModel();
        ImagePlus image = null;
        try {
            session.load(modelPath.toString(), contract);
            String error = session.validate();
            if (error != null) throw new AssertionError(error);
            Path namesFile = root.resolve(stem + ".txt");
            if (Files.exists(namesFile)) session.loadClassNamesFromFile(namesFile.toString());
            else session.loadClassNamesFromMetadata();

            image = IJ.openImage(imagePath.toString());
            if (image == null) throw new AssertionError("Could not open " + imagePath);
            int imgW = image.getWidth();
            int imgH = image.getHeight();

            ORT_2nd_Detection plugin = new ORT_2nd_Detection();
            Preprocessor.Result prep = ORT_2nd_Detection.prepareInput(session, image.getProcessor());
            writeLittleEndianFloats(output.resolve("input.f32"), prep.tensor());

            List<Detection> candidates;
            List<Detection> postNms;
            long start = System.nanoTime();
            try (OrtSession.Result result = session.runInference(prep.tensor())) {
                long millis = (System.nanoTime() - start) / 1_000_000;
                OnnxTensor outTensor = (OnnxTensor) result.get(0);
                float[][] data = ((float[][][]) outTensor.getValue())[0];
                candidates = YoloDecoder.decodeDetection(data, contract, session.getNumClasses(),
                        imgW, imgH, SCORE_THRESHOLD, prep.transform(),
                        session.getInputWidth(), session.getInputHeight());
                postNms = PostProcessors.applyNms(candidates, contract, NMS_THRESHOLD);
                System.out.printf(Locale.ROOT, "Inference time: %d ms, %d classes%n",
                        millis, session.getNumClasses());
            }

            try (PrintWriter csv = new PrintWriter(Files.newBufferedWriter(
                    output.resolve("detections.csv"), StandardCharsets.UTF_8))) {
                csv.println("ClassID,Score,X,Y,W,H");
                for (Detection d : candidates)
                    csv.printf(Locale.ROOT, "%d,%.6f,%.3f,%.3f,%.3f,%.3f%n",
                            d.classId(), d.score(), d.x(), d.y(), d.width(), d.height());
            }

            // Exercise the real plugin entry at the same thresholds and
            // confirm the shared ResultsTable holds the post-NMS rows.
            OrtUtil.setSlot(0, session);
            setStatic(ORT_2nd_Detection.class, "slotChoice", 0);
            setStatic(ORT_2nd_Detection.class, "scoreThreshold", (double) SCORE_THRESHOLD);
            setStatic(ORT_2nd_Detection.class, "nmsThreshold", (double) NMS_THRESHOLD);
            setField(plugin, "imp", image);
            WindowManager.setTempCurrentImage(image);
            ResultsTable.getResultsTable().reset();
            plugin.run(image.getProcessor());
            int tableRows = ResultsTable.getResultsTable().size();
            if (tableRows != postNms.size())
                throw new AssertionError("ResultsTable rows " + tableRows
                        + " != post-NMS detections " + postNms.size());

            System.out.printf(Locale.ROOT, "PASS %s: %d pre-NMS, %d post-NMS detections%n",
                    onnxName, candidates.size(), postNms.size());
        } finally {
            OrtUtil.setSlot(0, null);
            session.release();
            if (image != null) image.close();
        }
    }

    private static void writeLittleEndianFloats(Path path, float[] values) throws Exception {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(values);
        Files.write(path, buffer.array());
    }

    private static void setStatic(Class<?> type, String name, Object value) throws Exception {
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        if (value instanceof Integer i) field.setInt(null, i);
        else if (value instanceof Double d) field.setDouble(null, d);
        else field.set(null, value);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Class<?> type = target.getClass();
        while (type != null) {
            try {
                Field field = type.getDeclaredField(name);
                field.setAccessible(true);
                field.set(target, value);
                return;
            } catch (NoSuchFieldException e) {
                type = type.getSuperclass();
            }
        }
        throw new NoSuchFieldException(name);
    }
}
