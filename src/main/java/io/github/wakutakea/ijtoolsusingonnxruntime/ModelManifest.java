// ModelManifest.java - Reader for the manifest written next to an exported ONNX model
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A bundled model manifest: the JSON file the Python export scripts
 * ({@code yolo/download_and_convert.py}, {@code yolo/convert.py},
 * {@code RF-DETR/export_rfdetr.py}) write next to an exported ONNX
 * model, recording its provenance, export settings, input/output
 * shapes and SHA-256.
 *
 * <p>For YOLO-family contracts the manifest is stronger evidence than
 * output-shape heuristics, so a loaded model prefers the manifest's
 * class count over the contract's shape-based estimation, and the
 * contract validation cross-checks the manifest task (Phase 3, P1-6).
 * RF-DETR contracts never consult a manifest: their validation stays
 * the strict ONNX-metadata contract in {@code RfDetrInference}.
 *
 * <p>Lookup order for a model file {@code <dir>/<name>.onnx}:
 * <ol>
 *   <li>{@code <dir>/<name>.onnx.manifest.json} (YOLO export layout)</li>
 *   <li>{@code <dir>/manifest.json} (RF-DETR export layout)</li>
 * </ol>
 *
 * <p>The reader is deliberately tolerant: it extracts only the two
 * fields the Java side acts on ({@code num_classes} and {@code task})
 * with anchored patterns, and treats anything unparsable as absent.
 */
public final class ModelManifest {

    private static final Pattern NUM_CLASSES = Pattern.compile("\"num_classes\"\\s*:\\s*(\\d+)");
    private static final Pattern TASK = Pattern.compile("\"task\"\\s*:\\s*\"([^\"]+)\"");

    private final Path path;
    private final int numClasses;                 // -1 when not recorded
    private final String taskLabel;               // raw task string, null when absent
    private final ModelContract.Task task;        // mapped task, null when absent/unknown

    private ModelManifest(Path path, int numClasses, String taskLabel) {
        this.path = path;
        this.numClasses = numClasses;
        this.taskLabel = taskLabel;
        this.task = mapTask(taskLabel);
    }

    /** Maps an export-side task string ("detect", "segment", ...) to a contract task. */
    private static ModelContract.Task mapTask(String label) {
        if (label == null) return null;
        return switch (label.toLowerCase()) {
            case "detect" -> ModelContract.Task.DETECTION;
            case "classify" -> ModelContract.Task.CLASSIFICATION;
            case "pose" -> ModelContract.Task.POSE;
            case "segment" -> ModelContract.Task.SEGMENTATION;
            case "obb" -> ModelContract.Task.OBB;
            default -> null;
        };
    }

    /**
     * Finds and reads the manifest bundled with a model file.
     *
     * @return the manifest, or null when no manifest file exists or it
     *         records neither a class count nor a task
     */
    public static ModelManifest findFor(String modelPath) {
        Path model = Path.of(modelPath);
        Path dir = model.getParent();
        if (dir == null) dir = Path.of(".");
        Path[] candidates = {
            dir.resolve(model.getFileName() + ".manifest.json"),
            dir.resolve("manifest.json"),
        };
        for (Path candidate : candidates) {
            ModelManifest m = read(candidate);
            if (m != null) return m;
        }
        return null;
    }

    private static ModelManifest read(Path file) {
        try {
            if (!Files.isRegularFile(file)) return null;
            String text = Files.readString(file, StandardCharsets.UTF_8);
            int numClasses = -1;
            Matcher nc = NUM_CLASSES.matcher(text);
            if (nc.find()) numClasses = Integer.parseInt(nc.group(1));
            String taskLabel = null;
            Matcher t = TASK.matcher(text);
            if (t.find()) taskLabel = t.group(1);
            if (numClasses < 0 && taskLabel == null) return null;
            return new ModelManifest(file, numClasses, taskLabel);
        } catch (IOException | RuntimeException e) {
            return null; // A broken manifest must never break model loading.
        }
    }

    /** The manifest file that was read. */
    public Path path() { return path; }

    /** Recorded class count, or -1 when the manifest does not record one. */
    public int numClasses() { return numClasses; }

    /** Recorded task string as written by the export script, or null. */
    public String taskLabel() { return taskLabel; }

    /** Recorded task mapped to a contract task, or null when absent/unknown. */
    public ModelContract.Task task() { return task; }
}
