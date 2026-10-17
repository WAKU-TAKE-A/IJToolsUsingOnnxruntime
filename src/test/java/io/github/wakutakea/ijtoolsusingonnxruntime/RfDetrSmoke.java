package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.measure.ResultsTable;
import ij.process.ByteProcessor;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/** Real-model smoke test, run by RF-DETR/verify_rfdetr.py in a separate Java 17 JVM. */
public final class RfDetrSmoke {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]);
        String[][] cases = {
            {"detection-nano", "dog.jpg", "RF_DETR_DETECTION"},
            {"segmentation-nano", "dog.jpg", "RF_DETR_SEGMENTATION"},
            {"keypoint-preview", "person.jpg", "RF_DETR_POSE"}
        };
        for (String[] item : cases) {
            Path output = root.resolve("verification").resolve(item[0]);
            Files.createDirectories(output);
            ImagePlus image = IJ.openImage(root.resolve("samples").resolve(item[1]).toString());
            if (image == null) throw new AssertionError("Cannot open " + item[1]);
            MyOrtSession session = new MyOrtSession();
            try {
                session.load(root.resolve(item[0]).resolve("model.onnx").toString(),
                        MyOrtSession.ModelType.valueOf(item[2]), MyOrtSession.CoordFormat.RF_DETR);
                session.loadClassNamesFromMetadata();
                String error = session.validateFormat();
                if (error != null) throw new AssertionError(error);
                float[] input = RfDetrInference.preprocess(image.getProcessor(),session.getInputWidth(),session.getInputHeight());
                ByteBuffer bytes = ByteBuffer.allocate(input.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                bytes.asFloatBuffer().put(input);
                Files.write(output.resolve("input.f32"), bytes.array());
                var prediction = RfDetrInference.predict(session,image.getProcessor(),.3f);
                if (prediction.detections().isEmpty()) throw new AssertionError("No detections: " + item[0]);
                StringBuilder csv = new StringBuilder("Query,ClassID,Score,X,Y,W,H\n");
                StringBuilder keypoints = new StringBuilder("Detection,Keypoint,X,Y,Score\n");
                int index = 0, maskPixels = 0, visiblePoints = 0;
                for (var d : prediction.detections()) {
                    csv.append(String.format(Locale.ROOT,"%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g%n",
                            d.query(),d.classId(),d.score(),d.x(),d.y(),d.width(),d.height()));
                    if (d.keypoints() != null) for (int k=0;k<d.keypoints().length;k++) {
                        float[] p = d.keypoints()[k];
                        keypoints.append(String.format(Locale.ROOT,"%d,%d,%.9g,%.9g,%.9g%n",index,k,p[0],p[1],p[2]));
                        if (p[2] > .5) visiblePoints++;
                    }
                    if (prediction.masks() != null) {
                        ByteProcessor mask = new ByteProcessor(image.getWidth(),image.getHeight());
                        for (int y=0;y<image.getHeight();y++) for (int x=0;x<image.getWidth();x++) {
                            if (RfDetrInference.maskValue(prediction.masks()[d.query()],x,y,image.getWidth(),image.getHeight()) > 0) {
                                mask.set(x,y,255); maskPixels++;
                            }
                        }
                        if (!new ij.io.FileSaver(new ImagePlus("mask",mask)).saveAsPng(output.resolve("mask-"+index+".png").toString()))
                            throw new AssertionError("Could not save mask");
                    }
                    index++;
                }
                if (prediction.masks() != null && maskPixels == 0) throw new AssertionError("Empty segmentation masks");
                if (session.getModelType() == MyOrtSession.ModelType.RF_DETR_POSE && visiblePoints == 0)
                    throw new AssertionError("No visible keypoints");
                Files.writeString(output.resolve("detections.csv"),csv.toString());
                Files.writeString(output.resolve("keypoints.csv"),keypoints.toString());
                // Render and verify the shared ImageJ presentation (table, mask ROI, keypoint ROI and overlay).
                RfDetrResults.render(prediction,session,image,0,.5,.5,true,false,true,80,false,true);
                ResultsTable rt = ResultsTable.getResultsTable();
                if (rt.size() != prediction.detections().size()) throw new AssertionError("Results table mismatch");
                rt.saveAs(output.resolve("results.csv").toString());
                Overlay overlay = image.getOverlay();
                if (overlay == null || overlay.size() == 0) throw new AssertionError("No overlay ROIs");
                if (!new ij.io.FileSaver(image.flatten()).saveAsPng(output.resolve("overlay.png").toString()))
                    throw new AssertionError("Could not save overlay");
                int count = rt.size();
                RfDetrResults.render(prediction,session,image,0,.5,.5,true,false,false,80,false,false);
                if (rt.size() != 2*count) throw new AssertionError("Append failed");
                RfDetrResults.render(prediction,session,image,0,.5,.5,true,false,false,80,false,true);
                if (rt.size() != count) throw new AssertionError("Refresh failed");
                // Exercise the actual existing plugin entry point, with GUI output disabled in this headless JVM.
                OrtUtil.setSlot(0,session);
                Object plugin = switch (session.getModelType()) {
                    case RF_DETR_DETECTION -> new ORT_2nd_Detection();
                    case RF_DETR_SEGMENTATION -> new ORT_2nd_Segmentation();
                    case RF_DETR_POSE -> new ORT_2nd_Pose();
                    default -> throw new AssertionError("Unexpected task");
                };
                Field manager = plugin.getClass().getDeclaredField("showRoiManager");
                manager.setAccessible(true); manager.setBoolean(null,false);
                var filter = (ij.plugin.filter.ExtendedPlugInFilter)plugin;
                filter.setup("",image);
                filter.run(image.getProcessor());
                // Test the user's dog.jpg reproduction through the real plugin, including the threshold field.
                var filtered = RfDetrInference.applyNms(prediction,.1f);
                var unsuppressed = RfDetrInference.applyNms(prediction,1f);
                if (unsuppressed.detections().size() != count) throw new AssertionError("IoU=1 should retain boxes");
                Field nms = plugin.getClass().getDeclaredField("nmsThreshold");
                nms.setAccessible(true);
                double previousThreshold = nms.getDouble(null);
                nms.setDouble(null,.1);
                try { filter.run(image.getProcessor()); }
                finally { nms.setDouble(null,previousThreshold); }
                if (session.getModelType() != MyOrtSession.ModelType.RF_DETR_POSE
                        && ResultsTable.getResultsTable().size() != filtered.detections().size())
                    throw new AssertionError("Plugin did not apply NMS threshold");
                if (session.getModelType() == MyOrtSession.ModelType.RF_DETR_DETECTION) {
                    long before = prediction.detections().stream().filter(d -> d.classId()==1).count();
                    long after = filtered.detections().stream().filter(d -> d.classId()==1).count();
                    if (before != 2 || after != 1 || filtered.detections().stream()
                            .noneMatch(d -> d.classId()==1 && d.score() > .59f))
                        throw new AssertionError("dog.jpg person duplicate was not removed correctly");
                }
                RfDetrResults.render(filtered,session,image,0,.5,.5,true,false,true,80,false,true);
                Path nmsOutput = output.resolve("nms-0.10");
                Files.createDirectories(nmsOutput);
                ResultsTable.getResultsTable().saveAs(nmsOutput.resolve("results.csv").toString());
                if (!new ij.io.FileSaver(image.flatten()).saveAsPng(nmsOutput.resolve("overlay.png").toString()))
                    throw new AssertionError("Could not save NMS overlay");
                System.out.printf(Locale.ROOT,"PASS NMS %s: %d -> %d detections at IoU=0.10%n",
                        item[0],count,filtered.detections().size());
                System.out.printf(Locale.ROOT,"PASS %s: %d detections, %d visible keypoints, %d mask pixels, %d ms%n",
                        item[0],count,visiblePoints,maskPixels,prediction.inferenceMillis());
            } finally {
                OrtUtil.setSlot(0,null);
                session.release();
                image.close();
            }
        }
    }
}
