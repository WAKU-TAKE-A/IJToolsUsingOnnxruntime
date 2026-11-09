// Preprocessor.java - The single preprocessing implementation for all model families
package io.github.wakutakea.ijtoolsusingonnxruntime;

import ij.process.Blitter;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;

/**
 * Applies a {@link PreprocessSpec} to an image. This class is the one
 * implementation behind the former {@code OrtUtil.preprocess} (YOLO
 * letterbox) and {@code RfDetrInference.preprocess} (RF-DETR stretch);
 * both now delegate here. The arithmetic of each variant is carried
 * over unchanged from those implementations (including the exact
 * division-based [0,1] scaling), so produced tensors are identical.
 */
public final class Preprocessor {
    private Preprocessor() {}

    /**
     * Coordinate transform between original image space and model input
     * space. For LETTERBOX, {@code scale/padLeft/padTop} are the
     * letterbox parameters; for STRETCH there is no padding and decoders
     * restore coordinates from normalized values directly.
     */
    public record Transform(float scale, float padLeft, float padTop, boolean letterbox) {}

    /** Preprocessing output: the NCHW input tensor and its coordinate transform. */
    public record Result(float[] tensor, Transform transform) {}

    public static Result apply(ImageProcessor ip, PreprocessSpec spec, int targetW, int targetH) {
        if (spec.resize() == PreprocessSpec.ResizeMode.STRETCH) {
            return new Result(stretch(ip, spec, targetW, targetH),
                              new Transform(1f, 0f, 0f, false));
        }
        return new Result(letterbox(ip, spec, targetW, targetH),
                          transformFor(spec, ip.getWidth(), ip.getHeight(), targetW, targetH));
    }

    /** The transform for an image of the given size, without running the preprocessing. */
    public static Transform transformFor(PreprocessSpec spec, int srcW, int srcH, int targetW, int targetH) {
        if (spec.resize() != PreprocessSpec.ResizeMode.LETTERBOX) {
            return new Transform(1f, 0f, 0f, false);
        }
        float[] p = OrtUtil.calcLetterboxParams(srcW, srcH, targetW, targetH);
        return new Transform(p[0], p[1], p[2], true);
    }

    // ---------------------------------------------------------------
    // LETTERBOX variant (formerly OrtUtil.preprocess)
    // ---------------------------------------------------------------

    private static float[] letterbox(ImageProcessor ip, PreprocessSpec spec, int targetW, int targetH) {
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
        padded.setColor(new Color(spec.padRgb(), spec.padRgb(), spec.padRgb()));
        padded.fill();
        padded.copyBits(resized, padLeft, padTop, Blitter.COPY);

        // ARGB int[] -> NCHW float[]
        int[] pixels    = (int[]) padded.getPixels();
        int   planeSize = targetH * targetW;
        float[] nchw    = new float[3 * planeSize];
        boolean bgrRaw  = spec.channels() == PreprocessSpec.ChannelOrder.BGR;
        for (int i = 0; i < planeSize; i++) {
            int px = pixels[i];
            float r = ((px >> 16) & 0xFF);
            float g = ((px >>  8) & 0xFF);
            float b = ( px        & 0xFF);

            if (bgrRaw) {
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
        }
        return nchw;
    }

    // ---------------------------------------------------------------
    // STRETCH variant (formerly RfDetrInference.preprocess)
    // ---------------------------------------------------------------

    /** Half-pixel bilinear interpolation without antialiasing, on float RGB values. */
    private static float[] stretch(ImageProcessor ip, PreprocessSpec spec, int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid input size");
        int[] rgb = (int[]) ip.convertToRGB().getPixels();
        int srcW = ip.getWidth(), srcH = ip.getHeight(), plane = width * height;
        float[] result = new float[3 * plane];
        float[] mean = spec.mean(), std = spec.std();
        float scaleX = (float) srcW / width, scaleY = (float) srcH / height;
        for (int y = 0; y < height; y++) {
            float sy = Math.max(0, (y + 0.5f) * scaleY - 0.5f);
            int y0 = (int) sy, y1 = Math.min(y0 + 1, srcH - 1);
            float fy = sy - y0;
            for (int x = 0; x < width; x++) {
                float sx = Math.max(0, (x + 0.5f) * scaleX - 0.5f);
                int x0 = (int) sx, x1 = Math.min(x0 + 1, srcW - 1);
                float fx = sx - x0;
                for (int c = 0; c < 3; c++) {
                    int shift = 16 - c * 8;
                    float a = ((rgb[y0 * srcW + x0] >> shift) & 255) / 255f;
                    float b = ((rgb[y0 * srcW + x1] >> shift) & 255) / 255f;
                    float d = ((rgb[y1 * srcW + x0] >> shift) & 255) / 255f;
                    float e = ((rgb[y1 * srcW + x1] >> shift) & 255) / 255f;
                    float v = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
                    result[c * plane + y * width + x] = (v - mean[c]) / std[c];
                }
            }
        }
        return result;
    }
}
