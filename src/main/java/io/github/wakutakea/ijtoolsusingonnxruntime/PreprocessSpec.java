// PreprocessSpec.java - Declarative description of a model's input preprocessing
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * The preprocessing contract of a model format, declared once in
 * {@link ModelContract} instead of being encoded in call-site flags
 * (the old {@code boolean isYolox} conflated channel order and value
 * range; the RF-DETR variant hid behind a same-named method with the
 * opposite geometry).
 *
 * @param resize   letterbox (aspect preserved, padded) or stretch
 * @param padRgb   padding gray value for LETTERBOX (YOLO uses 114)
 * @param channels channel order written to the tensor
 * @param scale    value scale: 1/255 for [0,1] inputs, 1.0 for raw [0,255]
 * @param mean     per-channel normalization mean, or null for none
 * @param std      per-channel normalization std, or null for none
 */
public record PreprocessSpec(ResizeMode resize, int padRgb, ChannelOrder channels,
                             float scale, float[] mean, float[] std) {

    public enum ResizeMode { LETTERBOX, STRETCH }
    public enum ChannelOrder { RGB, BGR }

    /** YOLO family: letterbox with 114-gray padding, RGB, [0,1]. */
    public static final PreprocessSpec YOLO_LETTERBOX =
            new PreprocessSpec(ResizeMode.LETTERBOX, 114, ChannelOrder.RGB, 1.0f / 255.0f, null, null);

    /** YOLOX: letterbox with 114-gray padding, BGR, raw [0,255]. */
    public static final PreprocessSpec YOLOX_LETTERBOX =
            new PreprocessSpec(ResizeMode.LETTERBOX, 114, ChannelOrder.BGR, 1.0f, null, null);

    /** RF-DETR: stretch resize, RGB, ImageNet mean/std normalization. */
    public static final PreprocessSpec RFDETR_STRETCH =
            new PreprocessSpec(ResizeMode.STRETCH, 0, ChannelOrder.RGB, 1.0f / 255.0f,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});
}
