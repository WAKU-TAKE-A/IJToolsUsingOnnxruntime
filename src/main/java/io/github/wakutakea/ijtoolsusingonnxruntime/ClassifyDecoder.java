// ClassifyDecoder.java - Decodes classification output into ranked entries
package io.github.wakutakea.ijtoolsusingonnxruntime;

import java.util.ArrayList;
import java.util.List;

/**
 * Classification decoder. Classification is the declared exception to
 * the unified {@link Detection} representation: it produces ranked
 * {@link Entry} records instead of bounding boxes. The top-K selection
 * is the former plugin's selection sort, ported unchanged (including
 * its tie ordering).
 */
public final class ClassifyDecoder {
    private ClassifyDecoder() {}

    /** One classification result: class index and its score. Rank is the position in the returned list. */
    public record Entry(int classId, float score) {}

    /** Returns up to {@code topK} entries, best first. */
    public static List<Entry> decode(float[] probs, int topK) {
        int[] idx = getTopK(probs, topK);
        List<Entry> out = new ArrayList<>(idx.length);
        for (int i : idx) {
            out.add(new Entry(i, probs[i]));
        }
        return out;
    }

    /** Top-K indices by selection sort (moved from ORT_2nd_Classify, unchanged). */
    private static int[] getTopK(float[] probs, int k) {
        int n = probs.length;
        int actualK = Math.min(k, n);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        // Simple selection sort for top-K (fine for typical class counts)
        for (int i = 0; i < actualK; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (probs[indices[j]] > probs[indices[maxIdx]]) {
                    maxIdx = j;
                }
            }
            int temp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = temp;
        }

        int[] result = new int[actualK];
        System.arraycopy(indices, 0, result, 0, actualK);
        return result;
    }
}
