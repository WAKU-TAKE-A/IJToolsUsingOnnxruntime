// RenderOptions.java - Display/output options for ResultsRenderer
package io.github.wakutakea.ijtoolsusingonnxruntime;

/**
 * Everything {@link ResultsRenderer} needs to know about how results
 * should be presented. This replaces the long boolean argument lists
 * the RF-DETR entry point used to take (P1-4): each caller now states
 * its options by name.
 *
 * @param slot              slot number written to the results table
 * @param maskThreshold     mask binarization threshold (segmentation)
 * @param kptThreshold      keypoint visibility threshold (pose)
 * @param showResultsTable  write rows to the Results table
 * @param showRoiManager    add ROIs to the ROI Manager
 * @param showOverlay       add ROIs to the image overlay
 * @param overlayAlpha      fill alpha for mask ROIs (0-255)
 * @param exportMask        open a union mask image (segmentation)
 * @param enableRefreshData clear the table/ROIs/overlay before writing
 * @param enableLog         log a summary line (used by the RF-DETR entry point)
 */
public record RenderOptions(int slot, double maskThreshold, double kptThreshold,
                            boolean showResultsTable, boolean showRoiManager, boolean showOverlay,
                            int overlayAlpha, boolean exportMask,
                            boolean enableRefreshData, boolean enableLog) {

    public static Builder builder(int slot) {
        return new Builder(slot);
    }

    /** Builder with the historical RF-DETR defaults (mask/kpt 0.5, alpha 80, displays off). */
    public static final class Builder {
        private final int slot;
        private double maskThreshold = 0.5;
        private double kptThreshold = 0.5;
        private boolean showResultsTable;
        private boolean showRoiManager;
        private boolean showOverlay;
        private int overlayAlpha = 80;
        private boolean exportMask;
        private boolean enableRefreshData;
        private boolean enableLog;

        private Builder(int slot) {
            this.slot = slot;
        }

        public Builder maskThreshold(double v) { this.maskThreshold = v; return this; }
        public Builder kptThreshold(double v) { this.kptThreshold = v; return this; }
        public Builder showResultsTable(boolean v) { this.showResultsTable = v; return this; }
        public Builder showRoiManager(boolean v) { this.showRoiManager = v; return this; }
        public Builder showOverlay(boolean v) { this.showOverlay = v; return this; }
        public Builder overlayAlpha(int v) { this.overlayAlpha = v; return this; }
        public Builder exportMask(boolean v) { this.exportMask = v; return this; }
        public Builder enableRefreshData(boolean v) { this.enableRefreshData = v; return this; }
        public Builder enableLog(boolean v) { this.enableLog = v; return this; }

        public RenderOptions build() {
            return new RenderOptions(slot, maskThreshold, kptThreshold, showResultsTable,
                    showRoiManager, showOverlay, overlayAlpha, exportMask, enableRefreshData, enableLog);
        }
    }
}
