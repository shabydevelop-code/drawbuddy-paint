"""Read-only audit of ink thresholding and skeleton thinning.

Separates uncertain source pixels from pixels lost or added by thinning.
Does not change drawing definitions or Paint.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from trace_reference import ZhangSuenThinner


class InkPipelineAudit:
    def __init__(self, threshold=95, band=25):
        if not 0 <= threshold <= 255 or not 0 <= band <= 127:
            raise ValueError("threshold must be 0..255 and band 0..127")
        self.threshold, self.band = threshold, band

    @staticmethod
    def components(mask):
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        return int(count-1), int(sum(stats[i, cv2.CC_STAT_AREA] < 12
                                    for i in range(1, count)))

    def run(self, reference_path, diagnostic_path, report_path):
        image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(reference_path)
        darkness = np.max(image, axis=2)
        ink = darkness < self.threshold
        lower = darkness < max(0, self.threshold-self.band)
        upper = darkness < min(256, self.threshold+self.band)
        uncertain = upper & ~lower
        skeleton = ZhangSuenThinner().thin(ink.astype(np.uint8)) > 0
        # A skeleton pixel should always be a subset of thresholded ink.
        outside = skeleton & ~ink
        # Quantify how well the skeleton represents the ink at a 2px tolerance.
        dilated = cv2.dilate(skeleton.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        unreached = ink & ~dilated
        source_components, source_small = self.components(ink)
        skeleton_components, skeleton_small = self.components(skeleton)
        canvas = np.full((*ink.shape, 3), 255, np.uint8)
        canvas[ink] = (205, 205, 205)
        canvas[skeleton] = (0, 0, 0)
        canvas[uncertain & ~skeleton] = (0, 200, 255)
        canvas[unreached] = (0, 0, 255)
        canvas[outside] = (255, 0, 0)
        report = {
            "threshold": self.threshold, "uncertainty_band": self.band,
            "source_ink_pixels": int(ink.sum()),
            "uncertain_threshold_pixels": int(uncertain.sum()),
            "skeleton_pixels": int(skeleton.sum()),
            "ink_pixels_farther_than_2px_from_skeleton": int(unreached.sum()),
            "skeleton_pixels_outside_ink": int(outside.sum()),
            "ink_components_8_connected": source_components,
            "skeleton_components_8_connected": skeleton_components,
            "small_ink_components_under_12px": source_small,
            "small_skeleton_components_under_12px": skeleton_small,
            "note": "Component changes are indicators, not proof of topology errors; threshold uncertainty is not necessarily missing ink."
        }
        diagnostic_path, report_path = Path(diagnostic_path), Path(report_path)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(diagnostic_path), canvas):
            raise RuntimeError(f"Cannot save {diagnostic_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("No changes to drawing or Paint.")


def main():
    parser = argparse.ArgumentParser(description="Audit ink threshold and thinning separately")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--output", type=Path, default=Path("output/ink-pipeline-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/ink-pipeline-report.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--band", type=int, default=25)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    InkPipelineAudit(args.threshold, args.band).run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
