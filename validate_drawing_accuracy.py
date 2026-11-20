"""Compare reference ink, planned stroke raster, and optional Paint screenshot.

Read-only diagnostic. Uses foreground bounding-box alignment; reports
approximate discrepancies, not a proof of incorrect individual strokes.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_fill_regions import FillRegionAuditor


class DrawingAccuracyValidator:
    def __init__(self, threshold=95, tolerance=3):
        self.threshold = threshold
        self.tolerance = tolerance

    def ink(self, image):
        return (np.max(image, axis=2) < self.threshold).astype(np.uint8)

    @staticmethod
    def crop_and_fit(mask, shape):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("No dark strokes found for alignment")
        crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        h, w = shape
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    def compare(self, reference, target):
        radius = self.tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        near_reference = cv2.dilate(reference.astype(np.uint8), kernel) > 0
        near_target = cv2.dilate(target.astype(np.uint8), kernel) > 0
        missing = reference & ~near_target
        extra = target & ~near_reference
        total_ref, total_target = int(reference.sum()), int(target.sum())
        return missing, extra, {
            "reference_ink_pixels": total_ref,
            "target_ink_pixels": total_target,
            "missing_reference_pixels": int(missing.sum()),
            "extra_target_pixels": int(extra.sum()),
            "reference_coverage": round(1-int(missing.sum())/max(1,total_ref), 4),
            "target_precision": round(1-int(extra.sum())/max(1,total_target), 4),
            "tolerance_pixels": radius,
            "alignment": "foreground bounding boxes (approximate)"
        }

    def run(self, reference_path, drawing_path, output_path, report_path, paint_path=None):
        reference_image = cv2.imread(str(reference_path))
        if reference_image is None:
            raise FileNotFoundError(reference_path)
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        raster, _ = FillRegionAuditor().rasterize(drawing["steps"])
        planned = raster > 0
        # Compare only tight ink extents; avoid canvas-margin distortion.
        ys, xs = np.nonzero(planned)
        if len(xs) == 0:
            raise ValueError("Drawing has no visible ink")
        target = planned[ys.min():ys.max()+1, xs.min():xs.max()+1]
        source = self.crop_and_fit(self.ink(reference_image), target.shape)
        missing, extra, metrics = self.compare(source, target)
        image = np.full((*target.shape, 3), 255, np.uint8)
        image[source & target] = (0, 0, 0)
        image[source & ~target] = (0, 0, 0)
        image[target & ~source] = (0, 0, 0)
        image[missing] = (0, 0, 255)  # red: source ink not covered
        image[extra] = (255, 0, 0)  # blue: drawing ink not in source
        metrics["planned_vs_reference"] = dict(metrics)
        if paint_path is not None:
            screenshot = cv2.imread(str(paint_path))
            if screenshot is None:
                raise FileNotFoundError(paint_path)
            actual = self.crop_and_fit(self.ink(screenshot), target.shape)
            _, _, actual_metrics = self.compare(target, actual)
            metrics["paint_vs_planned"] = actual_metrics
            # Paint differences are evaluated separately, not conflated with tracing.
        header = np.full((65, image.shape[1], 3), 255, np.uint8)
        cv2.putText(header, "RED: missing source ink | BLUE: extra planned ink",
                    (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)
        cv2.putText(header, "Approximate bbox alignment; inspect before fixing",
                    (3, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)
        output_path, report_path = Path(output_path), Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), np.vstack((header, image))):
            raise RuntimeError(f"Cannot save {output_path}")
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print("No drawing or Paint modifications.")


def main():
    parser = argparse.ArgumentParser(description="Read-only geometric accuracy check")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-leak-test.json"))
    parser.add_argument("--paint", type=Path, help="Optional local Paint screenshot")
    parser.add_argument("--output", type=Path, default=Path("output/accuracy-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/accuracy-report.json"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    DrawingAccuracyValidator().run(args.reference, args.drawing, args.output, args.report, args.paint)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
