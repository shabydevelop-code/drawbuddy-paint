"""Audit loss of source ink during thinning, tracing and path simplification.

Read-only: compares rasterized raw skeleton paths against simplified paths
in source-image coordinates, with no Paint mapping or color boundaries.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from trace_reference import InkMask, ZhangSuenThinner, SkeletonPathTracer


class InkSimplificationAudit:
    def __init__(self, threshold=95, tolerance=1):
        self.threshold = threshold
        self.tolerance = tolerance

    @staticmethod
    def rasterize(paths, shape):
        canvas = np.zeros(shape, dtype=np.uint8)
        for path in paths:
            if len(path) > 1:
                cv2.polylines(canvas, [np.asarray(path, np.int32)],
                              False, 255, 1, cv2.LINE_8)
        return canvas > 0

    def run(self, reference_path, output_path, report_path):
        mask = InkMask(self.threshold).from_image(reference_path)
        skeleton = ZhangSuenThinner().thin(mask) > 0
        # Raw and simplified use the exact same tracer and topology.
        raw_paths = SkeletonPathTracer(min_length=2, simplify=0).trace(skeleton.astype(np.uint8))
        simplified_paths = SkeletonPathTracer(min_length=2, simplify=0.65).trace(skeleton.astype(np.uint8))
        raw = self.rasterize(raw_paths, mask.shape)
        simplified = self.rasterize(simplified_paths, mask.shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (2*self.tolerance+1, 2*self.tolerance+1))
        near_simple = cv2.dilate(simplified.astype(np.uint8), kernel) > 0
        near_raw = cv2.dilate(raw.astype(np.uint8), kernel) > 0
        missing = raw & ~near_simple
        extra = simplified & ~near_raw
        image = np.full((*mask.shape, 3), 255, np.uint8)
        image[mask > 0] = (220, 220, 220)
        image[raw] = (0, 0, 0)
        image[missing] = (0, 0, 255)
        image[extra] = (255, 0, 0)
        report = {
            "raw_paths": len(raw_paths), "simplified_paths": len(simplified_paths),
            "raw_raster_pixels": int(raw.sum()),
            "simplified_raster_pixels": int(simplified.sum()),
            "raw_pixels_missing_after_simplification": int(missing.sum()),
            "simplified_pixels_outside_raw_tolerance": int(extra.sum()),
            "tolerance_source_pixels": self.tolerance,
            "note": "Measures simplification against traced skeleton; does not validate thinning or source ink threshold."
        }
        output_path, report_path = Path(output_path), Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Cannot save {output_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("No drawing or Paint changes.")


def main():
    parser = argparse.ArgumentParser(description="Measure detail loss from ink path simplification")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--output", type=Path, default=Path("output/ink-simplification-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/ink-simplification-report.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.tolerance <= 10:
        parser.error("--tolerance must be between 0 and 10")
    InkSimplificationAudit(args.threshold, args.tolerance).run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
