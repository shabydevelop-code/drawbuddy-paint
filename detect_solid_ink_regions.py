"""Find solid dark regions that need area preservation, not just thin strokes.

Conservative, read-only candidate extraction. No Paint bucket operations.
The source can be any raster illustration, not only the dog example.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from trace_reference import InkMask


class SolidInkRegionDetector:
    def __init__(self, threshold=95, core_radius=4, min_core_area=12,
                 min_region_area=35):
        self.threshold = threshold
        self.core_radius = core_radius
        self.min_core_area = min_core_area
        self.min_region_area = min_region_area

    def detect(self, image_path):
        mask = InkMask(self.threshold).from_image(image_path).astype(np.uint8)
        h, w = mask.shape
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # Pixels at least 'core_radius' away from white are robustly thick.
        cores = (distance >= self.core_radius).astype(np.uint8)
        n, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(
            cores, connectivity=8)
        accepted_cores = np.zeros_like(cores)
        for i in range(1, n):
            if core_stats[i, cv2.CC_STAT_AREA] >= self.min_core_area:
                accepted_cores[core_labels == i] = 1

        # Restrict growth to source ink; do not flood through thin connecting
        # strokes. Local thickness >= 2px keeps broad areas while avoiding
        # arbitrary object semantics.
        thick = ((distance >= 2.0) & (mask > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            thick, connectivity=8)
        candidates = []
        regions = np.zeros_like(mask)
        for i in range(1, count):
            region = labels == i
            if stats[i, cv2.CC_STAT_AREA] < self.min_region_area:
                continue
            if not np.any(region & (accepted_cores > 0)):
                continue
            # A source-dark candidate, not an approved Paint fill boundary.
            regions[region] = 1
            ys, xs = np.nonzero(region)
            candidates.append({
                "bounds": [int(xs.min()), int(ys.min()),
                           int(xs.max()), int(ys.max())],
                "area": int(region.sum()),
                "max_half_width_px": round(float(distance[region].max()), 2),
                "status": "REVIEW_SOLID_REGION"
            })
        return mask, regions, candidates

    def run(self, image_path, output_path, report_path):
        mask, regions, candidates = self.detect(image_path)
        display = np.full((*mask.shape, 3), 255, dtype=np.uint8)
        display[mask > 0] = (0, 0, 0)
        # Green means source-dark thick area worth preserving, not a safe
        # automatic fill instruction.
        display[regions > 0] = (85, 175, 65)
        contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (0, 110, 0), 1)
        report = {
            "source": str(image_path),
            "threshold": self.threshold,
            "core_radius_px": self.core_radius,
            "candidates": candidates,
            "count": len(candidates),
            "note": ("These are source-coordinate dark-area candidates only. "
                     "They may include thick outlines and joined shapes. "
                     "No Paint fill or drawing edit is authorized.")
        }
        output_path, report_path = Path(output_path), Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), display):
            raise RuntimeError(f"Cannot save {output_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Solid ink candidates: {len(candidates)}")
        for c in candidates:
            print(json.dumps(c))
        print(f"Saved: {output_path}, {report_path}; original drawing unchanged.")


def main():
    parser = argparse.ArgumentParser(description="Detect thick black areas in source artwork")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--output", type=Path, default=Path("output/solid-ink-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/solid-ink-report.json"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    SolidInkRegionDetector().run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
