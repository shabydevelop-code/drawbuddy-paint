"""Read-only diagnostic combining ink thickness and skeleton branch context.

Find broad ink interiors with distance-transform peaks; compare their local
thickness to the reference's typical thin stroke. Candidates are not fills.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from trace_reference import InkMask, ZhangSuenThinner


class InkSkeletonFusionAnalyzer:
    def __init__(self, threshold=95, min_area=12, relative_width=2.2):
        self.threshold = threshold
        self.min_area = min_area
        self.relative_width = relative_width

    def analyze(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        skeleton = ZhangSuenThinner().thin(ink)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        widths = 2.0 * distance[skeleton > 0]
        if not len(widths):
            raise ValueError("Reference contains no ink skeleton")
        # The lower quartile represents narrow strokes, avoiding a fixed
        # pixel threshold that would only work for one reference image.
        baseline = float(np.percentile(widths, 25))
        threshold = max(2.0, baseline * self.relative_width)
        thick = ((2.0 * distance >= threshold) & (ink > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            thick, connectivity=8)
        display = np.full((*ink.shape, 3), 255, np.uint8)
        display[ink > 0] = (0, 0, 0)
        # Blue is the skeleton; yellow is a candidate's broad ink interior.
        display[skeleton > 0] = (255, 0, 0)
        items = []
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if area < self.min_area:
                continue
            component = labels[y:y+h, x:x+w] == i
            local_skeleton = skeleton[y:y+h, x:x+w] > 0
            local_distance = distance[y:y+h, x:x+w]
            on_axis = component & local_skeleton
            axis_widths = 2.0 * local_distance[on_axis]
            max_width = float(2.0 * local_distance[component].max())
            # Skeleton support and actual ink area are both retained.
            # A long thick outline can score high; no auto-fill decision.
            support = int(np.count_nonzero(on_axis))
            aspect = max(w, h) / max(1, min(w, h))
            items.append({
                "bounds": [x, y, x+w-1, y+h-1],
                "broad_ink_area": area,
                "skeleton_support_pixels": support,
                "ink_area_per_skeleton_pixel": round(area/max(1, support), 2),
                "aspect_ratio": round(aspect, 3),
                "max_ink_width": round(max_width, 2),
                "median_axis_width": (round(float(np.median(axis_widths)), 2)
                                      if len(axis_widths) else None),
                "status": "REVIEW_BROAD_INK",
            })
            roi = display[y:y+h, x:x+w]
            roi[component] = (0, 200, 255)
            roi[on_axis] = (255, 0, 0)
        return display, {
            "reference": str(reference),
            "baseline_stroke_width": round(baseline, 2),
            "broad_ink_width_threshold": round(threshold, 2),
            "candidate_count": len(items),
            "candidates": items,
            "warning": "Diagnostic only. Broad outlines may be candidates; no automatic fill or drawing edits.",
        }

    def run(self, reference, output, report):
        display, result = self.analyze(reference)
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), display):
            raise RuntimeError(f"Cannot save {output}")
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"},
                         indent=2))
        print(f"Saved {output} and {report}; no Paint or drawing changes.")


def main():
    p = argparse.ArgumentParser(description="Compare source ink regions with skeleton")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/ink-fusion-diagnostic.png"))
    p.add_argument("--report", type=Path, default=Path("output/ink-fusion-report.json"))
    p.add_argument("--relative-width", type=float, default=2.2)
    p.add_argument("--publish-diagnostic", action="store_true")
    args = p.parse_args()
    InkSkeletonFusionAnalyzer(relative_width=args.relative_width).run(
        args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
