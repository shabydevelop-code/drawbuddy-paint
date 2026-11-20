"""Conservative, read-only reconstruction of broad ink candidates.

Grow from broad-ink seeds only inside original ink, stopping at thin necks
and at a fixed local distance. This is NOT semantic object extraction.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from trace_reference import InkMask, ZhangSuenThinner


class BoundedInkReconstructor:
    def __init__(self, threshold=95, relative_width=2.2,
                 min_seed_area=12, growth_factor=1.5):
        self.threshold = threshold
        self.relative_width = relative_width
        self.min_seed_area = min_seed_area
        self.growth_factor = growth_factor

    def analyze(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        skeleton = ZhangSuenThinner().thin(ink)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        widths = 2.0 * distance[skeleton > 0]
        if widths.size == 0:
            raise ValueError("Reference contains no ink skeleton")
        baseline = float(np.percentile(widths, 25))
        seed_halfwidth = max(1.0, baseline * self.relative_width / 2.0)
        seed_mask = ((distance >= seed_halfwidth) & (ink > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            seed_mask, connectivity=8)
        # The original ink mask is a hard constraint. A thin-neck barrier
        # blocks propagation onto neighboring ordinary outline strokes.
        min_halfwidth = max(0.8, baseline * 0.30)
        traversable = ((ink > 0) & (distance >= min_halfwidth)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        display = np.full((*ink.shape, 3), 255, np.uint8)
        display[ink > 0] = (0, 0, 0)
        display[skeleton > 0] = (255, 0, 0)
        occupied = np.zeros_like(ink, dtype=np.uint8)
        items = []
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if area < self.min_seed_area:
                continue
            # Local bound prevents following an entire connected contour.
            radius = max(1, int(np.ceil(baseline * self.growth_factor)))
            x0, y0 = max(0, x-radius), max(0, y-radius)
            x1, y1 = min(ink.shape[1], x+w+radius), min(ink.shape[0], y+h+radius)
            seed = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            allowed = traversable[y0:y1, x0:x1]
            grown = seed.copy()
            for _ in range(radius):
                grown = cv2.bitwise_and(cv2.dilate(grown, kernel), allowed)
                grown[seed > 0] = 1
            # Do not assign pixels to two separate candidates.
            local_occupied = occupied[y0:y1, x0:x1]
            overlap = int(np.count_nonzero((grown > 0) & (local_occupied > 0)))
            grown[local_occupied > 0] = 0
            local_occupied[grown > 0] = 1
            grown_area = int(np.count_nonzero(grown))
            roi = display[y0:y1, x0:x1]
            roi[grown > 0] = (255, 200, 0)  # cyan-ish reconstructed footprint
            roi[seed > 0] = (0, 220, 255)  # yellow original seed
            items.append({
                "seed_bounds": [x, y, x+w-1, y+h-1],
                "seed_area": area,
                "reconstructed_area": grown_area,
                "growth_ratio": round(grown_area / area, 3),
                "overlap_pixels_excluded": overlap,
                "max_growth_pixels": radius,
                "status": "REVIEW_RECONSTRUCTED_INK",
            })
        return display, {
            "reference": str(reference),
            "baseline_stroke_width": round(baseline, 3),
            "seed_halfwidth": round(seed_halfwidth, 3),
            "minimum_growth_halfwidth": round(min_halfwidth, 3),
            "candidate_count": len(items),
            "candidates": items,
            "warning": "Bounded reconstruction is not a verified semantic shape or a fill instruction. No drawing changes.",
        }

    def run(self, reference, output, report):
        display, result = self.analyze(reference)
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), display):
            raise RuntimeError(f"Cannot save {output}")
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
        print(f"Saved {output} and {report}; no drawing edits.")


def main():
    p = argparse.ArgumentParser(description="Bounded reconstruction of broad source ink")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/ink-reconstruction-diagnostic.png"))
    p.add_argument("--report", type=Path, default=Path("output/ink-reconstruction-report.json"))
    p.add_argument("--publish-diagnostic", action="store_true")
    args = p.parse_args()
    BoundedInkReconstructor().run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
