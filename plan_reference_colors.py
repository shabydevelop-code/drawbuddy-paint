"""Plan source-matched flat fills for any stroke-only drawing, without editing Paint.

Usage: python plan_reference_colors.py --drawing drawings/dog-closure-test.json
Produces output/color-plan.json and output/color-preview.png.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor
from validate_fill_regions import FillRegionValidator


class ReferenceColorPlanner:
    def __init__(self, min_area=100, minimum_coverage=0.65, max_spread=32):
        self.auditor = FillRegionAuditor(min_area=min_area)
        self.aligner = FillRegionValidator()
        self.min_area = min_area
        self.minimum_coverage = minimum_coverage
        self.max_spread = max_spread

    def plan(self, reference, drawing):
        ink, offset = self.auditor.rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = self.aligner.reference_to_canvas(reference, drawing, (w, h), offset)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (ink == 0).astype(np.uint8), connectivity=4)
        preview = np.full((h, w, 3), 255, dtype=np.uint8)
        fills = []
        review = []
        for i in range(1, count):
            x, y, rw, rh, area = map(int, stats[i])
            if area < self.min_area:
                continue
            if x == 0 or y == 0 or x+rw == w or y+rh == h:
                continue
            component = (labels == i).astype(np.uint8)
            inner = cv2.erode(component, np.ones((5, 5), np.uint8)) > 0
            # Reject ink, background and antialiasing when sampling source colors.
            pixels = aligned[inner]
            foreground = np.min(pixels, axis=1) < 235
            selected = pixels[foreground]
            coverage = float(len(selected)) / max(1, int(inner.sum()))
            if len(selected) < 20 or coverage < self.minimum_coverage:
                review.append({"region": i, "reason": "insufficient source foreground",
                               "coverage": round(coverage, 3)})
                continue
            # Median is robust to outlines and minor highlights.
            rgb = np.median(selected[:, ::-1], axis=0)
            deviations = np.linalg.norm(selected[:, ::-1].astype(float)-rgb, axis=1)
            spread = float(np.percentile(deviations, 75))
            if spread > self.max_spread:
                review.append({"region": i, "reason": "multiple source colors or shading",
                               "spread": round(spread, 1)})
                continue
            # Maximum distance from ink for a safe bucket seed.
            distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
            seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            if distance[seed_y, seed_x] < 3:
                review.append({"region": i, "reason": "no safe fill seed"})
                continue
            color = [int(round(c)) for c in rgb]
            fills.append({"region": i, "seed": [int(seed_x+offset[0]), int(seed_y+offset[1])],
                          "rgb": color, "area": area, "coverage": round(coverage, 3),
                          "spread": round(spread, 1)})
            preview[labels == i] = color[::-1]
        preview[ink > 0] = (0, 0, 0)
        return {"fills": fills, "review": review, "alignment": "approximate"}, preview

    def run(self, reference_path, drawing_path, plan_path, preview_path):
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(reference_path)
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        plan, preview = self.plan(reference, drawing)
        for path in (plan_path, preview_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(plan_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Cannot write preview: {preview_path}")
        print(f"Proposed fills: {len(plan['fills'])}; review needed: {len(plan['review'])}")
        print(f"Saved: {plan_path} and {preview_path}")
        print("No Paint actions performed. Preview uses approximate alignment; review before filling.")


def main():
    parser = argparse.ArgumentParser(description="Generate source-color fill proposal")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--plan", type=Path, default=Path("output/color-plan.json"))
    parser.add_argument("--preview", type=Path, default=Path("output/color-preview.png"))
    parser.add_argument("--publish-preview", action="store_true")
    args = parser.parse_args()
    ReferenceColorPlanner().run(args.reference, args.drawing, args.plan, args.preview)
    if args.publish_preview:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.preview)


if __name__ == "__main__":
    main()
