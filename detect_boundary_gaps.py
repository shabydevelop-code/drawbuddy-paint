"""Read-only detection of color-boundary endpoints near existing ink.

Usage: python detect_boundary_gaps.py drawings/dog.json
Creates output/gap-diagnostic.png; does not change drawing paths.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor


class BoundaryGapDetector:
    """Identify short candidate endpoint-to-ink gaps, without auto-joining."""

    def __init__(self, max_gap=12, min_gap=2, min_separation=5):
        self.max_gap = max_gap
        self.min_gap = min_gap
        self.min_separation = min_separation
        self.auditor = FillRegionAuditor()

    def detect(self, drawing):
        ink_steps = [s for s in drawing["steps"]
                     if s.get("tool") in ("stroke", "closed_path")
                     and s.get("name", "").startswith("Ink")]
        color_steps = [s for s in drawing["steps"]
                       if s.get("tool") == "stroke"
                       and s.get("name", "").startswith("Color boundary")]
        if not ink_steps:
            raise ValueError("No named Ink strokes; cannot identify target lines")
        if not color_steps:
            raise ValueError("No named Color boundary strokes")
        all_steps = ink_steps + color_steps
        raster, offset = self.auditor.rasterize(all_steps)
        ink, _ = self.auditor.rasterize(ink_steps)
        # Both rasterizers share the same origin only if their bounding boxes match.
        # Rebuild the ink mask on the combined coordinate system.
        ink = np.zeros_like(raster)
        ox, oy = offset
        for step in ink_steps:
            pts = np.asarray([[x-ox, y-oy] for x, y in step["points"]],
                             dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(ink, [pts], False, 255, 1)
        free = (ink == 0).astype(np.uint8)
        distance, nearest = cv2.distanceTransformWithLabels(
            free, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        h, w = ink.shape
        ink_y, ink_x = np.nonzero(ink)
        if not len(ink_x):
            raise ValueError("Ink raster is empty")
        # Build a mapping from distance-transform labels to ink coordinates.
        zero_labels = nearest[ink_y, ink_x]
        targets = dict(zip(zero_labels.tolist(),
                           zip(ink_x.tolist(), ink_y.tolist())))
        gaps = []
        seen = set()
        for step in color_steps:
            points = step["points"]
            if len(points) < 2 or points[0] == points[-1]:
                continue
            for end in (points[0], points[-1]):
                x, y = int(end[0]-ox), int(end[1]-oy)
                if not (0 <= x < w and 0 <= y < h):
                    continue
                d = float(distance[y, x])
                if not self.min_gap <= d <= self.max_gap:
                    continue
                target = targets.get(int(nearest[y, x]))
                if target is None:
                    continue
                tx, ty = target
                # Avoid repeatedly reporting the same junction.
                key = (round(x/self.min_separation), round(y/self.min_separation),
                       round(tx/self.min_separation), round(ty/self.min_separation))
                if key in seen:
                    continue
                seen.add(key)
                gaps.append((round(d, 1), (x+ox, y+oy), (tx+ox, ty+oy)))
        gaps.sort(key=lambda item: item[0])
        return gaps, raster, offset

    @staticmethod
    def save(path, raster, offset, gaps):
        image = np.full((*raster.shape, 3), 255, dtype=np.uint8)
        image[raster > 0] = (0, 0, 0)
        ox, oy = offset
        for distance, start, end in gaps:
            a = (int(start[0]-ox), int(start[1]-oy))
            b = (int(end[0]-ox), int(end[1]-oy))
            cv2.line(image, a, b, (255, 0, 255), 1, cv2.LINE_8)
            cv2.circle(image, a, 3, (0, 0, 255), -1)
            cv2.circle(image, b, 2, (255, 0, 0), -1)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Cannot save {path}")
        print(f"Saved: {path}")

    def run(self, drawing, output):
        gaps, raster, offset = self.detect(drawing)
        print(f"Candidate color-endpoint-to-ink gaps: {len(gaps)}")
        for d, start, end in gaps[:50]:
            print(f"  distance={d:4.1f}px endpoint={start} ink_target={end}")
        self.save(output, raster, offset, gaps)
        print("Candidate only: nearby ink may belong to an unrelated contour.")
        print("No drawing changes or automatic joins were made.")


def main():
    parser = argparse.ArgumentParser(description="Find possible color-boundary gaps")
    parser.add_argument("drawing", nargs="?", type=Path,
                        default=Path("drawings/dog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/gap-diagnostic.png"))
    parser.add_argument("--max-gap", type=float, default=12)
    parser.add_argument("--publish-diagnostic", action="store_true", help="Publish diagnostic image to GitHub")
    args = parser.parse_args()
    if not 2 <= args.max_gap <= 50:
        parser.error("--max-gap must be between 2 and 50")
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    BoundaryGapDetector(max_gap=args.max_gap).run(drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
