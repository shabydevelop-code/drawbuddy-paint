"""Audit potential Paint bucket-fill regions without changing drawing paths.

Usage: python audit_fill_regions.py drawings/dog.json
This is a raster approximation, not a guarantee of Paint fill behavior.
No report files or history are created.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class FillRegionAuditor:
    """Find enclosed white components in a simulated line drawing."""

    def __init__(self, margin=12, min_area=30, line_width=1):
        self.margin = margin
        self.min_area = min_area
        self.line_width = line_width

    def rasterize(self, steps):
        points = [p for step in steps if step.get("tool") in ("stroke", "closed_path")
                  for p in step.get("points", [])]
        if not points:
            raise ValueError("Drawing has no stroke points")
        xs, ys = zip(*points)
        x0, y0 = min(xs) - self.margin, min(ys) - self.margin
        x1, y1 = max(xs) + self.margin, max(ys) + self.margin
        canvas = np.zeros((y1-y0+1, x1-x0+1), dtype=np.uint8)
        for step in steps:
            if step.get("tool") not in ("stroke", "closed_path"):
                continue
            coords = step.get("points", [])
            if len(coords) < 2:
                continue
            local = np.asarray([[x-x0, y-y0] for x, y in coords], np.int32)
            # Width is an approximation; actual Paint Pencil rasterization varies.
            thickness = max(self.line_width, int(step.get("width", 1)))
            cv2.polylines(canvas, [local], step["tool"] == "closed_path",
                          255, thickness=thickness, lineType=cv2.LINE_8)
        return canvas, (x0, y0)

    def analyze(self, drawing):
        ink, (x0, y0) = self.rasterize(drawing["steps"])
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (ink == 0).astype(np.uint8), connectivity=4)
        h, w = ink.shape
        regions = []
        exterior_area = 0
        for i in range(1, count):
            x, y, width, height, area = map(int, stats[i])
            touches_edge = (x == 0 or y == 0 or
                            x + width == w or y + height == h)
            if touches_edge:
                exterior_area += area
                continue
            if area >= self.min_area:
                cx, cy = centers[i]
                regions.append((area, round(cx+x0), round(cy+y0)))
        regions.sort(reverse=True)
        return regions, exterior_area, (w, h)

    def report(self, drawing):
        regions, exterior, dimensions = self.analyze(drawing)
        print(f"Drawing: {drawing.get('name', '(unnamed)')}")
        print(f"Raster: {dimensions[0]}x{dimensions[1]}")
        print(f"Enclosed candidate regions (area >= {self.min_area}): {len(regions)}")
        print(f"Exterior/background white pixels: {exterior}")
        print("Largest enclosed regions (area px; approximate Paint x,y fill seed):")
        for area, x, y in regions[:30]:
            print(f"  {area:>7} px  seed=({x}, {y})")
        print("WARNING: This cannot identify intended color regions or prove")
        print("that Paint's bucket will not leak. Open boundaries merge with")
        print("the exterior and are therefore not listed as enclosed regions.")


def main():
    parser = argparse.ArgumentParser(description="Audit closed fill regions in DrawBuddy")
    parser.add_argument("drawing", type=Path, nargs="?",
                        default=Path("drawings/dog.json"))
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--line-width", type=int, default=1)
    args = parser.parse_args()
    if args.min_area < 1 or args.line_width < 1:
        parser.error("--min-area and --line-width must be positive")
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    FillRegionAuditor(min_area=args.min_area,
                      line_width=args.line_width).report(drawing)


if __name__ == "__main__":
    main()
