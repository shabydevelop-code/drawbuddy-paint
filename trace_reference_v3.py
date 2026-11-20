"""Reference-traced dog v3: original centerlines, measured widths, human-readable order.

Usage: python trace_reference_v3.py assets/dog-reference.png
Requires: opencv-python, numpy.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from trace_reference import PaintCanvasMapper
from trace_reference_v2 import InkMask, ZhangSuenThinner, SkeletonPathTracer


class StrokeWidthAnalyzer:
    """Estimate source ink diameter along each centerline using a distance map."""

    def __init__(self, mask):
        self.distances = cv2.distanceTransform(
            (mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    def measure(self, points):
        h, w = self.distances.shape
        values = [2.0 * float(self.distances[y, x])
                  for x, y in points if 0 <= x < w and 0 <= y < h]
        return float(np.median(values)) if values else 1.0

    @staticmethod
    def assign_widths(paths, diameters):
        """Map measured widths to Paint's discrete 1..4 sizes.

        Quantile bins preserve relative ink weight without requiring
        knowledge of the original image's resolution.
        """
        if not diameters:
            return []
        q25, q50, q80 = np.percentile(diameters, [25, 50, 80])
        if q80 - q25 < 0.7:
            return [2] * len(paths)
        return [1 if d < q25 else 2 if d < q50 else 3 if d < q80 else 4
                for d in diameters]


class DrawingSequencePlanner:
    """Plan a stable top-to-bottom visual narrative, then minimize pen-up moves.

    Region rules are based on normalized source geometry, not the screen
    coordinates. They are deliberately approximate, never alter geometry.
    """

    REGION_NAMES = (
        "Head and face outline", "Ears", "Eyes nose and mouth",
        "Collar and tag", "Body and chest", "Legs and paws", "Tail"
    )

    @staticmethod
    def classify(points, bounds):
        x0, y0, x1, y1 = bounds
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = ((min(xs) + max(xs)) / 2 - x0) / max(1, x1 - x0)
        cy = ((min(ys) + max(ys)) / 2 - y0) / max(1, y1 - y0)
        span_x = (max(xs) - min(xs)) / max(1, x1 - x0)
        span_y = (max(ys) - min(ys)) / max(1, y1 - y0)
        if cy < 0.48:
            if cx < 0.25 or cx > 0.79:
                return 1
            if cy > 0.36 and 0.37 < cx < 0.76 and span_y < 0.20:
                return 3
            if span_x < 0.22 and span_y < 0.25:
                return 2
            return 0
        if cx < 0.26 and cy > 0.50:
            return 6
        if cy > 0.79:
            return 5
        if cy < 0.55 and 0.35 < cx < 0.76 and span_y < 0.18:
            return 3
        return 4

    @staticmethod
    def _distance(a, b):
        return math.dist(a, b)

    def plan(self, paths, widths):
        coords = [p for path in paths for p in path]
        bounds = (min(p[0] for p in coords), min(p[1] for p in coords),
                  max(p[0] for p in coords), max(p[1] for p in coords))
        groups = [[] for _ in self.REGION_NAMES]
        for index, path in enumerate(paths):
            groups[self.classify(path, bounds)].append(index)
        result = []
        cursor = (bounds[0] + (bounds[2]-bounds[0]) * .5, bounds[1])
        for region, members in enumerate(groups):
            remaining = set(members)
            while remaining:
                # Evaluate both orientations of every path; avoid needless jumps.
                options = []
                for i in remaining:
                    path = paths[i]
                    options.append((self._distance(cursor, path[0]), i, False))
                    options.append((self._distance(cursor, path[-1]), i, True))
                _, index, reverse = min(options)
                path = list(reversed(paths[index])) if reverse else paths[index]
                result.append((region, index, path, widths[index]))
                cursor = path[-1]
                remaining.remove(index)
        return result


class V3DrawingExporter:
    """Emit only existing DrawingEngine stroke commands."""

    def save(self, plan, mapped_paths, output):
        steps = []
        for sequence, (region, original_index, _, width) in enumerate(plan, 1):
            points = mapped_paths[sequence - 1]
            if len(points) < 2 or len(set(map(tuple, points))) < 2:
                continue
            steps.append({
                "name": f"{DrawingSequencePlanner.REGION_NAMES[region]} {sequence:03d}",
                "tool": "stroke", "color": "black", "smooth": False,
                "width": int(width), "points": points
            })
        if not steps:
            raise ValueError("No drawable paths")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "name": "Dog traced v3 - measured widths and visual drawing order",
            "steps": steps
        }, indent=2), encoding="utf-8")
        counts = {i: sum(1 for s in steps if s["width"] == i) for i in range(1, 5)}
        print(f"Saved {len(steps)} ordered strokes to {output}; Paint widths: {counts}")


class V3TracingPipeline:
    def __init__(self, threshold=95, min_length=12):
        self.ink = InkMask(threshold)
        self.thinner = ZhangSuenThinner()
        self.tracer = SkeletonPathTracer(min_length)
        self.mapper = PaintCanvasMapper()
        self.planner = DrawingSequencePlanner()
        self.exporter = V3DrawingExporter()

    def run(self, source, output):
        mask = self.ink.from_image(source)
        paths = self.tracer.trace(self.thinner.thin(mask))
        if not paths:
            raise ValueError("No paths detected; try --threshold")
        analyzer = StrokeWidthAnalyzer(mask)
        diameters = [analyzer.measure(p) for p in paths]
        widths = analyzer.assign_widths(paths, diameters)
        plan = self.planner.plan(paths, widths)
        # Map after ordering to preserve exactly the same scale as v2.
        mapped = self.mapper.map([entry[2] for entry in plan])
        self.exporter.save(plan, mapped, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("drawings/dog-traced-v3.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=12)
    args = parser.parse_args()
    V3TracingPipeline(args.threshold, args.min_length).run(
        args.image, args.output)


if __name__ == "__main__":
    main()
