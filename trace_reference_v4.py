"""DrawBuddy v4: conservative gap repair, paint-leak audit and thinner strokes.

Usage: python trace_reference_v4.py assets/dog-reference.png
The audit is predictive; verify real Paint output before bucket filling.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from trace_reference import PaintCanvasMapper
from trace_reference_v2 import InkMask, ZhangSuenThinner, SkeletonPathTracer
from trace_reference_v3 import DrawingSequencePlanner, StrokeWidthAnalyzer


class BoundaryRepair:
    """Snap short dangling endpoints to nearby ink paths, never arbitrary regions."""

    def __init__(self, max_gap=3.0):
        self.max_gap = max_gap

    @staticmethod
    def _projection(p, a, b):
        vx, vy = b[0]-a[0], b[1]-a[1]
        length2 = vx*vx + vy*vy
        if not length2:
            return tuple(a), 0.0
        t = max(0.0, min(1.0, ((p[0]-a[0])*vx+(p[1]-a[1])*vy)/length2))
        return (a[0]+t*vx, a[1]+t*vy), t

    def repair(self, paths, mask):
        """Append separate short bridges. Do not alter traced paths or close loops blindly."""
        h, w = mask.shape
        # Only endpoints of open paths may need bridging.
        endpoints = [(i, side, tuple(path[0] if side == 0 else path[-1]))
                     for i, path in enumerate(paths) if len(path) > 1
                     and path[0] != path[-1] for side in (0, 1)]
        bridges, used = [], set()
        for i, side, p in endpoints:
            if (i, side) in used:
                continue
            candidates = []
            for j, other in enumerate(paths):
                if i == j:
                    continue
                for a, b in zip(other, other[1:]):
                    projected, _ = self._projection(p, a, b)
                    q = (int(round(projected[0])), int(round(projected[1])))
                    distance = math.dist(p, q)
                    if 1.1 <= distance <= self.max_gap:
                        candidates.append((distance, j, q))
            if not candidates:
                continue
            distance, j, q = min(candidates)
            # Require both endpoints to lie near actual source ink, not across
            # an intentional whitespace gap. The source can be anti-aliased.
            samples = np.linspace(p, q, max(3, int(math.ceil(distance * 3))))
            if any(not (0 <= int(round(x)) < w and 0 <= int(round(y)) < h)
                   for x, y in samples):
                continue
            near_ink = cv2.dilate(mask.astype(np.uint8),
                                 np.ones((3, 3), np.uint8))
            if not all(near_ink[int(round(y)), int(round(x))]
                       for x, y in samples):
                continue
            bridges.append([list(p), list(q)])
            used.add((i, side))
        return bridges


class PaintLeakAuditor:
    """Rasterize planned strokes and report flood-fill regions and dangling ends."""

    def inspect(self, paths, widths, shape):
        h, w = shape
        ink = np.zeros((h, w), dtype=np.uint8)
        for path, width in zip(paths, widths):
            points = np.asarray(path, dtype=np.int32)
            if len(points) >= 2:
                cv2.polylines(ink, [points], False, 255, max(1, int(width)),
                              lineType=cv2.LINE_8)
        # White connected to the canvas border represents a bucket leak.
        free = (ink == 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
        border = set(np.unique(np.concatenate(
            (labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
        enclosed = sorted(
            (int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, count)
             if k not in border), reverse=True)
        return {"enclosed_regions": len(enclosed),
                "largest_enclosed_areas": enclosed[:15],
                "exterior_connected": True,
                "warning": ("An enclosed-region count does not prove that all "
                            "intended color regions are sealed. Validate each "
                            "bucket seed on the final Paint screenshot.")}


class ConservativeWidthMapper:
    """Use only Paint sizes 1 and 2 to avoid the heavy pixelated v3 outline."""

    def assign(self, paths, mask):
        analyzer = StrokeWidthAnalyzer(mask)
        measured = np.asarray([analyzer.measure(path) for path in paths])
        if len(measured) < 2 or float(np.ptp(measured)) < 0.7:
            return [1] * len(paths)
        cutoff = float(np.percentile(measured, 70))
        return [2 if d > cutoff else 1 for d in measured]


class V4TracingPipeline:
    def __init__(self, threshold=95, min_length=12, max_gap=3):
        self.ink = InkMask(threshold)
        self.thinner = ZhangSuenThinner()
        self.tracer = SkeletonPathTracer(min_length)
        self.repair = BoundaryRepair(max_gap)
        self.widths = ConservativeWidthMapper()
        self.planner = DrawingSequencePlanner()
        self.mapper = PaintCanvasMapper()
        self.audit = PaintLeakAuditor()

    def run(self, source, output, report):
        mask = self.ink.from_image(source)
        paths = self.tracer.trace(self.thinner.thin(mask))
        if not paths:
            raise ValueError("No ink paths found")
        bridges = self.repair.repair(paths, mask)
        paths.extend(bridges)
        widths = self.widths.assign(paths, mask)
        plan = self.planner.plan(paths, widths)
        ordered = [p for _, _, p, _ in plan]
        mapped = self.mapper.map(ordered)
        steps = []
        for index, ((region, _, _, width), points) in enumerate(zip(plan, mapped), 1):
            if len(set(map(tuple, points))) < 2:
                continue
            steps.append({
                "name": f"{self.planner.REGION_NAMES[region]} {index:03d}",
                "tool": "stroke", "color": "black", "smooth": False,
                "width": int(width), "points": points
            })
        if not steps:
            raise ValueError("No drawable strokes")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "name": "Dog traced v4 - conservative repair and thinner lines",
            "steps": steps
        }, indent=2), encoding="utf-8")
        # Audit in final Paint coordinates, with a safety margin.
        canvas_w = max(p[0] for s in steps for p in s["points"]) + 20
        canvas_h = max(p[1] for s in steps for p in s["points"]) + 20
        audit = self.audit.inspect([s["points"] for s in steps],
                                   [s["width"] for s in steps],
                                   (canvas_h, canvas_w))
        audit.update({"source": str(source), "drawing": str(output),
                      "original_paths": len(paths) - len(bridges),
                      "repair_bridges": len(bridges),
                      "steps": len(steps),
                      "note": "Repairs are conservative; manual verification is required."})
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Created {output}: {len(steps)} strokes, {len(bridges)} short bridges")
        print(f"Audit: {report}; {audit['enclosed_regions']} enclosed regions")
        print("IMPORTANT: Not a guarantee against Paint bucket leaks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("drawings/dog-traced-v4.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("drawings/dog-traced-v4-audit.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-gap", type=float, default=3.0)
    args = parser.parse_args()
    V4TracingPipeline(args.threshold, args.min_length, args.max_gap).run(
        args.image, args.output, args.report)


if __name__ == "__main__":
    main()
