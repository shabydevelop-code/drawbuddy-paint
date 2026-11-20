"""Validate proposed color-boundary joins without changing the drawing.

Usage: python validate_gap_closures.py drawings/dog.json --publish-diagnostic
Green = passes conservative tests; red = rejected. No joins are applied.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor
from detect_boundary_gaps import BoundaryGapDetector


class GapClosureValidator:
    """Evaluate direction, line crossing, and connected fill-region topology."""

    def __init__(self, max_gap=12, min_alignment=0.65):
        self.detector = BoundaryGapDetector(max_gap=max_gap)
        self.min_alignment = min_alignment

    @staticmethod
    def region_count(raster):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (raster == 0).astype(np.uint8), connectivity=4)
        h, w = raster.shape
        enclosed = 0
        for i in range(1, count):
            x, y, rw, rh, area = map(int, stats[i])
            if x > 0 and y > 0 and x + rw < w and y + rh < h:
                enclosed += 1
        return enclosed

    @staticmethod
    def endpoint_direction(steps, start):
        """Use the first distinct vertex at the matching path endpoint."""
        for step in steps:
            if not step.get("name", "").startswith("Color boundary"):
                continue
            pts = step.get("points", [])
            if len(pts) < 2 or pts[0] == pts[-1]:
                continue
            for path in (pts, list(reversed(pts))):
                if tuple(path[0]) != tuple(start):
                    continue
                for neighbor in path[1:]:
                    delta = np.asarray(path[0], dtype=float) - neighbor
                    length = np.linalg.norm(delta)
                    if length >= 2:
                        return delta / length
        return None

    @staticmethod
    def path_group(steps, start):
        """Associate a detected endpoint with its source color path."""
        for index, step in enumerate(steps):
            if not step.get("name", "").startswith("Color boundary"):
                continue
            points = step.get("points", [])
            if len(points) < 2 or points[0] == points[-1]:
                continue
            if tuple(start) in (tuple(points[0]), tuple(points[-1])):
                return index
        return None

    def evaluate(self, drawing):
        gaps, raster, offset = self.detector.detect(drawing)
        ox, oy = offset
        before = self.region_count(raster)
        groups = {}
        for distance, start, end in gaps:
            direction = self.endpoint_direction(drawing["steps"], start)
            delta = np.asarray(end, dtype=float) - start
            norm = np.linalg.norm(delta)
            aligned = (direction is not None and norm > 0 and
                       float(np.dot(direction, delta / norm)) >= self.min_alignment)
            a = (int(start[0]-ox), int(start[1]-oy))
            b = (int(end[0]-ox), int(end[1]-oy))
            line = np.zeros_like(raster)
            cv2.line(line, a, b, 255, 1, cv2.LINE_8)
            middle = line.copy()
            cv2.circle(middle, a, 2, 0, -1)
            cv2.circle(middle, b, 2, 0, -1)
            crosses = bool(np.any((middle > 0) & (raster > 0)))
            reasons = []
            if not aligned:
                reasons.append("direction")
            if crosses:
                reasons.append("crossing")
            group_id = self.path_group(drawing["steps"], start)
            groups.setdefault(group_id, []).append(
                {"distance": distance, "start": start, "end": end,
                 "line": line, "reasons": reasons})

        accepted, rejected = [], []
        for group_id, members in groups.items():
            # A valid group must contain both distinct ends of the same path.
            endpoints = {tuple(member["start"]) for member in members}
            geometrically_valid = all(not m["reasons"] for m in members)
            group_reasons = []
            if group_id is None or len(endpoints) != 2:
                group_reasons.append("incomplete-path-group")
            if not geometrically_valid:
                group_reasons.append("group-geometry")
            if not group_reasons:
                joined = raster.copy()
                for member in members:
                    joined = cv2.bitwise_or(joined, member["line"])
                if self.region_count(joined) <= before:
                    group_reasons.append("group-no-new-enclosed-region")
            for member in members:
                reasons = list(member["reasons"]) + group_reasons
                item = (member["distance"], member["start"],
                        member["end"], reasons)
                (rejected if reasons else accepted).append(item)
        return accepted, rejected, raster, offset

    @staticmethod
    def save(path, accepted, rejected, raster, offset):
        image = np.full((*raster.shape, 3), 255, dtype=np.uint8)
        image[raster > 0] = (0, 0, 0)
        ox, oy = offset
        for collection, color in ((rejected, (0, 0, 255)),
                                  (accepted, (0, 170, 0))):
            for _, start, end, _ in collection:
                a = (int(start[0]-ox), int(start[1]-oy))
                b = (int(end[0]-ox), int(end[1]-oy))
                cv2.line(image, a, b, color, 2, cv2.LINE_8)
                cv2.circle(image, a, 2, color, -1)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Cannot save diagnostic: {path}")
        print(f"Saved: {path}")

    def run(self, drawing, output):
        accepted, rejected, raster, offset = self.evaluate(drawing)
        print(f"PASS group members: {len(accepted)}; REVIEW/REJECT: {len(rejected)}")
        for d, start, end, reasons in accepted + rejected:
            status = "PASS" if not reasons else ",".join(reasons)
            print(f"  {status}: {start} -> {end} ({d:.1f}px)")
        self.save(output, accepted, rejected, raster, offset)
        print("Grouped by both endpoints of the same color path.")
        print("PASS means locally plausible only; Paint behavior is unverified.")
        print("No drawing paths were modified.")


def main():
    parser = argparse.ArgumentParser(description="Conservative gap-closure validation")
    parser.add_argument("drawing", type=Path, nargs="?",
                        default=Path("drawings/dog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/closure-diagnostic.png"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    GapClosureValidator().run(drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
