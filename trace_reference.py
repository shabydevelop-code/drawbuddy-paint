"""Fast reference-image tracing for DrawBuddy.

Usage: python trace_reference.py assets/dog-reference.png
Requires: opencv-python and numpy.
No gap repairs, morphology, width estimation, region planning, or fill audit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class PaintCanvasMapper:
    """Fit all traced paths into a safe region below the Paint ribbon."""

    def __init__(self, left=310, top=155, right=850, bottom=540):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def map(self, paths):
        coords = np.asarray([point for path in paths for point in path], dtype=float)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        scale = min((self.right - self.left) / max(1, max_x - min_x),
                    (self.bottom - self.top) / max(1, max_y - min_y))
        pad_x = (self.right - self.left - (max_x - min_x) * scale) / 2
        pad_y = (self.bottom - self.top - (max_y - min_y) * scale) / 2
        return [[[round(self.left + pad_x + (x - min_x) * scale),
                  round(self.top + pad_y + (y - min_y) * scale)]
                 for x, y in path] for path in paths]





class InkMask:
    """Select dark pixels only; never alter the reference topology."""

    def __init__(self, threshold=95):
        self.threshold = threshold

    def from_image(self, filename):
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read reference: {filename}")
        return (np.max(image, axis=2) < self.threshold).astype(np.uint8)


class ColorBoundaryExtractor:
    """Find visible boundaries between differently colored interior regions.

    Uses color-channel edges rather than grayscale darkness, rejects edges
    adjacent to existing black ink, and filters tiny isolated fragments.
    It does not infer invisible or deliberately open boundaries.
    """

    def __init__(self, min_area=18, ink_clearance=4):
        self.min_area = min_area
        self.ink_clearance = ink_clearance

    def extract(self, filename, ink_mask):
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read reference: {filename}")
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        smooth = cv2.GaussianBlur(lab, (5, 5), 0)
        # Chromatic edges reveal orange/cream and brown/orange transitions
        # that the black-only line extractor cannot see.
        channels = (smooth[:, :, 1], smooth[:, :, 2])
        edges = np.zeros(ink_mask.shape, dtype=np.uint8)
        for channel in channels:
            edges = cv2.bitwise_or(edges, cv2.Canny(channel, 22, 55))
        # Avoid duplicating existing black outlines or their anti-aliased rims.
        radius = 2 * self.ink_clearance + 1
        excluded = cv2.dilate(ink_mask.astype(np.uint8),
                              cv2.getStructuringElement(
                                  cv2.MORPH_ELLIPSE, (radius, radius)))
        edges[excluded > 0] = 0
        # Remove isolated noise, but retain genuine open interior borders.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (edges > 0).astype(np.uint8), connectivity=8)
        clean = np.zeros_like(edges)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= self.min_area:
                clean[labels == label] = 1
        return clean


class ZhangSuenThinner:
    """Reduce thick ink to one-pixel centerlines without extra dependencies."""

    def thin(self, mask):
        image = (mask > 0).astype(np.uint8)
        for _ in range(500):
            changed = False
            for phase in (0, 1):
                p = np.pad(image, 1)
                n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:],
                     p[2:, 2:], p[2:, 1:-1], p[2:, :-2],
                     p[1:-1, :-2], p[:-2, :-2]]
                count = sum(n)
                transitions = sum(((n[k] == 0) & (n[(k + 1) % 8] == 1))
                                  for k in range(8))
                if phase == 0:
                    a = n[0] * n[2] * n[4]
                    b = n[2] * n[4] * n[6]
                else:
                    a = n[0] * n[2] * n[6]
                    b = n[0] * n[4] * n[6]
                delete = ((image == 1) & (count >= 2) & (count <= 6) &
                          (transitions == 1) & (a == 0) & (b == 0))
                if np.any(delete):
                    image[delete] = 0
                    changed = True
            if not changed:
                break
        return image


class SkeletonPathTracer:
    """Trace each undirected pixel edge once, preserving branches and loops."""

    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, min_length=2, simplify=0.65):
        self.min_length = min_length
        self.simplify = simplify

    def trace(self, skeleton):
        ys, xs = np.nonzero(skeleton)
        pixels = {(int(x), int(y)) for x, y in zip(xs, ys)}
        neighbors = {
            p: sorted(q for dx, dy in self.NEIGHBORS
                      if (q := (p[0] + dx, p[1] + dy)) in pixels)
            for p in pixels
        }
        # Avoid diagonal shortcuts around orthogonal 2x2 junctions.
        for p, adjacent in neighbors.items():
            neighbors[p] = [q for q in adjacent
                            if abs(q[0]-p[0]) + abs(q[1]-p[1]) == 1
                            or not ((q[0], p[1]) in pixels
                                    or (p[0], q[1]) in pixels)]
        used = set()
        paths = []

        def edge(a, b):
            return (a, b) if a < b else (b, a)

        def walk(start, next_point):
            points = [start, next_point]
            used.add(edge(start, next_point))
            previous, current = start, next_point
            while len(neighbors[current]) == 2:
                options = [q for q in neighbors[current] if q != previous]
                if not options or edge(current, options[0]) in used:
                    break
                following = options[0]
                used.add(edge(current, following))
                points.append(following)
                previous, current = current, following
            return points

        for p in sorted(pixels):
            if len(neighbors[p]) != 2:
                for q in neighbors[p]:
                    if edge(p, q) not in used:
                        paths.append(walk(p, q))
        for p in sorted(pixels):
            for q in neighbors[p]:
                if edge(p, q) not in used:
                    paths.append(walk(p, q))

        output = []
        for points in paths:
            coords = np.asarray(points, dtype=np.int32)
            if len(coords) < 2 or cv2.arcLength(coords, False) < self.min_length:
                continue
            approx = cv2.approxPolyDP(coords, self.simplify, False)
            reduced = approx.reshape(-1, 2).tolist()
            if len(reduced) >= 2:
                output.append(reduced)
        output.sort(key=lambda p: -cv2.arcLength(np.asarray(p, dtype=np.int32), False))
        return output





class ReferenceTracingPipeline:
    """Trace existing line centers directly to a Paint drawing definition."""

    def __init__(self, threshold=95, min_length=2):
        self.ink = InkMask(threshold)
        self.color_boundaries = ColorBoundaryExtractor()
        self.thinner = ZhangSuenThinner()
        self.tracer = SkeletonPathTracer(min_length)
        self.mapper = PaintCanvasMapper()

    def run(self, source, output):
        mask = self.ink.from_image(source)
        paths = self.tracer.trace(self.thinner.thin(mask))
        color_mask = self.color_boundaries.extract(source, mask)
        color_paths = self.tracer.trace(self.thinner.thin(color_mask))
        if not paths:
            raise ValueError("No ink paths found")
        # Map all paths together so the original and color edges align.
        combined = paths + color_paths
        mapped = self.mapper.map(combined)
        steps = [
            {"name": f"{'Ink' if i < len(paths) else 'Color boundary'} {i + 1:03d}",
             "tool": "stroke", "color": "black", "smooth": False,
             "width": 1, "points": points}
            for i, points in enumerate(mapped)
            if len(set(map(tuple, points))) >= 2
        ]
        if not steps:
            raise ValueError("No drawable strokes")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "name": "Dog traced from reference", "steps": steps
        }, indent=2), encoding="utf-8")
        print(f"Created {output}: {len(paths)} ink paths + {len(color_paths)} color boundary paths; {len(steps)} drawable strokes")


def main():
    parser = argparse.ArgumentParser(description="Fast reference tracing")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("drawings/dog.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=2)
    args = parser.parse_args()
    ReferenceTracingPipeline(args.threshold, args.min_length).run(
        args.image, args.output)


if __name__ == "__main__":
    main()
