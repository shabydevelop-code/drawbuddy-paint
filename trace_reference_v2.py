"""Centerline tracing (v2) of the exact reference for mouse-driven MS Paint.

Usage: python trace_reference_v2.py assets/dog-reference.png
Dependencies: opencv-python, numpy (no opencv-contrib required).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from trace_reference import PaintCanvasMapper, PaintDrawingExporter


class InkMask:
    """Separate near-black linework from colored fills."""

    def __init__(self, threshold=95):
        self.threshold = threshold

    def from_image(self, filename):
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read reference: {filename}")
        # Dark in every channel; avoid treating saturated brown/orange as ink.
        mask = (np.max(image, axis=2) < self.threshold).astype(np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               np.ones((2, 2), np.uint8))


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

    def __init__(self, min_length=12, simplify=0.65):
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


class CenterlineTracingPipeline:
    """Keep image processing, coordinate mapping and export independent."""

    def __init__(self, threshold=95, min_length=12):
        self.mask = InkMask(threshold)
        self.thinner = ZhangSuenThinner()
        self.tracer = SkeletonPathTracer(min_length)
        self.mapper = PaintCanvasMapper()
        self.exporter = PaintDrawingExporter()

    def run(self, image, output):
        ink = self.mask.from_image(image)
        skeleton = self.thinner.thin(ink)
        paths = self.tracer.trace(skeleton)
        if not paths:
            raise ValueError("No centerline paths detected; adjust --threshold")
        mapped = self.mapper.map(paths)
        self.exporter.save(mapped, output)


def main():
    parser = argparse.ArgumentParser(description="Trace centerlines into DrawBuddy")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("drawings/dog-traced-v2.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=12)
    args = parser.parse_args()
    CenterlineTracingPipeline(args.threshold, args.min_length).run(
        args.image, args.output)


if __name__ == "__main__":
    main()
