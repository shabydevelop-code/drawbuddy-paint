"""Fast reference-image tracing for DrawBuddy.

Usage: python trace_reference.py assets/dog-reference.png
Requires: opencv-python and numpy.
Uses connected color regions for interior boundaries; no guessed connectors or fill guarantee.
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


class ColorRegionSegmenter:
    """Segment broad color regions before deriving their shared boundaries.

    Small regions and near-identical neighboring colors are ignored to avoid
    turning shading into extra contour lines. No endpoint guessing is used.
    """

    def __init__(self, clusters=7, min_region_area=650, min_color_distance=42):
        self.clusters = clusters
        self.min_region_area = min_region_area
        self.min_color_distance = min_color_distance

    def extract(self, filename, ink_mask):
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read reference: {filename}")
        # Mean-shift smooths gradients while retaining visible color edges.
        smooth = cv2.pyrMeanShiftFiltering(image, 12, 24)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        h, w = ink_mask.shape
        # Fit the palette on a small, deterministic sample for predictable cost.
        sample = lab[::3, ::3].reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(7)
        _, _, palette = cv2.kmeans(
            sample, self.clusters, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5),
            2, cv2.KMEANS_PP_CENTERS)
        # Assign every pixel to its nearest palette color.
        pixels = lab.reshape(-1, 3).astype(np.float32)
        labels = np.empty(len(pixels), dtype=np.uint8)
        for start in range(0, len(pixels), 120000):
            block = pixels[start:start + 120000]
            distances = np.sum((block[:, None, :] - palette[None, :, :]) ** 2,
                               axis=2)
            labels[start:start + len(block)] = np.argmin(distances, axis=1)
        labels = cv2.medianBlur(labels.reshape(h, w), 5)
        # Discard small islands independently within each palette class.
        regions = np.full((h, w), -1, dtype=np.int16)
        region_colors = []
        region_count = 0
        for color in range(len(palette)):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (labels == color).astype(np.uint8), connectivity=8)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < self.min_region_area:
                    continue
                regions[components == component] = region_count
                region_colors.append(palette[color])
                region_count += 1
        # Derive each interface once, from adjacent pixels belonging to
        # different large regions with genuinely different colors.
        boundaries = np.zeros((h, w), dtype=np.uint8)
        for dy, dx in ((0, 1), (1, 0)):
            left = regions[:h-dy, :w-dx]
            right = regions[dy:, dx:]
            valid = (left >= 0) & (right >= 0) & (left != right)
            ys, xs = np.nonzero(valid)
            if not len(xs):
                continue
            colors = np.asarray(region_colors)
            delta = np.linalg.norm(colors[left[ys, xs]] -
                                   colors[right[ys, xs]], axis=1)
            strong = delta >= self.min_color_distance
            boundaries[ys[strong], xs[strong]] = 1
        # Reject color boundaries near black ink, including anti-aliased edges.
        # This mask is used only for new color boundaries; ink is untouched.
        distance = cv2.distanceTransform(
            (ink_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        boundaries[distance < 11.0] = 0
        return boundaries, region_count


class ColorPathOverlapFilter:
    """Reject color contours that duplicate the original black linework."""

    def __init__(self, min_length=24, near_distance=19, max_overlap=0.22):
        self.min_length = min_length
        self.near_distance = near_distance
        self.max_overlap = max_overlap

    def filter(self, paths, ink_mask):
        distance = cv2.distanceTransform(
            (ink_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        h, w = ink_mask.shape
        accepted = []
        for path in paths:
            if len(path) < 3:
                continue
            if cv2.arcLength(np.asarray(path, dtype=np.int32), False) < self.min_length:
                continue
            # Sample each segment, not just its simplified vertices.
            raster = np.zeros((h, w), dtype=np.uint8)
            for start, end in zip(path, path[1:]):
                cv2.line(raster, tuple(map(int, start)),
                         tuple(map(int, end)), 1, 1)
            values = distance[raster > 0]
            if not len(values):
                continue
            if float(np.mean(values < self.near_distance)) > self.max_overlap:
                continue
            accepted.append(path)
        return accepted


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
        self.color_regions = ColorRegionSegmenter()
        self.color_filter = ColorPathOverlapFilter()
        self.thinner = ZhangSuenThinner()
        self.tracer = SkeletonPathTracer(min_length)
        self.mapper = PaintCanvasMapper()

    def run(self, source, output):
        mask = self.ink.from_image(source)
        paths = self.tracer.trace(self.thinner.thin(mask))
        color_mask, region_count = self.color_regions.extract(source, mask)
        raw_color_paths = self.tracer.trace(self.thinner.thin(color_mask))
        color_paths = self.color_filter.filter(raw_color_paths, mask)
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
        print(f"Created {output}: {region_count} color regions, {len(paths)} ink paths, {len(color_paths)} shared color boundaries; {len(steps)} strokes")


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
