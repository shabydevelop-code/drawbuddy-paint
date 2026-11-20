"""Generate the current Paint drawing from the original reference image.

Usage: python trace_reference.py assets/dog-reference.png
Dependencies: opencv-python, numpy. All geometry is traced from the image.
The connectivity audit is predictive, not a guarantee of safe Paint filling.
"""
import argparse
import json
import math
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




class ReferenceTracingPipeline:
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
    parser = argparse.ArgumentParser(description="Generate the current Paint drawing from a reference image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("drawings/dog.json"))
    parser.add_argument("--report", type=Path, default=Path("drawings/dog-audit.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-gap", type=float, default=3.0)
    args = parser.parse_args()
    ReferenceTracingPipeline(args.threshold, args.min_length, args.max_gap).run(
        args.image, args.output, args.report)


if __name__ == "__main__":
    main()
