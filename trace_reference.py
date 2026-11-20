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


class ReferenceRegionMap:
    """Create a source-derived, recolored region map without semantic labels.

    The output retains original RGB values separately from artificial region IDs.
    This is analysis metadata, not yet Paint fill instructions.
    """

    def __init__(self, colors=10, min_area=180):
        self.colors = colors
        self.min_area = min_area

    def extract(self, source, mapper, reference_paths):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        smooth = cv2.pyrMeanShiftFiltering(image, 12, 24)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]
        sample = lab[::4, ::4].reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(7)
        _, _, palette = cv2.kmeans(
            sample, self.colors, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.4),
            3, cv2.KMEANS_PP_CENTERS)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        labels = np.empty(len(pixels), dtype=np.uint8)
        for start in range(0, len(pixels), 100000):
            block = pixels[start:start+100000]
            labels[start:start+len(block)] = np.argmin(
                ((block[:, None, :] - palette[None, :, :])**2).sum(axis=2), axis=1)
        labels = cv2.medianBlur(labels.reshape(h, w), 5)
        candidates = []
        for cluster in range(self.colors):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (labels == cluster).astype(np.uint8), connectivity=8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < self.min_area:
                    continue
                mask = (components == component).astype(np.uint8)
                contours, hierarchy = cv2.findContours(
                    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy is None:
                    continue
                for index, info in enumerate(hierarchy[0]):
                    if info[3] != -1:
                        continue
                    contour = cv2.approxPolyDP(contours[index], 0.9, True)
                    if len(contour) < 3:
                        continue
                    points = contour.reshape(-1, 2).tolist()
                    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    _, radius, _, seed = cv2.minMaxLoc(distance)
                    if radius < 2.0:
                        continue
                    # Only one outer contour per connected component is expected.
                    ys, xs = np.nonzero(mask)
                    rgb = np.median(image[ys, xs, ::-1], axis=0).astype(int).tolist()
                    holes = []
                    child = int(info[2])
                    while child != -1:
                        hole = cv2.approxPolyDP(contours[child], 0.9, True)
                        if len(hole) >= 3:
                            holes.append(hole.reshape(-1, 2).tolist())
                        child = int(hierarchy[0][child][0])
                    candidates.append((points, [int(seed[0]), int(seed[1])], rgb, area, holes))
        if not candidates:
            return []
        # Map with the original paths to keep coordinates aligned.
        all_paths = reference_paths + [r[0] for r in candidates] + [
            [r[1]] for r in candidates] + [
            hole for r in candidates for hole in r[4]]
        mapped = mapper.map(all_paths)
        offset = len(reference_paths)
        outlines = mapped[offset:offset+len(candidates)]
        seeds = mapped[offset+len(candidates):offset+2*len(candidates)]
        hole_paths = iter(mapped[offset+2*len(candidates):])
        regions = []
        for index, (entry, outline, seed) in enumerate(zip(candidates, outlines, seeds), 1):
            _, _, rgb, area, holes = entry
            mapped_holes = [next(hole_paths) for _ in holes]
            regions.append({
                "id": index, "analysis_color": [
                    (index * 73) % 256, (index * 151) % 256, (index * 199) % 256],
                "source_rgb": rgb, "source_area": area,
                "boundary": outline, "seed": seed[0], "holes": mapped_holes})
        return regions


class SolidInkContourExtractor:
    """Find isolated, broad ink components that can safely be bucket-filled.

    Uses source pixels and geometry, not object labels. Connected thick
    outlines and regions with internal holes are intentionally excluded.
    """

    def __init__(self, min_area=90, min_radius=4.0, min_compactness=0.12):
        self.min_area = min_area
        self.min_radius = min_radius
        self.min_compactness = min_compactness

    def extract(self, ink):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8)
        h, w = ink.shape
        results = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if area < self.min_area or x == 0 or y == 0 or x + width >= w or y + height >= h:
                continue
            region = (labels[y:y+height, x:x+width] == component).astype(np.uint8)
            distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
            _, radius, _, seed = cv2.minMaxLoc(distance)
            if radius < self.min_radius:
                continue
            contours, hierarchy = cv2.findContours(
                region.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            outer = [i for i, row in enumerate(hierarchy[0]) if row[3] == -1]
            if len(outer) != 1:
                continue
            i = outer[0]
            # Filling a contour with holes would erase source highlights.
            if hierarchy[0][i][2] != -1:
                continue
            contour = contours[i]
            perimeter = cv2.arcLength(contour, True)
            compactness = 4 * np.pi * cv2.contourArea(contour) / max(1.0, perimeter ** 2)
            if compactness < self.min_compactness:
                continue
            approx = cv2.approxPolyDP(contour, 0.65, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            points = [[int(px+x), int(py+y)] for px, py in approx]
            seed_point = [int(seed[0]+x), int(seed[1]+y)]
            results.append((points, seed_point))
        return results


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

    @staticmethod
    def _point_inside(point, vertices):
        x, y = point
        inside = False
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            if (a[1] > y) != (b[1] > y):
                crossing = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x < crossing:
                    inside = not inside
        return inside

    @staticmethod
    def _paint_fill_steps(regions):
        """Conservative Paint fills: only hole-free, non-overlapping large regions.

        Uses the three calibrated Paint swatches; never silently changes colors
        to an unrelated palette entry. Black outlines are drawn afterwards.
        """
        swatches = {"black": (0, 0, 0), "orange": (255, 127, 39),
                    "brown": (185, 122, 87)}
        candidates = []
        for region in regions:
            if region["holes"] or region["source_area"] < 650:
                continue
            points, seed = region["boundary"], region["seed"]
            if len(set(map(tuple, points))) < 3 or not ReferenceTracingPipeline._point_inside(seed, points):
                continue
            rgb = region["source_rgb"]
            color = min(swatches, key=lambda name: sum(
                (rgb[i] - swatches[name][i]) ** 2 for i in range(3)))
            if max(abs(rgb[i] - swatches[color][i]) for i in range(3)) > 38:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            if min(xs) < 310 or max(xs) > 850 or min(ys) < 155 or max(ys) > 540:
                continue
            if max(xs)-min(xs) < 6 or max(ys)-min(ys) < 6:
                continue
            candidates.append((region, color))
        # Nested and overlapping regions cannot safely be filled by a bucket
        # without a dedicated compositing pass. Exclude those regions.
        from drawbuddy.drawing_engine import DrawingEngine
        safe = []
        rejected = 0
        for region, color in candidates:
            if any(other["id"] != region["id"] and
                   ReferenceTracingPipeline._point_inside(
                       region["seed"], other["boundary"])
                   for other, _ in candidates):
                continue
            step = {
                "name": f"Region {region['id']:03d}",
                "tool": "closed_path", "color": color, "fill": color,
                "width": 1, "smooth": False,
                "points": region["boundary"], "fill_point": region["seed"]}
            try:
                DrawingEngine.verify_fill_boundary(step)
            except ValueError:
                rejected += 1
                continue
            safe.append(step)
        print(f"Fill geometry preflight: {len(safe)} safe, {rejected} rejected")
        return safe

    def run(self, source, output, fill_solid_ink=False, region_map=False, paint_fills=False):
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
        if fill_solid_ink:
            solid_regions = SolidInkContourExtractor().extract(mask)
            if solid_regions:
                mapped_regions = self.mapper.map(
                    combined + [points for points, _ in solid_regions] +
                    [[seed] for _, seed in solid_regions])
                region_paths = mapped_regions[len(combined):len(combined)+len(solid_regions)]
                region_seeds = mapped_regions[len(combined)+len(solid_regions):]
                fills = []
                for i, (points, seed) in enumerate(zip(region_paths, region_seeds), 1):
                    if len(set(map(tuple, points))) < 3:
                        continue
                    if not all(v >= 0 for p in points for v in p):
                        continue
                    if not self._point_inside(seed[0], points):
                        continue
                    fills.append({"name": f"Solid ink region {i:03d}",
                                  "tool": "closed_path", "color": "black",
                                  "fill": "black", "smooth": False, "width": 1,
                                  "points": points, "fill_point": seed[0]})
                # Draw fills first; original stroke geometry stays unchanged.
                steps = fills + steps
            print(f"Source-derived isolated solid regions: {len(solid_regions)}")
        if not steps:
            raise ValueError("No drawable strokes")
        output.parent.mkdir(parents=True, exist_ok=True)
        drawing = {"name": f"Traced from {Path(source).name}", "steps": steps}
        if region_map or paint_fills:
            drawing["analysis_regions"] = ReferenceRegionMap().extract(
                source, self.mapper, combined)
            print(f"Source-derived color regions mapped: {len(drawing['analysis_regions'])}")
            if paint_fills:
                fills = self._paint_fill_steps(drawing["analysis_regions"])
                drawing["steps"] = fills + drawing["steps"]
                print(f"Paint-ready conservative fills: {len(fills)}")
                print("Only calibrated palette colors are filled; other RGB regions remain metadata.")
            else:
                print("Region map is analysis-only; Paint does not yet support arbitrary RGB fills.")
        output.write_text(json.dumps(drawing, indent=2), encoding="utf-8")
        print(f"Created {output}: {region_count} color regions, {len(paths)} ink paths, {len(color_paths)} shared color boundaries; {len(steps)} strokes")



def raster_drawing(source, output, pixel_step=2):
    """Indexed 0/1 masks, run-length encoded per color; no contours or bucket.

    Every destination grid cell is background (0) or one palette color (1).
    Runs are merged horizontally to minimize mouse actions.
    """
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    from drawbuddy.components import RGB
    h, w = image.shape[:2]
    scale = min(520 / w, 365 / h)
    target_w = max(1, round(w * scale / pixel_step))
    target_h = max(1, round(h * scale / pixel_step))
    reduced = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB).astype(np.int16)
    palette = np.array(list(RGB.values()), dtype=np.int16)
    names = list(RGB)
    distance = np.sum((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2, axis=3)
    labels = np.argmin(distance, axis=2)
    # White/near-white is always background. Unmatched colors remain background,
    # never silently assigned a visually unrelated palette entry.
    background = np.min(rgb, axis=2) > 238
    matched = np.min(np.abs(rgb[:, :, None, :] - palette[None, None, :, :]), axis=3)
    valid = (~background) & (np.min(np.max(np.abs(rgb[:, :, None, :] - palette[None, None, :, :]), axis=3), axis=2) <= 65)
    # Preserve dark contours independently of nearest-palette matching.
    dark = np.max(rgb, axis=2) < 100
    labels[dark] = names.index("black")
    valid |= dark
    left = 310 + (520 - target_w * pixel_step) // 2
    top = 155 + (365 - target_h * pixel_step) // 2
    steps = []
    for color_index, color in enumerate(names):
        runs = []
        for y in range(target_h):
            x = 0
            while x < target_w:
                if not valid[y, x] or labels[y, x] != color_index:
                    x += 1
                    continue
                start = x
                while x + 1 < target_w and valid[y, x + 1] and labels[y, x + 1] == color_index:
                    x += 1
                runs.append([left + start * pixel_step, left + x * pixel_step + pixel_step - 1, top + y * pixel_step])
                x += 1
        for start in range(0, len(runs), 150):
            steps.append({"name": f"Raster {color} {start // 150 + 1:03d}",
                          "tool": "raster_runs", "color": color,
                          "runs": runs[start:start + 150]})
        print(f"Raster {color}: {len(runs)} horizontal spans")
    if not steps:
        raise ValueError("No supported palette pixels in reference")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"name": f"Raster from {source.name}", "steps": steps,
        "raster": {"pixel_step": pixel_step, "width": target_w, "height": target_h,
                   "supported_colors": names}}, indent=2), encoding="utf-8")
    print(f"Created {output}: {len(steps)} batches; no bucket fills")



def vector_painter(source, output):
    """Region-based brush coverage followed by original vector-like ink strokes.

    This is a brush-stroke approximation, not an SVG renderer. No bucket fills.
    """
    from drawbuddy.components import RGB
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    h, w = image.shape[:2]
    scale = min(520 / w, 365 / h)
    target_w, target_h = max(1, round(w * scale)), max(1, round(h * scale))
    reduced = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB).astype(np.int16)
    left = 310 + (520 - target_w) // 2
    top = 155 + (365 - target_h) // 2
    # Paint's size-4 pencil is a broad stroke. Erode coverage so that it
    # stays inside the color region and the fine contours remain visible.
    brush_width = 4
    row_stride = 5
    palette = np.array(list(RGB.values()), dtype=np.int16)
    names = list(RGB)
    diff = np.max(np.abs(rgb[:, :, None, :] - palette[None, None, :, :]), axis=3)
    nearest = np.argmin(np.sum((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2, axis=3), axis=2)
    valid = np.min(diff, axis=2) <= 65
    dark = np.max(rgb, axis=2) < 100
    nearest[dark] = names.index("black")
    valid |= dark
    brush_steps = []
    for index, name in enumerate(names):
        if name == "black":
            continue
        mask = ((nearest == index) & valid).astype(np.uint8)
        # Filter speckle islands; retain contiguous source-derived shapes.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for region in range(1, count):
            if stats[region, cv2.CC_STAT_AREA] >= 24:
                clean[labels == region] = 1
        interior = cv2.erode(clean, np.ones((5, 5), np.uint8), iterations=1)
        paths = []
        for y in range(2, target_h - 2, row_stride):
            x = 0
            while x < target_w:
                if not interior[y, x]:
                    x += 1
                    continue
                start = x
                while x + 1 < target_w and interior[y, x + 1]:
                    x += 1
                if x - start >= 3:
                    paths.append([[left + start, top + y], [left + x, top + y]])
                x += 1
        for start in range(0, len(paths), 100):
            brush_steps.append({
                "name": f"Brush {name} {start // 100 + 1:03d}",
                "tool": "brush_paths", "color": name, "width": brush_width,
                "paths": paths[start:start + 100]})
        print(f"Brush {name}: {len(paths)} region-interior paths")
    # Preserve the established contour extraction, independent of color coverage.
    ReferenceTracingPipeline().run(source, output)
    original = json.loads(output.read_text(encoding="utf-8"))
    steps = brush_steps + original["steps"]
    output.write_text(json.dumps({
        "name": f"Vector-style painter from {source.name}",
        "steps": steps,
        "painter": {"method": "region brush strokes + contour pass",
                    "palette": names, "brush_width": brush_width}
    }, indent=2), encoding="utf-8")
    print(f"Created {output}: {len(brush_steps)} brush batches + {len(original['steps'])} contour paths")


def main():
    parser = argparse.ArgumentParser(description="Fast reference tracing")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("drawings/dog.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--fill-solid-ink", action="store_true", help="Include isolated closed solid ink regions; excludes holes and connected outlines")
    parser.add_argument("--region-map", action="store_true", help="Store recolored connected regions and original RGB in JSON metadata")
    parser.add_argument("--vector-painter", action="store_true", help="Paint region interiors with broad brush strokes, then restore ink contours")
    parser.add_argument("--raster", action="store_true", help="Draw indexed pixel masks as horizontal Paint strokes, without bucket fills")
    parser.add_argument("--pixel-step", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--paint-fills", action="store_true", help="Generate safe fills for calibrated Paint colors before original strokes")
    args = parser.parse_args()
    if args.vector_painter:
        vector_painter(args.image, args.output)
        return
    if args.raster:
        raster_drawing(args.image, args.output, args.pixel_step)
        return
    ReferenceTracingPipeline(args.threshold, args.min_length).run(
        args.image, args.output, fill_solid_ink=args.fill_solid_ink, region_map=args.region_map, paint_fills=args.paint_fills)


if __name__ == "__main__":
    main()
