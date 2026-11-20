"""Vertical-first region-aware drawing with no repeated area sweeps.

Use long vertical strokes for filled regions. Horizontal strokes are reserved
for explicitly detected thin, long horizontal features in the source mask.
Only short diagonal/vertical connectors are permitted between fill strokes.
"""
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing

MAX_BRIDGE_PIXELS = 3
MIN_HORIZONTAL_LINE = 10


def _line_pixels(a, b):
    x0, y0 = a
    x1, y1 = b
    n = max(abs(x1 - x0), abs(y1 - y0))
    if not n:
        return [a]
    return [(round(x0 + (x1 - x0) * i / n),
             round(y0 + (y1 - y0) * i / n)) for i in range(n + 1)]


def _runs(mask, vertical=True):
    """Find maximal straight runs, grouped by their position in the region."""
    h, w = mask.shape
    result = []
    if vertical:
        for x in range(w):
            padded = np.pad(mask[:, x].astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            result.extend(((x, int(a)), (x, int(b))) for a, b in zip(starts, ends))
    else:
        for y in range(h):
            padded = np.pad(mask[y, :].astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            result.extend(((int(a), y), (int(b), y)) for a, b in zip(starts, ends))
    return result


def _horizontal_features(component):
    """Only preserve thin, clearly horizontal source lines as horizontal strokes.

    A long row is not a line if it is part of a vertically thick filled area.
    """
    h, w = component.shape
    features = np.zeros_like(component, dtype=np.uint8)
    if w < MIN_HORIZONTAL_LINE:
        return features
    for a, b in _runs(component, vertical=False):
        x0, y = a
        x1 = b[0]
        if x1 - x0 + 1 < MIN_HORIZONTAL_LINE:
            continue
        # Require thinness across the whole candidate line. Filled regions
        # have substantial neighboring pixels and must remain vertical fills.
        above = component[y - 1, x0:x1 + 1] if y else np.zeros(x1-x0+1, dtype=np.uint8)
        below = component[y + 1, x0:x1 + 1] if y + 1 < h else np.zeros(x1-x0+1, dtype=np.uint8)
        if np.count_nonzero(above | below) <= max(1, (x1 - x0 + 1) // 10):
            features[y, x0:x1 + 1] = 1
    return features


def _continuous_paths(mask, left, top):
    """Plan vertical region strokes; join only safe short nonhorizontal moves."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ids = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    covered = np.zeros_like(mask, dtype=np.uint8)
    paths = []
    for component_id in ids:
        x0, y0, cw, ch, _ = map(int, stats[component_id])
        region = (labels[y0:y0+ch, x0:x0+cw] == component_id).astype(np.uint8)
        horizontal = _horizontal_features(region)
        vertical = region & (1 - horizontal)
        # A long line is drawn horizontally only when the source genuinely
        # contains a thin horizontal feature, not as a fill/bridge shortcut.
        segments = _runs(vertical, vertical=True)
        horizontal_segments = _runs(horizontal, vertical=False)
        # Work from larger strokes to smaller details.
        segments.sort(key=lambda p: -abs(p[1][1] - p[0][1]))
        horizontal_segments.sort(key=lambda p: -abs(p[1][0] - p[0][0]))
        for group, is_horizontal in ((segments, False), (horizontal_segments, True)):
            current = None
            for local_a, local_b in group:
                a = (x0 + local_a[0], y0 + local_a[1])
                b = (x0 + local_b[0], y0 + local_b[1])
                if current is not None:
                    # Choose the nearer end of the next vertical run.
                    if not is_horizontal and max(abs(current[-1][0]-b[0]), abs(current[-1][1]-b[1])) < max(abs(current[-1][0]-a[0]), abs(current[-1][1]-a[1])):
                        a, b = b, a
                    bridge = _line_pixels(current[-1], a)
                    dx = abs(a[0] - current[-1][0])
                    dy = abs(a[1] - current[-1][1])
                    safe = (not is_horizontal and 0 < dx <= dy and
                            len(bridge) - 1 <= MAX_BRIDGE_PIXELS and
                            all(mask[y, x] and
                                (not covered[y, x] or i == 0 or i == len(bridge)-1)
                                for i, (x, y) in enumerate(bridge)))
                    if safe:
                        current.extend(bridge[1:])
                    else:
                        paths.append([[left + x, top + y] for x, y in current])
                        current = None
                if current is None:
                    current = [a]
                if a != b:
                    current.append(b)
                for x, y in _line_pixels(a, b):
                    if covered[y, x]:
                        raise RuntimeError('Duplicate planned area stroke')
                    covered[y, x] = 1
            if current is not None:
                if len(current) == 1:
                    current.append(current[0])
                paths.append([[left + x, top + y] for x, y in current])
    if not np.array_equal(covered != 0, mask != 0):
        raise RuntimeError('Vertical planner coverage mismatch')
    return paths


def build_continuous_color_drawing(source):
    raster = build_color_raster_drawing(source)
    info = raster['color']
    width, height = int(info['width']), int(info['height'])
    left = 310 + (240 - width) // 2
    top = 155 + (180 - height) // 2
    steps = []
    statistics = {}
    for step in raster['steps']:
        name = step['color']
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, x2, y in step['runs']:
            mask[y - top, x1 - left:x2 - left + 1] = 1
        paths = _continuous_paths(mask, left, top)
        if paths:
            steps.append({'name': f'Vertical-first {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
            statistics[name] = {'pixels': int(mask.sum()), 'pencil_lifts': len(paths),
                                'mouse_positions': sum(len(p) for p in paths)}
    if not steps:
        raise ValueError('No colored pixels to draw')
    return {'name': raster['name'] + '-vertical-first',
            'continuous': {'width': width, 'height': height,
                           'colors': len(steps), 'statistics': statistics},
            'steps': steps}
