"""Color-map pencil planning with bounded overlap and verified planned coverage.

Sweep runs once; connect only nearby endpoints when the connecting line is
same-color and has little already-painted overlap. Otherwise lift the pencil.
"""
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing

MAX_BRIDGE_PIXELS = 3


def _line_pixels(a, b):
    """All integer pixels touched by the candidate connector."""
    x0, y0 = a
    x1, y1 = b
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n == 0:
        return [a]
    return [(round(x0 + (x1 - x0) * i / n),
             round(y0 + (y1 - y0) * i / n)) for i in range(n + 1)]


def _continuous_paths(mask, left, top):
    """Draw each row run once; join only short, safe, low-overlap segments."""
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ids = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    covered = np.zeros_like(mask, dtype=np.uint8)
    paths = []
    overlaps = 0
    for component_id in ids:
        x0, y0, cw, ch, _ = map(int, stats[component_id])
        runs = []
        for y in range(y0, y0 + ch):
            row = (labels[y, x0:x0 + cw] == component_id).astype(np.int8)
            padded = np.pad(row, (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            row_runs = [(x0 + int(a), x0 + int(b), y)
                        for a, b in zip(starts, ends)]
            if (y - y0) % 2:
                row_runs.reverse()
            for a, b, yy in row_runs:
                runs.append(((a, yy), (b, yy)) if (y - y0) % 2 == 0
                            else ((b, yy), (a, yy)))
        current = None
        for start, end in runs:
            segment_pixels = _line_pixels(start, end)
            if current is not None:
                bridge = _line_pixels(current[-1], start)
                # A short diagonal is allowed only if every traversed pixel
                # belongs to the target color. Never redraw a long route.
                safe = (len(bridge) - 1 <= MAX_BRIDGE_PIXELS and
                        all(0 <= x < w and 0 <= y < h and
                            labels[y, x] == component_id for x, y in bridge))
                # Do not create a connector that mostly retraces old paint.
                repeated = sum(bool(covered[y, x]) for x, y in bridge[1:-1]) if safe else 0
                if safe and repeated <= 1:
                    current.extend([list(p) for p in bridge[1:]])
                else:
                    paths.append([[left + x, top + y] for x, y in current])
                    current = None
            if current is None:
                current = [list(start)]
            if end != start:
                current.append(list(end))
            # All pixels on a row run are covered exactly once by the sweep.
            for x, y in segment_pixels:
                if covered[y, x]:
                    raise RuntimeError('Unexpected duplicate row coverage')
                covered[y, x] = 1
        if current is not None:
            if len(current) == 1:
                current.append(current[0][:])
            paths.append([[left + x, top + y] for x, y in current])
    if not np.array_equal(covered != 0, mask != 0):
        raise RuntimeError('Coverage mismatch: some source pixels were skipped')
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
            steps.append({'name': f'Coverage-aware {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
            statistics[name] = {'pixels': int(mask.sum()), 'pencil_lifts': len(paths),
                                'mouse_positions': sum(len(p) for p in paths)}
    if not steps:
        raise ValueError('No colored pixels to draw')
    return {'name': raster['name'] + '-coverage-aware',
            'continuous': {'width': width, 'height': height,
                           'colors': len(steps), 'statistics': statistics,
                           'max_bridge_pixels': MAX_BRIDGE_PIXELS},
            'steps': steps}
