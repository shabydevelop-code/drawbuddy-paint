"""Natural Painter: trace real dark contours, then fill color regions vertically.

Preserves the quantized source palette and dimensions. Does not invent
horizontal fill strokes, black borders around colored regions, or long bridges.
"""
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing


def _append_path(paths, current, left, top):
    """PaintController requires two positions even for a one-pixel mark."""
    if not current:
        return
    if len(current) == 1:
        current = [current[0], current[0]]
    paths.append([[left + int(x), top + int(y)] for x, y in current])


def _vertical_region_paths(mask, left, top):
    """Fill a region using long alternating vertical strokes, with clean lifts."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    paths = []
    for component in sorted(range(1, count),
                            key=lambda i: -int(stats[i, cv2.CC_STAT_AREA])):
        x0, y0, w, h, _ = map(int, stats[component])
        strokes = []
        for x in range(x0, x0 + w):
            column = (labels[y0:y0 + h, x] == component).astype(np.int8)
            padded = np.pad(column, (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            strokes.extend(((x, y0 + int(a)), (x, y0 + int(b)))
                           for a, b in zip(starts, ends))
        # Zigzag by column, while allowing a short diagonal shift of one pixel.
        strokes.sort(key=lambda s: (s[0][0], s[0][1]))
        current = None
        for a, b in strokes:
            if a[0] % 2:
                a, b = b, a
            if current is not None:
                last = current[-1]
                # Join only directly adjacent, same-color pixels. No long
                # retracing and no horizontal connectors through a filled area.
                if (abs(last[0] - a[0]) == 1 and
                        abs(last[1] - a[1]) == 1 and
                        mask[a[1], a[0]] and mask[last[1], last[0]]):
                    current.append(a)
                else:
                    _append_path(paths, current, left, top)
                    current = None
            if current is None:
                current = [a]
            if a != b:
                current.append(b)
        _append_path(paths, current, left, top)
    return paths



def _interior_line_paths(mask, left, top):
    """Follow connected thin ink pixels as continuous strokes."""
    pixels = set(zip(*np.where(mask > 0)))
    if not pixels:
        return []
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1))
    neighbors = {}
    for y, x in pixels:
        adjacent = [(y + dy, x + dx) for dy, dx in offsets
                    if (y + dy, x + dx) in pixels]
        neighbors[(y, x)] = [q for q in adjacent
                             if q[0] == y or q[1] == x or
                             ((y, q[1]) not in pixels and
                              (q[0], x) not in pixels)]
    visited = set()
    paths = []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def follow(start, nxt):
        path = [start, nxt]
        visited.add(edge(start, nxt))
        previous, current = start, nxt
        while len(neighbors[current]) == 2:
            choices = [q for q in neighbors[current]
                       if q != previous and edge(current, q) not in visited]
            if not choices:
                break
            following = choices[0]
            visited.add(edge(current, following))
            path.append(following)
            previous, current = current, following
        if len(path) >= 3:
            _append_path(paths, [(x, y) for y, x in path], left, top)

    for start in sorted(pixels):
        if len(neighbors[start]) != 2:
            for nxt in neighbors[start]:
                if edge(start, nxt) not in visited:
                    follow(start, nxt)
    for start in sorted(pixels):
        for nxt in neighbors[start]:
            if edge(start, nxt) not in visited:
                follow(start, nxt)
    return paths


def build_natural_drawing(source):
    raster = build_color_raster_drawing(source)
    width, height = raster['color']['width'], raster['color']['height']
    left = 310 + (240 - width) // 2
    top = 155 + (180 - height) // 2
    masks = {}
    for step in raster['steps']:
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, x2, y in step['runs']:
            mask[y - top, x1 - left:x2 - left + 1] = 1
        masks[step['color']] = mask
    black = masks.get('black', np.zeros((height, width), dtype=np.uint8))
    contours, _ = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: -cv2.arcLength(c, True))
    steps = []
    traced = np.zeros_like(black)
    outlines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = contour  # Keep source contour pixels; no polygon simplification.
        points = [[left + int(p[0][0]), top + int(p[0][1])] for p in approx]
        if len(points) < 3:
            continue
        points.append(points[0])
        outlines.append(points)
        cv2.polylines(traced, [approx], True, 1, 1, lineType=cv2.LINE_8)
    if outlines:
        steps.append({'name': 'Natural dark outlines', 'tool': 'brush_paths',
                      'color': 'black', 'width': 1, 'paths': outlines})
    # Interior boundaries (including leg separation) are also real contours.
    # Trace these before the vertical fill instead of fragmenting them into
    # one-column hatch strokes.
    interior = black.copy()
    interior[traced != 0] = 0
    # Only trace thin interior linework; thick areas remain fill regions.
    distance = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
    thin = ((interior != 0) & (distance <= 1.5)).astype(np.uint8)
    details = _interior_line_paths(thin, left, top)
    if details:
        steps.append({'name': 'Natural interior connected lines',
                      'tool': 'brush_paths', 'color': 'black',
                      'width': 1, 'paths': details})
    covered = traced.copy()
    covered[thin != 0] = 1
    for color in ('brown', 'orange', 'black'):
        if color not in masks:
            continue
        fill = masks[color].copy()
        if color == 'black':
            fill[covered != 0] = 0
        paths = _vertical_region_paths(fill, left, top)
        if paths:
            steps.append({'name': f'Natural {color} region fill',
                          'tool': 'brush_paths', 'color': color,
                          'width': 1, 'paths': paths})
    if not steps:
        raise ValueError('No natural drawing strokes')
    for step in steps:
        if any(len(path) < 2 for path in step['paths']):
            raise RuntimeError('Natural planner emitted an invalid single-point stroke')
    return {'name': raster['name'] + '-natural-precision', 'mode': 'natural',
            'natural': {'width': width, 'height': height,
                        'outline_paths': len(outlines),
                        'interior_line_paths': len(details),
                        'fill_paths': sum(len(s['paths']) for s in steps[1:])},
            'steps': steps}
