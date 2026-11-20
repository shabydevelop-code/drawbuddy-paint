"""Natural Painter: trace real dark contours, then fill color regions vertically.

Preserves the quantized source palette and dimensions. Does not invent
horizontal fill strokes, black borders around colored regions, or long bridges.
"""
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing


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
                    paths.append([[left + x, top + y] for x, y in current])
                    current = None
            if current is None:
                current = [a]
            if a != b:
                current.append(b)
        if current is not None:
            if len(current) == 1:
                current.append(current[0])
            paths.append([[left + x, top + y] for x, y in current])
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
    contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: -cv2.arcLength(c, True))
    steps = []
    traced = np.zeros_like(black)
    outlines = []
    for contour in contours:
        if len(contour) < 4 or cv2.arcLength(contour, True) < 9:
            continue
        approx = cv2.approxPolyDP(contour, 0.45, True)
        points = [[left + int(p[0][0]), top + int(p[0][1])] for p in approx]
        if len(points) < 3:
            continue
        points.append(points[0])
        outlines.append(points)
        cv2.polylines(traced, [approx], True, 1, 1, lineType=cv2.LINE_8)
    if outlines:
        steps.append({'name': 'Natural dark outlines', 'tool': 'brush_paths',
                      'color': 'black', 'width': 1, 'paths': outlines})
    # Only omit contour pixels already traced; everything else gets a vertical
    # fill stroke. Keep palette mapping exactly as in the raster reference.
    for color in ('black', 'brown', 'orange'):
        if color not in masks:
            continue
        fill = masks[color].copy()
        if color == 'black':
            fill[traced != 0] = 0
        paths = _vertical_region_paths(fill, left, top)
        if paths:
            steps.append({'name': f'Natural {color} region fill',
                          'tool': 'brush_paths', 'color': color,
                          'width': 1, 'paths': paths})
    if not steps:
        raise ValueError('No natural drawing strokes')
    return {'name': raster['name'] + '-natural', 'mode': 'natural',
            'natural': {'width': width, 'height': height,
                        'outline_paths': len(outlines),
                        'fill_paths': sum(len(s['paths']) for s in steps[1:])},
            'steps': steps}
