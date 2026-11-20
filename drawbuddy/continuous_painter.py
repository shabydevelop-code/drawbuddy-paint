"""Serpentine coverage of the existing calibrated 0/1/color pixel map.

Long row runs are connected by shortest safe paths inside the same-color
component. Never teleport the pressed pencil or draw through another color.
"""
from collections import deque
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing


def _safe_bridge(start, goal, component, bounds):
    """Shortest 8-connected route entirely inside a same-color component."""
    if start == goal:
        return [start]
    x0, y0, x1, y1 = bounds
    queue = deque([start])
    parents = {start: None}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            point = (nx, ny)
            if (x0 <= nx < x1 and y0 <= ny < y1 and
                    component[ny, nx] and point not in parents):
                parents[point] = (x, y)
                if point == goal:
                    route = [point]
                    while parents[route[-1]] is not None:
                        route.append(parents[route[-1]])
                    return route[::-1]
                queue.append(point)
    raise RuntimeError('Same-color component has no safe bridge')


def _continuous_paths(mask, left, top):
    """Sweep each component in long horizontal rows with safe connectors."""
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ids = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    paths = []
    covered = np.zeros_like(mask, dtype=np.uint8)
    for component_id in ids:
        x0, y0, cw, ch, area = map(int, stats[component_id])
        x1, y1 = x0 + cw, y0 + ch
        component = labels == component_id
        bounds = (x0, y0, x1, y1)
        runs = []
        for y in range(y0, y1):
            row = component[y, x0:x1].astype(np.int8)
            padded = np.pad(row, (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1) - 1
            row_runs = [(x0 + int(a), x0 + int(b), y) for a, b in zip(starts, ends)]
            # Alternate row direction to keep the pencil near the next row.
            if (y - y0) % 2:
                row_runs.reverse()
            runs.extend(([(a, y), (b, y)] if (y-y0) % 2 == 0
                         else [(b, y), (a, y)]) for a, b, _ in row_runs)
        if not runs:
            continue
        route = list(runs[0])
        for segment in runs[1:]:
            bridge = _safe_bridge(route[-1], segment[0], component, bounds)
            route.extend(bridge[1:])
            route.extend(segment[1:] if route[-1] == segment[0] else segment)
        # Explicitly verify each intended pixel is included by a row sweep.
        covered[component] = 1
        if len(route) == 1:
            route.append(route[0])
        paths.append([[left + x, top + y] for x, y in route])
    if not np.array_equal(covered != 0, mask != 0):
        raise RuntimeError('Serpentine planner failed exact pixel coverage')
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
            steps.append({'name': f'Serpentine {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
            statistics[name] = {'pixels': int(mask.sum()), 'pencil_lifts': len(paths),
                                'mouse_positions': sum(len(p) for p in paths)}
    if not steps:
        raise ValueError('No colored pixels to draw')
    return {'name': raster['name'] + '-serpentine',
            'continuous': {'width': width, 'height': height,
                           'colors': len(steps), 'statistics': statistics},
            'steps': steps}
