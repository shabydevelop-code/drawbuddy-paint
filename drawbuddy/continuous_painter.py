"""Plan same-color continuous pencil paths, drawing large regions before details."""
import cv2
import numpy as np
from .binary_painter import build_color_raster_drawing


def _continuous_paths(mask, left, top):
    """Cover every ink pixel, preferring long straight sweeps over DFS zigzags."""
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ids = sorted(range(1, count), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    paths = []
    covered = np.zeros_like(mask, dtype=np.uint8)
    # One component per stroke. At each junction, travel in the current
    # direction if possible, then prefer an unvisited pixel to backtracking.
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, -1), (1, -1), (-1, 1))
    for component in ids:
        x0, y0, cw, ch, _ = map(int, stats[component])
        ys, xs = np.where(labels[y0:y0+ch, x0:x0+cw] == component)
        if not len(xs):
            continue
        # Start at a boundary; largest components are processed first.
        sx, sy = x0 + int(xs[0]), y0 + int(ys[0])
        visited = {(sx, sy)}
        route = [(sx, sy)]
        stack = [(sx, sy, None)]
        while stack:
            x, y, heading = stack[-1]
            options = []
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        labels[ny, nx] == component and (nx, ny) not in visited):
                    straight = heading == (dx, dy)
                    # Favor continuing a long horizontal or vertical sweep.
                    options.append((0 if straight else 1,
                                    0 if dy == 0 else 1 if dx == 0 else 2,
                                    nx, ny, dx, dy))
            if options:
                _, _, nx, ny, dx, dy = min(options)
                visited.add((nx, ny))
                route.append((nx, ny))
                stack.append((nx, ny, (dx, dy)))
            else:
                stack.pop()
                if stack:
                    route.append((stack[-1][0], stack[-1][1]))
        for x, y in visited:
            covered[y, x] = 1
        if len(route) == 1:
            route.append(route[0])
        paths.append([[left+x, top+y] for x, y in route])
    if not np.array_equal(covered != 0, mask != 0):
        raise RuntimeError('Continuous planner failed exact pixel coverage')
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
            mask[y-top, x1-left:x2-left+1] = 1
        paths = _continuous_paths(mask, left, top)
        if paths:
            # Same color throughout, large regions first; tiny facial marks last.
            steps.append({'name': f'Continuous {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
            statistics[name] = {'pixels': int(mask.sum()), 'pencil_lifts': len(paths),
                                'mouse_positions': sum(len(p) for p in paths)}
    if not steps:
        raise ValueError('No colored pixels to draw')
    return {'name': raster['name'] + '-continuous',
            'continuous': {'width': width, 'height': height,
                           'colors': len(steps), 'statistics': statistics},
            'steps': steps}
