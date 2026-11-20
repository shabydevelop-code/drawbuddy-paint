"""Optimize the existing 0/1/color map into continuous pencil paths.

Eight-connected pixel paths join same-color diagonal neighbors. Every
connected component is covered in one mouse-down stroke, with backtracking
inside already-painted pixels when necessary.
"""
import numpy as np
from .binary_painter import build_color_raster_drawing


def _continuous_paths(mask, left, top):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    paths = []
    # Include diagonal neighbors: fewer disconnected islands and pencil lifts.
    # Every endpoint is an actual same-color pixel; no off-mask bridges.
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, -1), (1, -1), (-1, 1))
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            visited[sy, sx] = 1
            route = [(sx, sy)]
            stack = [(sx, sy, 0)]
            while stack:
                x, y, direction = stack[-1]
                if direction == len(neighbors):
                    stack.pop()
                    if stack:
                        route.append((stack[-1][0], stack[-1][1]))
                    continue
                stack[-1] = (x, y, direction + 1)
                dx, dy = neighbors[direction]
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    stack.append((nx, ny, 0))
                    route.append((nx, ny))
            # One-point components are represented by a stationary pencil tap.
            if len(route) == 1:
                route.append(route[0])
            paths.append([[left + x, top + y] for x, y in route])
    return paths


def build_continuous_color_drawing(source):
    raster = build_color_raster_drawing(source)
    info = raster['color']
    width, height = int(info['width']), int(info['height'])
    left = 310 + (240 - width) // 2
    top = 155 + (180 - height) // 2
    steps = []
    stats = {}
    for step in raster['steps']:
        name = step['color']
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, x2, y in step['runs']:
            mask[y - top, x1 - left:x2 - left + 1] = 1
        paths = _continuous_paths(mask, left, top)
        if paths:
            steps.append({'name': f'Continuous {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
            stats[name] = {'pixels': int(mask.sum()), 'pencil_lifts': len(paths),
                           'mouse_positions': sum(len(p) for p in paths)}
    if not steps:
        raise ValueError('No colored pixels to draw')
    return {'name': raster['name'] + '-continuous',
            'continuous': {'width': width, 'height': height,
                           'colors': len(steps), 'statistics': stats},
            'steps': steps}
