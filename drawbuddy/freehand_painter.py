"""Plan ordered, directional freehand strokes from image regions."""
from pathlib import Path
import cv2
import numpy as np
from .components import RGB


def _segments(values):
    """Contiguous occupied intervals along one stroke axis."""
    positions = np.flatnonzero(values)
    if len(positions) == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(chunk[0]), int(chunk[-1])) for chunk in np.split(positions, breaks)]


def _region_paths(mask, left, top):
    """Trace a region boundary, then cover it in ordered long-axis strokes."""
    ys, xs = np.where(mask != 0)
    if len(xs) < 4:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    horizontal = (x1 - x0) >= (y1 - y0)
    paths = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(contour) < 3:
            continue
        simplified = cv2.approxPolyDP(contour, 0.65, True)[:, 0, :]
        coords = [[int(x) + left, int(y) + top] for x, y in simplified]
        if len(coords) >= 3:
            paths.append(coords + [coords[0]])

    # 3px pencil strokes, spaced 3px, along the dominant region axis.
    # Each stroke stays within its component and never joins separate islands.
    # A deterministic serpentine ordering minimizes travel across the canvas.
    if horizontal:
        lines = list(range(y0 + 1, y1 + 1, 3))
        for i, y in enumerate(lines):
            intervals = _segments(mask[y, x0:x1 + 1] != 0)
            if i % 2:
                intervals = list(reversed(intervals))
            for a, b in intervals:
                a += x0
                b += x0
                if b - a < 2:
                    continue
                endpoints = [[a + left, y + top], [b + left, y + top]]
                paths.append(endpoints if i % 2 == 0 else endpoints[::-1])
    else:
        lines = list(range(x0 + 1, x1 + 1, 3))
        for i, x in enumerate(lines):
            intervals = _segments(mask[y0:y1 + 1, x] != 0)
            if i % 2:
                intervals = list(reversed(intervals))
            for a, b in intervals:
                a += y0
                b += y0
                if b - a < 2:
                    continue
                endpoints = [[x + left, a + top], [x + left, b + top]]
                paths.append(endpoints if i % 2 == 0 else endpoints[::-1])
    return paths


def build_freehand_drawing(source, max_width=240, max_height=180):
    """Draw large connected shapes first, using lengthwise/crosswise strokes.

    This is a shape-oriented freehand planner, not the raster printer.
    """
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {source}')
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int32)
    names = ('brown', 'orange', 'black', 'white')
    palette = np.array([RGB[n] for n in names], dtype=np.int32)
    labels = np.argmin(np.sum((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2,
                                  axis=3, dtype=np.int32), axis=2)
    left = 310 + (max_width - width) // 2
    top = 155 + (max_height - height) // 2
    steps = []
    for index, name in enumerate(names[:-1]):
        mask = (labels == index).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        # Work through each connected shape in descending area order.
        component_ids = sorted(range(1, count),
                               key=lambda i: int(stats[i, cv2.CC_STAT_AREA]),
                               reverse=True)
        for component_id in component_ids:
            if stats[component_id, cv2.CC_STAT_AREA] < 8:
                continue
            x, y, w, h, _ = map(int, stats[component_id])
            local = (components[y:y+h, x:x+w] == component_id).astype(np.uint8) * 255
            paths = _region_paths(local, left + x, top + y)
            if paths:
                steps.append({'name': f'Freehand {name} region {component_id}',
                              'tool': 'brush_paths', 'color': name,
                              'width': 3, 'paths': paths})
    if not steps:
        raise ValueError('No drawable nonwhite regions found')
    return {'name': f'Freehand {source.stem}',
            'freehand': {'width': width, 'height': height, 'left': left,
                         'top': top, 'regions': len(steps),
                         'paths': sum(len(s['paths']) for s in steps)},
            'steps': steps}
