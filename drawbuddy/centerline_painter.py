"""Experimental 2px centerline drawing; leaves stable lineart mode untouched."""
from pathlib import Path
import cv2
import numpy as np
from .natural_painter import _interior_line_paths


def _thin(binary):
    """Zhang-Suen thinning: remove outer ink pixels without breaking strokes."""
    ink = (binary != 0).astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            p = np.pad(ink, 1)
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:],
                 p[2:, 2:], p[2:, 1:-1], p[2:, :-2],
                 p[1:-1, :-2], p[:-2, :-2]]
            degree = sum(n)
            turns = sum((n[i] == 0) & (n[(i + 1) % 8] == 1)
                        for i in range(8))
            if phase == 0:
                safe = ((n[0] * n[2] * n[4] == 0) &
                        (n[2] * n[4] * n[6] == 0))
            else:
                safe = ((n[0] * n[2] * n[6] == 0) &
                        (n[0] * n[4] * n[6] == 0))
            remove = ((ink != 0) & (degree >= 2) & (degree <= 6) &
                      (turns == 1) & safe)
            if np.any(remove):
                ink[remove] = 0
                changed = True
        if not changed:
            return ink


def build_centerline_drawing(source, max_width=240, max_height=180):
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {source}')
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    image = cv2.resize(image, (width, height),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Only dark source ink is thinned; no Canny double edges.
    dark = (gray < 105).astype(np.uint8)
    skeleton = _thin(dark)
    left = 310 + (max_width - width) // 2
    top = 155 + (max_height - height) // 2
    paths = _interior_line_paths(skeleton, left, top)
    if not paths:
        raise ValueError('No centerline paths detected')
    # A separate thin color-boundary pass, with no filled regions.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chroma = ((hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 80)).astype(np.uint8)
    contours, _ = cv2.findContours(chroma, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    guard = cv2.dilate(skeleton, np.ones((3, 3), np.uint8))
    colored = []
    for contour in contours:
        if len(contour) < 12:
            continue
        run = []
        for p in contour[:, 0, :]:
            x, y = map(int, p)
            if guard[y, x]:
                if len(run) >= 3:
                    colored.append([[left + a, top + b] for a, b in run])
                run = []
            else:
                run.append((x, y))
        if len(run) >= 3:
            colored.append([[left + a, top + b] for a, b in run])
    steps = []
    if colored:
        steps.append({'name': 'Color region borders (thin)',
                      'tool': 'brush_paths', 'color': 'orange',
                      'width': 1, 'paths': colored})
    steps.append({'name': 'Dark centerlines (2px)',
                  'tool': 'brush_paths', 'color': 'black',
                  'width': 2, 'paths': paths})
    return {'name': f'Experimental centerline {source.stem}',
            'lineart': {'width': width, 'height': height,
                        'centerline_paths': len(paths),
                        'colored_paths': len(colored), 'fill': False},
            'steps': steps}
