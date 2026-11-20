"""Experimental 2px centerline drawing; leaves stable lineart mode untouched."""
from pathlib import Path
import cv2
import numpy as np


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


def _trace_all(mask, left, top):
    """Cover every skeleton pixel, including branches and isolated endpoints."""
    pixels = set(zip(*np.where(mask != 0)))
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0),
               (1, 1), (1, -1), (-1, -1), (-1, 1))
    neighbors = {}
    for y, x in pixels:
        candidates = [(y + dy, x + dx) for dy, dx in offsets
                      if (y + dy, x + dx) in pixels]
        neighbors[y, x] = [p for p in candidates
                           if p[0] == y or p[1] == x or
                           ((y, p[1]) not in pixels and (p[0], x) not in pixels)]
    used = set()
    visited_pixels = set()
    paths = []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, nxt):
        chain = [start, nxt]
        used.add(edge(start, nxt))
        while True:
            previous, current = chain[-2:]
            if len(neighbors[current]) != 2:
                break
            choices = [q for q in neighbors[current]
                       if q != previous and edge(current, q) not in used]
            if not choices:
                break
            q = choices[0]
            used.add(edge(current, q))
            chain.append(q)
        visited_pixels.update(chain)
        paths.append([[left + x, top + y] for y, x in chain])

    for p in sorted(pixels):
        if len(neighbors[p]) != 2:
            for q in neighbors[p]:
                if edge(p, q) not in used:
                    walk(p, q)
    for p in sorted(pixels):
        for q in neighbors[p]:
            if edge(p, q) not in used:
                walk(p, q)
    for p in sorted(pixels - visited_pixels):
        paths.append([[left + p[1], top + p[0]]] * 2)
        visited_pixels.add(p)
    if visited_pixels != pixels:
        raise RuntimeError('Centerline planner omitted source skeleton pixels')
    return paths


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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = (gray < 105).astype(np.uint8)
    skeleton = _thin(dark)
    left = 310 + (max_width - width) // 2
    top = 155 + (max_height - height) // 2
    paths = _trace_all(skeleton, left, top)
    if not paths:
        raise ValueError('No centerline paths detected')

    # Every colored region boundary is retained unless it coincides with
    # dark ink. Separate color boundaries from dark line centerlines.
    chroma = ((hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 80)).astype(np.uint8)
    contours, _ = cv2.findContours(chroma, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    guard = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    colored = []
    for contour in contours:
        if len(contour) < 8:
            continue
        run = []
        for p in list(contour[:, 0, :]) + [contour[0, 0, :]]:
            x, y = map(int, p)
            if guard[y, x]:
                if len(run) >= 2:
                    colored.append([[left + a, top + b] for a, b in run])
                run = []
            else:
                run.append((x, y))
        if len(run) >= 2:
            colored.append([[left + a, top + b] for a, b in run])

    # Produce a plan preview BEFORE Paint starts, to inspect missing details.
    preview = np.full((height, width, 3), 255, dtype=np.uint8)
    for path in colored:
        pts = np.array([[x - left, y - top] for x, y in path], np.int32)
        cv2.polylines(preview, [pts], False, (0, 130, 255), 1)
    for path in paths:
        pts = np.array([[x - left, y - top] for x, y in path], np.int32)
        cv2.polylines(preview, [pts], False, (0, 0, 0), 2)
    output = Path('output')
    output.mkdir(exist_ok=True)
    preview_path = output / 'centerline-plan-preview.png'
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f'Could not write {preview_path}')

    steps = []
    if colored:
        steps.append({'name': 'Color region borders (thin)',
                      'tool': 'brush_paths', 'color': 'orange',
                      'width': 1, 'paths': colored})
    steps.append({'name': 'Complete dark centerlines (2px)',
                  'tool': 'brush_paths', 'color': 'black',
                  'width': 2, 'paths': paths})
    return {'name': f'Centerline {source.stem}',
            'lineart': {'width': width, 'height': height,
                        'centerline_paths': len(paths),
                        'colored_paths': len(colored),
                        'skeleton_pixels': int(np.count_nonzero(skeleton)),
                        'preview': str(preview_path), 'fill': False},
            'steps': steps}
