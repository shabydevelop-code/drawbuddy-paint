"""Extract meaningful black line art for freehand Paint pencil strokes."""
from pathlib import Path
import cv2
import numpy as np


def build_lineart_drawing(source, max_width=240, max_height=180):
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {source}')
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 45, 110)
    # Keep only meaningful, continuous boundaries; suppress isolated texture.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
    left = 310 + (max_width - w) // 2
    top = 155 + (max_height - h) // 2
    paths = []
    for contour in contours:
        length = cv2.arcLength(contour, False)
        if length < 14:
            continue
        # Use open contour paths to avoid artificial closing segments.
        simplified = cv2.approxPolyDP(contour, 0.65, False)[:, 0, :]
        coords = [[left + int(x), top + int(y)] for x, y in simplified]
        deduped = []
        for point in coords:
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        if len(deduped) >= 2:
            paths.append(deduped)
    if not paths:
        raise ValueError('No significant contours detected')
    return {'name': f'Line art {source.stem}',
            'lineart': {'width': w, 'height': h, 'left': left,
                        'top': top, 'paths': len(paths)},
            'steps': [{'name': 'Black freehand outlines', 'tool': 'brush_paths',
                       'color': 'black', 'width': 1, 'paths': paths}]}
