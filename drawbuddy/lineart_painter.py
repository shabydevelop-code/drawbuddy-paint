"""Trace source outlines and internal linework without any color filling."""
from pathlib import Path
import cv2
import numpy as np


def build_lineart_drawing(source, max_width=240, max_height=180):
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
    # Find all visible dark linework, including leg separators and small
    # facial details, not only the outer silhouette.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = ((gray < 105) & (hsv[:, :, 2] < 145)).astype(np.uint8) * 255
    # Canny finds the border of the dog against white even where the fur is
    # brown rather than black. A light blur suppresses color quantization noise.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.6), 55, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dark_contours, _ = cv2.findContours(dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates = [(c, False) for c in contours] + [(c, True) for c in dark_contours]
    candidates.sort(key=lambda item: -cv2.arcLength(item[0], item[1]))
    left = 310 + (max_width - width) // 2
    top = 155 + (max_height - height) // 2
    covered = np.zeros((height, width), dtype=np.uint8)
    paths = []
    for contour, closed in candidates:
        if len(contour) < 3:
            continue
        length = cv2.arcLength(contour, closed)
        if length < 5:
            continue
        # Do not redraw the same black contour when it is present in both
        # the edge map and the dark-ink map.
        contour_mask = np.zeros_like(covered)
        cv2.polylines(contour_mask, [contour], closed, 255, 1)
        count = cv2.countNonZero(contour_mask)
        if not count:
            continue
        overlap = cv2.countNonZero(cv2.bitwise_and(contour_mask, covered))
        if overlap / count > 0.55:
            continue
        # Retain the actual pixel-by-pixel geometry; simplification previously
        # removed small but important leg and face segments.
        coords = [[left + int(p[0][0]), top + int(p[0][1])] for p in contour]
        if closed and len(coords) >= 3:
            coords.append(coords[0])
        if len(coords) < 2:
            continue
        paths.append(coords)
        cv2.polylines(covered, [contour], closed, 255, 1)
    if not paths:
        raise ValueError('No significant contours detected')
    return {'name': f'Complete line art {source.stem}',
            'lineart': {'width': width, 'height': height, 'left': left,
                        'top': top, 'paths': len(paths), 'fill': False},
            'steps': [{'name': 'All black source contours and internal lines',
                       'tool': 'brush_paths', 'color': 'black',
                       'width': 1, 'paths': paths}]}
