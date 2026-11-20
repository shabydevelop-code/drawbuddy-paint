"""Plan freehand mouse strokes from an image using contours, not raster rows."""
from pathlib import Path
import cv2
import numpy as np
from .components import RGB


def build_freehand_drawing(source, max_width=240, max_height=180):
    """Generate organic contour-following pencil paths for Paint.

    Each ink region is covered by successive inset contour strokes. No
    horizontal raster spans or bitmap pasting are used.
    """
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {source}')
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)
    names = ('brown', 'orange', 'black', 'white')
    palette = np.array([RGB[n] for n in names], dtype=np.int16)
    labels = np.argmin(np.sum((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2,
                                  axis=3, dtype=np.int32), axis=2)
    left = 310 + (max_width - width) // 2
    top = 155 + (max_height - height) // 2
    steps = []
    for index, name in enumerate(names[:-1]):
        mask = (labels == index).astype(np.uint8) * 255
        paths = []
        # Successive inset contours produce continuous curved mouse strokes
        # rather than the printer's horizontal pixel runs.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for depth in range(max(width, height)):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                break
            for contour in contours:
                if len(contour) < 2:
                    continue
                arc = cv2.arcLength(contour, True)
                epsilon = min(0.7, max(0.15, arc * 0.0008))
                points = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :]
                coords = [[int(x) + left, int(y) + top] for x, y in points]
                if len(coords) >= 2:
                    coords.append(coords[0])
                    paths.append(coords)
            mask = cv2.erode(mask, kernel, iterations=1)
            if not np.any(mask):
                break
        if paths:
            steps.append({'name': f'Freehand {name}', 'tool': 'brush_paths',
                          'color': name, 'width': 1, 'paths': paths})
    if not steps:
        raise ValueError('No drawable nonwhite contours found')
    return {'name': f'Freehand {source.stem}',
            'freehand': {'width': width, 'height': height, 'left': left,
                         'top': top, 'paths': sum(len(s['paths']) for s in steps)},
            'steps': steps}
