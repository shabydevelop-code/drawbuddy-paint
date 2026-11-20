"""Binary (0/1) image-to-Paint planner.

0 = white background, 1 = black ink. The mask is resized to the safe Paint
canvas and encoded as horizontal runs, preserving source pixels rather than
tracing or inventing vector contours.
"""
from pathlib import Path

import cv2
import numpy as np


def build_binary_drawing(source, max_width=240, max_height=180, threshold=None):
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    height, width = image.shape
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    gray = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if threshold is None:
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        mask = (gray < threshold).astype(np.uint8)
    # White Paint background is assumed. Each 1 is an actual source ink pixel.
    left = 310 + (max_width - new_width) // 2
    top = 155 + (max_height - new_height) // 2
    runs = []
    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        for start, end in zip(starts, ends):
            runs.append([left + int(start), left + int(end), top + y])
    if not runs:
        raise ValueError("Binary mask contains no ink pixels")
    return {
        "name": f"{source.stem}-binary",
        "mode": "binary-01",
        "binary": {"zero": "white", "one": "black",
                   "width": new_width, "height": new_height,
                   "ink_pixels": int(mask.sum()), "runs": len(runs)},
        "steps": [{"name": "binary black ink", "tool": "raster_runs",
                   "color": "black", "runs": runs}],
    }
