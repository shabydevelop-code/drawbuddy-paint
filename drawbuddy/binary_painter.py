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


def build_region_drawing(source, max_width=500, max_height=350, threshold=None):
    """Plan filled connected regions, including alternating black/white holes.

    Contour hierarchy is painted parent-first; child holes are restored in white.
    No pixel-by-pixel raster-run execution is used in this mode.
    """
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    h, w = image.shape
    scale = min(max_width / w, max_height / h)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = cv2.resize(image, target, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be 0..255")
        binary = np.where(gray < threshold, 255, 0).astype(np.uint8)
    # Two white guard pixels prevent contours touching the edge of the work area.
    binary = cv2.copyMakeBorder(binary, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("No binary regions found")
    tree = hierarchy[0]
    left = 310 + (max_width - target[0]) // 2 - 2
    top = 155 + (max_height - target[1]) // 2 - 2
    steps = []
    def visit(index, depth):
        contour = contours[index]
        # Degenerate one/two-pixel contours are not safely bucket-fillable.
        if len(contour) >= 3 and cv2.contourArea(contour) >= 12:
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, contours, index, 255, thickness=cv2.FILLED)
            # A child contour is processed later; the seed only needs to be
            # inside this contour, not necessarily outside all its children.
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, radius, _, seed = cv2.minMaxLoc(dist)
            if radius >= 1.5:
                points = [[left + int(p[0][0]), top + int(p[0][1])] for p in contour]
                steps.append({"name": f"binary region {len(steps)+1} depth {depth}",
                              "tool": "binary_region", "color": "black" if depth % 2 == 0 else "white",
                              "points": points,
                              "fill_point": [left + int(seed[0]), top + int(seed[1])]})
        child = int(tree[index][2])
        while child != -1:
            visit(child, depth + 1)
            child = int(tree[child][0])
    for i, entry in enumerate(tree):
        if int(entry[3]) == -1:
            visit(i, 0)
    if not steps:
        raise ValueError("No sufficiently large closed binary regions; try --binary-runs")
    return {"name": f"{source.stem}-binary-regions", "mode": "binary-regions",
            "binary": {"zero": "white", "one": "black", "width": target[0],
                       "height": target[1], "regions": len(steps)},
            "steps": steps}
