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
            # Build the same 2px contour that Paint will draw. Select a
            # seed only from a raster-enclosed component, never from the
            # filled contour polygon (which may contain exterior pixels).
            line = np.zeros(binary.shape, dtype=np.uint8)
            cv2.polylines(line, [contour], True, 255, 2, lineType=cv2.LINE_8)
            free = (line == 0).astype(np.uint8)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(
                free, connectivity=4)
            exterior = int(labels[0, 0])
            enclosed = [i for i in range(1, count)
                        if i != exterior and stats[i, cv2.CC_STAT_AREA] >= 12]
            if enclosed:
                region = max(enclosed, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                interior = (labels == region).astype(np.uint8)
                distance = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
                _, radius, _, seed = cv2.minMaxLoc(distance)
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


def build_outline_drawing(source, max_width=500, max_height=350, threshold=None):
    """Draw only the exterior 0/1 contour, without filling or gap closing."""
    source = Path(source)
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    height, width = gray.shape
    scale = min(max_width / width, max_height / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if threshold is None:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be 0..255")
        ink = np.where(gray < threshold, 255, 0).astype(np.uint8)
    # Draw only the largest external contour of the unmodified 0/1 mask.
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No exterior contour found in binary image")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        raise ValueError("Exterior contour is too small")
    contour = cv2.approxPolyDP(contour, epsilon=0.55, closed=True)
    left = 310 + (max_width - size[0]) // 2
    top = 155 + (max_height - size[1]) // 2
    points = [[left + int(p[0][0]), top + int(p[0][1])] for p in contour]
    if len(points) < 3:
        raise ValueError("Exterior contour has fewer than three points")
    points.append(points[0])
    steps = [{"name": "outer silhouette only", "tool": "stroke",
              "color": "black", "width": 1, "points": points}]

    # Preserve the approved exterior and all previously selected inner contours.
    # Also trace independent black components and small enclosed details
    # directly from the same 0/1 mask. No morphology, filling, or skeletons.
    all_contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is not None:
        tree = hierarchy[0]
        roots = [i for i, h in enumerate(tree) if int(h[3]) == -1]
        outer_index = max(roots, key=lambda i: cv2.contourArea(all_contours[i]))
        candidates = [i for i, h in enumerate(tree)
                      if int(h[3]) == outer_index and cv2.contourArea(all_contours[i]) >= 20]
        candidates.sort(key=lambda i: cv2.contourArea(all_contours[i]), reverse=True)
        selected = [candidates[n - 1] for n in (10, 12, 11, 9, 4, 5, 6, 7, 8, 1, 2, 3)
                    if len(candidates) >= n]
        # Roots other than the approved silhouette are disconnected ink marks.
        # Small holes can represent facial details; include them if measurable.
        selected.extend(i for i, h in enumerate(tree)
                        if i != outer_index and i not in selected
                        and ((int(h[3]) == -1 and cv2.contourArea(all_contours[i]) >= 1)
                             or (int(h[3]) == outer_index
                                 and cv2.contourArea(all_contours[i]) >= 1)))
        for index in selected:
            detail = cv2.approxPolyDP(all_contours[index], epsilon=0.55, closed=True)
            detail_points = [[left + int(p[0][0]), top + int(p[0][1])] for p in detail]
            if len(detail_points) >= 3:
                detail_points.append(detail_points[0])
                steps.append({"name": f"binary detail {index}", "tool": "stroke",
                              "color": "black", "width": 1, "points": detail_points})
    return {"name": f"{source.stem}-all-binary-contours",
            "mode": "binary-all-contours",
            "binary": {"zero": "white", "one": "black", "width": size[0],
                       "height": size[1], "outlines": len(steps), "fill_enabled": False},
            "steps": steps}
