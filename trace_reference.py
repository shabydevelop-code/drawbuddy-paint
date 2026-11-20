"""Trace the actual reference illustration into Paint mouse strokes.

Run: python trace_reference.py assets/dog-reference.png
Dependencies: opencv-python, numpy
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class ReferenceLineExtractor:
    """Find dark ink boundaries in the source image; never redraw from memory."""

    def __init__(self, threshold=95, minimum_length=28):
        self.threshold = threshold
        self.minimum_length = minimum_length

    def extract(self, filename):
        image = cv2.imread(str(filename))
        if image is None:
            raise FileNotFoundError(f"Cannot read reference image: {filename}")
        # Require all three channels to be dark: brown/orange fills stay excluded.
        ink = (np.max(image, axis=2) < self.threshold).astype(np.uint8) * 255
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        paths = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < self.minimum_length:
                continue
            approx = cv2.approxPolyDP(contour, 1.35, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            paths.append(approx.tolist() + [approx[0].tolist()])
        paths.sort(key=lambda p: -cv2.arcLength(np.asarray(p, dtype=np.int32), True))
        return paths


class PaintCanvasMapper:
    """Fit all traced paths into a safe region below the Paint ribbon."""

    def __init__(self, left=310, top=155, right=850, bottom=540):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def map(self, paths):
        coords = np.asarray([point for path in paths for point in path], dtype=float)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        scale = min((self.right - self.left) / max(1, max_x - min_x),
                    (self.bottom - self.top) / max(1, max_y - min_y))
        pad_x = (self.right - self.left - (max_x - min_x) * scale) / 2
        pad_y = (self.bottom - self.top - (max_y - min_y) * scale) / 2
        return [[[round(self.left + pad_x + (x - min_x) * scale),
                  round(self.top + pad_y + (y - min_y) * scale)]
                 for x, y in path] for path in paths]


class PaintDrawingExporter:
    """Write the existing declarative DrawBuddy format, without Paint UI logic."""

    def save(self, paths, filename):
        steps = [
            {"name": f"Reference contour {index:03d}", "tool": "stroke",
             "color": "black", "smooth": False, "width": 2, "points": points}
            for index, points in enumerate(paths, 1) if len(set(map(tuple, points))) > 2
        ]
        if not steps:
            raise ValueError("No ink contours found in the reference image")
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps({
            "name": "Reference-traced dog: contours only, no color",
            "steps": steps}, indent=2), encoding="utf-8")
        print(f"Created {filename} with {len(steps)} traced contours")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("drawings/dog-traced.json"))
    args = parser.parse_args()
    paths = ReferenceLineExtractor().extract(args.image)
    mapped = PaintCanvasMapper().map(paths)
    PaintDrawingExporter().save(mapped, args.output)


if __name__ == "__main__":
    main()
