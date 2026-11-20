"""Compare source color variation with enclosed Paint drawing regions.

Usage: python validate_fill_regions.py assets/dog-reference.png drawings/dog.json
Read-only diagnostic. Alignment is approximate; no strokes are changed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor


class FillRegionValidator:
    """Flag enclosed areas mixing distinct reference colors and open backgrounds."""

    def __init__(self, min_area=100, color_distance=28, min_fraction=0.12):
        self.auditor = FillRegionAuditor(min_area=min_area)
        self.min_area = min_area
        self.color_distance = color_distance
        self.min_fraction = min_fraction

    @staticmethod
    def reference_to_canvas(reference, drawing, size, offset):
        """Approximate affine alignment using visible foreground bounding boxes."""
        points = [p for step in drawing["steps"]
                  if step.get("tool") in ("stroke", "closed_path")
                  for p in step.get("points", [])]
        if not points:
            raise ValueError("No stroke points")
        coords = np.asarray(points, dtype=np.int32)
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        # Foreground includes colored fur, not just the black contour.
        bg = np.all(reference >= 245, axis=2)
        ys, xs = np.where(~bg)
        if not len(xs):
            raise ValueError("Reference contains no visible foreground")
        sx0, sy0, sx1, sy1 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = reference[sy0:sy1+1, sx0:sx1+1]
        target_w, target_h = max(1, xmax-xmin+1), max(1, ymax-ymin+1)
        resized = cv2.resize(crop, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)
        w, h = size
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        dx, dy = int(xmin-offset[0]), int(ymin-offset[1])
        canvas[dy:dy+target_h, dx:dx+target_w] = resized
        return canvas

    def validate(self, reference_path, drawing):
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(reference_path)
        ink, offset = self.auditor.rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = self.reference_to_canvas(reference, drawing, (w, h), offset)
        # Quantize broad source colors deterministically, suppressing shading.
        smooth = cv2.pyrMeanShiftFiltering(aligned, 12, 24)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        sample = lab[::3, ::3].reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(7)
        _, _, palette = cv2.kmeans(
            sample, 7, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5),
            2, cv2.KMEANS_PP_CENTERS)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        labels = np.empty(len(pixels), dtype=np.uint8)
        for start in range(0, len(pixels), 100000):
            block = pixels[start:start+100000]
            labels[start:start+len(block)] = np.argmin(
                np.sum((block[:, None] - palette[None, :])**2, axis=2), axis=1)
        labels = cv2.medianBlur(labels.reshape(h, w), 5)
        count, components, stats, centers = cv2.connectedComponentsWithStats(
            (ink == 0).astype(np.uint8), connectivity=4)
        candidates = []
        exterior = []
        for i in range(1, count):
            x, y, rw, rh, area = map(int, stats[i])
            if area < self.min_area:
                continue
            component = components == i
            touches_border = x == 0 or y == 0 or x+rw == w or y+rh == h
            # Exclude pixels near black outlines and pale background.
            interior = component & (cv2.erode(component.astype(np.uint8),
                        np.ones((5, 5), np.uint8)) > 0)
            interior &= np.min(aligned, axis=2) < 240
            selected = labels[interior]
            if not len(selected):
                continue
            hist = np.bincount(selected, minlength=len(palette))
            major = np.flatnonzero(hist >= len(selected)*self.min_fraction)
            separated = [(int(a), int(b)) for j, a in enumerate(major)
                         for b in major[j+1:]
                         if np.linalg.norm(palette[a]-palette[b]) >= self.color_distance]
            item = (area, round(centers[i][0]+offset[0]),
                    round(centers[i][1]+offset[1]), len(major), len(separated))
            (exterior if touches_border else candidates).append(item)
        candidates.sort(reverse=True)
        return candidates, exterior

    def report(self, reference_path, drawing):
        candidates, exterior = self.validate(reference_path, drawing)
        print(f"Enclosed regions checked: {len(candidates)}")
        for area, x, y, colors, differences in candidates:
            state = "REVIEW mixed source colors" if differences else "candidate"
            print(f"  {state}: area={area}, seed=({x},{y}), color_groups={colors}")
        if exterior:
            print("Exterior region contains reference foreground; open contours")
            print("or alignment differences may cause bucket-fill leakage.")
        print("NOTE: source-to-drawing alignment and color grouping are")
        print("approximate. REVIEW does not prove a missing line. No edits made.")


def main():
    parser = argparse.ArgumentParser(description="Check source colors vs Paint fill regions")
    parser.add_argument("reference", type=Path,
                        nargs="?", default=Path("assets/dog-reference.png"))
    parser.add_argument("drawing", type=Path,
                        nargs="?", default=Path("drawings/dog.json"))
    args = parser.parse_args()
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    FillRegionValidator().report(args.reference, drawing)


if __name__ == "__main__":
    main()
