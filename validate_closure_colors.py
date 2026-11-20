"""Compare source-color consistency before and after proposed gap closures.

Usage: python validate_closure_colors.py assets/dog-reference.png drawings/dog.json --publish-diagnostic
Read-only: never modifies drawing JSON or launches Paint.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from validate_gap_closures import GapClosureValidator
from validate_fill_regions import FillRegionValidator


class ClosureColorValidator:
    def __init__(self, min_area=100, min_fraction=0.12, color_distance=28):
        self.closure = GapClosureValidator()
        self.alignment = FillRegionValidator()
        self.min_area = min_area
        self.min_fraction = min_fraction
        self.color_distance = color_distance

    @staticmethod
    def quantize(image):
        smooth = cv2.pyrMeanShiftFiltering(image, 12, 24)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        sample = lab[::3, ::3].reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(7)
        _, _, palette = cv2.kmeans(
            sample, 7, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5),
            2, cv2.KMEANS_PP_CENTERS)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        labels = np.empty(len(pixels), dtype=np.uint8)
        for i in range(0, len(pixels), 100000):
            block = pixels[i:i+100000]
            labels[i:i+len(block)] = np.argmin(
                np.sum((block[:, None] - palette[None, :])**2, axis=2), axis=1)
        return cv2.medianBlur(labels.reshape(lab.shape[:2]), 5), palette

    def analyze(self, raster, aligned, labels, palette):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (raster == 0).astype(np.uint8), connectivity=4)
        h, w = raster.shape
        mixed_mask = np.zeros((h, w), dtype=np.uint8)
        valid_regions = mixed_regions = 0
        for index in range(1, count):
            x, y, rw, rh, area = map(int, stats[index])
            if (area < self.min_area or x == 0 or y == 0 or
                    x+rw == w or y+rh == h):
                continue
            component = (components == index).astype(np.uint8)
            interior = cv2.erode(component, np.ones((5, 5), np.uint8)) > 0
            interior &= np.min(aligned, axis=2) < 240
            selected = labels[interior]
            if len(selected) < self.min_area // 2:
                continue
            valid_regions += 1
            hist = np.bincount(selected, minlength=len(palette))
            major = np.flatnonzero(hist >= len(selected)*self.min_fraction)
            is_mixed = any(
                np.linalg.norm(palette[a]-palette[b]) >= self.color_distance
                for j, a in enumerate(major) for b in major[j+1:])
            if is_mixed:
                mixed_regions += 1
                mixed_mask[component > 0] = 255
        return valid_regions, mixed_regions, mixed_mask

    @staticmethod
    def render(raster, mixed):
        image = np.full((*raster.shape, 3), 255, dtype=np.uint8)
        image[mixed > 0] = (170, 215, 255)
        image[raster > 0] = (0, 0, 0)
        return image

    def run(self, reference_path, drawing, output):
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(reference_path)
        accepted, rejected, original, offset = self.closure.evaluate(drawing)
        aligned = self.alignment.reference_to_canvas(
            reference, drawing, (original.shape[1], original.shape[0]), offset)
        labels, palette = self.quantize(aligned)
        updated = original.copy()
        ox, oy = offset
        for _, start, end, *_ in accepted:
            cv2.line(updated, (int(start[0]-ox), int(start[1]-oy)),
                     (int(end[0]-ox), int(end[1]-oy)), 255, 1, cv2.LINE_8)
        old_count, old_mixed, old_mask = self.analyze(
            original, aligned, labels, palette)
        new_count, new_mixed, new_mask = self.analyze(
            updated, aligned, labels, palette)
        panels = []
        for title, raster, mask in (
                (f"BEFORE: {old_mixed}/{old_count} mixed", original, old_mask),
                (f"AFTER: {new_mixed}/{new_count} mixed", updated, new_mask)):
            panel = self.render(raster, mask)
            header = np.full((32, panel.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(header, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(np.vstack((header, panel)))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), np.hstack(panels)):
            raise RuntimeError(f"Cannot save diagnostic: {output}")
        print(f"Candidate joins simulated: {len(accepted)}; rejected: {len(rejected)}")
        print(f"Before: {old_count} sampled enclosed regions, {old_mixed} mixed")
        print(f"After:  {new_count} sampled enclosed regions, {new_mixed} mixed")
        print(f"Mixed-region change: {new_mixed-old_mixed:+d}")
        print(f"Saved: {output}")
        print("Orange shading marks regions containing multiple source-color clusters.")
        print("Counts are approximate; source alignment and shading may cause false positives.")
        print("No drawing files or Paint canvas were modified.")


def main():
    parser = argparse.ArgumentParser(description="Check color-region consistency of proposed joins")
    parser.add_argument("reference", nargs="?", type=Path,
                        default=Path("assets/dog-reference.png"))
    parser.add_argument("drawing", nargs="?", type=Path,
                        default=Path("drawings/dog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/closure-color-diagnostic.png"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    ClosureColorValidator().run(args.reference, drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
