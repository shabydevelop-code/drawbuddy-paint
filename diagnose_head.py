"""Create a source/0-1/contour diagnostic of the dog's head; never launches Paint.

Usage: python diagnose_head.py --binary-source assets/dog-reference.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary-source", type=Path, required=True)
    parser.add_argument("--binary-threshold", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("output/head-diagnostic.png"))
    args = parser.parse_args()
    if args.binary_threshold is not None and not 0 <= args.binary_threshold <= 255:
        parser.error("--binary-threshold must be 0..255")
    gray = cv2.imread(str(args.binary_source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"Cannot read source: {args.binary_source}")
    height, width = gray.shape
    scale = min(500 / width, 350 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(gray, size,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if args.binary_threshold is None:
        _, ink = cv2.threshold(resized, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ink = np.where(resized < args.binary_threshold, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        # Red: contours large enough for the previous 20px area threshold.
        # Blue: smaller contours; these might explain missing marks.
        area = cv2.contourArea(contour)
        color = (0, 0, 230) if area >= 20 else (230, 70, 0)
        cv2.drawContours(overlay, contours, i, color, 1)
    head_height = max(1, round(size[1] * 0.48))
    panels = [resized[:head_height], 255 - ink[:head_height],
              overlay[:head_height]]
    labels = ["SOURCE (RESIZED)", "EXACT 0/1 MASK", "CONTOURS RED>=20 BLUE<20"]
    zoom = 2
    tile_w, tile_h = size[0] * zoom, head_height * zoom
    canvas = np.full((tile_h + 48, tile_w * 3, 3), 255, np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        canvas[48:, i * tile_w:(i + 1) * tile_w] = panel
        cv2.putText(canvas, label, (i * tile_w + 12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"Could not save {args.output}")
    print(f"Saved: {args.output}")
    print(f"Contours: {len(contours)} | source: {width}x{height} | mask: {size}")
    print("Diagnostic only: no Paint drawing, no upload.")


if __name__ == "__main__":
    main()
