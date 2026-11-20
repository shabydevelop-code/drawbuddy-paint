"""Inspect 0/1 internal contours without drawing anything in Paint.

Run: python preview_inner_contours.py --binary-source assets/dog-reference.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preview 0/1 inner contours without Paint")
    parser.add_argument("--binary-source", type=Path, required=True)
    parser.add_argument("--binary-threshold", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("output/inner-contours-preview.png"))
    args = parser.parse_args()
    if args.binary_threshold is not None and not 0 <= args.binary_threshold <= 255:
        parser.error("--binary-threshold must be 0..255")
    gray = cv2.imread(str(args.binary_source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"Cannot read image: {args.binary_source}")
    h, w = gray.shape
    scale = min(500 / w, 350 / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if args.binary_threshold is None:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ink = np.where(gray < args.binary_threshold, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        parser.error("No contours found")
    tree = hierarchy[0]
    roots = [i for i, entry in enumerate(tree) if entry[3] == -1]
    outer = max(roots, key=lambda i: cv2.contourArea(contours[i]))
    # Only contours directly inside the selected silhouette are candidates.
    candidates = [i for i, entry in enumerate(tree)
                  if int(entry[3]) == outer and cv2.contourArea(contours[i]) >= 20]
    candidates.sort(key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    tile_w, tile_h = size[0] + 24, size[1] + 52
    tiles = []
    for index, contour_index in enumerate(candidates, 1):
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        cv2.drawContours(tile, contours, outer, (190, 190, 190), 1, offset=(12, 40))
        cv2.drawContours(tile, contours, contour_index, (0, 0, 220), 2, offset=(12, 40))
        area = cv2.contourArea(contours[contour_index])
        cv2.putText(tile, f"INNER {index} | area={area:.0f}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        cv2.drawContours(tile, contours, outer, (0, 0, 0), 1, offset=(12, 40))
        cv2.putText(tile, "NO INNER CONTOURS FOUND", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
    columns = min(3, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    preview = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, columns)
        preview[y * tile_h:(y + 1) * tile_h, x * tile_w:(x + 1) * tile_w] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), preview):
        raise OSError(f"Cannot save preview: {args.output}")
    print(f"Preview: {args.output} | inner candidates: {len(candidates)}")
    print("Gray = existing outer silhouette; red = one candidate per panel. No Paint actions.")


if __name__ == "__main__":
    main()
