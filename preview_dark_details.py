"""Preview several strict black-ink 0/1 thresholds before changing Paint drawing.

Run: python preview_dark_details.py --binary-source assets/dog-reference.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare strict dark-detail binary masks")
    parser.add_argument("--binary-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/dark-details-preview.png"))
    parser.add_argument("--thresholds", type=int, nargs="+", default=[55, 80, 105, 130, 155])
    args = parser.parse_args()
    if any(not 0 <= threshold <= 255 for threshold in args.thresholds):
        parser.error("Thresholds must be between 0 and 255")
    gray = cv2.imread(str(args.binary_source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"Cannot read source: {args.binary_source}")
    h, w = gray.shape
    scale = min(500 / w, 350 / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    head_h = max(1, round(size[1] * 0.48))
    zoom = 2
    tile_w, tile_h = size[0] * zoom, head_h * zoom
    variants = [("SOURCE", resized[:head_h])]
    for threshold in args.thresholds:
        # Strictly darker than threshold = ink (1); all other pixels = white (0).
        mask = np.where(resized < threshold, 0, 255).astype(np.uint8)
        variants.append((f"0/1 DARK THRESHOLD {threshold}", mask[:head_h]))
    cols = min(3, len(variants))
    rows = (len(variants) + cols - 1) // cols
    preview = np.full((rows * (tile_h + 44), cols * tile_w, 3), 245, dtype=np.uint8)
    for i, (label, panel) in enumerate(variants):
        row, col = divmod(i, cols)
        x, y = col * tile_w, row * (tile_h + 44)
        cv2.putText(preview, label, (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.64, (0, 0, 0), 2, cv2.LINE_AA)
        enlarged = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        preview[y + 44:y + 44 + tile_h, x:x + tile_w] = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), preview):
        raise OSError(f"Cannot save preview: {args.output}")
    print(f"Saved: {args.output} | thresholds: {args.thresholds}")
    print("Preview only: no changes to Paint drawing or current 0/1 planner.")


if __name__ == "__main__":
    main()
