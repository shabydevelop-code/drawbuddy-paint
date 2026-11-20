"""Compare a Paint drawing with the source reference at pixel resolution.

Examples:
  python compare_images.py --reference output/raster-source-color.png --actual output/paint-crop.png
  python compare_images.py --reference output/raster-source-color.png --actual paint-diagnostic.png --crop 310 155 240 180

The crop is in screenshot coordinates, not Paint-window-relative coordinates.
Images must have identical dimensions after cropping. No silent resizing.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compare(reference, actual, tolerance=12):
    if reference.shape != actual.shape:
        raise ValueError(
            f"Size mismatch: reference {reference.shape[1]}x{reference.shape[0]}, "
            f"actual {actual.shape[1]}x{actual.shape[0]}. "
            "Supply an accurately aligned crop; automatic scaling is disabled."
        )
    difference = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    max_channel = difference.max(axis=2)
    error_mask = (max_channel > tolerance).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(error_mask, connectivity=8)
    regions = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        regions.append({"x": x, "y": y, "width": width,
                        "height": height, "error_pixels": area})
    regions.sort(key=lambda region: region["error_pixels"], reverse=True)
    return {
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "tolerance_per_channel": int(tolerance),
        "exact_pixel_match_percent": round(100 * float(np.mean(max_channel == 0)), 3),
        "within_tolerance_percent": round(100 * float(np.mean(error_mask == 0)), 3),
        "mean_absolute_channel_error": round(float(difference.mean()), 3),
        "error_pixels": int(error_mask.sum()),
        "error_regions": regions,
    }, max_channel, error_mask


def compare_paint_canvas(paint, reference_path, width, height, left, top,
                         output=Path("output/comparison"), tolerance=12):
    """Capture the exact drawing rectangle from Paint and compare it."""
    import pyautogui as pg

    reference = load_rgb(reference_path)
    if reference.shape[:2] != (height, width):
        raise ValueError("Reference dimensions do not match planned Paint drawing")
    x, y = paint.point(left, top)
    screenshot = pg.screenshot()
    if x < 0 or y < 0 or x + width > screenshot.width or y + height > screenshot.height:
        raise RuntimeError("Paint comparison crop falls outside the display")
    actual = np.asarray(screenshot.crop((x, y, x + width, y + height)).convert("RGB"))
    report, max_channel, mask = compare(reference, actual, tolerance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    heatmap = cv2.applyColorMap(max_channel, cv2.COLORMAP_JET)
    heatmap[mask == 0] = (0, 0, 0)
    overlay = cv2.cvtColor(actual, cv2.COLOR_RGB2BGR).copy()
    overlay[mask != 0] = (0, 0, 255)
    cv2.imwrite(str(output / "error-heatmap.png"), heatmap)
    cv2.imwrite(str(output / "error-overlay.png"), overlay)
    cv2.imwrite(str(output / "actual-crop.png"), cv2.cvtColor(actual, cv2.COLOR_RGB2BGR))
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Paint comparison: exact match = "
          f"{report['exact_pixel_match_percent']}%, "
          f"within tolerance = {report['within_tolerance_percent']}%")
    print("Comparison files:", output)
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare a Paint result against its source pixels.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Crop the actual screenshot to the reference canvas area")
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("output/comparison"))
    args = parser.parse_args()
    if not 0 <= args.tolerance <= 255:
        parser.error("--tolerance must be between 0 and 255")
    reference, actual = load_rgb(args.reference), load_rgb(args.actual)
    if args.crop:
        x, y, width, height = args.crop
        if min(x, y) < 0 or min(width, height) <= 0:
            parser.error("--crop requires nonnegative coordinates and positive dimensions")
        if x + width > actual.shape[1] or y + height > actual.shape[0]:
            parser.error("--crop exceeds the actual image bounds")
        actual = actual[y:y + height, x:x + width]
    report, max_channel, mask = compare(reference, actual, args.tolerance)
    args.output.mkdir(parents=True, exist_ok=True)
    heatmap = cv2.applyColorMap(max_channel, cv2.COLORMAP_JET)
    heatmap[mask == 0] = (0, 0, 0)
    overlay = cv2.cvtColor(actual, cv2.COLOR_RGB2BGR).copy()
    overlay[mask != 0] = (0, 0, 255)
    cv2.imwrite(str(args.output / "error-heatmap.png"), heatmap)
    cv2.imwrite(str(args.output / "error-overlay.png"), overlay)
    cv2.imwrite(str(args.output / "actual-crop.png"), cv2.cvtColor(actual, cv2.COLOR_RGB2BGR))
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "error_regions"}
    summary["region_count"] = len(report["error_regions"])
    print(json.dumps(summary, indent=2))
    print("Comparison files:", args.output)


if __name__ == "__main__":
    main()
