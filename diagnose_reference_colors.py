"""Visualize why source-color regions were not filled.

Usage: python diagnose_reference_colors.py --publish-diagnostic
No Paint modifications. Uses the same raster and acceptance thresholds as planner.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from plan_reference_colors import ReferenceColorPlanner


def run(reference_path, drawing_path, output):
    planner = ReferenceColorPlanner()
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise FileNotFoundError(reference_path)
    drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
    ink, offset = planner.auditor.rasterize(drawing["steps"])
    h, w = ink.shape
    aligned = planner.aligner.reference_to_canvas(reference, drawing, (w, h), offset)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (ink == 0).astype(np.uint8), connectivity=4)
    # BGR diagnostic colors; white = unclassified tiny area.
    colors = {"open/exterior": (210, 220, 245), "no source": (0, 220, 255),
              "mixed/shaded": (255, 100, 200), "unsafe seed": (90, 90, 255),
              "fill candidate": (150, 225, 150)}
    counts = {key: 0 for key in colors}
    diagnostic = np.full((h, w, 3), 255, dtype=np.uint8)
    for i in range(1, count):
        x, y, rw, rh, area = map(int, stats[i])
        if area < planner.min_area:
            continue
        component = (components == i).astype(np.uint8)
        if x == 0 or y == 0 or x+rw == w or y+rh == h:
            status = "open/exterior"
        else:
            inner = cv2.erode(component, np.ones((5, 5), np.uint8)) > 0
            pixels = aligned[inner]
            selected = pixels[np.min(pixels, axis=1) < 235]
            coverage = len(selected)/max(1, int(inner.sum()))
            if len(selected) < 20 or coverage < planner.minimum_coverage:
                status = "no source"
            else:
                median = np.median(selected[:, ::-1], axis=0)
                spread = float(np.percentile(
                    np.linalg.norm(selected[:, ::-1].astype(float)-median, axis=1), 75))
                if spread > planner.max_spread:
                    status = "mixed/shaded"
                else:
                    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
                    status = "unsafe seed" if float(distance.max()) < 3 else "fill candidate"
        counts[status] += 1
        diagnostic[component > 0] = colors[status]
    diagnostic[ink > 0] = (0, 0, 0)
    header = np.full((78, w, 3), 255, dtype=np.uint8)
    cv2.putText(header, "GREEN fill | YELLOW no source | PINK mixed/shaded",
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1)
    cv2.putText(header, "PALE BLUE open/exterior | RED unsafe seed",
                (4, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1)
    cv2.putText(header, "Raster simulation; Paint behavior may differ",
                (4, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack((header, diagnostic))):
        raise RuntimeError(f"Cannot save: {output}")
    print("Region diagnostics:", counts)
    print("Saved:", output)
    print("No Paint edits performed.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose source-color fill candidates")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--output", type=Path, default=Path("output/color-diagnostic.png"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    run(args.reference, args.drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
