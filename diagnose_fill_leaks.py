"""Locate possible small gaps connecting source foreground to canvas exterior.

Read-only geometry diagnostic; candidates are NOT automatic closure instructions.
Usage: python diagnose_fill_leaks.py --publish-diagnostic
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor
from validate_fill_regions import FillRegionValidator


class FillLeakDiagnostic:
    def __init__(self, radius=3, min_area=80):
        self.radius = radius
        self.min_area = min_area
        self.auditor = FillRegionAuditor()
        self.aligner = FillRegionValidator()

    def run(self, reference_path, drawing_path, output):
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(reference_path)
        ink, offset = self.auditor.rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = self.aligner.reference_to_canvas(reference, drawing, (w, h), offset)
        foreground = (np.min(aligned, axis=2) < 235).astype(np.uint8)
        # Only compare raster topology; never add lines to the drawing.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*self.radius+1, 2*self.radius+1))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        # Avoid edge artifacts in morphology and count only regions that
        # become enclosed after closing.
        def regions(mask):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (mask == 0).astype(np.uint8), connectivity=4)
            exterior = np.zeros(mask.shape, dtype=bool)
            for i in range(1, count):
                x, y, rw, rh, area = map(int, stats[i])
                if x == 0 or y == 0 or x+rw == w or y+rh == h:
                    exterior |= labels == i
            return exterior, labels, stats, count

        exterior_before, _, _, _ = regions(ink)
        exterior_after, labels, stats, count = regions(closed)
        newly_enclosed = exterior_before & ~exterior_after & (closed == 0)
        diagnostic = np.full((h, w, 3), 255, dtype=np.uint8)
        diagnostic[ink > 0] = (0, 0, 0)
        # Mark reference foreground connected to exterior before simulated closure.
        diagnostic[exterior_before & (foreground > 0)] = (210, 220, 245)
        accepted = []
        for i in range(1, count):
            x, y, rw, rh, area = map(int, stats[i])
            if area < self.min_area or x == 0 or y == 0 or x+rw == w or y+rh == h:
                continue
            component = labels == i
            if not np.any(component & newly_enclosed):
                continue
            if np.count_nonzero(component & (foreground > 0)) < self.min_area//2:
                continue
            # Pixels that closing added along the newly enclosed component boundary.
            added = (closed > 0) & (ink == 0)
            ring = cv2.dilate(component.astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
            candidates = added & ring
            if not np.any(candidates):
                continue
            ys, xs = np.nonzero(candidates)
            for px, py in zip(xs, ys):
                cv2.circle(diagnostic, (int(px), int(py)), 2, (0, 0, 255), -1)
            diagnostic[component & (foreground > 0)] = (145, 215, 150)
            # Reapply markers above the colored region.
            for px, py in zip(xs, ys):
                cv2.circle(diagnostic, (int(px), int(py)), 2, (0, 0, 255), -1)
            accepted.append({"region": int(i), "area": area,
                             "candidate_pixels": int(len(xs)),
                             "candidate_bounds": [int(xs.min()+offset[0]),
                                                  int(ys.min()+offset[1]),
                                                  int(xs.max()+offset[0]),
                                                  int(ys.max()+offset[1])]})
        header = np.full((70, w, 3), 255, dtype=np.uint8)
        cv2.putText(header, "RED: possible gaps | GREEN: newly enclosed source",
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        cv2.putText(header, "PALE BLUE: foreground leaking into exterior",
                    (4, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        cv2.putText(header, f"Simulated closing radius={self.radius}px; verify manually",
                    (4, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), np.vstack((header, diagnostic))):
            raise RuntimeError(f"Could not save {output}")
        print(f"Potentially recovered enclosed regions: {len(accepted)}")
        for item in accepted:
            print(item)
        print("No changes to drawing or Paint. Morphological closure is diagnostic only.")
        print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Find likely small exterior fill leaks")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--output", type=Path, default=Path("output/leak-diagnostic.png"))
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.radius <= 10:
        parser.error("--radius must be between 1 and 10")
    FillLeakDiagnostic(radius=args.radius).run(args.reference, args.drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
