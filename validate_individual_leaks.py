"""Test each morphology-derived gap candidate independently.

Only candidates that actually disconnect source foreground from the exterior
are reported as plausible. This is a read-only diagnostic, not a drawing edit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_fill_regions import FillRegionAuditor
from validate_fill_regions import FillRegionValidator


class IndividualLeakValidator:
    def __init__(self, radius=3, min_gain=80, max_added=35):
        self.radius, self.min_gain, self.max_added = radius, min_gain, max_added

    @staticmethod
    def exterior(ink):
        h, w = ink.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (ink == 0).astype(np.uint8), connectivity=4)
        exterior = np.zeros((h, w), dtype=bool)
        for i in range(1, count):
            x, y, rw, rh, _ = map(int, stats[i])
            if x == 0 or y == 0 or x+rw == w or y+rh == h:
                exterior |= labels == i
        return exterior

    def run(self, reference_path, drawing_path, output):
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise FileNotFoundError(reference_path)
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        auditor = FillRegionAuditor()
        ink, (ox, oy) = auditor.rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = FillRegionValidator.reference_to_canvas(
            reference, drawing, (w, h), (ox, oy))
        foreground = np.min(aligned, axis=2) < 235
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*self.radius+1, 2*self.radius+1))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        added = ((closed > 0) & (ink == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            added, connectivity=8)
        before = self.exterior(ink)
        image = np.full((h, w, 3), 255, dtype=np.uint8)
        image[ink > 0] = (0, 0, 0)
        image[before & foreground] = (210, 220, 245)
        results = []
        for i in range(1, count):
            x, y, rw, rh, area = map(int, stats[i])
            if area > self.max_added:
                continue
            candidate = labels == i
            trial = ink.copy()
            trial[candidate] = 255
            after = self.exterior(trial)
            gained = before & ~after & (trial == 0)
            fg_gain = int(np.count_nonzero(gained & foreground))
            bg_gain = int(np.count_nonzero(gained & ~foreground))
            # Reject additions that do not fix a leak, or that enclose a
            # predominantly background area. A successful candidate is
            # still only a hypothesis requiring visual verification.
            plausible = fg_gain >= self.min_gain and fg_gain > 2*bg_gain
            if fg_gain == 0 and bg_gain == 0:
                continue
            color = (0, 165, 0) if plausible else (0, 0, 255)
            image[candidate] = color
            results.append({"bounds": [x+ox, y+oy, x+rw-1+ox, y+rh-1+oy],
                            "added_pixels": area, "foreground_recovered": fg_gain,
                            "background_enclosed": bg_gain,
                            "status": "PLAUSIBLE" if plausible else "REJECT"})
        header = np.full((65, w, 3), 255, dtype=np.uint8)
        cv2.putText(header, "GREEN: individually plausible  RED: rejected",
                    (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(header, "PALE BLUE: source foreground connected to exterior",
                    (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), np.vstack((header, image))):
            raise RuntimeError(f"Cannot write {output}")
        print(f"Individually evaluated: {len(results)}; plausible: {sum(x['status']=='PLAUSIBLE' for x in results)}")
        for item in results:
            print(json.dumps(item))
        print("No joins applied; morphology may still alter intended linework.")
        print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Validate individual candidate fill leak closures")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--output", type=Path, default=Path("output/individual-leaks.png"))
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.radius <= 10:
        parser.error("--radius must be between 1 and 10")
    IndividualLeakValidator(radius=args.radius).run(args.reference, args.drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
