"""Test groups of small morphology-derived joins without modifying drawing.

Starts with the combined candidate mask, then greedily removes unnecessary
components while preserving recovered foreground. A heuristic, not a proof.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_fill_regions import FillRegionAuditor
from validate_fill_regions import FillRegionValidator
from validate_individual_leaks import IndividualLeakValidator


class GroupLeakValidator:
    def __init__(self, radius=3, max_component_pixels=35, min_foreground=80):
        self.radius = radius
        self.max_component_pixels = max_component_pixels
        self.min_foreground = min_foreground

    def run(self, reference_path, drawing_path, output):
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise FileNotFoundError(reference_path)
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        ink, (ox, oy) = FillRegionAuditor().rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = FillRegionValidator.reference_to_canvas(
            reference, drawing, (w, h), (ox, oy))
        foreground = np.min(aligned, axis=2) < 235
        before = IndividualLeakValidator.exterior(ink)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*self.radius+1, 2*self.radius+1))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        additions = ((closed > 0) & (ink == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            additions, connectivity=8)
        candidates = [i for i in range(1, count)
                      if 0 < stats[i, cv2.CC_STAT_AREA] <= self.max_component_pixels]
        def evaluate(ids):
            trial = ink.copy()
            for i in ids:
                trial[labels == i] = 255
            after = IndividualLeakValidator.exterior(trial)
            gained = before & ~after & (trial == 0)
            fg = int(np.count_nonzero(gained & foreground))
            bg = int(np.count_nonzero(gained & ~foreground))
            return fg, bg, gained

        selected = set(candidates)
        full_fg, full_bg, _ = evaluate(selected)
        # Only reduce groups that show meaningful recovery without mostly
        # enclosing background. Otherwise report no safe group.
        valid = full_fg >= self.min_foreground and full_fg > 2*full_bg
        if valid:
            # Removing components one by one reveals combinations that are
            # necessary collectively, even if none works in isolation.
            for i in sorted(selected, key=lambda k: (stats[k, cv2.CC_STAT_AREA], k), reverse=True):
                trial = selected - {i}
                fg, bg, _ = evaluate(trial)
                if fg >= self.min_foreground and fg > 2*bg and fg >= 0.95*full_fg:
                    selected = trial
                    full_fg, full_bg = fg, bg
        else:
            selected.clear()
        image = np.full((h, w, 3), 255, dtype=np.uint8)
        image[before & foreground] = (210, 220, 245)
        image[ink > 0] = (0, 0, 0)
        for i in candidates:
            image[labels == i] = (0, 160, 0) if i in selected else (0, 0, 255)
        if selected:
            _, _, gained = evaluate(selected)
            image[gained & foreground] = (150, 225, 150)
            for i in selected:
                image[labels == i] = (0, 160, 0)
        header = np.full((72, w, 3), 255, dtype=np.uint8)
        for line, y in (("GREEN: required join group | RED: excluded candidate", 20),
                        ("PALE BLUE: foreground still connected to exterior", 43),
                        ("Simulation only: no drawing modifications", 64)):
            cv2.putText(header, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 0), 1)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), np.vstack((header, image))):
            raise RuntimeError(f"Cannot save {output}")
        report = {"candidates": len(candidates), "selected": len(selected),
                  "foreground_recovered": full_fg if valid else 0,
                  "background_enclosed": full_bg if valid else 0,
                  "status": "REVIEW_GROUP" if selected else "NO_SAFE_GROUP",
                  "joins": [{"bounds": [int(stats[i, cv2.CC_STAT_LEFT]+ox),
                                       int(stats[i, cv2.CC_STAT_TOP]+oy),
                                       int(stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH]-1+ox),
                                       int(stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT]-1+oy)],
                             "pixels": int(stats[i, cv2.CC_STAT_AREA])}
                            for i in sorted(selected)]}
        print(json.dumps(report, indent=2))
        print("Candidate groups are hypothetical; no edits to Paint or drawing.")
        print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate groups of possible fill leak closures")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--output", type=Path, default=Path("output/group-leaks.png"))
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.radius <= 10:
        parser.error("--radius must be between 1 and 10")
    GroupLeakValidator(radius=args.radius).run(args.reference, args.drawing, args.output)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
