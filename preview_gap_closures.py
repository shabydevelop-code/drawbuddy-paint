"""Preview validated gap joins and compare enclosed fill regions, without editing JSON.

Usage: python preview_gap_closures.py drawings/dog.json --publish-preview
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from validate_gap_closures import GapClosureValidator


class ClosurePreview:
    def __init__(self):
        self.validator = GapClosureValidator()

    def run(self, drawing, output):
        accepted, rejected, original, offset = self.validator.evaluate(drawing)
        ox, oy = offset
        preview = original.copy()
        for _, start, end, *_ in accepted:
            a = (int(start[0]-ox), int(start[1]-oy))
            b = (int(end[0]-ox), int(end[1]-oy))
            cv2.line(preview, a, b, 255, 1, cv2.LINE_8)
        before = self.validator.region_count(original)
        after = self.validator.region_count(preview)
        # Three panels: original, proposed connections, and post-join linework.
        original_view = cv2.cvtColor(255-original, cv2.COLOR_GRAY2BGR)
        preview_view = cv2.cvtColor(255-preview, cv2.COLOR_GRAY2BGR)
        marked = original_view.copy()
        for _, start, end, *_ in accepted:
            cv2.line(marked, (int(start[0]-ox), int(start[1]-oy)),
                     (int(end[0]-ox), int(end[1]-oy)), (0, 170, 0), 2)
        panels = []
        for label, image in (
                (f"BEFORE: {before} regions", original_view),
                (f"PROPOSED: {len(accepted)} joins", marked),
                (f"AFTER: {after} regions", preview_view)):
            header = np.full((30, image.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(header, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.53, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(np.vstack((header, image)))
        result = np.hstack(panels)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result):
            raise RuntimeError(f"Cannot save preview: {output}")
        print(f"Accepted candidate joins: {len(accepted)}; rejected: {len(rejected)}")
        print(f"Enclosed regions before: {before}; after: {after}; change: {after-before:+d}")
        print(f"Saved: {output}")
        print("Preview only: no Paint actions, no JSON modifications.")
        print("Simulated topology is not proof of intended color boundaries.")


def main():
    parser = argparse.ArgumentParser(description="Preview accepted joins and fill regions")
    parser.add_argument("drawing", nargs="?", type=Path,
                        default=Path("drawings/dog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/closure-preview.png"))
    parser.add_argument("--publish-preview", action="store_true")
    args = parser.parse_args()
    drawing = json.loads(args.drawing.read_text(encoding="utf-8"))
    ClosurePreview().run(drawing, args.output)
    if args.publish_preview:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
