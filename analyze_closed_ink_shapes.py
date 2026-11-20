"""Read-only closed-ink contour and hole hierarchy diagnostic.

Uses source ink topology, not width thresholds. Connected outlines can
still join semantic objects: candidates require visual verification.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from trace_reference import InkMask


class ClosedInkShapeAnalyzer:
    def __init__(self, threshold=95, min_area=35):
        self.threshold = threshold
        self.min_area = min_area

    def analyze(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        # CCOMP gives each foreground ink component an outer contour
        # and its immediate background holes. It does not invent closures.
        contours, hierarchy = cv2.findContours(
            ink.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        display = np.full((*ink.shape, 3), 255, np.uint8)
        display[ink > 0] = (0, 0, 0)
        items = []
        if hierarchy is None:
            return display, {"reference": str(reference), "candidates": [],
                             "candidate_count": 0, "hole_count": 0}
        h = hierarchy[0]
        for i, contour in enumerate(contours):
            if h[i][3] != -1:
                continue
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, height = cv2.boundingRect(contour)
            holes = []
            child = int(h[i][2])
            while child != -1:
                hole_area = float(cv2.contourArea(contours[child]))
                if hole_area >= self.min_area:
                    holes.append({"area": round(hole_area, 2),
                                  "bounds": list(map(int, cv2.boundingRect(contours[child])))})
                    cv2.drawContours(display, contours, child, (255, 0, 0), 2)
                child = int(h[child][0])
            cv2.drawContours(display, contours, i, (0, 170, 0), 2)
            perimeter = float(cv2.arcLength(contour, True))
            items.append({
                "bounds": [x, y, x+w-1, y+height-1],
                "outer_area": round(area, 2),
                "perimeter": round(perimeter, 2),
                "hole_count": len(holes),
                "holes": holes,
                "closed_contour": True,
                "status": "TOPOLOGICAL_CANDIDATE_ONLY",
            })
        return display, {
            "reference": str(reference),
            "candidate_count": len(items),
            "hole_count": sum(x["hole_count"] for x in items),
            "candidates": items,
            "warning": "Closed ink topology is not a semantic object boundary. Connected ink may contain multiple shapes. No drawing edits.",
        }

    def run(self, reference, output, report):
        display, data = self.analyze(reference)
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), display):
            raise RuntimeError(f"Cannot save {output}")
        report.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in data.items() if k != "candidates"}, indent=2))
        print(f"Saved {output} and {report}; no drawing edits.")


def main():
    p = argparse.ArgumentParser(description="Audit closed ink contours and holes")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/closed-ink-shapes-diagnostic.png"))
    p.add_argument("--report", type=Path, default=Path("output/closed-ink-shapes-report.json"))
    p.add_argument("--publish-diagnostic", action="store_true")
    a = p.parse_args()
    ClosedInkShapeAnalyzer().run(a.reference, a.output, a.report)
    if a.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(a.output)


if __name__ == "__main__":
    main()
