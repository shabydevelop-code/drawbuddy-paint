"""Read-only skeleton branch segmentation diagnostic for reference ink.

Segments are cut at junctions and measured independently. This is not a
semantic fill classifier and never changes Paint or drawing JSON.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from trace_reference import InkMask, ZhangSuenThinner


class InkBranchSegmenter:
    def __init__(self, threshold=95, junction_radius=2, min_pixels=5):
        self.threshold = threshold
        self.junction_radius = junction_radius
        self.min_pixels = min_pixels

    def analyze(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        skeleton = ZhangSuenThinner().thin(ink)
        # Count 8-neighbors. A junction has at least three; suppress diagonal
        # shortcuts at corners by counting 0->1 neighbor transitions too.
        padded = np.pad(skeleton, 1)
        neighbors = [padded[:-2, 1:-1], padded[:-2, 2:],
                     padded[1:-1, 2:], padded[2:, 2:],
                     padded[2:, 1:-1], padded[2:, :-2],
                     padded[1:-1, :-2], padded[:-2, :-2]]
        degree = sum(neighbors)
        transitions = sum(((neighbors[k] == 0) &
                           (neighbors[(k+1) % 8] == 1))
                          for k in range(8))
        junctions = ((skeleton > 0) & (degree >= 3) &
                     (transitions >= 3)).astype(np.uint8)
        size = 2*self.junction_radius+1
        exclusion = cv2.dilate(junctions, np.ones((size, size), np.uint8))
        segments_mask = skeleton.copy()
        segments_mask[exclusion > 0] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            segments_mask, connectivity=8)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        # Pixelwise output: white background, black ink, blue junctions,
        # green thicker segments, red thin segments.
        display = np.full((*ink.shape, 3), 255, dtype=np.uint8)
        display[ink > 0] = (0, 0, 0)
        display[skeleton > 0] = (0, 0, 255)
        items = []
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if area < self.min_pixels:
                continue
            component = labels[y:y+h, x:x+w] == i
            widths = 2.0 * distance[y:y+h, x:x+w][component]
            p10, p50, p90 = np.percentile(widths, [10, 50, 90])
            variation = float(p90 / max(p10, 1.0))
            # Thickness is descriptive, not proof of a filled shape.
            thick = bool(p50 >= 2 * 5)
            roi = display[y:y+h, x:x+w]
            roi[component] = (0, 150, 0) if thick else (0, 0, 255)
            items.append({
                "bounds": [x, y, x+w-1, y+h-1],
                "skeleton_pixels": area,
                "width_p10": round(float(p10), 2),
                "width_median": round(float(p50), 2),
                "width_p90": round(float(p90), 2),
                "width_variation": round(variation, 3),
                "category": "THICK_BRANCH_REVIEW" if thick else "THIN_BRANCH",
            })
        display[junctions > 0] = (255, 0, 0)
        return display, {
            "reference": str(reference),
            "junction_count_pixels": int(junctions.sum()),
            "segment_count": len(items),
            "thick_review_count": sum(x["category"] == "THICK_BRANCH_REVIEW"
                                      for x in items),
            "segments": items,
            "warning": "Skeleton branches are not semantic regions. Green means thick branch only; never auto-fill.",
        }

    def run(self, reference, output, report):
        display, data = self.analyze(reference)
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), display):
            raise RuntimeError(f"Cannot save {output}")
        report.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in data.items() if k != "segments"},
                         indent=2))
        print(f"Saved {output} and {report}; no drawing edits.")


def main():
    p = argparse.ArgumentParser(description="Segment reference skeleton at junctions")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/ink-branch-diagnostic.png"))
    p.add_argument("--report", type=Path, default=Path("output/ink-branch-report.json"))
    p.add_argument("--publish-diagnostic", action="store_true")
    args = p.parse_args()
    InkBranchSegmenter().run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
