"""Classify compact solid-ink cores separately from long thick strokes.

Read-only heuristic. A green region is a candidate for later validation,
not permission to fill it in Paint.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from trace_reference import InkMask


class SolidRegionClassifier:
    def __init__(self, threshold=95, core_radius=5, min_core_area=18,
                 max_aspect=3.0, min_compactness=0.14):
        self.threshold = threshold
        self.core_radius = core_radius
        self.min_core_area = min_core_area
        self.max_aspect = max_aspect
        self.min_compactness = min_compactness

    def classify(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        core = (distance >= self.core_radius).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        approved = np.zeros_like(ink)
        rejected = np.zeros_like(ink)
        items = []
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if area < self.min_core_area:
                continue
            component = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
            compactness = 4*np.pi*area/(perimeter*perimeter) if perimeter else 0
            aspect = max(w,h)/max(1,min(w,h))
            # A long thick line can contain a broad core. Reject by both
            # elongation and boundary compactness, not width alone.
            plausible = aspect <= self.max_aspect and compactness >= self.min_compactness
            target = approved if plausible else rejected
            target[y:y+h, x:x+w][component > 0] = 1
            items.append({"bounds": [x,y,x+w-1,y+h-1],
                          "core_area": area,
                          "aspect_ratio": round(aspect,3),
                          "compactness": round(float(compactness),3),
                          "max_half_width": round(float(distance[y:y+h,x:x+w][component>0].max()),2),
                          "status": "REVIEW_COMPACT_CORE" if plausible else "REJECT_ELONGATED_CORE"})
        return ink, approved, rejected, items

    def run(self, reference, output, report):
        ink, approved, rejected, items = self.classify(reference)
        display = np.full((*ink.shape,3),255,np.uint8)
        display[ink>0] = (0,0,0)
        display[rejected>0] = (0,0,255)
        display[approved>0] = (0,160,0)
        # Cores are intentionally shown, not complete source shapes:
        # region growth could cross into connected outline strokes.
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True,exist_ok=True)
        report.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(output),display):
            raise RuntimeError(f"Cannot save {output}")
        summary = {"reference":str(reference),"core_radius":self.core_radius,
                   "review_candidates":sum(i["status"]=="REVIEW_COMPACT_CORE" for i in items),
                   "rejected":sum(i["status"]=="REJECT_ELONGATED_CORE" for i in items),
                   "regions":items,
                   "warning":"Only compact cores classified. Not full fill masks; no drawing edits."}
        report.write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2))
        print(f"Saved {output}; no Paint or drawing changes.")


def main():
    p=argparse.ArgumentParser(description="Classify compact black-ink cores")
    p.add_argument("--reference",type=Path,default=Path("assets/dog-reference.png"))
    p.add_argument("--output",type=Path,default=Path("output/solid-classifier-diagnostic.png"))
    p.add_argument("--report",type=Path,default=Path("output/solid-classifier-report.json"))
    p.add_argument("--publish-diagnostic",action="store_true")
    a=p.parse_args()
    SolidRegionClassifier().run(a.reference,a.output,a.report)
    if a.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(a.output)


if __name__=="__main__":
    main()
