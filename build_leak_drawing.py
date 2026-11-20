"""Build a separate experimental drawing from geometry-validated leak joins.

Candidate pixels are not copied into Paint. Only short straight segments
between two existing ink anchors are allowed. Fail closed if the group
cannot be represented or does not reproduce the topology improvement.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from audit_fill_regions import FillRegionAuditor
from validate_fill_regions import FillRegionValidator
from validate_individual_leaks import IndividualLeakValidator
from drawing_sequence_planner import DrawingSequencePlanner


class LeakDrawingBuilder:
    def __init__(self, radius=3, max_component=35, max_join=9):
        self.radius, self.max_component, self.max_join = radius, max_component, max_join

    @staticmethod
    def foreground_gain(before, trial, foreground):
        after = IndividualLeakValidator.exterior(trial)
        gained = before & ~after & (trial == 0)
        return int(np.count_nonzero(gained & foreground)), int(np.count_nonzero(gained & ~foreground))

    def build(self, drawing, reference):
        ink, (ox, oy) = FillRegionAuditor().rasterize(drawing["steps"])
        h, w = ink.shape
        aligned = FillRegionValidator.reference_to_canvas(reference, drawing, (w, h), (ox, oy))
        foreground = np.min(aligned, axis=2) < 235
        before = IndividualLeakValidator.exterior(ink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.radius+1, 2*self.radius+1))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        additions = ((closed > 0) & (ink == 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(additions, connectivity=8)
        candidates = [i for i in range(1, count) if 0 < stats[i, cv2.CC_STAT_AREA] <= self.max_component]
        # Reproduce the group validator's conservative selection.
        def trial_for(ids):
            trial = ink.copy()
            for i in ids:
                trial[labels == i] = 255
            return trial
        selected = set(candidates)
        full_fg, full_bg = self.foreground_gain(before, trial_for(selected), foreground)
        if full_fg < 80 or full_fg <= 2*full_bg:
            raise ValueError("No group with sufficient foreground recovery")
        for i in sorted(selected, key=lambda k: (stats[k, cv2.CC_STAT_AREA], k), reverse=True):
            reduced = selected - {i}
            fg, bg = self.foreground_gain(before, trial_for(reduced), foreground)
            if fg >= 80 and fg > 2*bg and fg >= 0.95*full_fg:
                selected, full_fg, full_bg = reduced, fg, bg

        segments = []
        rejected = []
        for i in sorted(selected):
            mask = labels == i
            # A join must touch original ink in two distinct neighborhoods.
            ring = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            touches = (ring & (ink > 0)).astype(np.uint8)
            n, clusters, st, centers = cv2.connectedComponentsWithStats(touches, connectivity=8)
            anchors = [(int(round(centers[j][0])), int(round(centers[j][1])))
                       for j in range(1, n) if st[j, cv2.CC_STAT_AREA] > 0]
            if len(anchors) != 2 or np.linalg.norm(np.subtract(*anchors)) > self.max_join:
                rejected.append(i)
                continue
            line = np.zeros_like(ink)
            cv2.line(line, anchors[0], anchors[1], 255, 1, cv2.LINE_8)
            # Do not cross pre-existing unrelated strokes or leave the
            # candidate neighborhood. Require close raster correspondence.
            expanded = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if np.any((line > 0) & (ink == 0) & ~expanded):
                rejected.append(i)
                continue
            segments.append((i, anchors, line))

        if not segments:
            raise ValueError("No pixel candidates passed stroke-geometry validation")
        # Evaluate the real straight-line raster, not morphological pixels.
        simulated = ink.copy()
        for _, _, line in segments:
            simulated = cv2.bitwise_or(simulated, line)
        fg, bg = self.foreground_gain(before, simulated, foreground)
        if fg < 80 or fg <= 2*bg:
            raise ValueError(f"Straight-line joins fail topology check: foreground={fg}, background={bg}")
        # Minimize the representable joins while retaining the topology gain.
        for item in segments[:]:
            test = ink.copy()
            for other in segments:
                if other is not item:
                    test = cv2.bitwise_or(test, other[2])
            gain, background = self.foreground_gain(before, test, foreground)
            if gain >= 80 and gain > 2*background and gain >= 0.95*fg:
                segments.remove(item)
                fg, bg = gain, background
        result = deepcopy(drawing)
        result["name"] = drawing.get("name", "Drawing") + " - leak closure experiment"
        for i, (a, b), _ in segments:
            result["steps"].append({"name": f"Leak join {i:03d}", "tool": "stroke",
                                    "color": "black", "smooth": False, "width": 1,
                                    "points": [[a[0]+ox, a[1]+oy], [b[0]+ox, b[1]+oy]]})
        result, order = DrawingSequencePlanner().plan(result)
        return result, {"candidate_group": len(selected), "geometric_joins": len(segments),
                        "rejected_geometry": len(rejected),
                        "foreground_recovered": fg, "background_enclosed": bg,
                        "sequence": order}

    def run(self, reference_path, source_path, output_path):
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise FileNotFoundError(reference_path)
        source, output = Path(source_path).resolve(), Path(output_path).resolve()
        if source == output or output.name != "dog-leak-test.json":
            raise ValueError("Use separate canonical output drawings/dog-leak-test.json")
        drawing = json.loads(source.read_text(encoding="utf-8"))
        result, report = self.build(drawing, reference)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Created experimental drawing: {output}; original unchanged")


def main():
    parser = argparse.ArgumentParser(description="Build topology-checked leak-closure experiment")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-closure-test.json"))
    parser.add_argument("--output", type=Path, default=Path("drawings/dog-leak-test.json"))
    args = parser.parse_args()
    LeakDrawingBuilder().run(args.reference, args.drawing, args.output)


if __name__ == "__main__":
    main()
