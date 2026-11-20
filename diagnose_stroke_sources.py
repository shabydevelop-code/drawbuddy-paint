"""Separate reference-ink errors from added color-boundary geometry.

Read-only, generic diagnostic. Compares planned black-ink strokes with source
dark ink, and separately highlights color-boundary strokes and other joins.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_fill_regions import FillRegionAuditor
from validate_drawing_accuracy import DrawingAccuracyValidator


class StrokeSourceDiagnostic:
    def __init__(self, tolerance=3):
        self.accuracy = DrawingAccuracyValidator(tolerance=tolerance)
        self.auditor = FillRegionAuditor()

    def run(self, reference_path, drawing_path, output_path, report_path):
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise FileNotFoundError(reference_path)
        drawing = json.loads(Path(drawing_path).read_text(encoding="utf-8"))
        steps = drawing["steps"]
        # Use a common raster frame so separate classes stay aligned.
        full, offset = self.auditor.rasterize(steps)
        ink_steps = [s for s in steps if s.get("name", "").startswith("Ink ")]
        color_steps = [s for s in steps if s.get("name", "").startswith("Color boundary")]
        other_steps = [s for s in steps if s not in ink_steps and s not in color_steps]
        if not ink_steps:
            raise ValueError("No named Ink strokes; cannot classify source tracing")
        def mask_for(items):
            canvas = np.zeros_like(full)
            for step in items:
                points = step.get("points", [])
                if step.get("tool") not in ("stroke", "closed_path") or len(points) < 2:
                    continue
                pts = np.asarray([[x-offset[0], y-offset[1]] for x, y in points], np.int32)
                cv2.polylines(canvas, [pts], step["tool"] == "closed_path",
                              255, max(1, int(step.get("width", 1))), cv2.LINE_8)
            return canvas > 0
        ink = mask_for(ink_steps)
        color = mask_for(color_steps)
        other = mask_for(other_steps)
        # Align against the tight extent of ALL planned strokes, not just ink.
        ys, xs = np.nonzero(full)
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        reference_ink = self.accuracy.crop_and_fit(
            self.accuracy.ink(reference), (y1-y0, x1-x0))
        ink, color, other = ink[y0:y1, x0:x1], color[y0:y1, x0:x1], other[y0:y1, x0:x1]
        missing, extra, metrics = self.accuracy.compare(reference_ink, ink)
        # Source comparison is meaningful for traced ink only. Added color
        # boundaries are intentionally not judged as missing source ink.
        canvas = np.full((*ink.shape, 3), 255, np.uint8)
        canvas[reference_ink | ink] = (0, 0, 0)
        canvas[missing] = (0, 0, 255)
        canvas[extra] = (255, 0, 0)
        canvas[color & ~ink] = (0, 170, 255)  # orange: derived color boundary
        canvas[other & ~ink & ~color] = (180, 0, 180)  # purple: joins/other
        header = np.full((80, canvas.shape[1], 3), 255, np.uint8)
        lines = ("RED missing ink | BLUE extra ink",
                 "ORANGE color boundaries | PURPLE other joins",
                 "Approximate bbox alignment; no edits")
        for line, y in zip(lines, (18, 42, 65)):
            cv2.putText(header, line, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (0, 0, 0), 1)
        report = {"source_ink_comparison": metrics,
                  "stroke_counts": {"ink": len(ink_steps),
                                    "color_boundaries": len(color_steps),
                                    "other": len(other_steps)},
                  "color_boundary_pixels": int(color.sum()),
                  "other_join_pixels": int(other.sum()),
                  "warning": "Approximate alignment; no automatic geometry corrections"}
        output_path, report_path = Path(output_path), Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), np.vstack((header, canvas))):
            raise RuntimeError(f"Cannot save {output_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("No changes to drawing or Paint.")


def main():
    parser = argparse.ArgumentParser(description="Separate ink and derived-boundary discrepancies")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--drawing", type=Path, default=Path("drawings/dog-leak-test.json"))
    parser.add_argument("--output", type=Path, default=Path("output/stroke-source-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/stroke-source-report.json"))
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    StrokeSourceDiagnostic().run(args.reference, args.drawing, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
