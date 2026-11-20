"""Validate reconstructed ink candidates against source pixels.

Pixel agreement is measurable; semantic completeness is not. This read-only
audit never approves regions for drawing or filling automatically.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from reconstruct_ink_regions import BoundedInkReconstructor
from trace_reference import InkMask


class InkReconstructionValidator:
    def __init__(self, threshold=95):
        self.threshold = threshold

    def analyze(self, reference):
        reconstructed_view, source_report = BoundedInkReconstructor(
            threshold=self.threshold).analyze(reference)
        ink = InkMask(self.threshold).from_image(reference).astype(bool)
        # Reconstruction uses these exact diagnostic colors in BGR.
        seed = np.all(reconstructed_view == (0, 220, 255), axis=2)
        grown = np.all(reconstructed_view == (255, 200, 0), axis=2)
        reconstructed = seed | grown
        false_positive = reconstructed & ~ink
        # "Missing" is defined within a candidate's local neighborhood,
        # not against an unknown semantic ground-truth shape.
        display = np.full((*ink.shape, 3), 255, np.uint8)
        display[ink] = (0, 0, 0)
        display[reconstructed] = (0, 180, 0)
        display[false_positive] = (0, 0, 255)
        items = []
        for candidate in source_report["candidates"]:
            x0, y0, x1, y1 = candidate["seed_bounds"]
            radius = candidate["max_growth_pixels"]
            x0, y0 = max(0, x0-radius), max(0, y0-radius)
            x1 = min(ink.shape[1], x1+radius+1)
            y1 = min(ink.shape[0], y1+radius+1)
            local_ink = ink[y0:y1, x0:x1]
            local_recon = reconstructed[y0:y1, x0:x1]
            # Multiple candidate neighborhoods can overlap: these figures
            # are local context metrics, not exclusive per-object scores.
            missed = local_ink & ~local_recon
            excess = local_recon & ~local_ink
            # Source boundary support: what fraction of reconstructed edge
            # touches source background? Low support implies truncation or
            # an interior cut, not necessarily a bad semantic region.
            mask = local_recon.astype(np.uint8)
            edge = (mask > 0) & (cv2.erode(mask, np.ones((3,3), np.uint8)) == 0)
            source_edge = (local_ink & (cv2.erode(
                local_ink.astype(np.uint8), np.ones((3,3), np.uint8)) == 0))
            boundary_support = int(np.count_nonzero(edge & source_edge))
            edge_pixels = int(np.count_nonzero(edge))
            items.append({
                "seed_bounds": candidate["seed_bounds"],
                "local_reference_ink_pixels": int(np.count_nonzero(local_ink)),
                "local_reconstructed_pixels": int(np.count_nonzero(local_recon)),
                "local_missing_ink_pixels": int(np.count_nonzero(missed)),
                "local_excess_pixels": int(np.count_nonzero(excess)),
                "boundary_source_support_fraction": round(
                    boundary_support / max(1, edge_pixels), 3),
                "note": "Local window may contain adjacent outlines; missing pixels are not necessarily part of this candidate.",
            })
        return display, {
            "reference": str(reference),
            "candidate_count": len(items),
            "reconstructed_pixels": int(np.count_nonzero(reconstructed)),
            "excess_pixels_outside_source_ink": int(np.count_nonzero(false_positive)),
            "candidates": items,
            "decision": "REVIEW_REQUIRED",
            "warning": "No semantic ground-truth masks exist. Pixel metrics cannot certify full shape boundaries or authorize filling.",
        }

    def run(self, reference, output, report):
        display, data = self.analyze(reference)
        output, report = Path(output), Path(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), display):
            raise RuntimeError(f"Cannot save {output}")
        report.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in data.items() if k != "candidates"},
                         indent=2))
        print(f"Saved {output} and {report}; no drawing changes.")


def main():
    p = argparse.ArgumentParser(description="Audit bounded ink reconstruction")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/ink-validation-diagnostic.png"))
    p.add_argument("--report", type=Path, default=Path("output/ink-validation-report.json"))
    p.add_argument("--publish-diagnostic", action="store_true")
    args = p.parse_args()
    InkReconstructionValidator().run(args.reference, args.output, args.report)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
