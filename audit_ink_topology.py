"""Read-only topology audit of source ink and its thinned skeleton.

Uses dual connectivity (8-connected ink, 4-connected background) to count
components and holes. Endpoint/junction counts are diagnostic estimates.
No drawing or Paint changes.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from trace_reference import InkMask, ZhangSuenThinner


class InkTopologyAudit:
    @staticmethod
    def topology(mask):
        binary = (mask > 0).astype(np.uint8)
        components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Pad so the outside background is unambiguously one component.
        padded = np.pad(1 - binary, 1, constant_values=1)
        background_count, _, _, _ = cv2.connectedComponentsWithStats(
            padded, connectivity=4)
        return {
            "components": int(components - 1),
            "holes": int(background_count - 2),
            "euler_characteristic": int(components - background_count + 1),
            "isolated_components_under_12_pixels": int(sum(
                stats[i, cv2.CC_STAT_AREA] < 12 for i in range(1, components)))
        }

    @staticmethod
    def skeleton_nodes(mask):
        binary = (mask > 0).astype(np.uint8)
        padded = np.pad(binary, 1)
        neighbors = [padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                     padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                     padded[1:-1, :-2], padded[:-2, :-2]]
        # Count transitions around each pixel rather than raw 8-neighbor
        # degree, which overcounts branches near diagonal contacts.
        transitions = np.zeros_like(binary, dtype=np.uint8)
        for k in range(8):
            transitions += ((neighbors[k] == 0) &
                            (neighbors[(k+1) % 8] == 1)).astype(np.uint8)
        endpoint = (binary > 0) & (transitions == 1)
        junction = (binary > 0) & (transitions >= 3)
        isolated = (binary > 0) & (sum(neighbors) == 0)
        return endpoint, junction, isolated

    def run(self, reference_path, output_path, report_path, threshold=95):
        mask = InkMask(threshold).from_image(reference_path) > 0
        skeleton = ZhangSuenThinner().thin(mask.astype(np.uint8)) > 0
        source_topology = self.topology(mask)
        skeleton_topology = self.topology(skeleton)
        ends, junctions, isolated = self.skeleton_nodes(skeleton)
        report = {
            "threshold": threshold,
            "source_ink": source_topology,
            "skeleton": skeleton_topology,
            "delta": {
                key: skeleton_topology[key] - source_topology[key]
                for key in ("components", "holes", "euler_characteristic")
            },
            "skeleton_node_estimates": {
                "endpoints": int(ends.sum()),
                "junction_pixels": int(junctions.sum()),
                "isolated_pixels": int(isolated.sum())
            },
            "interpretation": (
                "Nonzero component/hole deltas flag possible thinning topology changes. "
                "Zero deltas do not prove local geometry preservation. "
                "Junction pixels are not distinct junction clusters. "
                "Filled dark areas are deliberately reduced to centerlines."
            )
        }
        h, w = mask.shape
        left = np.full((h, w, 3), 255, dtype=np.uint8)
        left[mask] = (0, 0, 0)
        right = np.full((h, w, 3), 255, dtype=np.uint8)
        right[skeleton] = (0, 0, 0)
        # Mark topology-sensitive skeleton nodes for visual inspection.
        right[ends] = (0, 0, 255)
        right[junctions] = (255, 0, 0)
        right[isolated] = (0, 165, 255)
        combined = np.concatenate([left, right], axis=1)
        header = np.full((66, combined.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, "LEFT: thresholded ink | RIGHT: skeleton",
                    (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        cv2.putText(header, "RED endpoints | BLUE junction pixels | ORANGE isolated",
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1)
        output_path, report_path = Path(output_path), Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), np.vstack([header, combined])):
            raise RuntimeError(f"Cannot write {output_path}")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("Read-only topology audit; no Paint or drawing edits.")


def main():
    parser = argparse.ArgumentParser(description="Compare ink and skeleton topology")
    parser.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    parser.add_argument("--output", type=Path, default=Path("output/ink-topology-diagnostic.png"))
    parser.add_argument("--report", type=Path, default=Path("output/ink-topology-report.json"))
    parser.add_argument("--threshold", type=int, default=95)
    parser.add_argument("--publish-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    InkTopologyAudit().run(args.reference, args.output, args.report, args.threshold)
    if args.publish_diagnostic:
        from drawbuddy.result_publisher import ResultPublisher
        ResultPublisher(Path(__file__).resolve().parent).publish_file(args.output)


if __name__ == "__main__":
    main()
