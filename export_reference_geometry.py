"""Export a generic geometry graph from reference ink, without semantic guesses.

The graph records all contour loops, their hierarchy, and shared connected
ink component. No Paint commands or fill operations are produced.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from trace_reference import InkMask


class ReferenceGeometryGraph:
    def __init__(self, threshold=95, simplify=0.65, min_area=2):
        self.threshold = threshold
        self.simplify = simplify
        self.min_area = min_area

    def build(self, reference):
        ink = InkMask(self.threshold).from_image(reference).astype(np.uint8)
        # TREE retains arbitrary nesting (outer boundary, hole, island...).
        contours, hierarchy = cv2.findContours(
            ink.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count, component_labels = cv2.connectedComponents(ink, connectivity=8)
        nodes = []
        index_to_id = {}
        if hierarchy is None:
            return {"schema_version": 1, "reference": str(reference),
                    "image_size": [int(ink.shape[1]), int(ink.shape[0])],
                    "nodes": [], "relations": [], "warning": "No ink detected"}
        for i, contour in enumerate(contours):
            area = abs(float(cv2.contourArea(contour)))
            if area < self.min_area or len(contour) < 3:
                continue
            epsilon = self.simplify
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            points = simplified.reshape(-1, 2).astype(int).tolist()
            if len(points) < 3:
                continue
            node_id = f"contour_{i}"
            index_to_id[i] = node_id
            x, y, w, h = cv2.boundingRect(contour)
            # Use a boundary pixel as a component identifier, rather than
            # assuming nested holes are separate foreground objects.
            px, py = map(int, contour[0, 0])
            component_id = int(component_labels[py, px])
            nodes.append({
                "id": node_id,
                "kind": "closed_contour",
                "points": points,
                "bounds": [x, y, x+w-1, y+h-1],
                "area": round(area, 2),
                "ink_component_id": component_id if component_id > 0 else None,
                "depth": None,
                "role": "UNRESOLVED",
                "semantic_object": None,
                "fill_approved": False,
            })
        relations = []
        for i, node_id in index_to_id.items():
            parent = int(hierarchy[0][i][3])
            depth = 0
            cursor = parent
            while cursor >= 0:
                depth += 1
                cursor = int(hierarchy[0][cursor][3])
            node = next(n for n in nodes if n["id"] == node_id)
            node["depth"] = depth
            node["role"] = "ink_outer_or_island" if depth % 2 == 0 else "background_hole"
            if parent in index_to_id:
                relations.append({"type": "contained_by", "from": node_id,
                                  "to": index_to_id[parent]})
        # The same ink component may span multiple meaningful objects.
        return {
            "schema_version": 1,
            "reference": str(reference),
            "image_size": [int(ink.shape[1]), int(ink.shape[0])],
            "nodes": nodes,
            "relations": relations,
            "ink_component_count": count - 1,
            "warning": "Topology only. Holes and components are not semantic shapes; no fills approved.",
        }

    def run(self, reference, output):
        data = self.build(reference)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(json.dumps({
            "nodes": len(data["nodes"]),
            "relations": len(data["relations"]),
            "ink_components": data.get("ink_component_count", 0),
            "output": str(output),
            "fill_approved": False,
        }, indent=2))


def main():
    p = argparse.ArgumentParser(description="Export reference contour geometry graph")
    p.add_argument("--reference", type=Path, default=Path("assets/dog-reference.png"))
    p.add_argument("--output", type=Path, default=Path("output/reference-geometry.json"))
    args = p.parse_args()
    ReferenceGeometryGraph().run(args.reference, args.output)


if __name__ == "__main__":
    main()
