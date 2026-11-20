"""Build an experimental Paint drawing with joins integrated into source paths.

Usage: python build_closure_drawing.py
Writes drawings/dog-closure-test.json; never modifies drawings/dog.json.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from validate_gap_closures import GapClosureValidator
from drawing_sequence_planner import DrawingSequencePlanner


class ClosureDrawingBuilder:
    def __init__(self):
        self.validator = GapClosureValidator()

    def build(self, drawing):
        accepted, rejected, original, offset = self.validator.evaluate(drawing)
        if not accepted:
            raise ValueError("No accepted joins; refusing to create an unchanged test drawing")
        updated = deepcopy(drawing)
        updated["name"] = f"{drawing.get('name', 'Drawing')} - closure test"
        # Attach each accepted join to its original color-boundary stroke.
        # This preserves step order and avoids revisiting the drawing at the end.
        remaining = {tuple(start): tuple(end)
                     for _, start, end, *_ in accepted}
        if len(remaining) != len(accepted):
            raise ValueError("Ambiguous duplicate closure endpoints")
        integrated = 0
        for step in updated["steps"]:
            if step.get("tool") != "stroke" or not step.get("name", "").startswith("Color boundary"):
                continue
            points = step["points"]
            if len(points) < 2 or points[0] == points[-1]:
                continue
            first, last = tuple(points[0]), tuple(points[-1])
            first_target = remaining.pop(first, None)
            last_target = remaining.pop(last, None)
            if first_target is not None:
                if first_target != first:
                    points.insert(0, list(first_target))
                integrated += 1
            if last_target is not None:
                if last_target != last:
                    points.append(list(last_target))
                integrated += 1
            if first_target is not None or last_target is not None:
                # Smoothing can curve a short closing segment away from its
                # validated straight-line geometry.
                step["smooth"] = False
        if remaining or integrated != len(accepted):
            raise ValueError(f"Unmatched closure endpoints: {sorted(remaining)}")
        updated, zones = DrawingSequencePlanner().plan(updated)
        print('Drawing order: ' + ' -> '.join(f'{name} ({count})' for name, count in zones.items()))
        return updated, len(accepted), len(rejected)

    def run(self, source, output):
        source = Path(source).resolve()
        output = Path(output).resolve()
        if source == output:
            raise ValueError("Output must differ from original drawing")
        if output.name != "dog-closure-test.json":
            raise ValueError("Use the canonical test filename dog-closure-test.json")
        drawing = json.loads(source.read_text(encoding="utf-8"))
        result, accepted, rejected = self.build(drawing)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Created: {output}")
        print(f"Accepted joins integrated into original paths: {accepted}; rejected excluded: {rejected}")
        print(f"Original unchanged: {source}")
        print("Experimental drawing only; validate the Paint screenshot before accepting.")


def main():
    parser = argparse.ArgumentParser(description="Build separate Paint closure test")
    parser.add_argument("--source", type=Path, default=Path("drawings/dog.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("drawings/dog-closure-test.json"))
    args = parser.parse_args()
    ClosureDrawingBuilder().run(args.source, args.output)


if __name__ == "__main__":
    main()
