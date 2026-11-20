"""Build an experimental Paint drawing from validated gap joins.

Usage: python build_closure_drawing.py
Writes drawings/dog-closure-test.json; never modifies drawings/dog.json.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from validate_gap_closures import GapClosureValidator


class ClosureDrawingBuilder:
    def __init__(self):
        self.validator = GapClosureValidator()

    def build(self, drawing):
        accepted, rejected, original, offset = self.validator.evaluate(drawing)
        if not accepted:
            raise ValueError("No accepted joins; refusing to create an unchanged test drawing")
        updated = deepcopy(drawing)
        updated["name"] = f"{drawing.get('name', 'Drawing')} - closure test"
        for index, (_, start, end, *_) in enumerate(accepted, 1):
            updated["steps"].append({
                "name": f"Validated closure {index:03d}",
                "tool": "stroke",
                "color": "black",
                "smooth": False,
                "width": 1,
                "points": [list(map(int, start)), list(map(int, end))]
            })
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
        print(f"Accepted joins added: {accepted}; rejected joins excluded: {rejected}")
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
