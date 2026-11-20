"""Generic entry point for DrawBuddy drawing definitions."""
import argparse
import json
import math
from pathlib import Path
from drawbuddy.drawing_engine import DrawingEngine


def main():
    parser = argparse.ArgumentParser(description='Draw a saved lesson in Microsoft Paint.')
    parser.add_argument('drawing', help='Drawing name (e.g. dog), without .json')
    parser.add_argument('--speed', type=float, default=None, help='Stroke speed multiplier (e.g. 4, 20, 50, 100)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    drawings_dir = root / 'drawings'
    name = args.drawing
    if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name or not name.replace('-', '').replace('_', '').isalnum():
        parser.error('Drawing must be a simple name, such as dog')
    definition = drawings_dir / (name + '.json')
    if not definition.is_file():
        choices = ', '.join(sorted(p.stem for p in drawings_dir.glob('*.json'))) or '(none)'
        parser.error(f'Drawing {name!r} not found. Available: {choices}')
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    if not isinstance(config, dict):
        parser.error('config.json must contain a JSON object')
    speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
    if type(speed) not in (float, int) or not math.isfinite(speed) or speed <= 0 or speed > 100:
        parser.error('drawing speed must be greater than 0 and at most 100')
    drawing = DrawingEngine.load(definition)
    DrawingEngine(speed=float(speed)).execute(drawing, root / 'paint-diagnostic.png')


if __name__ == '__main__':
    main()
