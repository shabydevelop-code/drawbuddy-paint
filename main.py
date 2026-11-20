"""Generic entry point for DrawBuddy drawing definitions."""
import argparse
import json
import math
from pathlib import Path
from drawbuddy.drawing_engine import DrawingEngine
from drawbuddy.result_publisher import ResultPublisher
from compare_images import compare_paint_canvas
from drawbuddy.binary_painter import build_binary_drawing, build_region_drawing, build_outline_drawing, build_color_raster_drawing


def main():
    parser = argparse.ArgumentParser(description='Draw a saved lesson in Microsoft Paint.')
    parser.add_argument('drawing', nargs='?', help='Drawing name (e.g. dog), without .json')
    parser.add_argument('--speed', type=float, default=None, help='Stroke speed multiplier (e.g. 4, 20, 50, 100)')
    parser.add_argument('--no-publish-result', action='store_true', help='Skip automatic Paint canvas capture and GitHub upload')
    parser.add_argument('--binary-source', type=Path, help='Print the source as 0/1 horizontal pixel runs (default)')
    parser.add_argument('--binary-fill', action='store_true', help='Experimental contour and bucket fill mode (unsafe until validated)')
    parser.add_argument('--binary-runs', action='store_true', help='Explicitly select the default 0/1 row-by-row printing')
    parser.add_argument('--binary-outlines', action='store_true', help='Use the previous experimental contour drawing')
    parser.add_argument('--color-raster', action='store_true', help='Print RGB colors as separate 0/1 raster runs')
    parser.add_argument('--no-compare', action='store_true', help='Skip automatic Paint vs source comparison')
    parser.add_argument('--binary-threshold', type=int, default=None, help='Optional grayscale cutoff 0..255 (default: Otsu)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    drawings_dir = root / 'drawings'
    if args.binary_source:
        if args.drawing:
            parser.error('Specify either a drawing name or --binary-source, not both')
        if args.binary_threshold is not None and not 0 <= args.binary_threshold <= 255:
            parser.error('--binary-threshold must be 0..255')
        if sum((args.binary_runs, args.binary_fill, args.binary_outlines, args.color_raster)) > 1:
            parser.error('Choose only one raster/outline mode')
        planner = (build_binary_drawing if args.binary_runs else
                   build_region_drawing if args.binary_fill else
                   build_outline_drawing if args.binary_outlines else build_color_raster_drawing)
        if planner is build_color_raster_drawing:
            drawing = planner(args.binary_source)
        else:
            drawing = planner(args.binary_source, threshold=args.binary_threshold)
        if "binary" in drawing:
            print(f'Binary 0/1: {drawing["binary"]}')
        elif "color" in drawing:
            print(f'Color raster: {drawing["color"]}')
    else:
        name = args.drawing
        if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\\\' in name or not name.replace('-', '').replace('_', '').isalnum():
            parser.error('Drawing must be a simple name, such as dog')
        definition = drawings_dir / (name + '.json')
        if not definition.is_file():
            choices = ', '.join(sorted(p.stem for p in drawings_dir.glob('*.json'))) or '(none)'
            parser.error(f'Drawing {name!r} not found. Available: {choices}')
        drawing = DrawingEngine.load(definition)
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    if not isinstance(config, dict):
        parser.error('config.json must contain a JSON object')
    speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
    if type(speed) not in (float, int) or not math.isfinite(speed) or speed <= 0 or speed > 100:
        parser.error('drawing speed must be greater than 0 and at most 100')
    engine = DrawingEngine(speed=float(speed))
    engine.execute(drawing, root / 'paint-diagnostic.png')
    if args.binary_source and not args.no_compare:
        # Compare the actual pixels against the resized source, not the
        # quantized palette preview. The difference includes palette loss.
        info = drawing.get('color') or drawing.get('binary')
        if not info or 'width' not in info or 'height' not in info:
            raise RuntimeError('Raster drawing dimensions missing')
        width, height = info['width'], info['height']
        max_width, max_height = 240, 180
        left = 310 + (max_width - width) // 2
        top = 155 + (max_height - height) // 2
        compare_paint_canvas(
            engine.paint, root / 'output' / 'raster-source-color.png',
            width, height, left, top,
            output=root / 'output' / 'comparison'
        )
    if not args.no_publish_result:
        publisher = ResultPublisher(root)
        publisher.capture(engine.paint)
        publisher.publish()


if __name__ == '__main__':
    main()
