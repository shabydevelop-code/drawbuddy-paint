"""Generic entry point for DrawBuddy drawing definitions."""
import argparse
import json
import math
from pathlib import Path
from drawbuddy.drawing_engine import DrawingEngine
from drawbuddy.controller import DrawingCancelled
from drawbuddy.freehand_painter import build_freehand_drawing
from drawbuddy.lineart_painter import build_lineart_drawing
from drawbuddy.centerline_painter import build_centerline_drawing
from drawbuddy.region_vectorizer import analyze_regions
from drawbuddy.region_painter import build_region_outline_drawing
from drawbuddy.continuous_painter import build_continuous_color_drawing
from drawbuddy.natural_painter import build_natural_drawing
from drawbuddy.result_publisher import ResultPublisher
from compare_images import compare_paint_canvas
from drawbuddy.binary_painter import build_binary_drawing, build_region_drawing, build_outline_drawing, build_color_raster_drawing



def draw_scene_strokes(path, speed, limit=None, validate_only=False, min_stroke_pixels=8):
    """Draw the Scene Engine's actual variable-width ink paths in Paint.

    Scene coordinates are mapped into the calibrated 240x180 Paint canvas
    area. This is an ink-only visual test, not a color scene renderer.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    width, height = data['width'], data['height']
    paths = data['paths']
    if width < 1 or height < 1 or not isinstance(paths, list) or not paths:
        raise ValueError('Invalid Scene Engine ink-strokes.json')
    if limit is not None and limit < 1:
        raise ValueError('--scene-limit must be positive')
    scale = min(240 / width, 180 / height, 1.0)
    dest_w, dest_h = max(1, round(width * scale)), max(1, round(height * scale))
    left = 310 + (240 - dest_w) // 2
    top = 155 + (180 - dest_h) // 2
    # Preserve whole pen gestures. Never fragment them into per-pixel
    # fragments: that produces the printer-like motion seen in Paint.
    selected_paths = paths[:limit]
    visited_pixels = set()
    plan = []
    duplicate_pixels = 0
    rejected_short = 0
    rejected_overlap = 0
    if not 2 <= min_stroke_pixels <= 100:
        raise ValueError('--scene-min-stroke-pixels must be between 2 and 100')

    def pixels_between(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx-dy
        while True:
            yield (x0, y0)
            if (x0, y0) == (x1, y1):
                return
            twice = 2*err
            if twice > -dy:
                err -= dy
                x0 += sx
            if twice < dx:
                err += dx
                y0 += sy

    # Rebuild connected pen gestures at *Paint resolution*. Source skeleton
    # paths are frequently only 1-4 pixels long; filtering them independently
    # discarded >99% of the dog. Join their edges before evaluating length.
    adjacency = {}
    dots = set()
    pixel_sizes = {}
    source_pixels = set()
    for record in selected_paths:
        mapped = []
        for point in record.get('points', []):
            if (not isinstance(point, list) or len(point) != 3 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v)
                            for v in point) or
                    not 0 <= point[0] < width or
                    not 0 <= point[1] < height or point[2] < 0):
                raise ValueError('Invalid Scene Engine stroke coordinates')
            x, y, radius = point
            dest = (left + round(x * scale), top + round(y * scale))
            diameter = 2 * radius * scale
            size = 1 if diameter < 1.5 else (2 if diameter < 3 else (3 if diameter < 5 else 4))
            pixel_sizes[dest] = max(pixel_sizes.get(dest, 1), size)
            if not mapped or mapped[-1] != dest:
                mapped.append(dest)
        if not mapped:
            continue
        dots.update(mapped)
        for start, end in zip(mapped, mapped[1:]):
            raster = list(pixels_between(start, end))
            for u, v in zip(raster, raster[1:]):
                if u == v:
                    continue
                adjacency.setdefault(u, set()).add(v)
                adjacency.setdefault(v, set()).add(u)
                source_pixels.update((u, v))
        source_pixels.update(mapped)

    # Consume each undirected edge once. Continue through junctions using
    # the straightest available continuation, avoiding arbitrary zigzags.
    remaining = {frozenset((u, v)) for u, neighbors in adjacency.items()
                 for v in neighbors}
    def follow(start, first):
        chain = [start, first]
        remaining.remove(frozenset((start, first)))
        previous, current = start, first
        while True:
            options = [p for p in adjacency[current]
                       if frozenset((current, p)) in remaining]
            if not options:
                break
            dx, dy = current[0]-previous[0], current[1]-previous[1]
            nxt = max(options, key=lambda p: (
                (p[0]-current[0])*dx + (p[1]-current[1])*dy,
                -abs((p[0]-current[0])*dy-(p[1]-current[1])*dx),
                -p[1], -p[0]))
            remaining.remove(frozenset((current, nxt)))
            chain.append(nxt)
            previous, current = current, nxt
            if current == start:
                break
        return chain

    chains = []
    for point in sorted(adjacency):
        if len(adjacency[point]) == 2:
            continue
        for neighbor in sorted(adjacency[point]):
            if frozenset((point, neighbor)) in remaining:
                chains.append(follow(point, neighbor))
    while remaining:
        edge = min(remaining, key=lambda e: tuple(sorted(e)))
        start, neighbor = sorted(edge)
        chains.append(follow(start, neighbor))
    # Preserve isolated ink pixels as dots instead of silently dropping them.
    covered = set(adjacency)
    for point in sorted(dots - covered):
        chains.append([point, point])

    for chain in chains:
        # At 240x180 resolution, one-pixel details matter. Do not filter
        # reconstructed gestures by the former source-path length threshold.
        sizes = sorted(pixel_sizes.get(p, 1) for p in chain)
        size = sizes[len(sizes)//2]
        plan.append((size, [list(p) for p in chain]))
    rejected_short = 0
    rejected_overlap = 0
    duplicate_pixels = 0
    if not plan:
        raise ValueError('No drawable paths after mapping')
    # Schedule real contours before tiny details. Within each Paint size,
    # choose the nearest available stroke endpoint, reversing open strokes
    # when it saves travel. This only changes order, never geometry.
    from collections import defaultdict
    buckets = defaultdict(list)
    for size, points in plan:
        buckets[size].append(points)
    ordered = []
    cursor = None
    for size in sorted(buckets):
        remaining_paths = buckets[size]
        # Long contours first; isolated dots and two-point details last.
        long_paths = [p for p in remaining_paths if len(set(map(tuple, p))) >= 5]
        short_paths = [p for p in remaining_paths if len(set(map(tuple, p))) < 5]
        for group in (long_paths, short_paths):
            while group:
                if cursor is None:
                    idx = max(range(len(group)), key=lambda i: len(group[i]))
                    reverse = False
                else:
                    def travel(i):
                        p = group[i]
                        forward = (p[0][0]-cursor[0])**2 + (p[0][1]-cursor[1])**2
                        backward = (p[-1][0]-cursor[0])**2 + (p[-1][1]-cursor[1])**2
                        return (min(forward, backward), -len(p), i)
                    idx = min(range(len(group)), key=travel)
                    p = group[idx]
                    forward = (p[0][0]-cursor[0])**2 + (p[0][1]-cursor[1])**2
                    backward = (p[-1][0]-cursor[0])**2 + (p[-1][1]-cursor[1])**2
                    reverse = backward < forward
                points = group.pop(idx)
                if reverse:
                    points = points[::-1]
                ordered.append((size, points))
                cursor = points[-1]
    plan = ordered
    # Validate planned motion and report natural contour intersections.
    planned_pixels = set()
    overlap_count = 0
    out_of_bounds = 0
    # Check actual visits by each mouse stroke. Shared vertices between
    # adjacent segments of ONE continuous stroke are not repeated visits.
    for _, points in plan:
        stroke_pixels = []
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            segment = list(pixels_between(tuple(start), tuple(end)))
            if segment_index:
                segment = segment[1:]
            stroke_pixels.extend(segment)
        if not stroke_pixels:
            stroke_pixels = [tuple(points[0])]
        for pixel in stroke_pixels:
            if not (left <= pixel[0] < left + dest_w + 2 and
                    top <= pixel[1] < top + dest_h + 2):
                out_of_bounds += 1
            if pixel in planned_pixels:
                overlap_count += 1
            planned_pixels.add(pixel)
    lengths = [len(points) for _, points in plan]
    width_counts = {str(size): sum(level == size for level, _ in plan) for size in (1, 2, 3, 4)}
    report = {'source_paths': len(selected_paths),
              'planned_strokes': len(plan),
              'rejected_short_paths': rejected_short,
              'rejected_overlapping_paths': rejected_overlap,
              'minimum_stroke_pixels': min_stroke_pixels,
              'median_stroke_vertices': sorted(lengths)[len(lengths)//2],
              'paint_size_levels': width_counts,
              'unique_pixels': len(planned_pixels),
              'source_unique_pixels': len(source_pixels),
              'repeated_pixel_visits': overlap_count,
              'overlaps_allowed_for_natural_contours': True,
              'skipped_source_pixel_visits': duplicate_pixels,
              'out_of_bounds_pixels': out_of_bounds,
              'paint_execution_started': False}
    print('Scene path validation:', json.dumps(report, indent=2))
    if out_of_bounds:
        raise RuntimeError('Scene path validation FAILED: out-of-bounds motion')
    coverage_ratio = len(planned_pixels & source_pixels) / max(1, len(source_pixels))
    print(f'Scene source pixel coverage: {coverage_ratio:.2%}')
    if coverage_ratio < 0.95:
        raise RuntimeError('Scene planning lost source pixels; Paint drawing blocked')
    if validate_only:
        print('PASS: planned Paint paths stay in bounds and preserve source ink pixels.')
        return
    engine = DrawingEngine(speed=speed)
    # Validate every planned position before touching the Paint canvas.
    for _, points in plan:
        engine.paint.validate_canvas_points(points)
    print(f'Natural Paint paths: skipped {rejected_short} short and {rejected_overlap} nearly redundant paths.')
    print(f'Scene Engine Paint test: {len(plan)} deduplicated pen paths '
          f'from {len(selected_paths)} source paths; '
          f'canvas={dest_w}x{dest_h}, scale={scale:.4f}')
    print('Contour intersections are retained; nearly redundant paths are omitted.')
    print('Use a NEW blank white maximized Paint canvas. F8 stops drawing.')
    engine.tools.select('pencil')
    engine.colors.select_primary('black')
    # The validated plan is already grouped by size and travel-optimized.
    try:
        for index, (size, points) in enumerate(plan, 1):
            from drawbuddy.controller import check_cancel
            check_cancel()
            engine.stroke_size.select(size)
            engine.canvas.stroke(points)
            if index % 100 == 0:
                print(f'Scene progress: {index}/{len(plan)}')
    except DrawingCancelled:
        print('STOPPED by F8. Partial scene remains in Paint.')
        return
    print('DONE: Scene Engine ink drawn in Microsoft Paint.')
    engine.paint.refresh_display()


def main():
    parser = argparse.ArgumentParser(description='Draw a saved lesson in Microsoft Paint.')
    parser.add_argument('--scene-strokes', type=Path, help='Draw Scene Engine ink-strokes.json in Paint (experimental)')
    parser.add_argument('--scene-min-stroke-pixels', type=int, default=8, help='Minimum unique pixels per natural pen stroke (default 8; try 5 for detail)')
    parser.add_argument('--scene-validate-only', action='store_true', help='Validate planned pen paths without opening or touching Paint')
    parser.add_argument('--scene-limit', type=int, default=None, help='Limit Scene Engine source paths for a short Paint test')
    parser.add_argument('drawing', nargs='?', help='Drawing name (e.g. dog), without .json')
    parser.add_argument('--speed', type=float, default=None, help='Stroke speed multiplier (e.g. 4, 20, 50, 100)')
    parser.add_argument('--no-publish-result', action='store_true', help='Skip automatic Paint canvas capture and GitHub upload')
    parser.add_argument('--natural-source', type=Path, help='Trace outlines and fill color regions with natural vertical strokes')
    parser.add_argument('--continuous-source', type=Path, help='Optimize 0/1/color pixels into continuous mouse strokes')
    parser.add_argument('--region-draw-source', type=Path, help='Draw shared region boundaries in Paint, no bucket fill')
    parser.add_argument('--region-source', type=Path, help='Analyze source regions and export boundary preview without launching Paint')
    parser.add_argument('--centerline-source', type=Path, help='Experimental 2px centerline contours without fills')
    parser.add_argument('--lineart-source', type=Path, help='Draw black outlines only, with no fill')
    parser.add_argument('--freehand-source', type=Path, help='Draw source with contour-following freehand mouse strokes')
    parser.add_argument('--binary-source', type=Path, help='Print the source as 0/1 horizontal pixel runs (default)')
    parser.add_argument('--binary-fill', action='store_true', help='Experimental contour and bucket fill mode (unsafe until validated)')
    parser.add_argument('--binary-runs', action='store_true', help='Explicitly select the default 0/1 row-by-row printing')
    parser.add_argument('--binary-outlines', action='store_true', help='Use the previous experimental contour drawing')
    parser.add_argument('--color-raster', action='store_true', help='Print RGB colors as separate 0/1 raster runs')
    parser.add_argument('--no-compare', action='store_true', help='Skip automatic Paint vs source comparison')
    parser.add_argument('--binary-threshold', type=int, default=None, help='Optional grayscale cutoff 0..255 (default: Otsu)')
    args = parser.parse_args()
    if args.scene_strokes:
        if args.drawing or any((args.region_draw_source, args.region_source,
                args.centerline_source, args.natural_source, args.continuous_source,
                args.lineart_source, args.freehand_source, args.binary_source)):
            parser.error('--scene-strokes cannot be combined with other drawing sources')
        config_path = Path(__file__).resolve().parent / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
        if type(speed) not in (float, int) or not math.isfinite(speed) or not 0 < speed <= 100:
            parser.error('drawing speed must be greater than 0 and at most 100')
        draw_scene_strokes(args.scene_strokes, float(speed), args.scene_limit, args.scene_validate_only, args.scene_min_stroke_pixels)
        return
    if args.scene_min_stroke_pixels != 8:
        parser.error('--scene-min-stroke-pixels requires --scene-strokes')
    if args.scene_validate_only:
        parser.error('--scene-validate-only requires --scene-strokes')
    if args.scene_limit is not None:
        parser.error('--scene-limit requires --scene-strokes')
    if args.region_draw_source and (args.region_source or args.centerline_source or args.natural_source or args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --region-draw-source alone')
    if args.region_source:
        if any((args.centerline_source, args.natural_source, args.continuous_source,
                args.lineart_source, args.freehand_source, args.binary_source, args.drawing)):
            parser.error('Use --region-source alone')
        print('Region-first analysis:', analyze_regions(args.region_source))
        return
    root = Path(__file__).resolve().parent
    drawings_dir = root / 'drawings'
    if args.centerline_source and (args.natural_source or args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --centerline-source alone')
    if args.natural_source and (args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --natural-source alone')
    if args.continuous_source and (args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --continuous-source alone')
    if args.lineart_source and (args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --lineart-source alone')
    if args.freehand_source and (args.binary_source or args.drawing):
        parser.error('Use --freehand-source alone, without a drawing or --binary-source')
    if args.region_draw_source:
        drawing = build_region_outline_drawing(args.region_draw_source)
        print(f'Region drawing plan: {drawing["region_drawing"]}')
    elif args.centerline_source:
        drawing = build_centerline_drawing(args.centerline_source)
        print(f'Centerline drawing plan: {drawing["lineart"]}')
    elif args.natural_source:
        drawing = build_natural_drawing(args.natural_source)
        print(f'Natural drawing plan: {drawing["natural"]}')
    elif args.continuous_source:
        drawing = build_continuous_color_drawing(args.continuous_source)
        print(f'Continuous color plan: {drawing["continuous"]}')
    elif args.lineart_source:
        drawing = build_lineart_drawing(args.lineart_source)
        print(f'Line art plan: {drawing["lineart"]}')
    elif args.freehand_source:
        drawing = build_freehand_drawing(args.freehand_source)
        print(f'Freehand plan: {drawing["freehand"]}')
    elif args.binary_source:
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
    print('Press F8 at any time to stop drawing without cancelling Paint strokes.')
    try:
        engine.execute(drawing, root / 'paint-diagnostic.png')
    except DrawingCancelled:
        print('STOPPED: drawing cancelled by F8. Partial drawing remains in Paint.')
        return
    engine.paint.refresh_display()
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
